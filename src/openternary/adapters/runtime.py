"""Lazy runtime loading behind the model adapter boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openternary.adapters.registry import adapter_name, select_adapter
from openternary.config.schema import AppConfig
from openternary.services.errors import CapabilityError


def normalize_diffusion_weights(root: Path) -> None:
    """Publish standard Diffusers shard names without rewriting tensor payloads."""
    import json

    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return
    data = json.loads(index.read_text(encoding="utf-8"))
    mapping = {name: name.replace("model-", "diffusion_pytorch_model-", 1) for name in set(data["weight_map"].values())}
    for old, new in mapping.items():
        (root / old).rename(root / new)
    data["weight_map"] = {key: mapping[name] for key, name in data["weight_map"].items()}
    (root / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    index.unlink()


def loader_classes(config: AppConfig, source: Path) -> tuple[Any, Any]:
    name = adapter_name(select_adapter(config, source))
    if name == "gemma4":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        return AutoModelForMultimodalLM, AutoProcessor
    if name == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        return AutoModelForCausalLM, AutoTokenizer
    raise CapabilityError("LLM evaluation cannot evaluate a Diffusers component; select a diffusion evaluator plugin")


def load_processor(config: AppConfig, source: Path) -> Any:
    _, processor_cls = loader_classes(config, source)
    return processor_cls.from_pretrained(str(source), trust_remote_code=False, local_files_only=True)


def load_model(config: AppConfig, source: Path, dtype: Any, device_map: Any) -> Any:
    from openternary.services.artifacts import MANIFEST, validate_artifact

    model_cls, _ = loader_classes(config, source)
    manifest = validate_artifact(source) if (source / MANIFEST).is_file() else None
    if manifest and manifest["format"] == "ternary-packed":
        raise CapabilityError("export packed weights to safetensors before model loading; no native packed runtime")
    if manifest and manifest["format"] == "torchao":
        return _load_torchao(config, source, dtype, device_map, model_cls)
    try:
        model = model_cls.from_pretrained(
            str(source), dtype=dtype, device_map=device_map, trust_remote_code=False, local_files_only=True
        )
    except TypeError:
        model = model_cls.from_pretrained(
            str(source), torch_dtype=dtype, device_map=device_map, trust_remote_code=False, local_files_only=True
        )
    _install_rotations(model, source, manifest)
    return model


def _install_rotations(model: Any, source: Path, manifest: dict[str, Any] | None) -> None:
    """Attach the input maps required by rotated ternary weights after reload."""
    import json

    import torch
    from safetensors.torch import load_file

    metadata_path = source / "openternary" / "rotations.json"
    if not metadata_path.is_file():
        report_path = source / "quantization.json"
        if report_path.is_file() and json.loads(report_path.read_text(encoding="utf-8")).get("rotation"):
            raise ValueError("rotated weights are missing their required input transform")
        return
    if manifest is None or manifest.get("format") != "safetensors":
        raise ValueError("rotated weights require a validated safetensors artifact")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("method") != "hadamard-learned-cayley"
        or metadata.get("transform_order") != ["normalized-hadamard", "learned-cayley"]
        or metadata.get("matrix_file") != "openternary/rotations.safetensors"
    ):
        raise ValueError("unsupported rotation artifact contract")
    report = json.loads((source / "quantization.json").read_text(encoding="utf-8"))
    if report.get("rotation") != metadata:
        raise ValueError("rotation metadata does not match quantization report")
    matrices = load_file(str(source / metadata["matrix_file"]), device="cpu")
    entries = metadata.get("rotations")
    if not isinstance(entries, dict) or not entries or matrices.keys() != entries.keys():
        raise ValueError("rotation matrices do not match metadata")
    from openternary.quant.rotation import apply_block_rotation, hadamard_last_dim

    handles = []
    for module_name, entry in sorted(entries.items()):
        matrix = matrices[module_name]
        block_size = entry["block_size"]
        module = model.get_submodule(module_name)
        if (
            block_size != 128
            or matrix.shape != (128, 128)
            or matrix.dtype != torch.float32
            or not torch.isfinite(matrix).all()
            or module.weight.ndim != 2
            or module.weight.shape[-1] % block_size
        ):
            raise ValueError(f"invalid saved rotation or target module: {module_name}")
        if (matrix.T @ matrix - torch.eye(block_size)).abs().max().item() > 1e-4:
            raise ValueError(f"saved rotation is not orthogonal: {module_name}")
        rotation = matrix.to(module.weight.device)

        def rotate_input(
            _module: Any, inputs: tuple[torch.Tensor, ...], *, rotation: torch.Tensor = rotation
        ) -> tuple[torch.Tensor, ...]:
            transformed = hadamard_last_dim(inputs[0].float(), rotation.shape[0])
            transformed = apply_block_rotation(transformed, rotation).to(inputs[0].dtype)
            return (transformed, *inputs[1:])

        handles.append(module.register_forward_pre_hook(rotate_input))
    model._openternary_rotation_hooks = handles


def _load_torchao(config: AppConfig, source: Path, dtype: Any, device_map: Any, model_cls: Any) -> Any:
    import torch
    from transformers import AutoConfig

    from openternary.backends.torchao import TorchAOBackend

    raw_config = AutoConfig.from_pretrained(str(source), trust_remote_code=False, local_files_only=True)
    model = model_cls.from_config(raw_config, torch_dtype=dtype, trust_remote_code=False)
    expected = set(model.state_dict())
    seen: set[str] = set()
    for name, tensor in TorchAOBackend().load_weights(source):
        if name not in expected:
            raise ValueError(f"backend state contains an unexpected key: {name}")
        prefix, _, leaf = name.rpartition(".")
        module = model.get_submodule(prefix) if prefix else model
        original = getattr(module, leaf)
        if tuple(original.shape) != tuple(tensor.shape):
            raise ValueError(f"backend state shape mismatch: {name}")
        if tensor.is_floating_point() and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if isinstance(original, torch.nn.Parameter):
            setattr(module, leaf, torch.nn.Parameter(tensor, requires_grad=False))
        else:
            setattr(module, leaf, tensor)
        seen.add(name)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    # Only tied output embeddings may be omitted by the original snapshot.
    allowed_missing = {"lm_head.weight"} if getattr(raw_config, "tie_word_embeddings", False) else set()
    if expected - seen - allowed_missing:
        raise ValueError(f"backend state is incomplete: {sorted(expected - seen - allowed_missing)[:8]}")
    device = (
        device_map.get("", "cpu") if isinstance(device_map, dict) else "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return model.to(device).eval()
