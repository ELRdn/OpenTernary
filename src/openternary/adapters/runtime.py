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
        return model_cls.from_pretrained(
            str(source), dtype=dtype, device_map=device_map, trust_remote_code=False, local_files_only=True
        )
    except TypeError:
        return model_cls.from_pretrained(
            str(source), torch_dtype=dtype, device_map=device_map, trust_remote_code=False, local_files_only=True
        )


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
