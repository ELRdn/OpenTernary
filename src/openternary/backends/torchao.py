"""TorchAO 0.18 Int8WeightOnlyConfig v2 bridge; runtime acceptance is pending."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

from openternary.adapters.registry import select_adapter
from openternary.config.schema import AppConfig
from openternary.experiment.metadata import write_json
from openternary.services.errors import CapabilityError

PINNED_VERSION = "0.18.0"


class TorchAOBackend:
    api_version = 1

    def convert(self, source: Path, destination: Path, config: AppConfig) -> Any:
        if importlib.metadata.version("torchao") != PINNED_VERSION:
            raise CapabilityError(f"TorchAO API bridge requires torchao=={PINNED_VERSION}")
        if config.device == "cuda":
            raise CapabilityError("TorchAO converter writes CPU state; GPU runtime must be validated separately")
        import torch
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        from torchao.quantization.granularity import PerGroup, PerTensor

        from openternary.export.packed import _copy_interface
        from openternary.quant.fake_quant import ConvertReport
        from openternary.services.artifacts import canonical_hash, file_hash
        from openternary.services.reporting import progress
        from openternary.services.snapshot import SnapshotReader

        destination.mkdir(parents=True, exist_ok=False)
        _copy_interface(source, destination)
        (destination / "tensors").mkdir()
        adapter = select_adapter(config, source)
        q = config.quantization
        granularity = PerGroup(q.group_size) if q.scale_granularity == "per_group" else PerTensor()
        quant_config = Int8WeightOnlyConfig(granularity=granularity, set_inductor_config=False)
        entries = []
        with SnapshotReader(source) as reader:
            names = reader.keys()
            for index, name in enumerate(names):
                tensor = reader.get_tensor(name)
                info = adapter.classify(name, list(tensor.shape), str(tensor.dtype))
                if info.quantizable:
                    module = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False, device="meta")
                    module.weight = torch.nn.Parameter(tensor, requires_grad=False)
                    quantize_(module, quant_config)
                    tensor = module.weight.detach()
                path = destination / "tensors" / f"{index:06d}.pt"
                torch.save({"weight": tensor}, path)
                entries.append(
                    {
                        "name": name,
                        "file": path.relative_to(destination).as_posix(),
                        "sha256": file_hash(path),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "quantizable": info.quantizable,
                    }
                )
                progress("torchao_tensors", index + 1, len(names))
        content = canonical_hash(entries)
        report = {
            "schema_version": 1,
            "backend": "torchao",
            "backend_version": PINNED_VERSION,
            "scheme": "int8-weight-only",
            "weight_dtype": "int8",
            "activation_dtype": "preserve",
            "per_tensor": entries,
            "content_fingerprint": content,
            "model_reload": "not_run",
            "runtime_validation": "pending",
            "native_kernel": "unverified",
        }
        write_json(destination / "quantization.json", report)
        files = {p.relative_to(destination).as_posix(): file_hash(p) for p in destination.rglob("*") if p.is_file()}
        total = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
        return ConvertReport(destination, report, content, files, len(entries), 0, total, total)

    def load_weights(self, source: Path) -> Any:
        """Yield verified state tensors; caller owns architecture construction and device."""
        if importlib.metadata.version("torchao") != PINNED_VERSION:
            raise CapabilityError(f"loading requires torchao=={PINNED_VERSION}")
        import torch
        import torchao.quantization  # noqa: F401 - registers safe tensor classes

        from openternary.services.artifacts import member_path, validate_artifact

        validate_artifact(source)
        report = json.loads((source / "quantization.json").read_text(encoding="utf-8"))
        seen: set[str] = set()
        for row in report["per_tensor"]:
            if row["name"] in seen:
                raise ValueError("duplicate backend tensor")
            seen.add(row["name"])
            payload = torch.load(member_path(source, row["file"]), map_location="cpu", weights_only=True)
            tensor = payload["weight"]
            if list(tensor.shape) != row["shape"] or str(tensor.dtype) != row["dtype"]:
                raise ValueError("backend tensor metadata mismatch")
            yield row["name"], tensor
