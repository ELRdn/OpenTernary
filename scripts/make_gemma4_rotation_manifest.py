"""Freeze the learned rotations for one already selected Gemma 4 target set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--rotation-dir", type=Path, required=True)
    parser.add_argument("--other-rotation-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    quant_report = json.loads((args.quantized / "quantization.json").read_text(encoding="utf-8"))
    selected = sorted(row["name"] for row in quant_report["per_tensor"] if row["quantizable"])
    if not selected or quant_report.get("group_size") != 128:
        raise ValueError("expected a selected G128 ternary target set")
    rotations = {}
    for weight_name in selected:
        module_name = weight_name.removesuffix(".weight")
        if module_name.endswith(".self_attn.q_proj"):
            layer = int(module_name.split(".layers.")[1].split(".")[0])
            path = args.rotation_dir / f"gemma4-q{layer}-learned-rotation-20260924.safetensors"
        elif args.other_rotation_dir:
            path = args.other_rotation_dir / f"{module_name}.safetensors"
        else:
            raise ValueError(f"no learned rotation for selected module: {module_name}")
        report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if report["module"] != module_name or Path(report["source"]).resolve() != source:
            raise ValueError(f"rotation provenance mismatch: {module_name}")
        rotations[module_name] = {"file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    plan = {"schema_version": 1, "source": str(source), "rotations": rotations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rotation_count": len(rotations), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
