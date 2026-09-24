"""Freeze a validation-selected Gemma 4 hybrid for CLI conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openternary.config.loader import dump_config_yaml, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--control-report", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--exclude-q-layer", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.control_report.read_text(encoding="utf-8"))
    all_targets = {row["name"] for row in report["per_tensor"] if row["quantizable"]}
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        selected = set(selection["selected_names"])
    else:
        selected = {name for name in all_targets if ".self_attn.q_proj.weight" in name}
    if args.exclude_q_layer is not None:
        if args.selection:
            raise ValueError("--selection and --exclude-q-layer cannot be combined")
        excluded_name = f"model.language_model.layers.{args.exclude_q_layer}.self_attn.q_proj.weight"
        if excluded_name not in selected:
            raise ValueError(f"q layer is not in the canonical target set: {excluded_name}")
        selected.remove(excluded_name)
    if len(all_targets) != 205 or not selected or not selected <= all_targets:
        raise ValueError("canonical target/selected-module contract mismatch")
    selected_hash = hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest()
    if args.selection and selected_hash != selection["selected_names_sha256"]:
        raise ValueError("selection names fingerprint mismatch")
    cfg = load_config(
        config_path="configs/gemma4-e2b.yaml",
        cli_overrides={
            "model.id": str(args.source.resolve()),
            "model.revision": "6befbaca7398925921802abd1f277b495b78b738",
            "quantization.scale_granularity": "per_group",
            "quantization.group_size": 128,
            "quantization.mixed_precision": dict.fromkeys(sorted(all_targets - selected), "preserve"),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dump_config_yaml(cfg), encoding="utf-8")
    manifest = {
        "source": str(args.source.resolve()),
        "control_report": str(args.control_report.resolve()),
        "selected_names": sorted(selected),
        "selected_names_sha256": selected_hash,
        "ternary_module_count": len(selected),
        "ternary_parameter_count": sum(row["param_count"] for row in report["per_tensor"] if row["name"] in selected),
        "eligible_module_count": len(all_targets),
        "eligible_parameter_count": sum(
            row["param_count"] for row in report["per_tensor"] if row["name"] in all_targets
        ),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
