"""Reproducible quality evaluation orchestration.

Datasets are supplied as frozen JSON artifacts.  The runner never downloads or
selects evaluation data and never promotes a measurement to scientific
acceptance by itself.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from openternary.benchmark.quality import (
    evaluate_quality,
    score_responses,
    validate_instruction_splits,
    validate_text_splits,
)
from openternary.config.schema import AppConfig
from openternary.services.identity import snapshot_identity
from openternary.services.resources import measured, phase


def _load_dataset(path: pathlib.Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid quality dataset JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("quality dataset requires schema_version=1")

    provenance = payload.get("dataset")
    required_provenance = ("id", "revision", "license", "source")
    if not isinstance(provenance, dict) or any(not provenance.get(key) for key in required_provenance):
        raise ValueError(f"quality dataset provenance requires {required_provenance}")

    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"calibration", "validation", "test"}:
        raise ValueError("quality dataset requires calibration, validation and test splits")
    for split_name, records in splits.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f"quality dataset split is empty: {split_name}")
        ids: list[str] = []
        for record in records:
            if not isinstance(record, dict) or any(not record.get(key) for key in ("id", "text", "language")):
                raise ValueError(f"invalid quality record in {split_name}")
            ids.append(str(record["id"]))
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate quality record IDs in {split_name}")
    validate_text_splits({name: [record["text"] for record in records] for name, records in splits.items()})

    instruction_cases = payload.get("instruction_cases")
    if not isinstance(instruction_cases, dict) or set(instruction_cases) != {"validation", "test"}:
        raise ValueError("instruction_cases requires validation and test splits")
    all_instruction_ids: list[str] = []
    instruction_prompts: dict[str, list[str]] = {}
    for split_name, cases in instruction_cases.items():
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"instruction split is empty: {split_name}")
        prompts: list[str] = []
        for case in cases:
            if not isinstance(case, dict) or any(not case.get(key) for key in ("id", "prompt", "expected")):
                raise ValueError(f"invalid instruction case in {split_name}")
            all_instruction_ids.append(str(case["id"]))
            prompts.append(str(case["prompt"]))
        instruction_prompts[split_name] = prompts
    if len(all_instruction_ids) != len(set(all_instruction_ids)):
        raise ValueError("instruction case IDs must be globally unique")
    instruction_audit = validate_instruction_splits(instruction_prompts)

    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return payload, hashlib.sha256(canonical).hexdigest(), instruction_audit


def validate_quality_dataset(path: pathlib.Path | str) -> dict[str, Any]:
    """Validate a frozen dataset without loading a model or creating a run."""
    payload, fingerprint, instruction_audit = _load_dataset(pathlib.Path(path))
    return {
        "dataset_fingerprint": fingerprint,
        "dataset": payload["dataset"],
        "split_counts": {name: len(records) for name, records in payload["splits"].items()},
        "instruction_counts": {name: len(cases) for name, cases in payload["instruction_cases"].items()},
        "instruction_data_audit": instruction_audit,
    }


@measured
def run_quality_benchmark(
    config: AppConfig,
    dataset_path: pathlib.Path | str,
    *,
    snapshot_path: pathlib.Path | str | None = None,
    model: Any | None = None,
    processor: Any | None = None,
    split: str,
    max_length: int,
    stride: int,
) -> dict[str, Any]:
    """Evaluate one frozen validation/test split with an exact-dtype model."""

    if split not in {"validation", "test"}:
        raise ValueError("quality split must be validation or test")
    if (model is None) != (processor is None):
        raise ValueError("model and processor must be supplied together")
    payload, dataset_fingerprint, instruction_data_audit = _load_dataset(pathlib.Path(dataset_path))
    from openternary.utils.seed import seed_everything

    seed_everything(config.seed)

    resolved_snapshot: pathlib.Path | None = None
    if model is None:
        from openternary.adapters.runtime import load_model, load_processor
        from openternary.benchmark.runner import _dtype_to_torch_dtype, _resolve_device_map
        from openternary.utils.hf_cache import resolve_snapshot

        resolved_snapshot = (
            pathlib.Path(snapshot_path)
            if snapshot_path is not None
            else resolve_snapshot(config.model.id, config.model.revision)
        )
        requested_dtype = _dtype_to_torch_dtype(config.dtype)
        device_map = _resolve_device_map(config.device)
        phase("load", device_map.get("", "cpu") if isinstance(device_map, dict) else "cuda:0")
        processor = load_processor(config, resolved_snapshot)
        model = load_model(config, resolved_snapshot, requested_dtype, _resolve_device_map(config.device))

    assert model is not None and processor is not None
    model.eval()
    actual_dtype: str = config.dtype
    actual_device: str = "cpu" if config.device == "auto" else config.device
    try:
        first_parameter = next(model.parameters())
        actual_dtype = str(first_parameter.dtype)
        actual_device = str(first_parameter.device)
        if resolved_snapshot is not None:
            from openternary.benchmark.runner import _dtype_to_torch_dtype

            requested_dtype = _dtype_to_torch_dtype(config.dtype)
            if first_parameter.dtype != requested_dtype:
                raise RuntimeError(
                    f"quality dtype mismatch: requested {requested_dtype}, loaded {first_parameter.dtype}"
                )
    except StopIteration:
        pass

    phase("inference", actual_device)
    splits = payload["splits"]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and callable(processor):
        tokenizer = processor
    if tokenizer is None:
        raise ValueError("processor does not expose a text tokenizer")
    perplexity = evaluate_quality(
        model,
        tokenizer,
        splits,
        split=split,
        max_length=max_length,
        stride=stride,
        device=actual_device,
    )
    by_language = perplexity["by_language"]
    if "en" not in by_language or "ja" not in by_language:
        raise ValueError("quality split requires both en and ja records")

    from openternary.benchmark.runner import _run_single_prompt

    cases = payload["instruction_cases"][split]
    generation = config.benchmark.generation
    generation_kwargs = {
        "do_sample": generation.do_sample,
        "num_beams": generation.num_beams,
        "max_new_tokens": generation.max_new_tokens,
    }
    responses: dict[str, str] = {}
    for case in cases:
        result = _run_single_prompt(
            processor=processor,
            model=model,
            prompt=case["prompt"],
            thinking=config.benchmark.thinking,
            generation_kwargs=generation_kwargs,
        )
        responses[case["id"]] = str(result["parsed_output_text"])
    instructions = score_responses(cases, responses)
    accuracy = instructions["exact_match_accuracy"]
    if accuracy is None:
        raise ValueError("instruction cases must contain expected answers")

    protocol = {
        "version": "quality-runner-v1",
        "split": split,
        "max_length": max_length,
        "stride": stride,
        "generation": generation_kwargs,
        "thinking": config.benchmark.thinking,
        "seed": config.seed,
        "dtype": config.dtype,
        "device": actual_device,
    }
    protocol_fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "report_schema_version": 3,
        "identity": snapshot_identity(resolved_snapshot),
        "protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint,
        "dataset": payload["dataset"],
        "dataset_fingerprint": dataset_fingerprint,
        "instruction_data_audit": instruction_data_audit,
        "model": {
            "id": config.model.id,
            "revision": config.model.revision,
            "snapshot_path": str(resolved_snapshot) if resolved_snapshot is not None else None,
            "requested_dtype": config.dtype,
            "actual_dtype": actual_dtype,
            "actual_device": actual_device,
        },
        "perplexity": perplexity,
        "instructions": instructions,
        "summary": {
            "general_ppl": by_language["en"]["perplexity"],
            "japanese_ppl": by_language["ja"]["perplexity"],
            "instruction_score": float(accuracy) * 100.0,
            "collapse_count": instructions["flagged_outputs"],
        },
        "scientific_acceptance": False,
        "acceptance_note": "measurement only; compare matched controls with the preregistered gate",
    }


__all__ = ["run_quality_benchmark", "validate_quality_dataset"]
