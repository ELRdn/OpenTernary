"""Compare utility — behavioral Δ without quality claim (Phase 3)."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from typing import Any


def _load_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid report JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return result


def _find_benchmark_json(run_dir: pathlib.Path) -> dict[str, Any] | None:
    # Try benchmark.json directly or in run_dir
    for cand in [run_dir / "benchmark.json", run_dir]:
        if cand.is_file() and cand.suffix == ".json":
            return _load_json(cand)
        if cand.is_dir():
            p = cand / "benchmark.json"
            if p.exists():
                return _load_json(p)
    return None


def _find_quantization_json(run_dir: pathlib.Path) -> dict[str, Any] | None:
    for cand in [run_dir / "quantization.json", run_dir / "metrics.json"]:
        if cand.exists():
            j = _load_json(cand)
            if j and ("quantization" in j or "scale_granularity" in j or "content_fingerprint" in j):
                # metrics.json wraps quantization under "quantization"
                if "quantization" in j:
                    return j["quantization"]  # type: ignore[no-any-return]
                return j  # type: ignore[no-any-return]
    # also check artifacts/snapshot/quantization.json
    p = run_dir / "artifacts" / "snapshot" / "quantization.json"
    if p.exists():
        return _load_json(p)
    return None


def _find_quality_json(run_dir: pathlib.Path) -> dict[str, Any] | None:
    candidate = run_dir if run_dir.is_file() and run_dir.name == "quality.json" else run_dir / "quality.json"
    if not candidate.exists():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid quality JSON: {candidate}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid quality JSON object: {candidate}")
    return payload


def _validate_quality_report(report: dict[str, Any], *, label: str) -> dict[str, str] | None:
    """Validate schema-v2 audit fields before any quality acceptance calculation."""
    schema_version = report.get("report_schema_version")
    if schema_version is None:
        return None
    if schema_version not in {2, 3}:
        raise ValueError(f"{label} quality report schema must be 2 or 3")

    protocol = report.get("protocol")
    protocol_fingerprint = report.get("protocol_fingerprint")
    if not isinstance(protocol, dict) or not isinstance(protocol_fingerprint, str):
        raise ValueError(f"{label} quality report requires protocol and protocol_fingerprint")
    expected_protocol_fingerprint = hashlib.sha256(
        json.dumps(
            protocol,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if protocol_fingerprint != expected_protocol_fingerprint:
        raise ValueError(f"{label} quality protocol_fingerprint does not match protocol")

    model = report.get("model")
    required_model_fields = (("revision",) if schema_version == 2 else ()) + ("actual_dtype", "actual_device")
    if not isinstance(model, dict) or any(
        not isinstance(model.get(key), str) or not model[key] for key in required_model_fields
    ):
        raise ValueError(f"{label} quality model requires {required_model_fields}")

    instructions = report.get("instructions")
    if not isinstance(instructions, dict):
        raise ValueError(f"{label} quality instructions must be an object")
    results = instructions.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{label} quality instructions.results must be non-empty")

    flagged_outputs = 0
    exact_matches: list[bool] = []
    result_ids: list[str] = []
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise ValueError(f"{label} quality response {index} must be an object")
        response_id = row.get("id")
        response_text = row.get("response_text")
        response_sha256 = row.get("response_sha256")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError(f"{label} quality response {index} requires id")
        if not isinstance(response_text, str) or not isinstance(response_sha256, str):
            raise ValueError(f"{label} quality response {response_id} requires text and response_sha256")
        expected_hash = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
        if response_sha256 != expected_hash:
            raise ValueError(f"{label} quality response_sha256 mismatch: {response_id}")
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, dict) or not isinstance(diagnostics.get("flags"), list):
            raise ValueError(f"{label} quality response diagnostics invalid: {response_id}")
        flagged_outputs += bool(diagnostics["flags"])
        exact_match = row.get("exact_match")
        if exact_match is not None:
            if not isinstance(exact_match, bool):
                raise ValueError(f"{label} quality exact_match invalid: {response_id}")
            exact_matches.append(exact_match)
        result_ids.append(response_id)
    if len(result_ids) != len(set(result_ids)):
        raise ValueError(f"{label} quality response IDs must be unique")

    recorded_flagged = instructions.get("flagged_outputs")
    if (
        not isinstance(recorded_flagged, int)
        or isinstance(recorded_flagged, bool)
        or recorded_flagged != flagged_outputs
    ):
        raise ValueError(f"{label} quality flagged_outputs mismatch")
    recorded_scored = instructions.get("exact_match_scored")
    if (
        not isinstance(recorded_scored, int)
        or isinstance(recorded_scored, bool)
        or recorded_scored != len(exact_matches)
    ):
        raise ValueError(f"{label} quality exact_match_scored mismatch")
    recorded_accuracy = instructions.get("exact_match_accuracy")
    expected_accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else None
    if expected_accuracy is None:
        if recorded_accuracy is not None:
            raise ValueError(f"{label} quality exact_match_accuracy mismatch")
    elif (
        not isinstance(recorded_accuracy, (int, float))
        or isinstance(recorded_accuracy, bool)
        or not math.isfinite(float(recorded_accuracy))
        or not math.isclose(float(recorded_accuracy), expected_accuracy, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"{label} quality exact_match_accuracy mismatch")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{label} quality summary must be an object")

    perplexity = report.get("perplexity")
    by_language = perplexity.get("by_language") if isinstance(perplexity, dict) else None
    if not isinstance(by_language, dict):
        raise ValueError(f"{label} quality perplexity.by_language must be an object")
    for language, summary_key in (("en", "general_ppl"), ("ja", "japanese_ppl")):
        row = by_language.get(language)
        if not isinstance(row, dict):
            raise ValueError(f"{label} quality perplexity requires language {language}")
        nll_sum = row.get("nll_sum")
        scored_tokens = row.get("scored_tokens")
        mean_nll = row.get("mean_nll")
        language_ppl = row.get("perplexity")
        if (
            not isinstance(nll_sum, (int, float))
            or isinstance(nll_sum, bool)
            or not math.isfinite(float(nll_sum))
            or float(nll_sum) < 0
            or not isinstance(scored_tokens, int)
            or isinstance(scored_tokens, bool)
            or scored_tokens <= 0
            or not isinstance(mean_nll, (int, float))
            or isinstance(mean_nll, bool)
            or not math.isfinite(float(mean_nll))
            or not isinstance(language_ppl, (int, float))
            or isinstance(language_ppl, bool)
            or not math.isfinite(float(language_ppl))
            or float(language_ppl) <= 0
        ):
            raise ValueError(f"{label} quality perplexity metrics invalid for {language}")
        if not math.isclose(float(mean_nll), float(nll_sum) / scored_tokens, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{label} quality mean_nll mismatch for {language}")
        if not math.isclose(float(language_ppl), math.exp(float(mean_nll)), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{label} quality perplexity mismatch for {language}")
        summary_value = summary.get(summary_key)
        if (
            not isinstance(summary_value, (int, float))
            or isinstance(summary_value, bool)
            or not math.isclose(float(summary_value), float(language_ppl), rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(f"{label} quality {summary_key} does not match perplexity")

    if summary.get("collapse_count") != flagged_outputs:
        raise ValueError(f"{label} quality collapse_count mismatch")
    if expected_accuracy is not None:
        instruction_score = summary.get("instruction_score")
        if (
            not isinstance(instruction_score, (int, float))
            or isinstance(instruction_score, bool)
            or not math.isclose(float(instruction_score), expected_accuracy * 100.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"{label} quality instruction_score mismatch")

    if schema_version == 3:
        identity = report.get("identity")
        environment = report.get("measurement_environment")
        if not isinstance(identity, dict) or not isinstance(environment, dict):
            raise ValueError(f"{label} schema 3 requires identity and measurement_environment")
        if identity.get("status") == "resolved":
            fingerprint = identity.get("source_fingerprint")
            if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                raise ValueError(f"{label} invalid source fingerprint")
            interface = identity.get("interface_files")
            if not isinstance(interface, dict):
                raise ValueError(f"{label} invalid interface inventory")
            from openternary.services.artifacts import canonical_hash

            expected = canonical_hash(interface) if interface else None
            if identity.get("interface_fingerprint") != expected:
                raise ValueError(f"{label} interface fingerprint mismatch")
        if environment.get("actual_device") != model["actual_device"]:
            raise ValueError(f"{label} measurement device mismatch")
    return {key: str(model.get(key)) for key in ("revision", "actual_dtype", "actual_device")}


def compare_runs(
    baseline_dir: pathlib.Path | str,
    quantized_dir: pathlib.Path | str,
) -> dict[str, Any]:
    """Compare two runs behaviorally.

    Returns dict with:
    - protocol_match (bool or None if missing)
    - result_match (bool or None)
    - result_observation: "same" | "different" | "unknown"
    - details per prompt token diff (if benchmarks available)
    - quantization content fingerprints (if available)
    """
    b_dir = pathlib.Path(baseline_dir)
    q_dir = pathlib.Path(quantized_dir)
    if not b_dir.exists() or not q_dir.exists():
        raise ValueError("comparison inputs must exist")

    b_bench = _find_benchmark_json(b_dir)
    q_bench = _find_benchmark_json(q_dir)

    b_quant = _find_quantization_json(b_dir)
    q_quant = _find_quantization_json(q_dir)
    b_quality = _find_quality_json(b_dir)
    q_quality = _find_quality_json(q_dir)
    if not any((b_bench, b_quant, b_quality)) or not any((q_bench, q_quant, q_quality)):
        raise ValueError("each comparison input must contain a non-empty benchmark, quantization or quality report")
    for label, report in (("baseline", b_bench), ("candidate", q_bench)):
        if report is not None:
            rows = report.get("results")
            if not isinstance(report.get("suite"), dict) or not isinstance(rows, list) or not rows:
                raise ValueError(f"{label} benchmark requires suite and non-empty results")
            ids = [row.get("id") for row in rows if isinstance(row, dict)]
            if (
                len(ids) != len(rows)
                or any(not isinstance(key, str) or not key for key in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError(f"{label} benchmark requires unique result IDs")

    quality_gate: dict[str, Any] | None = None
    quality_protocol_match: bool | None = None
    dataset_match: bool | None = None
    b_quality_protocol: str | None = None
    q_quality_protocol: str | None = None
    b_dataset: str | None = None
    q_dataset: str | None = None
    quality_report_schema_match: bool | None = None
    model_revision_match: bool | None = None
    actual_dtype_match: bool | None = None
    actual_device_match: bool | None = None
    if b_quality is not None or q_quality is not None:
        if b_quality is None or q_quality is None:
            raise ValueError("both runs must contain quality.json for quality comparison")
        b_model = _validate_quality_report(b_quality, label="baseline")
        q_model = _validate_quality_report(q_quality, label="candidate")
        if (b_model is None) != (q_model is None):
            raise ValueError("quality report schema mismatch")
        quality_report_schema_match = (
            (b_quality.get("report_schema_version") == q_quality.get("report_schema_version"))
            if b_model is not None
            else None
        )
        if b_model is not None and q_model is not None:
            model_revision_match = b_model["revision"] == q_model["revision"]
            actual_dtype_match = b_model["actual_dtype"] == q_model["actual_dtype"]
            actual_device_match = b_model["actual_device"] == q_model["actual_device"]
            if not model_revision_match and b_quality.get("report_schema_version") == 2:
                raise ValueError("quality model revision mismatch")
            if not actual_dtype_match:
                raise ValueError("quality actual dtype mismatch")
            if not actual_device_match:
                raise ValueError("quality actual device mismatch")
        b_summary = b_quality.get("summary")
        q_summary = q_quality.get("summary")
        if not isinstance(b_summary, dict) or not isinstance(q_summary, dict):
            raise ValueError("quality.json requires an object summary")
        b_quality_protocol = b_quality.get("protocol_fingerprint")
        q_quality_protocol = q_quality.get("protocol_fingerprint")
        if not isinstance(b_quality_protocol, str) or not isinstance(q_quality_protocol, str):
            raise ValueError("quality.json requires protocol_fingerprint")
        quality_protocol_match = b_quality_protocol == q_quality_protocol
        if not quality_protocol_match:
            raise ValueError("quality protocol fingerprint mismatch")
        b_dataset = b_quality.get("dataset_fingerprint")
        q_dataset = q_quality.get("dataset_fingerprint")
        if not isinstance(b_dataset, str) or not isinstance(q_dataset, str):
            raise ValueError("quality.json requires dataset_fingerprint")
        dataset_match = b_dataset == q_dataset
        if not dataset_match:
            raise ValueError("quality dataset fingerprint mismatch")
        from openternary.benchmark.acceptance import compare_quality_metrics

        identity_eligible = False
        if b_quality.get("report_schema_version") == q_quality.get("report_schema_version") == 3:
            b_identity, q_identity = b_quality["identity"], q_quality["identity"]
            if b_identity.get("status") == q_identity.get("status") == "resolved":
                for key in ("source_fingerprint", "interface_fingerprint"):
                    if b_identity.get(key) != q_identity.get(key):
                        raise ValueError(f"quality {key} mismatch")
                if b_quality["measurement_environment"] != q_quality["measurement_environment"]:
                    raise ValueError("quality measurement environment mismatch")
                identity_eligible = (
                    bool(b_identity.get("interface_fingerprint"))
                    and b_identity.get("scope") == q_identity.get("scope") == "model"
                )
        quality_gate = (
            compare_quality_metrics(b_summary, q_summary)
            if identity_eligible
            else {
                "accepted": False,
                "status": "insufficient_evidence",
                "reason": "schema 3 with matched source/interface/runtime and non-synthetic model evidence required",
            }
        )

    # Protocol comparison
    protocol_match: bool | None = None
    result_match: bool | None = None
    result_observation = "unknown"

    if b_bench and q_bench:
        b_proto = b_bench.get("suite", {}).get("fingerprint") if isinstance(b_bench.get("suite"), dict) else None
        q_proto = q_bench.get("suite", {}).get("fingerprint") if isinstance(q_bench.get("suite"), dict) else None
        if b_proto and q_proto:
            protocol_match = b_proto == q_proto
            if b_bench.get("identity") or q_bench.get("identity"):
                b_identity, q_identity = b_bench.get("identity", {}), q_bench.get("identity", {})
                protocol_match = (
                    protocol_match
                    and all(
                        b_identity.get(key) is not None and b_identity.get(key) == q_identity.get(key)
                        for key in ("source_fingerprint", "interface_fingerprint")
                    )
                    and b_bench.get("measurement_environment") == q_bench.get("measurement_environment")
                    and b_bench.get("dtype") == q_bench.get("dtype")
                    and b_bench.get("seed") == q_bench.get("seed")
                    and b_bench.get("warmup") == q_bench.get("warmup")
                )
        # result
        b_res = b_bench.get("result_fingerprint")
        q_res = q_bench.get("result_fingerprint")
        if b_res and q_res:
            result_match = b_res == q_res
            result_observation = "same" if result_match else "different"
    elif quality_protocol_match is not None:
        protocol_match = quality_protocol_match

    # Token-level diffs
    token_diffs: list[dict[str, Any]] = []
    if b_bench and q_bench:
        b_results = {r["id"]: r for r in b_bench.get("results", []) if isinstance(r, dict)}
        q_results = {r["id"]: r for r in q_bench.get("results", []) if isinstance(r, dict)}
        for pid in sorted(set(b_results.keys()) | set(q_results.keys())):
            b_ids = b_results.get(pid, {}).get("generated_token_ids")
            q_ids = q_results.get(pid, {}).get("generated_token_ids")
            same = b_ids == q_ids if (b_ids is not None and q_ids is not None) else None
            token_diffs.append({"id": pid, "same": same, "baseline_tokens": b_ids, "quantized_tokens": q_ids})

    # Quantization fingerprints
    b_content_fp = None
    q_content_fp = None
    if b_quant:
        b_content_fp = b_quant.get("content_fingerprint")
    if q_quant:
        q_content_fp = q_quant.get("content_fingerprint")
    content_match = None
    if b_content_fp and q_content_fp:
        content_match = b_content_fp == q_content_fp

    return {
        "baseline_dir": str(b_dir),
        "quantized_dir": str(q_dir),
        "protocol_match": protocol_match,
        "result_match": result_match,
        "result_observation": result_observation,
        "result_is_pass_fail": False,  # explicit: observational, not gate
        "token_diffs": token_diffs,
        "baseline_protocol_fingerprint": (
            b_bench.get("suite", {}).get("fingerprint") if b_bench else b_quality_protocol
        ),
        "quantized_protocol_fingerprint": (
            q_bench.get("suite", {}).get("fingerprint") if q_bench else q_quality_protocol
        ),
        "baseline_dataset_fingerprint": b_dataset,
        "quantized_dataset_fingerprint": q_dataset,
        "dataset_match": dataset_match,
        "quality_report_schema_match": quality_report_schema_match,
        "model_revision_match": model_revision_match,
        "actual_dtype_match": actual_dtype_match,
        "actual_device_match": actual_device_match,
        "baseline_result_fingerprint": b_bench.get("result_fingerprint") if b_bench else None,
        "quantized_result_fingerprint": q_bench.get("result_fingerprint") if q_bench else None,
        "baseline_content_fingerprint": b_content_fp,
        "quantized_content_fingerprint": q_content_fp,
        "content_match": content_match,
        "quality_gate": quality_gate,
    }
