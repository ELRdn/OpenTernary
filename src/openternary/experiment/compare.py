"""Compare utility — behavioral Δ without quality claim (Phase 3)."""

from __future__ import annotations

import json
import pathlib
from typing import Any


def _load_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except Exception:
        return None


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

    b_bench = _find_benchmark_json(b_dir)
    q_bench = _find_benchmark_json(q_dir)

    b_quant = _find_quantization_json(b_dir)
    q_quant = _find_quantization_json(q_dir)

    # Protocol comparison
    protocol_match: bool | None = None
    result_match: bool | None = None
    result_observation = "unknown"

    if b_bench and q_bench:
        b_proto = b_bench.get("suite", {}).get("fingerprint") if isinstance(b_bench.get("suite"), dict) else None
        q_proto = q_bench.get("suite", {}).get("fingerprint") if isinstance(q_bench.get("suite"), dict) else None
        if b_proto and q_proto:
            protocol_match = b_proto == q_proto
        # result
        b_res = b_bench.get("result_fingerprint")
        q_res = q_bench.get("result_fingerprint")
        if b_res and q_res:
            result_match = b_res == q_res
            result_observation = "same" if result_match else "different"

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
        "baseline_protocol_fingerprint": b_bench.get("suite", {}).get("fingerprint") if b_bench else None,
        "quantized_protocol_fingerprint": q_bench.get("suite", {}).get("fingerprint") if q_bench else None,
        "baseline_result_fingerprint": b_bench.get("result_fingerprint") if b_bench else None,
        "quantized_result_fingerprint": q_bench.get("result_fingerprint") if q_bench else None,
        "baseline_content_fingerprint": b_content_fp,
        "quantized_content_fingerprint": q_content_fp,
        "content_match": content_match,
    }
