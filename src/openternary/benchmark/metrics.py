"""Benchmark metrics helpers — summary ↔ metrics.json 変換."""

from __future__ import annotations

from typing import Any


def to_metrics(
    runner_result: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """runner の result dict から metrics.json 用の集約 dict を生成."""
    summary = runner_result.get("summary", {})
    suite_info = runner_result.get("suite", {})
    model = runner_result.get("model", {})
    generation = runner_result.get("generation", {})
    # suite_info が dict の場合と文字列の場合の両対応
    suite_name = suite_info.get("name") if isinstance(suite_info, dict) else str(suite_info)
    return {
        "status": "completed",
        "run_id": run_id,
        "suite": suite_name,
        "suite_version": suite_info.get("version") if isinstance(suite_info, dict) else None,
        "protocol_fingerprint": suite_info.get("fingerprint") if isinstance(suite_info, dict) else None,
        "result_fingerprint": runner_result.get("result_fingerprint"),
        "model": model,
        "generation": generation,
        "thinking": runner_result.get("thinking"),
        "summary": summary,
        "num_prompts": summary.get("num_prompts"),
        "avg_generation_latency_ms": summary.get("avg_generation_latency_ms"),
        "output_tokens_per_second": summary.get("output_tokens_per_second"),
    }
