"""Compare matched eager-attention BF16 and saved ternary research reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from openternary.benchmark.acceptance import compare_quality_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline_bytes = args.baseline.read_bytes()
    candidate_bytes = args.candidate.read_bytes()
    baseline = json.loads(baseline_bytes)
    candidate = json.loads(candidate_bytes)
    measured = candidate.get("preceding_quality")
    if (
        not isinstance(measured, dict)
        or candidate.get("eager_attention") is not True
        or baseline.get("research_execution", {}).get("text_attention_implementation") != "eager"
        or baseline.get("research_execution", {}).get("source_identity") != measured.get("source_identity")
        or baseline.get("protocol_fingerprint") != measured.get("protocol_fingerprint")
        or baseline.get("dataset_fingerprint") != measured.get("dataset_fingerprint")
    ):
        raise ValueError("eager attention source, protocol, or data mismatch")
    gate = compare_quality_metrics(baseline["summary"], measured["summary"])
    result = {
        "status": "matched_eager_attention_research_comparison",
        "baseline_report_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "candidate_report_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "artifact_sha256": candidate["artifact_sha256"],
        "protocol_fingerprint": baseline["protocol_fingerprint"],
        "dataset_fingerprint": baseline["dataset_fingerprint"],
        "baseline_summary": baseline["summary"],
        "candidate_summary": measured["summary"],
        "quality_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"quality_gate": gate, "candidate_summary": measured["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
