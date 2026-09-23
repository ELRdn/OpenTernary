"""CLI compare tests — behavioral Δ observational."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import shutil
import uuid

import pytest
from typer.testing import CliRunner

from openternary.cli.main import app

runner = CliRunner()


def _quality_v2_payload(
    summary: dict[str, float | int],
    *,
    dataset: str = "data-v1",
    revision: str = "revision-v1",
    actual_dtype: str = "torch.bfloat16",
    actual_device: str = "cuda:0",
    response: str = "answer",
) -> dict[str, object]:
    protocol_payload = {
        "version": "quality-runner-v1",
        "split": "validation",
        "max_length": 128,
        "stride": 64,
    }
    protocol_fingerprint = hashlib.sha256(
        json.dumps(
            protocol_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "report_schema_version": 2,
        "protocol": protocol_payload,
        "protocol_fingerprint": protocol_fingerprint,
        "dataset_fingerprint": dataset,
        "model": {
            "revision": revision,
            "actual_dtype": actual_dtype,
            "actual_device": actual_device,
        },
        "instructions": {
            "results": [
                {
                    "id": "case-1",
                    "response_text": response,
                    "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                    "diagnostics": {"flags": []},
                    "exact_match": True,
                }
            ],
            "exact_match_scored": 1,
            "exact_match_accuracy": float(summary["instruction_score"]) / 100.0,
            "flagged_outputs": int(summary["collapse_count"]),
        },
        "perplexity": {
            "by_language": {
                language: {
                    "nll_sum": math.log(float(summary[summary_key])),
                    "scored_tokens": 1,
                    "mean_nll": math.log(float(summary[summary_key])),
                    "perplexity": float(summary[summary_key]),
                }
                for language, summary_key in (("en", "general_ppl"), ("ja", "japanese_ppl"))
            }
        },
        "summary": summary,
    }


def _make_benchmark_dir(base: pathlib.Path, name: str, result_ids: list[int]) -> pathlib.Path:
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    bench = {
        "suite": {"name": "smoke", "version": "1", "fingerprint": "sha256:abc"},
        "result_fingerprint": f"sha256:{''.join(str(x) for x in result_ids)}",
        "results": [{"id": f"p{i}", "generated_token_ids": [result_ids[i]]} for i in range(len(result_ids))],
    }
    (run_dir / "benchmark.json").write_text(json.dumps(bench), encoding="utf-8")
    return run_dir


def test_cli_compare_reports_behavioral_delta() -> None:
    tmp = pathlib.Path.cwd() / f"test_compare_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        base = _make_benchmark_dir(tmp, "baseline", [1, 2, 3])
        quant_same = _make_benchmark_dir(tmp, "quant_same", [1, 2, 3])
        quant_diff = _make_benchmark_dir(tmp, "quant_diff", [1, 2, 99])

        # same result -> observation same
        result = runner.invoke(app, ["compare", str(base), str(quant_same)])
        assert result.exit_code == 0, result.output
        assert "observational" in result.output.lower() or "observation" in result.output.lower()

        # different result -> observation different, but still exit 0 (not pass/fail)
        result2 = runner.invoke(app, ["compare", str(base), str(quant_diff)])
        assert result2.exit_code == 0, result2.output
        assert "different" in result2.output.lower() or "same" in result2.output.lower()

        # Check compare.json written somewhere (run dir created)
        # compare creates a new run dir in runs/ — we don't know name, but result should contain token diffs
        from openternary.experiment.compare import compare_runs

        cmp = compare_runs(base, quant_diff)
        assert cmp["protocol_match"] is True
        assert cmp["result_observation"] == "different"
        assert cmp["result_is_pass_fail"] is False
        cmp_same = compare_runs(base, quant_same)
        assert cmp_same["result_observation"] == "same"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_compare_missing_args_exit_2() -> None:
    result = runner.invoke(app, ["compare"])
    assert result.exit_code == 2


def test_compare_runs_preserves_legacy_observations_without_acceptance(tmp_path) -> None:
    from openternary.experiment.compare import compare_runs

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "quality.json").write_text(
        json.dumps(
            {
                "protocol_fingerprint": "proto-v1",
                "dataset_fingerprint": "data-v1",
                "summary": {
                    "general_ppl": 10.0,
                    "japanese_ppl": 20.0,
                    "instruction_score": 80.0,
                    "collapse_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (candidate / "quality.json").write_text(
        json.dumps(
            {
                "protocol_fingerprint": "proto-v1",
                "dataset_fingerprint": "data-v1",
                "summary": {
                    "general_ppl": 9.8,
                    "japanese_ppl": 20.2,
                    "instruction_score": 79.0,
                    "collapse_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    result = compare_runs(baseline, candidate)
    assert result["protocol_match"] is True
    assert result["dataset_match"] is True
    assert result["baseline_protocol_fingerprint"] == "proto-v1"
    assert result["quantized_protocol_fingerprint"] == "proto-v1"
    assert result["baseline_dataset_fingerprint"] == "data-v1"
    assert result["quantized_dataset_fingerprint"] == "data-v1"
    assert result["quality_gate"]["accepted"] is False
    assert result["quality_gate"]["status"] == "insufficient_evidence"

    output = tmp_path / "comparison"
    cli_result = runner.invoke(app, ["compare", str(baseline), str(candidate), "--output", str(output)])
    assert cli_result.exit_code == 0, cli_result.output
    acceptance = json.loads((output / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["accepted"] is False


def test_compare_runs_rejects_mismatched_quality_protocol_or_data(tmp_path) -> None:
    from openternary.experiment.compare import compare_runs

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    summary = {
        "general_ppl": 10.0,
        "japanese_ppl": 20.0,
        "instruction_score": 80.0,
        "collapse_count": 0,
    }
    (baseline / "quality.json").write_text(
        json.dumps({"protocol_fingerprint": "proto-v1", "dataset_fingerprint": "data-v1", "summary": summary}),
        encoding="utf-8",
    )
    (candidate / "quality.json").write_text(
        json.dumps({"protocol_fingerprint": "proto-v2", "dataset_fingerprint": "data-v1", "summary": summary}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="quality protocol fingerprint mismatch"):
        compare_runs(baseline, candidate)

    (candidate / "quality.json").write_text(
        json.dumps({"protocol_fingerprint": "proto-v1", "dataset_fingerprint": "data-v2", "summary": summary}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="quality dataset fingerprint mismatch"):
        compare_runs(baseline, candidate)


def test_compare_runs_validates_schema_v2_report_integrity_and_runtime(tmp_path) -> None:
    from openternary.experiment.compare import compare_runs

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    control = {
        "general_ppl": 10.0,
        "japanese_ppl": 20.0,
        "instruction_score": 100.0,
        "collapse_count": 0,
    }
    improved = {
        "general_ppl": 9.0,
        "japanese_ppl": 18.0,
        "instruction_score": 100.0,
        "collapse_count": 0,
    }
    (baseline / "quality.json").write_text(json.dumps(_quality_v2_payload(control)), encoding="utf-8")
    (candidate / "quality.json").write_text(json.dumps(_quality_v2_payload(improved)), encoding="utf-8")

    result = compare_runs(baseline, candidate)
    assert result["quality_report_schema_match"] is True
    assert result["model_revision_match"] is True
    assert result["actual_dtype_match"] is True
    assert result["actual_device_match"] is True

    tampered = _quality_v2_payload(improved)
    tampered["instructions"]["results"][0]["response_sha256"] = "0" * 64  # type: ignore[index]
    (candidate / "quality.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="response_sha256"):
        compare_runs(baseline, candidate)

    tampered = _quality_v2_payload(improved)
    tampered["protocol"]["stride"] = 32  # type: ignore[index]
    (candidate / "quality.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol_fingerprint does not match protocol"):
        compare_runs(baseline, candidate)

    tampered = _quality_v2_payload(improved)
    tampered["summary"]["general_ppl"] = 8.0  # type: ignore[index]
    (candidate / "quality.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="general_ppl does not match perplexity"):
        compare_runs(baseline, candidate)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("revision", "revision-v2", "model revision mismatch"),
        ("actual_dtype", "torch.float32", "actual dtype mismatch"),
        ("actual_device", "cpu", "actual device mismatch"),
    ],
)
def test_compare_runs_rejects_schema_v2_runtime_mismatch(tmp_path, field, value, message) -> None:
    from openternary.experiment.compare import compare_runs

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    summary = {
        "general_ppl": 10.0,
        "japanese_ppl": 20.0,
        "instruction_score": 100.0,
        "collapse_count": 0,
    }
    control = _quality_v2_payload(summary)
    changed = _quality_v2_payload(summary)
    changed["model"][field] = value  # type: ignore[index]
    (baseline / "quality.json").write_text(json.dumps(control), encoding="utf-8")
    (candidate / "quality.json").write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        compare_runs(baseline, candidate)


def test_compare_runs_rejects_malformed_quality_json(tmp_path) -> None:
    from openternary.experiment.compare import compare_runs

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "quality.json").write_text("{not-json", encoding="utf-8")
    (candidate / "quality.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid quality JSON"):
        compare_runs(baseline, candidate)
