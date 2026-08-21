"""CLI compare tests — behavioral Δ observational."""

from __future__ import annotations

import json
import pathlib
import shutil
import uuid

from typer.testing import CliRunner

from openternary.cli.main import app

runner = CliRunner()


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
