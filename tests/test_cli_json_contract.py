"""Freeze the machine contract (`--json` envelope, exit codes) before the TUI work.

The TUI/Rich layer must be purely additive: for every probe below, stdout must stay
exactly one JSON document whose envelope values and result *shape* match the golden
file. Shapes (key names, nesting, list-vs-dict) are compared instead of raw values
so the test is stable across OS, Python and optional-dependency environments.

Regenerate intentionally with: OPENTERNARY_UPDATE_GOLDEN=1 pytest tests/test_cli_json_contract.py
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "cli_json_contract.json"

PROBES: dict[str, list[str]] = {
    "doctor": ["doctor"],
    "backends_list": ["backends", "list"],
    "backends_info_ternary": ["backends", "info", "ternary"],
    "backends_info_unknown": ["backends", "info", "no-such-backend"],
    "plan_default": ["plan"],
    "plugins": ["plugins"],
    "quantize_dry_run": ["quantize", "--dry-run"],
    "inspect_dry_run": ["inspect", "--dry-run"],
    "calibrate_dry_run": ["calibrate", "--dry-run"],
    "benchmark_dry_run": ["benchmark", "--dry-run"],
    "quality_dry_run_missing_profile": ["quality", "--dry-run"],
    "unknown_option": ["quantize", "--no-such-option"],
    "compare_without_runs": ["compare"],
    "cache_list_empty": ["cache", "list", "{tmp}"],
}

# Values that are part of the contract and must match exactly.
ENVELOPE_EXACT = ("schema_version", "command", "exit_code", "execution", "artifact_validation", "quality_acceptance")
# Environment-dependent keys whose value shape may legitimately vary.
OPAQUE_KEYS = {"dependencies", "runtime_probe", "plugins", "environment"}


def shape(value: Any, key: str | None = None) -> Any:
    if key in OPAQUE_KEYS:
        return "opaque"
    if isinstance(value, dict):
        return {k: shape(v, k) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return "scalar"


def run_probe(args: list[str], tmp: pathlib.Path) -> tuple[int, dict[str, Any]]:
    argv = [a.replace("{tmp}", str(tmp)) for a in args]
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
        "HIP_VISIBLE_DEVICES": "-1",
        "PYTHONIOENCODING": "utf-8",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "openternary", *argv, "--json"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        timeout=120,
    )
    stdout = proc.stdout.decode("utf-8").strip()
    assert stdout.count("\n") == 0, f"--json must print exactly one line on stdout, got:\n{stdout}"
    return proc.returncode, json.loads(stdout)


def describe(code: int, doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "process_exit": code,
        "envelope": {k: doc[k] for k in ENVELOPE_EXACT},
        "error_type": type(doc["error"]).__name__,
        "keys": sorted(doc),
        "result_shape": shape(doc["result"]),
    }


@pytest.fixture(scope="module")
def observed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("cache-root")
    return {name: describe(*run_probe(args, tmp)) for name, args in PROBES.items()}


def test_json_contract_matches_golden(observed: dict[str, Any]) -> None:
    if os.environ.get("OPENTERNARY_UPDATE_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(observed, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert sorted(observed) == sorted(golden)
    for name in golden:
        assert observed[name] == golden[name], f"--json contract changed for probe {name!r}"


def test_process_exit_equals_envelope_exit(observed: dict[str, Any]) -> None:
    for name, row in observed.items():
        assert row["process_exit"] == row["envelope"]["exit_code"], name
