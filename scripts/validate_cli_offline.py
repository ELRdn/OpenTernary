"""Validate CLI contracts with empty caches and CPU fixtures, never pretrained models.

Run using the already selected Python environment; this script installs nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFERRED: tuple[str, ...] = ()


def main() -> int:
    scratch = Path(os.environ.get("OPENTERNARY_VALIDATION_ROOT", str(ROOT / ".tmp")))
    scratch.mkdir(parents=True, exist_ok=True)
    report_path = scratch / "cli-offline-validation.json"
    junit = scratch / "cli-offline-tests.xml"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "HIP_VISIBLE_DEVICES": "-1",
        "ROCR_VISIBLE_DEVICES": "-1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
    }
    with tempfile.TemporaryDirectory(prefix="cli-validation-", dir=scratch) as temporary:
        working = Path(temporary)
        shutil.copytree(ROOT / "configs", working / "configs")
        env["HF_HOME"] = str(working / "empty-hf-cache")
        env["HF_HUB_CACHE"] = str(working / "empty-hf-cache/hub")
        temporary_files = working / "temporary-files"
        temporary_files.mkdir()
        env.update({key: str(temporary_files) for key in ("TMP", "TEMP", "TMPDIR")})
        env["MYPY_CACHE_DIR"] = str(scratch / "mypy-cache")
        env["RUFF_CACHE_DIR"] = str(scratch / "ruff-cache")
        commands = [
            ([sys.executable, "-m", "ruff", "check", "src", "tests"], ROOT),
            ([sys.executable, "-m", "ruff", "format", "--check", "src", "tests"], ROOT),
            ([sys.executable, "-m", "mypy", "src/openternary"], ROOT),
            (
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(ROOT / "tests"),
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--tb=short",
                    f"--junitxml={junit}",
                    f"--basetemp={working / 'pytest'}",
                    *[f"--ignore={ROOT / 'tests' / name}" for name in DEFERRED],
                ],
                working,
            ),
        ]
        results = []
        for command, cwd in commands:
            result = subprocess.run(command, cwd=cwd, env=env)
            results.append({"command": command, "exit_code": result.returncode})
            if result.returncode:
                break
    report = {
        "scope": "offline_cli_contracts",
        "real_model_execution": False,
        "model_quality_acceptance": "not_run",
        "python": sys.executable,
        "checks": results,
        "deferred_test_files": list(DEFERRED),
    }
    if junit.is_file() and len(results) == len(commands):
        report["pytest"] = [suite.attrib for suite in ET.parse(junit).getroot()]
    report["passed"] = len(results) == len(commands) and all(row["exit_code"] == 0 for row in results)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Offline CLI report: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
