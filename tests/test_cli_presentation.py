"""Human-facing presentation: language, help layout, tables, single error panel, honest colours."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openternary.cli.main import app
from openternary.ui import brand, i18n
from openternary.ui.theme import badge

ROOT = Path(__file__).resolve().parents[1]


def cli(*args: str, lang: str = "en", encoding: str = "utf-8") -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "OPENTERNARY_LANG": lang, "PYTHONIOENCODING": encoding, "COLUMNS": "120", "PYTHONUTF8": "0"}
    return subprocess.run([sys.executable, "-m", "openternary", *args], cwd=ROOT, env=env, capture_output=True)


class Stream(io.StringIO):
    def __init__(self, encoding: str) -> None:
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._encoding


@pytest.mark.parametrize(
    ("env", "stream_encoding", "expected"),
    [
        ({"OPENTERNARY_LANG": "en"}, "utf-8", "en"),
        ({"OPENTERNARY_LANG": "ja"}, "utf-8", "ja"),
        ({"OPENTERNARY_LANG": "ja"}, "cp1252", "en"),  # cannot encode Japanese -> never crash
        ({"OPENTERNARY_LANG": "", "LANG": "ja_JP.UTF-8"}, "utf-8", "ja"),
        ({"OPENTERNARY_LANG": "", "LANG": "en_US.UTF-8"}, "utf-8", "en"),
        ({"OPENTERNARY_LANG": "fr"}, "utf-8", "en"),
    ],
)
def test_language_detection(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], stream_encoding: str, expected: str
) -> None:
    for name in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert i18n.detect(Stream(stream_encoding)) == expected


def test_every_message_has_both_languages() -> None:
    assert all(set(entry) == {"en", "ja"} and all(entry.values()) for entry in i18n.M.values())


def test_help_is_grouped_and_free_of_internal_phase_names() -> None:
    result = cli("--help")
    out = result.stdout.decode("utf-8")
    assert result.returncode == 0
    positions = [out.index(panel) for panel in ("Workflow", "Evaluate & compare", "Environment & artifacts")]
    assert positions == sorted(positions)
    assert out.index("inspect") < out.index("quantize") < out.index("calibrate") < out.index("Evaluate & compare")
    assert "Phase" not in out


def test_japanese_help_and_legacy_code_page_fallback() -> None:
    ja = cli("--help", lang="ja").stdout.decode("utf-8")
    assert "ワークフロー" in ja and "評価・比較" in ja
    legacy = cli("--help", lang="ja", encoding="cp1252")
    assert legacy.returncode == 0
    assert b"Workflow" in legacy.stdout  # English instead of an encoding error


def test_doctor_renders_tables_not_raw_json() -> None:
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert not result.stdout.lstrip().startswith("{")
    assert "ternary" in result.stdout and "torchao" in result.stdout


def test_machine_output_is_unchanged_by_the_human_view() -> None:
    result = CliRunner().invoke(app, ["doctor", "--json"])
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert envelope["result"]["backends"][0]["name"] == "ternary"


def assert_single_error_panel(err: str) -> None:
    """One panel and nothing printed outside it; independent of where long paths fold."""
    lines = [line for line in err.splitlines() if line.strip()]
    assert sum(line.lstrip().startswith(("╭", "┌")) for line in lines) == 1, err  # exactly one panel
    assert all(line.lstrip()[0] in "╭┌│╰└" for line in lines), err  # no second, plain echo


@pytest.mark.parametrize("depth", [1, 19, 40, 77])  # long paths fold inside the panel (Windows runners)
def test_errors_are_printed_once_with_context(tmp_path: Path, depth: int) -> None:
    missing = tmp_path / ("d" * depth) / "missing.yaml"
    result = cli("inspect", "--config", str(missing))
    err = result.stderr.decode("utf-8")
    assert result.returncode == 2
    assert result.stdout == b""
    assert_single_error_panel(err)
    assert "Config file not found" in err
    unfolded = "".join(line.strip().strip("│").strip() for line in err.splitlines())
    assert "missing.yaml" in unfolded


def test_context_becomes_the_json_error_when_there_is_no_exception() -> None:
    result = CliRunner().invoke(app, ["quality", "--dry-run", "--json"])
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert envelope["exit_code"] == 2
    assert envelope["error"] == "quality requires --data with a frozen dataset JSON"


def test_missing_dependency_gets_an_install_hint() -> None:
    from openternary.ui.render import error_panel

    console = __import__("rich.console", fromlist=["Console"]).Console(file=io.StringIO(), width=120)
    console.print(error_panel(4, "No module named 'torch'", None, "ImportError"))
    assert "uv sync --extra ml" in console.file.getvalue()


def test_completed_is_never_drawn_in_the_success_colour() -> None:
    assert badge("accepted").spans[0].style == brand.OK
    assert badge("completed").spans[0].style != brand.OK
    assert badge(None).plain == "—"
