"""Progress events (--events-jsonl): additive schema, throttling, safety, no-op without a stream."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openternary.cli.main import app
from openternary.services import reporting
from openternary.services.reporting import Operation, current, milestone, progress, track

ORIGINAL_KEYS = {"schema_version", "sequence", "command", "run_id", "stage", "completed", "total"}
ADDITIVE_KEYS = {"unit", "elapsed_s", "detail"}


def rows(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


@pytest.fixture
def operation() -> tuple[Operation, io.StringIO]:
    stream = io.StringIO()
    op = Operation("test", stream=stream)
    token = current.set(op)
    yield op, stream
    current.reset(token)


def test_rows_keep_original_keys_and_add_new_ones(operation: tuple[Operation, io.StringIO]) -> None:
    _, stream = operation
    progress("convert.tensors", 1, 4, unit="tensor", tensor="w")
    (row,) = rows(stream)
    assert set(row) == ORIGINAL_KEYS | ADDITIVE_KEYS
    assert row["schema_version"] == 1
    assert row["unit"] == "tensor" and row["detail"] == {"tensor": "w"}
    assert isinstance(row["elapsed_s"], float) and row["elapsed_s"] >= 0


def test_throttle_drops_bursts_but_keeps_first_and_last(operation: tuple[Operation, io.StringIO]) -> None:
    _, stream = operation
    for i in range(1000):
        progress("calibrate.steps", i, 1000, unit="step")
    progress("calibrate.steps", 1000, 1000, unit="step")
    got = rows(stream)
    assert got[0]["completed"] == 0
    assert got[-1]["completed"] == got[-1]["total"] == 1000
    assert len(got) < 20  # 1001 calls within milliseconds collapse to a handful of rows
    assert [r["sequence"] for r in got] == list(range(1, len(got) + 1))


def test_throttle_is_per_stage_and_milestones_always_pass(operation: tuple[Operation, io.StringIO]) -> None:
    _, stream = operation
    progress("a", 1, 10)
    progress("b", 1, 10)  # a different stage is not throttled by "a"
    progress("a", 2, 10)  # throttled
    milestone("calibrate.checkpoint", checkpoint="step_1.pt")
    milestone("calibrate.checkpoint", checkpoint="step_2.pt")
    assert [(r["stage"], r["completed"]) for r in rows(stream)] == [
        ("a", 1),
        ("b", 1),
        ("calibrate.checkpoint", None),
        ("calibrate.checkpoint", None),
    ]


def test_detail_values_are_json_safe(operation: tuple[Operation, io.StringIO]) -> None:
    _, stream = operation
    milestone("x", loss=math.nan, best=math.inf, path=Path("runs/a"), nested={"v": [1.5, -math.inf]})
    (row,) = rows(stream)
    assert row["detail"] == {"loss": None, "best": None, "path": str(Path("runs/a")), "nested": {"v": [1.5, None]}}


def test_track_reports_each_item(operation: tuple[Operation, io.StringIO], monkeypatch: pytest.MonkeyPatch) -> None:
    _, stream = operation
    monkeypatch.setattr(reporting, "MIN_INTERVAL_S", 0.0)
    assert list(track(["x", "y", "z"], "export.tensors", unit="tensor")) == ["x", "y", "z"]
    assert [r["completed"] for r in rows(stream)] == [0, 1, 2, 3]
    assert {r["total"] for r in rows(stream)} == {3}


def test_progress_without_an_operation_is_a_no_op() -> None:
    assert current.get() is None
    progress("convert.tensors", 1, 2)  # library use: nothing to write, must not raise
    milestone("calibrate.cache", status="reused")


def test_search_reports_candidate_completion(operation: tuple[Operation, io.StringIO], tmp_path: Path) -> None:
    """run_search with recorded measurements (no models) reports each finished candidate."""
    from openternary.services.optimize import SearchBudget, run_search
    from tests.test_search_product import fake_executor, inputs

    _, stream = operation
    cfg, profile, candidates = inputs(tmp_path)
    output = tmp_path / "search"
    run_search(cfg, profile, candidates, output, SearchBudget(), target_vram=200, executor=fake_executor([]))
    got = rows(stream)
    stages = [r["stage"] for r in got]
    assert stages.count("candidate_started") == len(candidates)  # pre-existing stage name unchanged
    done = [r for r in got if r["stage"] == "search.candidates"]
    assert [r["completed"] for r in done] == list(range(1, len(candidates) + 1))
    assert all(r["total"] == len(candidates) and r["detail"]["status"] == "completed" for r in done)


def test_cli_events_have_new_keys_and_monotonic_sequence(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    result = CliRunner().invoke(app, ["doctor", "--json", "--events-jsonl", str(events)])
    assert result.exit_code == 0
    got = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [r["stage"] for r in got] == ["started", "completed"]
    assert all(set(r) == ORIGINAL_KEYS | ADDITIVE_KEYS for r in got)
    assert [r["sequence"] for r in got] == [1, 2]


def test_events_file_parent_directories_are_created(tmp_path: Path) -> None:
    events = tmp_path / "runs" / "nested" / "events.jsonl"
    result = CliRunner().invoke(app, ["doctor", "--json", "--events-jsonl", str(events)])
    assert result.exit_code == 0, result.output
    assert events.is_file()


def test_existing_events_file_is_rejected_with_a_clear_message(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text("previous run\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["doctor", "--json", "--events-jsonl", str(events)])
    assert result.exit_code == 2
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert envelope["exit_code"] == 2 and "already exists" in envelope["error"]
    assert events.read_text(encoding="utf-8") == "previous run\n"  # never overwritten
