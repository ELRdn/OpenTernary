"""The Textual TUI: forms derived from the CLI, navigation, typed deletion, plan screen, launch rules."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import DataTable, Input, TabbedContent

from openternary.services import jobs as jobsvc
from openternary.ui.app import catalog
from openternary.ui.app.app import OpenTernaryApp, short
from openternary.ui.app.dialogs import Choice, JobForm, Picker, PlanScreen, TypedConfirm
from openternary.ui.app.widgets import Command, CommandBar, Menu, rank

ROOT = Path(__file__).resolve().parents[1]


# --- catalog ------------------------------------------------------------------------------------
def test_every_cli_command_has_a_form_except_the_tui_itself() -> None:
    names = {spec.name for spec in catalog.specs()}
    for expected in ("inspect", "quantize", "calibrate", "compare", "doctor", "setup", "update", "cache remove"):
        assert expected in names
    assert "ui" not in names
    for spec in catalog.specs():
        hidden = {"json_output", "quiet", "events_jsonl", "dry_run", "yes"}
        assert not hidden & {f.name for f in spec.fields}, spec.name  # the TUI supplies these itself


def test_forms_know_dry_run_and_confirmation() -> None:
    assert catalog.spec("quantize").dry_run and not catalog.spec("quantize").confirm
    assert catalog.spec("setup").dry_run and catalog.spec("setup").confirm
    assert not catalog.spec("doctor").dry_run


def test_build_args() -> None:
    compare = catalog.spec("compare")
    args = catalog.build_args(
        compare, {"baseline": "runs/a", "quantized": "runs/b", "candidate": "runs/c 'runs/d e'", "verbose": True}
    )
    assert args[0] == "compare" and args[-2:] == ["runs/a", "runs/b"]
    assert args.count("--candidate") == 2 and "runs/d e" in args and "--verbose" in args
    remove = catalog.spec("cache remove")
    with pytest.raises(ValueError, match="required"):
        catalog.build_args(remove, {"target": "", "root": "cache"})
    values = {"target": "cache/model-a", "root": "cache", "execute": True}
    assert catalog.build_args(remove, values) == ["cache", "remove", "--root", "cache", "--execute", "cache/model-a"]
    assert catalog.destructive_target(remove, values) == "model-a"
    assert catalog.destructive_target(remove, {**values, "execute": False}) is None


def test_long_run_names_keep_both_ends() -> None:
    name = "2026-11-03_gemma-4-E2B-it-qat-q4_0-unquantized-naive-g128-009"
    assert short(name).startswith("2026-11-03_gemma") and short(name).endswith("g128-009") and len(short(name)) == 44
    assert short("short") == "short"


# --- app ----------------------------------------------------------------------------------------
FAKE = r"""
import json, os, signal, sys, time
if os.environ.get("OPENTERNARY_BREAK_IS_INTERRUPT") and hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, signal.default_int_handler)
open(sys.argv[1], "x").write(json.dumps({"stage": "started", "elapsed_s": 0}) + "\n")
try:
    time.sleep(float(sys.argv[2]))
    print(json.dumps({"exit_code": 0, "result": {}, "error": None}))
except KeyboardInterrupt:
    print(json.dumps({"exit_code": 130, "result": {}, "error": "KeyboardInterrupt"})); sys.exit(130)
"""


def make_run(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    return path


@pytest.fixture
def app_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENTERNARY_LANG", "en")
    script = tmp_path / "fake.py"
    script.write_text(FAKE, encoding="utf-8")
    seconds = {"value": 30.0}
    manager = jobsvc.JobManager(
        tmp_path / "jobs",
        command=lambda args, events: [sys.executable, str(script), str(events), str(seconds["value"])],
    )

    def factory() -> OpenTernaryApp:
        return OpenTernaryApp(runs_root=tmp_path / "runs", manager=manager, check_updates=False, animate=False)

    factory.manager = manager  # type: ignore[attr-defined]
    factory.seconds = seconds  # type: ignore[attr-defined]
    yield factory
    for job in manager.running():
        manager.cancel(job, force=True)


def run(app: OpenTernaryApp, scenario: Any) -> None:
    async def main() -> None:
        async with app.run_test(size=(130, 40)) as pilot:
            await until(pilot, lambda: app.summary.get("runs") is not None)  # first load finished
            await scenario(app, pilot)

    asyncio.run(main())


async def until(pilot: Any, condition: Any, timeout: float = 15.0) -> None:
    """Wait for a UI state instead of a fixed sleep: screen changes, worker results and child
    processes take longer on busy CI runners (Windows especially)."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("UI did not reach the expected state in time")
        await pilot.pause(0.05)


KEYS = {"/": "slash", " ": "space", "-": "minus", ".": "full_stop", "_": "underscore", ":": "colon"}


async def type_text(pilot: Any, text: str) -> None:
    for char in text:
        await pilot.press(KEYS.get(char, char))


def bar(app: OpenTernaryApp) -> CommandBar:
    return app.query_one(CommandBar)


async def command(app: OpenTernaryApp, pilot: Any, text: str) -> None:
    """Type a command at the ❯ prompt and press Enter, as a person would."""
    await type_text(pilot, text)
    await pilot.press("enter")


def menu_ids(menu: Menu) -> list[str]:
    return [item.id for item in menu.items]


async def choose(pilot: Any, menu: Menu, item_id: str) -> None:
    """Move the ❯ to ``item_id`` with the arrow keys and press Enter."""
    menu.focus()
    await until(pilot, lambda: menu.has_focus)
    while menu.current is None or menu.current.id != item_id:
        await pilot.press("down")
    await pilot.press("enter")


def test_rank_puts_prefix_matches_first() -> None:
    commands = [Command(n, d, lambda words: None) for n, d in (("cancel", ""), ("calibrate", ""), ("upgrade", "x"))]
    assert [c.name for c in rank(commands, "ca")] == ["cancel", "calibrate"]
    assert [c.name for c in rank(commands, "rad")] == ["upgrade"]  # substring of the name
    assert len(rank(commands, "")) == 3


def test_prompt_has_focus_and_slash_lists_commands(app_factory: Any) -> None:
    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        prompt = bar(app).input
        assert prompt.has_focus  # the ❯ prompt is where you start, like a coding-agent CLI
        await type_text(pilot, "/")
        await until(pilot, lambda: bar(app).is_open)
        ids = menu_ids(bar(app).menu)
        assert ids[0] == "new" and {"calibrate", "compare", "runs", "probe", "upgrade", "cache:remove"} <= set(ids)
        await type_text(pilot, "cal")
        await until(pilot, lambda: menu_ids(bar(app).menu)[:1] == ["calibrate"])
        await pilot.press("tab")  # completes the highlighted command
        await until(pilot, lambda: prompt.value == "/calibrate ")
        assert not bar(app).is_open  # typing arguments now
        await pilot.press("escape")
        await until(pilot, lambda: prompt.value == "")

    run(app_factory(), scenario)


def test_slash_commands_switch_tabs_and_arrows_move_the_list(app_factory: Any, tmp_path: Path) -> None:
    make_run(tmp_path / "runs", "2026-11-01_a")
    make_run(tmp_path / "runs", "2026-11-02_b")

    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        tabs = app.query_one("#tabs", TabbedContent)
        for name in ("jobs", "doctor", "update", "home", "runs"):
            await command(app, pilot, f"/{name}")
            await until(pilot, lambda name=name: tabs.active == name)
        table = app.query_one("#runs-table", DataTable)
        await until(pilot, lambda: table.row_count == 2)
        assert bar(app).input.has_focus and table.cursor_row == 0
        await pilot.press("down")  # no command list open: ↑↓ move the current tab's list
        await until(pilot, lambda: table.cursor_row == 1)

    run(app_factory(), scenario)


def test_deleting_a_run_needs_its_exact_name(app_factory: Any, tmp_path: Path) -> None:
    run_dir = make_run(tmp_path / "runs", "2026-11-03_demo")

    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        await command(app, pilot, "/runs")
        await until(pilot, lambda: app.query_one("#runs-table", DataTable).row_count == 1)
        await command(app, pilot, "/delete")
        await until(pilot, lambda: isinstance(app.screen, TypedConfirm) and bool(app.screen.query("#actions")))
        menu = app.screen.query_one("#actions", Menu)
        assert menu.items[0].disabled and menu.current is not None and menu.current.id == "back"
        await type_text(pilot, "2026-11-03")
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert isinstance(app.screen, TypedConfirm) and run_dir.exists()  # a partial name does nothing
        await pilot.press("escape")
        await until(pilot, lambda: not isinstance(app.screen, TypedConfirm))
        await command(app, pilot, "/delete")
        await until(pilot, lambda: isinstance(app.screen, TypedConfirm) and bool(app.screen.query("#typed")))
        await type_text(pilot, "2026-11-03_demo")
        menu = app.screen.query_one("#actions", Menu)
        await until(pilot, lambda: not menu.items[0].disabled)
        await choose(pilot, menu, "delete")
        await until(pilot, lambda: not run_dir.exists())

    run(app_factory(), scenario)


def test_new_job_picker_form_plan_and_start(app_factory: Any) -> None:
    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        await command(app, pilot, "/new")
        await until(pilot, lambda: isinstance(app.screen, Picker) and bool(app.screen.query("#choices")))
        await type_text(pilot, "insp")
        await until(pilot, lambda: app.screen.query_one("#choices", Menu).current.id == "inspect")  # type: ignore[union-attr]
        await pilot.press("enter")
        await until(pilot, lambda: isinstance(app.screen, JobForm) and bool(app.screen.query("#f-config")))
        app.screen.query_one("#f-config", Input).value = "configs/gemma4-e2b.yaml"
        await choose(pilot, app.screen.query_one("#actions", Menu), "review")
        await until(pilot, lambda: isinstance(app.screen, PlanScreen) and bool(app.screen.query("#actions")))
        plan = app.screen
        assert isinstance(plan, PlanScreen) and plan.args == ["inspect", "--config", "configs/gemma4-e2b.yaml"]
        actions = plan.query_one("#actions", Menu)
        await until(pilot, lambda: not actions.items[0].disabled, timeout=60)  # once the dry run has passed
        assert "dry-run" in str(plan.query_one("#preview").render())
        await choose(pilot, actions, "start")
        await until(pilot, lambda: app.query_one("#tabs", TabbedContent).active == "jobs")
        (job,) = app_factory.manager.jobs()
        assert job.args == ["inspect", "--config", "configs/gemma4-e2b.yaml"] and job.status == "running"

    run(app_factory(), scenario)


def test_command_with_arguments_goes_straight_to_the_plan(app_factory: Any) -> None:
    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        await command(app, pilot, "/quantize --config configs/gemma4-e2b.yaml")
        await until(pilot, lambda: isinstance(app.screen, PlanScreen))
        assert app.screen.args == ["quantize", "--config", "configs/gemma4-e2b.yaml"]  # type: ignore[attr-defined]

    run(app_factory(), scenario)


def test_start_is_unavailable_until_the_preview_answers(app_factory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PlanScreen, "_preview", lambda self: None)  # a dry run that has not answered yet

    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        screen = PlanScreen(catalog.spec("quantize"), ["quantize", "--config", "configs/gemma4-e2b.yaml"])
        app.push_screen(screen)
        await until(pilot, lambda: app.screen is screen and bool(screen.query("#actions")))
        actions = screen.query_one("#actions", Menu)
        assert actions.items[0].disabled and actions.current is not None and actions.current.id == "back"
        actions.focus()
        await pilot.press("up", "enter")  # the ❯ cannot land on Start
        await until(pilot, lambda: app.screen is not screen)
        assert app_factory.manager.jobs() == []

    run(app_factory(), scenario)


def test_start_stays_unavailable_when_the_dry_run_fails(app_factory: Any) -> None:
    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        screen = PlanScreen(catalog.spec("quantize"), ["quantize", "--config", "configs/missing.yaml"])
        app.push_screen(screen)
        await until(pilot, lambda: app.screen is screen and bool(screen.query("#actions")))
        await until(pilot, lambda: "exit" in str(screen.query_one("#preview").render()), timeout=60)
        actions = screen.query_one("#actions", Menu)
        assert actions.items[0].disabled and actions.current is not None and actions.current.id == "back"
        actions.focus()
        await pilot.press("up", "enter")  # the ❯ cannot land on Start, so Enter means Back
        await until(pilot, lambda: app.screen is not screen)
        assert app_factory.manager.jobs() == []

    run(app_factory(), scenario)


def test_compare_picks_two_runs_from_lists(app_factory: Any, tmp_path: Path) -> None:
    base = make_run(tmp_path / "runs", "2026-11-01_base")
    cand = make_run(tmp_path / "runs", "2026-11-02_cand")

    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        await until(pilot, lambda: len(app.run_list) == 2)
        await command(app, pilot, "/compare")
        await until(pilot, lambda: isinstance(app.screen, Picker) and bool(app.screen.query("#choices")))
        await choose(pilot, app.screen.query_one("#choices", Menu), str(base))
        await until(pilot, lambda: isinstance(app.screen, Picker) and app.screen.title_text == "Candidate")
        choices = app.screen.query_one("#choices", Menu)
        assert menu_ids(choices) == [str(cand)]  # the baseline is not offered again
        await choose(pilot, choices, str(cand))
        await until(pilot, lambda: isinstance(app.screen, JobForm) and bool(app.screen.query("#f-baseline")))
        assert app.screen.query_one("#f-baseline", Input).value == str(base)
        assert app.screen.query_one("#f-quantized", Input).value == str(cand)

    run(app_factory(), scenario)


def test_quitting_with_a_running_job_asks_first(app_factory: Any) -> None:
    async def scenario(app: OpenTernaryApp, pilot: Any) -> None:
        job = app.start_job(["calibrate"])
        await command(app, pilot, "/quit")
        await until(pilot, lambda: isinstance(app.screen, Choice) and bool(app.screen.query("#actions")))
        assert menu_ids(app.screen.query_one("#actions", Menu)) == ["leave", "stop", "back"]
        await choose(pilot, app.screen.query_one("#actions", Menu), "back")
        await until(pilot, lambda: not isinstance(app.screen, Choice))
        assert app.is_running and app_factory.manager.refresh(job).status == "running"

    run(app_factory(), scenario)


# --- launch rules -------------------------------------------------------------------------------
def cli(*args: str) -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "OPENTERNARY_LANG": "en", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-m", "openternary", *args], cwd=ROOT, env=env, capture_output=True)


def test_without_a_terminal_no_arguments_still_prints_help() -> None:
    result = cli()
    assert result.returncode == 2 and b"Usage" in result.stdout  # unchanged CLI behaviour


def test_ui_command_refuses_without_a_terminal() -> None:
    result = cli("ui")
    assert result.returncode == 2 and b"interactive terminal" in result.stderr


def test_machine_paths_never_import_textual() -> None:
    env = {**os.environ, "OPENTERNARY_LANG": "en"}
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "openternary", "doctor", "--json"],
        cwd=ROOT,
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0
    assert b" textual" not in result.stderr  # -X importtime lists every imported module on stderr
