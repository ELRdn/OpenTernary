"""The OpenTernary TUI: Home, Runs, Jobs, Compare, Doctor, Update.

Heavy work always runs as the ordinary CLI in a child process (``services.jobs``); this process only
draws. Reading state (runs, doctor, update status) happens in worker threads so the UI never stalls.
"""

from __future__ import annotations

import datetime
import os
import time
from pathlib import Path
from typing import Any, Literal

from rich.console import Group, RenderableType
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.theme import Theme
from textual.widgets import Button, Checkbox, DataTable, Footer, Label, Select, Static, TabbedContent, TabPane

from openternary import __version__
from openternary.services import jobs as jobsvc
from openternary.services import runs as runsvc
from openternary.ui import brand
from openternary.ui.app import catalog
from openternary.ui.app.dialogs import Choice, JobForm, PlanScreen, TypedConfirm
from openternary.ui.i18n import t
from openternary.ui.theme import badge

YORU = Theme(
    name="yoru",
    primary=brand.ACCENT,
    secondary=brand.INFO,
    accent=brand.ACCENT,
    foreground=brand.TEXT,
    background=brand.BG,
    surface=brand.SURFACE,
    panel=brand.BORDER,
    success=brand.OK,
    warning=brand.WARN,
    error=brand.ERR,
    dark=True,
)

TABS = ("home", "runs", "jobs", "compare", "doctor", "update")
FPS = 12


def reduced_motion() -> bool:
    flags = (os.environ.get("OPENTERNARY_REDUCED_MOTION"), os.environ.get("NO_MOTION"))
    return any(v not in (None, "", "0") for v in flags)


def clock(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


def short(name: str, width: int = 44) -> str:
    """Run names differ at both ends (model slug, -001 suffix): keep both, elide the middle."""
    if len(name) <= width:
        return name
    head = (width - 1) // 2
    return name[:head] + "…" + name[len(name) - (width - 1 - head) :]


def ago(moment: datetime.datetime) -> str:
    return moment.astimezone().strftime("%m-%d %H:%M")


def kv(rows: list[tuple[str, RenderableType | str | None]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=brand.MUTED, no_wrap=True)
    table.add_column()
    for key, value in rows:
        table.add_row(key, value if value is not None else Text("—", style="dim"))
    return table


# --- Home ---------------------------------------------------------------------------------------
class Mark(Static):
    """The tomoe symbol, turning clockwise (owner decision D17) unless motion is reduced."""

    def __init__(self, animate: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.frame = 0
        self.animate_mark = animate and not reduced_motion()

    def on_mount(self) -> None:
        self.update(brand.mark(24, 0))
        if self.animate_mark:
            self.set_interval(1 / FPS, self._tick)

    def _tick(self) -> None:
        self.frame = (self.frame + 1) % brand.mark_frame_count()
        self.update(brand.mark(24, self.frame))


class Home(Vertical):
    def compose(self) -> ComposeResult:
        yield Static(id="banner")
        with Horizontal(id="home-body"):
            yield Mark(animate=self.app.motion, id="mark")  # type: ignore[attr-defined]
            with Vertical(id="home-info"):
                yield Static(id="home-summary")
                with Horizontal(classes="buttons left"):
                    yield Button(t("tui.new_job"), id="home-new", variant="primary")
                    yield Button(t("tui.tab.runs"), id="home-runs")
                    yield Button(t("tui.tab.doctor"), id="home-doctor")

    def on_resize(self) -> None:
        self.query_one("#banner", Static).update(brand.banner(self.size.width))

    def show(self, summary: dict[str, Any]) -> None:
        self.query_one("#banner", Static).update(brand.banner(self.size.width or 120))
        title = Text.assemble((f"OpenTernary {__version__}", f"bold {brand.ACCENT}"))
        tagline = Text(t("tui.tagline"), style=brand.MUTED)
        rows: list[tuple[str, RenderableType | str | None]] = [
            (t("tui.update"), summary.get("update")),
            ("PyTorch", summary.get("torch")),
            (t("tui.runs_folder"), summary.get("runs")),
            (t("tui.last_run"), summary.get("last_run")),
            (t("tui.tab.jobs"), summary.get("jobs")),
        ]
        trits = Text()
        for trit, label in ((1, "+1"), (0, " 0"), (-1, "-1")):
            trits.append("━━ ", style=brand.TRIT[trit])
            trits.append(label + "   ")
        self.query_one("#home-summary", Static).update(Group(title, tagline, Text(""), kv(rows), Text(""), trits))


# --- Runs ---------------------------------------------------------------------------------------
class Runs(Horizontal):
    BINDINGS = [
        Binding("x", "delete", t("tui.delete")),
        Binding("b", "baseline", t("tui.as_baseline")),
        Binding("c", "candidate", t("tui.as_candidate")),
        Binding("r", "refresh", t("tui.refresh")),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="runs-left"):
            yield Label("", id="runs-root", classes="muted")
            table: DataTable[Any] = DataTable(id="runs-table", cursor_type="row", zebra_stripes=True)
            yield table
        with VerticalScroll(id="runs-detail", classes="detail"):
            yield Static(t("tui.select_run"), id="run-info")

    def on_mount(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.add_columns(t("tui.col.status"), t("tui.col.quality"), t("col.kind"), t("col.name"), t("tui.col.started"))
        self.runs: dict[str, runsvc.RunInfo] = {}

    def show(self, root: Path, runs: list[runsvc.RunInfo]) -> None:
        self.query_one("#runs-root", Label).update(t("tui.runs_in", path=str(root.resolve())))
        table = self.query_one("#runs-table", DataTable)
        selected = self.selected()
        table.clear()
        self.runs = {str(run.path): run for run in runs}
        for run in runs:
            table.add_row(
                badge(run.status),
                badge(run.quality),
                run.kind or "—",
                short(run.name),
                ago(run.started),
                key=str(run.path),
            )
        if not runs:
            self.query_one("#run-info", Static).update(t("tui.no_runs", path=str(root)))
        elif selected and str(selected.path) in self.runs:
            table.move_cursor(row=table.get_row_index(str(selected.path)))

    def selected(self) -> runsvc.RunInfo | None:
        table = self.query_one("#runs-table", DataTable)
        if not table.row_count:
            return None
        key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        return self.runs.get(str(key))

    @on(DataTable.RowHighlighted, "#runs-table")
    def _highlighted(self) -> None:
        run = self.selected()
        if run is None:
            return
        files = Table.grid(padding=(0, 2))
        files.add_column()
        files.add_column(justify="right", style=brand.MUTED)
        for name, size in runsvc.files(run.path, limit=25):
            files.add_row(name, f"{size:,}")
        rows: list[tuple[str, RenderableType | str | None]] = [
            (t("col.name"), run.name),
            ("run_id", run.run_id),
            (t("tui.col.status"), badge(run.status)),
            (t("tui.col.quality"), badge(run.quality)),
            (t("col.kind"), run.kind),
            (t("plan.model"), run.model),
            (t("tui.col.started"), run.started.astimezone().strftime("%Y-%m-%d %H:%M:%S")),
            (t("tui.path"), str(run.path)),
        ]
        if run.error:
            rows.append((t("tui.error"), Text(run.error, style=brand.ERR)))
        self.query_one("#run-info", Static).update(Group(kv(rows), Text(""), Text(t("tui.files"), "bold"), files))

    def action_refresh(self) -> None:
        self.app.refresh_runs()  # type: ignore[attr-defined]

    def action_delete(self) -> None:
        run = self.selected()
        if run is not None:
            self.app.delete_run(run)  # type: ignore[attr-defined]

    def action_baseline(self) -> None:
        if run := self.selected():
            self.app.pick_for_compare("baseline", run)  # type: ignore[attr-defined]

    def action_candidate(self) -> None:
        if run := self.selected():
            self.app.pick_for_compare("candidate", run)  # type: ignore[attr-defined]


# --- Jobs ---------------------------------------------------------------------------------------
def stages_view(progress: jobsvc.Progress, running: bool) -> RenderableType:
    if not progress.stages:
        return Text(t("tui.waiting_events") if running else t("tui.no_events"), style=brand.MUTED)
    table = Table(box=None, pad_edge=False, show_header=True, header_style=brand.MUTED, expand=True)
    table.add_column(t("tui.col.stage"), no_wrap=True, ratio=3)
    table.add_column("", ratio=3)
    table.add_column(t("tui.col.progress"), justify="right", no_wrap=True, ratio=2)
    table.add_column("ETA", justify="right", no_wrap=True, ratio=1)
    table.add_column(t("tui.col.detail"), ratio=3, overflow="ellipsis", no_wrap=True)
    for stage in progress.stages.values():
        current = stage.name == progress.current and running and not stage.done
        glyph = Text("◐ " if current else "✔ " if stage.done else "· ", style=brand.INFO if current else brand.MUTED)
        name = Text.assemble(glyph, (stage.name, "bold" if current else ""))
        if stage.total:
            bar: RenderableType = ProgressBar(
                total=float(stage.total),
                completed=float(stage.completed or 0),
                width=24,
                complete_style=brand.ACCENT,
                finished_style=brand.INFO,
                style=brand.BORDER,
            )
            count = f"{stage.completed or 0}/{stage.total}" + (f" {stage.unit}" if stage.unit else "")
        else:
            bar, count = Text(""), "—" if stage.completed is None else str(stage.completed)
        detail = "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in stage.detail.items() if v is not None
        )
        table.add_row(name, bar, count, clock(stage.eta_s()) if current else "", Text(detail, style=brand.MUTED))
    return table


class Jobs(Vertical):
    BINDINGS = [
        Binding("n", "new", t("tui.new_job")),
        Binding("k", "cancel", t("tui.cancel_job")),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="jobs-left"):
            table: DataTable[Any] = DataTable(id="jobs-table", cursor_type="row", zebra_stripes=True)
            yield table
        with Vertical(id="jobs-detail", classes="detail"):
            yield Static(t("tui.no_jobs"), id="job-head")
            yield Static("", id="job-stages")
            yield Label(t("tui.log"), classes="section")
            with VerticalScroll(id="job-log-box"):
                yield Static("", id="job-log")

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.add_column(t("tui.col.status"), key="status", width=34)
        table.add_column(t("tui.col.command"), key="command")
        table.add_column(t("tui.col.started"), key="started")
        table.add_column(t("tui.col.elapsed"), key="elapsed")
        self.tails: dict[str, tuple[jobsvc.EventTail, jobsvc.Progress]] = {}
        self.jobs: dict[str, jobsvc.Job] = {}
        self.focus_job: str | None = None

    def progress(self, job: jobsvc.Job) -> jobsvc.Progress:
        if job.id not in self.tails:
            self.tails[job.id] = (jobsvc.EventTail(job.events_path), jobsvc.Progress())
        tail, progress = self.tails[job.id]
        for row in tail.read():
            progress.feed(row)
        return progress

    def show(self, jobs: list[jobsvc.Job]) -> None:
        table = self.query_one("#jobs-table", DataTable)
        current = self.selected_id()
        same = list(self.jobs) == [job.id for job in jobs]
        self.jobs = {job.id: job for job in jobs}
        if not same:
            table.clear()
        for job in jobs:
            progress = self.progress(job) if job.status == "running" else None
            elapsed = (job.finished_at or time.time()) - job.started_at
            status = badge(job.status)
            if progress and progress.current:
                status.append(f"  {progress.current}", style=brand.MUTED)
            if same:  # update in place: keeps the cursor and scroll position while jobs tick
                table.update_cell(job.id, "status", status)
                table.update_cell(job.id, "elapsed", clock(elapsed))
                continue
            table.add_row(
                status,
                job.command,
                datetime.datetime.fromtimestamp(job.started_at).strftime("%m-%d %H:%M"),
                clock(elapsed),
                key=job.id,
            )
        target = self.focus_job or current
        if target and target in self.jobs:
            table.move_cursor(row=table.get_row_index(target))
            self.focus_job = None
        self.show_detail()

    def selected_id(self) -> str | None:
        table = self.query_one("#jobs-table", DataTable)
        if not table.row_count:
            return None
        return str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)

    @on(DataTable.RowHighlighted, "#jobs-table")
    def show_detail(self) -> None:
        job = self.jobs.get(self.selected_id() or "")
        if job is None:
            self.query_one("#job-head", Static).update(t("tui.no_jobs"))
            self.query_one("#job-stages", Static).update("")
            self.query_one("#job-log", Static).update("")
            return
        progress = self.progress(job)
        running = job.status == "running"
        line = "openternary " + " ".join(job.args)
        rows: list[tuple[str, RenderableType | str | None]] = [
            (t("tui.col.status"), badge(job.status)),
            (t("tui.col.command"), Text(line, style=brand.ACCENT)),
            (t("tui.col.elapsed"), clock((job.finished_at or time.time()) - job.started_at)),
        ]
        if running and job.cancel_requested:
            rows.append(("", Text(t("tui.cancel_sent"), style=brand.WARN)))
        envelope = job.envelope() if not running else None
        if envelope:
            result = envelope.get("result") or {}
            rows += [
                ("exit", str(envelope.get("exit_code"))),
                (t("tui.col.quality"), badge(envelope.get("quality_acceptance"))),
            ]
            if result.get("run_dir"):
                rows.append((t("tui.path"), str(result["run_dir"])))
            if envelope.get("error"):
                rows.append((t("tui.error"), Text(str(envelope["error"]), style=brand.ERR)))
        self.query_one("#job-head", Static).update(kv(rows))
        self.query_one("#job-stages", Static).update(stages_view(progress, running))
        box = self.query_one("#job-log-box", VerticalScroll)
        following = box.scroll_y >= box.max_scroll_y - 1  # keep following the end unless scrolled up
        self.query_one("#job-log", Static).update(Text("\n".join(job.stderr_tail(200)), style=brand.MUTED))
        if following:
            box.call_after_refresh(box.scroll_end, animate=False)

    def action_new(self) -> None:
        self.app.new_job()  # type: ignore[attr-defined]

    def action_cancel(self) -> None:
        job = self.jobs.get(self.selected_id() or "")
        if job is not None:
            self.app.cancel_job(job)  # type: ignore[attr-defined]


# --- Compare ------------------------------------------------------------------------------------
class Compare(Vertical):
    def compose(self) -> ComposeResult:
        yield Static(t("tui.compare_intro"), classes="muted")
        yield Label(t("tui.baseline"), classes="section")
        yield Select([], id="cmp-baseline", prompt=t("tui.pick_run"))
        yield Label(t("tui.candidate"), classes="section")
        yield Select([], id="cmp-candidate", prompt=t("tui.pick_run"))
        yield Checkbox(t("tui.require_acceptance"), id="cmp-require")
        with Horizontal(classes="buttons left"):
            yield Button(t("tui.compare_go"), id="cmp-go", variant="primary")
        yield Static("", id="cmp-error", classes="error")

    def show(self, runs: list[runsvc.RunInfo]) -> None:
        options = [(f"{run.name}  ({run.kind or '—'} · {run.status})", str(run.path)) for run in runs]
        for ident in ("#cmp-baseline", "#cmp-candidate"):
            select = self.query_one(ident, Select)
            value = select.value
            select.set_options(options)
            if value is not Select.NULL and any(v == value for _, v in options):
                select.value = value

    def pick(self, which: str, run: runsvc.RunInfo) -> None:
        self.query_one(f"#cmp-{which}", Select).value = str(run.path)

    @on(Button.Pressed, "#cmp-go")
    def _go(self) -> None:
        baseline = self.query_one("#cmp-baseline", Select).value
        candidate = self.query_one("#cmp-candidate", Select).value
        if baseline is Select.NULL or candidate is Select.NULL or baseline == candidate:
            self.query_one("#cmp-error", Static).update(t("tui.compare_pick_two"))
            return
        self.query_one("#cmp-error", Static).update("")
        values = {
            "baseline": baseline,
            "quantized": candidate,
            "require_acceptance": self.query_one("#cmp-require", Checkbox).value,
        }
        self.app.new_job("compare", values)  # type: ignore[attr-defined]


# --- Doctor / Update ----------------------------------------------------------------------------
class Doctor(VerticalScroll):
    def compose(self) -> ComposeResult:
        with Horizontal(classes="buttons left"):
            yield Button(t("tui.refresh"), id="doc-refresh")
            yield Button(t("tui.probe_runtime"), id="doc-probe")
            yield Button(t("tui.setup_torch"), id="doc-setup", variant="primary")
        yield Static(t("tui.loading"), id="doctor-view")

    def show(self, renderable: RenderableType) -> None:
        self.query_one("#doctor-view", Static).update(renderable)

    @on(Button.Pressed, "#doc-refresh")
    def _refresh(self) -> None:
        self.app.refresh_doctor()  # type: ignore[attr-defined]

    @on(Button.Pressed, "#doc-probe")
    def _probe(self) -> None:
        self.app.new_job("doctor", {"probe_runtime": True})  # type: ignore[attr-defined]

    @on(Button.Pressed, "#doc-setup")
    def _setup(self) -> None:
        self.app.new_job("setup")  # type: ignore[attr-defined]


class Update(VerticalScroll):
    def compose(self) -> ComposeResult:
        with Horizontal(classes="buttons left"):
            yield Button(t("tui.check_now"), id="upd-check")
            yield Button(t("tui.update_now"), id="upd-go", variant="primary")
        yield Static(t("tui.loading"), id="update-view")

    def show(self, renderable: RenderableType, can_update: bool) -> None:
        self.query_one("#update-view", Static).update(renderable)
        self.query_one("#upd-go", Button).disabled = not can_update

    @on(Button.Pressed, "#upd-check")
    def _check(self) -> None:
        self.app.refresh_update(force=True)  # type: ignore[attr-defined]

    @on(Button.Pressed, "#upd-go")
    def _go(self) -> None:
        self.app.new_job("update")  # type: ignore[attr-defined]


# --- App ----------------------------------------------------------------------------------------
class OpenTernaryApp(App[None]):
    TITLE = "OpenTernary"
    CSS = """
    Screen { background: $background; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 1 2; }
    .muted { color: $text-muted; }
    .error { color: $error; }
    .warn { color: $warning; margin-top: 1; }
    .section { margin-top: 1; color: $text-muted; text-style: bold; }
    .buttons { height: auto; margin-top: 1; align-horizontal: right; }
    .buttons.left { align-horizontal: left; margin-top: 0; margin-bottom: 1; }
    .buttons Button { margin-right: 1; }
    .detail { border-left: tall $panel; padding: 0 0 0 2; }
    #banner { height: auto; margin-bottom: 1; }
    #home-body { height: auto; }
    #mark { width: 28; height: auto; margin-right: 3; }
    #home-info { height: auto; width: 1fr; }
    #home-summary { height: auto; }
    #runs-left { width: 3fr; }
    #runs-detail { width: 2fr; }
    #jobs-left { height: auto; max-height: 9; }
    #jobs-table { height: auto; max-height: 9; }
    #jobs-detail { height: 1fr; border-left: none; border-top: tall $panel; padding: 1 0 0 0; margin-top: 1; }
    #runs-root { margin-bottom: 1; }
    #job-head { height: auto; }
    #job-stages { height: auto; margin-top: 1; }
    #job-log-box { height: 1fr; }
    Compare Select { width: 100%; max-width: 110; }
    ModalScreen { align: center middle; background: $background 70%; }
    .dialog { width: 80; max-width: 95%; height: auto; max-height: 90%; border: round $primary;
              background: $surface; padding: 1 2; }
    .dialog.wide { width: 110; }
    .dialog-title { text-style: bold; color: $primary; margin-bottom: 1; }
    .dialog.form { height: 90%; }
    #fields { height: 1fr; margin-top: 1; }
    .field { height: auto; margin-bottom: 1; }
    .field Input, .field SelectCurrent { background: $panel; }
    .field Input:focus, .field Select:focus SelectCurrent { background: $panel-lighten-1; }
    #preview-box { height: auto; max-height: 22; border: round $panel; padding: 0 1; margin-top: 1; }
    """
    BINDINGS = [
        Binding("n", "new_job", t("tui.new_job")),
        Binding("q", "quit_app", t("tui.quit")),
        *[Binding(str(i + 1), f"tab('{name}')", t(f"tui.tab.{name}"), show=False) for i, name in enumerate(TABS)],
    ]

    def __init__(
        self,
        runs_root: Path | None = None,
        manager: jobsvc.JobManager | None = None,
        *,
        check_updates: bool = True,
        animate: bool = True,
    ) -> None:
        super().__init__()
        self.runs_root = runs_root or Path("runs")
        self.manager = manager or jobsvc.JobManager()
        self.check_updates = check_updates and not os.environ.get("OPENTERNARY_NO_UPDATE_CHECK")
        self.motion = animate
        self.summary: dict[str, Any] = {}
        self.run_list: list[runsvc.RunInfo] = []

    def compose(self) -> ComposeResult:
        with TabbedContent(initial="home", id="tabs"):
            with TabPane(t("tui.tab.home"), id="home"):
                yield Home()
            with TabPane(t("tui.tab.runs"), id="runs"):
                yield Runs()
            with TabPane(t("tui.tab.jobs"), id="jobs"):
                yield Jobs()
            with TabPane(t("tui.tab.compare"), id="compare"):
                yield Compare()
            with TabPane(t("tui.tab.doctor"), id="doctor"):
                yield Doctor()
            with TabPane(t("tui.tab.update"), id="update"):
                yield Update()
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(YORU)
        self.theme = "yoru"
        self.sub_title = __version__
        self.refresh_runs()
        self.refresh_jobs()
        self.refresh_doctor()
        self.refresh_update(force=False)
        self.set_interval(0.5, self.refresh_jobs)
        self.run_worker(self.manager.prune, thread=True, group="prune")  # keep the newest 200 records

    # navigation
    def action_tab(self, name: str) -> None:
        self.query_one("#tabs", TabbedContent).active = name

    @on(TabbedContent.TabActivated)
    def _tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Put the keyboard on the tab's main list so its keys (delete, cancel, …) work at once."""
        target = {"runs": "#runs-table", "jobs": "#jobs-table"}.get(event.pane.id or "")
        if target:
            self.query_one(target).focus()

    @on(Button.Pressed, "#home-new")
    def action_new_job(self) -> None:
        self.new_job()

    @on(Button.Pressed, "#home-runs")
    def _home_runs(self) -> None:
        self.action_tab("runs")

    @on(Button.Pressed, "#home-doctor")
    def _home_doctor(self) -> None:
        self.action_tab("doctor")

    # home summary
    def _home(self) -> None:
        self.query_one(Home).show(self.summary)

    # runs
    @work(thread=True, exclusive=True, group="runs")
    def refresh_runs(self) -> None:
        runs = runsvc.list_runs(self.runs_root)
        self.call_from_thread(self._show_runs, runs)

    def _show_runs(self, runs: list[runsvc.RunInfo]) -> None:
        self.run_list = runs
        self.query_one(Runs).show(self.runs_root, runs)
        self.query_one(Compare).show(runs)
        self.summary["runs"] = t("tui.runs_count", count=len(runs), path=str(self.runs_root))
        if runs:
            last = runs[0]
            self.summary["last_run"] = Text.assemble(badge(last.status), f"  {last.name}")
        self._home()

    def delete_run(self, run: runsvc.RunInfo) -> None:
        def done(confirmed: bool | None) -> None:
            if not confirmed:
                return
            try:
                runsvc.delete_run(self.runs_root, run.path)
            except (ValueError, OSError) as exc:
                self.notify(str(exc), severity="error", timeout=8)
                return
            self.notify(t("tui.deleted", name=run.name))
            self.refresh_runs()

        self.push_screen(TypedConfirm(t("tui.delete_run"), run.name, str(run.path)), done)

    def pick_for_compare(self, which: str, run: runsvc.RunInfo) -> None:
        self.query_one(Compare).pick(which, run)
        self.notify(t(f"tui.picked_{which}", name=run.name))

    # jobs
    def refresh_jobs(self) -> None:
        jobs = self.manager.jobs()
        self.query_one(Jobs).show(jobs)
        running = [job for job in jobs if job.status == "running"]
        self.summary["jobs"] = t("tui.jobs_running", count=len(running)) if running else t("tui.jobs_idle")
        finished_now = [j for j in jobs if j.id in getattr(self, "_watching", set()) and j.status != "running"]
        for job in finished_now:
            severity: Literal["information", "error"] = "information" if job.status == "completed" else "error"
            self.notify(
                t("tui.job_finished", command=job.command, status=t(f"tui.status.{job.status}")), severity=severity
            )
            self.refresh_runs()
        self._watching = {job.id for job in running}
        self._home()

    def new_job(self, command: str | None = None, values: dict[str, Any] | None = None) -> None:
        def planned(choice: tuple[catalog.Spec, list[str], str | None] | None) -> None:
            if choice is None:
                return
            spec, args, typed = choice

            def confirmed(start: bool | None) -> None:
                if start:
                    self.start_job([*args, "--yes"] if spec.confirm else args)
                else:  # back to the form with what was typed
                    self.new_job(spec.name)

            self.push_screen(PlanScreen(spec, args, typed), confirmed)

        self.push_screen(JobForm(command, values), planned)

    def start_job(self, args: list[str]) -> jobsvc.Job:
        job = self.manager.start(args)
        jobs_pane = self.query_one(Jobs)
        jobs_pane.focus_job = job.id
        self.refresh_jobs()
        self.action_tab("jobs")
        self.notify(t("tui.job_started", command=job.command))
        return job

    def cancel_job(self, job: jobsvc.Job) -> None:
        if job.status != "running":
            return
        if job.cancel_requested:
            buttons = [("force", t("tui.force_stop"), "error"), ("back", t("tui.back"), "default")]
            message = t("tui.force_message")
        else:
            buttons = [("cancel", t("tui.cancel_job"), "warning"), ("back", t("tui.back"), "default")]
            message = t("tui.cancel_message")

        def chosen(answer: str | None) -> None:
            if answer in ("cancel", "force"):
                self.manager.cancel(job, force=answer == "force")
                self.refresh_jobs()

        self.push_screen(Choice(t("tui.cancel_job"), message, buttons), chosen)

    # doctor / update
    @work(thread=True, exclusive=True, group="doctor")
    def refresh_doctor(self) -> None:
        from openternary.services import torch_setup as ts
        from openternary.services.planning import doctor
        from openternary.ui import render

        result = doctor()
        profile = ts.read_profile() or {}
        torch = ts.installed_torch()
        if torch is None:
            line: RenderableType = Text(t("tui.torch_missing"), style=brand.WARN)
        else:  # same rule as `doctor`: the recorded check counts only for the build that is installed
            status = (profile.get("verification") or {}).get("status") if profile.get("version") == torch else None
            backend = f" · {profile['backend']}" if profile.get("version") == torch and profile.get("backend") else ""
            line = Text.assemble(torch + backend + "  ", badge(status or "not_run"))
        self.call_from_thread(self._show_doctor, render.doctor(result), line)

    def _show_doctor(self, renderable: RenderableType, torch: RenderableType) -> None:
        self.query_one(Doctor).show(renderable)
        self.summary["torch"] = torch
        self._home()

    @work(thread=True, exclusive=True, group="update")
    def refresh_update(self, force: bool = False) -> None:
        from openternary.services import update as upd

        if self.check_updates or force:
            status = upd.check(force=force, fetch=lambda: upd.fetch_pypi_json(timeout=3.0))
        else:
            status = upd.cached_notice() or upd.Status(upd.current_version(), None, False, "", source="cache")
        mode = upd.install_mode()
        plan = upd.plan_upgrade(mode, has_profile=False, uv=upd.find_uv())
        self.call_from_thread(self._show_update, status, plan)

    def _show_update(self, status: Any, plan: Any) -> None:
        if status.error:
            state = Text(t("update.check_failed"), style=brand.WARN)
        elif status.published is False:
            state = Text(t("update.not_published"), style=brand.MUTED)
        elif status.update_available:
            state = Text(t("update.available", version=status.latest), style=f"bold {brand.WARN}")
        elif status.latest:
            state = Text.assemble(("✔ ", brand.OK), t("update.up_to_date"))
        else:
            state = Text(t("tui.not_checked"), style=brand.MUTED)
        rows: list[tuple[str, RenderableType | str | None]] = [
            (t("update.current"), status.current),
            (t("update.latest"), status.latest),
            (t("update.mode"), plan.mode),
        ]
        parts: list[RenderableType] = [kv(rows), Text(""), state]
        if not plan.automatic and plan.reason:
            parts += [Text(""), Text(t(f"update.manual.{plan.reason}"), style=brand.MUTED)]
        if plan.commands:
            parts += [Text(""), Text(t("update.steps"), style="bold")]
            parts += [Text("  " + " ".join(cmd), style=brand.ACCENT) for cmd in plan.commands]
        self.query_one(Update).show(Group(*parts), bool(status.update_available and plan.automatic))
        self.summary["update"] = state
        self._home()

    # quit
    def action_quit_app(self) -> None:
        running = self.manager.running()
        if not running:
            self.exit()
            return

        def chosen(answer: str | None) -> None:
            if answer == "leave":
                self.exit()
            elif answer == "stop":
                for job in running:
                    self.manager.cancel(job)
                self.exit()

        buttons = [
            ("leave", t("tui.quit_leave"), "primary"),
            ("stop", t("tui.quit_stop"), "warning"),
            ("back", t("tui.back"), "default"),
        ]
        self.push_screen(Choice(t("tui.quit"), t("tui.quit_running", count=len(running)), buttons), chosen)
