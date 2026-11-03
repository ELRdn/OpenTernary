"""Modal screens: the job form, the plan (confirmation) screen, typed-name deletion, quit."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from openternary.ui import brand
from openternary.ui.app import catalog
from openternary.ui.i18n import t


def _configs() -> list[str]:
    root = Path("configs")
    return sorted(str(p).replace("\\", "/") for p in root.glob("*.y*ml")) if root.is_dir() else []


class JobForm(ModalScreen[tuple[catalog.Spec, list[str], str | None] | None]):
    """Choose a command and fill in its options. Returns (spec, args, name to type) or None."""

    BINDINGS = [("escape", "dismiss_form", t("tui.back"))]

    def __init__(self, command: str | None = None, values: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.specs = list(catalog.specs())
        self.command = command if command in {s.name for s in self.specs} else "inspect"
        self.values = dict(values or {})

    def compose(self) -> ComposeResult:
        options = [(s.name, s.name) for s in self.specs]
        with Vertical(id="dialog", classes="dialog wide form"):
            yield Label(t("tui.new_job"), classes="dialog-title")
            yield Select(options, value=self.command, allow_blank=False, compact=True, id="command")
            yield Static("", id="command-help", classes="muted")
            yield VerticalScroll(id="fields")
            yield Static("", id="form-error", classes="error")
            with Horizontal(classes="buttons"):
                yield Button(t("tui.back"), id="cancel")
                yield Button(t("tui.review_plan"), id="review", variant="primary")

    async def on_mount(self) -> None:
        await self._build(self.command)

    async def _build(self, name: str) -> None:
        spec = catalog.spec(name)
        self.query_one("#command-help", Static).update(spec.help)
        self.query_one("#form-error", Static).update("")
        fields = self.query_one("#fields", VerticalScroll)
        await fields.remove_children()
        rows = []
        for item in spec.fields:
            label = item.flag or f"<{item.name}>"
            if item.required:
                label += " *"
            preset = self.values.get(item.name, item.default)
            parts: list[Any] = [Label(Text.assemble((label, "bold"), ("  " + item.help, brand.MUTED)))]
            if item.kind == "flag":
                parts = [Checkbox(f"{label}  {item.help}", value=bool(preset), compact=True, id=f"f-{item.name}")]
            elif item.choices:
                choices = [(c, c) for c in item.choices]
                value = preset if preset in item.choices else Select.NULL
                parts.append(
                    Select(choices, value=value, allow_blank=not item.required, compact=True, id=f"f-{item.name}")
                )
            else:
                suggester = SuggestFromList(_configs()) if item.name == "config" else None
                text = "" if preset in (None, False) else str(preset)
                placeholder = t("tui.multiple_hint") if item.multiple else ""
                parts.append(
                    Input(text, placeholder=placeholder, suggester=suggester, compact=True, id=f"f-{item.name}")
                )
            rows.append(Vertical(*parts, classes="field"))
        await fields.mount(*rows)
        self.values = {}  # presets apply to the first command shown only

    @on(Select.Changed, "#command")
    async def _command_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, str) and event.value != self.command:
            self.command = event.value
            await self._build(event.value)

    def _collect(self) -> dict[str, Any]:
        spec = catalog.spec(self.command)
        values: dict[str, Any] = {}
        for item in spec.fields:
            widget = self.query_one(f"#f-{item.name}")
            if isinstance(widget, Checkbox):
                values[item.name] = widget.value
            elif isinstance(widget, Select):
                values[item.name] = None if widget.value is Select.NULL else widget.value
            elif isinstance(widget, Input):
                values[item.name] = widget.value
        return values

    @on(Button.Pressed, "#review")
    def _review(self) -> None:
        spec = catalog.spec(self.command)
        values = self._collect()
        try:
            args = catalog.build_args(spec, values)
        except ValueError as exc:
            self.query_one("#form-error", Static).update(str(exc))
            return
        self.dismiss((spec, args, catalog.destructive_target(spec, values)))

    @on(Button.Pressed, "#cancel")
    def action_dismiss_form(self) -> None:
        self.dismiss(None)


class PlanScreen(ModalScreen[bool]):
    """Exactly what will run, the command's own --dry-run output, then Start. Deleting jobs also
    require typing the target's name."""

    BINDINGS = [("escape", "back", t("tui.back"))]

    def __init__(self, spec: catalog.Spec, args: list[str], typed: str | None = None) -> None:
        super().__init__()
        self.spec, self.args, self.typed = spec, args, typed
        self.preview_ok = not spec.dry_run

    def compose(self) -> ComposeResult:
        line = "openternary " + shlex.join(self.args) + (" --yes" if self.spec.confirm else "")
        with Vertical(id="dialog", classes="dialog wide"):
            yield Label(t("tui.plan_title"), classes="dialog-title")
            yield Static(Text(line, style=f"bold {brand.ACCENT}"), id="plan-command")
            note = t("tui.preview_running") if self.spec.dry_run else t("tui.no_preview")
            with VerticalScroll(id="preview-box"):
                yield Static(note, id="preview", classes="muted")
            if self.typed:
                yield Label(t("tui.type_to_delete", name=self.typed), classes="warn")
                yield Input(placeholder=self.typed, id="typed")
            with Horizontal(classes="buttons"):
                yield Button(t("tui.back"), id="back")
                yield Button(t("tui.start"), id="start", variant="error" if self.typed else "primary")

    def on_mount(self) -> None:
        self._sync()
        if self.spec.dry_run:
            self._preview()

    @work(thread=True, exclusive=True)
    def _preview(self) -> None:
        from openternary.services.jobs import preview

        code, text = preview(self.args)
        self.app.call_from_thread(self._show_preview, code, text)

    def _show_preview(self, code: int, text: str) -> None:
        self.preview_ok = code == 0
        body = Text(text or t("tui.preview_empty"))
        if code != 0:
            body = Text.assemble((t("tui.preview_failed", code=code) + "\n\n", f"bold {brand.ERR}"), body)
        self.query_one("#preview", Static).update(body)
        self._sync()

    def _sync(self) -> None:
        typed_ok = True
        if self.typed:
            typed_ok = self.query_one("#typed", Input).value.strip() == self.typed
        self.query_one("#start", Button).disabled = not (self.preview_ok and typed_ok)

    @on(Input.Changed, "#typed")
    def _typed(self) -> None:
        self._sync()

    @on(Button.Pressed, "#start")
    def _start(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#back")
    def action_back(self) -> None:
        self.dismiss(False)


class TypedConfirm(ModalScreen[bool]):
    """Generic destructive confirmation: the action runs only after the exact name is typed."""

    BINDINGS = [("escape", "back", t("tui.back"))]

    def __init__(self, title: str, name: str, detail: str = "") -> None:
        super().__init__()
        self.title_text, self.name_text, self.detail = title, name, detail

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog", classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            if self.detail:
                yield Static(self.detail, classes="muted")
            yield Label(t("tui.type_to_delete", name=self.name_text), classes="warn")
            yield Input(placeholder=self.name_text, id="typed")
            with Horizontal(classes="buttons"):
                yield Button(t("tui.back"), id="back")
                yield Button(t("tui.delete"), id="delete", variant="error", disabled=True)

    @on(Input.Changed, "#typed")
    def _typed(self, event: Input.Changed) -> None:
        self.query_one("#delete", Button).disabled = event.value.strip() != self.name_text

    @on(Input.Submitted, "#typed")
    def _submitted(self, event: Input.Submitted) -> None:
        if event.value.strip() == self.name_text:
            self.dismiss(True)

    @on(Button.Pressed, "#delete")
    def _delete(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#back")
    def action_back(self) -> None:
        self.dismiss(False)


class Choice(ModalScreen[str]):
    """A message with a few labelled buttons; returns the chosen id (``back`` on Escape)."""

    BINDINGS = [("escape", "back", t("tui.back"))]

    def __init__(self, title: str, message: str, buttons: list[tuple[str, str, str]]) -> None:
        super().__init__()
        self.title_text, self.message, self.buttons = title, message, buttons

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog", classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Static(self.message)
            with Horizontal(classes="buttons"):
                for ident, label, variant in self.buttons:
                    yield Button(label, id=ident, variant=variant)  # type: ignore[arg-type]

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id or "back")

    def action_back(self) -> None:
        self.dismiss("back")
