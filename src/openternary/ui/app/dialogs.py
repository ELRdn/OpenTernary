"""Modal screens, all driven by ❯ menus (↑↓ + Enter) rather than buttons: choose a command or a
run, fill in a job form, review the plan, confirm a deletion by typing its name, quit."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Checkbox, Input, Label, OptionList, Select, Static

from openternary.ui import brand
from openternary.ui.app import catalog
from openternary.ui.app.widgets import Item, Menu
from openternary.ui.i18n import t


def _configs() -> list[str]:
    root = Path("configs")
    return sorted(str(p).replace("\\", "/") for p in root.glob("*.y*ml")) if root.is_dir() else []


class Picker(ModalScreen[str | None]):
    """Type to filter, ↑↓ to choose, Enter to take it. Returns the chosen item id or None."""

    BINDINGS = [
        Binding("escape", "back", t("tui.back")),
        Binding("up", "move(-1)", show=False),
        Binding("down", "move(1)", show=False),
    ]

    def __init__(self, title: str, items: list[Item], note: str = "") -> None:
        super().__init__()
        self.title_text, self.items, self.note = title, items, note

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog wide"):
            yield Label(self.title_text, classes="dialog-title")
            if self.note:
                yield Static(self.note, classes="muted")
            yield Input(placeholder=t("tui.filter"), id="filter", compact=True)
            yield Menu(self.items, id="choices")

    def on_mount(self) -> None:
        self.query_one("#filter", Input).focus()

    @on(Input.Changed, "#filter")
    def _filter(self, event: Input.Changed) -> None:
        query = event.value.strip().lower()
        shown = [i for i in self.items if query in i.name.lower() or query in i.description.lower()]
        starts = [i for i in shown if i.name.lower().lstrip("/").startswith(query)]
        self.query_one("#choices", Menu).set_items(starts + [i for i in shown if i not in starts])

    def action_move(self, delta: int) -> None:
        menu = self.query_one("#choices", Menu)
        if menu.option_count:
            menu.highlighted = ((menu.highlighted or 0) + delta) % menu.option_count

    @on(Input.Submitted, "#filter")
    def _take(self) -> None:
        item = self.query_one("#choices", Menu).current
        if item is not None and not item.disabled:
            self.dismiss(item.id)

    @on(OptionList.OptionSelected, "#choices")
    def _chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_back(self) -> None:
        self.dismiss(None)


def command_items() -> list[Item]:
    return [Item(spec.name, "/" + spec.slash, spec.help) for spec in catalog.specs()]


class JobForm(ModalScreen[tuple[catalog.Spec, list[str], str | None] | None]):
    """Fill in one command's options (generated from the CLI). Returns (spec, args, name to type)."""

    BINDINGS = [Binding("escape", "back", t("tui.back")), Binding("ctrl+s", "review", t("tui.review_plan"))]

    def __init__(self, command: str, values: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.spec = catalog.spec(command)
        self.values = dict(values or {})

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog wide form"):
            yield Label(
                Text.assemble(t("tui.new_job"), "  ", ("/" + self.spec.name, brand.ACCENT)), classes="dialog-title"
            )
            yield Static(self.spec.help, classes="muted")
            with VerticalScroll(id="fields"):
                for item in self.spec.fields:
                    yield self._field(item)
            yield Static("", id="form-error", classes="error")
            yield Menu(
                [Item("review", t("tui.review_plan"), "ctrl+s"), Item("back", t("tui.back"), "esc")],
                id="actions",
            )

    def _field(self, item: catalog.Field) -> Vertical:
        label = item.flag or f"<{item.name}>"
        if item.required:
            label += " *"
        preset = self.values.get(item.name, item.default)
        if item.kind == "flag":
            return Vertical(
                Checkbox(f"{label}  {item.help}", value=bool(preset), compact=True, id=f"f-{item.name}"),
                classes="field",
            )
        title = Label(Text.assemble((label, "bold"), ("  " + item.help, brand.MUTED)))
        if item.choices:
            value = preset if preset in item.choices else Select.NULL
            choices = [(c, c) for c in item.choices]
            widget: Any = Select(choices, value=value, allow_blank=not item.required, compact=True, id=f"f-{item.name}")
        else:
            suggester = SuggestFromList(_configs()) if item.name == "config" else None
            text = "" if preset in (None, False) else str(preset)
            placeholder = t("tui.multiple_hint") if item.multiple else ""
            widget = Input(text, placeholder=placeholder, suggester=suggester, compact=True, id=f"f-{item.name}")
        return Vertical(title, widget, classes="field")

    def on_mount(self) -> None:
        first = next(iter(self.query("#fields Input, #fields Checkbox, #fields Select")), None)
        (first or self.query_one("#actions")).focus()

    @on(Input.Submitted, "#fields Input")
    def _next_field(self) -> None:
        self.focus_next()  # Enter moves on, like Tab; the last field leads to the ❯ menu

    def _collect(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for item in self.spec.fields:
            widget = self.query_one(f"#f-{item.name}")
            if isinstance(widget, Checkbox):
                values[item.name] = widget.value
            elif isinstance(widget, Select):
                values[item.name] = None if widget.value is Select.NULL else widget.value
            elif isinstance(widget, Input):
                values[item.name] = widget.value
        return values

    @on(OptionList.OptionSelected, "#actions")
    def _action(self, event: OptionList.OptionSelected) -> None:
        if event.option.id == "review":
            self.action_review()
        else:
            self.action_back()

    def action_review(self) -> None:
        values = self._collect()
        try:
            args = catalog.build_args(self.spec, values)
        except ValueError as exc:
            self.query_one("#form-error", Static).update(str(exc))
            return
        self.dismiss((self.spec, args, catalog.destructive_target(self.spec, values)))

    def action_back(self) -> None:
        self.dismiss(None)


class PlanScreen(ModalScreen[bool]):
    """Exactly what will run and the command's own --dry-run output, then ❯ Start / Back.
    Start stays unavailable until the dry run passes (and, for deletions, the name is typed)."""

    BINDINGS = [Binding("escape", "back", t("tui.back"))]

    def __init__(self, spec: catalog.Spec, args: list[str], typed: str | None = None) -> None:
        super().__init__()
        self.spec, self.args, self.typed = spec, args, typed
        self.preview_ok = not spec.dry_run

    def compose(self) -> ComposeResult:
        line = "openternary " + shlex.join(self.args) + (" --yes" if self.spec.confirm else "")
        with Vertical(classes="dialog wide"):
            yield Label(t("tui.plan_title"), classes="dialog-title")
            yield Static(Text(line, style=f"bold {brand.ACCENT}"), id="plan-command")
            note = t("tui.preview_running") if self.spec.dry_run else t("tui.no_preview")
            with VerticalScroll(id="preview-box"):
                yield Static(note, id="preview", classes="muted")
            if self.typed:
                yield Label(t("tui.type_to_delete", name=self.typed), classes="warn")
                yield Input(placeholder=self.typed, id="typed", compact=True)
            # Created unavailable: it opens only once the checks pass, so nothing can slip through early.
            start = t("tui.delete") if self.typed else t("tui.start")
            yield Menu(
                [
                    Item("start", start, t("tui.start_hint"), disabled=True, tone=brand.ERR if self.typed else None),
                    Item("back", t("tui.back"), "esc"),
                ],
                id="actions",
            )

    def on_mount(self) -> None:
        (self.query_one("#typed") if self.typed else self.query_one("#actions")).focus()
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

    @property
    def ready(self) -> bool:
        typed_ok = not self.typed or self.query_one("#typed", Input).value.strip() == self.typed
        return self.preview_ok and typed_ok

    def _sync(self) -> None:
        menu = self.query_one("#actions", Menu)
        label = t("tui.delete") if self.typed else t("tui.start")
        hint = t("tui.start_ready") if self.ready else t("tui.start_hint")
        tone = brand.ERR if self.typed else None
        menu.update_item(Item("start", label, hint, disabled=not self.ready, tone=tone))
        if self.ready:
            menu.highlighted = 0

    @on(Input.Changed, "#typed")
    def _typed(self) -> None:
        self._sync()

    @on(Input.Submitted, "#typed")
    def _typed_done(self) -> None:
        self.query_one("#actions").focus()

    @on(OptionList.OptionSelected, "#actions")
    def _action(self, event: OptionList.OptionSelected) -> None:
        if event.option.id == "start" and self.ready:
            self.dismiss(True)
        elif event.option.id == "back":
            self.dismiss(False)

    def action_back(self) -> None:
        self.dismiss(False)


class TypedConfirm(ModalScreen[bool]):
    """Destructive confirmation: ❯ Delete becomes available only after the exact name is typed."""

    BINDINGS = [Binding("escape", "back", t("tui.back"))]

    def __init__(self, title: str, name: str, detail: str = "") -> None:
        super().__init__()
        self.title_text, self.name_text, self.detail = title, name, detail

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            if self.detail:
                yield Static(self.detail, classes="muted")
            yield Label(t("tui.type_to_delete", name=self.name_text), classes="warn")
            yield Input(placeholder=self.name_text, id="typed", compact=True)
            yield Menu(
                [
                    Item("delete", t("tui.delete"), self.name_text, disabled=True, tone=brand.ERR),
                    Item("back", t("tui.back"), "esc"),
                ],
                id="actions",
            )

    def on_mount(self) -> None:
        self.query_one("#typed", Input).focus()

    @property
    def matches(self) -> bool:
        return self.query_one("#typed", Input).value.strip() == self.name_text

    @on(Input.Changed, "#typed")
    def _typed(self) -> None:
        menu = self.query_one("#actions", Menu)
        menu.update_item(Item("delete", t("tui.delete"), self.name_text, disabled=not self.matches, tone=brand.ERR))
        if self.matches:
            menu.highlighted = 0

    @on(Input.Submitted, "#typed")
    def _submitted(self) -> None:
        if self.matches:
            self.dismiss(True)

    @on(OptionList.OptionSelected, "#actions")
    def _action(self, event: OptionList.OptionSelected) -> None:
        if event.option.id == "delete" and self.matches:
            self.dismiss(True)
        elif event.option.id == "back":
            self.dismiss(False)

    def action_back(self) -> None:
        self.dismiss(False)


class Choice(ModalScreen[str]):
    """A message and a ❯ menu of answers; returns the chosen id (``back`` on Escape)."""

    BINDINGS = [Binding("escape", "back", t("tui.back"))]

    def __init__(self, title: str, message: str, items: list[Item]) -> None:
        super().__init__()
        self.title_text, self.message, self.items = title, message, items

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Static(self.message)
            yield Menu(self.items, id="actions")

    def on_mount(self) -> None:
        self.query_one("#actions").focus()

    @on(OptionList.OptionSelected, "#actions")
    def _chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id or "back")

    def action_back(self) -> None:
        self.dismiss("back")
