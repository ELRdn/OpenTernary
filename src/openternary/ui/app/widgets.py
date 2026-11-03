"""Selection widgets in the style of a coding-agent CLI: a ``❯`` prompt with ``/`` commands whose
matches appear as a list above it, and plain ``❯``-marked menus instead of buttons."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rich.table import Table
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Input, Label, OptionList
from textual.widgets.option_list import Option

from openternary.ui import brand

MARK = "❯"


@dataclass(frozen=True)
class Item:
    """One selectable row: ``name`` in the first column, ``description`` muted after it."""

    id: str
    name: str
    description: str = ""
    disabled: bool = False
    tone: str | None = None  # colour of the name when highlighted (default: accent)


class Menu(OptionList):
    """A list where the highlighted row carries ``❯`` and the accent colour; Enter selects it."""

    DEFAULT_CSS = """
    Menu { border: none; background: transparent; padding: 0; height: auto; max-height: 14; }
    Menu:focus { border: none; background-tint: transparent; }
    Menu > .option-list--option-highlighted { background: transparent; }
    Menu:focus > .option-list--option-highlighted { background: transparent; }
    Menu > .option-list--option-hover { background: $boost; }
    Menu > .option-list--option-disabled { color: $text-disabled; }
    """

    def __init__(self, items: Sequence[Item] = (), *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self.items: list[Item] = []
        self.width = 0
        self.set_items(items)

    def set_items(self, items: Sequence[Item], width: int | None = None) -> None:
        self.items = []  # cleared first: clearing re-runs the highlight watcher
        self.clear_options()
        self.items = list(items)
        self.width = width or max((len(item.name) for item in self.items), default=0) + 3
        self.add_options(Option(self._prompt(item, False), id=item.id, disabled=item.disabled) for item in self.items)
        first = next((i for i, item in enumerate(self.items) if not item.disabled), None)
        self.highlighted = first

    def _prompt(self, item: Item, active: bool) -> Table:
        """Two columns, like a coding-agent CLI: long descriptions wrap inside their own column."""
        on = active and not item.disabled
        name = Text(no_wrap=True, overflow="ellipsis")
        name.append(f"{MARK} " if on else "  ", style=f"bold {item.tone or brand.ACCENT}")
        name.append(item.name, style=f"bold {item.tone or brand.ACCENT}" if on else ("dim" if item.disabled else ""))
        grid = Table.grid(expand=True)
        grid.add_column(width=self.width + 2, no_wrap=True)
        grid.add_column(ratio=1)
        style = brand.ACCENT if on else ("dim" if item.disabled else brand.MUTED)
        grid.add_row(name, Text(item.description, style=style))
        return grid

    def watch_highlighted(self, highlighted: int | None) -> None:
        super().watch_highlighted(highlighted)
        for index, item in enumerate(self.items[: self.option_count]):
            self.replace_option_prompt_at_index(index, self._prompt(item, index == highlighted))

    def update_item(self, item: Item) -> None:
        """Replace one row (same id): its text and whether it can be chosen."""
        self.items = [item if i.id == item.id else i for i in self.items]
        (self.disable_option if item.disabled else self.enable_option)(item.id)
        if item.disabled and self.current is not None and self.current.id == item.id:
            self.highlighted = next((n for n, i in enumerate(self.items) if not i.disabled), None)
        self.watch_highlighted(self.highlighted)

    @property
    def current(self) -> Item | None:
        if self.highlighted is None or not self.items:
            return None
        return self.items[self.highlighted]

    def select_id(self, item_id: str) -> None:
        """Highlight ``item_id`` and choose it, exactly as Enter would."""
        for index, item in enumerate(self.items):
            if item.id == item_id:
                self.highlighted = index
                self.action_select()
                return
        raise KeyError(item_id)


@dataclass
class Command:
    name: str  # without the slash, e.g. "calibrate" or "cache:remove"
    description: str
    run: Callable[[list[str]], object] = field(repr=False)  # receives the words typed after the name
    group: str = ""


def rank(commands: Sequence[Command], query: str) -> list[Command]:
    """Prefix matches first, then names containing the text, then descriptions containing it."""
    query = query.lower()
    if not query:
        return list(commands)
    starts = [c for c in commands if c.name.lower().startswith(query)]
    inside = [c for c in commands if query in c.name.lower() and c not in starts]
    described = [c for c in commands if query in c.description.lower() and c not in starts and c not in inside]
    return starts + inside + described


class CommandBar(Vertical):
    """``❯`` prompt. Typing ``/`` (or any text) lists matching commands above it: ↑↓ choose,
    Enter or Tab runs/completes, Esc closes. With the list closed, ↑↓ move the current tab's list."""

    BINDINGS = [
        Binding("up", "move(-1)", show=False),
        Binding("down", "move(1)", show=False),
        Binding("tab", "complete", show=False),
        Binding("escape", "close", show=False),
    ]
    DEFAULT_CSS = """
    CommandBar { height: auto; padding: 0 1; }
    CommandBar #suggest { display: none; border-top: solid $panel; padding-top: 0; }
    CommandBar.open #suggest { display: block; }
    CommandBar #prompt-row { border-top: solid $panel; height: 2; }
    CommandBar #prompt-mark { width: 2; color: $primary; text-style: bold; }
    CommandBar #prompt { border: none; height: 1; padding: 0; background: transparent; }
    CommandBar #prompt:focus { border: none; background-tint: transparent; }
    """

    class Navigate(Message):
        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    def __init__(self, commands: Sequence[Command], placeholder: str = "") -> None:
        super().__init__()
        self.commands = list(commands)
        self.placeholder = placeholder
        self.matches: list[Command] = []

    def compose(self) -> ComposeResult:
        yield Menu(id="suggest")
        with Horizontal(id="prompt-row"):
            yield Label(MARK, id="prompt-mark")
            yield Input(placeholder=self.placeholder, id="prompt", compact=True)

    @property
    def input(self) -> Input:
        return self.query_one("#prompt", Input)

    @property
    def menu(self) -> Menu:
        return self.query_one("#suggest", Menu)

    @property
    def is_open(self) -> bool:
        return self.has_class("open")

    def _words(self) -> tuple[str, list[str]]:
        text = self.input.value.strip()
        head, _, rest = text.partition(" ")
        return head.lstrip("/"), rest.split()

    @on(Input.Changed, "#prompt")
    def _changed(self, event: Input.Changed) -> None:
        head, rest = self._words()
        if not event.value.strip() or rest or event.value.endswith(" "):
            self.set_class(False, "open")  # arguments are being typed: the command is chosen
            return
        self.matches = rank(self.commands, head)
        width = max((len(c.name) for c in self.commands), default=0) + 4
        self.menu.set_items([Item(c.name, "/" + c.name, c.description) for c in self.matches], width=width)
        self.set_class(bool(self.matches), "open")

    def action_move(self, delta: int) -> None:
        if self.is_open:
            menu = self.menu
            if menu.option_count:
                menu.highlighted = ((menu.highlighted or 0) + delta) % menu.option_count
        else:
            self.post_message(self.Navigate(delta))

    def action_complete(self) -> None:
        if self.is_open and self.menu.current:
            self.input.value = "/" + self.menu.current.id + " "
            self.input.cursor_position = len(self.input.value)
        else:
            self.screen.focus_next()

    def action_close(self) -> None:
        if self.is_open or self.input.value:
            self.input.value = ""
            self.set_class(False, "open")
        else:
            self.screen.focus_next()

    @on(Input.Submitted, "#prompt")
    def _submitted(self) -> None:
        head, rest = self._words()
        command = None
        if self.is_open and self.menu.current:
            command = next(c for c in self.commands if c.name == self.menu.current.id)
        elif head:
            command = next((c for c in self.commands if c.name == head), None) or next(
                iter(rank(self.commands, head)), None
            )
        self.input.value = ""
        self.set_class(False, "open")
        if command is not None:
            command.run(rest)
        elif head:
            self.app.notify(f"/{head}?", severity="warning")

    @on(OptionList.OptionSelected, "#suggest")
    def _clicked(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.input.value = "/" + (event.option.id or "")
        self._submitted()
        self.input.focus()

    def on_key(self, event: events.Key) -> None:
        if event.key == "slash" and not self.input.has_focus:  # "/" anywhere opens the prompt
            self.input.focus()
