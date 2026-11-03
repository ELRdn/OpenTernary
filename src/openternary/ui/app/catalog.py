"""Job forms derived from the CLI definition itself, so the TUI can never drift from the commands.

Every leaf command of the Typer app becomes a ``Spec``; every parameter becomes a ``Field``. Options
that the TUI supplies itself (``--json``, ``--events-jsonl``, ``--quiet``, ``--dry-run``, ``--yes``)
are hidden: the plan screen previews with ``--dry-run`` and confirming the plan is the ``--yes``.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import PurePath
from typing import Any

HIDDEN = {"json_output", "quiet", "events_jsonl", "dry_run", "yes", "help"}
NOT_JOBS = {("ui",)}  # the TUI does not launch itself
# Commands that delete when this flag is set; the plan screen then asks for the target's name.
DESTRUCTIVE: dict[tuple[str, ...], tuple[str, str]] = {("cache", "remove"): ("execute", "target")}


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # "argument" | "flag" | "option"
    flag: str | None  # the option string, e.g. "--config"
    help: str = ""
    required: bool = False
    default: Any = None
    choices: tuple[str, ...] = ()
    multiple: bool = False
    path: bool = False


@dataclass(frozen=True)
class Spec:
    path: tuple[str, ...]
    help: str
    panel: str | None
    fields: tuple[Field, ...] = field(default_factory=tuple)
    dry_run: bool = False
    confirm: bool = False  # has --yes: the plan screen is the confirmation

    @property
    def name(self) -> str:
        return " ".join(self.path)


def _field(param: Any) -> Field | None:
    if param.name in HIDDEN:
        return None
    is_argument = "Argument" in type(param).__name__
    opts = list(getattr(param, "opts", []) or [])
    long_opts = [o for o in opts if o.startswith("--")] or opts
    choices = tuple(str(c) for c in (getattr(param.type, "choices", None) or ()))
    default = param.default if not callable(param.default) else None
    return Field(
        name=param.name,
        kind="argument" if is_argument else "flag" if getattr(param, "is_flag", False) else "option",
        flag=None if is_argument else (long_opts[0] if long_opts else None),
        help=(getattr(param, "help", None) or "").strip(),
        required=bool(param.required),
        default=None if default in ((), []) else default,
        choices=choices,
        multiple=bool(getattr(param, "multiple", False) or getattr(param, "nargs", 1) == -1),
        path="Path" in type(param.type).__name__,
    )


def _walk(command: Any, prefix: tuple[str, ...] = ()) -> list[Spec]:
    if hasattr(command, "commands"):
        specs: list[Spec] = []
        for name, child in command.commands.items():
            if not getattr(child, "hidden", False):
                specs.extend(_walk(child, (*prefix, name)))
        return specs
    if prefix in NOT_JOBS:
        return []
    names = {p.name for p in command.params}
    fields = tuple(f for f in (_field(p) for p in command.params) if f is not None)
    return [
        Spec(
            path=prefix,
            help=(command.help or command.short_help or "").strip().splitlines()[0] if command.help else "",
            panel=getattr(command, "rich_help_panel", None),
            fields=fields,
            dry_run="dry_run" in names,
            confirm="yes" in names,
        )
    ]


@lru_cache(maxsize=1)
def specs() -> tuple[Spec, ...]:
    import typer.main

    from openternary.cli.main import app

    return tuple(_walk(typer.main.get_command(app)))


def spec(name: str) -> Spec:
    for item in specs():
        if item.name == name:
            return item
    raise KeyError(name)


def build_args(spec: Spec, values: dict[str, Any]) -> list[str]:
    """CLI arguments for a filled-in form. Blank text and unset flags are left to the CLI defaults."""
    args = list(spec.path)
    positional: list[str] = []
    for item in spec.fields:
        value = values.get(item.name)
        if item.kind == "flag":
            if value:
                args.append(item.flag or f"--{item.name.replace('_', '-')}")
            continue
        text = "" if value is None else str(value).strip()
        if not text:
            if item.required:
                raise ValueError(f"{item.flag or item.name} is required")
            continue
        parts = shlex.split(text, posix=True) if item.multiple else [text]
        if item.kind == "argument":
            positional.extend(parts)
        else:
            for part in parts:
                args.extend([item.flag or f"--{item.name.replace('_', '-')}", part])
    return [*args, *positional]


def destructive_target(spec: Spec, values: dict[str, Any]) -> str | None:
    """The name the user must type before a deleting job starts (None when nothing is deleted)."""
    rule = DESTRUCTIVE.get(spec.path)
    if rule is None or not values.get(rule[0]):
        return None
    target = str(values.get(rule[1]) or "").strip()
    return PurePath(target).name or target
