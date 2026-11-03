"""The documentation site cannot drift from the code: every command, option, configuration key and
environment variable must be documented in both languages, and the two languages must have the same pages."""

from __future__ import annotations

import re
import typing
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "website" / "content"
LANGS = ("en", "ja")

if not CONTENT.is_dir():  # the sdist ships the code and tests, not the website
    pytest.skip("website/ is not part of this checkout", allow_module_level=True)

COMMON = {"json_output", "quiet", "events_jsonl"}  # documented once, under "Options on every command"


def page(lang: str, name: str) -> str:
    return (CONTENT / lang / name).read_text(encoding="utf-8")


def leaf_commands() -> dict[str, Any]:
    import typer.main

    from openternary.cli.main import app

    found: dict[str, Any] = {}

    def walk(command: Any, path: tuple[str, ...] = ()) -> None:
        if hasattr(command, "commands"):
            for name, child in command.commands.items():
                if not getattr(child, "hidden", False):
                    walk(child, (*path, name))
        else:
            found[" ".join(path)] = command

    walk(typer.main.get_command(app))
    return found


def cli_sections(text: str) -> dict[str, str]:
    parts = re.split(r"^### `(.+?)`\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def test_both_languages_have_the_same_pages() -> None:
    def pages(lang: str) -> set[str]:
        return {str(p.relative_to(CONTENT / lang)) for p in (CONTENT / lang).rglob("*.md")}

    assert pages("en") == pages("ja")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\s+- (?:[\w ]+: )?([\w/-]+\.md)\s*$", nav, flags=re.M))
    assert listed == pages("en"), "mkdocs.yml nav and the pages differ"


@pytest.mark.parametrize("lang", LANGS)
def test_cli_reference_documents_every_command_and_option(lang: str) -> None:
    commands = leaf_commands()
    sections = cli_sections(page(lang, "reference/cli.md"))
    assert set(sections) == set(commands), "documented commands differ from the CLI"
    for name, command in commands.items():
        body = sections[name]
        documented = set(re.findall(r"^\| `(--[\w-]+|-\w|\[?[A-Z_]+\]?)`", body, flags=re.M))
        documented |= set(re.findall(r" / `(--[\w-]+|-\w)`", body))
        for param in command.params:
            if param.name in COMMON:
                continue
            if "Argument" in type(param).__name__:
                assert {param.name.upper(), f"[{param.name.upper()}]"} & documented, f"{name}: {param.name}"
            else:
                for flag in [*param.opts, *param.secondary_opts]:
                    assert flag in documented, f"{name}: {flag} is not documented ({lang})"
        real = {flag for p in command.params for flag in [*p.opts, *p.secondary_opts]}
        real |= {p.name.upper() for p in command.params} | {f"[{p.name.upper()}]" for p in command.params}
        assert documented <= real, f"{name}: documents options that do not exist: {documented - real}"


def config_keys() -> set[str]:
    from pydantic import BaseModel

    from openternary.config.schema import AppConfig

    keys: set[str] = set()

    def walk(model: type[BaseModel], prefix: str = "") -> None:
        for name, field in model.model_fields.items():
            nested = [c for c in (field.annotation, *typing.get_args(field.annotation)) if isinstance(c, type)]
            sub = next((c for c in nested if issubclass(c, BaseModel)), None)
            if sub is not None and name != "passes":
                walk(sub, f"{prefix}{name}.")
            else:
                keys.add(prefix + name)

    walk(AppConfig)
    return keys


@pytest.mark.parametrize("lang", LANGS)
def test_configuration_reference_documents_every_key(lang: str) -> None:
    text = page(lang, "reference/configuration.md")
    documented = set(re.findall(r"^\| `([\w.]+)` \|", text, flags=re.M))
    keys = config_keys()
    assert keys <= documented, f"undocumented configuration keys ({lang}): {sorted(keys - documented)}"
    assert documented <= keys, f"documented keys that do not exist ({lang}): {sorted(documented - keys)}"


@pytest.mark.parametrize("lang", LANGS)
def test_every_environment_variable_is_documented(lang: str) -> None:
    source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))
    names = set(re.findall(r"\"(OPENTERNARY_[A-Z_]+|OT_[A-Z_]+)\"", source))
    names |= {f"OPENTERNARY_INDEX_{p}" for p in ("PYTORCH", "AMD", "PYPI")}  # built with an f-string
    names.discard("OPENTERNARY_INDEX_")
    text = page(lang, "reference/environment-variables.md")
    missing = {n for n in names if f"`{n}`" not in text}
    assert not missing, f"undocumented environment variables ({lang}): {sorted(missing)}"


@pytest.mark.parametrize("lang", LANGS)
def test_exit_codes_match_the_contract(lang: str) -> None:
    text = page(lang, "reference/output.md")
    documented = set(re.findall(r"^\| `(\d+)` \|", text, flags=re.M))
    assert documented == {"0", "1", "2", "3", "4", "5", "6", "7", "130"}


def test_japanese_soft_line_breaks_are_joined_outside_code() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("cjk_linebreaks", ROOT / "website" / "hooks" / "cjk_linebreaks.py")
    assert spec is not None and spec.loader is not None
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    source = "ジョブの\nリアルタイム。\n\n```text\nこれは\nコード\n```\n- 項目の\n  続き\nEnglish\nwords\n"
    expected = "ジョブのリアルタイム。\n\n```text\nこれは\nコード\n```\n- 項目の続き\nEnglish\nwords\n"
    assert hook.join_cjk_lines(source) == expected
