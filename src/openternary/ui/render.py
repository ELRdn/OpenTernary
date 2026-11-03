"""Human renderers for command results. The --json envelope is produced elsewhere and untouched."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from openternary.ui import brand
from openternary.ui.i18n import t
from openternary.ui.theme import badge


def _table(*columns: str) -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    for column in columns:
        table.add_column(column)
    return table


def _section(title: str) -> Text:
    return Text(title, style=f"bold {brand.ACCENT}")


def _scalar(value: Any) -> Text:
    if value is None:
        return Text("—", style="dim")
    if isinstance(value, bool):
        return badge("passed" if value else "failed")
    if isinstance(value, list):
        return Text(", ".join(str(v) for v in value) if value else "—")
    return Text(str(value))


def _version_cell(row: dict[str, Any]) -> Text:
    if row.get("version"):
        return Text(str(row["version"]))
    required = row.get("required_version")
    return Text(t("requires", version=required), style="dim") if required else Text("—", style="dim")


def backends_table(backends: list[dict[str, Any]]) -> Table:
    """One column per backend, one row per attribute (long status words stay on one line)."""
    table = _table(t("col.key"), *(str(row.get("name")) for row in backends))
    rows: list[tuple[str, Any]] = [
        (t("col.installation"), lambda r: badge(r.get("installation"))),
        (t("col.quality"), lambda r: badge(r.get("quality"))),
        (t("col.convert"), lambda r: badge(r.get("operations", {}).get("convert"))),
        (t("col.model_reload"), lambda r: badge(r.get("operations", {}).get("model_reload"))),
        (t("col.native"), lambda r: badge(r.get("operations", {}).get("native_low_bit_execution"))),
        (t("col.formats"), lambda r: Text(", ".join(r.get("formats", [])))),
        (t("col.version"), _version_cell),
    ]
    for label, cell in rows:
        table.add_row(label, *(cell(row) for row in backends))
    return table


def doctor(result: dict[str, Any]) -> RenderableType:
    deps = _table(t("col.package"), t("col.version"))
    for name, version in result.get("dependencies", {}).items():
        deps.add_row(name, Text(str(version)) if version else badge("not_installed"))
    hardware = result.get("hardware")
    parts: list[RenderableType] = [
        Text.assemble(
            (f"Python {result.get('python')}", "bold"),
            ("  ·  ", "dim"),
            str(result.get("platform")),
        ),
        Text(""),
        _section(t("doctor.dependencies")),
        deps,
        _section(t("doctor.backends")),
        backends_table(result.get("backends", [])),
        Text.assemble(
            (f"{t('doctor.hardware')}: ", "bold"),
            t("doctor.probe_hint") if hardware == "not_probed" else str(hardware),
        ),
        Text.assemble((f"{t('doctor.model_validation')}: ", "bold"), badge(result.get("model_validation"))),
    ]
    probe = result.get("runtime_probe")
    if isinstance(probe, dict):
        parts += [Text(""), _section(t("doctor.runtime_probe")), key_values(probe)]
    return Panel(
        Group(*parts),
        title=Text(t("doctor.title"), style=f"bold {brand.ACCENT}"),
        title_align="left",
        border_style=brand.BORDER,
        expand=False,
    )


def plan(result: dict[str, Any]) -> RenderableType:
    config = result.get("config", {})
    model = config.get("model", {})
    quant = config.get("quantization", {})
    backend = result.get("backend", {})
    targets = result.get("targets") or []
    quantizable = sum(1 for item in targets if item.get("quantizable"))
    rows = _table(t("col.key"), t("col.value"))
    revision = model.get("revision")
    rows.add_row(t("plan.model"), f"{model.get('id')}" + (f" @ {str(revision)[:12]}" if revision else ""))
    rows.add_row(t("plan.backend"), f"{backend.get('name')} / {quant.get('scheme')} → {quant.get('weight_dtype')}")
    rows.add_row(t("plan.granularity"), f"{quant.get('scale_granularity')} / g{quant.get('group_size')}")
    rows.add_row(t("plan.adapter"), _scalar(result.get("adapter")))
    rows.add_row(
        t("plan.targets"),
        Text(f"{quantizable} / {len(targets)}") if targets else _scalar(result.get("target_count")),
    )
    rows.add_row(t("plan.device"), f"{result.get('actual_device')} / {result.get('actual_dtype')}")
    rows.add_row(t("plan.passes"), _scalar([p.get("name", p) for p in result.get("passes", [])] or None))
    parts: list[RenderableType] = [rows]
    unknown = result.get("unknown") or []
    if unknown:
        parts += [_section(t("plan.unknown"))] + [Text.assemble(("  ? ", brand.MUTED), str(u)) for u in unknown]
    return Panel(
        Group(*parts),
        title=Text(t("plan.title"), style=f"bold {brand.ACCENT}"),
        title_align="left",
        border_style=brand.BORDER,
        expand=False,
    )


def plugins(result: dict[str, Any]) -> RenderableType:
    items = result.get("plugins") or []
    if not items:
        return Text(t("plugins.none"), style="dim")
    table = _table(t("col.kind"), t("col.name"), t("col.version"), "status")
    for row in items:
        table.add_row(str(row.get("kind")), str(row.get("name")), _scalar(row.get("version")), badge(row.get("status")))
    return table


def key_values(result: dict[str, Any]) -> Table:
    table = _table(t("col.key"), t("col.value"))
    for key, value in result.items():
        if isinstance(value, dict):
            nested = ", ".join(f"{k}={v}" for k, v in value.items() if not isinstance(v, dict | list))
            table.add_row(key, Text(nested or "…", style="" if nested else "dim"))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            table.add_row(key, Text(f"{len(value)} items", style="dim"))
        elif isinstance(value, str) and value in {"installed", "not_installed", "not_run", "completed", "passed"}:
            table.add_row(key, badge(value))
        else:
            table.add_row(key, _scalar(value))
    return table


RENDERERS = {"doctor": doctor, "plan": plan, "plugins": plugins}


def render(console: Console, kind: str, result: dict[str, Any]) -> None:
    if kind == "backends":
        console.print(backends_table(result.get("backends", [])))
    elif kind == "backend":
        console.print(backends_table([result]))
    elif kind in RENDERERS:
        console.print(RENDERERS[kind](result))
    else:
        console.print(key_values(result))


def error_panel(code: int, message: str, context: str | None, exc_type: str | None) -> Panel:
    title = Text.assemble(
        ("✖ ", brand.ERR), (t(f"error.{code}") if f"error.{code}" in _known() else t("error.1"), "bold")
    )
    body: list[RenderableType] = []
    if context and context.strip() and context.strip() != message.strip():
        body.append(Text(context.rstrip(":"), style="bold"))
    body.append(Text(message))
    hint = _hint(code, message, exc_type)
    if hint:
        body += [Text(""), Text.assemble((f"{t('hint')}: ", f"bold {brand.INFO}"), hint)]
    return Panel(
        Group(*body),
        title=title,
        title_align="left",
        subtitle=Text(t("error.exit", code=code), style="dim"),
        subtitle_align="right",
        border_style=brand.ERR,
    )


def _known() -> set[str]:
    from openternary.ui.i18n import M

    return set(M)


def _hint(code: int, message: str, exc_type: str | None) -> str | None:
    if code == 4 or exc_type in {"ImportError", "ModuleNotFoundError"}:
        return t("hint.ml")
    if "hf download" in message:
        return t("hint.download")
    if code == 5:
        return t("hint.hardware")
    return None
