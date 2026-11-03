"""English / Japanese messages for human-facing output (never used for --json content).

Language: ``OPENTERNARY_LANG`` (``en`` / ``ja``), otherwise the OS locale. Japanese is only
chosen when the output stream can encode it, so legacy code pages (e.g. cp1252) fall back to
English instead of failing.
"""

from __future__ import annotations

import locale
import os
import sys
from functools import lru_cache
from typing import TextIO

LANGS = ("en", "ja")


def _encodable(stream: TextIO | None) -> bool:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "日本語".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _locale_is_japanese() -> bool:
    for name in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        value = os.environ.get(name, "")
        if value:
            return value.lower().startswith("ja")
    try:
        current = locale.getlocale()[0] or ""
    except ValueError:
        current = ""
    return current.lower().startswith(("ja", "japanese"))


def detect(stream: TextIO | None = None) -> str:
    requested = os.environ.get("OPENTERNARY_LANG", "").strip().lower()
    lang = requested if requested in LANGS else ("ja" if _locale_is_japanese() else "en")
    if lang == "ja" and not _encodable(stream if stream is not None else sys.stdout):
        return "en"
    return lang


@lru_cache(maxsize=1)
def lang() -> str:
    return detect()


M: dict[str, dict[str, str]] = {
    # --- app / help panels ---------------------------------------------------------------
    "app.help": {
        "en": "OpenTernary — ternary quantization toolkit. Add --json to any command for machine output.",
        "ja": "OpenTernary — 三値量子化ツールキット。各コマンドに --json を付けると機械向けの出力になります。",
    },
    "panel.workflow": {"en": "Workflow", "ja": "ワークフロー"},
    "panel.evaluate": {"en": "Evaluate & compare", "ja": "評価・比較"},
    "panel.environment": {"en": "Environment & artifacts", "ja": "環境・成果物"},
    # --- command short help (no internal phase names) ------------------------------------
    "help.inspect": {
        "en": "Inspect a model's architecture and quantizable modules (header-only by default).",
        "ja": "モデル構造と量子化対象を調べる（既定ではヘッダのみ読む）。",
    },
    "help.plan": {
        "en": "Check a quantization plan without downloads, tensor loading or writes.",
        "ja": "ダウンロード・テンソル読込・書込なしで量子化の計画を確認する。",
    },
    "help.quantize": {
        "en": "Quantize a model to a ternary fake-quant snapshot.",
        "ja": "モデルを三値fake-quantのsnapshotに量子化する。",
    },
    "help.calibrate": {
        "en": "Calibrate ternary scales/thresholds layer by layer (recon-scale).",
        "ja": "三値のscale・閾値を層ごとに校正する（recon-scale）。",
    },
    "help.export": {
        "en": "Export an artifact to another format; the source is never modified.",
        "ja": "成果物を別形式で書き出す（元は変更しない）。",
    },
    "help.benchmark": {
        "en": "Run the smoke inference benchmark.",
        "ja": "推論のsmokeベンチマークを実行する。",
    },
    "help.quality": {
        "en": "Evaluate frozen English/Japanese perplexity and instruction cases.",
        "ja": "固定した英日パープレキシティと指示ケースで品質を評価する。",
    },
    "help.evaluate": {
        "en": "Evaluate from a fixed profile (diffusion); image proxies never imply human acceptance.",
        "ja": "固定profileで評価する（Diffusion）。画像の代理指標は人の受入を意味しない。",
    },
    "help.compare": {
        "en": "Compare runs under the same conditions (observational unless acceptance is required).",
        "ja": "同条件のrunを比較する（受入を要求しない限り観測のみ）。",
    },
    "help.optimize": {
        "en": "Search explicit candidates within budgets; never selects on the test split.",
        "ja": "予算内で明示した候補を探索する（testでは選ばない）。",
    },
    "help.doctor": {
        "en": "Show installed packages and backends; never installs or runs a model.",
        "ja": "導入済みパッケージとbackendを表示する（インストールやモデル実行はしない）。",
    },
    "help.plugins": {
        "en": "List installed extensions without executing their code.",
        "ja": "導入済みの拡張を一覧する（コードは実行しない）。",
    },
    "help.cache_info": {"en": "Show cache configuration.", "ja": "キャッシュ設定を表示する。"},
    "help.backends": {"en": "Backend installation and capabilities.", "ja": "backendの導入状況と機能。"},
    "help.artifacts": {"en": "Artifact integrity and metadata.", "ja": "成果物の整合性とメタデータ。"},
    "help.cache": {
        "en": "Manage explicitly selected artifact roots.",
        "ja": "明示したartifact rootを管理する。",
    },
    # --- errors ----------------------------------------------------------------------------
    "error.1": {"en": "Unexpected failure", "ja": "想定外の失敗"},
    "error.2": {"en": "Invalid configuration, argument or input", "ja": "設定・引数・入力の誤り"},
    "error.3": {"en": "Unsupported capability", "ja": "非対応の機能"},
    "error.4": {"en": "Missing dependency", "ja": "必要な依存パッケージがない"},
    "error.5": {"en": "Required hardware is unavailable", "ja": "必要なハードウェアが使えない"},
    "error.6": {"en": "Acceptance criteria not met", "ja": "受入条件を満たしていない"},
    "error.7": {"en": "Search budget exhausted", "ja": "探索予算の上限に達した"},
    "error.130": {"en": "Interrupted", "ja": "中断された"},
    "error.exit": {"en": "exit {code}", "ja": "終了コード {code}"},
    "hint": {"en": "Hint", "ja": "ヒント"},
    "hint.ml": {
        "en": "Install the ML extras: uv sync --extra ml  (calibration also needs --extra calibration)",
        "ja": "ML用の追加依存を入れる: uv sync --extra ml（校正は --extra calibration も必要）",
    },
    "hint.download": {
        "en": "The model is not in the local Hugging Face cache; run the hf download command above.",
        "ja": "モデルがローカルのHugging Faceキャッシュにありません。上の hf download を実行してください。",
    },
    "hint.hardware": {
        "en": "Check the device with: openternary doctor --probe-runtime --device cuda",
        "ja": "デバイスを確認: openternary doctor --probe-runtime --device cuda",
    },
    # --- doctor / backends / plan ------------------------------------------------------------
    "doctor.title": {"en": "Environment", "ja": "環境"},
    "doctor.dependencies": {"en": "Dependencies", "ja": "依存パッケージ"},
    "doctor.backends": {"en": "Backends", "ja": "バックエンド"},
    "doctor.hardware": {"en": "Hardware", "ja": "ハードウェア"},
    "doctor.model_validation": {"en": "Model validation", "ja": "モデル検証"},
    "doctor.probe_hint": {
        "en": "not probed — openternary doctor --probe-runtime",
        "ja": "未確認 — openternary doctor --probe-runtime",
    },
    "doctor.runtime_probe": {"en": "Runtime probe", "ja": "実行時チェック"},
    "col.package": {"en": "Package", "ja": "パッケージ"},
    "col.version": {"en": "Version", "ja": "バージョン"},
    "col.backend": {"en": "Backend", "ja": "バックエンド"},
    "col.installation": {"en": "Installed", "ja": "導入"},
    "col.quality": {"en": "Quality", "ja": "品質"},
    "col.convert": {"en": "Convert", "ja": "変換"},
    "col.model_reload": {"en": "Model reload", "ja": "モデル再読込"},
    "col.native": {"en": "Native low-bit", "ja": "ネイティブ低bit"},
    "col.formats": {"en": "Formats", "ja": "形式"},
    "col.key": {"en": "Item", "ja": "項目"},
    "col.value": {"en": "Value", "ja": "値"},
    "col.name": {"en": "Name", "ja": "名前"},
    "col.kind": {"en": "Kind", "ja": "種類"},
    "plan.title": {"en": "Plan (no downloads, loads or writes)", "ja": "計画（ダウンロード・読込・書込なし）"},
    "plan.model": {"en": "Model", "ja": "モデル"},
    "plan.backend": {"en": "Backend / scheme", "ja": "backend / 方式"},
    "plan.granularity": {"en": "Scale / group", "ja": "scale / group"},
    "plan.adapter": {"en": "Adapter", "ja": "アダプタ"},
    "plan.targets": {"en": "Targets", "ja": "対象"},
    "plan.device": {"en": "Device / dtype", "ja": "デバイス / dtype"},
    "plan.passes": {"en": "Passes", "ja": "pass"},
    "plan.unknown": {"en": "Not determined offline", "ja": "オフラインでは未確定"},
    "plugins.none": {"en": "No extensions installed.", "ja": "導入済みの拡張はありません。"},
    "unknown": {"en": "unknown", "ja": "不明"},
    "requires": {"en": "requires {version}", "ja": "必要: {version}"},
    "none": {"en": "none", "ja": "なし"},
}


def t(key: str, **values: object) -> str:
    entry = M.get(key)
    if entry is None:
        return key
    text = entry.get(lang()) or entry["en"]
    return text.format(**values) if values else text
