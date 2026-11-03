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
    "help.setup": {
        "en": "Detect the GPU and install a matching PyTorch build (CUDA / ROCm / MPS / CPU) into this environment.",
        "ja": "GPUを検出し、合うPyTorch（CUDA / ROCm / MPS / CPU）をこの環境に入れる。",
    },
    "help.update": {
        "en": "Check PyPI for a newer OpenTernary, upgrade, and restore the PyTorch build chosen by setup.",
        "ja": "PyPIで新しい版を確認して更新し、setupで選んだPyTorchの構成を入れ直す。",
    },
    "update.title": {"en": "OpenTernary updates", "ja": "OpenTernaryの更新"},
    "update.current": {"en": "Installed", "ja": "インストール済み"},
    "update.latest": {"en": "Latest on PyPI", "ja": "PyPIの最新"},
    "update.up_to_date": {"en": "Up to date.", "ja": "最新です。"},
    "update.not_published": {
        "en": "OpenTernary has no release on PyPI yet; nothing to update.",
        "ja": "OpenTernaryはまだPyPIに公開されていません。更新するものはありません。",
    },
    "update.available": {"en": "A newer version is available: {version}", "ja": "新しい版があります: {version}"},
    "update.mode": {"en": "Installed as", "ja": "インストール方法"},
    "update.steps": {"en": "Steps", "ja": "手順"},
    "update.restore_step": {
        "en": "then restore the PyTorch build recorded by openternary setup",
        "ja": "続けて openternary setup で記録したPyTorchの構成を入れ直す",
    },
    "update.manual.editable": {
        "en": "This is a source checkout: update it with git (nothing is changed automatically).",
        "ja": "ソースからの導入です。git で更新してください（自動では何も変更しません）。",
    },
    "update.manual.windows_running_tool": {
        "en": "Windows cannot replace a running openternary; close it and run these in a new terminal:",
        "ja": "Windowsでは実行中の openternary を置き換えられません。終了してから新しいターミナルで次を実行してください:",
    },
    "update.manual.no_uv": {"en": "uv is not on PATH.", "ja": "uv が見つかりません。"},
    "update.confirm": {"en": "Upgrade to {version} now?", "ja": "{version} に更新しますか？"},
    "update.check_failed": {
        "en": "Could not reach PyPI to check for updates",
        "ja": "更新の確認のためにPyPIへ接続できませんでした",
    },
    "update.no_uv": {"en": "Upgrading needs uv on PATH", "ja": "更新には uv が必要です"},
    "update.failed": {"en": "The upgrade command failed", "ja": "更新コマンドが失敗しました"},
    "update.restore_failed": {
        "en": "Upgraded, but restoring the PyTorch build failed; retry with: {command}",
        "ja": "更新しましたが、PyTorchの構成の入れ直しに失敗しました。再実行: {command}",
    },
    "update.done": {"en": "Updated to {version}.", "ja": "{version} に更新しました。"},
    "update.notice": {
        "en": "OpenTernary {version} is available — openternary update",
        "ja": "OpenTernary {version} が出ています — openternary update",
    },
    "setup.hardware": {"en": "Detected hardware", "ja": "検出したハードウェア"},
    "setup.gpus_none": {"en": "no supported GPU found", "ja": "対応GPUは見つかりませんでした"},
    "setup.backend": {"en": "Backend", "ja": "方式"},
    "setup.candidates": {"en": "Available PyTorch builds for this PC", "ja": "このPCで使えるPyTorch"},
    "setup.legend": {
        "en": "★ recommended   ✔ verified on this PC   ● installed now",
        "ja": "★ おすすめ   ✔ このPCで動作確認済み   ● 現在インストール中",
    },
    "setup.more": {
        "en": "… {count} more (openternary setup --list --json)",
        "ja": "… ほか{count}件（openternary setup --list --json）",
    },
    "setup.none_listed": {
        "en": "No compatible build found for this backend.",
        "ja": "この方式で使えるビルドがありません。",
    },
    "setup.plan": {"en": "Install plan", "ja": "インストール計画"},
    "setup.target": {"en": "Into", "ja": "入れる先"},
    "setup.source": {"en": "Source", "ja": "配布元"},
    "setup.command": {"en": "Command", "ja": "実行するコマンド"},
    "setup.confirm": {
        "en": "Install now? This downloads several GB.",
        "ja": "インストールしますか？ 数GBダウンロードします。",
    },
    "setup.needs_yes": {
        "en": "Installing needs confirmation; rerun with --yes (or use --dry-run to only see the plan).",
        "ja": "インストールには確認が必要です。--yes を付けて再実行してください（計画だけなら --dry-run）。",
    },
    "setup.install_failed": {"en": "PyTorch installation failed", "ja": "PyTorchのインストールに失敗しました"},
    "setup.verify_failed": {
        "en": "Installed, but the device check failed",
        "ja": "インストールはしましたが、デバイスでの動作確認に失敗しました",
    },
    "setup.verification": {"en": "Device check", "ja": "デバイスでの動作確認"},
    "setup.profile": {"en": "Recorded in", "ja": "記録先"},
    "setup.index_unreachable": {
        "en": "Could not read any package index",
        "ja": "配布元の一覧を取得できませんでした",
    },
    "setup.python_mismatch": {
        "en": "No build for Python {python}; builds exist for {needed}. Reinstall with e.g. uv tool install --editable . --python {needed} --reinstall",
        "ja": "Python {python} 用のビルドがありません（{needed} 用はあります）。例: uv tool install --editable . --python {needed} --reinstall",
    },
    "setup.no_build": {
        "en": "No {backend} build is published for this platform",
        "ja": "この環境向けの {backend} ビルドは配布されていません",
    },
    "setup.already": {
        "en": "Already set up: {version} is installed and passed the device check here. Nothing to do "
        "(choose another build with --torch / --variant, or reinstall it with --restore).",
        "ja": "設定済みです: {version} がインストール済みで、このPCでの動作確認に合格しています。何もしません"
        "（別のビルドにするなら --torch / --variant、入れ直すなら --restore）。",
    },
    "error.cancelled": {"en": "Cancelled by the user.", "ja": "ユーザーが中断しました。"},
    "setup.no_profile": {
        "en": "Nothing to restore: openternary setup has not recorded a build in this environment",
        "ja": "戻す構成がありません。この環境では openternary setup の記録がありません",
    },
    "setup.index_errors": {"en": "Unreachable sources", "ja": "取得できなかった配布元"},
    "warn.cuda_driver_too_old": {
        "en": "The NVIDIA driver is older than every listed CUDA build; update the driver.",
        "ja": "NVIDIAドライバが、どのCUDAビルドよりも古いです。ドライバを更新してください。",
    },
    "warn.cuda_driver_unknown": {
        "en": "The NVIDIA driver's CUDA version is unknown; the newest build was picked.",
        "ja": "NVIDIAドライバの対応CUDAが分からないため、最新のビルドを選びました。",
    },
    "warn.rocm_on_wsl": {
        "en": "ROCm under WSL needs AMD's WSL driver and is not verified here.",
        "ja": "WSL上のROCmにはAMDのWSL用ドライバが必要で、ここでは未検証です。",
    },
    "warn.torchvision_missing": {
        "en": "No matching torchvision is published for this torch yet; only torch is installed (Gemma image processing may fail).",
        "ja": "このtorchに合うtorchvisionはまだ公開されていないため、torchだけを入れます（Gemmaの画像処理が動かない可能性があります）。",
    },
    "setup.no_torchvision": {"en": "no torchvision yet", "ja": "torchvision未公開"},
    "note.wsl_gpu_not_visible": {
        "en": "WSL: no GPU is visible inside WSL.",
        "ja": "WSL: WSLの中からGPUが見えていません。",
    },
    "note.amd_arch_unknown": {
        "en": "AMD GPU model not recognised; set OPENTERNARY_GPU_ARCH=gfxNNNN if needed.",
        "ja": "AMD GPUの型番を判別できません。必要なら OPENTERNARY_GPU_ARCH=gfxNNNN を指定してください。",
    },
    "setup.amd_device": {
        "en": "AMD kernels: {device} (all = every GPU family, several GB; --amd-device all|gfxNNNN to override)",
        "ja": "AMDカーネル: {device}（all は全GPU分で数GB。--amd-device all|gfxNNNN で変更できます）",
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
    "doctor.torch": {"en": "PyTorch", "ja": "PyTorch"},
    "doctor.torch_no_profile": {
        "en": "not set up by openternary setup",
        "ja": "openternary setup では未設定",
    },
    "doctor.torch_drift": {
        "en": "differs from the build openternary setup installed ({recorded}); restore it with: openternary setup --restore",
        "ja": "openternary setup で入れた構成（{recorded}）と違います。戻すには: openternary setup --restore",
    },
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
    "help.ui": {
        "en": "Open the full-screen interface (also: openternary with no arguments at a terminal).",
        "ja": "全画面のインターフェースを開く（ターミナルで引数なしの openternary でも開く）。",
    },
    "tui.needs_terminal": {
        "en": "The interface needs an interactive terminal. Use the commands instead (openternary --help).",
        "ja": "インターフェースには対話できるターミナルが必要です。コマンドを使ってください（openternary --help）。",
    },
    "tui.tab.home": {"en": "Home", "ja": "ホーム"},
    "tui.tab.runs": {"en": "Runs", "ja": "実行結果"},
    "tui.tab.jobs": {"en": "Jobs", "ja": "ジョブ"},
    "tui.tab.compare": {"en": "Compare", "ja": "比較"},
    "tui.tab.doctor": {"en": "Doctor", "ja": "診断"},
    "tui.tab.update": {"en": "Update", "ja": "更新"},
    "tui.tagline": {"en": "ternary LLM toolkit", "ja": "三値LLMツールキット"},
    "tui.update": {"en": "Update", "ja": "更新"},
    "tui.runs_folder": {"en": "Runs", "ja": "実行結果"},
    "tui.last_run": {"en": "Last run", "ja": "直近の実行"},
    "tui.runs_count": {"en": "{count} in {path}", "ja": "{path} に {count} 件"},
    "tui.jobs_running": {"en": "{count} running", "ja": "{count} 件実行中"},
    "tui.jobs_idle": {"en": "none running", "ja": "実行中なし"},
    "tui.torch_missing": {"en": "not installed — Doctor › Set up PyTorch", "ja": "未導入 — 診断 › PyTorchを導入"},
    "tui.not_checked": {"en": "Not checked yet.", "ja": "まだ確認していません。"},
    "tui.new_job": {"en": "New job", "ja": "新しいジョブ"},
    "tui.quit": {"en": "Quit", "ja": "終了"},
    "tui.back": {"en": "Back", "ja": "戻る"},
    "tui.refresh": {"en": "Refresh", "ja": "再読込"},
    "tui.loading": {"en": "Loading…", "ja": "読み込み中…"},
    "tui.delete": {"en": "Delete", "ja": "削除"},
    "tui.review_plan": {"en": "Review plan", "ja": "計画を確認"},
    "tui.multiple_hint": {"en": "several values: separate with spaces", "ja": "複数指定はスペース区切り"},
    "tui.plan_title": {
        "en": "Plan — nothing runs until you press Start",
        "ja": "計画 — 開始を押すまで何も実行しません",
    },
    "tui.preview_running": {
        "en": "Asking the command for its plan (--dry-run)…",
        "ja": "コマンドに計画を問い合わせています（--dry-run）…",
    },
    "tui.no_preview": {
        "en": "This command has no --dry-run; check the command line above.",
        "ja": "このコマンドには --dry-run がありません。上のコマンドを確認してください。",
    },
    "tui.preview_empty": {"en": "(the dry run printed nothing)", "ja": "（dry-run の出力はありません）"},
    "tui.preview_failed": {
        "en": "The dry run failed (exit {code}); fix the options before starting.",
        "ja": "dry-run が失敗しました（終了コード {code}）。設定を直してから開始してください。",
    },
    "tui.type_to_delete": {"en": "Type {name} to confirm deletion", "ja": "削除するには {name} と入力してください"},
    "tui.start": {"en": "Start", "ja": "開始"},
    "tui.as_baseline": {"en": "Use as baseline", "ja": "基準に設定"},
    "tui.as_candidate": {"en": "Use as candidate", "ja": "候補に設定"},
    "tui.col.status": {"en": "Status", "ja": "状態"},
    "tui.col.quality": {"en": "Quality", "ja": "品質"},
    "tui.col.started": {"en": "Started", "ja": "開始"},
    "tui.col.command": {"en": "Command", "ja": "コマンド"},
    "tui.col.elapsed": {"en": "Elapsed", "ja": "経過"},
    "tui.col.stage": {"en": "Stage", "ja": "段階"},
    "tui.col.progress": {"en": "Progress", "ja": "進捗"},
    "tui.col.detail": {"en": "Detail", "ja": "詳細"},
    "tui.runs_in": {"en": "Runs in {path}", "ja": "{path} の実行結果"},
    "tui.no_runs": {
        "en": "No runs in {path} yet. Start one with New job (n).",
        "ja": "{path} にはまだ実行結果がありません。新しいジョブ（n）から始められます。",
    },
    "tui.select_run": {"en": "Select a run to see its details.", "ja": "実行結果を選ぶと詳細を表示します。"},
    "tui.path": {"en": "Path", "ja": "場所"},
    "tui.error": {"en": "Error", "ja": "エラー"},
    "tui.files": {"en": "Files", "ja": "ファイル"},
    "tui.delete_run": {"en": "Delete this run directory?", "ja": "この実行結果のフォルダを削除しますか？"},
    "tui.deleted": {"en": "Deleted {name}", "ja": "{name} を削除しました"},
    "tui.picked_baseline": {"en": "Baseline for Compare: {name}", "ja": "比較の基準: {name}"},
    "tui.picked_candidate": {"en": "Candidate for Compare: {name}", "ja": "比較の候補: {name}"},
    "tui.no_jobs": {
        "en": "No jobs yet. Start one with New job (n).",
        "ja": "ジョブはまだありません。新しいジョブ（n）から始められます。",
    },
    "tui.log": {"en": "Log (stderr)", "ja": "ログ（stderr）"},
    "tui.waiting_events": {"en": "Waiting for the first progress event…", "ja": "最初の進捗イベントを待っています…"},
    "tui.no_events": {
        "en": "This job reported no progress stages.",
        "ja": "このジョブは進捗段階を報告しませんでした。",
    },
    "tui.cancel_job": {"en": "Cancel job", "ja": "ジョブを中止"},
    "tui.cancel_message": {
        "en": "Send an interrupt? The command stops cleanly, exits 130 and records the failure in the run.",
        "ja": "中断を送りますか？ コマンドは安全に止まり、終了コード130で終わり、失敗を実行結果に記録します。",
    },
    "tui.force_stop": {"en": "Force stop", "ja": "強制停止"},
    "tui.force_message": {
        "en": "An interrupt was already sent. Force-stop the process and its children? Nothing will be recorded.",
        "ja": "中断は送信済みです。プロセスと子プロセスを強制停止しますか？ 記録は残りません。",
    },
    "tui.cancel_sent": {
        "en": "Interrupt sent; waiting for the command to stop.",
        "ja": "中断を送りました。コマンドの停止を待っています。",
    },
    "tui.job_started": {"en": "Started: {command}", "ja": "開始: {command}"},
    "tui.job_finished": {"en": "{command}: {status}", "ja": "{command}: {status}"},
    "tui.status.completed": {"en": "completed", "ja": "完了"},
    "tui.status.failed": {"en": "failed", "ja": "失敗"},
    "tui.status.interrupted": {"en": "interrupted", "ja": "中断"},
    "tui.status.lost": {"en": "ended (no record)", "ja": "終了（記録なし）"},
    "tui.status.running": {"en": "running", "ja": "実行中"},
    "tui.compare_intro": {
        "en": "Compare benchmark runs. The result is observational; only a passed quality gate means accepted.",
        "ja": "ベンチマーク結果を比較します。結果は観察であり、品質ゲートに合格した場合だけが受入です。",
    },
    "tui.baseline": {"en": "Baseline", "ja": "基準"},
    "tui.candidate": {"en": "Candidate", "ja": "候補"},
    "tui.probe_runtime": {"en": "Probe runtime…", "ja": "実機で確認…"},
    "tui.setup_torch": {"en": "Set up PyTorch…", "ja": "PyTorchを導入…"},
    "tui.check_now": {"en": "Check now", "ja": "今すぐ確認"},
    "tui.update_now": {"en": "Update…", "ja": "更新…"},
    "tui.quit_running": {
        "en": "{count} job(s) still running. They can keep running in the background; reopen OpenTernary to keep monitoring them.",
        "ja": "{count} 件のジョブが実行中です。バックグラウンドで続けられます。OpenTernaryを開き直せば監視を再開できます。",
    },
    "tui.quit_leave": {"en": "Quit, keep running", "ja": "終了（実行は続ける）"},
    "tui.quit_stop": {"en": "Stop jobs and quit", "ja": "ジョブを止めて終了"},
    "tui.commands": {"en": "commands", "ja": "コマンド"},
    "tui.prompt_placeholder": {
        "en": "Type / for commands  (↑↓ choose · Enter run · Tab complete · Esc close)",
        "ja": "/ でコマンド一覧（↑↓ 選択・Enter 実行・Tab 補完・Esc 閉じる）",
    },
    "tui.filter": {"en": "Type to filter", "ja": "入力して絞り込み"},
    "tui.pick_command": {
        "en": "Choose what to run. ↑↓ to move, Enter to choose.",
        "ja": "実行するコマンドを選んでください。↑↓で移動、Enterで決定。",
    },
    "tui.start_hint": {"en": "available once the plan checks pass", "ja": "計画の確認が通ると選べます"},
    "tui.tip.new": {"en": "choose a command and start a job", "ja": "コマンドを選んでジョブを始める"},
    "tui.tip.runs": {
        "en": "browse results (↑↓ at the prompt moves the list)",
        "ja": "実行結果を見る（入力欄で↑↓を押すと一覧が動く）",
    },
    "tui.tip.setup": {"en": "install PyTorch for this PC", "ja": "このPCに合うPyTorchを入れる"},
    "tui.tip.compare": {"en": "choose two runs from lists and compare them", "ja": "一覧から2つ選んで比較する"},
    "tui.tip.baseline": {
        "en": "in Runs: use the highlighted run as the baseline",
        "ja": "実行結果タブ: 選択中の結果を基準にする",
    },
    "tui.tip.candidate": {
        "en": "in Runs: use the highlighted run as the candidate",
        "ja": "実行結果タブ: 選択中の結果を候補にする",
    },
    "tui.cmd.new": {"en": "Choose a command and start a job", "ja": "コマンドを選んでジョブを始める"},
    "tui.cmd.compare": {"en": "Compare two runs (choose them from lists)", "ja": "2つの実行結果を比較（一覧から選ぶ）"},
    "tui.cmd.home": {"en": "Show Home", "ja": "ホームを表示"},
    "tui.cmd.runs": {"en": "Show run results", "ja": "実行結果を表示"},
    "tui.cmd.jobs": {"en": "Show jobs and live progress", "ja": "ジョブと進捗を表示"},
    "tui.cmd.doctor": {"en": "Show the environment check", "ja": "環境の診断を表示"},
    "tui.cmd.update": {"en": "Show the update status", "ja": "更新の状況を表示"},
    "tui.cmd.cancel": {"en": "Cancel the highlighted job", "ja": "選択中のジョブを中止"},
    "tui.cmd.delete": {
        "en": "Delete the highlighted run (type its name to confirm)",
        "ja": "選択中の実行結果を削除（名前の入力で確認）",
    },
    "tui.cmd.baseline": {
        "en": "Use the highlighted run as the comparison baseline",
        "ja": "選択中の実行結果を比較の基準にする",
    },
    "tui.cmd.candidate": {
        "en": "Use the highlighted run as the comparison candidate",
        "ja": "選択中の実行結果を比較の候補にする",
    },
    "tui.cmd.refresh": {"en": "Reload runs, jobs and the environment check", "ja": "実行結果・ジョブ・診断を再読込"},
    "tui.cmd.check": {"en": "Check PyPI for a newer version now", "ja": "今すぐPyPIで新しい版を確認"},
    "help.cache_list": {
        "en": "List the managed artifacts under a cache root.",
        "ja": "キャッシュの場所にある管理対象のartifactを一覧する。",
    },
    "help.cache_remove": {
        "en": "Remove one managed artifact after checking nothing refers to it (--execute to delete).",
        "ja": "参照されていないことを確かめてから管理対象のartifactを1つ削除する（--executeで実行）。",
    },
    "help.backends_list": {
        "en": "List backends with their installation and capability status.",
        "ja": "backendと、その導入・対応状況を一覧する。",
    },
    "help.backends_info": {
        "en": "Show one backend's installation and capability details.",
        "ja": "1つのbackendの導入・対応状況の詳細を表示する。",
    },
    "help.artifacts_validate": {
        "en": "Check an artifact's manifest and file integrity (not model quality).",
        "ja": "artifactのmanifestとファイルの整合性を検査する（モデルの品質ではない）。",
    },
    "tui.start_ready": {"en": "Enter to start", "ja": "Enterで開始"},
}


def t(key: str, **values: object) -> str:
    entry = M.get(key)
    if entry is None:
        return key
    text = entry.get(lang()) or entry["en"]
    return text.format(**values) if values else text
