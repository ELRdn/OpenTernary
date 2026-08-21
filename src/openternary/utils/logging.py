"""Structured logging — Run ディレクトリを正本とする."""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import UTC, datetime

from rich.logging import RichHandler


def setup_logging(
    run_dir: pathlib.Path | None = None,
    level: int = logging.INFO,
    verbose: bool = False,
) -> logging.Logger:
    """ロギングを初期化.

    - コンソールは RichHandler
    - run_dir があれば logs/openternary.log と logs/events.jsonl にも出力
    """
    if verbose:
        level = logging.DEBUG

    # root logger をリセット（多重呼び出し対策）
    root = logging.getLogger()
    # 既存ハンドラをクリア（重複防止）
    for h in list(root.handlers):
        root.removeHandler(h)

    rich_handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    rich_handler.setLevel(level)
    root.addHandler(rich_handler)
    root.setLevel(level)

    if run_dir is not None:
        run_dir = pathlib.Path(run_dir)
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # ファイルハンドラ（通常ログ）
        file_handler = logging.FileHandler(log_dir / "openternary.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(file_handler)

        # events.jsonl は別途 capture_event で追記

    logger = logging.getLogger("openternary")
    return logger


def capture_event(run_dir: pathlib.Path, event: dict[str, object]) -> None:
    """events.jsonl に1イベント追記."""
    log_dir = pathlib.Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "events.jsonl"
    payload = {"timestamp": datetime.now(UTC).isoformat(), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
