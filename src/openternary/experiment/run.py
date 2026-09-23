"""Run directory management — 1 experiment = 1 directory = 1 reproducibility unit."""

from __future__ import annotations

import contextlib
import datetime
import pathlib
import re
import uuid
from collections.abc import Iterator
from contextvars import ContextVar

from filelock import FileLock

from openternary.config.loader import dump_config_yaml
from openternary.config.schema import AppConfig
from openternary.experiment.metadata import collect_environment, write_json

_HELD: ContextVar[frozenset[str]] = ContextVar("held_runs", default=frozenset())


def _slug_from_config(config: AppConfig) -> str:
    """config から slug を生成."""
    # model id の最後の要素 + method + group_size
    model_name = config.model.id.split("/")[-1].replace(".", "-")
    slug = f"{model_name}-{config.quantization.method}-g{config.quantization.group_size}"
    # 安全な文字列に正規化
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", slug)
    return slug


def _run_dir_name(config: AppConfig) -> str:
    """runs/<YYYY-MM-DD>_<slug>/ の名前を生成."""
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    slug = _slug_from_config(config)
    return f"{date}_{slug}"


def create_run(
    config: AppConfig,
    base_dir: str | pathlib.Path = "runs",
    run_id: str | None = None,
) -> tuple[pathlib.Path, str]:
    """Run ディレクトリを作成し、初期成果物を書き出す.

    Returns: (run_dir, run_id)
    """
    base = pathlib.Path(base_dir).resolve()
    original = (pathlib.Path(config.output) if config.output else base / _run_dir_name(config)).resolve()
    original.parent.mkdir(parents=True, exist_ok=True)
    run_id = run_id or uuid.uuid4().hex[:8]
    # Same destination uses the same lock across cwd, aliases and explicit/base outputs.
    lock_path = original.parent / ".run.lock"
    with FileLock(str(lock_path)):
        run_dir = original
        counter = 1
        while True:
            try:
                run_dir.mkdir(exist_ok=False)
                break
            except FileExistsError:
                run_dir = original.with_name(f"{original.name}-{counter:03d}")
                counter += 1

        from openternary.services.reporting import track_run

        track_run(run_dir, run_id)
        # logs / artifacts ディレクトリ
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        # config.yaml 保存
        (run_dir / "config.yaml").write_text(dump_config_yaml(config), encoding="utf-8")

        # environment.json
        env = collect_environment(config, run_id)
        write_json(run_dir / "environment.json", env)

        # model.json
        write_json(
            run_dir / "model.json",
            {"id": config.model.id, "revision": config.model.revision},
        )

        # metrics.json 初期
        write_json(run_dir / "metrics.json", {"status": "not_run", "run_id": run_id})

    return run_dir, run_id


@contextlib.contextmanager
def run_lock(run_dir: str | pathlib.Path) -> Iterator[None]:
    """Exclusive, process-scoped ownership of execution/resume/finalization."""
    from filelock import Timeout

    root = pathlib.Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run directory not found: {root}")
    key = str(root)
    if key in _HELD.get():
        yield
        return
    try:
        with FileLock(str(root / ".execution.lock"), timeout=0):
            token = _HELD.set(_HELD.get() | {key})
            try:
                yield
            finally:
                _HELD.reset(token)
    except Timeout as exc:
        raise ValueError(f"run is already in use: {root}") from exc
