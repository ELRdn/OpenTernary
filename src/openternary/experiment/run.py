"""Run directory management — 1 experiment = 1 directory = 1 reproducibility unit."""

from __future__ import annotations

import datetime
import pathlib
import re
import uuid

from filelock import FileLock

from openternary.config.loader import dump_config_yaml
from openternary.config.schema import AppConfig
from openternary.experiment.metadata import collect_environment, write_json


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
    base = pathlib.Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if config.output:
        # output が指定されていればそれを優先（絶対/相対両対応）
        run_dir = pathlib.Path(config.output)
        # 衝突時は連番サフィックス
        original = run_dir
        counter = 1
        while run_dir.exists():
            run_dir = pathlib.Path(f"{original}-{counter:03d}")
            counter += 1
    else:
        name = _run_dir_name(config)
        run_dir = base / name
        counter = 1
        original = run_dir
        while run_dir.exists():
            run_dir = pathlib.Path(f"{original}-{counter:03d}")
            counter += 1

    run_id = run_id or uuid.uuid4().hex[:8]

    # filelock で並行安全に作成
    lock_path = base / ".run.lock"
    with FileLock(str(lock_path)):
        run_dir.mkdir(parents=True, exist_ok=True)
        # 再チェック（ロック内で衝突が解決された場合）
        if run_dir.exists() and any(run_dir.iterdir()):
            # 既に別プロセスが作った場合は連番へ
            pass

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
