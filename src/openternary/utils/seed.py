"""Seed handling — PYTHONHASHSEED は子プロセス向けにのみ設定."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> dict[str, str]:
    """可能な範囲で乱数シードを固定.

    Returns: 設定内容の辞書（ログ用）
    """
    result: dict[str, str] = {}

    # stdlib
    random.seed(seed)
    result["random"] = str(seed)

    # PYTHONHASHSEED は現プロセスでは変更不可だが、子プロセスのために環境変数をセット
    os.environ["PYTHONHASHSEED"] = str(seed)
    result["PYTHONHASHSEED"] = str(seed) + " (subprocess only, not retroactive)"

    # numpy (optional)
    try:
        import numpy as np

        np.random.seed(seed)
        result["numpy"] = str(seed)
    except ImportError:
        result["numpy"] = "not installed, skipped"

    # torch (optional, Phase 0 では未インストール想定)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result["torch"] = str(seed)
    except ImportError:
        result["torch"] = "not installed, skipped"

    return result
