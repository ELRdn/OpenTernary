"""Configuration loader: defaults → YAML → CLI overrides."""

from __future__ import annotations

import pathlib
from typing import Any

import yaml
from pydantic import ValidationError

from openternary.config.schema import AppConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """override を base に再帰マージ（override 優先）."""
    result: dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    """dotted パス（例 quantization.group_size）で値をセット."""
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value


def load_config(
    config_path: str | pathlib.Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """設定を読み込み、AppConfig を返す.

    優先度: defaults < YAML < CLI overrides
    """
    # 1. defaults（Pydantic デフォルト）
    merged: dict[str, Any] = AppConfig().model_dump()

    # 2. YAML ファイル
    if config_path is not None:
        p = pathlib.Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        if not isinstance(yaml_data, dict):
            raise ValueError(f"Config YAML must be a mapping, got {type(yaml_data)}")
        merged = _deep_merge(merged, yaml_data)

    # 3. CLI overrides
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is None:
                continue
            # CLI_TO_CONFIG の対応表を尊重、ただし dotted パスも直接受け付ける
            from openternary.config.schema import CLI_TO_CONFIG

            dotted = CLI_TO_CONFIG.get(k, k)
            _set_dotted(merged, dotted, v)

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as e:
        # rich で整形する前の素のエラーをラップ
        raise ValueError(f"Configuration validation failed:\n{e}") from e


def dump_config_yaml(config: AppConfig) -> str:
    """AppConfig を YAML 文字列にダンプ（config.yaml 保存用）."""
    data = config.model_dump()
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
