"""Adapter preserving the existing ternary numerical implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openternary.config.schema import AppConfig


class TernaryBackend:
    api_version = 1

    def convert(self, source: Path, destination: Path, config: AppConfig) -> Any:
        from openternary.quant.fake_quant import convert_snapshot

        return convert_snapshot(source, destination, config)
