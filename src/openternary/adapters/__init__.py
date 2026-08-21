"""Adapters public API."""

from openternary.adapters.base import ModelAdapter, TensorInfo
from openternary.adapters.gemma4 import Gemma4Adapter, get_adapter

__all__ = ["ModelAdapter", "TensorInfo", "Gemma4Adapter", "get_adapter"]
