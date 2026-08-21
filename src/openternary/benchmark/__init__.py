"""Benchmark package — Phase 1b smoke."""

from openternary.benchmark.metrics import to_metrics
from openternary.benchmark.runner import run_benchmark
from openternary.benchmark.suites import (
    SMOKE_PROMPTS,
    SMOKE_SUITE_VERSION,
    protocol_fingerprint,
    result_fingerprint,
)

__all__ = [
    "SMOKE_PROMPTS",
    "SMOKE_SUITE_VERSION",
    "protocol_fingerprint",
    "result_fingerprint",
    "run_benchmark",
    "to_metrics",
]
