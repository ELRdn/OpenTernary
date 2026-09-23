"""Bounded process RSS sampling and phase-specific Torch allocator peaks."""

from __future__ import annotations

import functools
import importlib.metadata
import platform
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, TypeVar, cast

import psutil

_current: ContextVar[ResourceMeter | None] = ContextVar("resource_meter", default=None)
F = TypeVar("F", bound=Callable[..., dict[str, Any]])


def runtime_environment(device: str) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "diffusers", "torchao", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "actual_device": device,
    }
    if device.startswith("cuda"):
        import torch

        result.update(device_name=torch.cuda.get_device_name(device), hip=torch.version.hip, cuda=torch.version.cuda)
    return result


class ResourceMeter:
    def __init__(self) -> None:
        self.results: dict[str, Any] = {
            "measurement": "process_rss_sampled_20ms; torch_allocator_peak",
            "load_ram_bytes": None,
            "load_vram_bytes": None,
            "inference_ram_bytes": None,
            "inference_vram_bytes": None,
        }
        self.phase: str | None = None
        self.device = "unknown"
        self.started = 0.0
        self.rss = 0
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def finish(self) -> None:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.phase is None:
            return
        vram = reserved = None
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize(self.device)
            vram = torch.cuda.max_memory_allocated(self.device)
            reserved = torch.cuda.max_memory_reserved(self.device)
        elif self.device == "cpu":
            vram = reserved = 0
        self.results.update(
            {
                f"{self.phase}_ram_bytes": max(self.rss, psutil.Process().memory_info().rss),
                f"{self.phase}_vram_bytes": vram,
                f"{self.phase}_vram_reserved_bytes": reserved,
                f"{self.phase}_ms": (time.perf_counter() - self.started) * 1000,
            }
        )
        self.phase = None

    def begin(self, phase: str, device: str) -> None:
        self.finish()
        self.device = device
        if device.startswith("cuda"):
            import torch

            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        self.phase, self.started = phase, time.perf_counter()
        self.rss = psutil.Process().memory_info().rss
        self.stop = threading.Event()

        def sample() -> None:
            while not self.stop.wait(0.02):
                self.rss = max(self.rss, psutil.Process().memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True, name="openternary-resource-meter")
        self.thread.start()


def phase(name: str, device: str) -> None:
    meter = _current.get()
    if meter is not None:
        meter.begin(name, device)


def measured(function: F) -> F:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        meter = ResourceMeter()
        token = _current.set(meter)
        try:
            report = function(*args, **kwargs)
            meter.finish()
            report["resources"] = meter.results
            report["measurement_environment"] = runtime_environment(meter.device)
            if report.get("family") == "diffusion":
                from openternary.experiment.metadata import write_json

                output = kwargs.get("output") if "output" in kwargs else args[3]
                if output is not None:
                    write_json(output / "evaluation.json", report)
            return report
        finally:
            try:
                meter.finish()
            finally:
                _current.reset(token)

    return cast(F, wrapped)
