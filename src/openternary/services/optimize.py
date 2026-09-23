"""Bounded candidate orchestration. Measurements come from workers, never estimates."""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from openternary.backends import get_spec
from openternary.config.schema import AppConfig, BaseModel, QuantizationConfig
from openternary.experiment.metadata import write_json
from openternary.experiment.run import run_lock
from openternary.services.artifacts import canonical_hash, file_hash, validate_artifact
from openternary.services.errors import BudgetError
from openternary.services.planning import build_plan, doctor
from openternary.services.reporting import current
from openternary.utils.git import code_provenance


class QualityProfile(BaseModel):
    schema_version: Literal[1] = 1
    name: Literal["balanced-llm-v1"] = "balanced-llm-v1"
    split: Literal["validation"] = "validation"
    dataset: str
    baseline: str
    max_length: int = Field(default=512, ge=2)
    stride: int = Field(default=256, ge=1)

    @model_validator(mode="after")
    def valid_stride(self) -> QualityProfile:
        if self.stride >= self.max_length:
            raise ValueError("stride must be smaller than max_length")
        return self


class SearchBudget(BaseModel):
    max_candidates: int = Field(default=4, ge=1, le=256)
    wall_seconds: float = Field(default=3600, gt=0)
    gpu_seconds: float = Field(default=3600, gt=0)
    artifact_bytes: int = Field(default=20 * 1024**3, gt=0)


class Measurement(BaseModel):
    schema_version: Literal[1] = 1
    artifact_reloaded: Literal[True]
    quality_accepted: bool
    latency_ms: float = Field(gt=0)
    inference_vram_bytes: int = Field(ge=0)
    gpu_seconds: float = Field(ge=0)
    artifact_fingerprint: str = Field(min_length=64, max_length=64)
    protocol_fingerprint: str = Field(min_length=64, max_length=64)


Executor = Callable[[AppConfig, QualityProfile, Path, float], dict[str, Any]]


def _worker(
    config: AppConfig, profile: QualityProfile, folder: Path, timeout: float, artifact_cap: int | None = None
) -> dict[str, Any]:
    request = folder / "request.json"
    write_json(request, {"config": config.model_dump(), "profile": profile.model_dump()})
    from openternary.services.process import run_process

    def capacity_check() -> None:
        if artifact_cap is not None:
            used = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            if used >= artifact_cap:
                raise BudgetError("candidate exceeded remaining artifact capacity")

    with (folder / "worker.log").open("w", encoding="utf-8") as log:
        run_process(
            [sys.executable, "-m", "openternary.services.search_worker", str(request)],
            timeout=timeout,
            stdout=log,
            poll_check=capacity_check,
        )
    return json.loads((folder / "measurement.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def search_contract(config: AppConfig, profile: QualityProfile, candidates: list[QuantizationConfig]) -> dict[str, Any]:
    source = Path(config.model.id).resolve()
    if not source.is_dir():
        raise ValueError("optimize requires an explicit local snapshot")
    # Sources, interface, dataset, baseline, seed, settings and runtime all invalidate cached measurements.
    source_files = {
        p.relative_to(source).as_posix(): file_hash(p)
        for p in source.rglob("*")
        if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"}
    }
    baseline = Path(profile.baseline) / "quality.json"
    from openternary.experiment.metadata import collect_environment

    hardware = collect_environment(config, "search-planning")["hardware"]
    return {
        "source_files": source_files,
        "config": config.model_dump(exclude={"output"}),
        "profile": profile.model_dump(),
        "dataset_hash": file_hash(Path(profile.dataset)),
        "baseline_hash": file_hash(baseline),
        "runtime": {**doctor(), "hardware_identity": hardware},
        "code": code_provenance(),
        "candidates": [c.model_dump() for c in candidates],
        "backends": [get_spec(c.backend).describe() for c in candidates],
    }


def run_search(
    config: AppConfig,
    profile: QualityProfile,
    candidates: list[QuantizationConfig],
    output: Path,
    budget: SearchBudget,
    *,
    target_vram: int,
    objective: str = "speed",
    resume: bool = False,
    executor: Executor | None = None,
) -> dict[str, Any]:
    work_started = time.monotonic()
    if objective not in {"speed", "size"} or target_vram <= 0 or not candidates:
        raise ValueError("search requires candidates, target_vram > 0 and objective speed/size")
    if len({canonical_hash(c.model_dump()) for c in candidates}) != len(candidates):
        raise ValueError("duplicate search candidates")
    for candidate in candidates:
        candidate_config = config.model_copy(deep=True)
        candidate_config.quantization = candidate
        build_plan(candidate_config)
    contract = search_contract(config, profile, candidates)
    contract.update(target_vram=target_vram, objective=objective)
    fingerprint = canonical_hash(contract)
    output = output.resolve()
    if resume:
        if not (output / "search.json").is_file():
            raise ValueError("resume requires an existing search journal")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        journal: dict[str, Any] = {
            "schema_version": 1,
            "contract_fingerprint": fingerprint,
            "contract": contract,
            "candidates": [],
            "wall_seconds": 0.0,
            "gpu_seconds": 0.0,
            "status": "running",
            "best": None,
        }
        if resume:
            journal = json.loads((output / "search.json").read_text(encoding="utf-8"))
            if journal.get("schema_version") != 1 or journal.get("contract_fingerprint") != fingerprint:
                raise ValueError("search source/config/profile/runtime contract changed; start a new search")
        start = work_started
        previous_wall = float(journal["wall_seconds"])
        journal["best"] = None

        def save() -> None:
            journal["wall_seconds"] = previous_wall + time.monotonic() - start
            write_json(output / "search.json", journal)

        save()
        operation = current.get()
        try:
            for index, candidate in enumerate(candidates):
                key = canonical_hash({"contract": fingerprint, "candidate": candidate.model_dump()})
                prior = next(
                    (
                        row
                        for row in reversed(journal["candidates"])
                        if row["key"] == key and row["status"] == "completed"
                    ),
                    None,
                )
                if prior:
                    recorded = Measurement.model_validate(prior["measurement"])
                    if prior.get("measurement_fingerprint") != canonical_hash(recorded.model_dump()):
                        raise ValueError("cached candidate measurement changed")
                    prior["feasible"] = recorded.quality_accepted and recorded.inference_vram_bytes <= target_vram
                    manifest = validate_artifact(output / prior["folder"] / "artifact")
                    if manifest["manifest_fingerprint"] != prior["measurement"]["artifact_fingerprint"]:
                        raise ValueError("cached candidate artifact changed")
                    continue
                used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                wall_left = budget.wall_seconds - (previous_wall + time.monotonic() - start)
                gpu_left = (
                    budget.gpu_seconds - float(journal["gpu_seconds"]) if config.device != "cpu" else float("inf")
                )
                if (
                    len(journal["candidates"]) >= budget.max_candidates
                    or min(wall_left, gpu_left) <= 0
                    or used >= budget.artifact_bytes
                ):
                    journal["status"] = "budget_exhausted"
                    break
                # Failed/incomplete attempts are never reused as cache hits or overwritten.
                folder = output / f"candidate-{index:04d}-attempt-{len(journal['candidates']):04d}"
                folder.mkdir(exist_ok=False)
                row: dict[str, Any] = {
                    "key": key,
                    "folder": folder.name,
                    "status": "running",
                    "config": candidate.model_dump(),
                }
                journal["candidates"].append(row)
                save()
                if operation:
                    operation.event("candidate_started", index, len(candidates))
                candidate_config = config.model_copy(deep=True)
                candidate_config.quantization = candidate
                candidate_started = time.monotonic()
                try:
                    response = (
                        executor(candidate_config, profile, folder, min(wall_left, gpu_left))
                        if executor
                        else _worker(
                            candidate_config, profile, folder, min(wall_left, gpu_left), budget.artifact_bytes - used
                        )
                    )
                    data = Measurement.model_validate(response)
                    manifest = validate_artifact(folder / "artifact")
                    if data.artifact_fingerprint != manifest["manifest_fingerprint"]:
                        raise ValueError("worker measured a different artifact")
                    if not math.isfinite(data.latency_ms):
                        raise ValueError("non-finite latency")
                    journal["gpu_seconds"] += max(
                        data.gpu_seconds, time.monotonic() - candidate_started if config.device != "cpu" else 0
                    )
                    measured = data.model_dump()
                    row.update(
                        status="completed",
                        measurement=measured,
                        measurement_fingerprint=canonical_hash(measured),
                        feasible=data.quality_accepted and data.inference_vram_bytes <= target_vram,
                        artifact_bytes=sum(info["bytes"] for info in manifest["files"].values()),
                    )
                except (TimeoutError, BudgetError) as exc:
                    row.update(status="budget_exhausted", reason=str(exc))
                    journal["status"] = "budget_exhausted"
                    if config.device != "cpu":
                        journal["gpu_seconds"] += time.monotonic() - candidate_started
                    break
                except Exception as exc:
                    row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                    if config.device != "cpu":
                        journal["gpu_seconds"] += time.monotonic() - candidate_started
                finally:
                    save()
                used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                if (
                    used > budget.artifact_bytes
                    or journal["wall_seconds"] >= budget.wall_seconds
                    or journal["gpu_seconds"] >= budget.gpu_seconds
                ):
                    journal["status"] = "budget_exhausted"
                    break
            else:
                journal["status"] = "completed"
            feasible = [r for r in journal["candidates"] if r.get("status") == "completed" and r.get("feasible")]
            if journal["status"] != "budget_exhausted":
                if feasible:
                    protocols = {r["measurement"]["protocol_fingerprint"] for r in feasible}
                    if len(protocols) != 1:
                        raise ValueError("candidate measurement protocols differ")
                    best = min(
                        feasible,
                        key=lambda r: r["measurement"]["latency_ms"] if objective == "speed" else r["artifact_bytes"],
                    )
                    journal["best"] = best["folder"]
                else:
                    journal["status"] = "no_feasible_candidate"
        except KeyboardInterrupt:
            journal["status"] = "interrupted"
            if journal["candidates"] and journal["candidates"][-1]["status"] == "running":
                journal["candidates"][-1]["status"] = "interrupted"
            raise
        except Exception:
            journal["status"] = "failed"
            raise
        finally:
            save()
        return journal
