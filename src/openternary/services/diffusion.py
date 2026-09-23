"""Reproducible diffusion measurements, explicitly separate from human quality acceptance."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from openternary.config.schema import AppConfig, BaseModel
from openternary.experiment.metadata import write_json
from openternary.services.artifacts import canonical_hash, file_hash, validate_artifact
from openternary.services.errors import CapabilityError
from openternary.services.identity import snapshot_identity
from openternary.services.resources import measured, phase


class DiffusionProfile(BaseModel):
    schema_version: Literal[1] = 1
    family: Literal["diffusion"] = "diffusion"
    name: str = Field(min_length=1)
    prompts: list[str] = Field(min_length=1, max_length=256)
    seeds: list[int] = Field(min_length=1, max_length=256)
    steps: int = Field(default=20, ge=1, le=1000)
    width: int = Field(default=512, ge=8, le=8192)
    height: int = Field(default=512, ge=8, le=8192)
    guidance_scale: float = Field(default=3.5, ge=0)
    component: Literal["transformer", "unet"] | None = None
    warmup: bool = True

    @model_validator(mode="after")
    def validate_cases(self) -> DiffusionProfile:
        if len(self.prompts) != len(self.seeds) or any(seed < 0 for seed in self.seeds):
            raise ValueError("one nonnegative seed is required per prompt")
        if self.width % 8 or self.height % 8 or any(not prompt.strip() for prompt in self.prompts):
            raise ValueError("image dimensions must be multiples of 8 and prompts must be non-empty")
        return self


@measured
def evaluate_diffusion(
    config: AppConfig, source: Path, profile: DiffusionProfile, output: Path, component_artifact: Path | None = None
) -> dict[str, Any]:
    if bool(profile.component) != bool(component_artifact):
        raise ValueError("component and component_artifact must be supplied together")
    from openternary.adapters.registry import component_inventory

    components = component_inventory(source)
    if profile.component and not any(row["name"] == profile.component for row in components):
        raise CapabilityError(f"pipeline does not contain {profile.component}")
    import torch
    from diffusers import DiffusionPipeline

    from openternary.benchmark.runner import _dtype_to_torch_dtype, _resolve_device_map

    resolved = _resolve_device_map(config.device)
    device = resolved.get("", "cpu") if isinstance(resolved, dict) else "cuda:0"
    dtype = _dtype_to_torch_dtype(config.dtype)
    phase("load", device)
    replacements: dict[str, Any] = {}
    manifest = None
    if component_artifact is not None:
        manifest = validate_artifact(component_artifact)
        if manifest["format"] != "safetensors":
            raise CapabilityError("diffusion component evaluation currently requires safetensors")
        if manifest.get("provenance", {}).get("component") != profile.component:
            raise ValueError("component artifact identity does not match profile")
        origin = manifest.get("provenance", {}).get("identity", {})
        if origin.get("source_fingerprint") != snapshot_identity(source).get("source_fingerprint"):
            raise ValueError("component artifact source identity does not match pipeline")
        import diffusers

        from openternary.adapters.registry import read_object

        class_name = read_object(component_artifact / "config.json").get("_class_name")
        cls = getattr(diffusers, str(class_name), None)
        if cls is None or not isinstance(cls, type) or not issubclass(cls, diffusers.ModelMixin):
            raise CapabilityError(f"unsupported diffusion component class: {class_name}")
        loader: Any = cls
        replacements[str(profile.component)] = loader.from_pretrained(
            str(component_artifact), torch_dtype=dtype, local_files_only=True
        )
        from openternary.services.snapshot import SnapshotReader

        loaded_state = replacements[str(profile.component)].state_dict()
        with SnapshotReader(component_artifact) as reader:
            if set(reader.keys()) != set(loaded_state):
                raise ValueError("diffusion component reload tensor inventory mismatch")
            for name in reader.keys():  # noqa: SIM118 - SnapshotReader is not a mapping
                expected = reader.get_tensor(name).to(dtype=dtype)
                if not torch.equal(expected, loaded_state[name].cpu()):
                    raise ValueError(f"diffusion component reload tensor mismatch: {name}")
    started = time.perf_counter()
    pipeline = DiffusionPipeline.from_pretrained(str(source), torch_dtype=dtype, local_files_only=True, **replacements)
    pipeline = pipeline.to(device)
    load_ms = (time.perf_counter() - started) * 1000
    phase("inference", device)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "profile": profile.model_dump(exclude={"component"}),
        "identity": snapshot_identity(source),
        "dtype": config.dtype,
        "device": device,
        "scheduler": type(pipeline.scheduler).__name__,
        "scheduler_config": dict(pipeline.scheduler.config),
    }
    rows = []

    def generate(index: int) -> Any:
        import numpy as np
        from PIL import Image

        pixels = pipeline(
            prompt=profile.prompts[index],
            generator=torch.Generator(device=device).manual_seed(profile.seeds[index]),
            num_inference_steps=profile.steps,
            width=profile.width,
            height=profile.height,
            guidance_scale=profile.guidance_scale,
            output_type="np",
        ).images[0]
        if not np.isfinite(pixels).all():
            raise ValueError("diffusion produced non-finite pixels")
        return Image.fromarray((pixels.clip(0, 1) * 255).round().astype("uint8"))

    with torch.inference_mode():
        if profile.warmup:
            generate(0)
        for index, prompt in enumerate(profile.prompts):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            image = generate(index)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latency = (time.perf_counter() - started) * 1000
            path = output / f"image-{index:04d}.png"
            if path.exists():
                raise FileExistsError(f"image already exists: {path}")
            image.save(path)
            from PIL import ImageStat

            statistics = ImageStat.Stat(image.convert("RGB"))
            rows.append(
                {
                    "id": index,
                    "prompt": prompt,
                    "seed": profile.seeds[index],
                    "file": path.name,
                    "sha256": file_hash(path),
                    "latency_ms": latency,
                    "pixel_mean": statistics.mean,
                    "pixel_stddev": statistics.stddev,
                }
            )
    report = {
        "report_schema_version": 1,
        "family": "diffusion",
        "protocol": protocol,
        "protocol_fingerprint": canonical_hash(protocol),
        "profile_fingerprint": canonical_hash(profile.model_dump()),
        "component_artifact_fingerprint": manifest["manifest_fingerprint"] if manifest else None,
        "model_load_ms": load_ms,
        "results": rows,
        "quality_acceptance": {"accepted": False, "status": "human_review_required"},
        "note": "Pixel statistics and generation timing are observations, not image quality acceptance.",
    }
    write_json(output / "evaluation.json", report)
    return report
