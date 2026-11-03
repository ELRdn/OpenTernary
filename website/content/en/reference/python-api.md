---
description: "The stable version-1 Python API in openternary.api: configuration, planning, conversion, export, validation, evaluation and search."
---

# Python API

`openternary.api` gives Python code the same services the CLI uses, without a subprocess. It is the stable,
**version 1** interface. Importing it never loads PyTorch: heavy dependencies are imported only by the functions
that need them.

## Example

This runs on the tiny model from [the first run](../getting-started/first-run.md):

```python
from pathlib import Path

from openternary.api import (
    build_plan,
    convert_artifact,
    doctor,
    export_artifact,
    load_config,
    validate_artifact,
)

config = load_config(cli_overrides={
    "model.id": "tiny-llama",
    "quantization.scale_granularity": "per_group",
    "quantization.group_size": 64,
    "device": "cpu",
})

plan = build_plan(config)
print("targets:", sum(t["quantizable"] for t in plan["targets"]))

report = convert_artifact(Path("tiny-llama"), Path("api-out/ternary"), config)
print("content fingerprint:", report.content_fingerprint)

export_artifact(Path("api-out/ternary"), Path("api-out/packed"), "ternary-packed")
print("validation:", validate_artifact(Path("api-out/packed"))["artifact_validation"])

print("torch:", doctor()["dependencies"].get("torch"))
```

```text
targets: 14
content fingerprint: sha256:2a25a7bffeeb0d0103ab21c3d42336bd72df75359e6bccd564c33fb16475f54f
validation: file_integrity
torch: 2.15.0
```

The fingerprint is the same one the CLI printed for the same model and settings. Unlike the CLI, these functions
do not create run directories. Where the results go is up to you.

## Functions

### Configuration and planning

`load_config(config_path=None, cli_overrides=None) -> AppConfig`
:   Load a YAML file (or the defaults) and apply overrides given as dotted keys, for example
    `{"model.id": "./m", "quantization.group_size": 64}`. Validation is as strict as on the command line.

`AppConfig`
:   The Pydantic model of the whole configuration (see [Configuration](configuration.md)).
    `config.model_dump()` gives a plain dict.

`build_plan(config, *, operation="quantize") -> dict`
:   What `plan` shows: resolved configuration, adapter, backend, and `targets`, a list of tensors with
    `quantizable`. Reads headers only.

`plan(config_path=None, overrides=None) -> dict`
:   Shortcut for `load_config` followed by `build_plan`.

`doctor() -> dict`
:   What `openternary doctor --json` returns in `result`: Python, platform, dependencies, backends.

### Conversion and artifacts

`convert_artifact(source, destination, config) -> ConvertReport`
:   Quantize the snapshot at `source` into a new artifact at `destination`, with a manifest. The report has
    `content_fingerprint`, `file_hashes`, `shard_count`, `tensor_payload_bytes`, `shard_file_bytes`,
    `snapshot_total_bytes` and `quantization_json`.

`export_artifact(source, destination, format="safetensors") -> dict`
:   Export as `safetensors`, `ternary-packed`, `torchao` or a plugin format. GGUF has its own function,
    `openternary.export.gguf.export_gguf`, because it needs a converter.

`validate_artifact(root) -> dict`
:   Check an artifact's manifest and file hashes. For tensor-level checks use
    `openternary.services.tensor_validation.validate_tensors`.

### Evaluation

`benchmark(config, snapshot_path, suite) -> dict`
:   Run the smoke benchmark (`suite="smoke"`) on a snapshot and return the `benchmark.json` content.

`quality(config, dataset_path, *, snapshot_path=None, model=None, processor=None, split, max_length, stride) -> dict`
:   Run the quality evaluation and return the `quality.json` content. Pass either a `snapshot_path` or an already
    loaded `model` and `processor` together.

`compare_many(baseline, candidates) -> dict`
:   Compare one baseline run directory with a list of candidate run directories.

Diffusion evaluation is in `openternary.services.diffusion.evaluate_diffusion`.

### Search

`run_search(config, profile, candidates, output, budget, *, target_vram, objective="speed", resume=False, executor=None) -> dict`
:   The engine behind `optimize`. `profile` is a `QualityProfile`, `candidates` a list of quantization
    configurations, `budget` a `SearchBudget`, and `target_vram` is in bytes.

`QualityProfile(dataset, baseline, name="balanced-llm-v1", split="validation", max_length=512, stride=256)`
:   The validation profile. `split` can only be `validation`.

`SearchBudget(max_candidates=4, wall_seconds=3600, gpu_seconds=3600, artifact_bytes=20 GiB)`
:   Search limits.

## Stability

Names exported from `openternary.api` follow the version-1 contract: existing behaviour does not change
incompatibly. Everything else (`openternary.services`, `openternary.quant`, …) is internal and may change between
releases. Errors are raised as exceptions. The CLI maps them to [exit codes](output.md#exit-codes).
