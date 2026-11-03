---
description: "Every file OpenTernary writes in a run directory and an artifact, and what each contains."
---

# Run directory

## Layout

```text
runs/<name>/
├── config.yaml            always   the fully resolved configuration
├── environment.json       always   versions, git commit, Python, OS, hardware, seed, the config
├── model.json             always   {"id": …, "revision": …}
├── metrics.json           always   status, run_id and the headline results
├── last_failure.json      sometimes  a failed retry of a run that had already completed
├── .execution.lock        while running  held by the process that owns the run
├── logs/
│   └── openternary.log
├── artifacts/             model snapshots, checkpoints and caches
└── <result file>          one per command, see below
```

### Naming

| `--output` | Folder |
|---|---|
| not given | `runs/<YYYY-MM-DD>_<model name>-<method>-g<group size>`, e.g. `runs/2026-11-03_tiny-llama-naive-g128` |
| given | that path |

If the folder exists, `-001`, `-002`, … is appended. A run is never written into an existing folder, except by
recovery commands (`calibrate --resume`, `--finalize`, `--materialize-only`), which deliberately reuse their
`--output`. A lock file, `runs/.run.lock`, makes the choice of name safe when several processes start at once.

## Common files

### `config.yaml`

The configuration that actually ran, after flags, YAML and defaults were merged. It contains every key, so it can be
passed back with `--config` to repeat the run. See [Configuration](configuration.md).

### `environment.json`

| Key | |
|---|---|
| `run_id` | 8 hexadecimal characters, unique per run |
| `timestamp` | start time, UTC, ISO 8601 |
| `openternary_version` | e.g. `0.1.0a2` |
| `python_version`, `python_executable` | the interpreter that ran |
| `uv_lock_hash` | first 16 hex characters of the SHA-256 of `uv.lock` in the working directory, or `unknown` |
| `git_commit`, `git_branch`, `code_provenance` | the source revision, when run from a git checkout |
| `seed`, `PYTHONHASHSEED` | seeds |
| `config` | the configuration (same as `config.yaml`) |
| `hardware` | platform, processor, RAM in GB, GPU name, whether CUDA/ROCm is available |

### `metrics.json`

Always has `status` and `run_id`.

| `status` | Meaning |
|---|---|
| `not_run` | created, nothing finished yet |
| `completed` | the command finished (not a quality verdict) |
| `failed` | it failed: `error_message` and `exit_code` are added |
| `interrupted` | interrupted (exit code 130) |

Completed runs add the command's headline numbers, for example the quantization summary or calibration losses. If a
run that had already completed is retried and the retry fails, `metrics.json` keeps the earlier result and the new
failure goes to `last_failure.json` (same fields as a failed `metrics.json`).

## Result files by command

| Command | Result file | Main contents |
|---|---|---|
| `inspect` | `inspection.json` | adapter, architecture, every tensor (`name`, `shape`, `dtype`, `param_count`, `role`, `quantizable`, `exclude_reason`), summary, memory estimate, fingerprint |
| `quantize` | `quantization.json` | settings, per-tensor statistics, total groups, content fingerprint, bit accounting (`packed_estimate`) |
| `calibrate` | `calibration.json` | method and settings, per-layer losses, train and held-out results, contamination check, peak VRAM, data provenance |
| `benchmark` | `benchmark.json` | suite and protocol fingerprint, per-prompt tokens and text, result fingerprint, latency, tokens/s, resources |
| `quality` | `quality.json` | report schema 3: identity, protocol, dataset fingerprint and audit, perplexity per language, instruction results, `summary`, resources, measurement environment |
| `evaluate` | `evaluation.json` | profile, images (hashes, timings, pixel statistics), `human_review_required` |
| `compare` | `compare.json`, `acceptance.json` | matching checks, per-prompt diffs or quality changes; the verdict |
| `optimize` | `search.json` (+ journal, candidate folders) | every candidate with its measurement and feasibility; the outcome |

`quality.json`'s `summary` holds the four numbers the gate uses: `general_ppl`, `japanese_ppl`,
`instruction_score` and `collapse_count`.

## Artifacts

An artifact is a model folder with a manifest. In a run it is `artifacts/snapshot` (`quantize`) or
`artifacts/calibrated_snapshot` (`calibrate`). An `export` output is an artifact folder of its own, not a run.

```text
<artifact>/
├── openternary-manifest.json     format, schema, provenance and the SHA-256 of every file
├── config.json, tokenizer files, chat template, generation_config.json   copied from the source
├── openternary-fixture.json      only for synthetic test models; never accepted as quality evidence
│
│   safetensors snapshot:
├── model-00001-of-0000N.safetensors
├── model.safetensors.index.json
├── quantization.json
│
│   ternary-packed:
├── packed.json                   layout of codes, scales, shapes and padding
└── tensors/000000.safetensors …  one file per tensor
```

The manifest records `schema_version`, `format`, `execution`, `artifact_validation`, `model_reload`,
`quality_acceptance`, the source and its interface (tokenizer and processor files), the configuration, backend,
adapter, component, passes, a code hash, the hash and size of every file, and the history of earlier conversions.
`openternary artifacts validate` checks an artifact against it.

## Calibration extras

```text
artifacts/
├── calibrated_snapshot/   the result
├── checkpoint/            module_00000.pt, …: resumable state, with a manifest of hashes
├── calibration_state/     learned parameters per layer (safetensors)
├── activation_cache/      captured teacher activations (with activation_cache_train/, activation_cache_held/)
└── threshold_metrics.jsonl
```

Activation caches can be much larger than the model. They can be deleted once you no longer need to resume the run
or reuse the cache.
