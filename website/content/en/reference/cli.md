---
description: Every OpenTernary command and option, with defaults, notes and examples.
---

# CLI reference

Every command and option of OpenTernary 0.1.0a2. `openternary <command> --help` shows the same options in your
terminal, and the TUI builds its forms from the same definitions.

```text
openternary [--version] [--help] <command> [arguments] [options]
```

| Global option | |
|---|---|
| `--version` / `-V` | Print the version (and the git commit for source installs) and exit |
| `--help` | Show help and exit |
| `--install-completion` / `--show-completion` | Install, or print, shell completion |

With no command, in an interactive terminal, `openternary` opens the [TUI](../getting-started/tui.md). Elsewhere
it prints help and exits with code `2`.

## Options on every command

Every command (except `ui`) also accepts:

| Option | |
|---|---|
| `--json` | Print one versioned JSON result on stdout; human output goes to stderr. See [Output](output.md). |
| `--quiet` | Suppress progress output; the result and errors are still printed |
| `--events-jsonl FILE` | Write progress events to a **new** file (its folder is created). Not with `--dry-run`. |

The model argument (`[MODEL]`) is optional where a config file can provide `model.id`.


## Workflow

### `inspect`

Inspect a model's architecture and quantizable modules (header-only by default).

| Argument / option | Description |
|---|---|
| `[MODEL]` | Model: a local folder, or a Hugging Face ID in your local cache |
| `--config` / `-c` | YAML configuration file; flags override it |
| `--output` / `-o` | Run (or artifact) folder to create; default `runs/<date>_<slug>` |
| `--seed` | Random seed (default 42) |
| `--device` | `auto`, `cpu` or `cuda` (also for AMD ROCm) |
| `--dtype` | Compute dtype: `bf16` (default), `fp16`, `fp32` |
| `--load-weights` | Also read weight values for distribution statistics (needs PyTorch) |
| `--dry-run` | Resolve and check everything, print the plan, create nothing |
| `--verbose` / `-v` | More detailed log output |

Reads safetensors headers and `config.json` only, unless `--load-weights`. Writes `inspection.json` in a new run. Guide: [Inspect and plan](../guides/inspect-and-plan.md).

```bash
openternary inspect google/gemma-4-E2B-it-qat-q4_0-unquantized
```

### `plan`

Check a quantization plan without downloads, tensor loading or writes.

| Argument / option | Description |
|---|---|
| `[MODEL]` | Model: a local folder, or a Hugging Face ID in your local cache |
| `--config` / `-c` | YAML configuration file; flags override it |
| `--backend` | Quantization backend: `ternary` (default), `torchao`, or a plugin |
| `--scheme` | Backend scheme: `absmean` (ternary), `int8-weight-only` (torchao) |
| `--weight-dtype` | Representation of quantized weights: `ternary`, `int8` |
| `--component` | Diffusers pipeline component to target, e.g. `transformer` |

Creates no run directory and writes nothing. Per-group settings come from `--config` (or use `quantize --dry-run`). Unknowns are reported as unknown.

```bash
openternary plan ./my-model --backend ternary --scheme absmean --json
```

### `quantize`

Quantize a model to a ternary fake-quant snapshot.

| Argument / option | Description |
|---|---|
| `[MODEL]` | Model: a local folder, or a Hugging Face ID in your local cache |
| `--config` / `-c` | YAML configuration file; flags override it |
| `--output` / `-o` | Run (or artifact) folder to create; default `runs/<date>_<slug>` |
| `--seed` | Random seed (default 42) |
| `--device` | `cpu` or `auto`; `cuda` is refused (conversion runs on the CPU) |
| `--dtype` | Refused for `quantize`: the snapshot keeps the source dtype |
| `--group-size` | Values per group with `per_group` (default 128) |
| `--scale-granularity` | `per_tensor` (default) or `per_group` |
| `--grouping-scheme` | Grouping definition: `last-dim-rowwise-v1` (the only one) |
| `--backend` | Quantization backend: `ternary` (default), `torchao`, or a plugin |
| `--scheme` | Backend scheme: `absmean` (ternary), `int8-weight-only` (torchao) |
| `--weight-dtype` | Representation of quantized weights: `ternary`, `int8` |
| `--component` | Diffusers pipeline component to target, e.g. `transformer` |
| `--dry-run` | Resolve and check everything, print the plan, create nothing |
| `--verbose` / `-v` | More detailed log output |

Writes `runs/<run>/artifacts/snapshot` and `quantization.json`. Conversion runs on the CPU and keeps the source dtype. Guide: [Quantize](../guides/quantize.md).

```bash
openternary quantize ./my-model --scale-granularity per_group --group-size 128 --output runs/g128
```

### `calibrate`

Calibrate ternary scales/thresholds layer by layer (recon-scale).

| Argument / option | Description |
|---|---|
| `[MODEL]` | Model: a local folder, or a Hugging Face ID in your local cache |
| `--config` / `-c` | YAML configuration file; flags override it |
| `--output` / `-o` | Run (or artifact) folder to create; default `runs/<date>_<slug>` |
| `--seed` | Random seed (default 42) |
| `--device` | `auto`, `cpu` or `cuda` (also for AMD ROCm) |
| `--dtype` | Compute dtype: `bf16` (default), `fp16`, `fp32` |
| `--dry-run` | Resolve and check everything, print the plan, create nothing |
| `--verbose` / `-v` | More detailed log output |
| `--resume` | Continue from the latest checkpoint in `--output` |
| `--resume-from` | Continue from this checkpoint |
| `--materialize-only` | Skip training; write the calibrated snapshot from a checkpoint |
| `--init-from` | Warm-start from an earlier calibration run (not with `--resume`) |
| `--finalize` | Regenerate `calibration.json`/`metrics.json` from saved state, without optimizing |
| `--preflight` | Check source, canonical targets, environment and budget only; needs `--output` |

Needs a config with a `calibration` section (`enabled` is set for you). `--finalize`, `--materialize-only`, `--resume` and `--preflight` work on an existing `--output`. Guide: [Calibrate](../guides/calibrate.md).

```bash
openternary calibrate --config calib.yaml --output runs/calib
openternary calibrate --config calib.yaml --output runs/calib --resume
```

### `export`

Export an artifact to another format; the source is never modified.

| Argument / option | Description |
|---|---|
| `RUN` | Run or artifact folder to export (never modified) |
| `--format` | `safetensors` (default), `ternary-packed`, `torchao`, `gguf` |
| `--output` / `-o` | New folder to write; must not exist |
| `--converter` | GGUF: path to llama.cpp's `convert_hf_to_gguf.py` (required for `gguf`) |
| `--output-dtype` | GGUF output: `f16` (default), `bf16`, `f32` |
| `--converter-python` | GGUF: Python to run the converter with (an isolated environment) |
| `--timeout-seconds` | GGUF: stop the converter after this many seconds (default 3600) |
| `--dry-run` | Resolve and check everything, print the plan, create nothing |

Never modifies the source and never overwrites the destination. Writes atomically with a manifest. Guide: [Export](../guides/export.md).

```bash
openternary export runs/g128 --format ternary-packed --output runs/g128-packed
```

## Evaluate and compare

### `benchmark`

Run the smoke inference benchmark.

| Argument / option | Description |
|---|---|
| `[MODEL]` | Model: a local folder, or a Hugging Face ID in your local cache |
| `--config` / `-c` | YAML configuration file; flags override it |
| `--output` / `-o` | Run (or artifact) folder to create; default `runs/<date>_<slug>` |
| `--seed` | Random seed (default 42) |
| `--device` | `auto`, `cpu` or `cuda` (also for AMD ROCm) |
| `--dtype` | Compute dtype: `bf16` (default), `fp16`, `fp32` |
| `--suite` | Benchmark suite: `smoke` (the only one) |
| `--thinking` / `--no-thinking` | Gemma 4 thinking mode (default off) |
| `--dry-run` | Resolve and check everything, print the plan, create nothing |
| `--verbose` / `-v` | More detailed log output |

Five fixed prompts, greedy decoding. Records protocol and result fingerprints, tokens, latency and resources. Guide: [Evaluate and compare](../guides/evaluate-and-compare.md#benchmark-the-smoke-suite).

```bash
openternary benchmark runs/g128/artifacts/snapshot --device cuda --dtype bf16 --output runs/g128-bench
```

### `quality`

Evaluate frozen English/Japanese perplexity and instruction cases.

| Argument / option | Description |
|---|---|
| `[MODEL]` | Model: a local folder, or a Hugging Face ID in your local cache |
| `--data` | Frozen quality dataset JSON (required) |
| `--split` | `validation` (default) or `test` |
| `--max-length` | Perplexity context length (default 512) |
| `--stride` | Perplexity sliding-window stride (default 256) |
| `--config` / `-c` | YAML configuration file; flags override it |
| `--output` / `-o` | Run (or artifact) folder to create; default `runs/<date>_<slug>` |
| `--seed` | Random seed (default 42) |
| `--device` | `auto`, `cpu` or `cuda` (also for AMD ROCm) |
| `--dtype` | Compute dtype: `bf16` (default), `fp16`, `fp32` |
| `--dry-run` | Validate the dataset without loading the model |
| `--verbose` / `-v` | More detailed log output |

English/Japanese perplexity and exact-answer instruction cases on one split of a frozen dataset. Writes `quality.json` (report schema 3). Guide: [quality](../guides/evaluate-and-compare.md#quality-perplexity-and-instructions).

```bash
openternary quality runs/g128/artifacts/snapshot --data quality.json --device cuda --output runs/g128-q
```

### `evaluate`

Evaluate from a fixed profile (diffusion); image proxies never imply human acceptance.

| Argument / option | Description |
|---|---|
| `MODEL` | Diffusers pipeline folder |
| `--profile` | Evaluation profile JSON (required) **(required)** |
| `--config` / `-c` | YAML configuration file; flags override it |
| `--output` / `-o` | Run (or artifact) folder to create; default `runs/<date>_<slug>` |
| `--component-artifact` | Quantized component to swap in (safetensors) |
| `--evaluator` | Evaluator: `diffusion` (default) or a plugin |
| `--dry-run` | Resolve and check everything, print the plan, create nothing |

Generates images under a fixed profile and records hashes, timings and pixel statistics. Results always need human review. Guide: [Diffusion models](../guides/diffusion.md).

```bash
openternary evaluate ./pipeline --profile profile.json --component-artifact runs/t/artifacts/snapshot
```

### `compare`

Compare runs under the same conditions (observational unless acceptance is required).

| Argument / option | Description |
|---|---|
| `[BASELINE]` | Baseline run folder |
| `[QUANTIZED]` | Candidate run folder |
| `--config` / `-c` | YAML configuration file; flags override it |
| `--output` / `-o` | Run (or artifact) folder to create; default `runs/<date>_<slug>` |
| `--seed` | Random seed (default 42) |
| `--device` | `auto`, `cpu` or `cuda` (also for AMD ROCm) |
| `--dtype` | Compute dtype: `bf16` (default), `fp16`, `fp32` |
| `--candidate` | Another candidate, compared with the same baseline (repeatable) |
| `--require-acceptance` | Exit with code 6 unless the quality gate accepts |
| `--dry-run` | Resolve and check everything, print the plan, create nothing |
| `--verbose` / `-v` | More detailed log output |

Refuses runs measured under different conditions (exit 2). Benchmark runs: observational. Quality runs: judged by the quality gate when the evidence is complete. Writes `compare.json` and `acceptance.json`.

```bash
openternary compare runs/base-q runs/g128-q --require-acceptance --output runs/gate
```

### `optimize`

Search explicit candidates within budgets; never selects on the test split.

| Argument / option | Description |
|---|---|
| `MODEL` | Model to search on |
| `--search-space` | JSON list of quantization configurations (required) **(required)** |
| `--quality-profile` | Validation profile JSON (required) **(required)** |
| `--target-vram` | Feasibility limit for inference VRAM, e.g. `12GiB` (required) **(required)** |
| `--output` / `-o` | Search folder (journal, candidates, `search.json`) |
| `--config` / `-c` | YAML configuration file; flags override it |
| `--objective` | `speed` (default: lowest latency) or `size` (smallest artifact) |
| `--max-candidates` | At most this many candidates, 1–256 (default 4) |
| `--wall-seconds` | Wall-clock budget in seconds (default 3600) |
| `--gpu-seconds` | GPU-time budget in seconds (default 3600) |
| `--max-artifact-bytes` | Disk budget for candidate artifacts (default `20GiB`) |
| `--resume` / `--no-resume` | Continue an interrupted search with the same inputs |
| `--dry-run` | Resolve and check everything, print the plan, create nothing |

Evaluates each candidate in a fresh worker. Exit `6` when no candidate is feasible, `7` when a budget runs out; no best candidate is published in either case. Guide: [Constrained search](../guides/optimize.md).

```bash
openternary optimize ./my-model --search-space space.json --quality-profile profile.json --target-vram 12GiB --dry-run
```

## Environment and artifacts

### `ui`

Open the full-screen interface (also: openternary with no arguments at a terminal).

| Argument / option | Description |
|---|---|
| `--runs` | Folder that holds run directories (default `runs`) |

Opens only in an interactive terminal; otherwise exits with code 2. `openternary` with no arguments does the same. Guide: [The full-screen interface](../getting-started/tui.md).

### `doctor`

Show installed packages and backends; never installs or runs a model.

| Argument / option | Description |
|---|---|
| `--probe-runtime` | Run a small FP32/BF16 test on the device (loads no model) |
| `--device` | Device for `--probe-runtime`: `cpu` (default) or `cuda` |

Never installs or loads a model. In an interactive terminal it checks PyPI for updates at most once a day.

```bash
openternary doctor --probe-runtime --device cuda
```

### `setup`

Detect the GPU and install a matching PyTorch build (CUDA / ROCm / MPS / CPU) into this environment.

| Argument / option | Description |
|---|---|
| `--backend` | `auto` (default), `cuda`, `rocm`, `mps`, `cpu` |
| `--torch` | A specific PyTorch version, e.g. `2.13.0` or `2.13.0+rocm10.0.0` |
| `--variant` | A specific build variant, e.g. `cu130`, `rocm10.0.0`, `cpu` |
| `--amd-device` | AMD kernels: `auto` (this GPU if published), `all`, or a gfx target |
| `--list` | Show the detected hardware and every compatible build |
| `--restore` | Reinstall exactly the build the last `setup` recorded |
| `--profile` | With `--restore`: read the record from this file |
| `--pre` | Include pre-release builds |
| `--yes` / `-y` | Do it without asking (required outside a terminal and with `--json`) |
| `--dry-run` | Resolve and check everything, print the plan, create nothing |

Installs into the environment that runs `openternary` and tests the device afterwards (exit `5` if the test fails). Needs uv. Guide: [Installation](../getting-started/installation.md#3-install-pytorch-for-your-hardware-openternary-setup).

```bash
openternary setup --list
openternary setup --backend rocm --amd-device gfx1201
```

### `update`

Check PyPI for a newer OpenTernary, upgrade, and restore the PyTorch build chosen by setup.

| Argument / option | Description |
|---|---|
| `--check` | Only check PyPI for a newer release |
| `--pre` | Consider pre-releases (default: yes if the installed version is one) |
| `--yes` / `-y` | Do it without asking (required outside a terminal and with `--json`) |
| `--dry-run` | Resolve and check everything, print the plan, create nothing |

Upgrades according to how OpenTernary was installed, then restores the PyTorch build that `setup` recorded. On Windows uv-tool installs it prints the commands instead.

```bash
openternary update --check
```

### `plugins`

List installed extensions without executing their code.

Lists installed plugins from entry-point metadata without importing their code. See [Plugins](plugins.md).

### `cache-info`

Show cache configuration.

Shows where uv and Hugging Face keep their caches.

### `backends list`

List backends with their installation and capability status.

For each backend: installation, quality maturity, and what has been verified (conversion, model reload, native low-bit execution).

### `backends info`

Show one backend's installation and capability details.

| Argument / option | Description |
|---|---|
| `NAME` | Backend name, e.g. `ternary`, `torchao` |

```bash
openternary backends info torchao --json
```

### `artifacts validate`

Check an artifact's manifest and file integrity (not model quality).

| Argument / option | Description |
|---|---|
| `PATH` | Artifact folder |
| `--level` | `files` (default: manifest + file hashes) or `tensors` (also open every tensor) |

Never modifies the artifact. `file_integrity` / `tensor_integrity` do not imply that the model loads or is accurate. Guide: [Manage artifacts](../guides/artifacts.md).

### `cache list`

List the managed artifacts under a cache root.

| Argument / option | Description |
|---|---|
| `ROOT` | Folder to search for artifacts |

```bash
openternary cache list runs --json
```

### `cache remove`

Remove one managed artifact after checking nothing refers to it (--execute to delete).

| Argument / option | Description |
|---|---|
| `TARGET` | Artifact to remove |
| `--root` | Folder within which references are checked; the target must be inside it (required) **(required)** |
| `--execute` | Actually delete; without it, only check and report |
| `--dry-run` | Check only (the default without `--execute`) |

Refuses artifacts that are still referenced inside `--root`, the root itself, and anything outside it. References from outside the root are reported as `unknown`.

```bash
openternary cache remove runs/old-export --root runs            # check
openternary cache remove runs/old-export --root runs --execute  # delete
```
