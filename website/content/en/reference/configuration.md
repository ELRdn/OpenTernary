---
description: "Every key of the OpenTernary YAML configuration, with types, defaults, allowed values and how flags override it."
---

# Configuration

Commands that take `--config` read a YAML file. Everything that defines an experiment can be written down there,
so that the run's saved `config.yaml` describes it completely.

## Rules

- **Precedence:** command-line flags > YAML file > built-in defaults.
- **Strict:** unknown keys, non-finite numbers (`.nan`, `.inf`), out-of-range values and contradictory
  settings are errors (exit code `2`). Nothing is silently ignored.
- **The resolved configuration is saved** as `config.yaml` in every run, including defaults you did not write.
- **Model revisions are not inherited.** If you give a model ID other than the default (on the command line or as
  `model.id`), `model.revision` is not taken from the default Gemma model. Set it yourself, or it is `null`.
- `--dry-run` on any command prints the resolved configuration without running anything.

Flags map to keys as follows:

| Flag | Key |
|---|---|
| `MODEL` argument | `model.id` |
| `--seed` | `seed` |
| `--device` | `device` |
| `--dtype` | `dtype` |
| `--output` | `output` |
| `--group-size` | `quantization.group_size` |
| `--scale-granularity` | `quantization.scale_granularity` |
| `--grouping-scheme` | `quantization.grouping_scheme` |
| `--backend`, `--scheme`, `--weight-dtype` | `quantization.backend`, `.scheme`, `.weight_dtype` |
| `--component` | `model.component` |
| `--suite` | `benchmark.suite` |
| `--thinking` / `--no-thinking` | `benchmark.thinking` |

## A complete example

```yaml
schema_version: 1
model:
  id: google/gemma-4-E2B-it-qat-q4_0-unquantized
  revision: 6befbaca7398925921802abd1f277b495b78b738
  adapter: auto
quantization:
  backend: ternary
  scheme: absmean
  weight_dtype: ternary
  scale_granularity: per_group
  group_size: 128
  target: {attention: true, mlp: true}
  mixed_precision: {}
  passes: []
calibration:
  enabled: true
  method: recon-scale
  dataset: wiki-tiny
  num_samples: 32
  seq_len: 128
  steps: 50
  lr: 0.001
  optimizer: adam
  loss: mse
  checkpoint_interval: 10
  held_out_ratio: 0.2
benchmark:
  suite: smoke
  thinking: false
  generation: {do_sample: false, num_beams: 1, max_new_tokens: 64}
  warmup: true
runtime:
  low_vram: true
  vram: {max_fraction: 0.8, reserve_mb: 1024, min_free_mb: 768}
  microbatch: {auto: true, min_size: 1}
seed: 42
device: cuda
dtype: bf16
```

## Top level

| Key | Type | Default | Description |
|---|---|---|---|
| `schema_version` | `1` | `1` | Configuration schema version. Only `1` exists. |
| `seed` | int | `42` | Seed for Python, NumPy and PyTorch. |
| `device` | `auto` \| `cpu` \| `cuda` | `auto` | Device for running models. `cuda` also means AMD GPUs under ROCm. |
| `dtype` | `bf16` \| `fp16` \| `fp32` | `bf16` | Dtype for running models (benchmark, quality, calibration). Snapshots keep the source dtype. |
| `output` | string \| null | `null` | Run folder. `null`: `runs/<date>_<model>-<method>-g<group size>`. |

## `model`

| Key | Type | Default | Description |
|---|---|---|---|
| `model.id` | string | `google/gemma-4-E2B-it-qat-q4_0-unquantized` | Local folder, or a Hugging Face ID in your local cache. |
| `model.revision` | string \| null | the pinned Gemma commit for the default ID; otherwise `null` | Hugging Face commit to use. Pin it for reproducible results. |
| `model.adapter` | `auto` \| `gemma4` \| `transformers` \| `diffusers` | `auto` | Which adapter classifies and loads the model. `auto` decides from the files. |
| `model.component` | string \| null | `null` | Diffusers pipeline component to target, for example `transformer`. |

## `quantization`

| Key | Type | Default | Description |
|---|---|---|---|
| `quantization.backend` | string | `ternary` | `ternary`, `torchao`, or an installed plugin. |
| `quantization.scheme` | string | `absmean` | The backend's scheme: `absmean` for ternary, `int8-weight-only` for TorchAO. |
| `quantization.weight_dtype` | string | `ternary` | Representation of quantized weights: `ternary`, `int8`. |
| `quantization.activation_dtype` | `preserve` | `preserve` | Activations are never quantized. |
| `quantization.scale_granularity` | `per_tensor` \| `per_group` | `per_tensor` | One scale per tensor, or per group. |
| `quantization.group_size` | int ≥ 1 | `128` | Values per group with `per_group`. |
| `quantization.grouping_scheme` | `last-dim-rowwise-v1` | `last-dim-rowwise-v1` | How groups are formed: consecutive values along each row of the last dimension. |
| `quantization.target.attention` | bool | `true` | Quantize attention projections. |
| `quantization.target.mlp` | bool | `true` | Quantize MLP projections. |
| `quantization.mixed_precision` | map of tensor name → `preserve` \| `ternary` | `{}` | Per-tensor overrides. Names must be quantizable targets ([find them](../guides/inspect-and-plan.md#inspect-what-is-in-the-model)). |
| `quantization.passes` | list of `{name, options}` | `[]` | Pre-quantization passes, applied in order: `clip` (`options: {max_abs: <finite number > 0>}`), `noop` (no options), or plugins. |
| `quantization.backend_options` | map of string → string/int/float/bool | `{}` | Extra options for the backend. |
| `quantization.method` | `naive` \| null | `naive` | Legacy name for the ternary AbsMean method. Ternary only. |
| `quantization.codebook` | list of int \| null | `[-1, 0, 1]` | Legacy. Must be `[-1, 0, 1]` for ternary. |

## `calibration`

Used by `calibrate`. See [Calibration](../concepts/calibration.md) for what each setting does.

| Key | Type | Default | Description |
|---|---|---|---|
| `calibration.enabled` | bool | `false` | Set automatically by `calibrate`. Writing `false` explicitly and running `calibrate` is an error. |
| `calibration.method` | `recon-scale` \| `recon-threshold` \| `recon-soft-to-hard` | `recon-scale` | What is learned. |
| `calibration.window` | `per-layer` | `per-layer` | Reconstruction window. Only per-layer exists. |
| `calibration.dataset` | `synthetic` \| `wiki-tiny` | `synthetic` | Calibration text. `synthetic` is for pipeline checks only. |
| `calibration.dataset_revision` | string (40+ chars) | `b08601e04326c79dfdd32d625aee71d232d685c3` | Commit of `Salesforce/wikitext` for `wiki-tiny`. |
| `calibration.allow_dataset_fallback` | bool | `false` | Allow falling back to `synthetic` if `wiki-tiny` cannot be loaded. |
| `calibration.num_samples` | int 1–1024 | `32` | Number of calibration sequences. |
| `calibration.seq_len` | int 16–512 | `128` | Tokens per sequence. |
| `calibration.held_out_ratio` | float 0.05–0.5 | `0.2` | Share of samples held out from optimization. |
| `calibration.steps` | int ≥ 1 | `50` | Optimizer steps per layer. |
| `calibration.lr` | float > 0 | `0.001` | Learning rate. |
| `calibration.optimizer` | `adam` \| `sgd` | `adam` | Optimizer. |
| `calibration.loss` | `mse` \| `l1` | `mse` | Reconstruction loss. |
| `calibration.checkpoint_interval` | int ≥ 1 | `10` | Save a checkpoint every this many steps. |
| `calibration.seed` | int \| null | `null` | Calibration seed. `null`: use the top-level `seed`. |
| `calibration.init_from` | string \| null | `null` | Earlier calibration run to warm-start from (also `--init-from`). |
| `calibration.threshold_enabled` | bool | `false` | Must be `true` exactly when `method` is `recon-threshold`. |
| `calibration.threshold_lr` | float > 0 \| null | `null` | Learning rate for thresholds. `null`: use `lr`. |
| `calibration.threshold_init_ratio` | float in (0, 1) | `0.5` | Initial threshold, as a ratio of the scale. |
| `calibration.threshold_granularity` | `per_tensor` \| `per_group` | `per_group` | One threshold per tensor or per group. |
| `calibration.threshold_estimator` | `clipped-ste` | `clipped-ste` | Gradient estimator for the hard threshold. |
| `calibration.threshold_ste_width` | float > 0 | `0.1` | Width of the clipped straight-through window. |
| `calibration.threshold_eps` | float in (0, 0.5) | `0.01` | Keeps the threshold ratio away from 0 and 1. |
| `calibration.temperature_schedule` | `linear` \| `cosine` \| `exponential` | `linear` | Soft-to-hard annealing schedule. |
| `calibration.temperature_start` | float > 0 | `1.0` | Initial temperature. Must be ≥ `temperature_end`. |
| `calibration.temperature_end` | float > 0 | `0.05` | Final soft temperature. |
| `calibration.hard_fraction` | float in [0, 1) | `0.1` | Final share of steps that use hard codes. |
| `calibration.zero_logit_bias` | float | `0.0` | Optional fixed bias toward code 0 in the soft path. The final hardening does not use it. |

## `benchmark`

| Key | Type | Default | Description |
|---|---|---|---|
| `benchmark.suite` | `smoke` | `smoke` | The benchmark suite. |
| `benchmark.suites` | list | `["smoke"]` | Deprecated alias. If present it must equal `[suite]`. |
| `benchmark.thinking` | bool | `false` | Gemma 4 thinking mode. |
| `benchmark.warmup` | bool | `true` | Run one warm-up prompt before timing. |
| `benchmark.generation.do_sample` | bool | `false` | Sampling. `false` is greedy decoding. |
| `benchmark.generation.num_beams` | int ≥ 1 | `1` | Beam count. |
| `benchmark.generation.max_new_tokens` | int 1–2048 | `64` | Maximum generated tokens per prompt. |

Changing any of these changes the benchmark's protocol fingerprint, so runs with different settings cannot be
compared.

## `runtime`

Memory policy for calibration on GPUs.

| Key | Type | Default | Description |
|---|---|---|---|
| `runtime.low_vram` | bool | `true` | Keep only the layer being optimized on the GPU. |
| `runtime.vram.max_fraction` | float in (0, 1] | `0.8` | Largest share of GPU memory to use. |
| `runtime.vram.reserve_mb` | int ≥ 0 | `1024` | Memory left free for the desktop and other programs. |
| `runtime.vram.min_free_mb` | int ≥ 0 | `768` | Stop allocating below this much free memory. |
| `runtime.microbatch.auto` | bool | `true` | Retry with a smaller microbatch after an out-of-memory error. |
| `runtime.microbatch.min_size` | int ≥ 1 | `1` | Smallest microbatch. |

## Ready-made files

The repository's [`configs/`](https://github.com/ELRdn/OpenTernary/tree/main/configs) folder has the configurations
used in the research. They are not included in the PyPI package, so download them from GitHub or clone the
repository.

| File | What it is |
|---|---|
| `gemma4-e2b.yaml` | BF16 smoke baseline for Gemma 4 E2B (pinned revision) |
| `gemma4-e2b-calibration.yaml` | `recon-scale`, G128, `wiki-tiny`, 32 × 128, 50 steps |
| `gemma4-e2b-calibration-smoke.yaml` | the same, at 8 × 64 and 2 steps: a pipeline check |
| `gemma4-e2b-threshold.yaml` | `recon-threshold` |
| `gate-real-rocm-5.yaml` | the 5-module AMD GPU gate configuration |
