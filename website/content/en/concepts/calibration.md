---
description: "How OpenTernary's layer-local calibration works (recon-scale, recon-threshold, recon-soft-to-hard), what it stores, and what the research has found so far."
---

# Calibration

Plain AbsMean quantization picks every scale and threshold with a fixed formula. **Calibration** chooses them by
looking at what the original model actually computes. The goal is to make each quantized layer produce outputs
as close as possible to the original layer's, on real inputs.

## Layer-local reconstruction

OpenTernary calibrates **one linear layer at a time** (`window: per-layer`):

```mermaid
flowchart LR
    T[Original model<br/>teacher] -- run on calibration text --> A[(Captured inputs and outputs<br/>of each target layer<br/>disk-backed cache)]
    A --> L{{For each layer:<br/>minimise ‖ y_teacher − F.linear(x, q·s) ‖}}
    L --> S[Learned parameters<br/>per layer]
    S --> M[Materialize:<br/>hard ternary snapshot]
```

1. **Capture.** The original model runs over the calibration text, and the inputs and outputs of every target
    layer are saved to disk. Large models never have to stay in memory during optimization, and the cache can
    be reused when the same teacher, data and targets come back.
2. **Optimize.** For each layer, the learnable parameters are adjusted for `steps` optimizer steps (Adam or SGD,
    learning rate `lr`) to minimize the reconstruction loss (`mse` or `l1`) between the teacher's output and the
    quantized layer's output, \(F.\mathrm{linear}(x,\; q \cdot s)\).
3. **Hold out.** A share of the samples (`held_out_ratio`, default 0.2) is never used for optimization. The loss
    on it is reported before and after, so you can see whether improvements generalize. OpenTernary checks that
    the training and held-out samples are disjoint by hash.
4. **Materialize.** The final parameters are applied with **hard** ternary codes and written as a normal
    fake-quant snapshot (`artifacts/calibrated_snapshot`), which you can benchmark, evaluate and export like the
    output of `quantize`.

The scales are parameterized so that the starting point equals plain AbsMean exactly. Step 0 reproduces the
uncalibrated model, and any change comes from the optimization.

## The three methods

| `calibration.method` | Learns | Notes |
|---|---|---|
| `recon-scale` | the scale of each tensor or group | The baseline. Codes are fixed by AbsMean; only magnitudes move. |
| `recon-threshold` | scales **and** the zero/non-zero threshold | Requires `threshold_enabled: true`. Gradients pass through the hard decision with a clipped straight-through estimator (`threshold_ste_width`). Thresholds can be per tensor or per group. |
| `recon-soft-to-hard` | scales, with soft ternary assignments during training | Codes start as a soft expectation and are annealed toward hard codes with a temperature schedule (`linear`, `cosine`, `exponential`, from `temperature_start` to `temperature_end`). The last `hard_fraction` of steps use hard codes. The saved snapshot is always hard. |

Configuration mistakes are rejected up front: `recon-threshold` without `threshold_enabled: true` (and the
reverse), a soft-to-hard schedule whose temperature would rise, unknown keys, and non-finite numbers.

## Data

| `calibration.dataset` | Source | Use |
|---|---|---|
| `synthetic` | generated token sequences | testing the pipeline only, never research evidence |
| `wiki-tiny` | WikiText from Hugging Face (`Salesforce/wikitext`, pinned to `dataset_revision`) | real calibration; needs the `datasets` package and a download on first use |

`num_samples` sequences of `seq_len` tokens are used. If `wiki-tiny` cannot be loaded, calibration **fails**.
It falls back to synthetic data only if you set `allow_dataset_fallback: true`, and the run then records the
dataset it actually used.

## Running on limited memory

Calibration is designed to run large models on consumer GPUs:

- **Low-VRAM placement** (`runtime.low_vram: true`, the default): only the layer being optimized is on the GPU.
- **VRAM budget:** OpenTernary uses at most `runtime.vram.max_fraction` (default 80%) of the GPU's memory, keeps
  `reserve_mb` free for your desktop and stops allocating below `min_free_mb`.
- **Microbatch shrinking:** on an out-of-memory error, the step is rolled back and retried with a smaller
  microbatch, down to `runtime.microbatch.min_size` (with `runtime.microbatch.auto: true`, the default).
- **Checkpoints:** every `checkpoint_interval` steps, per-layer state is saved with a manifest of file hashes.
  `--resume` continues from the latest checkpoint, `--resume-from` from a specific one, and
  `--materialize-only` writes the snapshot from a checkpoint without further training.

The [Calibrate guide](../guides/calibrate.md) shows the commands, including recovery after a crash.

## What the research has found so far

Calibration is OpenTernary's main research question, and the honest summary is that **it has not yet produced a
ternary Gemma 4 E2B that passes the quality gate.** From the recorded experiments:

- `recon-scale` lowers the reconstruction loss and the held-out loss on small configurations, and the pipeline
  (capture, checkpoints, resume, materialize, reload) works end to end on the real model.
- `recon-threshold` changes very few hard codes. In the full-model comparison, 166,481 of 1,835,532,288 codes
  (0.009%) changed compared with scale-only calibration.
- All three soft-to-hard schedules completed on all 205 target layers. After hardening and reloading, they
  produced no change in codes, an instruction score of 0 and 8 of 8 collapsed cases. No schedule passed the gate.
- The largest remaining error is concentrated in attention `q_proj`: 86% of the held-out squared error, with
  layer 0's `q_proj` alone at about 47%.

The current analysis is that the optimizer only owns the scales, while the code assignments and the hardening
boundary stay frozen. That is a structural limit, not a matter of training longer. See the
[research status](../project/status.md#research) and the
[decision record](https://github.com/ELRdn/OpenTernary/blob/main/docs/plans/p4-soft-to-hard-redesign-decision.md).
