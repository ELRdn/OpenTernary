---
description: "Run layer-local calibration, choose a method, resume after interruptions, materialize from checkpoints, and recover a run's records."
---

# Calibrate

`calibrate` tries to recover the quality lost by ternary conversion. It tunes each layer's parameters so that its
output matches the original model's. Read [Calibration](../concepts/calibration.md) first for how it works and
what results to expect.

## A minimal configuration

Calibration is configured in YAML. This is a small, complete file. It is a pipeline check on the tiny model from
[the first run](../getting-started/first-run.md), with synthetic data:

```yaml title="calib-smoke.yaml"
model:
  id: tiny-llama          # a folder, or a Hugging Face id in your cache
  revision: null
quantization:
  group_size: 64
  scale_granularity: per_group
calibration:
  enabled: true
  method: recon-scale
  dataset: synthetic      # pipeline check only: never research evidence
  num_samples: 8
  seq_len: 32
  steps: 5
  checkpoint_interval: 5
device: cpu
dtype: fp32
```

```bash
openternary calibrate --config calib-smoke.yaml --dry-run          # check; creates nothing
openternary calibrate --config calib-smoke.yaml --output runs/calib-smoke
```

It finishes in seconds and prints:

```text
    Calibrate — tiny-llama
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric          ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Dataset         │ synthetic │
│ Steps           │ 5         │
│ Initial loss    │ 0.009966  │
│ Final loss      │ 0.009901  │
│ Held-out before │ 0.009592  │
│ Held-out after  │ 0.009533  │
│ Scale delta     │ 0.000074  │
│ Contamination   │ False     │
└─────────────────┴───────────┘
```

`Initial` and `Final loss` are the training reconstruction loss. `Held-out before/after` is the same loss on
samples that were never trained on. `Contamination: False` means the training and held-out samples were checked
and found disjoint.

## Real calibration on Gemma 4 E2B

The research configuration uses WikiText and the pinned Gemma revision. It needs the `datasets` package
(`openternary setup` installs it), and WikiText is downloaded on first use. Ready-made files are in the
repository's [`configs/`](https://github.com/ELRdn/OpenTernary/tree/main/configs) folder:

| File | Purpose |
|---|---|
| [`gemma4-e2b-calibration.yaml`](https://github.com/ELRdn/OpenTernary/blob/main/configs/gemma4-e2b-calibration.yaml) | Phase 4.1 acceptance: `recon-scale`, G128, `wiki-tiny`, 32 samples × 128 tokens, 50 steps per layer |
| [`gemma4-e2b-calibration-smoke.yaml`](https://github.com/ELRdn/OpenTernary/blob/main/configs/gemma4-e2b-calibration-smoke.yaml) | Same model and targets, 8 × 64 tokens and 2 steps: checks the pipeline and progress output, not quality |
| [`gemma4-e2b-threshold.yaml`](https://github.com/ELRdn/OpenTernary/blob/main/configs/gemma4-e2b-threshold.yaml) | `recon-threshold`: learned thresholds |

```bash
openternary calibrate --config configs/gemma4-e2b-calibration.yaml --device cuda --output runs/e2b-recon-scale
```

`--device cuda` works for NVIDIA and AMD (ROCm) GPUs.

### Preflight

Before committing hours of GPU time to the research target, check it:

```bash
openternary calibrate --config configs/gemma4-e2b-calibration.yaml --preflight --output runs/e2b-preflight
```

`--preflight` loads nothing heavy. It checks that the source is the pinned revision, that the weights are present
and match their recorded SHA-256, that the canonical set of target layers (205 for Gemma 4 E2B) is found, and that
the protocol files and resource budget are in place. It writes `preflight.json` and exits with `0` on a pass or
`2` with the failed checks listed.

!!! note

    Preflight checks against the **canonical research target**, Gemma 4 E2B at its pinned revision. For any other
    model it reports `revision_pinned`, `canonical_target_count` and similar checks as failed. That is expected:
    use `--dry-run` to check other models.

## Choosing a method

Set `calibration.method` (see [the three methods](../concepts/calibration.md#the-three-methods)):

=== "recon-scale"

    ```yaml
    calibration:
      enabled: true
      method: recon-scale
    ```

=== "recon-threshold"

    ```yaml
    calibration:
      enabled: true
      method: recon-threshold
      threshold_enabled: true          # required with recon-threshold
      threshold_granularity: per_group
      threshold_init_ratio: 0.5
      threshold_lr: null               # null: use lr
    ```

=== "recon-soft-to-hard"

    ```yaml
    calibration:
      enabled: true
      method: recon-soft-to-hard
      temperature_schedule: cosine     # linear | cosine | exponential
      temperature_start: 1.0
      temperature_end: 0.05
      hard_fraction: 0.1               # final 10% of steps use hard codes
    ```

Warm-start from an earlier run's learned scales with `--init-from runs/<earlier-run>`. This cannot be combined
with `--resume`.

## Progress

Calibration reports detailed progress: dataset preparation, teacher loading, activation capture per batch, each
layer's steps with the current loss, checkpoints, held-out evaluation and materialization. In a terminal you see
it live. With `--events-jsonl FILE` it is also written as [JSON lines](../reference/output.md#progress-events-jsonl),
and the TUI's Jobs tab shows it as progress bars with ETAs.

## Interruptions and recovery

Long runs get interrupted. OpenTernary is designed to lose as little as possible.

| Situation | Command |
|---|---|
| Continue from the latest checkpoint | `openternary calibrate --config X.yaml --output runs/R --resume` |
| Continue from a specific checkpoint | `… --output runs/R --resume-from <checkpoint>` (files in `runs/R/artifacts/checkpoint/`) |
| Skip training: write the snapshot from the latest checkpoint | `… --output runs/R --materialize-only` |
| Rewrite `calibration.json` / `metrics.json` from existing state, without optimizing | `… --output runs/R --finalize` |

- Resuming reuses the existing run directory. It does not create `R-001`.
- Checkpoints store each layer's state with a manifest of file sizes and SHA-256 hashes. A damaged or mismatched
  checkpoint stops the run (`Integrity gate FAILED — fail closed, no files written`).
- Only one process can work on a run at a time. A second one gets `run is already in use`.
- ++ctrl+c++, or *Cancel* in the TUI, stops cleanly with exit code `130`, and the interruption is recorded in
  the run.

## Output

```text
runs/R/
├── calibration.json            # method, settings, per-layer losses, held-out results, peak VRAM, provenance
├── metrics.json                # status and headline losses
└── artifacts/
    ├── calibrated_snapshot/    # the result: a hard ternary fake-quant snapshot
    ├── checkpoint/             # resumable state (module_00000.pt, …)
    ├── calibration_state/      # learned parameters per layer
    ├── activation_cache*/      # captured teacher activations (reusable, can be large)
    └── threshold_metrics.jsonl
```

Evaluate `artifacts/calibrated_snapshot` exactly like a quantized snapshot:

```bash
openternary quality runs/R/artifacts/calibrated_snapshot --data quality.json --device cuda --output runs/R-quality
```

!!! warning "Disk space"

    Activation caches and checkpoints can be many times the model's size. Delete `activation_cache*` once a run
    is finished and you no longer need to resume or reuse it.
