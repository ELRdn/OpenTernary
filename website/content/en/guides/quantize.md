---
description: "Quantize a model to a ternary fake-quant snapshot with per-tensor or per-group scales, layer exclusions, passes, or the TorchAO backend."
---

# Quantize

`quantize` converts a model's linear layers to ternary and writes a **fake-quant snapshot**: a normal Hugging Face
model folder whose quantized weights take only three values per scale. You can load, benchmark, evaluate and
export it. [Ternary quantization](../concepts/ternary-quantization.md) explains what happens to the weights.

## The basic command

```bash
openternary quantize google/gemma-4-E2B-it-qat-q4_0-unquantized \
    --scale-granularity per_group --group-size 128 \
    --output runs/e2b-g128
```

The result is in `runs/e2b-g128/artifacts/snapshot`, next to the run's records (`config.yaml`,
`quantization.json`, …).

Always preview first. `--dry-run` resolves and checks everything and creates nothing:

```bash
openternary quantize google/gemma-4-E2B-it-qat-q4_0-unquantized --scale-granularity per_group --group-size 128 --dry-run
```

## Choosing scales

| Option | Default | Effect |
|---|---|---|
| `--scale-granularity per_tensor` | ✔ | one scale per tensor: smallest overhead, lowest fidelity |
| `--scale-granularity per_group` | | one scale per `--group-size` values along each row |
| `--group-size N` | 128 | values per group (with `per_group`); smaller groups are more faithful but store more scales |
| `--grouping-scheme last-dim-rowwise-v1` | ✔ | the only grouping scheme; see [grouping](../concepts/ternary-quantization.md#per-tensor-or-per-group-scales) |

Per-group G128 is the configuration used in OpenTernary's research. It costs 0.25 extra bits per weight for the
FP32 scales.

## Using a config file

Anything you can pass as a flag can also go in a YAML file, together with options that have no flag:

```yaml title="my-quant.yaml"
model:
  id: google/gemma-4-E2B-it-qat-q4_0-unquantized
  revision: 6befbaca7398925921802abd1f277b495b78b738
quantization:
  backend: ternary
  scheme: absmean
  scale_granularity: per_group
  group_size: 128
  target:
    attention: true
    mlp: true
  mixed_precision:                       # keep these tensors in full precision
    model.language_model.layers.0.self_attn.q_proj.weight: preserve
  passes:                                # applied in order before quantizing
    - name: clip
      options: {max_abs: 2.0}
device: cpu
```

```bash
openternary quantize --config my-quant.yaml --output runs/e2b-g128-keep-l0q
```

Flags override the file, and the file overrides defaults. The resolved result is saved as the run's
`config.yaml`. A model given on the command line replaces `model.id`, and a new model ID never inherits another
model's pinned revision. Every key is described in the [configuration reference](../reference/configuration.md).

!!! tip "Finding tensor names"

    `mixed_precision` keys must be exact tensor names. Get them from `inspection.json`
    ([Inspect and plan](inspect-and-plan.md#inspect-what-is-in-the-model)). A name that matches nothing is an error.

## Device and dtype

Conversion of a snapshot **always runs on the CPU**, tensor by tensor:

- `--device cpu` or `auto` (the default) are accepted. `--device cuda` is refused with exit code `3` instead of
  silently using the CPU.
- The snapshot keeps the source model's floating-point dtype (a BF16 model gives a BF16 snapshot). `--dtype` is
  therefore refused for `quantize` with exit code `2`. The dtype used for *running* the model is chosen when you
  benchmark or evaluate.

Memory use is bounded: roughly the largest single tensor plus one output shard (at most 512 MiB), not the whole
model. Disk use is about the size of the original model for the snapshot.

## The TorchAO backend

For comparison with an established 8-bit method, with `torchao==0.18.0` installed:

```bash
openternary quantize ./my-model --backend torchao --scheme int8-weight-only --weight-dtype int8 --output runs/my-int8
```

TorchAO artifacts can only be reloaded with the same TorchAO version, and cannot be exported to safetensors.

## What the run records

`quantization.json` has, among other things:

- the configuration that ran and the adapter that classified the tensors;
- per-tensor details: shape, dtype, scale statistics, number of groups, and a fingerprint of the scales (the full
  scale arrays are not stored in JSON);
- the **content fingerprint** of the snapshot, a hash over tensor names, shapes, dtypes and bytes;
- bit accounting: ideal, packed and scale overhead, kept separate (see
  [counting bits](../concepts/ternary-quantization.md#counting-bits-honestly)).

The snapshot itself contains `openternary-manifest.json` with the SHA-256 of every file, so you can check it at
any time with `openternary artifacts validate`.

## After quantizing

- Measure it: [Evaluate and compare](evaluate-and-compare.md). Expect a large quality drop from plain ternary
  conversion.
- Try to recover quality: [Calibrate](calibrate.md).
- See the real packed size: [Export](export.md).

## Migrating from older versions

| Old | Now |
|---|---|
| `quantize --method naive` | `--backend ternary --scheme absmean` (YAML `method: naive` still works) |
| `quantize --dtype bf16` | refused; the snapshot keeps the source dtype |
