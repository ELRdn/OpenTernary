---
description: "Get a model onto your disk, list its quantizable tensors with inspect, and preview a quantization with plan, all without loading weights."
---

# Inspect and plan

Before you quantize a model, find out what OpenTernary will do to it. Both commands in this guide read only file
headers: no PyTorch, no GPU and no downloads are needed.

## Getting a model

OpenTernary **never downloads models itself.** You give it either:

- a **folder** containing `config.json` and `*.safetensors` files (a Hugging Face snapshot, or the output of a
  previous OpenTernary command), or
- a **Hugging Face model ID**, which is looked up in your local Hugging Face cache only.

To put a model into the cache, use Hugging Face's own CLI. It is part of `huggingface_hub`:

```bash
uvx --from huggingface_hub hf auth login     # once; needed for gated models such as Gemma
uvx --from huggingface_hub hf download google/gemma-4-E2B-it-qat-q4_0-unquantized \
    --revision 6befbaca7398925921802abd1f277b495b78b738
```

Gemma models are gated: accept the licence on the model's Hugging Face page first. If a model is not in the
cache, OpenTernary stops with a message that tells you the exact `hf download` command to run.

!!! info "Pin the revision"

    A Hugging Face model can change after you download it. Always record and pass a revision (commit hash), as
    in the `--revision` above and in `model.revision` in a config file. The run's `model.json` records it, so
    comparisons can check that they used the same weights.

## `inspect`: what is in the model

```bash
openternary inspect google/gemma-4-E2B-it-qat-q4_0-unquantized
openternary inspect ./my-model --output runs/inspect-my-model
```

`inspect` reads the safetensors headers and `config.json`, then:

- picks an **adapter** (Gemma 4, generic Transformers, or Diffusers) for the architecture;
- lists every tensor with its shape, dtype and role, and whether it is **quantizable**;
- counts parameters and estimates sizes in BF16 and FP32;
- computes a fingerprint of the model's tensor inventory.

It writes `inspection.json` in a new run directory and prints a summary (see
[the first run](../getting-started/first-run.md#2-inspect-and-plan) for an example). The full tensor list is in
`inspection.json`, which is where to find exact names for `mixed_precision`.

Each entry in its `tensors` array has `name`, `shape`, `dtype`, `param_count`, `role`, `quantizable` and, for
tensors that stay in full precision, an `exclude_reason`:

```json
{"name": "lm_head.weight", "shape": [99, 128], "dtype": "F32", "param_count": 12672,
 "role": "lm_head", "quantizable": false,
 "exclude_reason": "high precision policy: lm_head (tie_word_embeddings)"}
```

```bash
# the names of the quantizable tensors
jq -r '.tensors[] | select(.quantizable) | .name' runs/<run>/inspection.json
```

`--load-weights` reads the actual weight values as well and adds distribution statistics. That needs PyTorch and
reads the whole model, one tensor at a time.

## `plan`: what quantize would do

```bash
openternary plan google/gemma-4-E2B-it-qat-q4_0-unquantized --backend ternary --scheme absmean
openternary plan ./my-model --config my-quant.yaml --json
```

`plan` resolves the full configuration (CLI flags over YAML over defaults), picks the adapter and backend, and
lists the target tensors. It creates **no run directory** and writes nothing. Anything it cannot know offline is
reported as unknown: whether the hardware can run it, whether the artifact will reload, and the model's quality.

`plan` takes the backend options (`--backend`, `--scheme`, `--weight-dtype`, `--component`) and `--config`.
For per-group settings, put them in the config file or preview with `quantize --dry-run`:

```bash
openternary quantize ./my-model --scale-granularity per_group --group-size 128 --dry-run
```

## Checking a config file before a long run

Every command that takes `--config` validates it completely before doing anything. Unknown keys, non-finite
numbers, unsupported codebooks and contradictory calibration settings are all errors (exit code `2`), never
silently ignored. The quickest way to check a config is a dry run:

```bash
openternary calibrate --config my-calibration.yaml --dry-run
```

It prints the resolved configuration as YAML and creates nothing. The plan screen in the
[TUI](../getting-started/tui.md#starting-a-job-form-plan-start) shows the same output before you can start a job.

## Next

[Quantize](quantize.md) the model.
