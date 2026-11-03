---
description: "Export artifacts as safetensors, 2-bit ternary-packed, TorchAO state, or GGUF through an explicitly chosen llama.cpp converter."
---

# Export

`export` writes an artifact in another format. **The source is never modified**, and an existing destination is
never overwritten.

```bash
openternary export <run-or-artifact> --format <format> --output <new folder>
```

The source can be a run directory (its `artifacts/snapshot` or calibrated snapshot is used) or an artifact folder
such as a previous export.

## Formats

| `--format` | What it writes | Limits |
|---|---|---|
| `safetensors` (default) | a plain snapshot copy, or a **restore** from `ternary-packed` | fake-quant weights are stored as floating point |
| `ternary-packed` | 2-bit codes, scales, shapes, padding, and the tensors that are not quantized | only inputs that can be reconstructed *exactly* from codes and scales; nothing can run it directly |
| `torchao` | TorchAO's own state, copied with its reload entry point | needs the same TorchAO version (0.18.0) to reload; cannot be converted to safetensors |
| `gguf` | a floating-point GGUF, made by a llama.cpp converter you choose | Llama, Mistral and Qwen2 metadata only; `f16`/`bf16`/`f32` output; see below |

## Packing and restoring

```bash
openternary export runs/e2b-g128 --format ternary-packed --output runs/e2b-g128-packed
openternary export runs/e2b-g128-packed --format safetensors --output runs/e2b-g128-restored
openternary artifacts validate runs/e2b-g128-restored --level tensors
```

The packed folder holds `packed.json` (the layout), `tensors/` (one file per tensor: 2-bit codes and scales for
quantized tensors, the original data for the rest), the model's config and tokenizer files, and the manifest.
[The 2-bit format](../concepts/ternary-quantization.md#the-2-bit-packing-format) describes the encoding.

The round trip is **lossless**: every restored tensor has the same values, dtype, shape and bytes as before
packing. This has been checked on CPU fixtures and in [the first run](../getting-started/first-run.md#4-pack-restore-and-validate).
The actual file size includes scales, padding and the unquantized tensors, and is reported separately from the
theoretical size.

Packing is refused, rather than approximated, when a snapshot cannot be reconstructed exactly from codes and
scales. That is the case, for example, for some calibrated snapshots whose thresholds would need information the
packed format does not store.

## GGUF through llama.cpp

OpenTernary does not include or download a GGUF converter. You point it at llama.cpp's `convert_hf_to_gguf.py`
on your machine:

```bash
openternary export runs/my-llama-restored --format gguf \
    --converter ~/src/llama.cpp/convert_hf_to_gguf.py \
    --converter-python ~/src/llama.cpp/.venv/bin/python \
    --output-dtype f16 \
    --output runs/my-llama-gguf
```

| Option | Default | |
|---|---|---|
| `--converter PATH` | required | the converter script; its SHA-256 is recorded in the manifest |
| `--converter-python PATH` | the Python running OpenTernary | an isolated environment with the converter's own dependencies |
| `--output-dtype` | `f16` | `f16`, `bf16` or `f32`; ternary GGUF types are not produced |
| `--timeout-seconds` | 3600 | the converter is stopped after this long; it and its children are cleaned up |

- The source must be safetensors with `llama`, `mistral` or `qwen2` architecture metadata. **The current Gemma
  research snapshots are not supported** by this bridge.
- A regular export checks the GGUF header and file hashes. Running the model is a separate check. The repository's
  [`scripts/validate_cli_gguf.py`](https://github.com/ELRdn/OpenTernary/blob/main/scripts/validate_cli_gguf.py)
  does that with a pinned llama.cpp: real reader, CPU generation.
- Verified so far: a synthetic Llama in FP32 through a pinned llama.cpp, read back and generating on the CPU.
  Other architectures, precisions and GPU execution are unverified.

## Atomic writes and manifests

Every export:

1. writes into a unique temporary folder next to the destination,
2. writes `openternary-manifest.json`: schema, format, the SHA-256 of every file, the source and its interface,
   the configuration, backend, adapter, component, passes and code hash,
3. validates the result,
4. and only then renames it to `--output`.

If anything fails, the destination does not appear. The manifest carries the conversion history forward, so a
restored snapshot still records that it came from a packed artifact, which came from a quantization run.

```bash
openternary artifacts validate runs/e2b-g128-packed            # manifest + every file hash
openternary artifacts validate runs/e2b-g128-packed --level tensors   # also open and check every tensor
```

## Migrating from older versions

| Old | Now |
|---|---|
| `export --format fake-quant` | `--format safetensors` |
| `export --format gguf` with no converter | an explicit `--converter` and a supported architecture are required |
