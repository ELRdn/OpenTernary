# OpenTernary Project Specification

## 1. Product Definition

OpenTernary is an open-source research toolkit for converting and studying open-weight LLMs under ternary weight constraints.

The software should make it possible to:

1. inspect a supported model,
2. identify quantizable components,
3. apply ternary quantization,
4. optionally calibrate/reconstruct the quantized model,
5. benchmark quality and efficiency,
6. export experimental or deployable artifacts.

---

## 2. Primary User

Initial primary user:

- local-LLM enthusiast
- ML/LLM quantization researcher
- OSS developer experimenting with low-bit inference

The initial project is not designed as a one-click consumer application.

---

## 3. Initial Target

### Model

Gemma 4 E2B instruction model.

### Modality

Text path first.

### Weight target

Ternary:

```text
-1
 0
+1
```

with group-wise scaling.

---

## 4. Goals

### G1 — Reproducible experiments

A run must be reconstructable from stored metadata.

### G2 — Modular quantization

Quantization method and model architecture must be separable.

### G3 — Honest evaluation

Quality losses must be visible.

### G4 — Japanese evaluation

Japanese quality is a first-class metric, not an anecdotal afterthought.

### G5 — CLI stability

GUI work begins only after core commands stabilize.

### G6 — Scale later

E2B is a proving ground for tooling and research methodology.

---

## 5. Non-Goals for Initial Releases

OpenTernary v0.x does not promise:

- state-of-the-art ternary quality
- exact Bonsai reproduction
- exact CAT-Q/ScaleQ reproduction
- training an LLM from scratch
- universal architecture support
- production inference kernels
- GUI
- mobile deployment
- MoE support
- perfect backward compatibility

---

## 6. Core Functional Requirements

### FR-01 Inspect

The tool shall report:

- model architecture
- module names/types
- parameter counts
- dtypes
- quantizable candidates
- excluded components

### FR-02 Quantize

The tool shall support a naive group-wise ternary method.

Inputs:

- model
- target modules
- group size
- threshold strategy
- scale strategy
- output path

Outputs:

- transformed model or fake-quant wrapper
- configuration
- tensor-level statistics

### FR-03 Benchmark

The tool shall benchmark:

- baseline model
- quantized model

and save machine-readable results.

### FR-04 Compare

The tool shall compare multiple experiment runs.

### FR-05 Calibrate

The tool shall support a calibration/reconstruction loop without requiring full pretraining.

### FR-06 Reproducibility

Each run shall save at minimum:

- timestamp
- git commit
- configuration
- model id/revision
- random seed
- hardware
- software versions
- dataset id/split
- result metrics

### FR-07 Export

Later versions shall support packed storage and runtime-specific formats.

---

## 7. Quality Requirements

### QR-01 Testability

Core ternary math must be unit tested independently of full models.

### QR-02 Failure visibility

NaN, overflow, unsupported modules, OOM, missing tensors, and incompatible model versions should fail loudly.

### QR-03 No silent fallback

If a requested module cannot be quantized, the tool must report it.

### QR-04 Determinism

Where practical, repeated runs with the same configuration should produce equivalent outputs.

### QR-05 Experiment isolation

Calibration data and held-out benchmark data must remain separate.

---

## 8. CLI Contract

Planned command groups:

```text
openternary inspect
openternary benchmark
openternary quantize
openternary calibrate
openternary compare
openternary export
```

Every modifying command should support:

```text
--config
--output
--seed
--device
--dtype
--dry-run
```

where technically applicable.

---

## 9. Configuration

Preferred format:

```yaml
model:
  id: google/gemma-4-e2b-it
  revision: null

quantization:
  method: naive
  codebook: [-1, 0, 1]
  group_size: 128
  target:
    attention: true
    mlp: true

calibration:
  enabled: false

benchmark:
  suites:
    - smoke
    - japanese
```

All CLI flags should ultimately map into the same internal configuration object.

---

## 10. Output Convention

Example:

```text
runs/
└─ 2026-08-20_e2b_naive_g128/
   ├─ config.yaml
   ├─ environment.json
   ├─ model.json
   ├─ quantization.json
   ├─ metrics.json
   ├─ logs/
   └─ artifacts/
```

---

## 11. Success Criteria for Gemma 4 E2B MVP

The MVP is successful if:

1. baseline inference runs,
2. target modules are identified,
3. naive ternary conversion completes,
4. converted model runs,
5. benchmark results are stored,
6. baseline and converted results can be compared,
7. the entire process is repeatable from a config file.

No quality threshold is required for the first MVP.
