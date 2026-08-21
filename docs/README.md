# OpenTernary

> Open-source tooling for researching, calibrating, benchmarking, and eventually deploying ternary LLMs.

**OpenTernary** is an experimental CLI-first project for exploring how existing open-weight language models can be converted toward ternary weights such as `{-1, 0, +1}` while retaining as much model quality as possible.

The project begins with **Gemma 4 E2B** as the first research target and focuses on building a reproducible quantization pipeline before attempting a GUI or large-model scaling.

> [!WARNING]
> OpenTernary is a research project. Early outputs may be slow, inaccurate, incompatible with common runtimes, or significantly worse than the original model.

---

## Quickstart (uv) — Phase 0

**推奨: uv で再現可能な環境を構築**

```bash
git clone <repo>
cd OpenTernary
uv sync
uv run openternary --help
uv run --frozen pytest
```

開発時チェック:

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy src/openternary
```

Fallback (pip):

```bash
pip install -e .
openternary --help
pytest
```

- Python 3.12 を推奨（`uv python pin 3.12` で固定、`requires-python >=3.11`）
- `uv.lock` + `.python-version` + `config.yaml` + `environment.json` で再現性を担保

詳細は `docs/ENVIRONMENT.md` を参照。

---

## Why OpenTernary?

Current low-bit LLM tooling is powerful, but the path from:

```text
open-weight model
    ↓
weight analysis
    ↓
ternary conversion
    ↓
calibration / reconstruction
    ↓
benchmark
    ↓
packed deployment
```

is still difficult to explore as a single, reproducible workflow.

OpenTernary aims to make that workflow inspectable and repeatable.

The first goal is **not** to immediately create the smallest possible model.

The first goal is:

> **Make a ternary Gemma 4 E2B that can be generated, loaded, benchmarked, and iterated on reproducibly.**

---

## Project Principles

1. **CLI first, GUI later**
   - Research interfaces change quickly.
   - Stabilize the quantization pipeline before building a desktop/web GUI.

2. **Measure before optimizing**
   - Every conversion must be compared against a fixed baseline.

3. **Quality per byte, not bit-count vanity**
   - Hybrid precision is allowed when it preserves substantially more capability.

4. **Reproducibility**
   - Every experiment should record config, model revision, code commit, hardware, dataset split, and metrics.

5. **Model adapters, not model-specific spaghetti**
   - Gemma 4 is the first target, not the only possible target.

6. **Research claims require evidence**
   - Do not claim CAT-Q, ScaleQ, Bonsai, BitNet, or any paper/framework reproduction unless the implementation has actually been verified against its source.

---

## Initial Scope

### First target

- Gemma 4 E2B instruction model
- Text path first
- Transformer linear layers first
- Embeddings / PLE / LM head / multimodal components handled separately

### First quantization target

Ternary weights:

```text
{-1, 0, +1}
```

with group-wise scaling.

### Initial development order

```text
Inspect
  ↓
Baseline benchmark
  ↓
Naive ternary fake quant
  ↓
Compare
  ↓
Calibration / reconstruction
  ↓
Japanese-focused evaluation
  ↓
Packed export
  ↓
Scale to larger Gemma 4 variants
  ↓
GUI
```

---

## Planned CLI

The following commands describe the intended UX. They are **not guaranteed to exist yet** (Phase 0 はスタブ).

```bash
openternary inspect MODEL
openternary benchmark MODEL --suite baseline
openternary quantize MODEL --method naive --group-size 128
openternary compare BASELINE QUANTIZED
openternary calibrate MODEL --config configs/gemma4-e2b.yaml
openternary export MODEL --format fake-quant
openternary export MODEL --format gguf
```

Example target flow:

```bash
openternary inspect google/gemma-4-E2B-it-qat-q4_0-unquantized

openternary benchmark google/gemma-4-E2B-it-qat-q4_0-unquantized \
  --output runs/e2b-baseline

openternary quantize google/gemma-4-E2B-it-qat-q4_0-unquantized \
  --method naive \
  --group-size 128 \
  --output runs/e2b-naive

openternary benchmark runs/e2b-naive \
  --output runs/e2b-naive-bench

openternary compare \
  runs/e2b-baseline \
  runs/e2b-naive-bench
```

Phase 0 では未実装コマンドは `--dry-run` で計画表示（exit 0）、通常実行は `Not implemented in Phase 0` で exit 1。

---

## Research Stages

### Stage A — Naive PTQ

Purpose: prove the pipeline.

- Load model
- Find quantizable tensors
- Group weights
- Compute scale
- Map weights to ternary values
- Dequantize for fake-quant inference
- Run benchmark
- Save experiment metadata

Performance may be poor. That is acceptable.

### Stage B — Calibration / Reconstruction

Purpose: preserve the original model's internal behavior.

Potential research directions include:

- learnable scale
- learnable threshold
- softened ternarization
- layer/window reconstruction
- teacher/student activation matching
- sensitivity-aware mixed precision

These methods must be implemented and validated incrementally.

### Stage C — Deployment

Purpose: turn research weights into something useful.

- packed ternary representation
- runtime integration
- GGUF or other practical export
- RAM / VRAM / throughput measurement

---

## Repository Structure

```text
OpenTernary/
├─ src/openternary/
│  ├─ cli/
│  ├─ config/
│  ├─ experiment/
│  ├─ utils/
│  ├─ adapters/
│  ├─ quant/
│  ├─ calibration/
│  ├─ benchmark/
│  └─ export/
├─ configs/
├─ docs/
├─ tests/
├─ pyproject.toml
├─ uv.lock
├─ .python-version
├─ README.md
├─ ROADMAP.md
├─ PROJECT_SPEC.md
├─ ARCHITECTURE.md
├─ RESEARCH_PLAN.md
├─ BENCHMARKS.md
├─ AGENTS.md
├─ EXPERIMENT_LOG.md
└─ CONTRIBUTING.md
```

---

## AI Development Workflow

OpenTernary is initially developed with:

- **DSH — lead development/research agent**
- **Codex — implementation/review sub-agent**

The exact division of responsibilities is defined in [`AGENTS.md`](./AGENTS.md).

High-level rule:

```text
DSH
Plan / Research / Integrate
        ↓
Codex
Focused implementation / tests / review
        ↓
DSH
Merge / benchmark / next experiment
```

No agent should silently modify the experimental target, benchmark protocol, or model scope.

---

## Definition of the First MVP

The first MVP is complete when all of the following work:

- [ ] Gemma 4 E2B can be inspected
- [ ] Quantizable tensors are listed correctly
- [ ] Baseline inference works
- [ ] Baseline benchmark can be reproduced
- [ ] Naive group-wise ternary fake quantization works
- [ ] Quantized model can run inference without NaN/crash
- [ ] Baseline vs ternary metrics can be compared
- [ ] Run configuration is saved automatically
- [ ] At least one Japanese evaluation set is held out from calibration
- [ ] Tests cover core ternary math and tensor mapping

The MVP does **not** require:

- competitive quality
- Bonsai-level results
- complete CAT-Q reproduction
- GGUF export
- GUI
- MoE support

---

## Long-Term Vision

OpenTernary should eventually become a model-agnostic research and conversion toolkit:

```text
Gemma
Qwen
Llama
DeepSeek-style models
Other open-weight LLMs
        ↓
OpenTernary
        ↓
Inspect
Quantize
Calibrate
Benchmark
Export
```

A later GUI can make these workflows accessible without hiding the underlying experiment configuration.

---

## Status

**Phase 3 — Model-Level Ternarization (bounded sharded fake-quant): CLOSED (2026-08-21)**

* Phase 0 (Foundation) ✅, Phase 1a (Inspection) ✅, Phase 1b (BF16 Baseline v1) ✅, Phase 2 (AbsMean + 2bit-v1) ✅, **Phase 3 (LD-RW per_group + bounded sharded fake-quant) ✅**
* Per-tensor `scale=mean(abs(W))` and per-group LD-RW `last-dim-rowwise-v1` vectorized, zero-branch, `G>=D` → `rows` groups, `00=-1/01=0/10=+1` packed `ceil(N/4)` per-tensor sum, `ideal_ternary_bits≈1.585` vs `packed` vs `scale_overhead` (per_tensor `205*32` vs per_group `14.3M*32`) separated
* Bounded sharded writer `512 MiB` flush-before-add, over-size single allowed, HF `blobs/` symlink dereferenced, content fingerprint MUST vs file hash SHOULD, JSON stores `scale_stats` + `scale_fingerprint` (no scale array)
* Whole-model `e2b-naive-pt` (12 shards, 10.2GB) and `e2b-naive-g128-rw` (14.3M groups, 12 shards) both loadable via `AutoModelForMultimodalLM` and smoke `protocol True` / `result observational different` PASS
* See `ROADMAP.md` for M1–M13 and `EXPERIMENT_LOG.md` for whole-model validation

Current focus:

> Phase 4 — Calibration / reconstruction (learnable scales) — `inspect→quantize→benchmark→compare` pipeline now stable

---

## License

License is **TBD before public release**.

The project must not copy code from external implementations until their licenses and attribution requirements have been checked.
