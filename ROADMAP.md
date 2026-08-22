# OpenTernary Roadmap

This roadmap prioritizes **research validity and a stable CLI pipeline** over GUI work.

> **Progress snapshot — 2026-08-21 (Phase 4.1 CLOSED)**
> - **Phase 0 — Repository Foundation: ✅ COMPLETE**
> - **Phase 1a — Gemma 4 E2B Inspection: ✅ COMPLETE** — 1951 tensors / 5,104,298,467 params / 205 quantizable (0.359605) / BF16 9.5075 GB / text 35 layers hidden 1536 vocab 262144, fingerprint `sha256:07fe44eae504218937187b75826a4c780cba2f76a9145fdc38f7efdd4d3a7b01` stable.
> - **Phase 1b — BF16 Smoke Inference Baseline v1: ✅ COMPLETE** — `AutoProcessor + AutoModelForMultimodalLM`, `bf16 exact`, `thinking=False`, `do_sample=False/num_beams=1`, suite `smoke v1` 5 prompts, warmup→5 timed, 2-run PASS: protocol `sha256:11ac43e68f9bc2079365b224078f4af93dd767c813a0d3d82485c0096507d52d`一致, result `sha256:907fbb73f91d0e97033c59cc1dae65817a9b507d8df4e3b65125a87b9327cfe8`一致.
> - **Phase 1 Gate: ✅ PASS** — Behavioral baseline frozen, not Quality (Phase 5).
> - **Phase 2 — Ternary Core & Packing Accounting: ✅ CLOSED** — `src/quant/{ternary,packing,accounting,metrics}.py` canonical AbsMean, 2bit-v1, per-tensor sum, G1–G19 PASS.
> - **Phase 3 — Model-Level Ternarization (bounded sharded fake-quant): ✅ CLOSED** — LD-RW per_group vectorized zero-branch, bounded sharded writer (512 MiB, flush-before-add, over-size single), HF blobs symlink dereference, content fingerprint (MUST) vs file hash (SHOULD), scale summary+fingerprint (no JSON scale array), `quantize`/`compare` CLI real, whole-model `e2b-naive-pt` (12 shards, 205 tensors, 14.3M groups for G128) + `e2b-naive-g128-rw` both loadable and smoke PASS, protocol match True, result observational, M1–M13 PASS.
> - **Phase 4.1 — Calibration recon-scale: ✅ CLOSED** — layer-local `codes*scale→F.linear` differentiable, Teacher activation disk-backed sharded, `inverse_softplus`初期化で `effective(step0)==orig`, `per_tensor/per_group` dequantize helper再利用, `steps=full sweep`定義, `sample hash isdisjoint`汚染証明, `calibrate` CLI real (dry-run/filesystemなし, checkpoint/resume), tiny fixture 4 samples/5 stepsで `best<initial`かつ `final<initial`かつ `heldout`改善、G4-1〜12 PASS.

---

# Phase 0 — Repository Foundation

**Goal:** Make the project safe to iterate with multiple AI coding agents.

**Status: ✅ Complete (2026-08-20 verified)**

## Deliverables

- [x] Python package skeleton (`src/openternary/`, `pyproject.toml` hatchling)
- [x] CLI entry point (`openternary` via `typer`, `src/openternary/cli/main.py`)
- [x] configuration system (`src/openternary/config/schema.py`, `loader.py`, `configs/gemma4-e2b.yaml`, `CLI_TO_CONFIG`)
- [x] structured logging (`src/openternary/utils/logging.py`, per-run `logs/`)
- [x] run directory convention (`src/openternary/experiment/run.py`, `runs/YYYY-MM-DD_<slug>/`)
- [x] deterministic seed handling (`src/openternary/utils/seed.py`)
- [x] unit-test framework (`tests/`, `pytest`)
- [x] lint / format / type-check setup (`ruff`, `mypy` — `pyproject.toml` + CI)
- [x] CI (`.github/workflows/ci.yml` — `uv sync --locked` / ruff / mypy / pytest + pip fallback)
- [x] documentation baseline (`README.md`, `ARCHITECTURE.md`, `PROJECT_SPEC.md`, `docs/ENVIRONMENT.md`, etc.)
- [x] model/download cache policy (`src/openternary/utils/hf_cache.py` cache-only + `safetensors_header.py` raw header, `HF_HOME`/`UV_CACHE_DIR`)
- [x] experiment metadata schema (`src/openternary/experiment/metadata.py`, `environment.json`/`config.yaml`/`model.json`/`metrics.json`)

## Gate

A new developer/agent can clone the repo, install dependencies, run tests, and execute:

```bash
openternary --help
```

without knowing internal implementation details.

**Result: PASS** — `uv sync --locked` → `uv run --frozen pytest -q` (9 tests) / `ruff check` / `mypy` all green.

---

# Phase 1 — Gemma 4 E2B Inspection + Baseline

**Goal:** Understand the model before changing it.

**Status: ✅ COMPLETE (Phase 1a + 1b closed 2026-08-21)**

## Phase 1a — Inspection

**Status: ✅ COMPLETE**

- [x] Gemma 4 adapter (`src/openternary/adapters/gemma4.py`, `base.py` — `Gemma4ForConditionalGeneration` / `model_type gemma4`)
- [x] tensor/module inventory (`src/openternary/inspect/engine.py` — `_collect_tensors` sorted, raw `struct "<Q" + json` header-only, no `torch` required)
- [x] quantizable-module classifier (`adapters/base.py` `_QUANTIZABLE_ATTENTION_RE` / `_QUANTIZABLE_MLP_RE`, `default_quantizable_role`)
- [x] excluded-module classifier (`_EXCLUDE_*` — `embed_tokens`, `lm_head`, `norm`, `PLE`, `vision_tower.*`, `audio_tower.*`)
- [x] parameter-count report (`inspection.json` `summary.total_params` / `quantizable_params` + per-tensor `param_count`)
- [x] dtype report (`dtype_report.by_dtype`, source `safetensors_header`)
- [x] memory estimate (`memory_estimate.bf16_GB` 9.5075 / `fp32_GB` 19.01 — `ternary: null` until Phase 2 packing defined)

Evidence (real checkpoint):
- `google/gemma-4-E2B-it-qat-q4_0-unquantized@6befbaca7398925921802abd1f277b495b78b738`, local `.../snapshots/6befbaca...`, `config.json` text 35 layers / hidden 1536 / vocab 262144, `model.safetensors` 10.2 GB
- `tensors 1951 / total_params 5,104,298,467 / quantizable 205 / ratio 0.359605 / BF16 9.5075 GB`
- `openternary inspect` → `exit 0`, fingerprint `sha256:07fe44eae504218937187b75826a4c780cba2f76a9145fdc38f7efdd4d3a7b01` stable (deterministic payload, provenance excluded)
- `inspect --load-weights` via `safe_open(..., framework="pt")` when `ml` extra installed

## Phase 1b — BF16 Smoke Inference Baseline v1

**Status: ✅ COMPLETE (2-run manual acceptance 2026-08-21)**

- [x] baseline inference command (`src/openternary/benchmark/runner.py` — `AutoProcessor` + `AutoModelForMultimodalLM`, `dtype=bf16 exact`, device `cpu`/`cuda`, greedy `do_sample=False/num_beams=1`, `thinking=False`)
- [x] baseline benchmark command (`src/openternary/cli/main.py benchmark` — real implementation, `--suite/--thinking/--device/--dtype` + `ml` dependency loud error `exit 2` + failure `metrics.json` writing)
- [x] baseline run artifact (`runs/baseline-smoke-cpu-{1,2}/` with `benchmark.json`/`metrics.json`/`config.yaml`/`environment.json`/`model.json`/`logs/`)

Evidence (real CPU runs, Manual Acceptance):
- Run 1: `runs/baseline-smoke-cpu-1/benchmark.json` — suite `smoke v1`, thinking `False`, dtype `bf16→torch.bfloat16`, 5 prompts, 160 total output tokens, avg 6636.47 ms, 4.82 tok/s, protocol `sha256:11ac43e68f9bc2079365b224078f4af93dd767c813a0d3d82485c0096507d52d` (FULL), result `sha256:907fbb73f91d0e97033c59cc1dae65817a9b507d8df4e3b65125a87b9327cfe8` (FULL)
- Run 2: `runs/baseline-smoke-cpu-2/benchmark.json` — same 160 tokens, avg 6237.82 ms, 5.13 tok/s, **same protocol + same result fingerprint** (latency variance only → PASS)
- Loader: `AutoProcessor`/`AutoModelForMultimodalLM` + `apply_chat_template(enable_thinking=False)` + `processor.parse_response(generated_ids, prefix=input_ids)` + warmup 1 → 5 timed
- Device: `device=auto` now resolves to `{"": "cpu"}` on CPU-only (no disk offload), `device=cpu` explicit also works
- Failure visibility: `ImportError` (torch/PIL/torchvision missing) or `device=auto` disk-offload now writes `metrics.json {status: failed, error_type, error_message}` instead of `not_run`

Meaning: **BF16 Smoke Inference Baseline v1** — behavioral/inference baseline, NOT a Quality benchmark. `result_fingerprint` change in Phase 2/3 means “deterministic output changed vs BF16 baseline”; quality degradation (accuracy/perplexity/Japanese) is Phase 5.

## Initial policy

Quantize first:

- attention linear projections
- MLP linear projections

Keep high precision initially:

- normalization
- embeddings
- PLE
- LM head
- multimodal encoders

## Gate

The baseline is reproducible across two runs within expected numerical variance.

**Result: ✅ PASS** — Inspection fingerprint stable + Benchmark protocol/result fingerprint stable across 2 real CPU BF16 runs (160 tokens, same output). See `EXPERIMENT_LOG.md` “Phase 1 — BF16 Smoke Inference Baseline v1” for full fingerprints.

---

# Phase 2 — Ternary Core & Packing Accounting

**Goal:** Establish mathematically correct PTQ foundation — `W → ternary → packed → accounting → roundtrip` before whole-model mutation.

**Status: ✅ CLOSED (2026-08-21) — G1–G19 PASS**

## Deliverables

- [x] AbsMean ternary quantizer `src/openternary/quant/ternary.py` — FP32 accumulation, zero-branch, device-preserving, `{-1,0,+1}` only, `round` ties-to-even, loud reject empty/non-float/NaN/Inf
- [x] dequantization `dequantize()` — `Q*scale` device-preserving, `scale==0` → zeros
- [x] deterministic 2-bit packing `src/openternary/quant/packing.py` — `00=-1/01=0/10=+1/11=reserved`, `c0=bits7-6…c3=bits1-0`, tail `00`, `len==ceil(N/4)`, `unpack(pack(Q))==Q` bit-exact, vectorized 9M-scale
- [x] size accounting `src/openternary/quant/accounting.py` — stdlib-only, `ideal_ternary_bits=N*log2(3)` vs `packed_weight_bits=Σ ceil(N_i/4)*8` (per-tensor sum, NOT total ceil), `logical_2bit_bits`, `byte_padding_bits`, `scale_overhead_bits` FP32 32bit/tensor, `excluded_original_bits`, `actual_file_size_bytes=null`
- [x] tensor metrics `src/openternary/quant/metrics.py` — MAE/MSE/RMSE/maxAbs, cosine (zero→null), zero/neg/pos ratio, `weight_only` vs `effective` compression split
- [x] inspect integration `src/openternary/inspect/engine.py` — `memory_estimate.ternary` + `ternary_estimator{estimate,fingerprint}` separated, `inspection_fingerprint` unchanged (Phase 1 contract preserved), header-only still no torch via `accounting`
- [x] real Gemma single tensor validation `scripts/validate_single_tensor.py` → `runs/2026-08-20_*/phase2_single_tensor_validation.json` — `model.language_model.layers.0.mlp.gate_proj.weight` [6144,1536] BF16 9,437,184 params packed 2,359,296 bytes, scale 0.0191, codes 32.6/33.7/33.7, roundtrip bit-exact, metrics finite, determinism PASS
- [x] tests `tests/test_quant_*.py` + `test_inspect_ternary_estimate.py` + `test_quant_no_torch_leak.py` (subprocess) — all PASS, `Q∈{-1,0,1}` shape preserved, thresholds ±0.5*scale, reserved 11 reject, tail padding strict
- [ ] group-wise partitioning / layer replacement / full-model fake-quant inference — **deferred to Phase 3** (Phase 2 is representation foundation, not end-to-end pipeline)
- [ ] `compare` CLI — still stub (Phase 3)

Current code state (2026-08-21):

- `src/openternary/quant/` now has `ternary/packing/accounting/metrics` (+ lazy `__init__.py`); `grouping/scale/threshold` deferred.
- `src/openternary/benchmark/` unchanged (smoke v1). `quantize`/`compare`/`calibrate` remain stubs (by design).
- `runs/2026-08-20_gemma-4-E2B-it-qat-q4_0-unquantized-naive-g128-001/phase2_single_tensor_validation.json` is the Phase 2 manual acceptance artifact.

## Experiments

Phase 2 is not a quality experiment; it is a representation invariant check. Group-size sweeps (32/64/128/256) and sensitivity are Phase 3.

## Gate

Phase 2 gates G1–G19 (incl. G16 FP32 accumulation, G17 device-preserving, G18 no torch leak, G19 loud reject):

**Result: ✅ PASS — all 19 gates verified (pytest/ruff/mypy green, real tensor roundtrip, Phase 1 inspect/benchmark no regression).**

---

# Phase 3 — Model-Level Ternarization (bounded sharded fake-quant)

**Goal:** Whole-snapshot `HF → ternary fake-quant → sharded BF16 reload → smoke` を bounded-memory で成立させる。

**Status: ✅ CLOSED (2026-08-21) — M1–M13 PASS**

## Deliverables

- [x] Group-wise LD-RW `src/openternary/quant/grouping.py` — `last-dim-rowwise-v1`, vectorized, per-row tail, `G>=D` → `rows` groups, zero-branch, `scales` per-row interleaved, `num_groups_for_shape` helper
- [x] Bounded sharded writer `src/openternary/quant/fake_quant.py` — `MAX_SHARD_SIZE=512 MiB`, `flush-before-add`, single>maxは単独over-size許可, `model-00001-of-0000N.safetensors` + `model.safetensors.index.json`, atomic tmp→rename, HF `blobs/` symlinkはdereferenceして内容コピー（snapshot内 or `blobs/` のみ許可）、`tensor_payload/shard_file/snapshot_total`分離
- [x] Config `scale_granularity` (`per_tensor`|`per_group`, default `per_tensor`) + `grouping_scheme` (`last-dim-rowwise-v1`) — `group_size`からの推論なし、明示指定
- [x] Content fingerprint `sha256(canonical(sorted[{name,dtype,shape,sha256(raw)}]))` — MUST, file hashはSHOULD, `scale_fingerprint`は `float32 LE bytes` のsha256, JSONには `scale_stats {min,max,mean,std(correction=0)}` と `num_groups` のみ（全scales配列は禁止）
- [x] `quantize` CLI real — `resolve_snapshot` → `convert_snapshot` → `artifacts/snapshot` + `quantization.json` + `metrics.json`, dry-runはrun作成なし, `--dtype`はwarnして無視（original dtype preserve）
- [x] `compare` CLI real + `src/openternary/experiment/compare.py` — `protocol_match` は必須一致, `result_observation` は `same`/`different` 観測でpass/failにしない, per-prompt token diff表示
- [x] Whole-model manual acceptance — `runs/e2b-naive-pt` per_tensor (205 tensors, 12 shards, 10.2GB payload) と `runs/e2b-naive-g128-rw` per_group G128 (14.3M groups, 12 shards) 両方生成 → `AutoModelForMultimodalLM` load → `smoke v1` 5 prompts warmup PASS, protocol `11ac43e...` 一致, resultは `499d390...` / `66e5977...` とbaseline `907fbb7...` からdifferent（観測）、`compare`で全prompt False（different）を確認
- [x] Tests `test_quant_grouping` (7), `test_quant_fake_quant` (4, mock sharded, symlink, flush-before-add), `test_cli_quantize` (3), `test_cli_compare` (2) — 計122 tests PASS

## Experiments

Per-tensor vs per_group G128 RW の2 runで比較:

- `packed_estimate` は `scale_overhead` が `per_tensor 205*32` vs `per_group 14.3M*32` で数百倍差、 `packed_weight_bits` は同一
- `fake_quant_artifact` はどちらも `tensor_payload 10.2GB` / `shard_file 10.2GB`（BF16 fakeなのでpackedより大きいのが正常）
- Smokeは両方 `protocol True` / `result different` / `output finite`

Group-size sweep (32/64/256) とMLP/Attention別はPhase 4以降のsensitivityへ deferred。

## Gate (M1–M13)

```
M1  Phase2 per-tensor invariants unchanged
M2  Group-wise LD-RW vectorized zero-branch 1 scale/group
M3  Bounded-memory sharded atomic (never full model in RAM)
M4  Quantizableのみfake-quant, excludedはexact preserve
M5  Valid HF sharded snapshot loadable with tokenizer/processor preserved (blobs dereferenced)
M6  quantize dry-run / success-failure artifacts / deterministic content fingerprint
M7  Loads via AutoModelForMultimodalLM and completes smoke without NaN/crash
M8  Protocol fingerprint equals baseline, result is observational NOT pass/fail
M9  compare reports behavioral Δ without quality claim
M10 Packed accounting uses grouped overhead, fake disk size reported separately
M11 Phase1/2 regression passes
M12 ruff+mypy+pytest green
M13 docs distinguish per_tensor/per_group, fake/packed, behavioral/quality, LD-RW/flat, blobs handling
```

**Result: ✅ PASS — all 13 gates verified (manual acceptance 2 snapshots + 2 benches + 2 compares, 122 tests, ruff/mypy green, bounded sharded proven).**

---

# Phase 4 — Calibration / Reconstruction Engine

**Goal:** Move beyond static PTQ.

**Status: Phase 4.1 ✅ CLOSED — 4.2+ IN_PROGRESS**

## Deliverables (Phase 4.1)

- [x] teacher activation capture (disk-backed sharded `artifacts/activation_cache/layer_*`, bounded, per-layer load/release)
- [x] reconstruction loss (`mse`/`l1`/`huber`, `total_squared_error/total_elements` 集約, NaN/Inf loud)
- [x] optimizer loop (layer-local `F.linear(codes*scale)` → MSE → `Adam(scaleのみ)`, `steps` =全205 Linearの1 full sweep, graph即解放)
- [x] learnable quantization parameters (non-zeroのみ `softplus(raw)+eps`, `inverse_softplus`で `effective(step0)==orig`, zero-groupはexact 0固定)
- [x] checkpoint/resume (`artifacts/checkpoint/step_*.pt`, loud error on corrupt, `--resume`はlatest, `--resume-from`明示)
- [x] calibration dataset loader (`synthetic`はCI専用 `text→tokenizer→input_ids`, 本番は `wiki-tiny`/`c4-tiny`, `allow_dataset_fallback=false`で暗黙fallback禁止)
- [x] training metrics (`calibration.json`: `loss_history`, `initial/final/best/relative`, `heldout_before/after`, `scale_fingerprint_before/after`, `mean_abs_scale_delta`, `train/heldout/smoke sample hashes` + `isdisjoint`汚染証明)
- [x] `calibrate` CLI real (dry-runはfilesystemなしで `Target modules N / Estimated cache`表示, `window`は`per-layer`のみ)
- [ ] early-stop support — deferred to 4.2

## Implementation progression

### 4.1 — Learnable scales ✅ CLOSED
`scale`のみを学習、Teacher capture→unload→`codes`固定×`scale`学習→bounded streamingでfinal materialization。

### 4.2 — Learnable thresholds (next)
`threshold`学習、per-group閾値探索。

### 4.3 — Soft-to-hard ternarization
STE / temperature annealing。

### 4.4 — Multi-layer/window reconstruction
`per-block` window, cross-layer。

### 4.5 — Research reproduction experiments
CAT-Q / ScaleQ 等は再現してから命名。

## Gate (Phase 4.1: G4-1〜12)

```
G4-1  dry-runはfilesystem変更なし、stdoutにconfig/dataset/cache estimate
G4-2  tiny fixture (4 samples,5 steps, CPU <5分) で calibrated_snapshot生成
G4-3  best_loss < initial かつ final < initial かつ finite（単調は要求しない）
G4-4  checkpoint step_2/step_4生成、--resumeはlatest step_4→step_5継続
G4-5  held-out loss改善 + sample hash isdisjointでsmoke汚染なし + smoke protocol True
G4-6  ruff/mypy/pytest green
G4-7  docs + fingerprints + dataset provenance保存
G4-8  scale_fingerprint_before != after かつ mean_abs_scale_delta >1e-9
G4-9  trainable params == scales count (他0)
G4-10 zero-groupは開始から終了までexact zero
G4-11 step0のeffective scales == naive original scales (inverse softplus)
G4-12 ActivationCache / final materializationはbounded streamingで全205 activation / full model同時RAM保持なし
```

**Result: ✅ PASS — tiny fixture `1.583→1.576` (relative 0.0045), heldout `1.460→1.455` (relative 0.0032), 141 tests, ruff/mypy green.**

Overall Phase 4 gate: Calibration consistently improves at least one held-out metric over naive ternary without using held-out evaluation data during optimization. → **Phase 4.1で達成、4.2で拡張。**

---

# Phase 5 — Japanese Retention Track

**Goal:** Optimize for useful Japanese capability rather than English-only aggregate scores.

## Deliverables

- [ ] Japanese held-out benchmark suite
- [ ] Japanese instruction following
- [ ] Japanese reasoning set
- [ ] Japanese knowledge set
- [ ] Japanese naturalness evaluation protocol
- [ ] calibration/evaluation contamination checks

## Experiment families

- general calibration
- Japanese-mixed calibration
- Japanese-heavy calibration
- reasoning-heavy calibration
- model-generated reasoning/calibration data

## Gate

A release candidate must report both:

- global/general metrics
- Japanese-specific retention metrics

---

# Phase 6 — Gemma 4 Special Components

**Goal:** Reduce total model size without destroying Gemma-specific behavior.

Research separately:

- [ ] PLE precision
- [ ] token embeddings
- [ ] LM head
- [ ] vision/audio components if supported
- [ ] hybrid precision policy

Candidate precision ladder:

```text
BF16
↓
INT8
↓
INT4
↓
Ternary
```

## Gate

Each component has an evidence-based precision choice rather than a blanket "ternary everything" rule.

---

# Phase 7 — Packed Ternary Export

**Goal:** Move from fake quantization to storage/runtime benefits.

## Deliverables

- [ ] packed ternary format
- [ ] group scale metadata
- [ ] deterministic serialization
- [ ] loading round-trip test
- [ ] packed-vs-fake-quant parity test
- [ ] size report
- [ ] runtime adapter
- [ ] GGUF feasibility/implementation study

## Gate

Packed weights reload to the same ternary codes and produce numerically equivalent dequantized weights within defined tolerance.

---

# Phase 8 — Scale Across Gemma 4

Progress only after E2B is stable.

Order:

1. E2B
2. E4B
3. 12B
4. 31B
5. MoE variant(s)

For every scale-up:

- reproduce baseline
- rerun sensitivity analysis
- do not assume E2B hyperparameters transfer

---

# Phase 9 — GUI

Only begin after the CLI interfaces and configuration schema stabilize.

Potential GUI features:

- model selection
- hardware detection
- quantization presets
- calibration dataset selector
- experiment monitor
- quality/size comparison
- benchmark dashboard
- export manager

The GUI should call the same core Python API used by the CLI.

---

# Phase 10 — Broader Model Support

Potential adapters:

- Qwen
- Llama-family
- DeepSeek-style architectures
- other open-weight LLMs

Requirements for a new adapter:

- tensor mapping
- module classification
- exclusions
- baseline benchmark
- adapter tests

---

# Release Milestones

## v0.1 — Ternary Hello World

Gemma 4 E2B loads, ternarizes, runs, and compares against baseline.

**Status 2026-08-21: 4/4 ✅ — `inspect` ✅, `benchmark` (Baseline v1) ✅, `quantize` (Phase 3 bounded sharded fake-quant, per_tensor & per_group G128) ✅, `compare` (behavioral Δ observational) ✅ → v0.1 CLOSED**

## v0.2 — Calibration

Automatic reconstruction/calibration improves naive ternary.

## v0.3 — Japanese Track

Japanese retention becomes a first-class benchmark target.

## v0.4 — Packed Model

Real storage reduction and reloadable packed weights.

## v0.5 — Multi-Gemma

At least two Gemma 4 sizes supported.

## v0.8 — Multi-Model

Adapter architecture proven outside Gemma.

## v1.0

A reproducible CLI pipeline supporting:

```text
inspect
quantize
calibrate
benchmark
compare
export
```

with documented model support and stable experiment metadata.

---

# Appendix — How to verify this snapshot

```bash
uv sync --extra ml --locked
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv run --frozen mypy src/openternary
uv run --frozen pytest -q -p no:cacheprovider  # 141 tests incl. calibration + grouping/fake_quant/cli_* + Phase 1/2
# Phase 2 single-tensor validation (requires ml extra + local snapshot)
uv run --frozen python scripts/validate_single_tensor.py
# Phase 3 whole-model fake-quant (requires ml extra + local snapshot, ~35s per_tensor, ~50s per_group)
uv run --frozen openternary quantize google/gemma-4-E2B-it-qat-q4_0-unquantized --scale-granularity per_tensor --output runs/e2b-naive-pt
uv run --frozen openternary quantize google/gemma-4-E2B-it-qat-q4_0-unquantized --scale-granularity per_group --group-size 128 --output runs/e2b-naive-g128-rw
# Phase 4.1 calibration (tiny fixture, CPU <5s, no Gemma load)
uv run --frozen openternary calibrate --dry-run
echo "calibration:\n  enabled: true\n  dataset: synthetic\n  num_samples: 4\n  seq_len: 16\n  steps: 5\n  checkpoint_interval: 2" > /tmp/calib.yaml
uv run --frozen openternary calibrate --config /tmp/calib.yaml --output runs/calib-tiny
uv run --frozen pytest -q -p no:cacheprovider tests/test_calibration_*
# header-only inspection (no ml needed)
uv run openternary inspect google/gemma-4-E2B-it-qat-q4_0-unquantized --output runs/inspect-smoke-manual
# BF16 baseline (requires ml extra + local checkpoint)
uv run openternary benchmark google/gemma-4-E2B-it-qat-q4_0-unquantized --suite smoke --no-thinking --dtype bf16 --output runs/baseline-smoke-cpu
# Fake-quant benchmark + compare
uv run openternary benchmark runs/e2b-naive-pt/artifacts/snapshot --suite smoke --output runs/e2b-naive-pt-bench
uv run openternary compare runs/baseline-smoke-cpu-1 runs/e2b-naive-pt-bench
# dependency smoke
uv run --extra ml python -c "import torch, torchvision; from PIL import Image; from transformers import Gemma4Processor; print('Gemma4 runtime dependencies OK')"
```
