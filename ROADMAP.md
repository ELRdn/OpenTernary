# OpenTernary Roadmap

This roadmap prioritizes **research validity and a stable CLI pipeline** over GUI work.

> **CLI product track:** [CLI Product Roadmap](CLI_ROADMAP.md) separately defines CLI-0–CLI-8 for reliability, shared contracts, backend/model integration, export, evaluation, and automation. Its milestones do not replace the research quality gates below.

> **Learned-rotation update — 2026-09-24:** All 35 text `q_proj` tensors were rotated using calibration-trained orthogonal matrices before G128 ternarization. The saved/reloaded mixed-precision snapshot passed the fixed quality gate on v2 validation and a new disjoint v3 test against BF16 on RX 9070 XT. This covers 35 of 205 canonical targets (2.5884% of total parameters); P7 all-205 acceptance remains open. Details: [rotation research record](docs/research/gemma4-q35-learned-rotation-20260924.md).

> **Follow-up — 2026-09-25:** A further disjoint v4 validation found that the same saved 35-target candidate fails the instruction gate (BF16 57.8125% vs 43.75%). All 205 locally learned rotations reduced held-out module error, but the saved 205-target model collapsed on v4 validation (instruction 0%, collapse 64/64). Research acceptance remains open; local reconstruction and isolated split passes do not establish robust model quality.

> **200-step rotation follow-up — 2026-09-25:** Retraining the 35 q projections for 200 steps improved mean local held-out relative MSE from 0.2180 to 0.0687. The saved/reloaded G128 snapshot passed v4 validation (BF16 and candidate instruction 57.8125%) but failed the independent v5 test instruction gate (59.375% vs 56.25%, a 3.125-point regression against the 2-point limit). The 205-target acceptance remains open.

> **All-205 200-step screen — 2026-09-25:** All 205 local rotations improved held-out error and preserved BF16 top-1 before ternary rounding on three short probes, but the in-memory hard G128 model still failed v4 validation (instruction 0%, collapse 64/64). This screen was not materialized as a second full snapshot. See the research record for the separate saved 50-step candidate and the 200-step screen.

> **Blockwise follow-up — 2026-09-25:** Hard codes and G128 scales for all 205 rotated targets were saved across 35 decoder blocks. Local block held error improved, but combining all blocks in memory still failed v4 validation (instruction 0%, collapse 56/64). Mixed English/Japanese calibration passed v4 for the first block's seven targets only; adding the next block failed. Quantized-upstream, joint-window, dense-latent, and answer-logit pilots also failed their model-level v4 gates. The 205-target P7 goal remains open. Evidence: [blockwise research record](docs/research/gemma4-blockwise-hard-ternary-20260925.md).

> **Eight-target rotation pilot — 2026-09-25:** Layer 0's seven hard-G128 projections plus an initialized learned rotation on layer 1 `q_proj` passed v4 validation and the disjoint v5 test after save/reload. The 200-step global rotation retraining selected by calibration held loss failed v4; the passing 96/32-row run selected step 0. This validates an eight-target rotate-then-ternarize artifact, not training improvement or all-205 quality acceptance. See the same [research record](docs/research/gemma4-blockwise-hard-ternary-20260925.md).

> **Fixed signed H1024 follow-up — 2026-09-25:** The Bonsai 2-inspired fixed signed Hadamard geometry was implemented with a 1024+512 split for Gemma 4's 1536-dimensional inputs and screened at H128/H256/H512/H1024. Signed H1024 improved 14-target quality over unrotated G128, but all fixed-rotation pilots failed v4. The all-205 in-memory H1024 candidate failed with English/Japanese PPL 7261.13/22939.74, instruction 0%, collapse 1/64. Least-squares G128 scale/code refit lowered weight MSE but did not pass model quality. The saved/reloaded 205-target acceptance and new untouched final test remain open. See [the signed Hadamard research record](docs/research/gemma4-fixed-signed-hadamard-20260925.md).

> **Mixed-geometry follow-up — 2026-09-25:** The saved eight-target base plus fixed signed H1024 on layer-1 `k_proj` and `up_proj` passed one in-memory v4 screen. The two added projections were saved as hard G128 codes/scales, then independently reloaded. Three identical GPU-artifact reloads produced 2 PASS and 1 FAIL because one instruction answer changed, despite bitwise checked weights. The 10-target candidate is not robustly accepted; runtime reproducibility and the remaining 195 targets are open.

> **Eager-attention follow-up — 2026-09-25:** The saved 10-target candidate passed eager-attention matched BF16 gates on v4 and v5, with repeatable v4 summary results in two independent processes. The default attention path remains numerically fragile; 195 targets and a new unopened final test remain. See [the signed Hadamard research record](docs/research/gemma4-fixed-signed-hadamard-20260925.md).

> **Eleven-target follow-up — 2026-09-25:** Adding fixed signed H1024 layer-2 `up_proj` to the ten-target base passed eager-attention v4 after save/reload in two independent processes and passed v5 once. GPU re-quantization matched the saved hard codes and BF16 weights across five repeats. Layer-2 `q_proj` and `k_proj` single additions failed v4 instruction regression. The remaining 194 canonical targets and an unopened final test are still required for P7.

> **Twelve-target follow-up — 2026-09-25:** The saved 11-target base plus fixed signed H1024 layer-2 `o_proj` passed matched eager-attention BF16 gates on v4 and v5. Independent saved reloads of both splits reproduced identical summary values. The remaining layer-2 `v_proj`, `gate_proj`, and `down_proj` single additions failed v4. P7 still requires the other 193 canonical targets, an unopened final test, and a deployable runtime.

> **Thirteen-target follow-up — 2026-09-25:** A GPU-materialized fixed signed H1024 layer-3 `down_proj` brought the saved/reloaded pilot to 13 canonical targets. It passed eager-attention v4 and v5 against matched BF16 controls. Layer-3 `o_proj` passed individually, but combining both failed the English-PPL gate; the other five layer-3 single additions failed. P7 still requires 192 other targets, an unopened final test, and a deployable runtime. See the [research record](docs/research/gemma4-fixed-signed-hadamard-20260925.md).

> **Seventeen-target follow-up — 2026-09-25:** A GPU-materialized signed H1024 layer-4 `q_proj`/`k_proj`/`o_proj`/`up_proj` block passed eager-attention matched BF16 gates after saved reload on v4 and v5. The saved v4 result equaled the in-memory screen, and an independent v5 repeat reproduced its summary and all 64 instruction responses. The other three layer-4 single additions failed v4. This remains a partial 17/205-target result; 188 canonical projections, a new unopened final test, and a deployable runtime remain for P7.

> **Eighteen-target follow-up — 2026-09-25:** Re-adding layer-3 `o_proj` to the saved 17-target base passed the eager-attention v4 and v5 gates after save/reload. An independent v5 run reproduced all summary metrics and 64 instruction responses. This interaction differs from its earlier failed pairing with layer-3 `down_proj` on the 12-target base. P7 still requires the other 187 canonical projections, a new unopened final test, and a deployable runtime; see the [research record](docs/research/gemma4-fixed-signed-hadamard-20260925.md).

> **Twenty-one-target follow-up — 2026-09-25:** The saved 18-target base allowed each remaining layer-1 role to pass alone, but all four together failed v4. The `v_proj`/`o_proj`/`gate_proj` triple passed matched eager-attention BF16 gates on v4 and v5 after saved reload, with independent repeats on both splits. The v4-selected `o_proj`/`gate_proj`/`down_proj` triple failed v5 twice, demonstrating selection sensitivity. P7 still requires 184 other targets, an unopened final test, and a deployable runtime; see the [research record](docs/research/gemma4-fixed-signed-hadamard-20260925.md).

> **Twenty-four-target follow-up — 2026-09-25:** All seven layer-5 fixed signed H1024 projections together failed the v4 English/Japanese PPL gates. The individually passing `q_proj`/`k_proj`/`down_proj` combination was GPU-materialized and passed matched eager-attention BF16 gates after saved reload on v4 and v5. Independent repeats reproduced both summary values and all 64 instruction responses. P7 still requires 181 other canonical projections, a new unopened final test, and a deployable runtime; see the [research record](docs/research/gemma4-fixed-signed-hadamard-20260925.md).

> **Twenty-eight-target follow-up — 2026-09-25:** All seven layer-6 signed H1024 projections together failed v4 English-PPL and instruction gates. The individually passing `q_proj`/`k_proj`/`v_proj`/`up_proj` combination was GPU-materialized as hard G128 and passed matched eager-attention BF16 gates on v4 and three saved v5 reloads. The v5 candidate varied on two of 64 generated answers between the first and later runs, though all passed. P7 still requires 177 other canonical projections, a new unopened final test, and a deployable packed runtime; see the [research record](docs/research/gemma4-fixed-signed-hadamard-20260925.md).

> **Layer-7 follow-up — 2026-09-25:** All seven layer-7 projections together failed v4 instruction quality on the saved 28-target base. Four individual additions passed, but their combined 32-target candidate failed. The saved/reloaded 31-target `q_proj`/`k_proj`/`up_proj` candidate passed v4, then failed v5 in two independent processes; the alternative `q_proj`/`k_proj`/`v_proj` triple passed v4 in memory but failed v5 English PPL. The strongest passing saved pilot remains 28/205. P7 still requires 177 more targets, an unopened final test, and native packed evaluation; see the [research record](docs/research/gemma4-fixed-signed-hadamard-20260925.md).

> **Validation screening update — 2026-09-22:** P0は固定revision/source hash/正準205対象/RX 9070 XTでPASS。P1は実ROCm OOM後のBF16 backward/Adam復旧とtransactional real-pathを確認。P2はcheckpoint v3、tiny mid-step再開同一性、全205 uninterrupted対終端checkpoint再開でloss/fingerprint/22 snapshot file hash完全一致を確認。P3は固定・分離済みvalidationでBF16/naive/scale-only/thresholdを同一protocol評価し、全ternary候補が崩壊gate不合格。P4はrunner/checkpoint/materializeへ接続しlinear/cosine/exponentialを全205でhard保存後評価したが、3候補とも不合格。hard code変化率0は、現実装がscaleのみ学習し、固定weight/reference scaleからhardeningするため構造的に不変と判明した。test splitは未使用、modulation/P5–P7は未実施。詳細は [foundation status](docs/PHASE4_FOUNDATION_STATUS.md) と [P4 redesign decision](docs/plans/p4-soft-to-hard-redesign-decision.md) を参照。

> **Active execution plan:** P0–P7 の固定予算、比較群、品質式、phase gate、P7受入成果物は [P0–P7 research acceptance plan](docs/plans/p0-p7-research-acceptance.md) を正とする。

> **Research planning update — 2026-09-18:** Phase 4.5 に回転ベース三値量子化の研究候補と実験前ゲートを追加。既存の 4.3 / 4.4 は維持する。これは計画の更新であり、回転の実装・実験・Gemma 品質改善を完了したという意味ではない。以下の 2026-08-23 snapshot は当時の実装記録であり、現時点の品質・CI 合格を保証しない。

> **Progress snapshot — 2026-08-23 (Phase 4.1 CLOSED, Phase 4.2 IN_PROGRESS)**
> - **Phase 0 — Repository Foundation: ✅ COMPLETE**
> - **Phase 1a — Gemma 4 E2B Inspection: ✅ COMPLETE** — 1951 tensors / 5,104,298,467 params / 205 quantizable (0.359605) / BF16 9.5075 GB / text 35 layers hidden 1536 vocab 262144, fingerprint `sha256:07fe44eae504218937187b75826a4c780cba2f76a9145fdc38f7efdd4d3a7b01` stable.
> - **Phase 1b — BF16 Smoke Inference Baseline v1: ✅ COMPLETE** — `AutoProcessor + AutoModelForMultimodalLM`, `bf16 exact`, `thinking=False`, `do_sample=False/num_beams=1`, suite `smoke v1` 5 prompts, warmup→5 timed, 2-run PASS: protocol `sha256:11ac43e68f9bc2079365b224078f4af93dd767c813a0d3d82485c0096507d52d`一致, result `sha256:907fbb73f91d0e97033c59cc1dae65817a9b507d8df4e3b65125a87b9327cfe8`一致.
> - **Phase 1 Gate: ✅ PASS** — Behavioral baseline frozen, not Quality (Phase 5).
> - **Phase 2 — Ternary Core & Packing Accounting: ✅ CLOSED** — `src/quant/{ternary,packing,accounting,metrics}.py` canonical AbsMean, 2bit-v1, per-tensor sum, G1–G19 PASS.
> - **Phase 3 — Model-Level Ternarization (bounded sharded fake-quant): ✅ CLOSED** — LD-RW per_group vectorized zero-branch, bounded sharded writer (512 MiB, flush-before-add, over-size single), HF blobs symlink dereference, content fingerprint (MUST) vs file hash (SHOULD), scale summary+fingerprint (no JSON scale array), `quantize`/`compare` CLI real, whole-model `e2b-naive-pt` (12 shards, 205 tensors, 14.3M groups for G128) + `e2b-naive-g128-rw` both loadable and smoke PASS, protocol match True, result observational, M1–M13 PASS.
> - **Phase 4.1 — Calibration recon-scale: ✅ CLOSED** — layer-local `codes*scale→F.linear` differentiable, Teacher activation disk-backed sharded, `inverse_softplus`初期化で `effective(step0)==orig`, `per_tensor/per_group` dequantize helper再利用, `steps=full sweep`定義, `sample hash isdisjoint`汚染証明, `calibrate` CLI real (dry-run/filesystemなし, checkpoint/resume), tiny fixture 4 samples/5 stepsで `best<initial`かつ `final<initial`かつ `heldout`改善、G4-1〜12 PASS.
> - **Phase 4.2 — Calibration recon-threshold (scale+threshold): 🚧 IN_PROGRESS (T4.2-1..6 ✅, T4.2-7/8 ⏳)** — `threshold+scale`分離・`clipped STE`（`threshold_ratio=eps+(1-2eps)*sigmoid(raw)`、`hard_threshold_codes`/`ste_threshold_codes` via `src/openternary/quant/threshold.py`）、`reference_scale` frozen × `reconstruction_scale` learnable、config `configs/gemma4-e2b-threshold.yaml` 準拠、runner 同時最適化＋bounded materialize 対応完了。CLI 表示・ドキュメント最終化は保留。

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

**Status: P0/P1 and P3 evaluation path PASS; P2 full-target terminal-resume identity PASS with full-target mid-optimizer identity unrun; P4 production screen completed with negative result**

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

## Phase 4.2 — Learnable Thresholds (recon-threshold) ✅ IMPLEMENTED / ❌ QUALITY GATE

**Goal:** `threshold` を `scale` と分離して学習し、`codes` 自体が最適化中に変化するパスを確立する。Phase 2/4.1 の暗黙 `±0.5*scale`（`round`）を `threshold_ratio ∈ (0,1)` で一般化し、clipped STE で微分可能にする。

**Spec:** `docs/plans/phase-4.2-threshold.md` §2（定義・STE・パラメタ化・Config・Runner・Materialize）、`ARCHITECTURE.md` §7、`configs/gemma4-e2b-threshold.yaml`（`PROJECT_SPEC.md` §6/§9 準拠）が正。

### Sub-tasks (T4.2-1..8 — 計画書 Table 準拠)

- [x] **T4.2-1 閾値パラメタ化ヘルパ** — `src/openternary/calibration/optimizer.py` に `build_threshold_params` / `get_effective_threshold_ratio` / `inverse_sigmoid_threshold_ratio` 追加。`ratio = eps + (1-2eps)*sigmoid(raw)`、`eps=0.01`、初期 `0.5` は `inverse_sigmoid(0.5)=0` （`build_scale_params` 既存ロジックは不変・zero_mask 固定を維持）
- [x] **T4.2-2 Threshold-aware quantize** — `src/openternary/quant/threshold.py`（新規）に `hard_threshold_codes` / `ste_threshold_codes` / `_expand_per_group` を実装。`threshold_ratio=0.5` で既存 `quantize_absmean`/`quantize_groupwise`（`round` 由来 `±0.5*scale`）と codes 一致、`0.3/0.7` で zero_ratio 単調性を確認。`ternary.py`/`grouping.py` の `round` ties-to-even 契約は不変
- [x] **T4.2-3 STE 最小実装 (clipped STE)** — `src/openternary/quant/threshold.py` 内 `_ClippedSTE(torch.autograd.Function)`：forward=`hard_gate`、backward=`surrogate` へ `grad_output` を素通し。`surrogate = clamp(0.5 + (u - thr)/(2*ste_width), 0, 1)`、`ste_width=0.1`、zero-ref グループは gate/surrogate 共に 0 で grad 遮断
- [x] **T4.2-4 Config 拡張** — `src/openternary/config/schema.py` `CalibrationConfig` に `threshold_enabled` / `threshold_lr` / `threshold_init_ratio` / `threshold_granularity` / `threshold_estimator: clipped-ste` / `threshold_ste_width` / `threshold_eps` / `init_from` 追加。`method: recon-scale | recon-threshold`（デフォルト `recon-scale` で Phase 4.1 と完全互換）、`window` は `per-layer` のみ維持。`configs/gemma4-e2b-threshold.yaml` は本スキーマに準拠
- [x] **T4.2-5 Runner 同時最適化** — `src/openternary/calibration/runner.py` で `raw_threshold` を per-module に追加、`Adam([{scales, lr}, {thresholds, threshold_lr}])` 同時最適化。各 forward で `reference_scale`（frozen AbsMean）と `eff_thr_ratio` から `ste_threshold_codes` で codes 再計算 → `w_hat = codes_ste * reconstruction_scale`（`_expand_per_group` で broadcast）。checkpoint に `raw_thresholds`/`threshold_eps`/`threshold_ste_width` を保存、`--resume` は `threshold_enabled` 不一致を loud error
- [x] **T4.2-6 Materialize 閾値対応** — `src/openternary/quant/fake_quant.py` `materialize_calibrated_snapshot` は `calibrated_state[codes]` をそのまま利用（threshold 反映済み hard codes）、`runner.py` 側で W を bounded に再ロードして `hard_threshold_codes` で最終 codes を確定。`threshold=0.3` で `content_fingerprint` が scale-only と差異、sharded roundtrip は bounded（`O(largest tensor + 512 MiB)`）を維持、`_run_materialize_only` も thresholds 復元対応
- [x] **T4.2-7 CLI 表示** — `calibrate --dry-run` に threshold設定を表示
- [x] **T4.2-8 ドキュメント** — `ROADMAP.md` / foundation status / `EXPERIMENT_LOG.md` を実測結果へ更新

依存順: `T4.2-1 → T4.2-2 → T4.2-3 → T4.2-5 → T4.2-6`、`T4.2-4` は並行可。`T4.2-7/8` は残作業。

**Config 正準:** `configs/gemma4-e2b-threshold.yaml`（`model.id=google/gemma-4-E2B-it-qat-q4_0-unquantized@6befbaca...`、`quantization: per_group G128 last-dim-rowwise-v1`、`calibration: method=recon-threshold / dataset=wiki-tiny / threshold_*` 一式）は `PROJECT_SPEC.md` §6（再現性要件: config/model/seed等を保存）および §9（YAML スキーマ）に準拠。`allow_dataset_fallback=false` / `held_out_ratio=0.2` / `seed=42` で汚染分離を担保。参照切れなし（`ARCHITECTURE.md` §7、`docs/plans/phase-4.2-threshold.md` §2.4/§6、`src/openternary/config/schema.py` と整合）。

## Implementation progression

### 4.1 — Learnable scales ✅ CLOSED
`scale`のみを学習、Teacher capture→unload→`codes`固定×`scale`学習→bounded streamingでfinal materialization。

### 4.2 — Learnable thresholds ✅ IMPLEMENTED / ❌ QUALITY GATE
`threshold`学習、per-group閾値探索。`reference_scale` frozen × `reconstruction_scale` learnable の分離、`clipped STE`（`threshold_ratio` → `hard gate` + `surrogate`）で `codes` を毎 forward 再計算。`configs/gemma4-e2b-threshold.yaml` が正準。

### 4.3 — Soft-to-hard ternarization

Linear/cosine/exponential temperature schedules、最終hard区間、soft ternary expectation、zero-code bias、deterministic hardening を本番runner/config/checkpoint/materializeへ接続。各scheduleを正準205対象・10 steps/module・validation-onlyで実行し、最終hard保存・再読込後に評価した。3候補とも指示0%、崩壊8/8、`code_change_ratio=0` で品質gate不合格。現実装ではoptimizerがscaleのみを所有し、hard assignmentは固定weight/reference scaleのmidpointで決まるため、この0はschedule長に依存しない構造的結果。winner不在のためzero-code bias比較は開かず、test splitも未使用。再実験前にparameterizationのarchitecture decisionが必要。

### 4.4 — Multi-layer/window reconstruction
`per-block` window, cross-layer。

### 4.5 — Research reproduction experiments

CAT-Q / ScaleQ / TWLA 等は再現してから命名。部分的な導入は `twla-inspired` / `rotated-ternary` とし、論文再現とは区別する。

#### Rotation-based ternary research candidate — 2026-09-18

**Status: PLANNED / UNVALIDATED on Gemma 4 E2B.** 実装・長時間実験の開始は別途承認を要する。モデル対象、正準205 Linear、既存の除外範囲は変更しない。回転不足が過去の出力崩壊の原因だったと断定しない。

**Hypothesis:** scale / threshold の調整に加え、量子化前の座標系を変えることで三値近似誤差を減らせる可能性がある。固定 Hadamard は低コストの比較候補であり、学習した三峰性整形や TWLA 全体の代替・再現ではない。

参考手法を分けて評価する:

- **TWLA / E2M-ATQ:** オフセットと scale を持つ非対称三値量子化。現行の対称 `codes * scale` とは異なるため、表現・保存・推論契約の変更は別途設計レビューする。
- **TWLA / KOTMS:** 三値向け分布を目指す学習可能な Kronecker 構造の直交回転。単純な固定 Hadamard 追加とは区別する。
- **TWLA / ILA-AMP:** 隣接層への影響も考慮する活性化混合精度。初期比較では導入せず、活性化は16bit（現行 BF16）に固定する。A4 / 混合精度は重み側の有効性確認後の別研究とする。
- **SpinQuant:** 学習回転とランダム回転の比較設計の参考。三値 Gemma での成功を示す根拠としては扱わない。

#### Entry gate — 新しい校正実験より先に解決すること

以下は前回監査を踏まえた未完了の確認・修正要件。古い成果物や tiny fixture の PASS だけでは代替しない。

- [ ] 元モデルの revision / 重みの完全性・可用性、正準205対象、除外モジュールの保存を確認する。異なる対象集合の過去 checkpoint を無条件に比較対照へ流用しない。
- [ ] padding を再構成 loss / 評価から除外し、実効 token 数・言語構成・長さ分布を記録する。calibration / validation / 最終 held-out を分離し、回転 seed の選定に最終評価データを使わない。
- [ ] module-major loss は同じ module の before / after と有効要素数で重み付けした集約で比較する。設定と実際の optimizer / loss / 対象範囲の一致を確認する。
- [ ] checkpoint / resume / materialize の対象集合一致、最終 module、欠損時の停止、cache の同一性を検証する。復旧で失われた必須 metadata を未確認のまま成功判定しない。
- [ ] microbatch の計算グラフ解放・OOM 時の復旧を含むメモリ上限、関連テスト・lint・型検査を確認する。
- [ ] Phase 5 の全完成を待たず、この研究比較に必要な perplexity と独立した日本語評価を先に用意する。評価設定・数値許容差・品質改善幅・許容劣化・時間/VRAM上限・中止条件を実験前に固定する。

#### Staged experiments — 段階ごとに合格してから拡大する

| Stage | 比較・検証対象 | 次へ進む条件 |
|---|---|---|
| R0 — Corrected control | 同一 revision / 対象 / データ / 校正予算で、BF16・naive・修正済み回転なし校正を比較 | 比較条件と品質評価が有効。既存の崩壊した出力だけを対照にしない |
| R1 — Function preservation | 量子化せず、単一 Linear → block → Gemma text path の回転前後、および保存・再読み込み後を比較 | FP32 と実使用 BF16 で事前定義した誤差内、有限値、除外範囲不変 |
| R2 — Fixed rotations | 回転なし / 正規化 Hadamard / ランダム符号付き Hadamard。まず小規模な対象で比較 | validation 上の誤差・品質と追加コストを報告。seed ごとの変動と負の結果も保存 |
| R3 — Learned rotations / asymmetric quantizer | 学習回転の有無 × 非対称量子化の有無を切り分ける | 単独効果と組み合わせ効果を確認。固定 Hadamard の不振だけで学習回転を否定しない |
| R4 — Whole-model acceptance | 選定した構成を正準205対象へ適用し、独立 held-out と反復実験で評価 | 事前登録した品質・日本語・資源ゲートを満たす。満たさなければ不採用または再設計 |

全比較で group size、対象集合、データ、dtype、生成条件を揃え、回転の学習・探索に使う追加予算も明記する。記録する指標は module 別再構成誤差、perplexity、日本語品質、出力崩壊、zero / sign 比率、scale / threshold、max-to-RMS、必要に応じ尖度、実測時間・RAM/VRAM・保存容量。局所 MSE の改善だけで全体品質の改善を宣言しない。

#### Implementation / export constraints — 設計レビュー事項

- 列ベクトル表記では `W x = (W R)(R^T x)`。PyTorch の行バッチ表記では `W_rot = W R`, `X_rot = X R` として `X_rot W_rot^T = X W^T`。正規化、転置、ランダム符号の掛け順を明記し、直交性と等価性をテストする。
- `G128` は量子化単位であり、回転ブロックを128にする必然性はない。`1536 = 12 * 128` は分割可能性だけを示す。固定 Hadamard が常に外れ値や三値誤差を減らすとは仮定しない。
- 回転した三値重みだけを通常の Linear に渡してはならない。対応する入力変換、または数学的に等価な融合が必要。逆回転を重みに戻すと一般に三値性が失われるため、それを packed ternary と呼ばない。
- 回転 wrapper / adapter、回転行列または再構成に必要な seed・方式・寸法・version、scale / offset の保存契約と round-trip を設計する。RoPE / 非線形演算 / normalization / shared KV / PLE をまたぐ融合は等価性の検証なしに行わない。
- Phase 7 の packed export は回転・offset の metadata と演算コストを含めて再検証する。fake-quant の動作確認と packed runtime の速度・容量改善は別の合格条件とする。
- TWLA の報告対象は LLaMA / Qwen3 系、実験環境は NVIDIA A6000。Gemma 4 / ROCm 対応は未検証として依存関係・演算ごとに確認する。`torch.cuda` という API 名だけで ROCm 非対応と断定しない。

#### Primary references

- [TWLA paper v2](https://arxiv.org/abs/2606.13054v2) — 手法と実験条件。固定回転だけを追加して再現と呼ばない。
- [TWLA official repository](https://github.com/Kishon-zzx/TWLA) / [KOTMS implementation](https://github.com/Kishon-zzx/TWLA/blob/master/scripts/KOTMS.py) — 入力変換と回転 metadata 保存の参考。導入時には commit を固定し、README の Apache-2.0 表記だけでなく由来コードのライセンス・帰属要件も確認する。
- [SpinQuant paper](https://arxiv.org/abs/2405.16406) — 回転選択と学習の比較。実装開始時に参照版を固定する。

上記は 2026-09-18 の文献・コード確認を受けた研究計画。実装済み機能や新しい実験結果は追加していない。

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

### Gate (Phase 4.2: G4.2-1〜12 — 計画書 §4 準拠、T4.2-7/8 完了後に最終判定)

```
G4.2-1  dry-runはfilesystem変更なし、threshold_enabled/threshold_init/cache estimateを表示
G4.2-2  tiny fixture (4 samples,5 steps, CPU <5分) で threshold_enabled=true が scale-only より best_loss ≤ 同等 かつ finite
G4.2-3  threshold_enabled=false 時に 4.1 と完全互換（loss_history / fingerprint が 4.1 と一致）
G4.2-4  checkpointに raw_thresholds が保存され、--resumeで復元され loss が継続（threshold_enabled 不一致は loud error）
G4.2-5  held-out lossが scale-only 比で改善 or 同等、かつ train/held/smoke の hash isdisjoint 汚染なし
G4.2-6  ruff/mypy/pytest green（新規 tests 含め 150+）
G4.2-7  docs + fingerprints + dataset provenance保存、threshold_fingerprint_before != after かつ mean_abs_threshold_delta >1e-9 (enabled時)
G4.2-8  trainable params == scales count + thresholds count（非ゼロのみ）、ゼログループは exact zero 維持
G4.2-9  step0の effective_threshold_ratio == 0.5、threshold=0.5 時に naive と codes 一致
G4.2-10 ActivationCache / materializeはbounded streaming維持、4.6GB embedでOOMなし
G4.2-11 materializeした calibrated_snapshotが threshold反映で content_fingerprintが scale-onlyと差異、かつ AutoModel load + smoke PASS
G4.2-12 --materialize-onlyが thresholdチェックポイントから正しく再現
```
→ T4.2-1..8 は実装・tiny/実ROCm経路・全205 validation screenで確認済み。保存後threshold候補は general PPL 315,223.93 / Japanese PPL 140,044,860.87 / instruction 0% / collapse 8/8 で品質gate不合格。実装完了と研究採用は分離する。

Overall Phase 4 gate: Calibration consistently improves at least one held-out metric over naive ternary without using held-out evaluation data during optimization. → **Phase 4.1 の tiny fixture での達成と、実 Gemma の品質保持は別判定。実モデルの科学的受け入れは未完了として扱い、4.2 の是正・検証と上記研究ゲートで確認する。**

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
