# OpenTernary Experiment Log

---

# Experiment: Phase 1 — BF16 Smoke Inference Baseline v1 (2-run reproducibility)

## Metadata

- Run 1 ID: `02619d17` (dir: `runs/baseline-smoke-cpu-1`)
- Run 2 ID: `02a3...` (dir: `runs/baseline-smoke-cpu-2` — see `environment.json` for full ID)
- Date: 2026-08-20T13:36:42Z (Run1), 2026-08-20T13:43:xxZ (Run2)
- Git commit: `unknown` (no git repo in manual smoke env — `environment.json: git_commit=unknown`)
- Branch: `unknown`
- Agent/operator: manual acceptance (CPU, Windows 11, uv 0.12.5)
- Model: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- Model revision: `6befbaca7398925921802abd1f277b495b78b738` (local `.../snapshots/6befbaca...`, 10.2 GB)
- Hardware: `AMD64 Family 25 Model 97 / 91.59 GB RAM / gpu_name=none / cuda_available=false` (CPU-only)
- OS: `Windows-11-10.0.26200-SP0`
- Python: `3.12.9` (`.venv`, `uv python pin 3.12`)
- PyTorch: `2.13.0+cpu`
- Torchvision: `0.28.0+cpu`
- Transformers: `5.15.1`
- Pillow: `12.3.0`
- Accelerate: `>=0.30` (via `ml` extra)
- Seed: `42`
- Device: `cpu` (`--device cpu` explicit; `device=auto` now also resolves to cpu on this hardware)
- Dtype: `bf16` exact (`requested bf16 → actual torch.bfloat16`, silent fallback forbidden)

---

## Objective

Freeze the **BF16 Smoke Inference Baseline v1** before any ternarization. Establish the behavioral/inference reference for `Δoutput` and reproducibility fingerprints.

Questions:
- Does the real Gemma 4 E2B checkpoint (1951 tensors, 5.1B params) load via `AutoProcessor + AutoModelForMultimodalLM` on CPU with `bf16 exact`?
- Does `smoke v1` (5 fixed prompts, `thinking=False`, greedy `do_sample=False/num_beams=1/max_new_tokens=64`) produce deterministic `generated_token_ids` across 2 runs?
- Do `protocol_fingerprint` and `result_fingerprint` match across runs (latency variance excluded)?

---

## Hypothesis

CPU BF16 inference with `device_map {"": "cpu"}` and `warmup=1` will succeed, and greedy generation will be deterministic (same `generated_token_ids`).

---

## Configuration

```yaml
# configs/gemma4-e2b.yaml (Phase 1b)
model:
  id: google/gemma-4-E2B-it-qat-q4_0-unquantized
  revision: 6befbaca7398925921802abd1f277b495b78b738
benchmark:
  suite: smoke
  thinking: false
  generation:
    do_sample: false
    num_beams: 1
    max_new_tokens: 64
  warmup: true
seed: 42
device: cpu  # auto now also resolves to cpu on CUDA-unavailable
dtype: bf16
```

`benchmark.json: suite.version=1`, generation `do_sample false / num_beams 1` (no temperature/top_p/top_k).

---

## Calibration Data

- Dataset: N/A (Baseline — no calibration)
- Known contamination risks: `smoke` suite is held-out from calibration by definition.

## Evaluation Data

- Suite: `smoke v1` — `SMOKE_SUITE_VERSION=1`, 5 fixed prompts
  - `smoke.en.short` — `Write a one-sentence greeting in English.`
  - `smoke.ja.short` — `日本語で一文の挨拶を書いてください。`
  - `smoke.ja.summary` — 要約 (OpenTernary文)
  - `smoke.reasoning.simple` — Bloops/Razzies/Loppies
  - `smoke.code.python` — `add(a,b)` code only
- Split: fixed
- Held out from calibration: yes
- Prompt count: 5

---

## Results

### Protocol / Result Fingerprints (FULL — do not truncate)

| Value | SHA-256 |
|---|---|
| `protocol_fingerprint` (suite/version/prompts/thinking/generation) | `sha256:11ac43e68f9bc2079365b224078f4af93dd767c813a0d3d82485c0096507d52d` |
| `result_fingerprint` (prompt ids + generated_token_ids) | `sha256:907fbb73f91d0e97033c59cc1dae65817a9b507d8df4e3b65125a87b9327cfe8` |

Both values read from `runs/baseline-smoke-cpu-1/benchmark.json` and `runs/baseline-smoke-cpu-2/benchmark.json` — **identical across both runs** (protocol and result).

Source files:
- `runs/baseline-smoke-cpu-1/benchmark.json: suite.fingerprint` / `result_fingerprint`
- `runs/baseline-smoke-cpu-2/benchmark.json: suite.fingerprint` / `result_fingerprint`

Log prefixes `sha256:11ac43e68f9...` / `sha256:907fbb73f91...` are truncated displays only — use FULL values above.

### Behavioral (NOT Quality)

This is **BF16 Smoke Inference Baseline** — quality scores (accuracy, perplexity, Japanese HQ) are NOT measured here. Phase 5 will measure quality degradation. `result_fingerprint` change in Phase 2/3 means “deterministic output changed vs BF16 baseline”.

### Efficiency

| Run | total_output_tokens | total_generation_ms | avg_latency_ms | tok/s (output) | model_load_ms |
|---|---:|---:|---:|---:|---:|
| 1 (`02619d17`) | 160 | 33182.37 | 6636.47 | 4.82 | 599.98 |
| 2 | 160 | 31189.11 | 6237.82 | 5.13 | 621.05 |

Latency variance is expected and **excluded from fingerprints**.

Per-prompt Run1 (all 5 output_tokens sum 160): en 8, ja.short 64, ja.summary 33, reasoning 36, code 19 — identical across runs.

### Phase 1a Inspection (for completeness)

- `runs/inspect-smoke-manual/inspection.json`: `tensors 1951 / total_params 5,104,298,467 / quantizable 205 / ratio 0.359605 / BF16 9.5075 GB / text layers 35 / hidden 1536 / vocab 262144`
- `inspection_fingerprint`: `sha256:07fe44eae504218937187b75826a4c780cba2f76a9145fdc38f7efdd4d3a7b01` stable.

### Failures / Diagnostics

Manual smoke failures before dependency fix (each left a `runs/*/metrics.json` with `status: failed` after fix):
- `torch` missing → `ImportError: torch is required` (loud error, `exit 2`)
- `Pillow` missing → `Gemma4Processor requires the PIL library`
- `torchvision` missing → processor import failure
- `device=auto` on CPU-only → `You are trying to offload the whole model to the disk` (fixed by resolving `auto → {"": "cpu"}` when `cuda unavailable`; not auto-falling back to disk_offload)

After `uv sync --extra ml --locked` and `uv add --optional ml pillow torchvision` (`pillow==12.3.0`, `torchvision==0.28.0+cpu`), both runs PASS.

No NaN, no OOM, BF16 exact verified (`requested torch.bfloat16 == actual`).

---

## Qualitative Observations

- All 5 prompts generated plausible outputs (en greeting, ja greetings list, ja summary correct, reasoning correct “No…”, code correct).
- `parse_response` with `prefix=input_ids` preserved.

---

## Interpretation

- ✅ Real Gemma 4 E2B loads and generates on CPU BF16 exact.
- ✅ `smoke v1` suite, `thinking=False`, greedy generation are frozen with fingerprints.
- ✅ 2-run reproducibility PASS at `generated_token_ids` level (stronger than text-only).
- ✅ Behavioral baseline frozen for `Δoutput` in Phase 2/3.
- ❌ No quality claim — do not report “quality degradation X%” from this suite alone.

---

## Next Experiment

**Phase 2 — Ternary Core & Packing Accounting** → see next entry (CLOSED 2026-08-21).

---

# Experiment: Phase 2 — Ternary Core & Packing Accounting (single real tensor roundtrip)

## Metadata

- Run ID: see `runs/2026-08-20_gemma-4-E2B-it-qat-q4_0-unquantized-naive-g128-001` (`phase2_single_tensor_validation.json`)
- Date: 2026-08-21TXX:XX:XXZ (see `environment.json` in same dir)
- Git commit: unknown (manual acceptance env)
- Branch: unknown
- Agent/operator: manual acceptance (CPU, Windows 11, uv 0.12.5, python 3.12.9, torch 2.13.0+cpu)
- Model: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- Model revision: `6befbaca7398925921802abd1f277b495b78b738` (local `.../snapshots/6befbaca...`)
- Hardware: AMD64 / 91GB RAM / cuda_available=false (CPU-only)
- Seed: 42 (quantization is deterministic, no sampling)
- Tensor: `model.language_model.layers.0.mlp.gate_proj.weight` — shape [6144, 1536] BF16, 9,437,184 params, selection policy `preferred-list-else-lexicographic-first` (sorted 205 quantizable, gate_proj found)

---

## Objective

Validate Phase 2 canonical primitives on one real quantizable Gemma 4 tensor without loading whole model: `quantize_absmean (FP32, device-preserving, zero-branch) → pack (2bit-v1 00/01/10/11) → unpack → dequantize → metrics`. Check G1–G10.

---

## Configuration

```yaml
quantization:
  scheme: absmean-per-tensor
  scale_dtype: fp32
  packing: 2bit-v1  # 00=-1,01=0,10=+1,11=reserved, c0=bits7-6, tail 00, len==ceil(N/4)
metrics:
  packing: vectorized (9M elements)
```

No grouping, no calibration, no benchmark quality.

---

## Results

### Quantization

- `scale = 0.01916767656803131` (FP32 accumulation `mean(abs(W.to(float32)))`)
- `Q ∈ {-1,0,1}` only — histogram zero 0.3257 / neg 0.3373 / pos 0.3369
- `shape preserved` [6144,1536]
- `device-preserving` (CPU→CPU; CUDA path tested in unit tests)
- `round` ties-to-even verified via unit tests `0.50*scale → 0`

### Packing

- `packed_bytes = 2,359,296` (`ceil(9437184/4)`), `packed_bits = 18,874,368`, `logical_2bit_bits = 18,874,368`, `byte_padding_bits = 0` (N divisible by 4)
- `unpack(pack(Q)) == Q` **bit-exact PASS** (flattened compare, strict=True)
- `tail padding 00` canonical, `reserved 11` rejected (unit tests), `len==ceil(N/4)` enforced

### Metrics (tensor diagnostics, not LM quality)

- `mae = 0.008733`, `mse = 0.000175`, `rmse = 0.0132`, `max_abs = ~0.03`
- `cosine_similarity = 0.8782` (non-null; zero tensor would be null)
- `weight_only_compression_ratio = 8.0` (BF16 2byte / packed 0.25byte), `effective = 7.998` inc. scale 4 bytes
- All finite, no NaN/Inf

### Accounting (whole-model estimate via header-only)

- `ideal_ternary_bits = quant_params * log2(3) ≈ 1.585 * N` (alphabet upper bound, not Shannon)
- `packed_weight_bits = Σ ceil(N_i/4)*8` per-tensor sum (NOT total ceil) — bug fixed
- Whole-model: quantizable 205 tensors, `packed_weight_bits` Σ, `byte_padding_bits`, `scale_overhead_bits = 205*32`, `excluded_original_bits` remains BF16

### Determinism / Reproducibility

- Repeat `quantize→pack` on same tensor → identical `codes/scale/packed` PASS
- `inspection_fingerprint` unchanged by Phase 2 (`ternary_estimator` is separate hash) — `test_fingerprint_payload_unchanged` PASS
- `header-only inspect` still no torch import via `accounting` — subprocess PASS

### Failures

- Pre-vectorized pack loop timed out on 9M (Python for-loop) — fixed by vectorized `flat+1` and byte stacking (9M → ~30ms)
- Initial `torch.equal(unpacked, codes)` failed due to shape mismatch (unpacked 1D vs codes 2D) — fixed to flattened compare

---

## Interpretation

- ✅ G1–G10 PASS — AbsMean FP32, dequant, packing, accounting, real tensor roundtrip all validated.
- ✅ Phase 1 inspect/benchmark no regression (pytest 100+ green, ruff/mypy green).
- ❌ No quality claim — tensor metrics ≠ LM quality; Phase 3 will do sensitivity/group-wise.
- Deferred: group-wise packing, full-model fake-quant inference, `compare` CLI, GGUF.

---

## Next Experiment

**Phase 3 — Model-Level Ternarization** → see next entry (CLOSED 2026-08-21).

---

# Experiment: Phase 3 — Model-Level Ternarization (bounded sharded fake-quant)

## Metadata

- Run IDs: `runs/e2b-naive-pt` (`039127a2`, per_tensor) and `runs/e2b-naive-g128-rw` (`5af24de5`, per_group G128 LD-RW)
- Date: 2026-08-21 17:53–17:57 JST
- Git commit: unknown (manual acceptance)
- Branch: unknown
- Agent/operator: manual acceptance (CPU, Windows 11, uv 0.12.5, python 3.12.9, torch 2.13.0+cpu, transformers 5.15.1)
- Model: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- Model revision: `6befbaca7398925921802abd1f277b495b78b738` (10.2GB, 1951 tensors, 205 quantizable)
- Hardware: AMD64 / 91GB RAM / cuda_available=false (CPU-only)
- Seed: 42
- Benchmark runs: `runs/e2b-naive-pt-bench` (`06179050`, 5.0s avg 6.23 tok/s) and `runs/e2b-naive-g128-bench` (`22c2a0e9`, 10.2s avg 6.25 tok/s)
- Compare runs: `runs/2026-08-21_...-016` (baseline vs pt) and `...-017` (baseline vs g128)

---

## Objective

Prove `whole-snapshot fake-quant` is bounded-memory, sharded, loadable and benchmarkable: per_tensor and per_group G128 LD-RW both produce HF sharded snapshots that load via `AutoModelForMultimodalLM` and complete `smoke v1` without NaN/crash, with `protocol True` and `result observational`.

---

## Configuration

```yaml
# per_tensor
quantization:
  method: naive
  codebook: [-1,0,1]
  scale_granularity: per_tensor
  grouping_scheme: last-dim-rowwise-v1
  group_size: 128  # inactive when per_tensor
  target: {attention: true, mlp: true}

# per_group G128 RW
quantization:
  scale_granularity: per_group
  grouping_scheme: last-dim-rowwise-v1
  group_size: 128
```

Benchmark: `suite smoke v1, thinking false, do_sample false, num_beams 1, max_new_tokens 64, warmup true, dtype bf16 exact, device auto→cpu`.

---

## Results

### Quantization (2 snapshots)

| Snapshot | scale_granularity | quantizable | total_groups | shards | tensor_payload | shard_file | snapshot_total | content FP (short) |
|---|---|---:|---:|---:|---:|---:|---|---|
| `e2b-naive-pt` | per_tensor | 205 | 205 | 12 | 10,208,596,934 | 10,208,849,294 | ~10.3GB inc. tokenizer | `sha256:8e4f8f2d...` |
| `e2b-naive-g128-rw` | per_group LD-RW | 205 | 14,340,096 | 12 | 10,208,596,934 | 10,208,849,294 | ~10.3GB | `sha256:5e6e8885...` |

- Per-tensor `scale_overhead 205*32`, per_group `14.3M*32` (=458M bits) vs `packed_weight_bits` same (per-tensor sum). `packed_estimate.estimated_total_bytes 6,996,416,250` (hypothetical 2-bit packed, not fake-quant size). `fake_quant_artifact` vs `packed_estimate` correctly separated.
- Per-tensor total_groups =205, per_group =14.3M (each `[8,4]` with G2 →16 groups, `[6144,1536]` with G128 →73,728 groups).
- `per_tensor` entries: `scale_fingerprint sha256:…` + `scale_stats {min,max,mean,std(correction=0)}` + `num_groups` + `zero_ratio` + `mae`; no `scales` array (JSON would be 100s MB otherwise). `std` for single scale =0 (not NaN).
- Conversion bounded: peak `O(largest tensor + 512 MiB)` (~805 MB embed + shard), never `O(model)`. `flush-before-add` and single>max over-size shard allowed verified via mock tests. HF `blobs/` symlink dereferenced correctly (test with sibling blobs mock).

### Benchmark (smoke v1, 5 prompts)

| Run | protocol FP | result FP | same as baseline? | output tokens | avg latency ms | load ms |
|---|---|---|---|---:|---:|---:|
| baseline `907fbb73` | `11ac43e...` | `907fbb73...` | — | 160 | ~6300 | ~610 |
| `e2b-naive-pt` `499d390...` | `11ac43e...` | `499d390...` | different (observational) | 156 | 5005 | 699 |
| `e2b-naive-g128` `66e5977...` | `11ac43e...` | `66e5977...` | different | 320 | 10237 | 641 |

- `protocol_match True` for both quantized vs baseline (suite/version/prompts/thinking/generation without model id, so fake-quant model correctly matches baseline protocol). `result_match False` (all 5 prompts token IDs differ — observational, NOT pass/fail per M8).
- No NaN/crash, `outputs finite`, `benchmark.json` valid, `warmup True`.

### Compare

- `compare baseline vs pt` → `protocol True`, `result different`, per-prompt `same False` for all 5, `result_is_pass_fail False` (observational). Same for `baseline vs g128`.
- Content fingerprints differ between per_tensor and per_group (`8e4f...` vs `5e6e...`) and vs source.

### Failures (fixed during Phase 3)

- Initial `safetensors.torch.save_file(dict_of_all_tensors)` would be O(model) — fixed to bounded sharded writer.
- `list[Tensor]` per-group loop would be slow and ambiguous grouping — fixed to LD-RW vectorized.
- `bytes(untyped_storage())` for BF16 sha took 22s per 8MB tensor — fixed to `view(torch.uint8).numpy().tobytes()` 2ms (P0).
- `group_size` from config implying granularity — fixed to explicit `scale_granularity`.
- `result_fingerprint != baseline` as gate — fixed to observational (M8).

---

## Interpretation

- ✅ M1–M13 PASS — Whole-model fake-quant is bounded, sharded, deterministic, loadable, benchmarkable, comparable.
- ✅ `inspect→quantize→benchmark→compare` pipeline now stable; `v0.1 Ternary Hello World` CLOSED.
- ❌ No quality claim — smoke is behavioral, not quality. Perplexity/MMLU/Japanese HQ are Phase 5.
- Deferred: per_group group-size sweep (32/64/256), MLP vs Attention sensitivity, mixed precision, calibration.

---

## Next Experiment

**Phase 4 — Calibration / Reconstruction** — teacher/student activation capture, reconstruction loss, learnable scales.

---

## 2026-09-22 — P3 matched validation controls and P4 soft-to-hard screen

### Metadata

- Branch/commit at start: `feat/calib-threshold` / `d37ace6c3e75291048a7bd12534618fe7b6d4059`
- Model/revision: `google/gemma-4-E2B-it-qat-q4_0-unquantized` / `6befbaca7398925921802abd1f277b495b78b738`
- Local source: `D:\AI\llm model\gemma-4-E2B-it-qat-q4_0-unquantized`
- Targets: canonical 205 Linear modules, per-group G128, BF16
- Hardware: AMD Radeon RX 9070 XT; PyTorch `2.13.0+rocm10.0.0`; HIP `7.15.26333`
- Calibration screen: WikiText pinned revision, 8 samples, sequence 128, 10 steps/module, Adam, seed 42, held-out 25%, microbatch auto (selected 6)
- Evaluation: validation only, max length 128, stride 64, greedy 64-token generation
- Quality protocol fingerprint: `f2daec8d279f0c07486a2e8377b69a39400f76761f08723698af4af02e3a973e`
- Quality dataset fingerprint: `a0b11176d3632c939e1912daa7076c585d7006f47af7dcbcfdc0fc0ad1ed648f`
- Frozen dataset file SHA-256: `81436b92856dd1a712d5340a2a8b4f9a358af235621ce821a2637c6a786fd899`

The frozen data is deterministically selected from WikiText `b08601e...`, SQuAD `7b6d24c...`, and JGLUE/JSQuAD `7f983b6...`. Calibration, validation, and test pass exact and character-5gram MinHash disjointness checks. The test split was not executed.

### Infrastructure evidence

- Production `quality` runner and CLI record provenance, split/protocol/data fingerprints, actual BF16/device, language PPL, exact instruction score, collapse diagnostics, and schema-v2 response text plus UTF-8 SHA-256 for direct audit.
- Quality dry-run now validates the complete frozen dataset contract and prints fingerprint `a0b11176d3632c939e1912daa7076c585d7006f47af7dcbcfdc0fc0ad1ed648f` without creating the requested output directory; malformed datasets fail before model loading.
- `compare` fails closed on missing/malformed quality JSON, one-sided reports, protocol/data mismatch, schema-v2 protocol-fingerprint self-inconsistency, response-hash inconsistency, detached PPL summaries, model-revision mismatch, actual-dtype mismatch, and actual-device mismatch; the CLI displays these matched contracts.
- Mid-module resume now fails closed if `current_raw_param`, `optimizer_state`, or CPU `rng_state` is absent (and also requires the current threshold when threshold learning is active); valid tiny interrupted runs still match their uninterrupted hard snapshots.
- Explicit `--init-from` now rejects missing checkpoints, corrupt/incomplete state, and source/target/quantization/dtype mismatches instead of silently continuing from a fresh initialization. Warm-start cache reuse validates every shard size and SHA-256 first; a corrupt optional cache is recaptured, while cache-manifest persistence failure stops capture and a complete scale checkpoint still warm-starts threshold calibration successfully. Materialize-only requires the resume contract and fails if its final report cannot be persisted.
- Actual ROCm allocator OOM at 480 MiB under a 3% cap was followed by successful BF16 backward and Adam; see `runs/rocm-oom-recovery-20260922-v1/report.json`.
- `runs/p3-scale-screen-v1` was interrupted after durable `step_02050.pt`, resumed, reused the activation cache, and completed hard materialization for all 205 targets.
- P4 schedule/config/checkpoint/materialize are production-connected; final reports say `assignment.final_state=hard`.
- `runs/p0-preflight-20260922-rx9070xt-v14/preflight.json` was executed from outside the repository after the final acceptance-plan update and passes. It gates the exact 40-hex canonical revision, canonical `model.safetensors` SHA-256 `33fe0cece08fb527ffefbd1a3a9ce73bd71073727993a283506293e5c6bf0137`, and exact 205-name inventory fingerprint `5c810d4f7f00f8ec3569b423504e235630f2a1d880d6d42350639cf6fc11fe2a`, while retaining all six current protocol hashes and Git state independently of the invocation directory. Separating canonical inspection identity from the local source locator restores the known path-independent inspection fingerprint `sha256:07fe44eae504218937187b75826a4c780cba2f76a9145fdc38f7efdd4d3a7b01`.

### P3 matched controls — saved/reloaded validation

| Control | General PPL | Japanese PPL | Instruction exact | Collapse | Decision |
|---|---:|---:|---:|---:|---|
| BF16 | 1,214.61 | 2,229.70 | 62.5 | 0 | reference |
| naive G128 | 287,747.46 | 212,961,126.33 | 0.0 | 8 | FAIL |
| scale-only | 281,853.76 | 160,282,304.67 | 0.0 | 8 | FAIL |
| threshold | 315,223.93 | 140,044,860.87 | 0.0 | 8 | FAIL |

Scale-only improved Japanese PPL relative to naive but still collapsed every instruction case and remained orders of magnitude behind BF16. Threshold improved Japanese PPL further but regressed general PPL. Positive components of the preregistered composite do not override mandatory collapse/regression gates.

All seven quality reports were rerun into `quality-*-validation-v3` with report schema v2. Protocol fingerprints, dataset fingerprints, summaries, and generated responses match the preceding reports exactly, and all 56 stored response hashes recompute correctly. The newly embedded instruction-split audit has fingerprint `a982aa563b50ec0381e14193e78562127af9490f6a82e9c5773cba7dfd0cd432`; maximum validation/test MinHash similarity is `0.1484375`, below the `0.9` rejection threshold. An independent SHA-256 audit of `chat_template.jinja`, model/generation/processor/tokenizer configs, and `tokenizer.json` gives the same interface fingerprint `f53f74ea4996fd9802bc42a03a42cfc9500ec84447653f5aed10bf8b39d7b47b` for BF16 and all six candidate snapshots; see `runs/p3-model-interface-audit-v1/audit.json`. BF16 returned ordinary answers such as `distributed adaptive message block switching`; the six ternary candidates returned only punctuation-like strings on every instruction item (for example 64 periods in the scale-only run). The collapse result is therefore inspectable from the saved response, not only from aggregate counters. The `compare-bf16-*-validation-audit-v3` reports have matching schema, protocol, data, model revision, actual dtype, and actual device, with `accepted=false` for every candidate.

### P4 hard-snapshot screen — no modulation

| Schedule | Final recon | Held-out before → after | General PPL | Japanese PPL | Instruction | Collapse | Hard code Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear | 1.454211 | 1.472153 → 1.439834 | 301,165.69 | 185,326,217.18 | 0.0 | 8 | 0.0 |
| cosine | 1.454181 | 1.472153 → 1.439804 | 298,709.06 | 190,855,174.30 | 0.0 | 8 | 0.0 |
| exponential | 1.453746 | 1.472153 → 1.439399 | 302,290.56 | 185,735,594.41 | 0.0 | 8 | 0.0 |

Each schedule used temperatures `1.0 → 0.05`, a final 10% hard region (one hard step in the 10-step screen), and zero logit bias `0.0`. All final content fingerprints differ because scales changed, but none changed the hard ternary assignment. Exponential had the best local reconstruction result while cosine had the best general PPL; all failed the final hard quality gate. There is no winner, so the preregistered winner-plus-zero-bias row was not run.

Post-run code-path diagnosis established that hard-code invariance is structural: only reconstruction scales are optimizer parameters; final hardening uses the frozen source weight and frozen reference scale at the fixed midpoint. Temperature and zero-code bias only affect the soft expectation used while fitting scales. The saved hard assignment therefore cannot change under this implementation, independent of schedule length. The dominant residual is also localized: layer 0 attention `q_proj` contributes about 47.2% of aggregate held-out squared error in every learned candidate, with held-out MSE remaining about 291. A larger run of the same parameterization is not authorized or scientifically motivated.

Across learned candidates, `q_proj` accounts for 86.47% of aggregate held-out squared error. A direct `sign(fake_quant_weight)` comparison of the saved scale-only and threshold snapshots reproduced the recorded global hard-code delta: 166,481 / 1,835,532,288 (`9.0699031e-5`). By projection, changed-code counts were 50,456 `up_proj`, 47,484 `gate_proj`, 38,458 `down_proj`, 15,395 `q_proj`, 12,422 `o_proj`, 1,248 `v_proj`, and 1,018 `k_proj`. The dominant layer-0 `q_proj` changed only 343 / 3,145,728 codes (`1.09037e-4`). This strengthens the bounded option-B proposal: first prove materially trainable hard assignment on that single dominant module rather than repeat another 205-target schedule.

### Resources and decision

- Candidate/control calibration directories initially consumed 75.03 GiB. The independent P2 uninterrupted run raised the seven heavy active runs to 88.016 GiB, still below the 96 GiB (80% of 120 GB) stop threshold; no further full snapshot candidate was started.
- Threshold/P4 reports recorded about 1.1–1.2 GB peak VRAM and zero OOM retries; the resumed scale report's zero peak is not treated as valid peak telemetry.
- Materialization logs observed process RSS up to about 18.6 GB while streaming the oversized embedding tensor and 512 MiB shards.
- Approximate calibration wall times were 6.3–7.7 minutes per uninterrupted learned candidate; the interrupted/resumed scale run covered about 15.8 minutes between config and final metrics.

### P2 full-target identity follow-up

`runs/p2-scale-identity-v2` completed an independent uninterrupted 205-target scale-only run. Compared with the existing terminal-checkpoint-resumed `runs/p3-scale-screen-v1`, the output-path-normalized config, complete loss history, final scale fingerprint, calibrated content fingerprint, and SHA-256 of all 22 materialized snapshot files match exactly. The uninterrupted snapshot totals 10,242,544,948 bytes. Evidence is stored in `identity-uninterrupted.json` and `p2-identity-comparison.json` inside the new run. Mid-optimizer identity remains proven on the tiny real path; the full-target pair specifically proves terminal-checkpoint resume and materialization identity.

A final readback audit verified all 205 state-file sizes and SHA-256 values in each of the six learned full-target schema-v3 checkpoints, exact equality between manifest modules and resume-contract targets, terminal cursor `205:0`, and final-hard reports. All 9,840 train/held cache files across those runs, totaling 18,585,040,944 bytes, were rehashed successfully; in every run the train/held contract fingerprints agree and the checkpoint cache fingerprint equals both cache manifests. Each materialized `quantization.json` also matches the calibration content fingerprint, reports 205 quantized / 1,951 total tensors, and references 12 model shards that are all present.

The complete 32-file materialized trees (22 top-level snapshot files plus 10 copied Hugging Face cache-metadata files) were byte-hashed into canonical sorted `{name,size,sha256}` aggregate digests. `p3-scale-screen-v1` and `p2-scale-identity-v2` both equal `3ed297b77bcb97b4181cd858161e927ccb2cf0f0b4d185c8e7808eaec3d79e86`. Threshold is `c3c3fd6076c9bb71b988242017da783e93f65f69812b8e37bd340254e0df6e31`; soft linear/cosine/exponential are `4dc912c3ae801010c19df62e05643e3847a75533f7cd1f0bbccfc128f7889f30`, `3769a0b76fd13877f758a90727f6128b17e0ce20b5e0740fe9edfbdb97d94c9d`, and `1f153b9d5d93bde648a599d51480b6fcd51a5e452850072e87f8542c919e0bd9`.

Decision: P3 evaluation infrastructure is accepted, but no ternary control is quality-acceptable. P4 integration is accepted as engineering work, while the P4 research gate fails. Keep the test split sealed; do not open modulation, P5, P6, or P7 based on these candidates. Diagnose unchanged hard assignments and whole-model collapse on a bounded validation-only design first.

Protocol follow-up: schema-v2 `model.id` currently records the loaded snapshot locator, so it differs by design between the BF16 source and each materialized candidate. This screen therefore relies on exact revision matching plus the independent six-file interface fingerprint audit. Before P7, the quality schema and compare contract should embed and match a stable immutable source-model/interface identity rather than treating locator equality as model identity.

Compatibility follow-up: `compare` retains its tested ability to read two schema-less legacy quality summaries. Such inputs lack the schema-v2 response, protocol-body, and runtime-integrity fields and were not used for this screen. P7 needs an explicit compatibility decision to reject legacy reports or force them to a non-accepting result.

---

## Template (copy for next experiment)

## Metadata

- Run ID:
- Date:
- Git commit:
- Branch:
- Agent/operator:
- Model:
- Model revision:
- Hardware:
- OS:
- Python:
- PyTorch:
- Seed:

## Objective

## Hypothesis

## Configuration

```yaml
model:
quantization:
calibration:
benchmark:
```

## Calibration Data

## Evaluation Data

## Results

### Quality

| Metric | Baseline | Experiment | Delta |
|---|---:|---:|---:|

### Japanese

### Efficiency

### Quantization Diagnostics

## Qualitative Observations

## Failures / Errors

## Interpretation

## Next Experiment
