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
