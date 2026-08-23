# OpenTernary Architecture

## 1. Design Goal

Keep three concerns independent:

```text
Model Architecture
       ×
Quantization Method
       ×
Evaluation / Export
```

Adding a new model should not require rewriting quantization math.

Adding a new quantization method should not require rewriting model adapters.

---

## 2. High-Level Data Flow

```text
Model ID / Path
      ↓
Model Adapter
      ↓
Module Inventory
      ↓
Quantization Plan
      ↓
Quantizer
      ↓
Fake-Quant / Quantized Model
      ↓
Calibration (optional)
      ↓
Benchmark
      ↓
Comparison
      ↓
Export
```

---

## 3. Package Layout

```text
openternary/
├─ cli/
│  ├─ inspect.py
│  ├─ quantize.py
│  ├─ calibrate.py
│  ├─ benchmark.py
│  ├─ compare.py
│  └─ export.py
│
├─ adapters/
│  ├─ base.py
│  └─ gemma4.py
│
├─ quant/
│  ├─ accounting.py   # stdlib — ideal/packed/scale accounting (header-only safe)
│  ├─ ternary.py      # torch — AbsMean per-tensor quantize/dequantize (FP32, zero-branch)
│  ├─ grouping.py     # torch — LD-RW per_group vectorized, per-row tail, zero-branch, interleaved scales
│  ├─ fake_quant.py   # torch — bounded sharded writer (512 MiB, flush-before-add, blobs dereference)
│  ├─ packing.py      # torch — 2-bit pack/unpack (canonical 00 padding, 11 reserved)
│  └─ metrics.py      # torch — MAE/MSE/cosine(null)/ratios
│
├─ calibration/
│  ├─ capture.py
│  ├─ losses.py
│  ├─ optimizer.py
│  └─ runner.py
│
├─ benchmark/
│  ├─ runner.py
│  ├─ suites.py
│  └─ metrics.py
│
├─ export/
│  ├─ fake_quant.py
│  ├─ packed.py
│  └─ gguf.py
│
├─ experiment/
│  ├─ run.py
│  ├─ metadata.py
│  └─ compare.py
│
└─ utils/
```

---

## 4. Model Adapter Contract

Each model adapter should expose concepts rather than raw naming assumptions.

Pseudo-interface:

```python
class ModelAdapter:
    def load(self, config): ...
    def architecture_info(self): ...
    def iter_quantizable_modules(self): ...
    def iter_sensitive_modules(self): ...
    def module_role(self, name): ...
    def replace_module(self, name, module): ...
    def prepare_text_input(self, text): ...
```

Gemma-specific details belong in `adapters/gemma4.py`.

---

## 5. Quantizer Contract

Phase 2 canonical baseline is **per-tensor AbsMean**:

```text
scale = mean(abs(W))  # FP32 accumulation: W.to(float32).abs().mean()
if scale == 0:
    Q = zeros_like(W)
else:
    Q = clamp(round(W / scale), -1, +1)  # device-preserving, ties-to-even
W_hat = Q * scale
```

Phase 3 adds **per-group LD-RW** (canonical `last-dim-rowwise-v1`):

```text
W.reshape(-1, last_dim) -> each row split into groups of G
for each group g:
    scale_g = mean(abs(g))  # FP32
    if scale_g==0: Q_g=0 else Q_g=clamp(round(g/scale_g),-1,1)
W_hat = Q * scale_g (per-group broadcast)
```

- Vectorized: full groups `reshape(-1, G).mean` + tail per row, no `list[Tensor]` per-group loop.
- Zero-branch per group, no `eps`, `scales` are `float32` per actual group, `codes` same shape as `W`.
- `G >= last_dim` → `num_groups = num_rows` (not 1), `per_group != per_tensor`.

Implementation: `openternary.quant.ternary.quantize_absmean` / `dequantize` and
`openternary.quant.grouping.quantize_groupwise` / `dequantize_groupwise` with
`TernaryTensor` / `GroupwiseResult(codes: int8, scales: float32[groups], shape, group_size, grouping_scheme="last-dim-rowwise-v1")`.

* `scale + eps` is forbidden — zero is handled by explicit branch (per tensor and per group).
* `w.numel()==0`, non-floating, NaN/Inf are rejected loudly (ValueError).
* `torch.round` ties-to-even semantics is canonical and documented.
* Device-preserving: `codes.device == w.device`; no implicit `.cpu()` in math layer.

```python
class Quantizer:
    def analyze(self, tensor): ...
    def quantize(self, tensor): ...
    def dequantize(self, qweight): ...
    def statistics(self): ...
```

The initial ternary result conceptually contains:

```text
codes         # int8 ∈ {-1,0,1}
scale         # float (canonical storage: FP32)
shape
orig_dtype
scale_dtype
scheme/version
```

Do not tie this internal representation directly to GGUF.

---

## 6. Ternary Tensor Representation

Logical representation (Phase 2):

```text
codes ∈ {-1, 0, +1}  # int8 intermediate
scale per tensor     # FP32 canonical (Python float in API)
```

Physical storage (Phase 2 2bit-v1):

```text
mapping: 00=-1, 01=0, 10=+1, 11=reserved/invalid
byte: (c0<<6)|(c1<<4)|(c2<<2)|c3, tail padding MUST be 00
len(packed) == ceil(N/4), unpack(pack(Q))==Q bit-exact
```

Why `log2(3)≈1.585` is **ideal alphabet bits**, not measured entropy:

> `N * log2(3)` is the information required to distinguish three states under the full ternary alphabet (equal-probability upper bound). A skewed distribution (e.g. 80% zeros) has lower Shannon entropy; future `empirical_entropy` can be computed from histogram.

Why 2 bits physically: fixed-width byte alignment requires 2 bits per weight; `logical_2bit_bits = N*2`, `packed_weight_bits = ceil(N/4)*8`, `byte_padding_bits = packed - logical`. Scale overhead (`fp32` 32 bit/tensor) and excluded tensors are counted separately.

Fake-quant path (per-tensor and per-group):

```text
FP tensor
   ↓
ternary codes  # int8 {-1,0,1} per tensor or per group
   ↓
scale(s) (FP32, 1 or groups)
   ↓
dequantized FP tensor  # Q*scale (broadcast per group)
   ↓
cast back to original dtype (BF16 preserved)
   ↓
normal framework matmul
```

This path is deliberately inefficient but easy to validate. `quantize` is device-preserving; `pack` produces CPU uint8 bytes. `accounting` is stdlib-only so header-only inspect never imports torch.

Bounded sharded writer (Phase 3):

```text
HF snapshot (10GB, 1951 tensors)
  ↓ sorted keys, safe_open streaming
  for each tensor: quantize (per_tensor/per_group) → W_hat → shard buffer
  buffer + next >512 MiB ? flush before add : add
  single tensor >512 MiB ? single over-size shard allowed
  ↓ 12 shards + model.safetensors.index.json + copied tokenizer/processor (HF blobs symlinks dereferenced)
```

`content_fingerprint = sha256(canonical(sorted[{name,dtype,shape,sha256(raw)}]))` is MUST (deterministic content), file hash is SHOULD. `scale_fingerprint` is `sha256(float32 LE bytes)`, JSON stores `scale_stats` + `num_groups` but never full `scales` array. `O(largest tensor + max shard size)` bounded, never `O(model)`.

Packed deployment is a later optimization; `ideal_ternary_bits` vs `packed_weight_bits` vs `scale_overhead` (per_tensor `205*32` vs per_group `14.3M*32`) vs `excluded_original` are separated, `packed_artifact_actual_bytes` stays null until true packed file, `fake_quant_artifact {tensor_payload, shard_file, snapshot_total, shard_count}` is reported separately.

---

## 7. Calibration Architecture

Calibration must be optional.

```text
Teacher / Original Model
          │
          ├──── capture target activation (disk-backed sharded, per-layer load/release)
          │
Calibration Input (tokenizer → input_ids, wiki-tiny / synthetic, held_out_ratio=0.2, isdisjoint)
          │
          └──── Quantized Model (layer-local F.linear)
                    │
                    ├──── reference_scale (frozen AbsMean) ─┐
                    │                                        ├─→ threshold ─→ hard codes ─┐
                    │   threshold_ratio (learnable) ────────┘                             │
                    │                                                                      ▼
                    └──── reconstruction_scale (learnable) ─────────────────────→ w_hat = codes * scale
                               │                                                              │
                               └──── capture reconstructed activation                        │
                                              │                                              │
                                              ▼                                              │
                                           Loss (mse/l1, total_squared_error/total_elements) │
                                              │                                              │
                                              ▼                                              │
                                        Optimizer (Adam: {scales, lr} + {thresholds, threshold_lr}) │
                                              │                                              │
                                              ▼                                              │
                                   Quantization Params (scales + thresholds) ◄──────────────┘
                                              │
                                              ▼
                                   Checkpoint (step_*.pt: raw_params + raw_thresholds) → Bounded materialize
```

Initially optimize small quantization-related parameters instead of every original model weight.

### 7.1 Phase 4.1 — Learnable Scales (recon-scale)

- `codes` は固定（`quantize_absmean` / `quantize_groupwise` の `round` 由来 `±0.5*scale`）。
- `effective_scale = softplus(raw_scale) + eps`（`eps=1e-6`）、zero-group（`scale==0`）は `where(zero_mask, 0, ...)` で exact 0 固定。
- 初期化: `raw = inverse_softplus(orig - eps)` で `effective(step0) == orig`（`atol 1e-6`）。
- `steps` = 全 205 Linear の 1 full sweep、graph は per-module `backward()` 後に即解放。

### 7.2 Phase 4.2 — Learnable Thresholds (recon-threshold): threshold+scale 分離と clipped STE

Phase 4.1 の暗黙 `Q=clamp(round(W/scale))` ＝ `±0.5*scale` を、`threshold_ratio ∈ (eps, 1-eps)` で一般化する。

**分離原則 (threshold+scale separation):**

| 役割 | 値 | 学習 | 用途 |
|---|---|---|---|
| `reference_scale` | `mean(abs(W))` FP32, frozen | ❌ | `codes` 割当専用（`hard_threshold_codes` の分母） |
| `threshold_ratio` | `eps + (1-2eps)*sigmoid(raw_thr)`、`eps=0.01` | ✅ `raw_threshold` | `effective_threshold = threshold_ratio * reference_scale`、codes の 0/±1 境界 |
| `reconstruction_scale` | `softplus(raw_scale)+eps` | ✅ `raw_scale` | `w_hat = codes * reconstruction_scale` の振幅 |

- `reference_scale` は元重みの AbsMean を frozen で保持し、threshold による code 変化と scale による振幅変化を直交させる。
- `threshold_ratio` は `sigmoid` で `(0,1)` → `eps` で `(0.01,0.99)` に clamp、初期 `0.5` は `inverse_sigmoid(0.5)=0`（`raw_thr=0`）。
- per_tensor: 1 threshold / tensor、per_group: 1 threshold / group（scales と同数、LD-RW `last-dim-rowwise-v1` の per-row interleaved 順序に準拠）。zero-group は threshold も学習対象外（`zero_mask` で gate/surrogate を 0 にマスク）。
- `threshold_estimator = clipped-ste` のみ（Phase 4.2）。`threshold_granularity = per_group` 推奨、`threshold_init_ratio=0.5`、`threshold_lr` は未指定なら `lr` を流用。

**Hard forward:**

```text
thr_abs = reference_scale * threshold_ratio          # per_tensor: scalar broadcast, per_group: _expand_per_group
hard_gate = (abs(W) > thr_abs) & (~zero_ref_mask)
hard_code = sign(W) * hard_gate                      # int8 ∈ {-1,0,1}, zero_ref では常に 0
```

`group_size=None` なら per_tensor、`group_size=G` なら `reference_scale`/`threshold_ratio` を `_expand_per_group(shape, G)` で `shape` に broadcast。`hard_threshold_codes(W, ref, thr, group_size)` が正準。

**Clipped STE (Straight-Through Estimator):**

`hard_gate` は閾値で非微分なため、backward は clipped surrogate で近似する（Phase 4.2 最小 STE、temperature annealing は 4.3 で導入）。

```python
u = abs(W) / reference_scale
margin = u - threshold_ratio_expanded
surrogate = clamp(0.5 + margin / (2*ste_width), 0, 1)  # ste_width=0.1
gate_ste = hard_gate.detach() - surrogate.detach() + surrogate  # forward==hard, backward via surrogate
codes_ste = sign(W) * gate_ste  # float {-1,0,1}, requires_grad は threshold_ratio に流れる
```

- `ste_width=0.1`（`config.threshold_ste_width`）が線形領域の幅、`|margin| < ste_width` のみで `d surrogate / d thr = -1/(2*width)`、外側は `clamp` で 0。
- `_ClippedSTE(torch.autograd.Function)` は `forward=hard_gate, backward=grad_output`（`gate_ste = hard - surr.detach() + surr` のため `dL/d surr = grad_output`）。
- `ste_threshold_codes(W, reference_scale, threshold_ratio, group_size, ste_width)` が正準。per_tensor は scalar `ref`/`thr` を broadcast、per_group は `_expand_per_group` で展開、zero_ref は `surrogate=0` で grad 遮断。

**Runner 同時最適化:**

- per-module `raw_threshold`（`build_threshold_params(n, init_ratio, eps)`) を `all_threshold_params` に追加、`optimizer = Adam([{scales, lr}, {thresholds, threshold_lr}])`。
- 各 forward で `eff_thr = get_effective_threshold_ratio(raw_thr, eps)` と `weight_tensor`（bounded に都度 `_load_weight`）から `ste_threshold_codes` で codes 再計算 → `w_hat = codes_ste * eff_scale_expanded` → `F.linear(inp, w_hat)` → MSE。
- `threshold_enabled=false` なら Phase 4.1 と完全互換（`codes` 固定パス）。`step0` では `thr=0.5` で `hard_threshold_codes == quantize_absmean/groupwise` を assert。
- Checkpoint (`artifacts/checkpoint/step_*.pt`) に `raw_thresholds`/`threshold_eps`/`threshold_ste_width` を追加、`--resume` は `threshold_enabled` 不一致を loud error、`--init-from` は Phase 4.1 成果物からの warm start に使用（`--resume` とは別）。

**Bounded materialize:**

- `calibration.json` に `threshold_fingerprint_before/after`（`sha256(threshold_ratio bytes)`、`before` は全 `0.5`）、`threshold_ratio_{mean,min,max,std}`、`code_fingerprint_before/after`（hard codes の `sha256`）、`code_change_ratio`、`zero/positive/negative_ratio_before/after` を追記。
- 最終 `calibrated_snapshot` は `hard_threshold_codes` で確定した codes（threshold 反映済み）と `effective_scales` を `materialize_calibrated_snapshot` へ渡し、sharded bounded writer（`512 MiB` flush-before-add）で生成。`content_fingerprint` は threshold ありで scale-only と差異が生じることが期待値。
- `artifacts/threshold_metrics.jsonl` に per-module の `threshold_after_{mean,min,max}` と `code_change_ratio` を保存。`_run_materialize_only` も thresholds を復元。

Potential trainable parameters (updated):

- [x] group scale (`reconstruction_scale`, Phase 4.1)
- [x] threshold (`threshold_ratio` via clipped STE, Phase 4.2)
- [ ] modulation terms (deferred)
- [ ] soft ternary parameters (Phase 4.3 temperature annealing)

---

## 8. Benchmark Isolation

Benchmark code must not know how the model was quantized.

It receives:

```text
model interface
dataset/suite
generation config
```

and produces structured metrics.

This allows:

```text
BF16
Q8
Q4
Naive Ternary
Calibrated Ternary
```

to use the same evaluator.

---

## 9. Experiment Artifact

Every run receives a unique ID.

Example metadata:

```json
{
  "run_id": "e2b-naive-g128-001",
  "model": "google/gemma-4-e2b-it",
  "method": "naive_ternary",
  "group_size": 128,
  "seed": 42,
  "git_commit": "..."
}
```

---

## 10. GPU / CPU Policy

The core code should not assume CUDA where unnecessary.

However:

- training/calibration backends may initially target CUDA because ecosystem support is stronger,
- inspection, metadata, some PTQ, and CPU tests should stay hardware-neutral where practical,
- AMD support should be treated as an explicit compatibility target, not assumed automatically.

---

## 11. GUI Boundary

The future GUI must call:

```text
OpenTernary Python API
```

and never reimplement quantization logic.

Desired layering:

```text
GUI
 ↓
CLI / Python API
 ↓
Core
 ↓
Model framework/runtime
```
