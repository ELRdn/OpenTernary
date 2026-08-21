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
          ├──── capture target activation
          │
Calibration Input
          │
          └──── Quantized Model
                    │
                    └──── capture reconstructed activation
                               │
                               ▼
                            Loss
                               │
                               ▼
                         Optimizer
                               │
                               ▼
                     Quantization Params
```

Initially optimize small quantization-related parameters instead of every original model weight.

Potential trainable parameters:

- group scale
- threshold
- modulation terms
- soft ternary parameters

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
