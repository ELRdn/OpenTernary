---
description: "What ternary quantization is, how OpenTernary computes codes and scales, per-tensor vs per-group, fake-quant vs packed storage, and bit accounting."
---

# Ternary quantization

## The idea

A language model's weights are usually stored as 16-bit floating-point numbers. **Ternary** quantization
replaces each weight with one of only three values:

$$
w \;\longrightarrow\; \hat w \in \{-s,\; 0,\; +s\}
$$

where \(s\) is a **scale** shared by a whole tensor or by a small group of weights. What you store per weight is
just a **code** \(q \in \{-1, 0, +1\}\), a *trit*. The scale is stored once per tensor or group.

This is attractive because:

- **Memory:** a trit carries \(\log_2 3 \approx 1.585\) bits of information, against 16 for BF16, so the
  weights could in principle be about 10× smaller.
- **Compute:** multiplying by −1, 0 or +1 needs no multiplications, only additions, subtractions and skips.
  Specialised kernels can exploit that.

The price is accuracy. A model trained in 16 bits was not designed to work with three values per weight. Models
*trained* to be ternary from the start, such as BitNet b1.58, can work well. Converting an existing model
**after** training, which is what OpenTernary studies, is much harder: naive conversion usually destroys the
model's abilities. OpenTernary exists to measure that loss honestly and to try to recover from it.

## How a weight becomes a trit (AbsMean)

OpenTernary's baseline method is **AbsMean**. For a tensor \(W\) (or one group of it):

$$
s = \operatorname{mean}\lvert W \rvert, \qquad
q = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{W}{s}\right), -1, +1\right), \qquad
\hat W = q \cdot s
$$

- The mean is accumulated in FP32.
- `round` is **round half to even**, PyTorch's `torch.round`. In practice a weight becomes 0 when
  \(\lvert w \rvert \le 0.5\,s\) and \(\pm1\) otherwise.
- If \(s = 0\), meaning all weights are zero, the codes are all zero. OpenTernary uses an explicit branch here
  instead of adding a small epsilon to \(s\), so the scale is exactly what the formula says.
- Empty tensors, non-floating-point tensors and tensors containing NaN or Inf are rejected with an error, never
  silently passed through.

A worked example, with four weights in one group. Their scale is
\(s = (0.90 + 0.05 + 0.30 + 1.20) / 4 = 0.6125\):

| | \(w_1\) | \(w_2\) | \(w_3\) | \(w_4\) |
|---|---:|---:|---:|---:|
| weights \(W\) | 0.90 | −0.05 | 0.30 | −1.20 |
| \(W / s\) | 1.47 | −0.08 | 0.49 | −1.96 |
| code \(q\) | <span class="ot-trit plus">+1</span> | <span class="ot-trit zero">0</span> | <span class="ot-trit zero">0</span> | <span class="ot-trit minus">−1</span> |
| \(\hat W = q\,s\) | 0.6125 | 0 | 0 | −0.6125 |

Small weights (−0.05, and 0.30, which is just under \(0.5\,s\)) become zero. The large ones keep only their sign
and share one magnitude. All the information in a weight's exact size is gone. That is why calibration
([next section](calibration.md)) tries to choose better scales and thresholds.

## Per-tensor or per-group scales

| `--scale-granularity` | One scale for… | Storage overhead (FP32 scale) | Fidelity |
|---|---|---|---|
| `per_tensor` (default) | the whole tensor | negligible | low: one magnitude for millions of weights |
| `per_group` | each group of `--group-size` consecutive weights in a row | \(32 / G\) bits per weight (0.25 at \(G = 128\)) | higher: each group gets its own magnitude |

Groups are formed by the **`last-dim-rowwise-v1`** scheme. The tensor is viewed as rows along its last dimension,
and each row is split into consecutive groups of \(G\) values. If a row's length is not a multiple of \(G\), the
last group of that row is shorter. If \(G\) is at least the row length, every row is one group, so `per_group`
never silently becomes `per_tensor`.

```text
row 0: [ g0: 128 values ][ g1: 128 values ] … [ g11: 128 values ]   ← 1536 values, G = 128
row 1: [ g12 ]           [ g13 ]            … [ g23 ]
…
```

For Gemma 4 E2B, per-group G128 gives about 14.3 million groups. Their scales are 57 MB in FP32. The
per-tensor version of the same model has 205 scales.

## What gets quantized

By default, the **linear layers of attention and MLP blocks**:

- attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

The **embeddings**, the **output head**, **norms** and **biases** stay in their original precision, as do
model-specific parts such as Gemma's per-layer embeddings and multimodal towers. The model **adapter** decides
which tensors qualify, and it never guesses about unknown tensors. You can:

- switch whole groups off: `quantization.target.attention: false` or `quantization.target.mlp: false`;
- keep individual tensors in full precision: `quantization.mixed_precision: {"model.layers.0.self_attn.q_proj.weight": preserve}`.

[Inspect and plan](../guides/inspect-and-plan.md) shows how to list the exact names before you quantize.

## Fake-quant, packed and native

The same ternary weights can be stored in three ways:

| Form | What is stored | Size | Runs in | Status in OpenTernary |
|---|---|---|---|---|
| **Fake-quant** | \(\hat W = q\,s\) as ordinary BF16/FP32 numbers | same as the original | any Transformers code | what `quantize` and `calibrate` produce |
| **Packed** | 2-bit codes + scales + the untouched tensors | ~2 bits per quantized weight + overhead | nothing yet: unpack first | `export --format ternary-packed`, lossless round trip |
| **Native kernel** | packed codes executed directly by a ternary matmul | small | a specialised runtime | **not provided** |

Fake-quant is deliberately inefficient. It lets you measure the *quality* of ternary weights with standard tools,
before anyone writes fast kernels. Packing proves that the weights really are ternary and shows the real file
size. Speed and memory gains need native kernels, which OpenTernary does not have yet.

### The 2-bit packing format

Each code takes 2 bits, four codes per byte, most significant first:

```text
code  →  bits      byte = (c0 << 6) | (c1 << 4) | (c2 << 2) | c3
 −1   →  00
  0   →  01        tail padding must be 00
 +1   →  10        length = ceil(N / 4) bytes
       11 = reserved (rejected when unpacking)
```

`unpack(pack(q)) == q` exactly. OpenTernary checks this, and a restored safetensors export is byte-for-byte equal
to the fake-quant snapshot it came from. Packing is refused when the input cannot be reconstructed exactly from
codes and scales, for example a snapshot calibrated with thresholds that would need extra information.

## Counting bits honestly

"1.58-bit model" is often used loosely. OpenTernary keeps the quantities separate:

| Quantity | Per quantized weight | Meaning |
|---|---|---|
| Ideal alphabet bits | \(\log_2 3 \approx 1.585\) | the information needed to tell three equally likely states apart; a skewed code distribution needs less in theory |
| Logical 2-bit | 2 | what a fixed-width 2-bit code uses |
| Packed bits | \(8 \lceil N/4 \rceil / N\) | 2 plus a few bits of byte padding per tensor |
| Scale overhead | \(32 / G\) (per group) | FP32 scales, counted separately |
| Excluded tensors | — | embeddings, head, norms… kept in full precision, counted separately |

The [first run](../getting-started/first-run.md) shows all of them at once. 294,912 quantized weights at 2 bits
take 73,728 bytes, the 4,608 FP32 scales take 18,432 bytes, and the 25,984 FP32 weights that are not quantized
take 103,936 bytes. The total, **196,096 bytes**, is exactly the CLI's `Packed estimate bytes`. In that tiny
model the parts that are *not* ternary dominate. In a large model they matter less, but they never vanish.

## Other backends

The `ternary` backend is the core of OpenTernary. A **TorchAO** backend (`--backend torchao --scheme
int8-weight-only --weight-dtype int8`) is included for comparison with an established 8-bit method; it requires
`torchao==0.18.0`. Third-party backends can be added as [plugins](../reference/plugins.md). `openternary backends
list` shows what is installed and what each backend has been verified to do.

## Passes

Before quantizing, optional **passes** can transform weights in order:

- `clip` with `max_abs`: clamps weights to \([-m, m]\). This changes the values. It is *not* a way of preserving
  quality.
- `noop`: does nothing (useful for testing the pass pipeline).

Passes are checked for order and duplication. Transformations that would need runtime changes, such as
rotations, are rejected as unsupported.

```yaml
quantization:
  passes:
    - name: clip
      options: {max_abs: 2.0}
```
