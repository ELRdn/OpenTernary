# Gemma 4 E2B: learned rotation of 35 q projections

Date: 2026-09-24. Model: `google/gemma-4-E2B-it-qat-q4_0-unquantized` at revision `6befbaca7398925921802abd1f277b495b78b738`. Hardware: RX 9070 XT, ROCm PyTorch, BF16 inference. This is a mixed-precision research candidate; the canonical 205-target goal remains open.

## Method

- Select all 35 text-path `self_attn.q_proj.weight` tensors for G128 ternary quantization; preserve the remaining 170 canonical Linear targets in BF16.
- For each selected tensor, optimize one shared 128×128 Cayley orthogonal matrix for 50 steps on calibration activation pairs. Apply a normalized Hadamard map first, then the learned matrix, to both weight rows and runtime input blocks. Minimize teacher-output relative MSE with hard G128 codes in the forward pass.
- Use only `runs/p3-threshold-screen-v1/artifacts/activation_cache_train` and `_held` for learning and local checkpoint selection. The 35 local held-output relative MSE values all improved; their mean changed from 0.2180 to 0.0742. This local metric is not a model-quality gate.
- Freeze a manifest with the original BF16 source and SHA-256 for every rotation. `openternary quantize --rotation-manifest` writes the transformed ternary weights, all 35 input maps, quantization report, and file-integrity manifest. The standard `quality`/`benchmark` loader installs and validates the input maps.

## Saved and reloaded quality

The output is `runs/gemma4-q35-rotated-materialized-20260924/artifacts/snapshot`. It contains 35 ternary targets and 132,120,576 ternary parameters, or 2.5884% of the model's 5,104,298,467 parameters. The 10.24 GB snapshot retains BF16 fake-quant weights; this result does not demonstrate packed-size or native ternary-kernel speed gains.

| Frozen split | Metric | BF16 | Saved rotated q35 | Change |
|---|---|---:|---:|---:|
| v2 validation | English PPL | 1072.998 | 302.608 | −71.80% |
| v2 validation | Japanese PPL | 1360.994 | 376.831 | −72.31% |
| v2 validation | instruction exact match | 54.6875% | 54.6875% | 0 points |
| v3 test | English PPL | 1296.731 | 342.937 | −73.55% |
| v3 test | Japanese PPL | 1680.002 | 435.205 | −74.09% |
| v3 test | instruction exact match | 43.7500% | 45.3125% | +1.5625 points |

Both splits had zero flagged collapse outputs. BF16 and candidate reports match on the frozen data and generation protocol, source/interface fingerprint, model revision, actual dtype/device, and measurement environment. `compare --require-acceptance` passed on both saved-model comparisons: v2 composite 0.50438, v3 composite 0.52146. The v3 evaluation texts have zero exact context overlap with v1 and v2; v3 test was measured after selection on v2 validation.

The in-memory GPU prototype and CPU-materialized snapshot have small numerical differences (v2 candidate English PPL 301.708 vs 302.608 and instruction exact match 53.125% vs 54.6875%). The saved/reloaded snapshot is the authoritative result. The unquantized, all-35-rotated BF16 function check had logits relative MSE `7.06e-5` and matched top-1 tokens on its probe prompt.

## Same-GPU smoke speed

One `benchmark --suite smoke` run per model used the same RX 9070 XT, BF16, seed, five prompts, generation protocol, and one warmup. BF16 generated 160 output tokens at 32.04 token/s (mean 998.85 ms/prompt); the rotated q35 snapshot generated 154 output tokens at 24.77 token/s (mean 1243.40 ms/prompt). Observed throughput is 22.69% lower and mean prompt latency 24.48% higher. PyTorch allocator inference peak was 10,297,021,952 versus 10,299,315,712 bytes (+2.19 MiB). Different generated token counts and a single timing run limit the speed conclusion; this fake-quant runtime performs BF16 matrix multiplication plus 35 input transforms.

## Evidence and limits

- Frozen v2 dataset: `data/quality/gemma4-e2b-quality-v2.json` (local, not committed with the model); v3: `data/quality/gemma4-e2b-quality-v3.json`. The freeze script records pinned source revisions and exclusion hashes.
- Validation reports: `runs/gemma4-bf16-quality-v2-validation-20260924/quality.json`, `runs/gemma4-q35-rotated-quality-v2-validation-20260924/quality.json`.
- Independent test reports: `runs/gemma4-bf16-quality-v3-test-20260924/quality.json`, `runs/gemma4-q35-rotated-quality-v3-test-20260924/quality.json`.
- Artifact manifest: `runs/gemma4-q35-rotated-materialized-20260924/artifacts/snapshot/openternary-manifest.json`.
- Speed reports: `runs/gemma4-bf16-smoke-rx9070-20260924/benchmark.json`, `runs/gemma4-q35-rotated-smoke-rx9070-20260924/benchmark.json`.
- These 64-question exact-match splits are narrow. Passing them does not establish broad instruction quality, Japanese fluency, or a 205-target ternary model.
- Packed export is rejected for rotated weights until an input-transform-aware packed runtime exists.

## Reproduction

The local `runs/` artifacts and licensed frozen data are intentionally outside Git. See [CLI guide](../CLI.md) for the manifest and quantize commands. Use the same snapshot and source revision, the frozen data file, `--split`, `--max-length 128`, `--stride 64`, `--device cuda`, `--dtype bf16`, `--seed 42`, and `configs/gemma4-e2b.yaml` for comparisons.
