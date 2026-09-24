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

A further disjoint v4 validation on 2026-09-25 **failed** for the same saved snapshot: BF16 instruction exact match 57.8125%, rotated q35 43.75% (−14.0625 points); English PPL 1355.436→347.235 and Japanese PPL 1725.479→509.831, with zero collapse flags. The fixed gate rejects the instruction regression. This negative result supersedes any claim that the v2/v3 pass demonstrates stable quality across datasets. The v4 instructions were not used for the earlier candidate selection. Many misses are exact-match formatting variants, but some answers are wrong; the preregistered gate remains unchanged.

Applying learned rotations to all 205 canonical targets also failed on v4 validation after saving and reloading: English PPL 5114.567, Japanese PPL 2,328,674.750, instruction exact match 0%, and collapse 64/64. All 205 individual held-out module errors improved (mean relative MSE 0.2288→0.1048), which did not prevent model-level failure. The full snapshot is `runs/gemma4-all205-rotated-materialized-20260925/artifacts/snapshot`; its content fingerprint is `sha256:1e9205eefd53757f20b45ac2e9b032f8e800d1dffdf282c7502b36a7a42595c3`.

The in-memory GPU prototype and CPU-materialized snapshot have small numerical differences (v2 candidate English PPL 301.708 vs 302.608 and instruction exact match 53.125% vs 54.6875%). The saved/reloaded snapshot is the authoritative result. The unquantized, all-35-rotated BF16 function check had logits relative MSE `7.06e-5` and matched top-1 tokens on its probe prompt.

## Same-GPU smoke speed

One `benchmark --suite smoke` run per model used the same RX 9070 XT, BF16, seed, five prompts, generation protocol, and one warmup. BF16 generated 160 output tokens at 32.04 token/s (mean 998.85 ms/prompt); the rotated q35 snapshot generated 154 output tokens at 24.77 token/s (mean 1243.40 ms/prompt). Observed throughput is 22.69% lower and mean prompt latency 24.48% higher. PyTorch allocator inference peak was 10,297,021,952 versus 10,299,315,712 bytes (+2.19 MiB). Different generated token counts and a single timing run limit the speed conclusion; this fake-quant runtime performs BF16 matrix multiplication plus 35 input transforms.

## 200-step follow-up (2026-09-25)

Retrained the same 35 q projections for 200 steps each using the same calibration train and held caches. All 35 improved their local held-out relative MSE, with the mean changing from 0.2180 before training to 0.0687 at the selected checkpoints. A validation-only in-memory screen passed v4, then `quantize --rotation-manifest` saved a new 35-target G128 snapshot at `runs/gemma4-q35-rotated-200steps-materialized-20260925/artifacts/snapshot`. Its content fingerprint is `sha256:1b5964d4cdbf9382043fc34d286f234596b4bb9c680d547a386a74998e2c8140`; 12 fake-quant shards hold 10,208,849,294 bytes. The model was reloaded in a separate process for both quality splits.

| Frozen split | Metric | BF16 | Saved 200-step q35 | Change |
|---|---|---:|---:|---:|
| v4 validation | English PPL | 1355.436 | 317.108 | −76.60% |
| v4 validation | Japanese PPL | 1725.479 | 471.905 | −72.65% |
| v4 validation | instruction exact match | 57.8125% | 57.8125% | 0 points |
| v5 test | English PPL | 1274.036 | 320.301 | −74.86% |
| v5 test | Japanese PPL | 1418.493 | 409.527 | −71.13% |
| v5 test | instruction exact match | 59.375% | 56.25% | −3.125 points |

Both runs reported zero collapse flags and matched the control on dataset/protocol, model revision, device, and dtype. `compare --require-acceptance` passed on v4 (composite 0.5224) and **failed on the independent v5 test** (composite 0.5016) because the fixed instruction regression limit is 2 points. The v5 test was first opened after the 200-step snapshot passed v4 validation; its 384 context/instruction rows have zero exact text overlap with v1–v4. Do not select another candidate on v5.

Same-GPU smoke on the saved 200-step snapshot: BF16 31.66 token/s (160 output tokens, 1010.59 ms/prompt); candidate 24.75 token/s (168 output tokens, 1357.31 ms/prompt). Observed throughput was 21.83% lower. This is one timing run with different output counts, and still uses BF16 matrix multiplication plus input transforms rather than packed ternary execution.

Evidence: `runs/gemma4-q35-rotation-200steps-20260925/v4-screen.json`; `runs/gemma4-q35-rotated-200steps-quality-v4-validation-20260925/quality.json`; `runs/gemma4-bf16-quality-v5-test-20260925/quality.json`; `runs/gemma4-q35-rotated-200steps-quality-v5-test-20260925/quality.json`; `runs/gemma4-q35-rotated-200steps-compare-v5-test-20260925/compare.json`; and the paired `*-smoke-rx9070-20260925/benchmark.json` files.

## All-205 200-step screen (2026-09-25)

Retrained the remaining 170 canonical targets for 200 steps and combined them with the 35 q rotations above. All 205 local held-out errors improved, with mean relative MSE 0.2288→0.0981; all manifest files passed source/hash, finite-value, and orthogonality checks (maximum absolute orthogonality error `1.31e-6`). Before ternary rounding, applying all 205 transforms to BF16 weights and inputs preserved top-1 at all 26 tested token positions over three short prompts, with logits relative MSE `9.76e-5` to `3.06e-4`.

The **in-memory** all-205 hard G128 candidate still failed v4 validation: English PPL 3657.364 versus BF16 1355.436, Japanese PPL 1,549,400.591 versus BF16 1725.479, instruction exact match 0% versus 57.8125%, and collapse 64/64 versus 0. Every fixed quality gate failed. These numbers are a screening result, not a saved/reloaded model result. Since the screen failed decisively, no second 10 GB all-205 snapshot was materialized. The earlier 50-step all-205 saved snapshot remains the separate, authoritative evidence for save/reload feasibility and failure.

Evidence: `runs/gemma4-all205-rotation-200steps-manifest-20260925.json`, `runs/gemma4-all205-rotation-200steps-function-20260925.json`, and `runs/gemma4-all205-rotation-200steps-v4-screen-20260925.{jsonl,all.quality.json}`.

An in-memory three-prompt layer-drift probe showed the error begins early: relative MSE at text layer 0 was 0.127–0.179 and at layer 2 was 0.657–0.779 versus BF16. Final-logit relative MSE was 0.319–0.411. Thus the full-model failure is already visible in the first few quantized blocks, despite the unquantized rotation function check passing. This short probe localizes the issue; it does not establish which individual module or calibration remedy is sufficient. Evidence: `runs/gemma4-all205-rotation-200steps-layer-drift-20260925.json`.

Keeping the first 3, 8, or 15 text layers in BF16 still left the remaining 184, 149, or 100 canonical targets collapsed on v4 validation: each had instruction 0% and collapse 64/64. The corresponding Japanese PPL values were approximately 1.51 million, 1.26 million, and 245 thousand. The problem is distributed beyond the first blocks. Evidence: `runs/gemma4-all205-rotation-200steps-layer-ablation-v4-20260925.jsonl` and its per-candidate quality reports.

Local method probes on the same calibration-held cache found that alternating weight-MSE-optimal ternary scales lowered weight MSE but could **worsen** output error (`layer9 o_proj` 0.309→0.515; `layer34 o_proj` 0.338→0.460). Training positive scales against calibration activations reduced `layer9 o_proj` held-output relative MSE from 0.309 to 0.232 and `layer0 up_proj` from 0.190 to 0.131. A separate per-128-input-block rotation, initialized from the trained shared rotation, reduced `layer9 o_proj` from 0.309 to 0.264. These are module-local research pilots; neither has a CLI materialization contract or a model-level quality pass. Evidence: `runs/gemma4-rotation-lloyd-local-probe-20260925.json`, `runs/gemma4-rotation-scale-pilot-20260925/`, and `runs/gemma4-per-block-rotation-pilot-20260925/`.

Training a per-group threshold together with the reconstruction scale on rotated `layer9 o_proj` changed 5.25% of hard codes in 200 steps and reduced held-output relative MSE from 0.309 to 0.201. Extending to 1000 steps selected step 960 with MSE 0.195 and 7.48% hard-code change. This directly exercises a trainable hardening path, unlike the earlier scale-only soft-to-hard schedule, but remains a one-module pilot with no whole-model quality evidence or CLI storage contract. Evidence: `runs/gemma4-rotation-threshold-pilot-20260925/layer9-o{,-1000}.{json,safetensors}`.

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
