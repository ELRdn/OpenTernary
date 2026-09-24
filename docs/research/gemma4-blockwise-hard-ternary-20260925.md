# Gemma 4 E2B rotated hard ternary: block reconstruction follow-up

Date: 2026-09-25. Target: `google/gemma-4-E2B-it-qat-q4_0-unquantized@6befbaca7398925921802abd1f277b495b78b738`, 205 canonical G128 Linear targets. Hardware: RX 9070 XT, Windows ROCm PyTorch, BF16 inference. Base code: `79e88a426026322a892ca5cfb82a581ff14ad25d` plus the research scripts in this change. The fixed comparison is v4 validation, seed 42, 128-token PPL windows, greedy instruction generation. The independent v5 test used for the earlier q35 candidate was **not** used to select these experiments.

## Saved hard blocks

All 35 text blocks were reconstructed separately from the existing 200-step learned rotation manifest. Blocks 0–14 each have seven canonical targets, and blocks 15–34 each have five: 205 in total. Each run used six WikiText train examples, two held examples, and 100 optimizer steps on BF16 upstream inputs. Every block replayed its BF16 teacher exactly before replacement, improved held block-output relative MSE, saved hard int8 `{-1,0,+1}` codes and G128 scales, reloaded exactly, and reproduced its stored BF16 weight from the codes and scales. All 35 artifact SHA-256 values match the JSON reports. Mean block held relative MSE was 0.170880→0.026662; total artifact size was 5,621,419,592 bytes and summed reported execution time was 1,631.854 seconds. These local measurements do not establish a runnable high-quality 205-target model.

The saved weights were combined in memory on the same GPU and screened on v4 validation. They were **not** exported as a CLI snapshot. The BF16 comparison was English PPL 1355.436, Japanese PPL 1725.479, instruction exact match 57.8125%, collapse 0/64.

| In-memory candidate | Targets | English PPL | Japanese PPL | Instruction | Collapse | Fixed gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| WikiText block 0 | 7 | 1352.726 | 2188.283 | 53.125% | 0/64 | FAIL |
| WikiText blocks 0–2 | 21 | 1588.779 | 3208.714 | 37.5% | 0/64 | FAIL |
| WikiText blocks 0–34 | 205 | 3158.996 | 645993.297 | 0% | 56/64 | FAIL |
| Quantized-upstream WikiText blocks 0–2 | 21 | 1641.219 | 3299.133 | 37.5% | 0/64 | FAIL |
| Mixed English/Japanese block 0 | 7 | 1315.369 | 1691.077 | 56.25% | 0/64 | PASS on v4 only |
| Mixed quantized-upstream blocks 0–1 | 14 | 1437.724 | 1779.645 | 51.5625% | 0/64 | FAIL |
| Mixed quantized-upstream blocks 0–2 | 21 | 1533.526 | 1991.718 | 45.3125% | 0/64 | FAIL |
| Mixed joint blocks 0–2 | 21 | 1643.539 | 1993.169 | 45.3125% | 0/64 | FAIL |
| Mixed block 0 + dense-latent block 1 | 14 | 1196.711 | 1774.092 | 48.4375% | 0/64 | FAIL |
| Mixed block 0 + QA-logit-distilled block 1 | 14 | 1320.963 | 1938.004 | 51.5625% | 0/64 | FAIL |

The mixed calibration source `data/calibration/gemma4-mixed-train-20260925.json` has SHA-256 `b9a3ca032d064d3f6d445356d276ab462ddb2b033d1f409dc2364a5793b995da`. It contains 48 training and 16 held English/Japanese text or chat prompts from pinned WikiText and JSQuAD **train** sources. Its builder excludes exact texts and IDs from frozen quality v1–v5. The follow-up QA calibration file has SHA-256 `c08e24766fe0cf05b99c7db026417892c4303d4e46ff8210758ee2929b734d53`, adding disjoint SQuAD and JSQuAD **train** answers for 96/32 train/held examples. The dataset files stay outside Git. The builders and their source revisions are in `scripts/`.

The block-0 saved-runtime probe found held MSE 0.037549 during training, 0.037535 with the saved BF16 weight in FP32 matmul, and 0.037542 in actual BF16 runtime. BF16 rounding is not the observed cause of model-level collapse. Training layers 1–2 on quantized upstream activations, and jointly optimizing blocks 0–2, did not recover v4 quality. Allowing dense BF16 latent weights to change hard assignments in block 1 improved its held block MSE to 0.063532 with 3.483% changed codes and improved English PPL, but the instruction and Japanese gates failed. Direct full-model gradient was feasible at 11.35–12.22 GB PyTorch allocated VRAM, yet scale-only logit distillation of block 0 failed v4. Distilling answer-token teacher logits on block 1 reduced held top-64 KL 0.227100→0.212594 but also failed v4. Lower local or logit loss alone did not predict the fixed quality gate.

Evidence is under `runs/gemma4-blockwise-20260925/`, `runs/gemma4-blockwise-upstream-20260925/`, `runs/gemma4-blockwise-mixed-20260925/`, `runs/gemma4-joint-three-block-20260925/`, `runs/gemma4-dense-latent-layer1-20260925/`, and `runs/gemma4-global-qa-kd-layer1-20260925/`. The screen reports include artifact hashes, matched source/interface and dataset/protocol fingerprints, metrics, gate decisions, time, and PyTorch peak allocation. No full-205 quality candidate passed, no new independent test was opened, and the single-block v4 pass is not P7 acceptance. Native packed rotated ternary inference remains unsupported.

## 2026-09-25 follow-up: residual and hard-code pilots

The canonical 205-target learned-rotation snapshot was materialized through the CLI, reloaded through the quality runner, and its file integrity was rechecked with `openternary artifacts validate` (exit 0, `file_integrity`). This proves a runnable BF16 fake-quant conversion, not quality acceptance or native packed inference. On the fixed v4 validation split, the saved 205-target snapshot had English PPL 5114.567, Japanese PPL 2328674.750, instruction exact match 0%, and collapse 64/64, versus the BF16 source's 1355.436, 1725.479, 57.8125%, and 0/64. Source/interface, dataset, and protocol identities match in the quality reports.

The layer-1 SVD residual pilot saved a rank-256 BF16 factor pair per ternary Linear. It requires 8,647,168 additional BF16 parameters in that block and therefore is a hybrid, not a pure ternary result. Using the residual as two runtime matrix multiplies improved PPL to 1313.848/1661.364 but left instruction exact match at 50% (v4 gate FAIL). A disjoint QA-answer pilot trained those factors against BF16 top-64 logits; QA held KL improved 0.193600→0.169535, but the saved/reloaded factorized model worsened on v4 to English/Japanese PPL 1725.226/1849.131 and instruction 46.875% (gate FAIL). This demonstrates that the held QA logit objective did not predict the frozen model-level gate.

A separate pilot trained the rotated hard G128 codes of block 1 using straight-through code logits while block 0 stayed hard ternary. At code learning rate 0.006, 120 steps changed 10.94% of codes and degraded held QA KL 0.22636→0.65983; best was step 0. At 0.0008, no codes changed in 150 steps. At 0.002, the best 60-step checkpoint changed 0.0159% of codes and improved held KL 0.22647→0.21325. The saved/reloaded hard ternary candidate still failed v4: English/Japanese PPL 1525.215/1946.180, instruction 51.5625%, collapse 0/64, against the same BF16 baseline. The low-rate scale-only best candidate also failed v4 (1457.218/1821.484, 51.5625%). No test split was opened for these failing candidates. Research artifacts are in `runs/gemma4-lowrank-layer1-*`, `runs/gemma4-global-qa-hard-layer1-*`, and their matching `*-screen-*` directories.

Same-device smoke v1 throughput was measured after reload on the AMD Radeon RX 9070 XT, Python 3.12, torch 2.13.0+rocm10.0.0, transformers 5.17.0, BF16, seed 42, greedy generation, and five identical prompts. The BF16 source produced 160 tokens in 5052.95 ms (31.66 output token/s, peak allocated VRAM 10,297,021,952 bytes); the saved 205-target rotated fake-quant model produced 320 tokens in 26674.04 ms (12.00 output token/s, 10,318,714,368 bytes). Its responses ran to the 64-token cap on all five prompts because quality collapsed. These are observed end-to-end fake-quant throughput values with different output lengths, not a controlled packed-kernel speedup result. The 35-q-only saved rotated candidate measured 24.75 output token/s in the same environment, but its prior independent v5 instruction gate failed by 3.125 points; it is not an accepted alternative.

The current bottleneck is model-level quality propagation through the first two blocks, despite lower local or held QA losses. Further full-205 training should be gated on a reproducible two-block candidate that passes the frozen validation protocol using only disjoint calibration examples. The benchmark thresholds and model target remain unchanged.

## Calibration split correction and layer-1 attribution

A subsequent audit found 17 shared Japanese passages between the original 64-row text/chat calibration and its appended QA examples; five crossed the calibration train/held boundary. The original QA held losses above are therefore diagnostic only and are not an independent generalization estimate. This does not change any frozen v4 quality result. The builders now reject shared Japanese base/QA passages. A larger calibration set, `data/calibration/gemma4-mixed-qa-large-train-20260925.json` (SHA-256 `b2e63dd3658baa1f96203e914895e6a8ca6f3508d68e7e03214590bb4ccaf1ad`), uses 384 train and 128 held rows from the pinned training sources. Its 512 rows have zero duplicate IDs, zero duplicate text bodies, and zero shared Japanese base/QA passages. It excludes quality v1–v5 texts and IDs as before.

With a stronger BF16 teacher top-1 objective on the original small set, the best 14-target hard-code candidate changed 0.2629% of block-1 codes. It reached v4 English PPL 1349.319 and instruction 56.25%, but Japanese PPL 1978.175 failed the 2% limit. Doubling Japanese sample loss worsened v4 to English/Japanese PPL 1418.749/2093.574 and instruction 51.5625%. On the corrected 512-row set, shuffled training with 384 steps selected step 192 by held objective (0.270720→0.263302, 0.1272% code changes). Its saved/reloaded 14-target v4 result was 1451.061/1876.145 PPL, 54.6875% instruction, collapse 0/64: FAIL. Thus increased calibration size, top-1 emphasis, and a Japanese weight did not yield a two-block accepted candidate.

To isolate block-1 sensitivity, each saved hard ternary projection was added separately to the accepted block-0 seven-target candidate. The full artifact was validated in every run, while only the named projection was applied; these are in-memory v4 screens, not independent test or CLI snapshots.

| Additional block-1 projection | Total ternary targets | English PPL | Japanese PPL | Instruction | v4 gate |
| --- | ---: | ---: | ---: | ---: | --- |
| q_proj | 8 | 1431.138 | 1921.485 | 53.125% | FAIL |
| k_proj | 8 | 1264.463 | 1655.046 | 54.6875% | FAIL |
| v_proj | 8 | 1303.257 | 1611.850 | 56.25% | PASS |
| o_proj | 8 | 1216.638 | 1588.342 | 53.125% | FAIL |
| up_proj | 8 | 1262.344 | 1643.017 | 53.125% | FAIL |
| gate_proj | 8 | 1467.827 | 1887.544 | 54.6875% | FAIL |
| down_proj | 8 | 1412.438 | 1728.827 | 50.0% | FAIL |

Only `v_proj` passed this validation gate as an isolated addition. It has not been materialized as a CLI snapshot or tested on an untouched split. Every other individual addition failed, so the full block-1 failure is not attributable solely to interactions among its seven projections. The experiment records are under `runs/gemma4-global-qa-hard-layer1-top1-*`, `runs/gemma4-global-qa-hard-layer1-large-*`, and `runs/gemma4-blockwise-role-screen-20260925/`.

## BF16 global rotation before hard ternary: layer-1 q pilot

The CLI already accepts a learned rotation plan with `quantize --rotation-manifest`; the previously saved/reloaded 205-target fake-quant snapshot exercised that path and failed model-level quality. The new `scripts/train_gemma4_global_rotation_q1.py` instead updates one 128-by-128 Cayley rotation using BF16 teacher top-64 output distillation, then hardens the rotated BF16 weight with the existing AbsMean G128 codes and scales. Seven layer-0 targets use the saved mixed-calibration hard block. The optimized q1 rotation and hard codes are saved, reloaded, and checked for exact weight reconstruction. This combination is an eight-target research artifact; the CLI does not yet train its rotation or import the separately reconstructed layer-0 block as a single `quantize` input.

The numerical preflight checked rotation orthogonality, pre-quantization linear equivalence, code/scale equality with `quantize_groupwise`, and finite STE gradients. The corrected 384/128-row calibration set above supplies disjoint calibration train/held examples. All quality comparisons below use the same RX 9070 XT, BF16 runtime, source and interface identity, 128-token PPL windows, and greedy instruction protocol. Both candidate evaluations load saved hard weights. The v5 test process independently reloads the two artifacts with `scripts/evaluate_gemma4_global_rotation_q1.py` and validates hashes, source identity, data fingerprint, and protocol fingerprint.

| Eight-target candidate | Split | English PPL | Japanese PPL | Instruction | Collapse | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| BF16 source | v4 validation | 1355.436 | 1725.479 | 57.8125% | 0/64 | Reference |
| 2 train / 1 held, best step 1 of 1 | v4 validation | 1136.751 | 1421.092 | 56.25% | 0/64 | PASS |
| 96 train / 32 held, best step 80 of 200 | v4 validation | 1450.878 | 1962.070 | 54.6875% | 0/64 | FAIL |
| 96 train / 32 held, best step 0 of 1 | v4 validation | 1128.167 | 1419.834 | 56.25% | 0/64 | PASS |
| BF16 source | v5 test | 1274.036 | 1418.493 | 59.375% | 0/64 | Reference |
| Saved/reloaded 96/32 step-0 candidate | v5 test | 1176.952 | 1167.171 | 57.8125% | 0/64 | PASS |

The 200-step run lowered calibration held objective from 0.201009 to 0.172625 at step 80, changing 11.214% of q1 hard codes, yet failed every non-collapse v4 gate. A lower teacher-output loss therefore did not predict model-level quality. The one-step 96/32 run selected its initial rotation because its held objective worsened at step 1; it validates the rotate-then-ternarize path, but does not show that end-to-end rotation retraining improved quality. The tiny 2/1 pilot did select a changed rotation but is insufficient evidence for a stable training gain. The v5 test split is disjoint from v4, but was previously opened for a different q35 experiment, so this is additional evidence rather than a pristine final test. No all-205 quality claim follows from eight targets, and native packed ternary execution remains unimplemented.

Artifacts: `runs/gemma4-global-rotation-q1-{smoke,early1}-20260925.{json,safetensors,quality.json}`, `runs/gemma4-global-rotation-q1-20260925.{json,safetensors,quality.json}`, and `runs/gemma4-global-rotation-q1-early1-v5-test-20260925.{json,quality.json}`. The selected step-0 artifact SHA-256 is `2c064956166078660a14853a8c9ffc262a6892373d848ca6f3d0cc733b124a03`; the layer-0 artifact SHA-256 is `06c3b261799a2b177664ba23162e9827daa827e898b70a1842c7d63569e0ca6f`.
