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
