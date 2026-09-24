# Fixed signed Hadamard rotation pilot for Gemma 4 E2B

Date: 2026-09-25. This is a validation-only research experiment on the canonical Gemma 4 E2B BF16 source and G128 Linear targets. It does not reproduce Bonsai 2 or establish a deployable ternary model.

## Source and geometry

The [Bonsai 2 official model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) and its [whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf), sections 2.1–2.4, describe ternary G128 weights with FP16 group scales and a fixed signed blockwise H1024 basis, with the matching activation transform at runtime. The whitepaper gives the column-vector transform `R = H S / sqrt(1024)`, where `S` is a fixed diagonal of signs; the row-vector implementation here applies `S H / sqrt(1024)` to both source weights and Linear inputs. It reports low-bit coverage of embeddings and LM head as well as attention and MLP projections, while retaining 0.0976% of language parameters above low bit. OpenTernary's canonical 205 targets cover the text attention and MLP Linear projections only, or 35.9605% of all Gemma 4 E2B parameters. These are different models and coverage definitions. The whitepaper does not specify a complete conversion or training recipe that would make a fixed transform by itself sufficient for another model.

The 205 Gemma 4 target input dimensions are 1536 (135), 2048 (28), 4096 (7), 6144 (15), and 12288 (20). H1024 does not divide 1536. This pilot partitions each input axis into the largest possible power-of-two blocks at or below the selected maximum, so 1536 becomes H1024 + H512. A fixed sign vector is derived from SHA-256 of the experiment seed and module name. The same orthogonal map is applied to BF16 source weights before hard G128 AbsMean quantization and to each activation before its Linear. Pre-quantization function preservation is checked per module before model evaluation.

The common comparison uses RX 9070 XT, the same source and interface fingerprints, quality dataset v4 validation, seed 42, 128-token PPL windows, and greedy instruction generation. The source BF16 results were English PPL 1355.436, Japanese PPL 1725.479, instruction exact match 57.8125%, collapse 0/64. The gate permits at most 2% PPL regression on each language, at most 2 instruction points of regression, zero collapse, and positive composite score. The v4 and v5 splits have already been opened in previous experiments; a new untouched test is required for any final candidate.

## Two-block screen (14 of 205 targets)

| Fixed signed transform | Mean module weight MSE | English PPL | Japanese PPL | Instruction | Collapse | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| None, G128 AbsMean | 0.000145142 | 3404.380 | 3638.618 | 37.5000% | 0/64 | FAIL |
| None, G128 least-squares refit (5 steps) | 0.000101712 | 2911.692 | 3695.732 | 40.6250% | 0/64 | FAIL |
| H128 | 0.000142001 | 1665.190 | 2210.231 | 50.0000% | 0/64 | FAIL |
| H256 | 0.000141998 | 2021.957 | 2569.314 | 46.8750% | 0/64 | FAIL |
| H512 | 0.000142049 | 1853.168 | 2809.780 | 45.3125% | 0/64 | FAIL |
| H1024 | 0.000142066 | 1783.612 | 2469.079 | 53.1250% | 0/64 | FAIL |
| H1024 without signs | 0.000142204 | 2180.740 | 3535.246 | 51.5625% | 0/64 | FAIL |
| H1024, G128 least-squares refit (5 steps) | 0.000101810 | 1816.503 | 2653.242 | 51.5625% | 0/64 | FAIL |

All candidates used the same seed and the same 14 modules. The four signed-rotation AbsMean candidates' mean local weight MSE barely varies, while the model-level quality varies substantially. Relative to no rotation, H1024 substantially improved both PPLs and instruction accuracy, so rotation is useful on this pilot. Fixed signs also improved H1024 over unsigned H1024. It still did not recover the first two blocks to the frozen BF16 gate. Least-squares refit lowered weight MSE further but did not close the quality gap. These are in-memory fake-quant screens, not saved/reloaded candidates or native packed inference.

## All-205 H1024 screen

The same signed H1024/G128 AbsMean method was applied to every canonical Linear target and each pre-quantization Linear function check passed. On matched v4 validation, the in-memory 205-target candidate reached English PPL 7261.130, Japanese PPL 22939.740, instruction exact match 0%, and collapse 1/64: **FAIL** against the same BF16 reference. Wall time was 1711.86 seconds (28.53 minutes) and PyTorch peak allocated VRAM was 10,733,558,272 bytes. This run is saved as `runs/gemma4-signed-h1024-all205-v4-20260925.json` and its `.quality.json` detail. The 205 hard weights were not saved as a reloadable snapshot because the in-memory gate failed. This result rules out this fixed-sign, greedy-segmented, AbsMean post-training conversion as an accepted all-target candidate on this seed and source; it does not establish that every H1024-based training method fails.

The earlier separately saved hard-code training candidate for these 14 targets, using learned H128 + Cayley rotations and answer-weighted logit training, also failed v4: 1316.482 / 1710.986 PPL, 51.5625% instruction, 0/64 collapse. That candidate changed 0.14045% of block-1 hard codes and improved its held objective, but the instruction gate worsened by 6.25 points.

## Artifacts and limits

The screen reports are under `runs/gemma4-signed-h{128,256,512,1024}-layers01-v4-20260925.json`, `runs/gemma4-unsigned-h1024-layers01-v4-20260925.json`, and `runs/gemma4-no-rotation-layers01-refit{0,5}-v4-20260925.json`, with matching `.quality.json` files. The code is `scripts/screen_gemma4_signed_hadamard.py`; the split-specific source and protocol checks are in the reports. The no-rotation and unsigned control reports retain the earlier generic `status` string; their `max_block`, `unsigned`, and `scale_refit_steps` fields identify the actual method. No final test split was used in these fixed Hadamard screens.

Least-squares G128 scale/code alternation reduced source-weight MSE to about 70% of canonical AbsMean on two representative layer-1 tensors. On layer-1 `q_proj`, it changed 11.47% of codes and raised the zero fraction from 32.56% to 44.03%. The H1024 + five-step scale/code-refit 14-target screen, however, worsened model-level quality to English/Japanese PPL 1816.503/2653.242 and instruction 51.5625%, with zero collapse: **FAIL**. Its artifact is `runs/gemma4-signed-h1024-layers01-refit5-v4-20260925.json`. Lower local weight MSE once again did not predict the frozen quality gate. Neither a new final test nor a 205-target saved artifact was produced for this failed candidate.
