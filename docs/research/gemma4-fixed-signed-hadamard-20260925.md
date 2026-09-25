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

The same signed H1024/G128 AbsMean method was applied to every canonical Linear target and each pre-quantization Linear function check passed. On matched v4 validation, the in-memory 205-target candidate reached English PPL 7261.130, Japanese PPL 22939.740, instruction exact match 0%, and collapse 1/64: **FAIL** against the same BF16 reference. Wall time was 1711.86 seconds (28.53 minutes) and PyTorch peak allocated VRAM was 10,733,558,272 bytes. The common quality runner recorded 1558.566 seconds of inference for this candidate versus 61.282 seconds for BF16, a 25.43× observed end-to-end slowdown; peak allocator VRAM was 9.996 GiB versus 9.988 GiB. Output lengths can differ because candidate quality failed, and this implementation runs FP32 Hadamard operations in Python hooks with dequantized BF16 weights. Thus the timing is an observed research implementation cost, not native packed ternary kernel throughput. This run is saved as `runs/gemma4-signed-h1024-all205-v4-20260925.json` and its `.quality.json` detail. The 205 hard weights were not saved as a reloadable snapshot because the in-memory gate failed. This result rules out this fixed-sign, greedy-segmented, AbsMean post-training conversion as an accepted all-target candidate on this seed and source; it does not establish that every H1024-based training method fails.

The earlier separately saved hard-code training candidate for these 14 targets, using learned H128 + Cayley rotations and answer-weighted logit training, also failed v4: 1316.482 / 1710.986 PPL, 51.5625% instruction, 0/64 collapse. That candidate changed 0.14045% of block-1 hard codes and improved its held objective, but the instruction gate worsened by 6.25 points.

## Artifacts and limits

The screen reports are under `runs/gemma4-signed-h{128,256,512,1024}-layers01-v4-20260925.json`, `runs/gemma4-unsigned-h1024-layers01-v4-20260925.json`, and `runs/gemma4-no-rotation-layers01-refit{0,5}-v4-20260925.json`, with matching `.quality.json` files. The code is `scripts/screen_gemma4_signed_hadamard.py`; the split-specific source and protocol checks are in the reports. The no-rotation and unsigned control reports retain the earlier generic `status` string; their `max_block`, `unsigned`, and `scale_refit_steps` fields identify the actual method. No final test split was used in these fixed Hadamard screens.

Least-squares G128 scale/code alternation reduced source-weight MSE to about 70% of canonical AbsMean on two representative layer-1 tensors. On layer-1 `q_proj`, it changed 11.47% of codes and raised the zero fraction from 32.56% to 44.03%. The H1024 + five-step scale/code-refit 14-target screen, however, worsened model-level quality to English/Japanese PPL 1816.503/2653.242 and instruction 51.5625%, with zero collapse: **FAIL**. Its artifact is `runs/gemma4-signed-h1024-layers01-refit5-v4-20260925.json`. Lower local weight MSE once again did not predict the frozen quality gate. Neither a new final test nor a 205-target saved artifact was produced for this failed candidate.

## Saved eight-target base plus H1024 additions

The previously saved/reloaded passing layer-0 seven-target block and layer-1 `q_proj` candidate were kept fixed. The other six layer-1 projections were individually quantized from the BF16 source in memory using signed H1024 and G128 AbsMean. Applying all six additions at once yielded English/Japanese PPL 1510.391/1976.189, instruction 53.125%, collapse 0/64: **FAIL** on v4. The result is `runs/gemma4-eight-plus-six-h1024-v4-20260925.json`. The eight-target base and its artifact hashes were reverified, but the six added tensors were not saved as a reloadable artifact. This negative result motivated one-role-at-a-time sensitivity screens.

| Single H1024 layer-1 addition to saved eight-target base | Total targets | English PPL | Japanese PPL | Instruction | v4 gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `v_proj` | 9 | 1189.477 | 1559.650 | 51.5625% | FAIL |
| `k_proj` | 9 | 1109.184 | 1403.972 | 56.2500% | PASS |
| `gate_proj` | 9 | 1380.456 | 1777.245 | 50.0000% | FAIL |
| `o_proj` | 9 | 1096.110 | 1524.281 | 53.1250% | FAIL |
| `up_proj` | 9 | 1197.494 | 1452.760 | 56.2500% | PASS |
| `down_proj` | 9 | 1173.356 | 1483.746 | 53.1250% | FAIL |

All six had zero collapse. The reports are `runs/gemma4-eight-plus-h1024-<role>-v4-20260925.json`. The two single-role passes have unsaved H1024 additions and use the previously opened v4 validation; they are not a new final test or a deployable 9-target artifact. Combining individually passing roles also requires its own model-level test.

Adding both `k_proj` and `up_proj` to the saved eight-target base passed one in-memory v4 run: English/Japanese PPL 1151.158/1402.692, instruction 56.25%, collapse 0/64. The two extra hard G128 tensors were then saved with codes and scales, and their artifact SHA-256 was checked on reload. A CPU-generated artifact differed from the GPU-generated candidate by one hard code among the `up_proj` weights and failed v4 (1153.100/1406.005, 54.6875%). A GPU-generated artifact had exactly the same codes and BF16 weights as the in-memory generation for both modules (SHA-256 `9718ddc80f0e2f926fcd7a51c3a02769b1be60d940c23d475733e2cc4d00a2c8`). Three independent reload evaluations of this identical GPU artifact returned:

| GPU artifact reload | English PPL | Japanese PPL | Instruction | v4 gate |
| --- | ---: | ---: | ---: | --- |
| First | 1158.177 | 1409.839 | 54.6875% | FAIL |
| Second | 1151.158 | 1402.692 | 56.2500% | PASS |
| Third | 1151.158 | 1402.692 | 56.2500% | PASS |

The first GPU reload differed from the two repeated results in one instruction response (`NASA` versus `had left NASA`). Source model weights, saved extra hard weights, codes, and scales were checked bitwise; 10 repeated GPU quantizations of each extra module produced identical codes. The one CPU/GPU hard-code difference therefore does not fully explain the GPU artifact's own run-to-run variation. The 10-target pilot is **not robustly accepted** by the fixed instruction gate. The underlying inference numerical nondeterminism or execution-state dependency remains to be diagnosed. All additional evidence remains on already opened v4 validation; no new final test was used.

### Reload reproducibility diagnosis

The saved 10-target artifact was reloaded for the exact v4 case `squad-validation-5725c6dcec44d21400f3d533` (expected `NASA`). Five successive generations within one process produced identical `NASA` token IDs and logits. Three further independent processes produced `NASA`, `had left NASA`, and `NASA`, respectively; two generations inside each process matched exactly. In the `NASA` process, the first-step logits for `NASA` and `had` were 24.5 and 24.375. In the `had left NASA` process, both were 24.5, so the greedy decision changed at a single BF16 logit step. The complete v4 quality run in another fresh process returned the passing 1151.158/1402.692 PPL and 56.25% instruction score, followed by three identical `NASA` diagnostic generations. This isolates the observed instruction failure to a near-tie at the first generated token but does not yet identify the operation that introduced the numerical difference.

`torch.use_deterministic_algorithms(True)`, cuDNN deterministic mode, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` did **not** eliminate the process-level variation: the third fresh process with these settings generated `had left NASA`. Instrumenting each decoder layer with a CPU hash yielded identical outputs at all 35 layer boundaries in four fresh processes; an earlier hash run did reproduce the alternate answer and differed already at the layer-0 output. Synchronizing at each decoder layer boundary without collecting hashes produced `NASA` in four fresh processes. Those small samples suggest an execution-state or GPU scheduling dependence, but do not establish a reliable fix or make the 10-target quality gate stable. The diagnostic mode and its saved JSON reports are provided by `scripts/evaluate_gemma4_global_rotation_q1.py` (`runs/gemma4-ten-h1024-nasa-*-20260925.json`). A synchronized candidate must be compared with a newly measured synchronized BF16 baseline before any quality or speed claim.

Further tests ruled out layer synchronization by itself as a fix: its fifth independent process returned `had left NASA`. More extensive tracing, including the layer-0 input and its attention/MLP projections, produced the same 45 activation hashes and `NASA` response in 12 independent processes. The tracing changes execution timing and is not evidence that the uninstrumented model is stable.

## Matched eager-attention follow-up

Gemma 4 text attention was switched from the default path to the Transformers eager implementation, with an assertion that all 35 text decoder layers used `eager`. This is a **new research execution protocol**; every control and candidate below used the same source, dataset, generation settings, RX 9070 XT, and eager attention. The candidate still consists of the same three saved, hash-checked artifacts: seven layer-0 targets, layer-1 `q_proj`, and signed H1024 layer-1 `k_proj`/`up_proj`. No weights were retrained or selected using the test results.

| Eager attention split | Model | English PPL | Japanese PPL | Instruction | Collapse | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v4 validation | BF16 | 1351.609 | 1722.587 | 57.8125% | 0/64 | reference |
| v4 validation | Saved 10-target G128 | 1154.391 | 1413.387 | 56.2500% | 0/64 | PASS |
| v5 test | BF16 | 1272.395 | 1427.770 | 59.3750% | 0/64 | reference |
| v5 test | Saved 10-target G128 | 1207.210 | 1128.605 | 59.3750% | 0/64 | PASS |

The v4 candidate full-quality run repeated in two independent processes with identical summary values. Six independent eager-attention single-case reloads also all answered `NASA`. The v4 composite score against the eager BF16 control was 0.109206; v5 was 0.091267. Each comparison matched its source identity, dataset fingerprint, and quality-runner protocol fingerprint. The runner's `quality-runner-v1` protocol hash does **not** encode the attention implementation; both scripts explicitly assert and record `eager`, so reports from different attention modes must not be compared as equivalent merely because their runner hashes match. The BF16 controls are saved at `runs/gemma4-bf16-eager-v{4,5}-research-20260925.json`; the candidate summaries and artifact hashes are in `runs/gemma4-ten-h1024-eager-quality-repeat-20260925.json` and `runs/gemma4-ten-h1024-eager-v5-research-20260925.json`. Hash-checked gate reports are `runs/gemma4-ten-h1024-eager-v{4,5}-compare-20260925.json`. The BF16 control and comparison code is in `scripts/evaluate_gemma4_bf16_eager_research.py` and `scripts/compare_gemma4_eager_research.py`. A crossed v4/v5 comparison was rejected without writing an artifact.

This is a promising **partial** pilot under eager attention, not a robustly accepted all-205 model. The default attention implementation still showed a process-level quality failure. Both v4 and v5 were opened in earlier research, so a newly separated final test is required for acceptance. A native packed ternary runtime was not exercised; the research model still uses dequantized BF16 hard weights and Python input-rotation hooks.
