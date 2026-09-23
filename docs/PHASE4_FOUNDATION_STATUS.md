# Phase 4 foundation repair — 2026-09-22

## Status and scope

**PARTIAL — P0/P1 and the P3 evaluation path are verified; P2 has full-target uninterrupted-versus-terminal-checkpoint identity plus tiny mid-optimizer identity, while full-target mid-optimizer identity remains unrun; P4 is production-connected but every screened hard snapshot failed the quality gate.** This is engineering and negative screening evidence, not Gemma quality acceptance or paper-grade reproduction.

Approved limits: 24 GPU-hours per research phase, 120 GB active artifacts, and an 80% resource stop threshold. P7 research acceptance and paper-grade reproduction remain separate gates. The 2026-09-22 validation screen plus the independent P2 identity run use 88.016 GiB across seven heavy runs, below the 96 GiB stop threshold. No further full snapshot candidate was started. No dependency change, commit, or push was performed.

Starting Git state: `feat/calib-threshold` at `d37ace6c3e75291048a7bd12534618fe7b6d4059`, with existing user changes preserved.

## P0 — Specification and live asset reconciliation

The fail-closed preflight command now records and verifies source hashes, the exact canonical target inventory, protocol hashes, runtime/GPU/disk/Git state, resource limits, and a bounded classification of existing runs without loading model tensors or starting training. A revision counts as pinned only when it is an exact 40-hex Git commit; mutable names such as `main` fail the preflight, and a different 40-hex revision also fails the canonical-target check. The canonical weight SHA-256 and exact 205-name inventory are gates rather than report-only metadata. Inspection identity is normalized to the canonical model ID while the local source locator is recorded separately, so the fingerprint is path-independent.

Verified against the local source:

- model: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- revision: `6befbaca7398925921802abd1f277b495b78b738`
- local source: `D:\AI\llm model\gemma-4-E2B-it-qat-q4_0-unquantized`
- source manifest: 9 files totaling 10,241,082,561 bytes (including one weight file), each recorded with SHA-256
- target contract: 205 quantizable modules, per-group G128, BF16
- runtime: PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333, AMD Radeon RX 9070 XT
- preflight result: PASS; `scientific_acceptance=false`

Evidence: `runs/p0-preflight-20260922-rx9070xt-v14/preflight.json` (executed from outside the repository after the final acceptance-plan update; exact commit, canonical revision, canonical weight hash, exact target inventory, all six current protocol hashes, and repository state pass; inspection fingerprint is the known `sha256:07fe44eae504218937187b75826a4c780cba2f76a9145fdc38f7efdd4d3a7b01`).

The existing-run inventory is classification only. Old runs are not promoted to controls or accepted evidence merely because they contain 205 state files.

## P1 — Calibration correctness and memory recovery

Implemented and tested:

- padding rows are removed using captured attention masks; maskless legacy caches fail closed
- MSE/L1 aggregation uses valid output element counts
- each microbatch immediately backpropagates its weighted contribution
- configured loss and optimizer are honored by the real runner
- target flags are shared with inspection/calibration rather than silently diverging
- an optimizer update is transactional across parameters, optimizer moments, CPU RNG, and CUDA RNG
- an OOM restores the entire failed logical update and retries the same step with a smaller microbatch
- actual microbatch chunks concatenate cached batches before one forward/backward
- effective microbatch and retry history is written to the report

An injected OOM after Adam state mutation produces the same loss history and materialized content fingerprint as an uninterrupted run, including on the real ROCm execution path. A hardware pressure probe on the RX 9070 XT produced an actual `OutOfMemoryError` after 480 MiB under a 3% allocator cap, released the allocation, and then completed BF16 backward plus an Adam step. Evidence: `runs/rocm-oom-recovery-20260922-v1/report.json`. The allocator probe and transaction-equivalence test are intentionally separate claims.

## P2 — Save, resume, and materialize contracts

New normal checkpoints use schema v3 and include:

- final-module state before the completed cursor advances
- per-state path, size, and SHA-256 manifest entries
- optimizer state plus CPU/CUDA RNG state
- source, target, quantization, calibration, and activation-cache contracts
- activation-cache shard paths, sizes, and SHA-256 values

Resume and materialize reject missing/corrupt state, missing cache shards, contract changes, incomplete target sets, and legacy schema checkpoints. A mid-module resume additionally requires the current parameter, optimizer state, and CPU RNG state (plus the current threshold when enabled); missing transaction state or an unloadable optimizer state now stops instead of silently continuing from a fresh state. Explicit `--init-from` also rejects missing checkpoints, corrupt/incomplete state, source/target/quantization/dtype mismatch, and no longer falls back to a fresh initialization. Reusable warm-start caches are accepted only after every shard size and SHA-256 matches; a damaged optional cache is recaptured, while cache-manifest persistence failure stops capture. Materialize-only requires the resume contract and treats final report persistence as part of success. Schema v2 is accepted only through the explicit audited recovery path. Same-directory and cross-directory interrupted/resumed tiny runs reproduce the uninterrupted hard-export fingerprint within the tested tolerance.

`runs/p3-scale-screen-v1` was interrupted after the durable final `step_02050.pt`, resumed with cache reuse, and completed final evaluation/materialization for all 205 targets. The independent uninterrupted `runs/p2-scale-identity-v2` run has the same output-path-normalized config, complete loss history, final scale fingerprint, calibrated content fingerprint, and SHA-256 for all 22 materialized snapshot files. This proves full-target terminal-checkpoint resume/materialization identity. Mid-optimizer identity is proven on the tiny real path; a full-target mid-optimizer interruption pair remains an explicitly unrun stricter check.

A fresh artifact audit rehashed all 9,840 train/held activation-cache shards across the six learned full-target runs (18,585,040,944 bytes): every size and SHA-256 matches its manifest, the train/held contract fingerprints agree, and each checkpoint cache fingerprint matches both cache manifests. Each final schema-v3 checkpoint also has 205 manifested state files whose hashes match, the manifest module set equals the 205-name resume contract, the cursor is `205:0`, and the reloaded report remains final-hard. Every materialized manifest agrees with its calibration content fingerprint, declares 205 quantized of 1,951 total tensors, and maps all tensors to 12 present model shards.

Every byte of every materialized snapshot tree was also rehashed (32 files each: 22 top-level snapshot files plus 10 copied Hugging Face cache-metadata files). The canonical aggregate is SHA-256 over sorted `{name,size,sha256}` entries. The resumed scale run and independent uninterrupted identity run are byte-identical at `3ed297b77bcb97b4181cd858161e927ccb2cf0f0b4d185c8e7808eaec3d79e86`; threshold is `c3c3fd6076c9bb71b988242017da783e93f65f69812b8e37bd340254e0df6e31`; soft linear/cosine/exponential are `4dc912c3ae801010c19df62e05643e3847a75533f7cd1f0bbccfc128f7889f30`, `3769a0b76fd13877f758a90727f6128b17e0ce20b5e0740fe9edfbdb97d94c9d`, and `1f153b9d5d93bde648a599d51480b6fcd51a5e452850072e87f8542c919e0bd9`.

## P3 — Evaluation and controls foundation

The opt-in quality library now provides:

- independent-document causal perplexity with each valid target token scored once
- per-language and overall NLL/PPL with protocol fingerprints
- NFKC exact overlap checks and deterministic character-5gram MinHash near-duplicate checks across calibration/validation/test
- closed-answer exact-match instruction scoring and conservative collapse diagnostics
- the preregistered balanced acceptance calculation:
  - 35% general-PPL relative improvement
  - 35% Japanese-PPL relative improvement
  - 30% instruction-score delta
  - composite must be positive
  - each PPL regression must be at most 2%
  - instruction regression must be at most 2 points
  - collapse count must be zero
- `compare` consumes matched `quality.json` files and writes `acceptance.json`; missing/malformed JSON, one-sided quality input, protocol/data mismatch, schema-v2 protocol-fingerprint self-inconsistency, response-hash inconsistency, detached PPL summaries, model-revision mismatch, actual-dtype mismatch, or actual-device mismatch fails closed

The production `quality` runner/CLI now writes a frozen protocol, source/data fingerprints, exact dtype/device, disjointness audit, language PPL, instruction score, collapse diagnostics, and report schema v2 audit fields containing each generated response plus its UTF-8 SHA-256. Quality dry-run parses and fully validates the frozen dataset, reports its fingerprint, and still creates no run directory; malformed JSON or split contracts fail before model loading. Both perplexity documents and instruction prompts fail closed on normalized exact or character-5gram MinHash cross-split overlap. `data/quality/gemma4-e2b-quality-v1.json` is deterministically generated from pinned WikiText, SQuAD, and JGLUE/JSQuAD revisions; its file SHA-256 is `81436b92856dd1a712d5340a2a8b4f9a358af235621ce821a2637c6a786fd899`, and offline regeneration matched exactly.

Matched validation runs share protocol fingerprint `f2daec8d...a973e` and dataset fingerprint `a0b11176...648f`:

| Control | General PPL | Japanese PPL | Instruction | Collapse |
|---|---:|---:|---:|---:|
| BF16 | 1,214.61 | 2,229.70 | 62.5 | 0 |
| naive G128 | 287,747.46 | 212,961,126.33 | 0.0 | 8 |
| scale-only | 281,853.76 | 160,282,304.67 | 0.0 | 8 |
| threshold | 315,223.93 | 140,044,860.87 | 0.0 | 8 |

All seven validation reports were rerun into `quality-*-validation-v3` with report schema v2 and the instruction-split audit embedded. Their protocol fingerprints, dataset fingerprints, metric summaries, and generated responses exactly match the preceding reports; every one of the 56 saved response hashes recomputes correctly. The instruction audit fingerprint is `a982aa563b50ec0381e14193e78562127af9490f6a82e9c5773cba7dfd0cd432`; maximum validation/test MinHash similarity is `0.1484375` against the `0.9` rejection threshold. An independent six-file model/tokenizer/chat interface audit also matches all seven snapshots with fingerprint `f53f74ea4996fd9802bc42a03a42cfc9500ec84447653f5aed10bf8b39d7b47b`; evidence is `runs/p3-model-interface-audit-v1/audit.json`. BF16 produces ordinary answer text and has zero collapse flags. Every ternary candidate produces symbol-only or repeated punctuation on all eight instruction cases, directly supporting the recorded `8/8` collapse result. The v3 BF16-versus-candidate comparisons report matching schema, protocol, data, model revision, actual BF16 dtype, and actual `cuda:0` device, then return `accepted=false`.

The evaluation/control-execution gate is now met, but every ternary control fails quality acceptance. Only `validation` was executed; the frozen `test` split remains untouched.

Compatibility boundary: `compare` still reads a pair of schema-less legacy quality summaries, as covered by the pre-existing backward-compatibility test. Those reports do not carry response hashes, recomputable protocol bodies, or runtime identity fields and therefore are not P7 acceptance evidence. Before final acceptance, decide whether to reject legacy quality reports outright or force their gate result to non-accepting; this is a benchmark compatibility decision rather than a silent patch.

## P4 — Soft-to-hard production screen

`openternary.quant.soft_ternary` implements and tests linear, cosine, and exponential temperature schedules, a final 10% hard region, differentiable ternary expectations, optional zero-code logit bias, and deterministic hardening. Config, runner, checkpoint v3, resume, final hard evaluation, and fail-closed materialization are connected. In a 10-step screen, a positive hard fraction always reserves at least one final hard step.

All three no-modulation schedules were run on 205 targets and judged only after hardening, saving, and reload:

| Schedule | Final recon loss | Held-out after | General PPL | Japanese PPL | Instruction | Collapse |
|---|---:|---:|---:|---:|---:|---:|
| linear | 1.454211 | 1.439834 | 301,165.69 | 185,326,217.18 | 0.0 | 8 |
| cosine | 1.454181 | 1.439804 | 298,709.06 | 190,855,174.30 | 0.0 | 8 |
| exponential | 1.453746 | 1.439399 | 302,290.56 | 185,735,594.41 | 0.0 | 8 |

All three retained the initial hard code assignment (`code_change_ratio=0`) and failed the preregistered quality gate. Code inspection shows this is structural, not evidence that the 10-step schedule was merely too short: the current optimizer owns only reconstruction scales, while final hardening is a deterministic midpoint decision from the frozen source weight and frozen reference scale. Temperature and `zero_logit_bias` affect the soft training path but cannot alter the saved hard assignment. New reports/checkpoints expose this as `assignment.trainable=false`, `hardening_source=frozen-weight-midpoint`, and `hardening_uses_zero_logit_bias=false`.

The collapse is also highly concentrated. Across the learned candidates, `q_proj` contributes 86.47% of aggregate held-out squared error. `model.language_model.layers.0.self_attn.q_proj` contributes about 47.2% by itself and remains near held-out MSE 291 for scale-only, threshold, and all three soft schedules. An independent sign scan of the saved scale-only and threshold snapshots reproduced the threshold report exactly: 166,481 of 1,835,532,288 hard codes changed (`9.0699031e-5`). Only 343 of 3,145,728 codes changed in the dominant layer-0 `q_proj` (`1.09037e-4`). Therefore simply extending the existing schedule has no demonstrated basis. There is no winning schedule, the zero-code-bias row was not opened, and P4 research acceptance is **FAIL / redesign required**. Soft-state or local reconstruction improvements are not promoted. The design decision and bounded next experiment are recorded in [P4 soft-to-hard redesign decision](plans/p4-soft-to-hard-redesign-decision.md).

## P5–P7

- P5 window reconstruction: not started; blocked by the failed P4 quality gate
- P6 rotation research: not started; the design and export constraints in `ROADMAP.md` remain gates
- P7 full 205-target acceptance: not started

## Validation evidence

| Check | Result |
|---|---|
| Focused P0–P4 regression suite | PASS on RX 9070 XT; real ROCm transaction and soft-to-hard paths executed without skips |
| Real ROCm OOM pressure | PASS; actual OOM followed by BF16 backward and Adam recovery |
| Full-target execution | scale-only uninterrupted and terminal-checkpoint-resumed outputs match; threshold and three P4 schedules completed 205/205 |
| Matched quality protocol | PASS on validation; all seven schema-v2 reports share protocol/data/interface fingerprints, preserve prior summaries/responses, and have valid response hashes |
| Scoped Ruff on changed/new Python files | PASS |
| Scoped Ruff format check | PASS, 35 changed/new Python files |
| mypy on `src/openternary` | PASS, 56 source files |
| pytest collection | 267 tests |
| Hermetic CLI inspect integration | PASS; local minimal Gemma 4 snapshot, no Hugging Face cache dependency |
| Full pytest | PASS, 267 tests |
| `git diff --check` | PASS (line-ending warnings only) |

`tests/test_cli_help.py::test_inspect_normal_creates_run_exit_0` now exercises the normal inspect path against a local minimal Gemma 4 snapshot, verifies both output files and the inspected tensor count, and no longer depends on the default Hugging Face cache. Repository-wide Ruff still reports 89 pre-existing findings in old helper scripts/tests; the current changed/new Python-file scope is green.

## Next fail-closed gates

1. Resolve the P4 parameterization decision before increasing optimization budget; the current design cannot change hard codes by construction.
2. If strict mid-optimizer whole-model equivalence is required, repeat the existing tiny mid-step identity test at full scale; current full-205 evidence covers uninterrupted versus terminal-checkpoint resume/materialization.
3. Before P7, add a stable immutable source-model/interface identity to the quality schema and compare contract. The current `model.id` is the loaded snapshot locator (and therefore intentionally differs between BF16 and materialized candidates); revision matching plus the independent interface audit covers this screen but should not be the final acceptance contract.
4. Decide the legacy quality-report policy for final acceptance: reject schema-less reports or force them to non-accepting status; only schema-v2 evidence was used in this screen.
5. Revisit P4 optimization budget/parameterization on a bounded subset, with validation-only fail-closed promotion.
6. Start P5/P6 only after a preceding hard saved candidate clears the quality/resource gate.

No quality recovery, paper reproduction, P5/P6 authorization, P7 whole-model acceptance, or release readiness is claimed.
