# P0–P7 research acceptance plan

> Status: Active, 2026-09-22
> Target: Gemma 4 E2B, canonical 205 Linear modules, per-group G128, BF16
> Model: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
> Revision: `6befbaca7398925921802abd1f277b495b78b738`

## 1. Decision boundaries

- Research acceptance and paper-grade reproduction are separate outcomes.
- Calibration/validation select candidates. The final test split is used once after selection.
- A tiny fixture, local reconstruction improvement, smoke output, or 205-file count is never a quality claim.
- Adoption is based on the final hard, saved, reloaded snapshot—not a soft training state.
- Missing source, data, cache, checkpoint, or protocol evidence stops the run.
- No target-model, canonical-target, calibration/evaluation split, or public storage-contract change is implicit.

## 2. Resource envelope

| Item | Limit |
|---|---:|
| GPU time | 24 hours per research phase |
| Active artifacts | 120 GB |
| Automatic stop | 80% of either limit without a passing intermediate gate |
| Seeds | one for screening; at least three for the selected P7 candidate |

Every long run records wall time, peak RAM/VRAM, disk growth, effective microbatch, retry count, model/source hashes, code revision, config, and data fingerprints.

## 3. Fixed quality gate

For candidate `c` versus its matched control `b`:

```text
general_change  = (general_ppl_c - general_ppl_b) / general_ppl_b
japanese_change = (japanese_ppl_c - japanese_ppl_b) / japanese_ppl_b
instruction_delta = instruction_score_c - instruction_score_b

composite = 0.35 * (-general_change)
          + 0.35 * (-japanese_change)
          + 0.30 * (instruction_delta / 100)
```

All conditions must pass:

- `composite > 0`
- general PPL regression `<= 2%`
- Japanese PPL regression `<= 2%`
- instruction regression `<= 2` points
- collapse count `== 0`
- protocol and data fingerprints match the control

## 4. Phase plan and gates

| Phase | Work | Exit gate | Current state |
|---|---|---|---|
| P0 | Reconcile exact source revision/weights, canonical targets, artifacts, runtime, roadmap, and budgets | Preflight PASS with source/target/protocol/runtime hashes | PASS; canonical revision, weight SHA-256, exact 205-name inventory, path-independent inspection fingerprint, and external-CWD invocation verified |
| P1 | Exclude padding; element-weight losses; honor config; immediate microbatch backward; transactional OOM shrink/retry | Tiny production path proves calculation, gradients, rollback, and memory-control reporting | PASS; actual ROCm OOM recovery plus transactional real-path test |
| P2 | Checkpoint final module, optimizer/RNG, source/cache identity, fail-closed missing data, deterministic materialize | Uninterrupted and resumed runs agree; saved/reloaded hard output agrees | Tiny mid-step identity PASS; full 205 uninterrupted vs terminal-checkpoint resume has identical loss history, fingerprints, and all 22 snapshot file hashes; six learned runs' 9,840 cache shards and 1,230 state files rehashed successfully |
| P3 | General/Japanese PPL, instruction following, collapse checks; BF16/naive/scale-only/threshold controls | Frozen disjoint data and verified matched evaluation implementation | Evaluation gate PASS; schema-v2 reports preserve prior metrics, save hash-verified responses, and audit instruction near-duplicates; all ternary controls fail quality on validation |
| P4 | Compare temperature schedules and zero-code modulation independently | Winner passes on final hard saved validation snapshot | Production integration PASS; linear/cosine/exponential hard snapshots FAIL; no winner |
| P5 | Scale Linear → one block → several adjacent blocks; include upstream quantization error | Improvement over layer-local under the same calibration budget, with cost reported | Not started |
| P6 | Prove non-quantized equivalence → fixed rotations → learned rotation/asymmetric ablations | Correct input/export contract and validation-selected candidate | Partially exercised: learned-rotation CLI input contract and q35 save/reload verified; 200-step q35 passed v4 validation but failed independent v5 test. Fixed/asymmetric ablations and phase acceptance remain open. |
| P7 | Apply selected configuration to all 205 targets; independent test and repeat seeds | Quality, reproducibility, resource, implementation, docs, CLI, and experiment record all pass | Exploratory saved 50-step and in-memory 200-step 205-target candidates both failed v4 validation; no selected P6 configuration or P7 acceptance. |

## 5. P3 matched controls

Use identical source revision, tokenizer, target set, split fingerprints, dtype, context/window policy, generation protocol, and evaluation code.

| Control | Purpose |
|---|---|
| BF16 | Upper behavioral reference |
| naive ternary G128 | Static quantization baseline |
| scale-only calibration | Isolate learned reconstruction scale |
| threshold calibration | Isolate threshold contribution |

Do not compare against an artifact with a different target set, missing source hash, recovered metadata, or different data budget.

## 6. P4 experiment matrix

Screen on validation with one seed and a fixed calibration budget:

1. threshold STE control
2. soft-to-hard linear schedule, no modulation
3. soft-to-hard cosine schedule, no modulation
4. soft-to-hard exponential schedule, no modulation
5. winning schedule plus one preregistered zero-code bias (not opened when rows 2–4 all fail)

Only one factor changes per row. The last 10% of optimizer steps is hard. Checkpoints record schedule, start/end temperatures, hard fraction, modulation, and current hard/soft state. Materialization rejects incomplete soft-state runs.

Current gate result: **FAIL / architecture decision required.** The implemented relaxation has no trainable assignment parameter. It optimizes reconstruction scales while both the source weights and reference scales used by final midpoint hardening remain frozen. Consequently, schedule choice and zero-code bias cannot change the saved hard codes. Do not spend a larger calibration budget on this parameterization. See [P4 soft-to-hard redesign decision](p4-soft-to-hard-redesign-decision.md).

## 7. P5 progression

1. One Linear: prove parity with P4 and measure local gain.
2. One block: reconstruct block output with every included module named.
3. Small adjacent-block window: feed activations produced by already quantized upstream modules.

At every stage, compare with layer-local under the same samples, tokens, optimizer steps, and wall/GPU budget. Record additional memory and time separately from quality.

## 8. P6 progression

1. Non-quantized FP32/BF16 function preservation for one Linear, one block, then text path.
2. Fixed normalized Hadamard and random signed Hadamard controls.
3. Learned rotation and asymmetric quantizer as separate factors before their combination.

The saved artifact includes transform type/version/dimensions/seed and all scale/offset metadata. Rotated ternary weights without the matching input transform are invalid.

## 9. P7 acceptance package

Required outputs:

- immutable source, code, config, tokenizer, and data fingerprints
- complete checkpoint/cache/state integrity manifests
- hard materialized snapshot and reload fingerprint
- per-module and aggregate reconstruction metrics
- matched P3 quality results on validation and untouched test
- at least three selected-candidate seeds with variance
- actual wall time, peak RAM/VRAM, disk, and OOM/retry records
- CLI commands and an experiment log sufficient for rerun
- explicit negative results and exclusions

P7 fails closed if any required target, metric, provenance field, seed, or artifact hash is missing.

## 10. Immediate next actions

1. Obtain the architecture decision in [P4 soft-to-hard redesign decision](p4-soft-to-hard-redesign-decision.md); the current parameterization cannot change saved hard codes.
2. After that decision, add only a bounded validation-only P4 redesign; do not use the frozen final test split.
3. Repeat the tiny mid-optimizer interruption test at full scale only if strict whole-model P2 equivalence is required; uninterrupted versus terminal-checkpoint identity is already proven for all 205 targets.
4. Before P7, embed a stable source/interface identity in quality reports and decide whether schema-less legacy quality reports are rejected or forced non-accepting.
5. Keep P5/P6 closed until a final hard saved candidate passes the preceding quality/resource gate.
