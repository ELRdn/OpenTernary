# P4 soft-to-hard redesign decision

Status: **DSH decision required / full-model experiments blocked**

Date: 2026-09-22

## Observed result

Linear, cosine, and exponential schedules completed all 205 targets under the same validation-only protocol. All three produced `code_change_ratio=0`, instruction score `0`, and collapse count `8/8` after hard materialization and reload. No schedule passed the quality gate, so the preregistered modulation row was not opened.

The dominant local residual is stable across candidates. Attention `q_proj` contributes 86.47% of aggregate held-out squared error. Layer 0 attention `q_proj` alone contributes about 47.2% and remains near held-out MSE 291 for scale-only, threshold, and all three schedules.

The existing threshold control does technically change hard assignments, but not enough to address that residual. A direct sign comparison of the saved scale-only and threshold snapshots matches the recorded aggregate exactly: 166,481 / 1,835,532,288 codes changed (`9.0699031e-5`). The dominant layer-0 `q_proj` changed only 343 / 3,145,728 codes (`1.09037e-4`). This is evidence for a bounded trainability test, not evidence that a larger run will recover quality.

## Root cause in the current contract

The optimizer owns reconstruction scales only. `soft_ternary_codes` computes an expectation from frozen source weights and frozen reference scales, but neither assignment logits nor the hardening boundary is trainable. Final materialization calls deterministic midpoint hardening from those same frozen values.

Therefore:

- temperature changes the soft expectation used while fitting scales;
- `zero_logit_bias` changes only that soft expectation;
- final hard codes are independent of both values;
- increasing steps cannot make the saved assignment change under this parameterization.

This contract is now machine-readable in calibration reports/checkpoints:

```json
{
  "trainable": false,
  "hardening_source": "frozen-weight-midpoint",
  "hardening_uses_zero_logit_bias": false
}
```

## Decision options

### A. Keep the current relaxation and rename its claim

Treat temperature annealing as scale-optimization conditioning only. Remove any claim that it learns ternary assignments. This is the smallest implementation change, but it does not address `code_change_ratio=0` or current quality collapse.

### B. Anneal a learnable threshold into the hard path — recommended bounded direction

Reuse the existing per-group threshold parameterization and clipped-STE checkpoint/materialization contract. Add temperature only as a surrogate-width or soft-gate schedule while preserving the learned threshold in final hardening. This has the lowest state and integration risk because threshold persistence, resume, and materialization already exist.

### C. Learn explicit assignment logits

Persist logits that determine the final ternary code. Per-weight three-way logits would add roughly three values per quantized weight and is incompatible with the present memory budget without a more compact design. Per-group logits alone cannot represent independent signs/codes inside a group. This option requires a new storage and optimizer design.

### D. Learn a group offset or asymmetric boundary

Persist one or more groupwise boundary/offset parameters and make final hardening consume them. This is cheaper than per-weight logits but changes the quantizer and export contract, so it must remain separate from rotation research.

## Recommended next experiment after DSH approval

Choose option B and run a bounded validation-only progression:

1. one Linear: `model.language_model.layers.0.self_attn.q_proj`;
2. compare threshold STE against threshold plus temperature schedule with identical samples, tokens, steps, seed, and optimizer budget;
3. require nonzero saved hard-code change, finite gradients, checkpoint/resume identity, and materialize/reload identity;
4. advance to one block only if hard held-out reconstruction improves over threshold control;
5. do not run all 205 targets until the saved hard candidate clears the existing quality and resource gates.

No test-split access, P5 window expansion, P6 rotation work, or full-model run is authorized by this note.
