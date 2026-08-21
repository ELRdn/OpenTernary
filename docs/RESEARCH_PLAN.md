# OpenTernary Research Plan

## 1. Central Research Question

How far can an existing open-weight LLM be pushed toward ternary weight representation while preserving useful capability, especially Japanese capability?

Initial subject:

> Gemma 4 E2B

---

## 2. Research Philosophy

OpenTernary should distinguish:

### Engineering result

"The model can be converted and loaded."

from:

### Research result

"This method retains more capability than baseline X under controlled conditions."

The first is necessary before the second is meaningful.

---

## 3. Primary Hypotheses

### H1 — Naive ternarization will substantially degrade quality

Expected, and useful as the baseline.

### H2 — Different layer types have different ternary sensitivity

Attention, MLP, embeddings, PLE, and LM head should not be assumed equally robust.

### H3 — Calibration/reconstruction can recover part of the naive quality loss

This is the main post-MVP research direction.

### H4 — Calibration data composition affects capability retention

Especially:

- Japanese
- reasoning
- coding
- instruction following

### H5 — Hybrid precision may outperform "100% ternary" on quality-per-byte

A slightly larger model may be substantially more useful.

### H6 — Results may change with model scale

Do not assume E2B findings transfer unchanged to E4B/12B/31B.

---

## 4. Baselines

Every major experiment should compare against relevant baselines.

Minimum:

1. original high-precision model
2. conventional low-bit baseline if available
3. naive ternary
4. calibrated ternary

Optional:

- alternative group sizes
- hybrid precision
- alternative threshold methods

---

## 5. Independent Variables

Examples:

- group size
- scale strategy
- threshold strategy
- target module classes
- excluded layers
- calibration sample count
- calibration token count
- calibration language mix
- optimizer
- learning rate
- epochs/steps
- soft-to-hard schedule
- reconstruction window size

---

## 6. Dependent Variables

### Model quality

- perplexity where appropriate
- instruction-following score
- reasoning score
- coding score
- Japanese score
- qualitative failure taxonomy

### Efficiency

- stored weight size
- process RAM
- VRAM
- load time
- tokens/sec
- calibration time

### Quantization behavior

- weight reconstruction error
- activation reconstruction error
- sparsity / zero ratio
- scale distribution
- layer sensitivity

---

## 7. Experiment Discipline

Each experiment changes as few major variables as practical.

Bad:

```text
change group size
change dataset
change threshold
change optimizer
change exclusions
```

all at once.

Better:

```text
Run A: group=128
Run B: group=64
```

with everything else fixed.

---

## 8. Calibration vs Evaluation

Strict separation:

```text
Calibration Set
≠
Validation Set
≠
Held-out Benchmark
```

Do not tune directly on the final held-out benchmark.

If Japanese prompts are used for calibration, maintain separate Japanese evaluation prompts.

---

## 9. Gemma 4 E2B Experiment Sequence

### Experiment 0

Baseline model only.

Purpose:
validate benchmark pipeline.

### Experiment 1

Naive ternary MLP only.

### Experiment 2

Naive ternary attention only.

### Experiment 3

Naive ternary attention + MLP.

### Experiment 4

Group-size sweep.

### Experiment 5

Layer sensitivity scan.

### Experiment 6

Sensitive-layer hybrid precision.

### Experiment 7

Learnable-scale calibration.

### Experiment 8

Learnable threshold.

### Experiment 9

Soft-to-hard ternary transition.

### Experiment 10

Window/layer reconstruction.

### Experiment 11

Japanese-mixed calibration.

### Experiment 12

Reasoning/Japanese calibration.

### Experiment 13

PLE/embedding precision study.

Only after these should the project seriously optimize packing/runtime.

---

## 10. Reproduction of External Research

When implementing ideas from external papers or repos:

1. record exact source and revision,
2. record which components were reproduced,
3. list intentional deviations,
4. compare on at least one overlapping setup when possible,
5. never label an implementation "CAT-Q" or similar merely because it was inspired by the paper.

Preferred naming during incomplete reproduction:

```text
catq-inspired
scaleq-inspired
soft-ternary-reconstruction
```

until validated.

---

## 11. Negative Results

Negative results belong in the repository.

Examples:

- "group 256 destroys Japanese instruction following"
- "ternarizing PLE increases size efficiency but sharply hurts quality"
- "calibration loss improves while held-out benchmark worsens"

These are valuable research outputs.

---

## 12. Completion Criteria for a Research Claim

A result is ready to be stated publicly when:

- config is saved,
- code commit is saved,
- benchmark split is fixed,
- at least one repeat run exists,
- comparison baseline exists,
- limitations are written,
- raw metrics are available.
