---
description: "Run directories, the four separate verdicts (execution, artifact validation, model reload, quality acceptance), the quality gate, data splits and status symbols."
---

# Runs and evidence

OpenTernary is built around one rule: **a number is only as good as the record of how it was produced.** This
page explains how that record is kept and how to read it.

## One experiment, one directory

Every command that does real work (`inspect`, `quantize`, `calibrate`, `benchmark`, `quality`, `evaluate`,
`compare`, `optimize`) creates a **run directory**:

```text
runs/2026-11-03_tiny-llama-naive-g128/
├── config.yaml        # the fully resolved configuration: CLI flags + YAML + defaults
├── environment.json   # OpenTernary version, git commit, Python, packages, OS, CPU/GPU, RAM, seed
├── model.json         # model id and revision
├── metrics.json       # status ("completed", "failed", …) and headline numbers
├── <result>.json      # quantization.json, quality.json, compare.json, … (one per command)
├── logs/openternary.log
└── artifacts/         # model snapshots, checkpoints, caches
```

- **Name:** without `--output`, runs go to `runs/<date>_<model>-<method>-g<group size>`. If that exists,
  `-001`, `-002`, … is appended, so nothing is ever overwritten. With `--output`, that path is used, with the
  same suffix rule.
- **Atomic writes:** JSON files are written to a temporary file and renamed, so an interrupted run never leaves a
  half-written `metrics.json`.
- **Locks:** a running command holds `.execution.lock`. A second process trying to resume or finalize the same
  run is refused, and so is deleting it from the TUI.
- **Failures are recorded too.** If a command fails after creating its run, `metrics.json` gets
  `status: failed` (or `interrupted`, exit code 130), the error message and the exit code. If the run had
  already completed once and a later retry fails, the earlier result is kept and the new failure goes to
  `last_failure.json`.

[Run directory reference](../reference/run-directory.md) lists every file.

!!! tip "The reproducibility unit"

    `config.yaml` + `environment.json` + `model.json` (with the revision) + the lockfile `uv.lock` in a
    source checkout are meant to be enough to repeat a run. Keep them with any number you report.

## Four questions, four answers

"It worked" can mean several different things. OpenTernary answers each question separately and never lets one
stand in for another:

| Question | Field | Possible values | What it does *not* tell you |
|---|---|---|---|
| Did the command finish? | `execution` | `planned`, `completed`, `failed`, `interrupted` | whether the result is any good |
| Are the files intact? | `artifact_validation` | `not_run`, `file_integrity`, `tensor_integrity` | whether the model loads |
| Does the model load again? | `model_reload` (in the manifest) | `not_run`, … | whether it is accurate |
| Is the quality acceptable? | `quality_acceptance` | `not_run`, or `{"accepted": …, "basis": …}` | anything outside the measured protocol |

Every `--json` result carries the first, second and fourth at the top level. A finished quantization is
`execution: completed`, `quality_acceptance: not_run`, and **that is the normal, honest outcome** of a
quantization. Quality is only ever established by comparing measurements.

!!! warning "`completed` is not success"

    In tables and the TUI, `completed` is shown with a neutral `●`, never a green tick. Only an accepted quality
    verdict gets `✔`.

## Comparing: observation vs acceptance

`compare` always checks first that the two runs are comparable: the same protocol, dataset, quality schema,
model revision, dtype and device. It refuses (exit code 2) when they are not.

**Benchmark comparisons are observations.** The smoke benchmark records the exact tokens each prompt produced.
`compare` reports whether the outputs are identical (`same` / `different`), prompt by prompt. Different output
from a quantized model is expected and is not a failure. Whether it is *worse* is a quality question.

**Quality comparisons can be accepted or rejected** by a fixed, preregistered gate, `balanced-quality-gate-v1`.
For a candidate \(c\) against its matched baseline \(b\):

$$
\Delta_{\text{gen}} = \frac{\text{PPL}^{\text{en}}_c - \text{PPL}^{\text{en}}_b}{\text{PPL}^{\text{en}}_b}, \quad
\Delta_{\text{ja}} = \frac{\text{PPL}^{\text{ja}}_c - \text{PPL}^{\text{ja}}_b}{\text{PPL}^{\text{ja}}_b}, \quad
\Delta_{\text{inst}} = \text{score}_c - \text{score}_b
$$

$$
\text{composite} = 0.35\,(-\Delta_{\text{gen}}) + 0.35\,(-\Delta_{\text{ja}}) + 0.30\,\frac{\Delta_{\text{inst}}}{100}
$$

The candidate is **accepted** only if all of these hold:

| Gate | Condition |
|---|---|
| composite | \(> 0\) |
| general (English) perplexity | rises by no more than 2% |
| Japanese perplexity | rises by no more than 2% |
| instruction score | drops by no more than 2 points |
| collapse | 0 collapsed cases |

The gate is **relative**: a candidate is judged against its own baseline. It is applied only when the evidence
is complete:

- both reports use quality report **schema 3**;
- the source model's identity is resolved and identical, with the same content fingerprints for weights and for
  the tokenizer/processor interface;
- the measurement environment is identical: device, dtype, library versions, OS;
- the model is real: a model marked synthetic (`openternary-fixture.json`) is never accepted.

Otherwise the verdict is `accepted: false` with `status: insufficient_evidence`. `compare --require-acceptance`
turns any non-acceptance into exit code **6**. The thresholds are fixed in code and in the research plan.
Changing them is a research decision, not a configuration option.

## Data splits

Quality datasets have three disjoint splits:

| Split | Used for |
|---|---|
| `calibration` | fitting: calibration data, never evaluation |
| `validation` | choosing between candidates (the default for `quality`) |
| `test` | the final, one-time measurement of the chosen candidate |

OpenTernary checks that no text appears in two splits, both exact matches after normalization and near-duplicates
(character 5-gram MinHash, similarity ≥ 0.9), and refuses datasets that overlap. `optimize` can never select
candidates on the test split.

## Unknown is not zero

When something was not measured, the result says `null`, never `0`. Peak VRAM on a CPU run, speed when the
measurement conditions differ, and model quality after a plain quantization are all `null` or `not_run`. Speed
rankings are only produced when the measurement conditions are identical.

## Status symbols

Human-readable output uses a symbol plus a word. The colour is on the symbol only, so the meaning survives
copy-paste and colour blindness.

| Symbol | Meaning | Examples |
|---|---|---|
| `✔` | accepted / passed | quality accepted, device test passed |
| `●` | installed / completed | a package is installed, a command finished |
| `◐` | implemented but unverified | conversion implemented, not yet validated on real models |
| `▲` | experimental / caution | backend quality is experimental, update available |
| `○` | not run / not installed | quality not run, optional package absent |
| `–` | unsupported | native low-bit execution for the ternary backend |
| `✖` | failed / rejected | a failed command, a rejected quality comparison |
