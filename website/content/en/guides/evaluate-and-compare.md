---
description: "Run the smoke benchmark and frozen English/Japanese quality evaluation, build a quality dataset, and compare runs against the quality gate."
---

# Evaluate and compare

OpenTernary measures a model in two ways and compares measurements in one:

| Command | Answers | Verdict |
|---|---|---|
| `benchmark` | Does it run? How fast? What exactly does it generate for 5 fixed prompts? | none: behaviour only |
| `quality` | How good is it at English and Japanese text, and at exact-answer instructions? | none on its own |
| `compare` | How do two runs differ under identical conditions? | quality gate, for quality runs |

Always measure the **original model** with exactly the same settings first. Every judgement is relative to that
baseline.

## `benchmark`: the smoke suite

```bash
openternary benchmark google/gemma-4-E2B-it-qat-q4_0-unquantized --device cuda --dtype bf16 --output runs/e2b-bench
openternary benchmark runs/e2b-g128/artifacts/snapshot          --device cuda --dtype bf16 --output runs/e2b-g128-bench
```

The suite `smoke v1` runs five fixed prompts, using the model's chat template, with greedy decoding
(`do_sample: false`, `num_beams: 1`, up to 64 new tokens), after one warm-up prompt:

| ID | Prompt |
|---|---|
| `smoke.en.short` | Write a one-sentence greeting in English. |
| `smoke.ja.short` | 日本語で一文の挨拶を書いてください。 |
| `smoke.ja.summary` | Summarize, in one Japanese sentence, a sentence about OpenTernary |
| `smoke.reasoning.simple` | A one-sentence syllogism question (Bloops, Razzies, Loppies) |
| `smoke.code.python` | Write a Python function add(a, b) that returns a + b. Reply with code only. |

`benchmark.json` records:

- the **protocol fingerprint**, a hash of the suite, prompts, chat-template setting and generation settings.
  Two runs are comparable only if it matches;
- the **result fingerprint**, a hash of all generated token IDs. Identical output gives an identical fingerprint;
- every prompt's generated tokens and text, latency, output tokens per second and model load time;
- resource measurements (RAM, and VRAM on GPUs) per phase, when they could be measured, and `null` otherwise.

Options: `--dtype bf16|fp16|fp32` (default `bf16`), `--device auto|cpu|cuda`, `--seed`, `--thinking/--no-thinking`
(Gemma 4's thinking mode, off by default). Generation settings can be changed in the config's `benchmark`
section. Changing them changes the protocol fingerprint.

The smoke suite has no quality score on purpose. It is for checking that a model runs, and for seeing *whether*
its outputs changed.

## `quality`: perplexity and instructions

```bash
openternary quality google/gemma-4-E2B-it-qat-q4_0-unquantized --data quality.json --device cuda --dtype bf16 --output runs/e2b-q
openternary quality runs/e2b-g128/artifacts/snapshot          --data quality.json --device cuda --dtype bf16 --output runs/e2b-g128-q
```

`quality` evaluates one split (`--split validation` by default, or `test`) of a frozen dataset:

**Perplexity** is computed separately for English and Japanese texts with a sliding window (`--max-length 512`,
`--stride 256`). Each document is scored independently, with no padding or truncation. Lower is better. It is
reported per language and overall.

**Instruction cases** are prompts with one exact expected answer. A response scores only if it is **exactly**
equal to `expected`. There is no normalization and no LLM judge, so spacing and punctuation count. The
`instruction_score` is the percentage of exact matches.

**Collapse detection** flags responses that are empty, contain only symbols, or consist of one short unit
repeated (for example `ああああああああ`). `collapse_count` is the number of flagged responses. Any collapse fails the
quality gate.

`quality.json` (report schema 3) also records the identity of the source model and its tokenizer/processor files
(content hashes), the device and dtype that were actually used, library and OS versions, and resource
measurements. `compare` needs all of these to accept a result.

`--dry-run` validates the dataset (schema, provenance, splits, overlaps) without loading the model.

### Quality datasets

A quality dataset is one JSON file with provenance, three text splits and two instruction splits:

```json
{
  "schema_version": 1,
  "dataset": {
    "id": "your-org/quality-v1",
    "revision": "v1",
    "license": "CC-BY-4.0",
    "source": [{"id": "where the texts came from", "revision": "…"}]
  },
  "splits": {
    "calibration": [{"id": "c1", "text": "…", "language": "en"}],
    "validation":  [{"id": "v1", "text": "…", "language": "en"}, {"id": "v2", "text": "…", "language": "ja"}],
    "test":        [{"id": "t1", "text": "…", "language": "en"}, {"id": "t2", "text": "…", "language": "ja"}]
  },
  "instruction_cases": {
    "validation": [{"id": "vi1", "prompt": "…", "expected": "…"}],
    "test":       [{"id": "ti1", "prompt": "…", "expected": "…"}]
  }
}
```

Rules, all enforced:

- `dataset.id`, `revision`, `license` and `source` are required. Provenance is part of the evidence.
- All three text splits and both instruction splits must exist and be non-empty. Every record needs `id`,
  `text` and `language`, and every case needs `id`, `prompt` and `expected`.
- IDs are unique within a split, and instruction IDs are unique across both splits.
- **No overlap:** no text may appear in two splits, either exactly (after Unicode NFKC and whitespace
  normalization) or as a near-duplicate (character 5-gram MinHash similarity ≥ 0.9).

The research dataset `OpenTernary/quality-squad-jsquad-v1` is built deterministically from WikiText (calibration),
SQuAD (English) and JSQuAD from JGLUE (Japanese). It uses 8 contexts per language per split and 4 instruction
cases per language per split, selected by the SHA-256 of record IDs. The builder is
`openternary.benchmark.quality_dataset.build_quality_dataset`. The source licences (CC-BY-SA, CC-BY) apply to the
built file.

!!! warning "Use the test split once"

    Choose between candidates on `validation`. Run `--split test` only for the final, chosen candidate.
    Evaluating many candidates on the test split and keeping the best one turns it into a validation set.

## `compare`

```bash
openternary compare runs/e2b-q runs/e2b-g128-q                         # observe
openternary compare runs/e2b-q runs/e2b-g128-q --require-acceptance    # gate: exit 6 if not accepted
openternary compare runs/e2b-q runs/e2b-g128-q --candidate runs/e2b-g64-q --candidate runs/e2b-calib-q
```

- The first argument is the **baseline**, the second the **candidate**. Add more candidates with `--candidate`
  (repeatable). Each is compared with the same baseline.
- Both runs must be the same kind (two benchmark runs, or two quality runs) with matching protocol, dataset,
  model revision, dtype and device. Otherwise `compare` stops with exit code `2` and names the mismatch.
- **Benchmark runs** give an observational comparison: is the output the same, overall and prompt by prompt?
- **Quality runs** are judged by the [quality gate](../concepts/runs-and-evidence.md#comparing-observation-vs-acceptance)
  when the evidence is complete. Perplexity may rise by at most 2% in each language, the instruction score may
  drop by at most 2 points, there may be no collapses, and the weighted composite must be positive.
- `--require-acceptance` makes rejection or incomplete evidence exit with code `6`, for scripts and CI.

`compare` writes a run with `compare.json` (all details) and `acceptance.json` (the verdict):

```json
{"accepted": false, "status": "insufficient_evidence",
 "reason": "schema 3 with matched source/interface/runtime and non-synthetic model evidence required"}
```

When the gate was applied, the verdict contains the per-metric changes, the composite score and the list of
failed gates, for example `["general_ppl_regression", "collapse"]`.

!!! tip "Pass `--output` to compare"

    Without `--output`, a compare run is named after the default model in the configuration
    (`…_gemma-4-E2B-it-qat-q4_0-unquantized-naive-g128`), not after the runs you compared, which can be confusing.

## In the TUI

`/compare` lets you pick the baseline and candidate from your runs. Alternatively, highlight a run in the
**Runs** tab and press ++b++ or ++c++ (or type `/baseline` and `/candidate`). The **Runs** table shows each
run's quality verdict.

## Migrating from older versions

| Old | Now |
|---|---|
| `benchmark --suite baseline` | `--suite smoke` (the only suite) |
| `compare` with nothing to compare succeeded | input error (exit `2`), recorded in the run |
| quality reports without a schema could pass | observational only, `accepted: false` |
