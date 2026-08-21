# OpenTernary Benchmark Protocol

## 1. Purpose

The benchmark system answers two different questions:

1. **Did ternarization preserve capability?**
2. **Was the size/performance trade-off worth it?**

A smaller file alone is not success.

Phase 1b establishes a **BF16 Smoke Inference Baseline** (behavioral, not quality-scored). True quality benchmarks (accuracy, MMLU, Japanese scores, perplexity) are Phase 5.

---

## 2. Required Comparison Matrix

Each release candidate should report:

| Variant | Quality | Japanese | RAM/VRAM | Size | tok/s |
|---|---:|---:|---:|---:|---:|
| Original | baseline | baseline | - | - | - |
| Conventional low-bit baseline | TBD | TBD | TBD | TBD | TBD |
| Naive Ternary | TBD | TBD | TBD | TBD | TBD |
| Calibrated Ternary | TBD | TBD | TBD | TBD | TBD |

Use identical prompts/generation settings for comparable rows.

Phase 1b smoke does **not** fill the Quality column — it records inference behavior (`Δoutput`, latency, tokens/sec) against the frozen BF16 baseline.

---

## 3. Benchmark Layers

### Layer A — Smoke (Phase 1b v1)

**BF16 Smoke Inference Baseline v1** — frozen before any ternarization.

- Suite: `smoke`, version `1`, 5 fixed prompts (see §3.1)
- Model: `google/gemma-4-E2B-it-qat-q4_0-unquantized@6befbaca...`
- Loader: `AutoProcessor` + `AutoModelForMultimodalLM` (`Gemma4ForConditionalGeneration`)
- Thinking: `false` (Baseline v1 — `baseline-smoke-think` is a future variant)
- Generation: `do_sample=false, num_beams=1, max_new_tokens=64` (greedy deterministic; no `temperature/top_p/top_k`)
- Precision: `bf16` exact — silent fallback to `fp32` is forbidden; `bf16` unsupported → loud error `exit 2`
- Artifacts: `benchmark.json` (per-prompt `generated_token_ids`, `raw_output_text`, `parsed_output_text`), `metrics.json` (summary), `protocol_fingerprint` + `result_fingerprint`

Fast checks after every important code change:

- model loads
- deterministic short prompt
- no NaN
- expected output shape
- chat template works
- `result_fingerprint` stable across reruns (latency excluded)

### Layer A — Smoke Suite v1 Prompts

| id | intent | prompt (exact) |
|---|---|---|
| `smoke.en.short` | short English instruction | `Write a one-sentence greeting in English.` |
| `smoke.ja.short` | short Japanese instruction | `日本語で一文の挨拶を書いてください。` |
| `smoke.ja.summary` | fixed summarization | `次の文を日本語で一文で要約してください:「OpenTernaryはGemma 4 E2Bを三値化する研究ツールで、再現可能なCLIパイプラインを構築することを目的としています。」` |
| `smoke.reasoning.simple` | simple reasoning | `If all Bloops are Razzies and some Razzies are Loppies, is it certain that some Bloops are Loppies? Answer yes/no and explain in one sentence.` |
| `smoke.code.python` | short code gen | `Write a Python function add(a, b) that returns a + b. Reply with code only.` |

Suite version `1` and `protocol_fingerprint` (`sha256:` over `suite/version/prompts/thinking/generation`) change whenever this table changes.

### Layer B — Regression

Small fixed suite for development.

Target categories:

- basic knowledge
- instruction following
- short reasoning
- Japanese
- code

### Layer C — Full Evaluation

Slower suite used for milestone comparisons.

### Layer D — Human/Qualitative Review

Especially important for Japanese naturalness.

Human review must never replace quantitative evaluation, but can reveal failure types that aggregate metrics miss.

---

## 4. Japanese Track

Maintain held-out Japanese prompts covering at least:

- natural conversation
- instruction following
- summarization
- factual QA
- reasoning
- coding explanation
- long-form coherence

Do not reuse these exact prompts for calibration.

---

## 5. Generation Controls

Store (and fingerprint) for every benchmark:

- `suite` name + `version` + `protocol_fingerprint`
- `thinking` mode
- `do_sample`, `num_beams`, `max_new_tokens`
- `seed`, `dtype` (requested vs actual), `device`
- `model` id + `revision` + `snapshot_path`
- chat template (via `AutoProcessor.apply_chat_template`)
- per-prompt `generated_token_ids` + `result_fingerprint`

Phase 1b explicitly does **not** store `temperature/top_p/top_k` — greedy baseline has no sampling params.

Comparisons are invalid if generation settings drift unnoticed. `protocol_fingerprint` catches drift.

---

## 6. Efficiency Metrics

Measure separately (Phase 1b records `model_load_ms`, `avg_generation_latency_ms`, `output_tokens_per_second`):

### Storage

- original weights
- fake-quant artifact
- packed artifact

### Memory

- peak RAM
- peak VRAM
- resident model memory

### Speed

- load time (`model_load_ms`)
- prompt processing rate
- generation rate (`output_tokens_per_second = total_output_tokens / total_generation_seconds`)
- warmup is excluded from timing (1 prompt warmup before measurement)

Fake-quant speed is not representative of a future packed ternary kernel and must be labeled accordingly.

---

## 7. Quantization Diagnostics

For each quantized layer/module:

- reconstruction error
- zero ratio
- scale statistics
- max/mean weight error
- activation error if captured

This makes benchmark failures diagnosable.

---

## 8. Contamination Rules

Any data used to optimize quantization parameters is calibration/training data.

It must not be counted as held-out evaluation.

`smoke` suite is **held-out from calibration** by definition. Document all overlap risks.

---

## 9. Benchmark Output

Phase 1b machine-readable (`runs/<slug>/benchmark.json`):

```json
{
  "suite": {"name": "smoke", "version": "1", "fingerprint": "sha256:..."},
  "model": {"id": "...", "revision": "..."},
  "thinking": false,
  "generation": {"do_sample": false, "num_beams": 1, "max_new_tokens": 64},
  "dtype": {"requested": "bf16", "actual": "torch.bfloat16"},
  "model_load_ms": 1234.5,
  "warmup": {"executed": true},
  "results": [
    {
      "id": "smoke.en.short",
      "prompt": "...",
      "raw_output_text": "...",
      "parsed_output_text": "...",
      "generated_token_ids": [123, 456],
      "input_tokens": 10,
      "output_tokens": 12,
      "latency_ms": 234.5
    }
  ],
  "result_fingerprint": "sha256:...",
  "summary": {
    "num_prompts": 5,
    "total_output_tokens": 60,
    "total_generation_ms": 1200,
    "avg_generation_latency_ms": 240,
    "output_tokens_per_second": 50.0
  }
}
```

`metrics.json` is the aggregated view derived from the above.

Human-readable reports may be generated from the same data.

---

## 10. First MVP Benchmark

Do not overbuild.

For the first Gemma 4 E2B naive experiment, require only:

- smoke suite (Phase 1b v1 — inference baseline, frozen)
- small general regression suite
- small held-out Japanese suite
- memory/size estimate
- quantization reconstruction error

The benchmark system can expand after the end-to-end pipeline works.
