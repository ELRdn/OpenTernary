---
description: "Search a list of quantization configurations within time, GPU-time, VRAM and disk budgets, with resumable journals and no selection on the test split."
---

# Constrained search

`optimize` tries several quantization configurations, measures each one, and picks the best one that passes the
quality gate and fits your VRAM target. It never invents configurations, never changes the model or the data, and
never looks at the test split.

## Inputs

**A search space**: a JSON list of quantization configurations, in the same form as the config file's
`quantization` section:

```json title="space.json"
[
  {"backend": "ternary", "scheme": "absmean", "weight_dtype": "ternary", "scale_granularity": "per_tensor"},
  {"backend": "ternary", "scheme": "absmean", "weight_dtype": "ternary", "scale_granularity": "per_group", "group_size": 128},
  {"backend": "ternary", "scheme": "absmean", "weight_dtype": "ternary", "scale_granularity": "per_group", "group_size": 64}
]
```

**A quality profile**: which validation data to measure on, and which baseline to compare with. Paths are
relative to the profile file:

```json title="profile.json"
{"schema_version": 1, "name": "balanced-llm-v1", "split": "validation",
 "dataset": "validation.jsonl", "baseline": "baseline-run", "max_length": 512, "stride": 256}
```

`split` can only be `validation`. `baseline` is a run directory with a `quality.json` of the original model,
measured the same way.

## Running a search

```bash
openternary optimize ./my-model \
    --search-space space.json --quality-profile profile.json \
    --target-vram 12GiB --objective speed \
    --max-candidates 4 --wall-seconds 3600 --gpu-seconds 3600 --max-artifact-bytes 20GiB \
    --output runs/search --dry-run
```

Remove `--dry-run` to run it. To continue an interrupted search, repeat the same command with `--resume`.

| Option | Default | Meaning |
|---|---|---|
| `--target-vram` | required | a candidate is feasible only if its measured inference VRAM is at or below this (`12GiB`, `8000MB`, …) |
| `--objective` | `speed` | among feasible candidates: `speed` picks the lowest latency, `size` the smallest artifact |
| `--max-candidates` | 4 | at most this many candidates are evaluated (1–256) |
| `--wall-seconds` | 3600 | total wall-clock budget |
| `--gpu-seconds` | 3600 | total GPU-time budget (non-CPU workers count their wall time, conservatively) |
| `--max-artifact-bytes` | `20GiB` | total disk budget for candidate artifacts |
| `--config` | | base configuration (device for evaluation, model revision, …) |

## How it works

1. Each candidate runs in a **fresh worker process**. It converts the model (on the CPU), saves the artifact,
    reloads it, measures quality on the validation split and records latency and memory.
2. The result is checked with the [quality gate](../concepts/runs-and-evidence.md#comparing-observation-vs-acceptance)
    against the profile's baseline. A candidate is **feasible** only if it is accepted *and* fits `--target-vram`.
3. Every step is written to a **journal**. On `--resume`, a finished candidate is reused only if its source,
    configuration, profile, baseline, runtime and code hash all match. Tampered outputs or measurements are
    detected and rejected.
4. Disk usage is monitored every 100 ms. Between checks a candidate can briefly go over the limit.
5. A GPU evaluation that fails is a failure. It is never replaced by a CPU result.

## Outcomes

| Outcome | Exit code | Meaning |
|---|---|---|
| a best candidate | `0` | the feasible candidate that is best for the objective; `search.json` has all candidates |
| `no_feasible_candidate` | `6` | every candidate was rejected by the gate or exceeded the VRAM target |
| `budget_exhausted` | `7` | a budget ran out before the search finished |

In the last two cases **no best candidate is published**. A "best of the failures" is never presented as a result.

The search runs candidates in separate processes, so the parent's progress events show one start and one finish
per candidate (`search.candidates`), not each candidate's internal conversion steps.

## Limits

The search machinery (selection, rejection, budgets, resume, tamper detection) has been verified with controlled
fixtures and synthetic models. Its usefulness on real models, and the accuracy of peak-VRAM and speed measurements
on real devices, are still to be established.
