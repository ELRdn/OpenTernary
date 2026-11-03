---
description: "The --json result envelope, exit codes, progress events (JSONL) with every stage, and the human-readable output conventions."
---

# Output, exit codes and events

These are OpenTernary's machine-readable contracts. They are versioned, and the TUI depends on them as much as
your scripts do. [Automation](../guides/automation.md) shows how to use them.

## The `--json` envelope

With `--json`, stdout carries exactly one JSON document, on one line. Human output goes to stderr.

```json
{
  "schema_version": 1,
  "command": "plan",
  "exit_code": 0,
  "execution": "planned",
  "artifact_validation": "not_run",
  "quality_acceptance": "not_run",
  "result": {},
  "error": null
}
```

| Field | Type | Values |
|---|---|---|
| `schema_version` | int | `1` |
| `command` | string | the command, e.g. `quantize`, `cache remove` |
| `exit_code` | int | the process exit code (see below) |
| `execution` | string | `planned` (a successful plan or dry run), `completed` (exit 0, or 6: it ran, the verdict was negative), `interrupted` (130), `failed` (anything else) |
| `artifact_validation` | string | `not_run`, `file_integrity`, `tensor_integrity` |
| `quality_acceptance` | string or object | `not_run`, or for `compare` an object such as `{"accepted": false, "basis": "validated quality reports only"}` |
| `result` | object | command-specific: paths, run ID, metrics, reports |
| `error` | string or null | `null` on success, otherwise the error message (the same text the error panel shows) |

`execution: completed` never means that quality was accepted, and `artifact_validation: file_integrity` never means
the model loads. The content does not depend on the display language.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | the operation or plan succeeded |
| `1` | unexpected runtime or I/O failure |
| `2` | invalid or missing configuration, argument or input |
| `3` | unsupported capability, or plugin API mismatch |
| `4` | a required dependency is not installed |
| `5` | required hardware or device unavailable |
| `6` | explicit acceptance criteria not met, or no feasible candidate |
| `7` | search budget exhausted |
| `130` | interrupted by the user |

## Progress events (JSONL)

`--events-jsonl FILE` writes one JSON object per line to a new file:

```json
{"schema_version": 1, "sequence": 3, "command": "quantize", "run_id": "e67150d8",
 "stage": "convert.tensors", "completed": 0, "total": 21, "unit": "tensor",
 "elapsed_s": 1.776, "detail": {"tensor": "lm_head.weight"}}
```

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `sequence` | int | increases by one per event within an operation |
| `command` | string | the command |
| `run_id` | string or null | `null` until the run exists |
| `stage` | string | see the table below |
| `completed`, `total` | int or null | progress within the stage. `null` when unknown |
| `unit` | string or null | what is counted (`tensor`, `step`, `batch`, …) |
| `elapsed_s` | number | seconds since the operation started |
| `detail` | object | stage-specific extra fields. Always an object. Non-finite numbers become `null` |

Fine-grained stages are throttled to at most **5 lines per second per stage**. The first line of a stage and the line
where `completed == total` are always written. The file must not exist beforehand, and `--events-jsonl` cannot be
combined with `--dry-run`.

### Stages

| Stage | Unit | `detail` | Emitted by |
|---|---|---|---|
| `started` / `completed` / `failed` / `interrupted` | | | every command |
| `run_created` | | | when the run directory is created |
| `phase` | | `name`, `device` | resource-measurement boundaries (load, inference, …) |
| `convert.tensors` | tensor | `tensor` | `quantize`, search candidates |
| `convert.flush` | | `shard`, `tensors` | a shard starts being written to disk |
| `convert.hash` | shard | | hashing written shards |
| `calibrate.dataset` | | `dataset`, `samples`, `seq_len` | preparing calibration text and tokenizer (first use includes a download) |
| `calibrate.load_teacher` | | `snapshot` | loading the original model |
| `calibrate.cache` | | `status` (`capturing` / `reused`), `source` | the activation cache |
| `capture.batches` | batch | `split` (`train` / `heldout`) | capturing teacher inputs and outputs |
| `calibrate.prepare` | | `modules` | recording the cache, releasing the teacher, preparing module state |
| `calibrate.modules` | module | `module`; on completion `last_loss` | start and end of each target layer |
| `calibrate.steps` | step | `module`, `module_index`, `modules`, `loss` | optimizer steps within a layer |
| `calibrate.checkpoint` | | `checkpoint`, `cursor` | a checkpoint was saved |
| `calibrate.evaluate` | module | `module` | train / held-out reconstruction error |
| `calibrate.materialize` | | `destination` | writing the calibrated snapshot |
| `materialize.tensors` | tensor | `tensor` | … per tensor |
| `materialize.flush` | | `shard`, `tensors` | … a shard starts being written |
| `materialize.hash` | shard | | … hashing |
| `calibrate.report` | | | writing metrics and `calibration.json` |
| `benchmark.prompts` | prompt | `prompt` | `benchmark` |
| `quality.perplexity` | language | `language` | `quality` |
| `quality.cases` | case | `case` | `quality` |
| `export.tensors` | tensor | `tensor` | packed export and restore |
| `export.converter` | | `status`, `converter` | the external GGUF converter |
| `validate.tensors` | tensor | `tensor` | `artifacts validate --level tensors` |
| `candidate_started` | | | `optimize`: a candidate starts |
| `search.candidates` | candidate | `status`, `folder` | `optimize`: a candidate finished |
| `diffusion.prompts` | prompt | | diffusion evaluation |

Search candidates run in separate processes, so their internal `convert.*` events do not appear in the parent's
file.

New fields and stages may be added within `schema_version: 1`. The meaning of existing fields does not change.
Parsers should ignore what they do not know.

## Human-readable output

Without `--json`:

- `doctor`, `backends`, `plan`, `plugins`, `artifacts validate`, `cache` and the result of every workflow command
  are shown as tables.
- Status is shown as a symbol plus a word, with colour on the symbol only. See
  [status symbols](../concepts/runs-and-evidence.md#status-symbols).
- Errors are printed **once**, on stderr, in a panel: what failed, the details, a hint if there is one, and the
  exit code.
- The language is `OPENTERNARY_LANG=en|ja`, or the OS language. It falls back to English when the terminal's
  encoding cannot show Japanese. Help texts, panel titles, table headings and error titles are translated.
  `--json` content never is.
- `NO_COLOR` turns colour off.
