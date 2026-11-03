---
description: "Use OpenTernary from scripts and CI: the --json result envelope, stdout/stderr separation, progress events, exit codes and non-interactive behaviour."
---

# Automation and scripting

Every OpenTernary command is designed to be driven by scripts, CI and other programs, including the TUI, which is
itself a client of these contracts. The full specification is in the [output reference](../reference/output.md).
This guide shows how to use it.

## `--json`: one result on stdout

```bash
openternary quantize tiny-llama --device cpu --output runs/ev --json > result.json
```

With `--json`, **stdout contains exactly one line**: a versioned JSON document. Everything for humans (progress,
tables, logs, errors) goes to **stderr**. In a terminal you see both, so the output can look doubled, but a
redirect captures only the JSON:

```json
{"schema_version": 1, "command": "quantize", "exit_code": 0,
 "execution": "completed", "artifact_validation": "file_integrity", "quality_acceptance": "not_run",
 "result": {"run_dir": "…/runs/ev", "run_id": "e67150d8", "artifact": "…/runs/ev/artifacts/snapshot", …},
 "error": null}
```

- `exit_code` repeats the process exit code.
- `execution`, `artifact_validation` and `quality_acceptance` are the [separate verdicts](../concepts/runs-and-evidence.md#four-questions-four-answers).
- `result` is command-specific.
- `error` is `null` on success, otherwise the error message.

The JSON content never depends on the display language (`OPENTERNARY_LANG`).

```bash
# shell: read fields with jq
openternary plan ./my-model --json | jq -r '.result.targets | map(select(.quantizable)) | length'
```

```powershell
# PowerShell
$r = openternary doctor --json | ConvertFrom-Json
$r.result.dependencies
```

`--quiet` suppresses progress output but still prints the result and any error.

## Exit codes

| Code | Meaning | Typical cause |
|---|---|---|
| `0` | success: the operation or plan finished | |
| `1` | unexpected runtime or I/O failure | a bug, a full disk |
| `2` | invalid configuration, argument or input | unknown config key, missing file, mismatched runs in `compare` |
| `3` | unsupported capability | `quantize --device cuda`, a plugin with the wrong API version |
| `4` | a required dependency is missing | quantizing without PyTorch |
| `5` | required hardware or device unavailable | `setup`'s GPU test failed |
| `6` | explicit acceptance criteria not met / no feasible candidate | `compare --require-acceptance`, `optimize` |
| `7` | search budget exhausted | `optimize` |
| `130` | interrupted | ++ctrl+c++, TUI cancel |

Exit code `6` still means the command **ran**: `execution` is `completed`, and the verdict was negative.

## Progress events: `--events-jsonl`

```bash
openternary quantize tiny-llama --device cpu --output runs/ev --json --events-jsonl events.jsonl
```

Each line of the file is one event:

```json
{"schema_version": 1, "sequence": 1, "command": "quantize", "run_id": null, "stage": "started", "completed": null, "total": null, "unit": null, "elapsed_s": 0.0, "detail": {}}
{"schema_version": 1, "sequence": 2, "command": "quantize", "run_id": "e67150d8", "stage": "run_created", "completed": null, "total": null, "unit": null, "elapsed_s": 0.004, "detail": {}}
{"schema_version": 1, "sequence": 3, "command": "quantize", "run_id": "e67150d8", "stage": "convert.tensors", "completed": 0, "total": 21, "unit": "tensor", "elapsed_s": 1.776, "detail": {"tensor": "lm_head.weight"}}
…
{"schema_version": 1, "sequence": 8, "command": "quantize", "run_id": "e67150d8", "stage": "completed", "completed": null, "total": null, "unit": null, "elapsed_s": 1.842, "detail": {}}
```

- The file must **not exist yet**, so that two runs cannot write into one file by accident. Its parent folder is
  created if needed.
- `--events-jsonl` cannot be combined with `--dry-run`, because a dry run writes nothing.
- `sequence` increases by one per event. Unknown totals and units are `null`.
- Fine-grained stages are throttled to at most 5 lines per second per stage, but the first line and the line with
  `completed == total` are always written.

Every stage, with its unit and detail fields, is listed in the
[progress events reference](../reference/output.md#progress-events-jsonl). To follow a long run from another
terminal:

```bash
tail -f events.jsonl | jq -c '{stage, completed, total, detail}'
```

## Non-interactive behaviour

OpenTernary never waits for input when nobody is there to answer:

- **Installs need `--yes`.** `setup` and `update` refuse to install anything without `--yes` when stdin is not
  a terminal or `--json` is used (exit code `2`).
- **No TUI in pipes.** With no arguments, `openternary` opens the TUI only when stdin and stdout are both
  terminals and `CI` is not set. Otherwise it prints help and exits with `2`. `OPENTERNARY_NO_TUI=1` forces this
  behaviour everywhere.
- **No network surprises.** The only automatic network request is the daily PyPI version check in an interactive
  `doctor` or the TUI. Set `OPENTERNARY_NO_UPDATE_CHECK=1` to turn it off. Models are never downloaded.

## Example: GitHub Actions

Model-free checks run on any runner without a GPU, PyTorch or model download:

```yaml title=".github/workflows/openternary.yml"
jobs:
  plan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v9
      - run: uv tool install openternary --python 3.12
      - run: openternary doctor --json > doctor.json
      - run: openternary calibrate --config configs/my-calibration.yaml --dry-run --json
```

A validation job on a self-hosted GPU runner can then run the real pipeline and gate on quality:

```bash
set -e
openternary quantize "$MODEL" --config quant.yaml --output runs/candidate --json > /dev/null
openternary quality runs/candidate/artifacts/snapshot --data quality.json --device cuda --output runs/candidate-q --json > /dev/null
openternary compare runs/baseline-q runs/candidate-q --require-acceptance --output runs/gate --json > gate.json
# exit code 6 here fails the job when the candidate does not pass
```

## From Python

The same services are available without a subprocess. See the [Python API](../reference/python-api.md):

```python
from openternary.api import load_config, build_plan

config = load_config(cli_overrides={"model.id": "./my-model"})
plan = build_plan(config)
print(sum(t["quantizable"] for t in plan["targets"]), "tensors to quantize")
```
