---
description: "Common OpenTernary errors and what to do about them, grouped by installation, models, running, evaluation and Windows."
---

# Troubleshooting

Errors are printed **once**, on stderr, as a panel that says what failed, the details, a hint when there is one,
and the exit code:

```text
╭─ ✖ Unsupported capability ───────────────────────────────────────────────────╮
│ Quantize error                                                               │
│ ternary snapshot conversion runs on CPU; choose --device cpu or auto         │
╰───────────────────────────────────────────────────────────────────── exit 3 ─╯
```

The exit code tells you the category (see [exit codes](automation.md#exit-codes)). With `--json`, the same message
is in the `error` field. Look for your message below.

## Installation and environment

??? question "`openternary: command not found` after `uv tool install`"

    uv's tool folder is not on your `PATH` yet. Run `uv tool update-shell` once and open a new terminal.
    `uv tool dir --bin` shows where the command was installed.

??? question "`torch is required …` (exit 4)"

    PyTorch is not installed in the environment that runs `openternary`. Run `openternary setup`. The hint in the
    message mentions `uv sync --extra ml`, which applies to a source checkout. For a `uv tool` install,
    `openternary setup` is the right command.

??? question "`doctor` says PyTorch `differs from the build openternary setup installed`"

    Something reinstalled `torch`, typically `uv sync`, a `pip install` of another package, or reinstalling the
    tool. Run `openternary setup --restore` to put the recorded GPU build back.

??? question "`setup` does not recognise my AMD GPU"

    Give it the gfx target: `OPENTERNARY_GPU_ARCH=gfx1201 openternary setup` (PowerShell:
    `$env:OPENTERNARY_GPU_ARCH="gfx1201"; openternary setup`). `rocminfo` on Linux, or AMD's documentation,
    gives the target for your card.

??? question "`setup` offers no build for my Python"

    Some builds exist only for certain Python versions; Windows ROCm, for example, only for 3.12 and 3.13.
    `setup` says which Python versions have a build. Reinstall with one of them:
    `uv tool install openternary --python 3.12 --reinstall`.

??? question "`setup` exits with code 5"

    The build installed, but the test on the GPU failed: FP32/BF16 matrix multiplication on the real device. Check
    your GPU driver, try another build (`openternary setup --list`, then `--torch` / `--variant`), or use
    `--backend cpu`.

??? question "`setup` / `update` exit with code 2 in a script"

    Outside an interactive terminal, and with `--json`, they need `--yes` to install anything.

??? question "`uv` errors about its version in a source checkout"

    The repository pins uv with `required-version` in `pyproject.toml`. Install that version of uv
    (`astral-sh/setup-uv` in CI).

## Models and files

??? question "`Local snapshot not found for '<model id>'`"

    OpenTernary never downloads models. Download the model first. The message contains the exact command, for
    example `hf download google/gemma-4-E2B-it-qat-q4_0-unquantized --revision 6bef…`. Run it as
    `uvx --from huggingface_hub hf download …`. For gated models, accept the licence on Hugging Face and run
    `hf auth login` first.

??? question "`Snapshot missing config.json` / `missing model.safetensors`"

    The folder is not a model snapshot. Point at the folder that directly contains `config.json` and the
    `*.safetensors` files. For OpenTernary runs, that is `runs/<run>/artifacts/snapshot`, or the run folder itself
    for `export`.

??? question "`mixed_precision key is not a supported target: …` (exit 2)"

    The tensor name does not exist or is not quantizable. Copy exact names from `inspection.json`
    ([Inspect and plan](inspect-and-plan.md#inspect-what-is-in-the-model)).

??? question "Configuration error on a key that looks right"

    Config files are strict: unknown keys are errors, never ignored. Check the spelling and the section against
    the [configuration reference](../reference/configuration.md). `benchmark.suites` (plural) is deprecated and
    must agree with `benchmark.suite`.

## Running commands

??? question "`quantize preserves source dtype; use --weight-dtype …` (exit 2)"

    `quantize` does not take `--dtype`: the snapshot keeps the source dtype. Choose the dtype for running the model
    when you `benchmark` or evaluate.

??? question "`ternary snapshot conversion runs on CPU` (exit 3)"

    Use `--device cpu` or leave it at `auto` for `quantize`. GPUs are used for evaluation and calibration.

??? question "`run is already in use`"

    Another process holds the run's `.execution.lock`: a calibration still running, or a resume started twice.
    Locks are released automatically when the process exits, including after a crash.

??? question "`events file already exists; pass a new path to --events-jsonl`"

    Event files are never appended to. Use a new file name per run.

??? question "`--events-jsonl cannot be used with --dry-run`"

    A dry run writes nothing, including events. Drop one of the two.

??? question "`--preflight` fails with `revision_pinned: False`, `canonical_target_count: False`, …"

    Preflight checks the canonical research target (Gemma 4 E2B at its pinned revision). For other models these
    checks fail by design. Use `calibrate --dry-run` instead.

??? question "Calibration runs out of GPU memory"

    Keep `runtime.low_vram: true` and `runtime.microbatch.auto: true` (the defaults), so OpenTernary retries with
    smaller microbatches. Lower `runtime.vram.max_fraction` if other programs need the GPU. Reduce
    `calibration.seq_len` or `num_samples`. Resume with `--resume` afterwards: nothing finished is lost.

??? question "The output appears twice with `--json`"

    In a terminal you see stdout (the JSON line) and stderr (the human view) together. Redirect stdout
    (`> result.json`) and only the JSON is captured.

## Evaluation and comparison

??? question "`quality requires --data with a frozen dataset JSON`"

    `quality` needs a dataset file. See [quality datasets](evaluate-and-compare.md#quality-datasets).

??? question "`split overlap` / `split near-duplicate` / `duplicate texts`"

    The dataset has the same or nearly the same text in two splits, or twice in one. That would contaminate the
    evaluation, so it is refused. Remove or replace the offending records.

??? question "`quality … mismatch` from `compare`"

    The two runs were not measured the same way (dataset, protocol, dtype, device, source model or environment),
    so comparing them would be meaningless. Re-run one of them with matching settings.

??? question "`compare` says `insufficient_evidence`"

    The gate was not applied, because the evidence is incomplete: an older report schema, a source model that was
    not identified, a different environment, or a synthetic model. The `reason` in `acceptance.json` says which.

??? question "My quantized model's outputs are gibberish"

    That is the expected result of plain ternary conversion of a model trained in 16 bits. See
    [Status](../project/status.md). Calibration is the research direction for recovering quality.

## Windows

??? question "Japanese text shows as `?` or the output is in English"

    If the console's encoding cannot show Japanese (cp1252, for example), OpenTernary switches to English instead
    of printing broken text. Use Windows Terminal, or set the console to UTF-8, or choose a language explicitly
    with `OPENTERNARY_LANG=en|ja`.

??? question "`update` prints commands instead of upgrading"

    A running `openternary.exe` cannot replace itself. Close OpenTernary and run the printed commands in a new
    terminal: `uv tool upgrade openternary`, then `openternary setup --restore`.

??? question "Closing the terminal stopped my TUI jobs"

    On Windows, jobs run in the terminal's console and stop when that window closes. Keep the window open, or
    minimise it, until they finish.

??? question "uv: access denied on its cache (`.lock`, `sdists-v9\.git`)"

    ```powershell
    uv cache clean
    # if that fails:
    Remove-Item -Force $env:LOCALAPPDATA\uv\cache\.lock
    Remove-Item -Recurse -Force $env:LOCALAPPDATA\uv\cache\sdists-v9\.git
    ```

## Still stuck?

Open an issue on [GitHub](https://github.com/ELRdn/OpenTernary/issues) with the command you ran, the full error
panel, the output of `openternary doctor --json`, and, for a run, its `config.yaml` and `metrics.json`.
