---
description: "The shortest path from nothing to a working OpenTernary install and a first ternary model."
---

# Getting started

These pages take you from nothing to a quantized, exported and evaluated model.

<div class="grid cards" markdown>

-   **1. [Installation](installation.md)**

    ---

    Install the `openternary` command, let `openternary setup` pick the right PyTorch for your GPU,
    and check everything with `openternary doctor`. Also covers updating and installing from source.

-   **2. [Your first run](first-run.md)**

    ---

    Build a tiny model on your CPU and run the whole pipeline in about a minute: inspect, quantize,
    pack, restore, validate, evaluate and compare. No download and no GPU needed.

-   **3. [The full-screen interface](tui.md)**

    ---

    Run the same commands from the TUI: choose them from a `❯` prompt, preview the plan, and watch
    jobs with live progress and ETA.

</div>

## What you need

| | Minimum | Notes |
|---|---|---|
| Python | 3.11 | 3.12 is recommended and is what the release is tested with most. Windows ROCm builds exist only for 3.12 and 3.13. |
| OS | Windows, Linux, macOS | WSL works. |
| [uv](https://docs.astral.sh/uv/) | any recent version | Needed for `openternary setup` and `update`. Use `pip` if you prefer, but `setup` still calls uv. |
| GPU | optional | NVIDIA (CUDA), AMD (ROCm) or Apple silicon (MPS). Quantizing runs on the CPU; GPUs speed up evaluation and calibration. |
| Disk | a few GB for small models | A ternary fake-quant snapshot is the size of the original model. Calibration adds activation caches and checkpoints. |

!!! tip "Already know the CLI?"

    Every command supports `--help`. The [CLI reference](../reference/cli.md) lists every flag, and
    [Automation](../guides/automation.md) covers `--json`, progress events and exit codes.
