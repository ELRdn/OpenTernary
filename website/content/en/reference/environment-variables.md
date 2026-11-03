---
description: "Every environment variable OpenTernary reads, and every file it writes outside your project."
---

# Environment variables

## OpenTernary

| Variable | Values | Effect |
|---|---|---|
| `OPENTERNARY_LANG` | `en`, `ja` | Display language for help, tables and errors. Default: the OS locale. Falls back to `en` when the terminal cannot encode Japanese. Never affects `--json`. |
| `OPENTERNARY_NO_TUI` | any non-empty value | `openternary` with no arguments prints help instead of opening the TUI. `openternary ui` refuses. |
| `OPENTERNARY_NO_UPDATE_CHECK` | any non-empty value | Never check PyPI for a newer version automatically (in `doctor` and the TUI). `update --check` still works. |
| `OPENTERNARY_STATE_DIR` | a folder | Where the TUI keeps job records. Default: see [below](#files-openternary-writes-outside-your-project). |
| `OPENTERNARY_GPU_ARCH` | e.g. `gfx1201` | Tell `setup` the gfx target of an AMD GPU it cannot identify. |
| `OPENTERNARY_INDEX_PYTORCH` | URL | Replace the PyTorch wheel index that `setup` reads (default `https://download.pytorch.org/whl`). |
| `OPENTERNARY_INDEX_AMD` | URL | Replace AMD's ROCm wheel index (default `https://stable.repo.amd.com/rocm/whl-next`). |
| `OPENTERNARY_INDEX_PYPI` | URL | Replace the PyPI index used by `setup` (default `https://pypi.org/simple`). |
| `OPENTERNARY_REDUCED_MOTION` | any non-empty value | Stop the spinning symbol in the TUI and the splash screen. |
| `OPENTERNARY_SPIN` | `cw`, `ccw` | Direction of the spinning symbol (default clockwise). |
| `OPENTERNARY_BREAK_IS_INTERRUPT` | `1` | Internal: set by the TUI for its child processes on Windows, so that Ctrl+Break is handled as an interrupt (exit code 130). |
| `OPENTERNARY_VALIDATION_ROOT` | a folder | Where `scripts/validate_cli_offline.py` writes its logs (default `.tmp`). |

## Standard variables OpenTernary respects

| Variable | Effect |
|---|---|
| `NO_COLOR` | No colour in output. |
| `NO_MOTION` | Same as `OPENTERNARY_REDUCED_MOTION`. |
| `CI` | When set, `openternary` with no arguments never opens the TUI. |
| `TERM=dumb` | Same: no TUI. |
| `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE`, … | Hugging Face's own variables decide where models are cached. OpenTernary only reads that cache. |
| `UV_CACHE_DIR` and other `UV_*` | Passed through to uv when `setup` and `update` run it. |
| `LOCALAPPDATA` (Windows), `XDG_STATE_HOME`, `XDG_CACHE_HOME` | Base folders for the files below. |

## Research and debugging switches

These exist for the calibration research and its gate tests. They change behaviour in ways that make results
unsuitable as evidence, so do not set them for normal use.

| Variable | Effect |
|---|---|
| `OT_LIMIT_MODULES` | Calibrate only the first N target layers. |
| `OT_SKIP_MATERIALIZE` | Skip writing the calibrated snapshot. |
| `OT_FORCE_ROCM`, `OT_FORCE_GPU` | Simulate a GPU backend for testing device code paths. |
| `OT_FORCE_LOW_BUDGET` | Force a tiny VRAM budget to exercise microbatch shrinking. |

## Files OpenTernary writes outside your project

| What | Windows | Linux / macOS |
|---|---|---|
| Update-check cache (1 day) | `%LOCALAPPDATA%\openternary\update-check.json` | `$XDG_CACHE_HOME/openternary/update-check.json` (default `~/.cache/…`) |
| TUI job records (newest 200 kept) | `%LOCALAPPDATA%\openternary\jobs\` | `$XDG_STATE_HOME/openternary/jobs/` (default `~/.local/state/…`) |
| PyTorch setup record | `openternary-torch.json` in the environment that runs `openternary` | same |

Everything else (runs, artifacts, events files) goes where you tell it to, by default under `runs/` in the current
folder.
