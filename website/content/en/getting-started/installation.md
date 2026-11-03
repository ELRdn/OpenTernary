---
description: "Install the openternary command, set up PyTorch for your GPU with openternary setup, and keep it up to date."
---

# Installation

There are three ways to install OpenTernary. Most people want the first one.

| Install | For | Command |
|---|---|---|
| **uv tool** (recommended) | Using OpenTernary from any folder | `uv tool install openternary --python 3.12` |
| **Project venv** | Using it from Python code or a notebook | `uv pip install openternary` |
| **From source** | Changing OpenTernary itself | `git clone` + `uv sync` |

## 1. Install uv

OpenTernary uses [uv](https://docs.astral.sh/uv/) to install itself and to install PyTorch. If you do not have it yet:

=== "Windows (PowerShell)"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "macOS / Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

Open a new terminal afterwards so that `uv` is on your `PATH`.

## 2. Install the `openternary` command

```bash
uv tool install openternary --python 3.12
```

This puts `openternary` on your `PATH` in an isolated environment of its own. If the command is not found,
run `uv tool update-shell` once and open a new terminal.

!!! info "Why `--python 3.12`?"

    Without it uv may pick the newest Python on your machine. OpenTernary is tested on 3.11 and 3.12, and some
    PyTorch builds exist only for particular versions. For example, AMD's Windows ROCm wheels exist only for
    Python 3.12 and 3.13.

The release is a pre-release (`0.1.0a2`). uv installs it because no stable version exists yet. Once a stable
version is published, use `uv tool install openternary --prerelease allow` to keep getting pre-releases.

Check that it works:

```console
$ openternary --version
openternary 0.1.0a2
```

At this point the **core** is installed. It can show help, plan quantizations, inspect model headers and
report on your environment. It does not include PyTorch: that is the next step.

## 3. Install PyTorch for your hardware: `openternary setup`

PyTorch comes in different builds for NVIDIA (CUDA), AMD (ROCm), Apple silicon (MPS) and CPU-only, and the
right one depends on your GPU, driver, OS and Python. `openternary setup` works this out for you:

```bash
openternary setup --list      # what was detected, and every compatible build
openternary setup --dry-run   # the recommended plan; installs nothing
openternary setup             # asks, installs, then tests the GPU
```

`setup` does the following:

1. **Detects your GPU.**

    | Hardware | How it is detected |
    |---|---|
    | NVIDIA | `nvidia-smi`, including the highest CUDA version your driver supports |
    | AMD on Windows | the PCI device ID of the GPU from `Win32_VideoController`, mapped to its gfx target |
    | AMD on Linux / WSL | `rocminfo`, or `/sys` if ROCm is not installed |
    | Apple silicon | MPS on macOS |

    If an AMD GPU is not recognised, tell it the target: `OPENTERNARY_GPU_ARCH=gfx1201 openternary setup`.

2. **Lists the builds that exist right now.** It reads the live indexes of the
    [PyTorch download site](https://download.pytorch.org/whl/) (CUDA, ROCm, CPU), of AMD (ROCm for Windows and
    Linux) and of PyPI. There is no built-in version table to go stale. Only wheels that install on *your*
    Python and OS are offered.

3. **Recommends one.** It prefers builds that come with a matching `torchvision` from the same source, stable
    over pre-release, and builds your driver can run. For NVIDIA that means a CUDA version at or below the
    driver's limit. A build that was already verified on this PC is preferred.

4. **Installs it into the environment that runs `openternary`**, together with the machine-learning packages
    OpenTernary needs (`transformers`, `safetensors`, `accelerate`, `datasets`, …), using
    `uv pip install --python <that environment>`.

5. **Tests it on the real device.** A new process runs FP32 and BF16 matrix multiplications on the GPU. The
    result and the exact build are recorded in `openternary-torch.json` inside that environment. A failed test
    exits with code `5`.

Useful options:

```bash
openternary setup --backend rocm            # force a backend: auto | cuda | rocm | mps | cpu
openternary setup --torch 2.13.0            # a specific version
openternary setup --variant cu130           # a specific build variant (cu130, rocm10.0.0, cpu, …)
openternary setup --amd-device gfx1201      # AMD: only the kernels for this GPU (smaller download)
openternary setup --pre                     # include pre-release builds
openternary setup --restore                 # reinstall exactly what setup recorded last time
```

For AMD, `--amd-device auto` (the default) downloads only your GPU's kernels when AMD publishes a package for
it, and all kernels otherwise.

!!! warning "`uv sync` and other installers can replace your PyTorch"

    If something reinstalls `torch` from PyPI, you silently lose the GPU build. `openternary doctor` compares the
    installed torch with what `setup` recorded and tells you. `openternary setup --restore` puts the recorded
    build back.

### Without a terminal prompt

In scripts, CI, or with `--json`, `setup` does not install anything unless you pass `--yes`. Without it, it
exits with code `2`:

```bash
openternary setup --yes --json
```

## 4. Check your environment: `openternary doctor`

```bash
openternary doctor                                # packages, backends, PyTorch build
openternary doctor --probe-runtime --device cuda  # also run a tiny FP32/BF16 test on the GPU
```

`doctor` never installs anything and never loads a model. Without `--probe-runtime` it only reads package
metadata. On ROCm, PyTorch calls the GPU `cuda` too, so use `--device cuda` for AMD GPUs as well.

Real output on a Linux machine with a CPU-only PyTorch that was not installed by `setup`:

```text
╭─ Environment ────────────────────────────────────────────────────────────────────╮
│ Python 3.12.3  ·  Linux-6.18.44-fc-v230-x86_64-with-glibc2.39                    │
│                                                                                  │
│ Dependencies                                                                     │
│ Package        Version                                                           │
│ ──────────────────────────────                                                   │
│ openternary    0.1.0a2                                                           │
│ torch          2.15.0                                                            │
│ safetensors    0.8.0                                                             │
│ transformers   5.22.1                                                            │
│ diffusers      ○ not installed                                                   │
│ torchao        ○ not installed                                                   │
│ datasets       ○ not installed                                                   │
│ Backends                                                                         │
│ Item             ternary                          torchao                        │
│ ──────────────────────────────────────────────────────────────────────────────── │
│ Installed        ● installed                      ○ not installed                │
│ Quality          ▲ experimental                   ▲ experimental                 │
│ Convert          ◐ implemented model unverified   ◐ implemented model unverified │
│ Model reload     ◐ unverified                     ◐ unverified                   │
│ Native low-bit   – unsupported                    ◐ unverified                   │
│ Formats          safetensors, ternary-packed      torchao                        │
│ Version          0.1.0a2                          requires 0.18.0                │
│ Hardware: not probed — openternary doctor --probe-runtime                        │
│ Model validation: ○ not run                                                      │
│ PyTorch: 2.15.0  (not set up by openternary setup)                               │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

The last line is the one to watch:

| `PyTorch:` line | Meaning |
|---|---|
| `2.15.0  (not set up by openternary setup)` | Some PyTorch is installed, but `setup` did not install it, so nothing records whether it works on your GPU. |
| `2.14.0+rocm10.1.0  · rocm · amd  ✔ passed` | `setup` installed this build and its GPU test passed. |
| `▲ differs from the build openternary setup installed (…)` | Something replaced the recorded build. Run `openternary setup --restore`. |

The status symbols are explained in [Runs and evidence](../concepts/runs-and-evidence.md#status-symbols).

## Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `ml` | `torch`, `safetensors`, `transformers`, `accelerate`, `pillow`, `torchvision` | quantizing, benchmarking, quality (`setup` installs these for you) |
| `calibration` | `datasets` | calibrating on `wiki-tiny` (`setup` installs it for you) |
| `torchao` | `torchao==0.18.0` | the TorchAO INT8 backend |
| `diffusion` | `diffusers==0.40.0` | Diffusers pipelines and `evaluate` |

To add an extra to the tool install, reinstall with it and then run `setup` again. The reinstall rebuilds the
environment and removes the GPU build:

```bash
uv tool install "openternary[diffusion,torchao]" --python 3.12 --reinstall
openternary setup
```

## Updating: `openternary update`

```bash
openternary update --check     # is there a newer release on PyPI?
openternary update --dry-run   # what would run
openternary update             # upgrade, then reinstall the PyTorch build setup recorded
```

- The only network request is PyPI's JSON API (`pypi.org/pypi/openternary/json`). The answer is cached for a day.
- The steps depend on how you installed: `uv tool upgrade openternary` for a uv tool,
  `uv pip install --upgrade openternary` for a venv. An editable source checkout is never changed: you are
  told to `git pull`.
- Upgrading a uv tool rebuilds its environment, which would drop your GPU build. `update` saves the `setup`
  record first and runs `openternary setup --restore --profile <saved copy> --yes` afterwards.
- **On Windows** a running program cannot replace itself, so for a uv tool install `update` prints the commands
  for you to run in a new terminal instead of running them:

    ```powershell
    uv tool upgrade openternary
    openternary setup --restore
    ```

- `openternary doctor` checks PyPI at most once a day, and only in an interactive terminal. It tells you when a
  newer version exists. Set `OPENTERNARY_NO_UPDATE_CHECK=1` to turn this off.

## Installing into a project environment

If you want to `import openternary` from your own code or a notebook:

```bash
uv venv --python 3.12
uv pip install openternary
uv run openternary setup
```

`setup` installs PyTorch into whatever environment runs it, here the project's `.venv`. With plain `pip`,
`pip install openternary` works the same way, but `setup` and `update` still need uv on your `PATH`.

## Installing from source

For working on OpenTernary itself. The repository pins its uv version (`required-version` in `pyproject.toml`),
so use that version of uv.

```bash
git clone https://github.com/ELRdn/OpenTernary.git
cd OpenTernary
uv sync                                  # core + dev tools, exactly as locked in uv.lock
uv run openternary --help
uv sync --extra ml --extra calibration   # add the ML stack (CPU/CUDA torch from PyPI)
uv run openternary setup                 # or replace torch with the build for your GPU
```

To get an `openternary` command that follows your checkout from any folder:

```bash
uv tool install --editable . --python 3.12
uv tool update-shell     # first time only; then open a new terminal
```

With `--editable`, changes from `git pull` apply immediately. After an update that adds dependencies, run
`uv tool install --editable . --reinstall`. Then run `openternary setup --restore`, because reinstalling
rebuilds the environment.

Development checks are described in [Contributing](../project/contributing.md).

## Uninstalling

```bash
uv tool uninstall openternary
```

This removes the command and its environment, including the PyTorch build that `setup` installed. Your runs and
models are not touched. Two other folders may be worth deleting: the TUI's job records (see
[Environment variables](../reference/environment-variables.md#files-openternary-writes-outside-your-project))
and the Hugging Face cache, if you downloaded models.
