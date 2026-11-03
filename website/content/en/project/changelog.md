---
description: "What changed in each OpenTernary release."
---

# Changelog

Versions follow [PEP 440](https://peps.python.org/pep-0440/). Pre-releases (`aN`) may change behaviour between
versions. The `--json` envelope, exit codes and progress-event schema are versioned contracts, and changes to them
are listed here.

## 0.1.0a2 — 2026-11-03

**The full-screen interface.**

- **New: `openternary` with no arguments in a terminal opens a TUI**, also available as `openternary ui`.
  Tabs for Home, Runs, Jobs, Compare, Doctor and Update.
- **Selection-style control**: a `❯` prompt at the bottom, where `/` lists commands with descriptions. Type to
  filter, use ↑↓ to choose, Enter to run, Tab to complete. Confirmations are `❯` menus too. There are no buttons.
- **Background jobs**: every job is the ordinary CLI in a child process, with live per-stage progress, rate and
  ETA, and the end of its log. Cancelling interrupts cleanly (exit 130), and force-stopping kills the process tree.
  Jobs keep running when the TUI is closed (Linux/macOS) and are picked up again when it reopens.
- **Forms generated from the CLI definition**, and a **plan screen** that shows the command's own `--dry-run`
  output. *Start* is unavailable until the dry run succeeds.
- **Safe deletion**: deleting a run or running `cache remove --execute` requires typing the target's name. Runs in
  use are refused.
- Entry point is now `openternary.cli.main:run`. `openternary.cli.main:app` still works.
- Without a terminal (pipes, `CI`, `TERM=dumb`, `OPENTERNARY_NO_TUI=1`), `openternary` with no arguments still
  prints help and exits with code 2. `--json` code paths never import the TUI library.
- Fix: editable installs report the version of the source tree, not stale package metadata.
- New dependency: `textual>=8.2`.

## 0.1.0a1 — 2026-11-03

**The first PyPI release.**

- **`openternary setup`**: detects NVIDIA, AMD and Apple GPUs, lists PyTorch builds from the live PyTorch, AMD and
  PyPI indexes, recommends one, installs it with the ML dependencies into the running environment, and tests it on
  the device. `--restore` reinstalls the recorded build. A build already verified on the PC is preferred. AMD
  downloads can be limited to one GPU's kernels.
- **`openternary update`**: checks PyPI (cached for a day), upgrades according to the install method, and restores
  the PyTorch build afterwards. On Windows uv-tool installs it prints the commands to run.
- **Readable output**: tables instead of raw JSON for human output, status symbols, one error panel per failure,
  and help in English and Japanese (`OPENTERNARY_LANG`).
- **Progress events** (`--events-jsonl`): `unit`, `elapsed_s` and `detail` fields, with calibration, capture,
  materialization, benchmark, quality, export, validation and search stages. Still `schema_version: 1`.
- **Licence**: Apache-2.0, with a NOTICE file and a trademark policy for the name and logo.
- **Brand**: the official tomoe symbol and the "Yoru" palette.

### Before 0.1.0a1

Development before the first release (August–September 2026) built the research pipeline and the CLI product:

- Phases 0–3: inspection, BF16 smoke baseline, AbsMean ternary core with 2-bit packing accounting, and bounded
  sharded whole-model fake quantization.
- Phase 4.1/4.2: layer-local calibration with learned scales and thresholds, checkpoints, resume, low-VRAM runtime
  and finalization recovery.
- CLI-0 to CLI-8: backends (ternary, TorchAO), adapters (Gemma 4, Transformers, Diffusers), passes, atomic exports
  with manifests, packed format, GGUF bridge, quality report schema 3 and the quality gate, constrained search,
  plugins, the versioned JSON contract and exit codes.

Migration notes from that period are in each guide's "Migrating from older versions" section and in
[docs/CLI.md](https://github.com/ELRdn/OpenTernary/blob/main/docs/CLI.md).
