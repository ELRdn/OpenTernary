---
description: "Exactly what OpenTernary has and has not verified: the CLI and its environments, the research results, and the release state."
---

# Status

Last updated for version **0.1.0a2** (November 2026). This page separates what has been **verified**, and how, from
what is only **implemented** and what is **open**. When in doubt, the repository's evidence records are the source:
[CLI implementation status](https://github.com/ELRdn/OpenTernary/blob/main/docs/CLI_IMPLEMENTATION_STATUS.md),
[validation evidence](https://github.com/ELRdn/OpenTernary/blob/main/docs/CLI_VALIDATION_EVIDENCE.md),
[research roadmap](https://github.com/ELRdn/OpenTernary/blob/main/ROADMAP.md) and
[experiment log](https://github.com/ELRdn/OpenTernary/blob/main/EXPERIMENT_LOG.md).

## In one paragraph

The **tooling works**: inspecting, quantizing, calibrating, packing, exporting, evaluating, comparing and searching
run end to end, have been tested on synthetic models with real libraries on Windows, WSL and an AMD GPU, and are
published on PyPI with a TUI. The **research goal is not met yet**: no ternary version of the target model,
Gemma 4 E2B, has passed the quality gate, and the latest experiments point to a structural limit in the current
calibration method.

## The CLI and its environments

Verified with synthetic models (seeded, randomly initialised Llama and Stable Diffusion pipelines), real libraries
and installed wheels:

| | Windows CPU | WSL Ubuntu 24.04 CPU | Windows, Radeon RX 9070 XT |
|---|---|---|---|
| Synthetic Llama: ternary conversion, passes, per-layer preserve, packed round trip | ✔ verified | ✔ verified | ✔ FP32/BF16 forward, reload in a separate process |
| TorchAO 0.18.0 INT8: convert, save, reload | ✔ verified | ✔ verified | ✔ FP32/BF16 forward, reload in a separate process |
| Diffusers 0.40.0: UNet swap, 32×32, 2 steps | ✔ verified | ✔ verified | ✔ FP32/BF16 generation, finite, saved |
| Quality report schema 3, search workers, resume | ✔ synthetic fixtures | ✔ synthetic fixtures | computation and resource measurement only; search quality checked on CPU |
| Pinned GGUF converter | ✔ isolated converter and reader, synthetic Llama FP32 | ✔ CPU runtime generates up to 8 tokens | ○ GPU execution not verified |
| Wheel install, real entry-point plugins | ✔ verified | ✔ verified | ✔ wheel in a copied ROCm environment |
| Pretrained models / image quality | ○ not verified | ○ not verified | ○ not verified |
| Native packed ternary kernel | – not provided | – not provided | – not provided |

Final validation run (offline, Windows CPU): **326 passed, 2 skipped, 0 failed**. The 2 skipped tests need ROCm and
passed separately in the ROCm environment. Ruff, formatting and mypy passed on every file. Measured GPU memory peaked
at about 84.6 MiB. Synthetic results are not extended to Gemma, Mistral, Qwen or other real models.

Since then, each change has gone through GitHub CI on Linux and Windows, including the TUI tests, a real Ctrl+Break
cancel test on Windows, and the job manager tests.

## PyTorch setup on real hardware

`openternary setup` on the owner's PC (Windows 11, Radeon RX 9070 XT, gfx1201) detected the GPU, installed AMD's
`torch 2.14.0+rocm10.1.0` with only the gfx1201 kernels, and passed the FP32/BF16 device test. That configuration
was also confirmed working from the PyPI release (`uv tool install openternary`). NVIDIA and Apple paths are
implemented against the official indexes but have not been run on real hardware by the project.

## Research

The research target is **Gemma 4 E2B** (`google/gemma-4-E2B-it-qat-q4_0-unquantized` at revision `6befbaca…`), with
205 canonical linear layers quantized per group (G128) and evaluated in BF16.

| Phase | Result |
|---|---|
| 0 Foundation | ✔ closed |
| 1 Inspection and BF16 smoke baseline | ✔ closed: fingerprints stable across two real CPU runs |
| 2 Ternary core and packing accounting | ✔ closed: 19 gates, real-tensor round trip bit-exact |
| 3 Whole-model ternarization | ✔ closed: per-tensor and per-group G128 snapshots of the whole model (12 shards, 10.2 GB) load and run the smoke suite; outputs differ from the baseline, as expected |
| 4.1 Calibration: learned scales | ✔ closed: lowers training and held-out reconstruction loss on fixtures; pipeline works on the real model |
| 4.2 Learned thresholds | implemented; AMD GPU runtime gate passed; changes very few codes (0.009% vs scale-only) |
| 4.3 Soft-to-hard | three schedules run on all 205 layers; after hardening all collapsed (instruction score 0, 8/8 collapses); **no schedule passed** |
| 5+ Japanese track, special components, packed export, scaling | not started as research phases |

**No quantized model has passed the quality gate.** The analysis of the soft-to-hard results finds that the
optimizer owns only the scales, while code assignments and the hardening boundary stay frozen. Attention `q_proj`
carries 86% of the held-out error, and layer 0's alone about 47%. A redesign decision is pending; see the
[decision record](https://github.com/ELRdn/OpenTernary/blob/main/docs/plans/p4-soft-to-hard-redesign-decision.md).
Until then, expect any OpenTernary ternary model to be far worse than the original.

## Releases

| Version | Date | Notes |
|---|---|---|
| 0.1.0a2 | 2026-11-03 | Full-screen TUI with background jobs; selection-style command prompt; see the [changelog](changelog.md) |
| 0.1.0a1 | 2026-11-03 | First PyPI release |

Both are pre-releases. There is no stable release yet.

## Not yet verified

- quality of any ternary model on real weights (the research question);
- native low-bit execution (no kernel is provided);
- GGUF on GPUs and for architectures other than Llama;
- `setup` on real NVIDIA and Apple hardware;
- long calibration runs monitored from the TUI, and TUI cancellation on real Windows hardware (covered by CI only).
