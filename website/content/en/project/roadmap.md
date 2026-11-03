---
description: "Where OpenTernary is going: the CLI product roadmap (CLI-0 to CLI-8), the research phases, and release milestones."
---

# Roadmap

OpenTernary has two roadmaps, kept separate on purpose. The **product** can be finished, reliable and well
documented while the **research** has not yet produced a good model, and the other way round.

## Product: the CLI

Tracked in [CLI_ROADMAP.md](https://github.com/ELRdn/OpenTernary/blob/main/CLI_ROADMAP.md). The initial
implementation of every stage is in place and verified on synthetic models, as described on the [Status](status.md) page.

| Stage | Goal | State |
|---|---|---|
| CLI-0 | success, failure and what actually happened always agree | ✔ implemented |
| CLI-1 | backend and external-use contracts: JSON, exit codes, config migration | ✔ implemented |
| CLI-2 | model-specific code confined to adapters | ✔ implemented |
| CLI-3 | a second, hardware-verified backend (TorchAO) | ✔ implemented |
| CLI-4 | ordered passes | ✔ implemented |
| CLI-5 | storage, packed representation and runtime bridges kept separate | ✔ implemented; no native packed kernel |
| CLI-6 | compare only measurements taken under the same conditions | ✔ implemented |
| CLI-7 | search within measured constraints and budgets | ✔ implemented; real-model usefulness open |
| CLI-8 | distribution and extension compatibility | ✔ PyPI, plugins, TUI |

Next for the product:

- run the same acceptance scripts on real models and hardware;
- verify the TUI's long-job monitoring and Windows cancellation on real machines;
- more ways to install and run it (for example a package-manager entry point), based on demand.

## Research

Tracked in [ROADMAP.md](https://github.com/ELRdn/OpenTernary/blob/main/ROADMAP.md), with budgets and gates in the
[P0–P7 research plan](https://github.com/ELRdn/OpenTernary/blob/main/docs/plans/p0-p7-research-acceptance.md).

| Phase | Topic | State |
|---|---|---|
| 0–3 | foundation, inspection, baseline, ternary core, whole-model fake quantization | ✔ closed |
| 4 | calibration and reconstruction: scales (4.1), thresholds (4.2), soft-to-hard (4.3), multi-layer windows (4.4), reproductions (4.5) | 4.1 closed; 4.2 and 4.3 run, no candidate passes; redesign pending |
| 5 | Japanese retention: held-out Japanese suites, Japanese-heavy calibration | planned |
| 6 | Gemma 4 special components: per-layer embeddings, multimodal parts | planned |
| 7 | packed ternary export as a research deliverable | planned |
| 8 | larger Gemma 4 variants | planned |
| 9 | GUI | the TUI exists; a GUI is later |
| 10 | broader model support: Qwen, Llama, other open-weight families | planned |

Research principles that will not change: gates are fixed before experiments, the test split is used once, a tiny
fixture or a smoke run is never a quality claim, and no reproduction of a paper or framework (BitNet, Bonsai,
CAT-Q, ScaleQ, …) is claimed unless it has been verified against its source.

## Release milestones

| Milestone | Meaning | State |
|---|---|---|
| v0.1 "Ternary hello world" | inspect, benchmark, quantize, compare work on the real model | ✔ reached in research; packaged as 0.1.0a1/a2 |
| v0.2 "Calibration" | calibration that improves held-out quality | open |
| v0.3 "Japanese track" | Japanese retention measured and reported | open |
