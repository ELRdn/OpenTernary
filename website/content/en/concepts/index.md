---
description: "The ideas behind OpenTernary: ternary weights, run directories and evidence, calibration, and the architecture."
---

# Concepts

Read these to understand *why* OpenTernary behaves as it does. The guides tell you *how* to do things.

<div class="grid cards" markdown>

-   **[Ternary quantization](ternary-quantization.md)**

    ---

    What restricting weights to −1, 0 and +1 means, how scales work, per-tensor vs per-group, fake-quant vs
    packed, and why "1.58 bits" usually means 2.

-   **[Runs and evidence](runs-and-evidence.md)**

    ---

    Every result lives in a run directory. Why "completed" never means "good", how the quality gate works, and
    what the status symbols mean.

-   **[Calibration](calibration.md)**

    ---

    How layer-by-layer reconstruction tries to win back quality: learned scales, learned thresholds and
    soft-to-hard assignments.

-   **[Architecture](architecture.md)**

    ---

    How the CLI, services, adapters, backends, passes, exporters and evaluators fit together, and where plugins
    plug in.

</div>
