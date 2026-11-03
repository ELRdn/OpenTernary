---
description: "Task-oriented guides for every OpenTernary workflow."
---

# Guides

Each guide covers one task from start to finish, with real models in mind. If you have not used OpenTernary
before, do [Your first run](../getting-started/first-run.md) first: it shows every step on a tiny model.

## The main workflow

| Step | Guide | Commands |
|---|---|---|
| 1 | [Inspect and plan](inspect-and-plan.md): get a model, see what can be quantized | `inspect`, `plan` |
| 2 | [Quantize](quantize.md): make a ternary snapshot | `quantize` |
| 3 | [Calibrate](calibrate.md): try to recover quality | `calibrate` |
| 4 | [Evaluate and compare](evaluate-and-compare.md): measure and judge | `benchmark`, `quality`, `compare` |
| 5 | [Export](export.md): pack, restore, GGUF | `export` |

## Beyond the basics

| Guide | For |
|---|---|
| [Diffusion models](diffusion.md) | quantizing a component of a Diffusers pipeline and evaluating images |
| [Constrained search](optimize.md) | trying several configurations within time, VRAM and disk budgets |
| [Manage artifacts](artifacts.md) | validating artifacts, listing and removing them safely |
| [Automation and scripting](automation.md) | `--json`, progress events, exit codes, CI |
| [Troubleshooting](troubleshooting.md) | common errors and what they mean |
