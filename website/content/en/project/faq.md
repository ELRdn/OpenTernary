---
description: "Frequently asked questions about OpenTernary, ternary models, hardware, results and the project."
---

# FAQ

## About the project

??? question "Can I use OpenTernary to make a small, fast model I can actually use?"

    Not yet. OpenTernary can produce ternary models, but no ternary model made with it has passed its quality gate:
    plain conversion destroys most of a model's abilities, and calibration has not recovered them so far. It also has
    no fast ternary kernels, so ternary models run at the speed of the original. Today it is a tool for **research**:
    measuring what ternary conversion costs and trying ways to reduce that cost. See [Status](status.md).

??? question "How is this different from BitNet?"

    BitNet b1.58 models are *trained* with ternary weights from the start. OpenTernary studies converting models that
    were already trained in 16 bits, *after* training. That is much harder, but it could apply to any open-weight
    model without training one from scratch. OpenTernary does not claim to reproduce BitNet or any other method.

??? question "How is this different from GGUF quantization (Q4_K, IQ2, …) or other low-bit tools?"

    Those formats are mature, fast and practical, and you should use them if you want a smaller model today.
    OpenTernary targets a more extreme point (three values per weight) and puts measurement, provenance and honest
    evaluation first. It can export to GGUF only as floating point, for supported architectures.

??? question "Why Gemma 4 E2B?"

    It is small enough to iterate on with a consumer GPU, modern, and has an instruction-tuned release, which makes
    quality loss visible. The research pins one exact revision so that results stay comparable. Gemma is the first
    target, not the only one: the adapters are generic, and any Transformers model with recognizable linear layers can
    be quantized.

??? question "Why is there so much emphasis on Japanese?"

    Aggregate English scores can hide large losses in other languages, and low-bit quantization often hurts
    lower-resource languages first. Japanese quality is treated as a first-class metric, measured separately from
    English and gated separately.

## Using it

??? question "Do I need a GPU?"

    No. Inspecting, planning and quantizing run on the CPU, and so do the tutorial, benchmarks and quality evaluation
    on small models. A GPU makes evaluation and calibration of real models practical.

??? question "Does it support NVIDIA, AMD and Apple GPUs?"

    `openternary setup` installs PyTorch for NVIDIA (CUDA), AMD (ROCm, on Windows and Linux) and Apple silicon (MPS).
    The project has verified AMD on Windows with a Radeon RX 9070 XT. NVIDIA and Apple use the official PyTorch builds
    but have not been run by the project yet.

??? question "Which models can I quantize?"

    Hugging Face Transformers models in safetensors format whose linear layers the adapter recognizes: Gemma 4 has a
    dedicated adapter, and the generic adapter handles Llama-style architectures. Diffusers pipeline components can
    be quantized too. Quality has not been verified for any of them.

??? question "Why does my quantized snapshot have the same size as the original?"

    `quantize` writes a *fake-quant* snapshot: the weights are ternary values stored as ordinary floats, so that any
    framework can load and evaluate them. `export --format ternary-packed` stores them at 2 bits each. See
    [fake-quant, packed and native](../concepts/ternary-quantization.md#fake-quant-packed-and-native).

??? question "Can I run the packed model?"

    Not directly: no runtime for the packed format exists yet. Restore it to safetensors with
    `export --format safetensors` to run it.

??? question "Does OpenTernary send data anywhere?"

    No telemetry. The only automatic network request is a version check against PyPI's public JSON API, at most once
    a day in an interactive `doctor` or the TUI (`OPENTERNARY_NO_UPDATE_CHECK=1` turns it off). `setup` reads public
    package indexes when you run it. `calibrate` with `wiki-tiny` downloads WikiText. Models are never downloaded.

??? question "Why does `compare` refuse my comparison?"

    Because the two runs were not measured under the same conditions, or because the evidence is incomplete (for
    example a synthetic model). A comparison under different conditions would produce a number that means nothing.
    See [Runs and evidence](../concepts/runs-and-evidence.md#comparing-observation-vs-acceptance).

## Results

??? question "The tutorial's ternary model had lower perplexity than the original. Is that real?"

    No. Both models are random, so their perplexities are noise, and a small difference either way means nothing.
    That is exactly why the quality gate refuses synthetic models. See
    [the first run, step 6](../getting-started/first-run.md#6-measure-quality-and-compare).

??? question "What does \"1.58 bits\" mean, and why does OpenTernary report 2?"

    \(\log_2 3 \approx 1.585\) is the information in one ternary digit. Stored in a fixed-width format, a ternary
    weight takes 2 bits, plus scales and the tensors that are not quantized. OpenTernary reports each of these
    separately. See [counting bits honestly](../concepts/ternary-quantization.md#counting-bits-honestly).
