---
description: "Quantize one component of a Diffusers pipeline and evaluate image generation with a fixed profile."
---

# Diffusion models

OpenTernary can also quantize a component of a [Diffusers](https://huggingface.co/docs/diffusers) pipeline, such
as the transformer or UNet, and generate images under a fixed protocol for comparison. Install the extra first:

```bash
uv tool install "openternary[diffusion]" --python 3.12 --reinstall   # diffusers==0.40.0
openternary setup                                                    # the reinstall removed the GPU build
```

## Quantize a component

A pipeline folder contains `model_index.json` and one subfolder per component. Choose the component explicitly:

```bash
openternary plan ./my-pipeline --component transformer
openternary quantize ./my-pipeline --component transformer --device cpu --output runs/pipe-transformer
```

Only recognizable **two-dimensional linear weights** are converted. Convolution weights and the VAE are left
alone. This is not low-bit support for whole pipelines. Output shards use Diffusers' standard names, so Diffusers
can load the component back.

## Evaluate with a fixed profile

Image quality is judged by people, so `evaluate` makes generation reproducible and records evidence, but never
passes an image as "good". Describe the protocol in a profile:

```json title="diffusion-profile.json"
{
  "schema_version": 1,
  "family": "diffusion",
  "name": "fixed-prompt-v1",
  "prompts": ["A red cube on a white table"],
  "seeds": [42],
  "steps": 20,
  "width": 512,
  "height": 512,
  "guidance_scale": 3.5,
  "warmup": true
}
```

```bash
openternary evaluate ./my-pipeline --profile diffusion-profile.json --dry-run      # check the profile
openternary evaluate ./my-pipeline --profile diffusion-profile.json --output runs/pipe-original
```

To evaluate the pipeline with your quantized component swapped in, add `"component": "transformer"` to the
profile and pass the artifact:

```bash
openternary evaluate ./my-pipeline --profile diffusion-profile.json \
    --component-artifact runs/pipe-transformer/artifacts/snapshot --output runs/pipe-ternary
```

The replacement must currently be safetensors.

## What is recorded

`evaluation.json` stores the prompts, seeds, steps, resolution and the pipeline's scheduler configuration, and,
for each image, its hash, the generation time and pixel statistics. Comparisons check that these settings match.

Every result carries `human_review_required`: pixel statistics and hashes can show *that* images changed, but not
whether they got worse. OpenTernary therefore never marks a diffusion result as quality-accepted.

## What has been verified

A synthetic Stable Diffusion pipeline at 32×32 pixels and 2 steps has been converted with its UNet swapped in, and
has generated finite images, on Windows and WSL CPUs and on a Radeon RX 9070 XT in FP32 and BF16. No real
pretrained pipeline and no image quality have been evaluated. See [Status](../project/status.md).
