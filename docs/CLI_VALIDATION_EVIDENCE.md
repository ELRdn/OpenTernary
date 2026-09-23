# CLI real-library validation evidence

Date: 2026-09-24. The synthetic evidence below was captured before the first push; pretrained evidence was added from the later RX 9070 XT run. Hosted results are tracked separately in [GitHub Actions](https://github.com/ELRdn/OpenTernary/actions/workflows/ci.yml).

## Environment and scope

- Windows CPU: Python 3.12.9, Torch 2.13.0, Transformers 5.15.1.
- WSL Ubuntu 24.04.3: Python 3.12.3, Torch 2.13.0+cpu, Transformers 5.15.1.
- Windows ROCm clone: Python 3.12.0, Torch 2.13.0+rocm10.0.0, Transformers 5.17.0, RX 9070 XT. Embedded HIP reports 7.15.26333.
- TorchAO 0.18.0 / Diffusers 0.40.0 / safetensors 0.8.0 / SentencePiece 0.2.2.
- Converter environment: Torch 2.11.0+cpu, Transformers 4.57.6, numpy 2.2.6, protobuf 4.25.9, as required by the pinned converter. Kept separate from the 2.13 CLI environments.
- llama.cpp commit `e6ab7c1a41054a888ada952eab4c886444c2f5ad`; archive SHA256 `033c29d5fda5a76af9fd0fc0ade185e5bcb2d8935f8165a30d96c1ed1355b663`. Every archived source file was checked. Runtime target at this revision is `llama-completion`.
- Artificial Llama: seed 42, 2 layers, hidden 128, intermediate 256, 4 attention heads/2 KV heads, local character tokenizer, maximum 8 generated tokens.
- Artificial diffusion: seed 42, local CLIP/UNet/VAE/DDIM pipeline, 32x32, 2 steps. No semantic/image-quality acceptance.
- Pretrained LLM: local Gemma 4 E2B revision `6befbaca7398925921802abd1f277b495b78b738`, model payload SHA256 `33fe0cece08fb527ffefbd1a3a9ce73bd71073727993a283506293e5c6bf0137`.
- Pretrained diffusion: `hf-internal-testing/tiny-stable-diffusion-torch`, saved locally with safetensors before conversion. The similarly named `tiny-stable-diffusion-pipe` is Flax-only and was rejected by the PyTorch pipeline loader.

## Evidence hashes

Reports and logs live under `E:/OpenTernary/cli-validation/reports/`. The table is a snapshot; each report references its per-case logs. Artifacts retain their original manifests; verification reports live outside them.

| Evidence | Report relative to reports/ | SHA256 |
|---|---|---|
| Windows CPU | `wheel-windows-final/validation.json` | `b18e1b38b9e20134698c9afe704d965332434eda66624b6d614522eb751f4686` |
| WSL Ubuntu 24.04 CPU | `wheel-wsl-final/validation.json` | `1fd9326246562a0b8f66869ea8424d914cc5c05e6afa877d5494b94e5effbdc9` |
| RX 9070 XT FP32 | `wheel-gpu-final-fp32/validation.json` | `8a01b878145b1af9327e33a163ea18abf04ab85e422770b6d6c9f75c1f583ee5` |
| RX 9070 XT BF16 | `wheel-gpu-final-bf16/validation.json` | `ddf0170c7dc5fd34fbca12f35d67d08a12ef28ef3740e6c83f958bf631800135` |
| GGUF converter/reader/runtime | `gguf-verified/validation.json` | `07f23f9e7b451c71a753ee6decb7009d2d38fe7a7480d22f25c27d3c73677ef5` |
| Windows plugins | `plugins-windows/report.json` | `be0caf6df348a738c7f199f61a18b45e573f7d7197ecb367c77aa1b22154e04e` |
| WSL plugins | `plugins-wsl/report.json` | `e79da3d5284a4157231800467c84d0853dbd907d264940d3db9394c6847a94d3` |
| Offline regression | `offline-release-candidate/cli-offline-validation.json` | `e8477d0c3282951f3d9958124985783e49f19e0d75feb037eea6881088cb8281` |
| Final offline regression after pretrained fixes | `pretrained-final-20260924/cli-offline-validation.json` | `cd684340415b592abf0f7b937641d4e9d0c53437677ccfd798a8d14a23b7af6b` |

Additional evidence: `offline-release-candidate/cli-offline-tests.xml`, `wsl-process-tests.xml`, `gpu-existing-regressions/tests.xml`, `gpu-resources/validation.json`, each GPU LLM's `reload.resources.json`, each diffusion `evaluation.json`, `task-changes.json`, `protected-source-check.json`, `resource-budget.json`, and `license-inventory.json`.

`baseline-20260923/` preserves 182 pre-task files and their hashes. Existing unrelated dirty changes remain part of the user's working tree. `task-changes.json` identifies changes against that baseline, not against Git HEAD.

## Reproduce

Use an isolated validation environment. Install the wheel from `E:/OpenTernary/cli-validation/dist/`, then pinned CPU or ROCm dependencies. Preserve the original research environments. Use `UV_CACHE_DIR` and temporary directories below the validation root. Do not set TorchAO compatibility-bypass flags.

```text
python D:/VibeCoding/OpenTernary/scripts/validate_cli_real.py --root E:/OpenTernary/cli-validation/reports/new-cpu --require-wheel
python D:/VibeCoding/OpenTernary/scripts/validate_cli_real.py --root E:/OpenTernary/cli-validation/reports/new-gpu --require-wheel --device cuda:0 --dtype bfloat16
python D:/VibeCoding/OpenTernary/scripts/validate_cli_plugins.py --root E:/OpenTernary/cli-validation/reports/new-plugin --uv uv
python D:/VibeCoding/OpenTernary/scripts/validate_cli_pretrained.py --model D:/path/to/gemma4 --quality-data D:/VibeCoding/OpenTernary/data/quality/gemma4-e2b-quality-v1.json --root E:/OpenTernary/cli-validation/pretrained-new --device cuda --dtype bf16 --require-wheel
```

On WSL use the corresponding `/mnt/d/` and `/mnt/e/` paths and the WSL Python. New output folders are required; runs are not silently overwritten. CPU CLI conversion stays on CPU; GPU inference is explicitly requested. GPU workers run sequentially under one lock, with a 4GiB Torch allocator cap. The harness totals recorded GPU case times against 1800 seconds. Resource measurements sample process RSS every 20ms and record Torch allocated/reserved peaks; they are not total board VRAM measurements.

GGUF source is fetched from the pinned GitHub commit archive (no model files), extracted under `tools/`, and its exact converter requirements are installed into a separate environment. Build the CPU runtime with:

```text
cmake -S <llama-source> -B <build> -G Ninja -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build <build> --target llama-completion -j 6
python scripts/validate_cli_gguf.py --root <new-output> --llama-source <pinned-source> --converter-python <converter-python> --runtime <llama-completion>
```

For a Windows converter with WSL runtime, add `--runtime-prefix '["wsl","-d","Ubuntu","--"]'` and `--runtime-model /mnt/e/.../artifact/model.gguf`. `validate_cli_gguf.py` verifies the clean commit or the exact source archive, exports FP32, uses that source's real reader to inspect architecture/shapes/types, and runs a bounded CPU generation. The report includes executable arguments and hashes. No claim is made for other architectures or lower GGUF precisions.

The workflow's `real-library-wheel` job runs the same CPU and plugin scripts on Windows/Linux. Local passes are not GitHub Actions passes. The offline suite is rerun with `OPENTERNARY_VALIDATION_ROOT` set to a folder under the validation root.

## Pretrained-model evidence

All files below are under `E:/OpenTernary/cli-validation/real-gemma4-20260923/`. The validation split was used; the test split was not opened.

| Evidence | Result | SHA256 |
|---|---|---|
| `quality-bf16/quality.json` | general PPL 1214.6145, Japanese PPL 2229.7044, instruction 62.5, collapse 0 | `3b0782df2abea9088fcf3773d3c464b1684bb84d55f1d4df5a452d436e7a0800` |
| `quality-ternary/quality.json` | 287747.46 / 212961126.33 / 0 / 8; required Gate exit 6 | `75ffc1481368fe5938f9912169e278a343dab6c24eb09405c90ca9da9d508660` |
| `quality-torchao/quality.json` | 1201.5218 / 2078.2776 / 62.5 / 0; required Gate accepted | `0977020ec54547c5186cfc85d22332679289b7e4242b14d23f59d38864b7da86` |
| `benchmark-bf16/benchmark.json` | 320 output tokens, 6.12 tok/s, inference VRAM 10,725,017,600 bytes | `e44afe6e8fbd99fef062c458cb358cb579f5816bda1bb1ffba484568b656fe09` |
| `benchmark-ternary/benchmark.json` | 320 output tokens, 6.63 tok/s, inference VRAM 10,297,021,952 bytes | `fd734afc8672f4120dd76a573f38123b38472d906e5910e0fd0c009b772c1b9d` |
| `benchmark-torchao/benchmark.json` | 158 output tokens, 18.12 tok/s, inference VRAM 8,513,800,704 bytes | `92d6d740b9d080f4cabb98501616fa6437c93bc353943b33b24715fc4299ae09` |
| `packed-roundtrip-byte-compare.json` | 1951 tensors and 10,208,596,934 bytes exactly equal | `ef77da42559cb55460fcbc7a1423cf6345f98ed821507edb8c04ee281856dd51` |
| `diffusion-pretrained-evaluate/evaluation.json` | pretrained UNet replacement, RX 9070 XT BF16, finite 32x32 output saved | `8f41e248ea58cfcc2a9e6e5db2e49a77898c7a9cdc4938e315fa6663148fb793` |
| `optimize-real-budget/search.json` | real Gemma worker stopped at 1MiB cap, exit 7, resume stable, no best | `d6a8dff1bf306b308bc80de4a1ceb6b0ef29564e667da22225bc6f448f2568c2` |

The Ternary snapshot contains 1951 tensors in 12 shards and is 10,243,172,030 bytes including interface files. The packed artifact is 7,060,122,575 bytes. Its safetensors restoration is 10,243,656,689 bytes because it uses one shard per tensor; the tensor payload is nevertheless byte-identical. The TorchAO artifact is 8,409,963,687 bytes and reloads all 1951 tensors. The final installed validation wheel SHA256 is `7fa3f10d25d866da036586bafa77f1b6b936e3c9574f22565fcb478a7c66ff60`.

TorchAO persisted the quantized weights as `torchao.quantization.Int8Tensor`. A GPU profiler probe observed `aten::to`, `aten::mm`, and `aten::mul`, while TorchAO reported that Triton was unavailable. This proves GPU execution and reduced weight storage, but it does not prove a native INT8 matmul kernel. The result is recorded as dequantize/ordinary-matmul execution.

## Limitations and license inventory

- Pretrained support is verified only for the fixed Gemma 4 and tiny Stable Diffusion inputs above. It does not certify arbitrary architectures or image quality.
- GPU execution does not establish a native INT8/ternary kernel. The real TorchAO probe used ordinary matmul after conversion; native packed Ternary execution is unsupported.
- Generic packaged source provenance may report Git revision `unknown` outside a checkout; per-package file hashes and wheel hashes remain available.
- License metadata inventory is recorded in `license-inventory.json`. Torch/TorchVision include multiple licenses; Diffusers/Transformers use Apache metadata. TorchAO package metadata lacks a license expression, so source-license review remains necessary. The project license remains **TBD** and requires the rights holder's decision. Stable publication was not performed.


Final measurement-condition checks: quality/benchmark APIs seed their own generators, quality schema 3 includes seed, and benchmark comparison checks seed and warmup parity. Affected contracts (34 tests) and the full 328-test suite passed again; installed-wheel quality/search passed in both Windows and WSL. Reports: `quality-seed-windows/quality-search-result.json` and `quality-seed-wsl/quality-search-result.json`. Earlier backend/GPU evidence remains valid for the unchanged conversion/loading/runtime modules.

The final distribution hashes are recorded externally in `release-candidate.json`, and every wheel Python source was checked against the workspace. The full environment metadata and license-file hashes are in `package-versions.json` and `license-inventory-full.json`; original research-environment versions are recorded in `original-env-check.json`.
