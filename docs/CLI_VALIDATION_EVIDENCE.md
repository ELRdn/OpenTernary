# CLI real-library validation evidence

Date: 2026-09-23. Local validation only; GitHub CI has not run for these changes.
All generated networks are artificial. No pretrained weights were downloaded or loaded.

## Environment and scope

- Windows CPU: Python 3.12.9, Torch 2.13.0, Transformers 5.15.1.
- WSL Ubuntu 24.04.3: Python 3.12.3, Torch 2.13.0+cpu, Transformers 5.15.1.
- Windows ROCm clone: Python 3.12.0, Torch 2.13.0+rocm10.0.0, Transformers 5.17.0, RX 9070 XT. Embedded HIP reports 7.15.26333.
- TorchAO 0.18.0 / Diffusers 0.40.0 / safetensors 0.8.0 / SentencePiece 0.2.2.
- Converter environment: Torch 2.11.0+cpu, Transformers 4.57.6, numpy 2.2.6, protobuf 4.25.9, as required by the pinned converter. Kept separate from the 2.13 CLI environments.
- llama.cpp commit `e6ab7c1a41054a888ada952eab4c886444c2f5ad`; archive SHA256 `033c29d5fda5a76af9fd0fc0ade185e5bcb2d8935f8165a30d96c1ed1355b663`. Every archived source file was checked. Runtime target at this revision is `llama-completion`.
- Artificial Llama: seed 42, 2 layers, hidden 128, intermediate 256, 4 attention heads/2 KV heads, local character tokenizer, maximum 8 generated tokens.
- Artificial diffusion: seed 42, local CLIP/UNet/VAE/DDIM pipeline, 32x32, 2 steps. No semantic/image-quality acceptance.

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

Additional evidence: `offline-release-candidate/cli-offline-tests.xml`, `wsl-process-tests.xml`, `gpu-existing-regressions/tests.xml`, `gpu-resources/validation.json`, each GPU LLM's `reload.resources.json`, each diffusion `evaluation.json`, `task-changes.json`, `protected-source-check.json`, `resource-budget.json`, and `license-inventory.json`.

`baseline-20260923/` preserves 182 pre-task files and their hashes. Existing unrelated dirty changes remain part of the user's working tree. `task-changes.json` identifies changes against that baseline, not against Git HEAD.

## Reproduce

Use an isolated validation environment. Install the wheel from `E:/OpenTernary/cli-validation/dist/`, then pinned CPU or ROCm dependencies. Preserve the original research environments. Use `UV_CACHE_DIR` and temporary directories below the validation root. Do not set TorchAO compatibility-bypass flags.

```text
python D:/VibeCoding/OpenTernary/scripts/validate_cli_real.py --root E:/OpenTernary/cli-validation/reports/new-cpu --require-wheel
python D:/VibeCoding/OpenTernary/scripts/validate_cli_real.py --root E:/OpenTernary/cli-validation/reports/new-gpu --require-wheel --device cuda:0 --dtype bfloat16
python D:/VibeCoding/OpenTernary/scripts/validate_cli_plugins.py --root E:/OpenTernary/cli-validation/reports/new-plugin --uv uv
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

## Limitations and license inventory

- All pretrained model support and quality acceptance remain unverified. Synthetic quality/search correctly returns no feasible candidate; controlled subprocess fixtures cover successful selection separately.
- GPU execution does not establish a native INT8/ternary kernel. CPU-only TorchAO emits warnings for unavailable CUDA extension libraries on WSL; tested CPU INT8 operations still complete without bypass flags or a silent device substitution.
- Generic packaged source provenance may report Git revision `unknown` outside a checkout; per-package file hashes and wheel hashes remain available.
- License metadata inventory is recorded in `license-inventory.json`. Torch/TorchVision include multiple licenses; Diffusers/Transformers use Apache metadata. TorchAO metadata lacks a license expression, so that field remains unknown pending source-license review. The project license remains **TBD**. No publication decision was made.


Final measurement-condition checks: quality/benchmark APIs seed their own generators, quality schema 3 includes seed, and benchmark comparison checks seed and warmup parity. Affected contracts (34 tests) and the full 328-test suite passed again; installed-wheel quality/search passed in both Windows and WSL. Reports: `quality-seed-windows/quality-search-result.json` and `quality-seed-wsl/quality-search-result.json`. Earlier backend/GPU evidence remains valid for the unchanged conversion/loading/runtime modules.

The final distribution hashes are recorded externally in `release-candidate.json`, and every wheel Python source was checked against the workspace. The full environment metadata and license-file hashes are in `package-versions.json` and `license-inventory-full.json`; original research-environment versions are recorded in `original-env-check.json`.
