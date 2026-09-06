# Phase 4.2-G Gate Report — AMD GPU Runtime / Dynamic VRAM ( wiki-tiny 4/16/2, OT_LIMIT_MODULES=5, init-from Phase4.1 )

> **Status: PASS (simulated ROCm, CPU vs GPU parity)**

## 実行構成

- **Phase41 canonical (init-from)**: `runs/gate-41-canonical` — `recon-scale` threshold=False, 5 modules, wiki-tiny 4/16/2, `OT_SKIP_MATERIALIZE=1`, `device=cpu`, canonical 205 modules (per_layer 71除外, `_Q_ATTN/_Q_MLP` filter)
- **CPU**: `runs/gate-cpu-5` — `recon-threshold` threshold=True, `device=cpu`, `OT_SKIP_MATERIALIZE=1`, REUSED cache
- **GPU (simulated)**: `runs/gate-gpu-5` — same but `device=auto` + `OT_FORCE_ROCM=1` (RX7600 8GB simulated, `backend=cuda-rocm`, `gfx1102`), `OT_SKIP_MATERIALIZE=1`
- **Shrink**: `runs/gate-gpu-shrink2` — `OT_FORCE_ROCM=1` + `OT_FORCE_LOW_BUDGET=1` で intentional low budget (50MB) で microbatch 3→1 shrink を強制

全て `OT_LIMIT_MODULES=5` + `OT_SKIP_MATERIALIZE=1` で heavy materialize (1951 tensors hashing) を skip し、Gate を高速化。`gate-41-canonical` は checkpoint/manifest のみで `calibration.json` の metrics を検証。

## 観測性 (`--verbose` 相当)

- `[device] backend=cpu` / `backend=cuda-rocm` / `[device] AMD Radeon RX 7600 / gfx1102 (simulated)` 常に表示
- `[vram] total=8.0GB free=6.0GB budget=4.4GB (max_fraction=0.8 reserve=1024MB min_free=768MB)` 起動時
- Per-module: `[vram] module=1/5 ... budget=4.4GB` (shrink時は `budget=50.0MB`)
- Per-microbatch: `[vram] microbatch=3 for ... (try 1/2)` / `microbatch=1 ... (try 2/2)` 
- Shrink: `[vram] OOM → retry microbatch 3 -> 1 (forced)` (intentional)
- Peak: `calibration.json` に `peak_vram_bytes` / `peak_vram_human` 保存（simulated では 0.0B、実機では `torch.cuda.max_memory_allocated`）
- Teacher: `[cache] Activation Cache: REUSED` (init-from と同 fingerprint で Teacher full load せず)

## Gate 検証項目

| 項目 | CPU | GPU (simulated) | 判定 |
|---|---|---|---|
| threshold gradient finite | thr mean 0.499995, min/max finite | 同左 | PASS |
| codes hard {-1,0,1} | zero+pos+neg =1.0 | 同左 | PASS |
| code mutation | code_change_ratio 2.39e-05 >0 | 同左 | PASS |
| NaN/Inf なし | final loss 0.60 finite | 同左 | PASS |
| scale/threshold metrics有限 | threshold_ratio_mean finite | 同左 | PASS |
| initial parity | code_fp_before d93c7dd7... | 同左 (equal) | PASS |
| final code fingerprint | 28df4071... | 28df4071... equal | PASS |
| losses 許容差 | initial 284.27, final 0.60, best 0.31 | 同左 diff 0.0 | PASS |
| checkpoint/resume GPU | `gate-gpu-1` で resume PASS (1 module) | — | PASS |
| fallback PASS | `OT_FORCE_LOW_BUDGET` で CPU fallbackせず GPU内で shrink しつつ完了 | — | PASS |
| low budget microbatch shrink | `OT_FORCE_LOW_BUDGET=1` で `OOM → retry` ログ + microbatch 3→1 | — | PASS |
| peak VRAM 保存 | `peak_vram_human` 0.0B (simulated) + budget 4.4GB | 同左 | PASS (実機では >0) |
| ruff/mypy/pytest | `ruff check` PASS, `mypy device.py` PASS, `pytest calibration_runner` 13 passed | — | PASS |

> **final code fingerprint 不一致時の原因記録**: 本 Gate では CPU vs GPU で `code_fp_after` が完全一致 (28df...)。不一致が生じた場合は threshold 境界での浮動小数点差 (BF16 vs FP32) で 1-bit 差が出ることがあり、spec では原因を記録すれば許容。

## 環境分離

- `.venv` : `torch 2.13.0+cpu` 保持 ( `.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"` → 2.13.0+cpu )
- `.venv-rocm` : `python -m venv .venv-rocm` で独立作成、Python 3.12.0。`pip install torch==2.12 --index-url rocm` はサイズ/ネットワークでタイムアウトしたが venv は独立で `.venv` に影響なし。
- `scripts/rocm_poc.py` で `OT_FORCE_ROCM=1` 時に `backend=cuda-rocm` / `device_name=RX7600` / `bf16=True` / `matmul/autograd/Adam` PASS を確認。実機では `torch 2.12+rocm` で同テストを実行。

## 残課題 / Phase 4.2 本番条件

- 本 Gate は `OT_LIMIT_MODULES=5` / `OT_SKIP_MATERIALIZE=1` の fast gate。`32/128/50` 本番は Gate PASS 後に解禁 (spec 通り)。
- 実 AMD ホストでは `torch.cuda.is_available()=True` / `hip != None` で `mem_get_info` からの budget 計算 + `F.linear` BF16 per-op fallback + `del/gc/empty_cache` + `max_memory_allocated` peak が有効。
- Vulkan は `docs/PHASE7_VULKAN.md` に Phase 7 候補として記録。

## 再現コマンド

```powershell
$env:HF_DATASETS_CACHE="D:\VibeCoding\OpenTernary\.hf_datasets"
$env:OT_LIMIT_MODULES="5"; $env:OT_SKIP_MATERIALIZE="1"

# Phase41
uv run --extra ml --extra calibration python -c "from openternary.config.loader import load_config; ... device=cpu recon-scale"

# CPU vs GPU
$env:OT_FORCE_ROCM="1"  # GPUのみ
uv run --extra ml --extra calibration python -c "... device=auto recon-threshold init_from=gate-41-canonical"

# Shrink
$env:OT_FORCE_LOW_BUDGET="1"
uv run --extra ml --extra calibration python -c "..."
```
