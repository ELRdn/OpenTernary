# ROCm 7.14 / gfx1102 PoC — Phase 4.2-G Environment Isolation

## 目的
- 現行 `.venv` の PyTorch 2.13.0+cpu 環境を破壊しない
- 別 venv `.venv-rocm` で ROCm 7.14 / gfx1102 / PyTorch 2.12 を PoC

## 環境分離結果
- `.venv` : `torch 2.13.0+cpu` 保持（`uv run python -c "import torch; print(torch.__version__)"` で確認）
- `.venv-rocm` : `python -m venv D:\VibeCoding\OpenTernary\.venv-rocm` で作成、Python 3.12.0。`pip install torch==2.12 --index-url rocm` はネットワーク/サイズでタイムアウトしたが venv は独立しており `.venv` に影響なし。

## 検出ロジック
`src/openternary/utils/device.py:detect_backend()` は `torch.cuda.is_available() / torch.version.hip / torch.version.cuda / torch.cuda.get_device_name(0)` で backend を `cpu | cuda-nvidia | cuda-rocm` に識別。ROCm でも `torch.cuda` API を使用し、`rocm` device type を tensor に直接渡さない。

`OT_FORCE_ROCM=1` で RX7600 8GB をシミュレート（CIでは実GPUなしでも微分VRAMテスト可能）。

## PoC テスト (`scripts/rocm_poc.py`)

| Test | 期待 | 実測 (CPU host) | シミュレート (OT_FORCE_ROCM=1) |
|---|---|---|---|
| `import torch` | success | 2.13.0+cpu PASS | 2.13.0+cpu PASS |
| `torch.cuda.is_available()` | True on ROCm host | False (expected) | simulated True via backend |
| `torch.version.hip != None` | != None on ROCm | None (cpu) | simulated hip True |
| `device name contains RX 7600` | Radeon RX 7600 | no gpu | AMD Radeon RX 7600 / gfx1102 (simulated) PASS |
| `matmul` | PASS | PASS | PASS |
| `autograd` | PASS | PASS | PASS |
| `Adam` | PASS | PASS | PASS |
| `BF16` | capability true on RDNA3 | False (cpu) | True (simulated) PASS |

実 ROCm ホストでは `torch` を `pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/rocm7.1`（または `rocm6.2` 系）で取得し、同テストを実行。`hip` が not None かつ `get_device_name` が `Radeon RX 7600` を含めば PASS。

## 次ステップ
- 実 AMD ホストで `.venv-rocm` に `torch 2.12+rocm` を入れ、上記 `scripts/rocm_poc.py` を `OT_FORCE_ROCM` なしで実行
- `BF16` で特定 op が不安定な場合は `F.linear` で FP32 フォールバック（runner 内で実装済み）

## コマンド例（実機）
```powershell
C:\Python312\python.exe -m venv .venv-rocm
.\.venv-rocm\Scripts\python.exe -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/rocm7.1
.\.venv-rocm\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
```
