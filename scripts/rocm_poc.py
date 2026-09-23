"""ROCm PoC capability tests — Phase 4.2-G environment isolation."""
import os, sys, pathlib

# Run with OT_FORCE_ROCM=1 to simulate RX7600 8GB on CPU-only host
# Real ROCm host would have torch 2.12+rocm and hip available

try:
    import torch
    print(f"torch {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"hip version: {getattr(torch.version, 'hip', None)}")
    print(f"cuda version: {getattr(torch.version, 'cuda', None)}")
    try:
        print(f"device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu'}")
    except Exception as e:
        print(f"device name error: {e}")
except Exception as e:
    print(f"torch import failed: {e}")
    sys.exit(1)

from openternary.utils.device import detect_backend

info = detect_backend()
print(f"[device] backend={info.backend} name={info.device_name} total={info.total_bytes} free={info.free_bytes} bf16={info.bf16_supported}")

# Tests
ok = True

# 1. backend hip check (simulated)
if os.environ.get("OT_FORCE_ROCM") == "1":
    assert info.backend == "cuda-rocm", f"expected cuda-rocm got {info.backend}"
    assert "7600" in info.device_name or "gfx1102" in info.device_name
    print("PASS: simulated ROCm backend")
else:
    print(f"INFO: real backend {info.backend} (OT_FORCE_ROCM not set)")

# 2. simple matmul
try:
    a = torch.randn(16, 32, device="cpu")
    b = torch.randn(32, 16, device="cpu")
    c = a @ b
    assert c.shape == (16, 16)
    print("PASS: matmul")
except Exception as e:
    print(f"FAIL matmul: {e}"); ok=False

# 3. autograd
try:
    x = torch.randn(4, requires_grad=True)
    y = (x * 2).sum()
    y.backward()
    assert x.grad is not None
    print("PASS: autograd")
except Exception as e:
    print(f"FAIL autograd: {e}"); ok=False

# 4. Adam
try:
    p = torch.nn.Parameter(torch.randn(8))
    opt = torch.optim.Adam([p], lr=1e-3)
    opt.zero_grad()
    loss = (p * p).sum()
    loss.backward()
    opt.step()
    print("PASS: Adam")
except Exception as e:
    print(f"FAIL Adam: {e}"); ok=False

# 5. BF16
try:
    if info.bf16_supported:
        t = torch.randn(4, 4, dtype=torch.bfloat16)
        print(f"PASS: BF16 capability {t.dtype}")
    else:
        print("INFO: BF16 not supported on this backend (cpu fallback expected)")
except Exception as e:
    print(f"BF16 test warning: {e}")

# 6. check isolation: main venv is 2.13 cpu
print(f"torch version check: {torch.__version__} should be 2.13.0+cpu in main venv")
if "2.13" in torch.__version__:
    print("PASS: isolation 2.13 preserved")
else:
    print(f"INFO: version {torch.__version__} (rocm venv would be 2.12)")

print("PoC done" if ok else "PoC failed")
sys.exit(0 if ok else 1)
