"""Hardware detection helpers."""

from __future__ import annotations


def get_device(preference: str = "auto") -> str:
    """デバイス選択。preference が auto なら cuda があれば cuda、なければ cpu."""
    if preference == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except ImportError:
            return "cpu"
    if preference in ("cpu", "cuda"):
        if preference == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but not available")
            except ImportError as e:
                raise RuntimeError("CUDA requested but torch not installed") from e
        return preference
    raise ValueError(f"Unknown device preference: {preference}")


def get_hardware_info() -> dict[str, str]:
    """ハードウェア情報を収集."""
    import platform

    import psutil

    info: dict[str, str] = {}
    info["platform"] = platform.platform()
    info["cpu"] = platform.processor() or "unknown"
    try:
        info["ram_gb"] = str(round(psutil.virtual_memory().total / (1024**3), 2))
    except Exception:
        info["ram_gb"] = "unknown"
    info["python"] = platform.python_version()
    # GPU
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_available"] = "true"
        else:
            info["gpu_name"] = "none"
            info["cuda_available"] = "false"
    except ImportError:
        info["gpu_name"] = "unknown (torch not installed)"
        info["cuda_available"] = "false"
    except Exception:
        info["gpu_name"] = "unknown"
        info["cuda_available"] = "false"
    return info
