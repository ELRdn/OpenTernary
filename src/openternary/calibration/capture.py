"""Teacher activation capture — disk-backed, bounded."""

from __future__ import annotations

import hashlib
import pathlib
import typing

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for calibration capture. Install with: uv sync --extra ml")


def sample_hash(input_ids: torch.Tensor) -> str:  # type: ignore[type-arg]
    """input_ids の SHA256 ハッシュ（contamination 証明用）."""
    _require_torch()
    # input_ids は 1-D or 2-D
    b = input_ids.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()  # type: ignore[union-attr]
    return hashlib.sha256(b).hexdigest()


def build_synthetic_dataloader(
    tokenizer: typing.Any,
    num_samples: int = 32,
    seq_len: int = 128,
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:  # type: ignore[type-arg]
    """Synthetic text → tokenizer → input_ids の dataloader（CI専用）."""
    _require_torch()
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    # 固定の synthetic text
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning systems require careful evaluation.",
        "OpenTernary is a research tool for ternary quantization.",
        "Artificial intelligence will transform many industries.",
        "The weather today is sunny with a chance of rain.",
        "Python is a popular programming language for data science.",
        "Transformers are powerful models for natural language processing.",
        "Quantization reduces model size while preserving capability.",
    ]
    texts: list[str] = []
    for i in range(num_samples):
        texts.append(base_texts[i % len(base_texts)] + f" Sample {i}.")

    batches: list[dict[str, torch.Tensor]] = []
    for txt in texts:
        enc = tokenizer(txt, truncation=True, max_length=seq_len, padding="max_length", return_tensors="pt")
        # enc is BatchEncoding with input_ids [1, seq_len]
        input_ids = enc["input_ids"][0]  # type: ignore[index]
        _am = enc.get("attention_mask")
        attention_mask = _am[0] if _am is not None else torch.ones_like(input_ids)  # type: ignore[index]
        batches.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return batches


class DiskActivationCache:
    """Disk-backed activation cache.

    Layout: cache_dir / layer_name_sanitized / batch_{idx}.pt
    Each file contains {"input": Tensor, "teacher_output": Tensor, "bias": Tensor|None}
    """

    def __init__(self, cache_dir: pathlib.Path) -> None:
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, layer_name: str, batch_idx: int, data: dict[str, torch.Tensor]) -> None:  # type: ignore[type-arg]
        _require_torch()
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        layer_dir = self.cache_dir / safe_name
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir / f"batch_{batch_idx:05d}.pt"
        # detach and cpu
        to_save = {k: v.detach().cpu() for k, v in data.items()}
        torch.save(to_save, str(path))  # type: ignore[union-attr]

    def load(self, layer_name: str, batch_idx: int) -> dict[str, torch.Tensor]:  # type: ignore[type-arg]
        _require_torch()
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        path = self.cache_dir / safe_name / f"batch_{batch_idx:05d}.pt"
        return torch.load(str(path), map_location="cpu", weights_only=True)  # type: ignore[union-attr,no-any-return]

    def list_layers(self) -> list[str]:
        if not self.cache_dir.exists():
            return []
        return [p.name for p in self.cache_dir.iterdir() if p.is_dir()]

    def count_batches(self, layer_name: str) -> int:
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        layer_dir = self.cache_dir / safe_name
        if not layer_dir.exists():
            return 0
        return len(list(layer_dir.glob("batch_*.pt")))


def capture_teacher_pairs(
    model: torch.nn.Module,  # type: ignore[type-arg]
    dataloader: list[dict[str, torch.Tensor]],  # type: ignore[type-arg]
    target_module_names: list[str],
    cache_dir: pathlib.Path,
) -> DiskActivationCache:
    """Teacher model から input/output を hook で capture し disk に保存.

    Args:
        model: Teacher model (eval mode, no_grad で forward)
        dataloader: list of {"input_ids", "attention_mask"}
        target_module_names: hook する nn.Linear の名前（model.named_modules の key）
        cache_dir: 保存先

    Returns:
        DiskActivationCache
    """
    _require_torch()
    model.eval()
    cache = DiskActivationCache(cache_dir)
    # Build module dict
    name_to_module: dict[str, nn.Module] = dict(model.named_modules())  # type: ignore[union-attr]
    # Validate targets exist and are Linear
    for n in target_module_names:
        if n not in name_to_module:
            raise ValueError(f"target module not found: {n}")
        # allow any Module, but warn if not Linear

    # Prepare hooks
    activations: dict[str, dict[str, torch.Tensor]] = {}  # type: ignore[type-arg]
    handles: list[typing.Any] = []

    def make_hook(name: str) -> typing.Any:
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:  # type: ignore[type-arg]
            # inputs is tuple, first is input activation
            inp = inputs[0].detach()
            out = output.detach()
            activations[name] = {"input": inp, "teacher_output": out}

        return hook

    for n in target_module_names:
        mod = name_to_module[n]
        h = mod.register_forward_hook(make_hook(n))
        handles.append(h)

    try:
        with torch.no_grad():  # type: ignore[union-attr]
            for batch_idx, batch in enumerate(dataloader):
                activations.clear()
                input_ids = batch["input_ids"].unsqueeze(0) if batch["input_ids"].dim() == 1 else batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None and attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                # Move to model device (usually cpu)
                device = next(model.parameters()).device  # type: ignore[union-attr]
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                # Forward
                kwargs: dict[str, typing.Any] = {"input_ids": input_ids}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                try:
                    model(**kwargs)  # type: ignore[operator]
                except TypeError:
                    # Fallback: try without attention_mask
                    model(input_ids)  # type: ignore[operator]
                # Save per layer
                for layer_name, data in list(activations.items()):
                    cache.save(layer_name, batch_idx, data)
    finally:
        for h in handles:
            h.remove()

    return cache
