"""Tests for disk-backed activation cache."""

import pathlib

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from openternary.calibration.capture import (  # noqa: E402
    DiskActivationCache,
    build_synthetic_dataloader,
    capture_teacher_pairs,
    sample_hash,
)


def test_sample_hash_deterministic() -> None:
    a = torch.tensor([1, 2, 3])
    h1 = sample_hash(a)
    h2 = sample_hash(a)
    assert h1 == h2
    b = torch.tensor([1, 2, 4])
    assert sample_hash(b) != h1


def test_disk_cache_save_load() -> None:
    import shutil
    import uuid

    tmp = pathlib.Path.cwd() / f"test_cache_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        cache = DiskActivationCache(tmp / "cache")
        data = {"input": torch.randn(2, 4), "teacher_output": torch.randn(2, 4)}
        cache.save("layer_0", 0, data)
        loaded = cache.load("layer_0", 0)
        assert torch.equal(loaded["input"], data["input"])
        assert cache.count_batches("layer_0") == 1
        assert cache.list_layers() == ["layer_0"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_capture_teacher_pairs_dummy() -> None:
    # Dummy model with 2 linears
    class Dummy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
            # For test, we just pass through fc1/fc2 sequentially on dummy hidden
            h = torch.randn(input_ids.shape[1], 4)
            h = self.fc1(h)
            h = self.fc2(h)
            return h

    model = Dummy()
    model.eval()

    # Create dummy tokenizer that returns fixed input_ids
    class DummyTokenizer:
        def __call__(
            self,
            text: str,
            truncation: bool = True,
            max_length: int = 8,
            padding: str = "max_length",
            return_tensors: str = "pt",
        ) -> dict[str, torch.Tensor]:  # type: ignore[no-untyped-def]
            # Return input_ids [1, max_length]
            import torch as _t

            return {"input_ids": _t.randint(0, 100, (1, max_length)), "attention_mask": _t.ones(1, max_length)}

    tok = DummyTokenizer()
    batches = build_synthetic_dataloader(tok, num_samples=4, seq_len=8, seed=42)  # type: ignore[arg-type]
    assert len(batches) == 4

    import shutil
    import uuid

    tmp2 = pathlib.Path.cwd() / f"test_act_{uuid.uuid4().hex[:6]}"
    tmp2.mkdir(parents=True, exist_ok=True)
    try:
        cache_dir = tmp2 / "act_cache"
        cache = capture_teacher_pairs(model, batches, ["fc1", "fc2"], cache_dir)
        assert cache.count_batches("fc1") == 4
        assert cache.count_batches("fc2") == 4
        # Ensure files are on disk, not just RAM
        assert (cache_dir / "fc1").exists()
        assert (cache_dir / "fc2").exists()
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)
