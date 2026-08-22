"""Tests for calibration dataset abstraction."""

import pytest

from openternary.calibration.dataset import (
    get_calibration_texts,
    hash_batches,
    load_synthetic_texts,
    split_train_held,
    tokenize_texts,
)


def test_load_synthetic_deterministic() -> None:
    a = load_synthetic_texts(4)
    b = load_synthetic_texts(4)
    assert a == b
    assert len(a) == 4


def test_split_train_held() -> None:
    texts = [f"text {i}" for i in range(10)]
    train, held = split_train_held(texts, 0.2)
    assert len(train) + len(held) == 10
    assert len(held) == 2
    assert len(train) == 8
    assert set(train).isdisjoint(set(held))


def test_split_small_n() -> None:
    texts = [f"t{i}" for i in range(4)]
    train, held = split_train_held(texts, 0.25)
    assert len(held) == 1
    assert len(train) == 3


def test_tokenize_and_hash() -> None:
    # dummy tokenizer
    class DummyTokenizer:
        def __call__(
            self,
            text: str,
            truncation: bool = True,
            max_length: int = 16,
            padding: str = "max_length",
            return_tensors: str = "pt",
        ) -> dict[str, object]:  # type: ignore[no-untyped-def]
            import torch as _t

            # simple: token ids from hash
            ids = _t.randint(0, 100, (1, max_length))
            return {"input_ids": ids, "attention_mask": _t.ones(1, max_length)}

    tok = DummyTokenizer()
    texts = ["hello world", "foo bar"]
    batches = tokenize_texts(texts, tok, 16)  # type: ignore[arg-type]
    assert len(batches) == 2
    hashes = hash_batches(batches)
    assert len(hashes) == 2
    assert hashes[0] != hashes[1]


def test_get_synthetic_no_fallback() -> None:
    texts, eff, fallback = get_calibration_texts("synthetic", 4, 42, False)
    assert eff == "synthetic"
    assert not fallback
    assert len(texts) == 4


def test_get_wiki_tiny_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    # mock datasets.load_dataset
    class FakeDS(list):
        pass

    fake_rows = [{"text": f"wiki text {i}"} for i in range(10)] + [{"text": "   "}, {"text": ""}]

    def fake_load_dataset(*args: object, **kwargs: object) -> FakeDS:  # type: ignore[no-untyped-def]
        return FakeDS(fake_rows)  # type: ignore[return-value]

    import sys
    import types

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    texts, eff, fallback = get_calibration_texts("wiki-tiny", 4, 42, False)
    assert eff == "wiki-tiny"
    assert not fallback
    assert len(texts) == 4
    # deterministic: same seed gives same order
    texts2, _, _ = get_calibration_texts("wiki-tiny", 4, 42, False)
    assert texts == texts2


def test_get_wiki_fallback_when_datasets_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    # ensure datasets not importable
    monkeypatch.delitem(sys.modules, "datasets", raising=False)

    # also make import fail by monkeypatching import
    original_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    # with fallback false, should raise
    with pytest.raises((ImportError, ValueError)):
        get_calibration_texts("wiki-tiny", 4, 42, False)

    # with fallback true, should fallback to synthetic
    texts, eff, fallback = get_calibration_texts("wiki-tiny", 4, 42, True)
    assert eff == "synthetic"
    assert fallback
    assert len(texts) == 4
