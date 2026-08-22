"""Calibration dataset abstraction — wiki-tiny / synthetic."""

from __future__ import annotations

import hashlib
import random
import typing

try:
    import torch  # noqa: F401
except ImportError:
    torch = None  # type: ignore[assignment]


SYNTHETIC_TEXTS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning systems require careful evaluation.",
    "OpenTernary is a research tool for ternary quantization.",
    "Artificial intelligence will transform many industries.",
    "The weather today is sunny with a chance of rain.",
    "Python is a popular programming language for data science.",
    "Transformers are powerful models for natural language processing.",
    "Quantization reduces model size while preserving capability.",
]

SMOKE_TEXTS: list[str] = [
    "Write a one-sentence greeting in English.",
    "日本語で一文の挨拶を書いてください。",
    "次の文を日本語で一文で要約してください:「OpenTernaryはGemma 4 E2Bを三値化する研究ツールで、再現可能なCLIパイプラインを構築することを目的としています。」",
    "If all Bloops are Razzies and some Razzies are Loppies, is it certain that some Bloops are Loppies? Answer yes/no and explain in one sentence.",
    "Write a Python function add(a, b) that returns a + b. Reply with code only.",
]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_input_ids(input_ids: typing.Any) -> str:
    """input_idsのSHA256（tensor or list）."""
    try:
        import torch as _t  # type: ignore[import]

        if isinstance(input_ids, _t.Tensor):
            b = input_ids.detach().cpu().contiguous().view(_t.uint8).numpy().tobytes()  # type: ignore[union-attr]
            return hashlib.sha256(b).hexdigest()
    except Exception:
        pass
    # fallback: list of ints
    if isinstance(input_ids, list):
        b = str(input_ids).encode("utf-8")
        return hashlib.sha256(b).hexdigest()
    # generic
    return hashlib.sha256(str(input_ids).encode("utf-8")).hexdigest()


def load_synthetic_texts(num_samples: int) -> list[str]:
    """Synthetic固定テキストからnum_samples件を生成."""
    texts: list[str] = []
    for i in range(num_samples):
        texts.append(SYNTHETIC_TEXTS[i % len(SYNTHETIC_TEXTS)] + f" Sample {i}.")
    return texts


def load_wikitext_texts(num_samples: int, seed: int) -> list[str]:
    """Salesforce/wikitext wikitext-2-raw-v1 trainからテキストを読込む."""
    try:
        import datasets  # type: ignore[import]
    except ImportError as e:
        raise ImportError("datasets is required for wiki-tiny. Install with: uv sync --extra calibration") from e

    try:
        ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")  # type: ignore[union-attr]
    except Exception as e:
        raise RuntimeError(f"failed to load wikitext dataset: {e}") from e

    # collect non-empty texts
    raw_texts: list[str] = []
    for row in ds:
        txt = row.get("text", "") if isinstance(row, dict) else str(row)
        if txt is None:
            continue
        txt = txt.strip()
        if not txt:
            continue
        # wikitext contains headers like " = = Title = =" — keep but not empty
        raw_texts.append(txt)

    if len(raw_texts) < num_samples:
        raise ValueError(f"not enough non-empty wikitext samples: have {len(raw_texts)}, need {num_samples}")

    # deterministic shuffle
    rng = random.Random(seed)
    indices = list(range(len(raw_texts)))
    rng.shuffle(indices)
    selected = [raw_texts[i] for i in indices[:num_samples]]
    return selected


def get_calibration_texts(
    dataset: str,
    num_samples: int,
    seed: int,
    allow_fallback: bool,
) -> tuple[list[str], str, bool]:
    """dataset名からテキストを取得、fallback可否を考慮.

    Returns: (texts, effective_dataset, fallback_used)
    """
    requested = dataset
    if requested == "synthetic":
        return load_synthetic_texts(num_samples), "synthetic", False

    if requested == "wiki-tiny":
        try:
            texts = load_wikitext_texts(num_samples, seed)
            return texts, "wiki-tiny", False
        except (ImportError, RuntimeError, ValueError, OSError):
            if not allow_fallback:
                raise
            # fallback to synthetic
            fallback_texts = load_synthetic_texts(num_samples)
            return fallback_texts, "synthetic", True

    # unknown dataset (c4-tiny等はConfigで弾かれるはずだが、念のため)
    raise ValueError(f"unsupported dataset: {requested}")


def split_train_held(texts: list[str], held_out_ratio: float) -> tuple[list[str], list[str]]:
    """TRAIN / HELD-OUTへ分割."""
    n = len(texts)
    n_held = max(1, int(n * held_out_ratio))
    n_train = n - n_held
    # textsは既にshuffle済みなので先頭をtrain、末尾をheldに
    return texts[:n_train], texts[n_train:]


def tokenize_texts(
    texts: list[str],
    tokenizer: typing.Any,
    seq_len: int,
) -> list[dict[str, typing.Any]]:
    """テキストをtokenizerでinput_idsへ変換（truncate/pad）。"""
    batches: list[dict[str, typing.Any]] = []
    for txt in texts:
        enc = tokenizer(txt, truncation=True, max_length=seq_len, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"][0]  # type: ignore[index]
        _am = enc.get("attention_mask")
        attention_mask = _am[0] if _am is not None else None  # type: ignore[index]
        # keep tensor
        item: dict[str, typing.Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            item["attention_mask"] = attention_mask
        # also keep text for hash fallback
        item["_text"] = txt
        batches.append(item)
    return batches


def hash_batches(batches: list[dict[str, typing.Any]]) -> list[str]:
    """batchesのsample-level hash（input_idsベース）。"""
    hashes: list[str] = []
    for b in batches:
        ids = b.get("input_ids")
        if ids is not None:
            try:
                hashes.append(_hash_input_ids(ids))
                continue
            except Exception:
                pass
        # fallback to text
        txt = b.get("_text", "")
        hashes.append(_hash_text(txt))
    return hashes
