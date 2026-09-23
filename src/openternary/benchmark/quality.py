"""Opt-in quality primitives; smoke v1 remains unchanged.

No benchmark data is downloaded or chosen here. Results are measurements, not
scientific acceptance; experiment owners must preregister their quality gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Iterable
from typing import Any

_MINHASH_PERMUTATIONS = 128
_MINHASH_THRESHOLD = 0.9
_SHINGLE_WIDTH = 5


def _minhash_signature(text: str) -> tuple[int, ...]:
    shingles = {text[index : index + _SHINGLE_WIDTH] for index in range(max(1, len(text) - _SHINGLE_WIDTH + 1))}
    if not shingles:
        shingles = {text}
    signature = []
    for permutation in range(_MINHASH_PERMUTATIONS):
        prefix = permutation.to_bytes(2, "little")
        signature.append(
            min(
                int.from_bytes(hashlib.blake2b(prefix + shingle.encode("utf-8"), digest_size=8).digest(), "little")
                for shingle in shingles
            )
        )
    return tuple(signature)


def _minhash_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def validate_text_splits(splits: dict[str, list[str]]) -> dict[str, Any]:
    """Fail closed on empty, duplicated or overlapping normalized exact texts.

    This is not a semantic/near-duplicate contamination proof.
    """
    if set(splits) != {"calibration", "validation", "test"}:
        raise ValueError("require calibration, validation and test splits")
    hashes: dict[str, list[str]] = {}
    normalized_splits: dict[str, list[str]] = {}
    for split, texts in splits.items():
        if not texts:
            raise ValueError(f"empty {split} split")
        normalized = [" ".join(unicodedata.normalize("NFKC", text).split()) for text in texts]
        if any(not text for text in normalized):
            raise ValueError("empty normalized text")
        normalized_splits[split] = normalized
        hashes[split] = [hashlib.sha256(text.encode()).hexdigest() for text in normalized]
        if len(set(hashes[split])) != len(texts):
            raise ValueError(f"duplicate texts in {split}")
    for left, right in (("calibration", "validation"), ("calibration", "test"), ("validation", "test")):
        if not set(hashes[left]).isdisjoint(hashes[right]):
            raise ValueError(f"split overlap: {left}/{right}")
        left_signatures = [_minhash_signature(text) for text in normalized_splits[left]]
        right_signatures = [_minhash_signature(text) for text in normalized_splits[right]]
        for left_signature in left_signatures:
            for right_signature in right_signatures:
                if _minhash_similarity(left_signature, right_signature) >= _MINHASH_THRESHOLD:
                    raise ValueError(f"split near-duplicate: {left}/{right} (MinHash >= {_MINHASH_THRESHOLD})")
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {
        "disjoint": True,
        "sample_hashes": hashes,
        "fingerprint": fingerprint,
        "normalization": "NFKC-whitespace-v1",
        "near_duplicate": {
            "method": "character-5gram-minhash-v1",
            "permutations": _MINHASH_PERMUTATIONS,
            "threshold": _MINHASH_THRESHOLD,
        },
        "scope": "normalized exact and cross-split MinHash near-duplicates",
    }


def validate_instruction_splits(splits: dict[str, list[str]]) -> dict[str, Any]:
    """Fail closed on duplicate or near-duplicate validation/test prompts."""
    if set(splits) != {"validation", "test"}:
        raise ValueError("instruction splits require validation and test")
    hashes: dict[str, list[str]] = {}
    normalized_splits: dict[str, list[str]] = {}
    for split, prompts in splits.items():
        if not prompts:
            raise ValueError(f"empty instruction {split} split")
        normalized = [" ".join(unicodedata.normalize("NFKC", prompt).split()) for prompt in prompts]
        if any(not prompt for prompt in normalized):
            raise ValueError("empty normalized instruction prompt")
        normalized_splits[split] = normalized
        hashes[split] = [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in normalized]
        if len(set(hashes[split])) != len(prompts):
            raise ValueError(f"duplicate instruction prompts in {split}")

    if not set(hashes["validation"]).isdisjoint(hashes["test"]):
        raise ValueError("instruction split overlap: validation/test")
    validation_signatures = [_minhash_signature(prompt) for prompt in normalized_splits["validation"]]
    test_signatures = [_minhash_signature(prompt) for prompt in normalized_splits["test"]]
    max_similarity = 0.0
    for validation_signature in validation_signatures:
        for test_signature in test_signatures:
            similarity = _minhash_similarity(validation_signature, test_signature)
            max_similarity = max(max_similarity, similarity)
            if similarity >= _MINHASH_THRESHOLD:
                raise ValueError(f"instruction split near-duplicate: validation/test (MinHash >= {_MINHASH_THRESHOLD})")
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {
        "disjoint": True,
        "sample_hashes": hashes,
        "fingerprint": fingerprint,
        "normalization": "NFKC-whitespace-v1",
        "near_duplicate": {
            "method": "character-5gram-minhash-v1",
            "permutations": _MINHASH_PERMUTATIONS,
            "threshold": _MINHASH_THRESHOLD,
            "maximum_cross_split_similarity": max_similarity,
        },
        "scope": "normalized exact and cross-split MinHash instruction prompts",
    }


def output_diagnostics(text: str, *, min_repetitions: int = 8, max_unit_length: int = 8) -> dict[str, Any]:
    """Conservative flags, not a language-quality or instruction-following score."""
    if min_repetitions < 2 or max_unit_length < 1:
        raise ValueError("invalid repetition detector settings")
    compact = "".join(text.split())
    flags = []
    if not compact:
        flags.append("empty")
    elif not any(character.isalnum() for character in compact):
        flags.append("symbols_only")
    for width in range(1, min(max_unit_length, len(compact) // min_repetitions) + 1):
        if len(compact) % width == 0 and compact == compact[:width] * (len(compact) // width):
            flags.append("repeated_unit")
            break
    return {
        "flags": flags,
        "characters": len(text),
        "detector_version": "exact-repeat-v1",
        "min_repetitions": min_repetitions,
        "max_unit_length": max_unit_length,
    }


def evaluate_perplexity(
    model: Any,
    sequences: Iterable[Any],
    *,
    max_length: int,
    stride: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Causal NLL, no cross-document context, each non-initial token scored once.

    Sequences must contain unpadded token IDs. The model is already loaded by
    the caller. No silent dtype conversion or download is performed.
    """
    import torch
    import torch.nn.functional as functional

    if not 1 <= stride < max_length:
        raise ValueError("require 1 <= stride < max_length")
    nll_sum = 0.0
    scored_tokens = 0
    previous_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for sequence in sequences:
                if sequence.ndim != 1 or sequence.numel() < 2:
                    raise ValueError("each unpadded sequence must contain at least two token IDs")
                next_target = 1
                for start in range(0, sequence.numel() - 1, stride):
                    end = min(start + max_length, sequence.numel())
                    ids = sequence[start:end].unsqueeze(0).to(device)
                    logits = model(input_ids=ids).logits
                    first = max(next_target, start + 1) - start
                    predictions = logits[:, first - 1 : -1, :].float()
                    targets = ids[:, first:]
                    if not torch.isfinite(predictions).all().item():
                        raise ValueError("non-finite evaluation logits")
                    loss_sum = functional.cross_entropy(
                        predictions.reshape(-1, predictions.shape[-1]), targets.reshape(-1), reduction="sum"
                    )
                    if not torch.isfinite(loss_sum).item():
                        raise ValueError("non-finite evaluation loss")
                    nll_sum += float(loss_sum.item())
                    scored_tokens += targets.numel()
                    next_target = end
                    if end == sequence.numel():
                        break
    finally:
        model.train(previous_training)
    if not scored_tokens:
        raise ValueError("no evaluation tokens")
    mean_nll = nll_sum / scored_tokens
    try:
        ppl = math.exp(mean_nll)
    except OverflowError as exc:
        raise ValueError("perplexity overflow") from exc
    return {"nll_sum": nll_sum, "scored_tokens": scored_tokens, "mean_nll": mean_nll, "perplexity": ppl}


def evaluate_quality(
    model: Any,
    tokenizer: Any,
    splits: dict[str, list[dict[str, str]]],
    *,
    split: str,
    max_length: int,
    stride: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Measure a supplied split; caller owns model loading and provenance.

    Records require text and language. Never select hyperparameters on test.
    Save the returned JSON alongside model/code revision and tokenizer identity.
    """
    if split not in {"validation", "test"}:
        raise ValueError("quality evaluation requires validation or test")
    audit = validate_text_splits({name: [item["text"] for item in items] for name, items in splits.items()})
    by_language: dict[str, dict[str, Any]] = {}
    languages = sorted({item["language"] for item in splits[split]})
    if any(not language for language in languages):
        raise ValueError("language labels must not be empty")
    for language in languages:
        sequences = (
            tokenizer(item["text"], return_tensors="pt", padding=False, truncation=False)["input_ids"][0]
            for item in splits[split]
            if item["language"] == language
        )
        by_language[language] = evaluate_perplexity(
            model, sequences, max_length=max_length, stride=stride, device=device
        )
    total_tokens = sum(result["scored_tokens"] for result in by_language.values())
    total_nll = sum(result["nll_sum"] for result in by_language.values())
    mean_nll = total_nll / total_tokens
    protocol = {
        "version": "quality-ppl-v1",
        "split": split,
        "max_length": max_length,
        "stride": stride,
        "data_fingerprint": audit["fingerprint"],
        "document_boundaries": "independent",
        "padding": False,
        "truncation": False,
    }
    return {
        "protocol": protocol,
        "protocol_fingerprint": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "data_audit": audit,
        "by_language": by_language,
        "overall": {
            "nll_sum": total_nll,
            "scored_tokens": total_tokens,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
        },
        "scientific_acceptance": False,
        "acceptance_note": "measurement only; provenance and preregistered quality gates still required",
    }


def score_responses(cases: list[dict[str, str]], responses: dict[str, str]) -> dict[str, Any]:
    """Score closed-answer instructions only; free-form responses get diagnostics.

    'expected' is an exact string, with no hidden normalization or LLM judge.
    Callers must save the cases and generation protocol before running a model.
    """
    ids = [case["id"] for case in cases]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(responses):
        raise ValueError("case and response IDs must match exactly and be unique")
    rows: list[dict[str, Any]] = []
    for case in cases:
        text = responses[case["id"]]
        rows.append(
            {
                "id": case["id"],
                "response_text": text,
                "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "diagnostics": output_diagnostics(text),
                "exact_match": text == case["expected"] if "expected" in case else None,
            }
        )
    scored = [row["exact_match"] for row in rows if row["exact_match"] is not None]
    return {
        "results": rows,
        "exact_match_scored": len(scored),
        "exact_match_accuracy": sum(scored) / len(scored) if scored else None,
        "flagged_outputs": sum(bool(row["diagnostics"]["flags"]) for row in rows),
        "scorer_version": "closed-answer-exact-v1",
        "scientific_acceptance": False,
    }
