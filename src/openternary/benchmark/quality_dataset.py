"""Deterministic construction of the frozen P3 quality dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

from openternary.benchmark.quality import validate_text_splits

QUALITY_DATASET_VERSION = "squad-jsquad-v1"


def _stable_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest())


def _unique_context_rows(rows: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _stable_rows(rows):
        context = " ".join(str(row.get("context", "")).split())
        answers = row.get("answers")
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        if len(context) < 128 or not row.get("question") or not answer_texts or context in seen:
            continue
        seen.add(context)
        selected.append({**row, "context": context})
        if len(selected) == count:
            return selected
    raise ValueError(f"not enough unique QA contexts: need {count}, found {len(selected)}")


def _instruction_case(row: dict[str, Any], *, prefix: str, language: str) -> dict[str, str]:
    answer = str(row["answers"]["text"][0]).strip()
    if language == "ja":
        prompt = (
            "次の文章を読み、質問に答えてください。答えだけを原文どおりに出力し、説明は付けないでください。\n"
            f"文章: {row['context']}\n質問: {row['question']}"
        )
    else:
        prompt = (
            "Read the passage and answer the question. Output only the exact answer from the passage, with no explanation.\n"
            f"Passage: {row['context']}\nQuestion: {row['question']}"
        )
    return {"id": f"{prefix}-{row['id']}", "prompt": prompt, "expected": answer}


def build_quality_dataset(
    *,
    calibration_texts: Sequence[str],
    squad_rows: Iterable[dict[str, Any]],
    jsquad_rows: Iterable[dict[str, Any]],
    source_revisions: dict[str, str],
    contexts_per_language_per_split: int = 8,
    instructions_per_language_per_split: int = 4,
) -> dict[str, Any]:
    """Build validation/test splits without depending on source row order."""

    if not calibration_texts or any(not text.strip() for text in calibration_texts):
        raise ValueError("calibration_texts must be non-empty")
    if not 0 < instructions_per_language_per_split <= contexts_per_language_per_split:
        raise ValueError("instruction count must be within the selected context count")
    required_revisions = {"wikitext", "squad", "jglue"}
    if set(source_revisions) != required_revisions or any(not source_revisions[key] for key in required_revisions):
        raise ValueError(f"source_revisions must contain exactly {sorted(required_revisions)}")

    per_source = contexts_per_language_per_split * 2
    english = _unique_context_rows(squad_rows, per_source)
    japanese = _unique_context_rows(jsquad_rows, per_source)
    split_rows = {
        "validation": (english[:contexts_per_language_per_split], japanese[:contexts_per_language_per_split]),
        "test": (english[contexts_per_language_per_split:], japanese[contexts_per_language_per_split:]),
    }
    splits: dict[str, list[dict[str, str]]] = {
        "calibration": [
            {"id": f"wikitext-train-{index:03d}", "text": text, "language": "en"}
            for index, text in enumerate(calibration_texts)
        ]
    }
    instruction_cases: dict[str, list[dict[str, str]]] = {}
    for split_name, (english_rows, japanese_rows) in split_rows.items():
        splits[split_name] = [
            {
                "id": f"squad-{split_name}-{row['id']}",
                "text": str(row["context"]),
                "language": "en",
            }
            for row in english_rows
        ] + [
            {
                "id": f"jsquad-{split_name}-{row['id']}",
                "text": str(row["context"]),
                "language": "ja",
            }
            for row in japanese_rows
        ]
        instruction_cases[split_name] = [
            _instruction_case(row, prefix=f"squad-{split_name}", language="en")
            for row in english_rows[:instructions_per_language_per_split]
        ] + [
            _instruction_case(row, prefix=f"jsquad-{split_name}", language="ja")
            for row in japanese_rows[:instructions_per_language_per_split]
        ]

    validate_text_splits({name: [row["text"] for row in records] for name, records in splits.items()})
    revision_material = json.dumps(
        {"builder": QUALITY_DATASET_VERSION, "sources": source_revisions},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "dataset": {
            "id": "OpenTernary/quality-squad-jsquad-v1",
            "revision": f"sha256:{hashlib.sha256(revision_material).hexdigest()}",
            "license": "WikiText: CC-BY-SA-3.0/GFDL; SQuAD: CC-BY-SA-4.0; JGLUE: CC-BY-4.0",
            "source": [
                {
                    "id": "Salesforce/wikitext",
                    "revision": source_revisions["wikitext"],
                    "split": "train",
                },
                {"id": "rajpurkar/squad", "revision": source_revisions["squad"], "split": "validation"},
                {
                    "id": "shunk031/JGLUE",
                    "revision": source_revisions["jglue"],
                    "config": "JSQuAD",
                    "split": "validation",
                },
            ],
            "selection": {
                "version": QUALITY_DATASET_VERSION,
                "ordering": "sha256(source record id)",
                "contexts_per_language_per_split": contexts_per_language_per_split,
                "instructions_per_language_per_split": instructions_per_language_per_split,
            },
        },
        "splits": splits,
        "instruction_cases": instruction_cases,
    }


__all__ = ["QUALITY_DATASET_VERSION", "build_quality_dataset"]
