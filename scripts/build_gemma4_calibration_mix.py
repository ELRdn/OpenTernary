"""Freeze disjoint English/Japanese text and chat calibration from training sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from openternary.calibration.dataset import WIKITEXT_REVISION, load_wikitext_texts

JGLUE_REVISION = "7f983b6d00db19b534b13ef2a27a7b31576ad812"


def normalized(value: str) -> str:
    return " ".join(value.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    excluded_texts: set[str] = set()
    excluded_ids: set[str] = set()
    quality_hashes = {}
    for path in args.quality:
        raw = path.read_bytes()
        quality_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        for split in ("calibration", "validation", "test"):
            for row in payload["splits"][split]:
                excluded_texts.add(normalized(str(row["text"])))
                excluded_ids.add(str(row["id"]).split("-", 2)[-1])
        for split in ("validation", "test"):
            for row in payload["instruction_cases"][split]:
                excluded_texts.add(normalized(str(row["prompt"])))

    english = []
    for index, value in enumerate(load_wikitext_texts(256, seed=20260925)):
        text = normalized(value)
        if text and text not in excluded_texts and len(text) > 120:
            english.append({"id": f"wikitext-train-{index}", "text": value, "language": "en"})
        if len(english) == 32:
            break
    path = hf_hub_download(
        "shunk031/JGLUE",
        repo_type="dataset",
        revision=JGLUE_REVISION,
        filename="JSQuAD/jglue-train.parquet",
    )
    japanese = []
    seen_contexts = set()
    rows = pq.read_table(path).to_pylist()
    rows.sort(key=lambda row: hashlib.sha256(str(row["id"]).encode()).hexdigest())
    for row in rows:
        context = normalized(str(row["context"]))
        row_id = str(row["id"])
        if (
            not context
            or context in excluded_texts
            or context in seen_contexts
            or row_id in excluded_ids
            or len(context) < 120
        ):
            continue
        seen_contexts.add(context)
        japanese.append(
            {
                "id": f"jsquad-train-{row_id}",
                "context": context,
                "question": str(row["question"]),
                "language": "ja",
            }
        )
        if len(japanese) == 32:
            break
    if len(english) != 32 or len(japanese) != 32:
        raise ValueError("insufficient disjoint calibration sources")

    train = []
    held = []
    for language, rows in (("en", english), ("ja", japanese)):
        for index, row in enumerate(rows):
            kind = "text" if index < 16 else "chat"
            if language == "en":
                text = row["text"]
                if kind == "chat":
                    text = f"Summarize the following passage in one sentence: {text[:500]}"
            else:
                text = row["context"]
                if kind == "chat":
                    text = f"質問: {row['question']}\n文章: {text[:400]}\n短く答えてください。"
            target = held if index % 4 == 3 else train
            target.append({"id": row["id"], "kind": kind, "language": language, "text": text})
    payload = {
        "schema_version": 1,
        "status": "calibration_only",
        "sources": {"wikitext_revision": WIKITEXT_REVISION, "jglue_train_revision": JGLUE_REVISION},
        "excluded_quality_sha256": quality_hashes,
        "train": train,
        "held": held,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train), "held": len(held), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
