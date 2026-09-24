"""Add disjoint SQuAD/JSQuAD training answers to the frozen mixed calibration set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
JGLUE_REVISION = "7f983b6d00db19b534b13ef2a27a7b31576ad812"


def normalized(value: str) -> str:
    return " ".join(value.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--quality", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base_bytes = args.base.read_bytes()
    base = json.loads(base_bytes)
    if base.get("schema_version") != 1 or base.get("status") != "calibration_only":
        raise ValueError("invalid base calibration set")
    excluded_ids = set()
    excluded_contexts = set()
    quality_hashes = {}
    for path in args.quality:
        raw = path.read_bytes()
        quality_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        dataset = json.loads(raw)
        for split in ("validation", "test"):
            for row in dataset["splits"][split]:
                excluded_ids.add(str(row["id"]).split("-", 2)[-1])
                excluded_contexts.add(normalized(str(row["text"])))
            for row in dataset["instruction_cases"][split]:
                excluded_ids.add(str(row["id"]).split("-", 2)[-1])
    squad = load_dataset("rajpurkar/squad", split="train", revision=SQUAD_REVISION)
    japanese_path = hf_hub_download(
        "shunk031/JGLUE",
        repo_type="dataset",
        revision=JGLUE_REVISION,
        filename="JSQuAD/jglue-train.parquet",
    )
    japanese = pq.read_table(japanese_path).to_pylist()
    rows_by_language = {"en": squad, "ja": japanese}
    qa_rows = []
    for language, rows in rows_by_language.items():
        ordered = sorted(rows, key=lambda row: hashlib.sha256(str(row["id"]).encode()).hexdigest())
        seen_contexts = set()
        selected = 0
        for row in ordered:
            context = normalized(str(row["context"]))
            answer_texts = row["answers"]["text"]
            answer_starts = row["answers"]["answer_start"]
            answer = str(answer_texts[0]).strip() if answer_texts else ""
            answer_start = int(answer_starts[0]) if answer_starts else -1
            row_id = str(row["id"])
            if (
                not answer
                or not 0 <= answer_start < 100
                or answer_start + len(answer) > 145
                or row_id in excluded_ids
                or context in excluded_contexts
                or context in seen_contexts
                or len(context) < 150
                or answer not in context[:160]
            ):
                continue
            seen_contexts.add(context)
            excerpt = context[:160]
            if language == "ja":
                prompt = (
                    "次の文章を読み、質問に答えてください。答えだけを原文どおりに出力し、説明は付けないでください。\n"
                    f"文章: {excerpt}\n質問: {row['question']}"
                )
            else:
                prompt = (
                    "Read the passage and answer the question. Output only the exact answer from the passage, with no explanation.\n"
                    f"Passage: {excerpt}\nQuestion: {row['question']}"
                )
            qa_rows.append(
                {
                    "id": f"{language}-qa-train-{row_id}",
                    "kind": "qa",
                    "language": language,
                    "text": prompt,
                    "answer": answer,
                }
            )
            selected += 1
            if selected == 32:
                break
        if selected != 32:
            raise ValueError(f"insufficient disjoint {language} QA training rows")
    train, held = [], []
    for index, row in enumerate(qa_rows):
        (held if index % 4 == 3 else train).append(row)
    train = [*base["train"], *train]
    held = [*base["held"], *held]
    if len({row["id"] for row in train + held}) != len(train + held):
        raise ValueError("duplicate calibration IDs")
    result = {
        "schema_version": 1,
        "status": "calibration_only",
        "base_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "sources": {**base["sources"], "squad_train_revision": SQUAD_REVISION},
        "excluded_quality_sha256": quality_hashes,
        "train": train,
        "held": held,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train), "held": len(held), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
