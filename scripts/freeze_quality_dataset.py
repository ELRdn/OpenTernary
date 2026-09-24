"""Freeze the revision-pinned P3 quality dataset to a canonical JSON file."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
JGLUE_REVISION = "7f983b6d00db19b534b13ef2a27a7b31576ad812"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--contexts-per-language-per-split", type=int, default=8)
    parser.add_argument("--instructions-per-language-per-split", type=int, default=4)
    parser.add_argument("--exclude-dataset", type=pathlib.Path, action="append", default=[])
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    from openternary.benchmark.quality_dataset import build_quality_dataset
    from openternary.calibration.dataset import load_wikitext_texts

    calibration_texts = load_wikitext_texts(
        args.calibration_samples,
        seed=42,
        revision=WIKITEXT_REVISION,
    )
    squad = load_dataset(
        "rajpurkar/squad",
        split="validation",
        revision=SQUAD_REVISION,
    )
    jsquad_path = hf_hub_download(
        repo_id="shunk031/JGLUE",
        repo_type="dataset",
        revision=JGLUE_REVISION,
        filename="JSQuAD/jglue-validation.parquet",
    )
    jsquad_rows = pq.read_table(jsquad_path).to_pylist()
    excluded_sha256s = []
    excluded_ids = set()
    excluded_contexts = set()
    for excluded_path in args.exclude_dataset:
        excluded_bytes = excluded_path.read_bytes()
        excluded_sha256s.append(hashlib.sha256(excluded_bytes).hexdigest())
        excluded = json.loads(excluded_bytes)
        excluded_ids.update(
            str(row["id"]).split("-", 2)[-1] for split in ("validation", "test") for row in excluded["splits"][split]
        )
        excluded_contexts.update(
            " ".join(str(row["text"]).split()) for split in ("validation", "test") for row in excluded["splits"][split]
        )
    if excluded_sha256s:
        squad = [
            row
            for row in squad
            if str(row["id"]) not in excluded_ids and " ".join(str(row["context"]).split()) not in excluded_contexts
        ]
        jsquad_rows = [
            row
            for row in jsquad_rows
            if str(row["id"]) not in excluded_ids and " ".join(str(row["context"]).split()) not in excluded_contexts
        ]
    payload = build_quality_dataset(
        calibration_texts=calibration_texts,
        squad_rows=squad,
        jsquad_rows=jsquad_rows,
        source_revisions={
            "wikitext": WIKITEXT_REVISION,
            "squad": SQUAD_REVISION,
            "jglue": JGLUE_REVISION,
        },
        contexts_per_language_per_split=args.contexts_per_language_per_split,
        instructions_per_language_per_split=args.instructions_per_language_per_split,
    )
    if excluded_sha256s:
        payload["dataset"]["id"] = f"OpenTernary/quality-squad-jsquad-disjoint-v{len(excluded_sha256s) + 1}"
        if len(excluded_sha256s) == 1:
            payload["dataset"]["selection"]["excluded_dataset_sha256"] = excluded_sha256s[0]
        else:
            payload["dataset"]["selection"]["excluded_dataset_sha256s"] = excluded_sha256s
        payload["dataset"]["selection"]["cross_dataset_exact_context_disjoint"] = True
        selected_ids = sorted(row["id"] for split in ("validation", "test") for row in payload["splits"][split])
        revision_input = json.dumps(
            {
                "sources": payload["dataset"]["source"],
                "selection": payload["dataset"]["selection"],
                "selected_ids": selected_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload["dataset"]["revision"] = f"sha256:{hashlib.sha256(revision_input).hexdigest()}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"frozen quality dataset: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
