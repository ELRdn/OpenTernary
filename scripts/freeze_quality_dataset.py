"""Freeze the revision-pinned P3 quality dataset to a canonical JSON file."""

from __future__ import annotations

import argparse
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"frozen quality dataset: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
