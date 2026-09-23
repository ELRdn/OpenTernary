"""Seeded Llama -> pinned llama.cpp converter -> real reader -> CPU generation.

Supply the externally provisioned source tree and converter Python. No download,
dependency installation or runtime build is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from validate_cli_real import fixture, write

COMMIT = "e6ab7c1a41054a888ada952eab4c886444c2f5ad"
ARCHIVE_SHA256 = "033c29d5fda5a76af9fd0fc0ade185e5bcb2d8935f8165a30d96c1ed1355b663"


def verify_source(source: Path) -> None:
    from openternary.services.artifacts import file_hash

    if (source / ".git").exists():
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
        if revision != COMMIT or changes:
            raise ValueError("llama.cpp must be the clean pinned commit")
        return
    archive = source.parent / "llama.cpp.tar.gz"
    if file_hash(archive) != ARCHIVE_SHA256:
        raise ValueError("llama.cpp archive does not match the pinned source")
    with tarfile.open(archive) as files:
        for entry in files:
            if not entry.isfile():
                continue
            name = Path(entry.name).relative_to(f"llama.cpp-{COMMIT}")
            stream = files.extractfile(entry)
            assert stream is not None
            if file_hash(source / name) != hashlib.sha256(stream.read()).hexdigest():
                raise ValueError(f"llama.cpp source changed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--converter-python", type=Path, required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--runtime-prefix", default="[]", help="JSON argv prefix, e.g. WSL launcher")
    parser.add_argument("--runtime-model", help="Path to root/artifact/model.gguf as seen by runtime")
    args = parser.parse_args()
    from openternary.export.gguf import export_gguf
    from openternary.services.artifacts import file_hash, validate_artifact
    from openternary.services.process import run_process

    verify_source(args.llama_source)
    args.root.mkdir(parents=True, exist_ok=False)
    source = fixture(args.root / "source")
    converter = args.llama_source / "convert_hf_to_gguf.py"
    artifact = args.root / "artifact"
    export_gguf(
        source, artifact, converter, output_dtype="f32", timeout_seconds=180, converter_python=args.converter_python
    )
    before = validate_artifact(artifact)["manifest_fingerprint"]
    code = """
import sys,json,numpy as np
sys.path.insert(0,sys.argv[1])
from gguf import GGUFReader
r=GGUFReader(sys.argv[2])
assert len(r.tensors)==21
assert r.fields['general.architecture'].contents()=='llama'
assert r.fields['llama.block_count'].contents()==2
assert r.fields['llama.embedding_length'].contents()==128
assert all(int(t.tensor_type)==0 and np.isfinite(t.data).all() for t in r.tensors)
out={'tensor_count':len(r.tensors),'finite':True,'tensors':[{'name':t.name,'shape':t.shape.tolist(),'type':int(t.tensor_type)} for t in r.tensors], 'metadata_fields':list(r.fields)}
open(sys.argv[3],'w',encoding='utf-8').write(json.dumps(out,indent=2))
"""
    with (args.root / "reader.log").open("w", encoding="utf-8") as log:
        run_process(
            [
                str(args.converter_python),
                "-c",
                code,
                str(args.llama_source / "gguf-py"),
                str(artifact / "model.gguf"),
                str(args.root / "reader.json"),
            ],
            timeout=60,
            stdout=log,
        )
    prefix = json.loads(args.runtime_prefix)
    if not isinstance(prefix, list) or not all(isinstance(s, str) for s in prefix):
        raise ValueError("runtime-prefix must be JSON string array")
    command = [
        *prefix,
        args.runtime,
        "-m",
        args.runtime_model or str(artifact / "model.gguf"),
        "-p",
        "hello",
        "-n",
        "8",
        "-c",
        "512",
        "-ngl",
        "0",
        "-t",
        "2",
        "--seed",
        "42",
        "--temp",
        "0",
        "--no-conversation",
    ]
    with (args.root / "runtime.log").open("w", encoding="utf-8") as log:
        run_process(command, timeout=60, stdout=log)
    runtime_log = (args.root / "runtime.log").read_text(encoding="utf-8", errors="replace")
    assert "eval time" in runtime_log and "n_predict = 8" in runtime_log
    assert validate_artifact(artifact)["manifest_fingerprint"] == before
    write(
        args.root / "validation.json",
        {
            "status": "passed",
            "scope": "synthetic_llama_f32",
            "pretrained": False,
            "quality_accepted": False,
            "llama_commit": COMMIT,
            "converter_sha256": file_hash(converter),
            "artifact_fingerprint": before,
            "command": command,
            "python": sys.executable,
            "evidence": {p.name: file_hash(p) for p in args.root.glob("*") if p.is_file()},
        },
    )


if __name__ == "__main__":
    main()
