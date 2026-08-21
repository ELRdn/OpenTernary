"""Inspect engine tests — header-only, no real 10GB needed."""

from __future__ import annotations

import json
import pathlib
import shutil
import struct
import uuid


def _write_snapshot(tmp: pathlib.Path, tensors: dict[str, tuple[str, list[int]]]) -> pathlib.Path:
    """Create a minimal snapshot dir with config.json + model.safetensors."""
    snapshot = tmp / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    # minimal gemma4 config
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "transformers_version": "5.6.2",
        "dtype": "bfloat16",
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "vocab_size": 262144,
        },
        "vision_config": {"hidden_size": 768, "num_hidden_layers": 16},
        "audio_config": {"hidden_size": 1024, "num_hidden_layers": 12},
    }
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    # safetensors
    header: dict[str, object] = {}
    offset = 0
    import math

    for name, (dtype, shape) in tensors.items():
        numel = math.prod(shape) if shape else 1
        nbytes = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1}.get(dtype, 2)
        size = numel * nbytes
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header_json = json.dumps(header).encode("utf-8")
    path = snapshot / "model.safetensors"
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(b"\x00" * offset)
    return snapshot


def test_run_inspection_header_only() -> None:
    from openternary.config.schema import AppConfig
    from openternary.inspect.engine import run_inspection

    tmp = pathlib.Path.cwd() / f"test_inspect_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _write_snapshot(
            tmp,
            {
                "model.layers.0.self_attn.q_proj.weight": ("BF16", [1536, 1536]),
                "model.layers.0.self_attn.o_proj.weight": ("BF16", [1536, 1536]),
                "model.layers.0.mlp.gate_proj.weight": ("BF16", [6144, 1536]),
                "model.embed_tokens.weight": ("BF16", [262144, 1536]),
                "model.norm.weight": ("BF16", [1536]),
            },
        )
        cfg = AppConfig(model={"id": "google/gemma-4-E2B-it-qat-q4_0-unquantized", "revision": "test-rev"})
        result = run_inspection(cfg, snapshot_path=snapshot, load_weights=False)
        assert result["inspection_fingerprint"].startswith("sha256:")
        assert result["summary"]["total_tensors"] == 5
        # quantizable: 3 (q_proj, o_proj, gate_proj)
        assert result["summary"]["quantizable_tensors"] == 3
        assert result["summary"]["total_params"] == sum(p for p in [2359296, 2359296, 9437184, 402653184, 1536])
        # Phase 2: ternary estimate is now structured dict (per-tensor packed sum), not None
        ternary = result["memory_estimate"]["ternary"]
        assert isinstance(ternary, dict)
        assert "packed_weight_bits" in ternary
        assert "ideal_ternary_bits" in ternary
        assert ternary["actual_file_size_bytes"] is None
        assert result["dtype_report"]["source"] == "safetensors_header"
        # tensors sorted
        names = [t["name"] for t in result["tensors"]]
        assert names == sorted(names)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fingerprint_stable_across_runs() -> None:
    from openternary.config.schema import AppConfig
    from openternary.inspect.engine import run_inspection

    tmp = pathlib.Path.cwd() / f"test_fp_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _write_snapshot(tmp, {"model.layers.0.self_attn.q_proj.weight": ("BF16", [1536, 1536])})
        cfg = AppConfig(model={"id": "google/gemma-4-E2B-it-qat-q4_0-unquantized", "revision": "rev1"})
        r1 = run_inspection(cfg, snapshot_path=snapshot)
        r2 = run_inspection(cfg, snapshot_path=snapshot)
        assert r1["inspection_fingerprint"] == r2["inspection_fingerprint"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_warning_layer_mismatch() -> None:
    from openternary.config.schema import AppConfig
    from openternary.inspect.engine import run_inspection

    tmp = pathlib.Path.cwd() / f"test_warn_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _write_snapshot(
            tmp,
            {
                "model.layers.0.self_attn.q_proj.weight": ("BF16", [1536, 1536]),
                "model.layers.5.self_attn.q_proj.weight": ("BF16", [1536, 1536]),
            },
        )
        cfg = AppConfig(model={"id": "google/gemma-4-E2B-it-qat-q4_0-unquantized", "revision": "rev1"})
        result = run_inspection(cfg, snapshot_path=snapshot)
        # text_config says 2 layers but we have layer 5 -> warning
        assert any("layer count mismatch" in w for w in result["warnings"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
