"""Inspect ternary estimate integration tests — header-only."""

from __future__ import annotations

import json
import pathlib
import shutil
import struct
import uuid


def _write_snapshot(tmp: pathlib.Path, tensors: dict[str, tuple[str, list[int]]]) -> pathlib.Path:
    snapshot = tmp / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
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


def test_inspect_has_ternary_estimator() -> None:
    from openternary.config.schema import AppConfig
    from openternary.inspect.engine import run_inspection

    tmp = pathlib.Path.cwd() / f"test_tern_est_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _write_snapshot(
            tmp,
            {
                "model.layers.0.self_attn.q_proj.weight": ("BF16", [1536, 1536]),
                "model.layers.0.mlp.gate_proj.weight": ("BF16", [6144, 1536]),
                "model.embed_tokens.weight": ("BF16", [262144, 1536]),
            },
        )
        cfg = AppConfig(model={"id": "google/gemma-4-E2B-it-qat-q4_0-unquantized", "revision": "test-rev"})
        result = run_inspection(cfg, snapshot_path=snapshot, load_weights=False)
        assert "ternary_estimator" in result
        est = result["ternary_estimator"]
        assert est["version"] == "1"
        assert est["scheme"] == "absmean-per-tensor"
        assert est["packing"] == "2bit-v1"
        assert est["scale_dtype"] == "fp32"
        assert "estimate" in est
        assert "fingerprint" in est
        assert est["fingerprint"].startswith("sha256:")
        # memory_estimate.ternary should be alias to estimate
        assert result["memory_estimate"]["ternary"] == est["estimate"]
        # ideal vs packed separation
        estimate = est["estimate"]
        assert "ideal_ternary_bits" in estimate
        assert "packed_weight_bits" in estimate
        assert "byte_padding_bits" in estimate
        assert estimate["actual_file_size_bytes"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fingerprint_payload_unchanged() -> None:
    from openternary.config.schema import AppConfig
    from openternary.inspect.engine import run_inspection  # noqa: F401 — keep import minimal

    tmp = pathlib.Path.cwd() / f"test_fp2_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _write_snapshot(tmp, {"model.layers.0.self_attn.q_proj.weight": ("BF16", [1536, 1536])})
        cfg = AppConfig(model={"id": "google/gemma-4-E2B-it-qat-q4_0-unquantized", "revision": "rev1"})
        result = run_inspection(cfg, snapshot_path=snapshot)
        # recompute fingerprint from payload builder should match inspection_fingerprint
        # Recreate what engine does internally: fingerprint_payload has ternary None
        # We verify that inspection_fingerprint is stable and not affected by ternary estimate
        r2 = run_inspection(cfg, snapshot_path=snapshot)
        assert result["inspection_fingerprint"] == r2["inspection_fingerprint"]
        # ternary fingerprint should also be deterministic
        assert result["ternary_estimator"]["fingerprint"] == r2["ternary_estimator"]["fingerprint"]
        # inspection fingerprint should NOT equal ternary fingerprint
        assert result["inspection_fingerprint"] != result["ternary_estimator"]["fingerprint"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_per_tensor_packed_sum() -> None:
    from openternary.config.schema import AppConfig
    from openternary.inspect.engine import run_inspection

    tmp = pathlib.Path.cwd() / f"test_packed_sum_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        # 2 quantizable tensors each 1 weight → packed 8+8=16, not 8
        snapshot = _write_snapshot(
            tmp,
            {
                "model.layers.0.self_attn.q_proj.weight": ("BF16", [1]),
                "model.layers.0.self_attn.k_proj.weight": ("BF16", [1]),
                "model.embed_tokens.weight": ("BF16", [10]),
            },
        )
        cfg = AppConfig(model={"id": "google/gemma-4-E2B-it-qat-q4_0-unquantized", "revision": "rev1"})
        result = run_inspection(cfg, snapshot_path=snapshot)
        est = result["ternary_estimator"]["estimate"]
        assert est["packed_weight_bits"] == 16
        assert est["logical_2bit_bits"] == 4
        assert est["byte_padding_bits"] == 12
        # excluded should include embed_tokens
        assert est["excluded_original_bits"] == 10 * 16
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
