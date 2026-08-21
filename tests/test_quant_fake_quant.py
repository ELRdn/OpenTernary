"""Fake-quant bounded sharded conversion tests — mock snapshots, no 10GB."""

from __future__ import annotations

import json
import pathlib
import shutil
import uuid

import pytest

torch = pytest.importorskip("torch")
import safetensors.torch  # noqa: E402

from openternary.config.loader import load_config  # noqa: E402
from openternary.quant.fake_quant import convert_snapshot  # noqa: E402


def _make_small_snapshot(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    # config.json minimal Gemma-like
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "transformers_version": "5.6.2",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 1000,
        },
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text(json.dumps({"version": "1"}), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({}), encoding="utf-8")
    # tensors: 2 quantizable (mlp.gate_proj, self_attn.q_proj), 2 excluded (embed, norm)
    tensors = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.randn(8, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(tensors, str(root / "model.safetensors"))
    return root


def test_convert_snapshot_per_tensor_bounded_sharded() -> None:
    tmp = pathlib.Path.cwd() / f"test_fakequant_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        src = _make_small_snapshot(tmp / "src")
        cfg = load_config(cli_overrides={"model.id": str(src), "quantization.scale_granularity": "per_tensor"})
        dst = tmp / "dst_per_tensor"
        # Use small shard size to force sharding (2 tensors ~ small, but set 1KB to force multiple shards)
        report = convert_snapshot(src, dst, cfg, max_shard_size=1024)
        assert dst.exists()
        assert report.shard_count >= 1
        # Check sharded files exist
        shards = list(dst.glob("model-*.safetensors"))
        assert len(shards) == report.shard_count
        # index exists
        assert (dst / "model.safetensors.index.json").exists()
        idx = json.loads((dst / "model.safetensors.index.json").read_text(encoding="utf-8"))
        assert "weight_map" in idx
        assert len(idx["weight_map"]) == 4
        # quantization.json exists and has summary, no full scales array
        qj = json.loads((dst / "quantization.json").read_text(encoding="utf-8"))
        assert qj["scale_granularity"] == "per_tensor"
        assert qj["num_quantizable_tensors"] == 2
        assert "content_fingerprint" in qj
        assert qj["fake_quant_artifact"]["shard_count"] == report.shard_count
        # per_tensor entries have scale_fingerprint but no scales array
        for entry in qj["per_tensor"]:
            if entry["quantizable"]:
                assert "scale_fingerprint" in entry
                assert "scales" not in entry
                assert "scale_stats" in entry
                # std should not be NaN
                assert entry["scale_stats"]["std"] == entry["scale_stats"]["std"]  # NaN check
        # excluded preserved exactly
        from safetensors import safe_open

        # Load dst tensors via index (for test, just open each shard and check embed preserved)
        src_tensors = {}
        with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f:
            for k in f.keys():  # noqa: SIM118
                src_tensors[k] = f.get_tensor(k)
        # collect dst tensors
        dst_tensors = {}
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for k in f.keys():  # noqa: SIM118
                    dst_tensors[k] = f.get_tensor(k)
        # excluded should be exactly equal
        assert torch.equal(src_tensors["model.embed_tokens.weight"], dst_tensors["model.embed_tokens.weight"])
        assert torch.equal(
            src_tensors["model.layers.0.input_layernorm.weight"], dst_tensors["model.layers.0.input_layernorm.weight"]
        )
        # quantizable should differ but be finite and no NaN, and changed (content fingerprint differs)
        assert not torch.equal(
            src_tensors["model.layers.0.mlp.gate_proj.weight"], dst_tensors["model.layers.0.mlp.gate_proj.weight"]
        )
        assert torch.isfinite(dst_tensors["model.layers.0.mlp.gate_proj.weight"]).all()

        # determinism: second conversion with same config should give same content fingerprint
        dst2 = tmp / "dst_per_tensor2"
        report2 = convert_snapshot(src, dst2, cfg, max_shard_size=1024)
        assert report2.content_fingerprint == report.content_fingerprint
        # file hashes should be same (SHOULD, not MUST, but with deterministic ordering they should match)
        assert report2.file_hashes == report.file_hashes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_convert_snapshot_per_group_rowwise() -> None:
    tmp = pathlib.Path.cwd() / f"test_fakequant_g_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        src = _make_small_snapshot(tmp / "src")
        cfg = load_config(
            cli_overrides={
                "model.id": str(src),
                "quantization.scale_granularity": "per_group",
                "quantization.group_size": 2,
                "quantization.grouping_scheme": "last-dim-rowwise-v1",
            }
        )
        dst = tmp / "dst_group"
        convert_snapshot(src, dst, cfg, max_shard_size=1024 * 1024)
        qj = json.loads((dst / "quantization.json").read_text(encoding="utf-8"))
        assert qj["scale_granularity"] == "per_group"
        assert qj["group_size"] == 2
        assert qj["grouping_scheme"] == "last-dim-rowwise-v1"
        # Check num_groups: each quantizable [8,4] with G=2 last-dim row-wise => per row 2 groups => 8*2=16 per tensor => total 32
        # Our small snapshot has 2 quantizable tensors
        assert qj["total_groups"] == 32
        # grouped overhead should be total_groups *32
        est = qj["packed_estimate"]
        assert est["grouped_scale_overhead_bits"] == 32 * 32
        assert est["scale_overhead_bits"] == 2 * 32  # per_tensor overhead
        # per_tensor entries should have num_groups =16 each for quantizable
        for e in qj["per_tensor"]:
            if e["quantizable"]:
                assert e["num_groups"] == 16
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_convert_snapshot_shard_flush_before_add() -> None:
    tmp = pathlib.Path.cwd() / f"test_shard_flush_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        src = tmp / "src"
        src.mkdir(parents=True, exist_ok=True)
        (src / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
        # Create 3 tensors each ~ 1KB (small) to test flush policy
        tensors = {
            "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),  # ~64 bytes
            "model.layers.0.mlp.up_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            "model.layers.0.mlp.down_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        }
        safetensors.torch.save_file(tensors, str(src / "model.safetensors"))
        cfg = load_config(cli_overrides={"model.id": str(src)})
        dst = tmp / "dst"
        # Set max_shard_size slightly larger than one tensor but smaller than 2 tensors to ensure flush before add
        # Each tensor ~64 bytes + overhead, so 100 bytes threshold will force 3 shards
        report = convert_snapshot(src, dst, cfg, max_shard_size=100)
        assert report.shard_count == 3
        shards = list(dst.glob("model-*.safetensors"))
        assert len(shards) == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_convert_snapshot_non_weight_files_and_symlink() -> None:
    tmp = pathlib.Path.cwd() / f"test_nw_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        src = tmp / "src"
        src.mkdir(parents=True, exist_ok=True)
        (src / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
        (src / "tokenizer.json").write_text(json.dumps({"test": 1}), encoding="utf-8")
        (src / "special_tokens_map.json").write_text(json.dumps({}), encoding="utf-8")
        tensors = {"model.layers.0.mlp.gate_proj.weight": torch.randn(4, 4, dtype=torch.bfloat16)}
        safetensors.torch.save_file(tensors, str(src / "model.safetensors"))
        # Create a symlink inside snapshot pointing to a file inside snapshot (should be copied)
        # and a symlink to blobs-like location (simulate HF cache)
        # For test, create blobs dir sibling
        blobs = tmp / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        blob_file = blobs / "abc123"
        blob_file.write_text("blob content", encoding="utf-8")
        # symlink inside src pointing to blobs
        try:
            (src / "blob_link").symlink_to(blob_file)
            has_symlink = True
        except Exception:
            has_symlink = False

        cfg = load_config(cli_overrides={"model.id": str(src)})
        dst = tmp / "dst"
        convert_snapshot(src, dst, cfg, max_shard_size=1024 * 1024)
        assert (dst / "config.json").exists()
        assert (dst / "tokenizer.json").exists()
        assert (dst / "special_tokens_map.json").exists()
        if has_symlink:
            # dereferenced copy should exist and contain blob content
            assert (dst / "blob_link").exists()
            assert not (dst / "blob_link").is_symlink()
            assert (dst / "blob_link").read_text(encoding="utf-8") == "blob content"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
