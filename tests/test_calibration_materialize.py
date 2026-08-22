"""Calibrated snapshot materialize integration — bounded sharded, HF loadable, fingerprint diff."""

from __future__ import annotations

import json
import pathlib
import shutil
import uuid

import pytest

torch = pytest.importorskip("torch")
import safetensors.torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from openternary.config.loader import load_config  # noqa: E402
from openternary.quant.fake_quant import materialize_calibrated_snapshot  # noqa: E402
from openternary.quant.grouping import quantize_groupwise  # noqa: E402
from openternary.quant.ternary import quantize_absmean  # noqa: E402


def _make_small_snapshot(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
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
        "hidden_size": 8,
        "torch_dtype": "bfloat16",
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text(json.dumps({"version": "1"}), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({}), encoding="utf-8")
    tensors = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.randn(8, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(tensors, str(root / "model.safetensors"))
    return root


def _tensor_sha(t: torch.Tensor) -> str:
    import hashlib

    cpu = t.detach().cpu().contiguous()
    try:
        b = cpu.view(torch.uint8).numpy().tobytes()
    except Exception:
        b = cpu.numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def test_materialize_per_tensor_hf_loadable_and_fingerprint_diff() -> None:
    tmp = pathlib.Path.cwd() / f"test_mat_pt_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        src = _make_small_snapshot(tmp / "src")
        cfg = load_config(cli_overrides={"model.id": str(src), "quantization.scale_granularity": "per_tensor"})
        # Build calibrated_state: quantize then perturb scales
        calibrated_state: dict[str, dict] = {}
        with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118
                if "gate_proj" in name or "q_proj" in name:
                    w = f.get_tensor(name)
                    tt = quantize_absmean(w.to(torch.float32))
                    # perturb scale by 5% to ensure fingerprint diff vs naive
                    perturbed_scale = torch.tensor([tt.scale * 1.05], dtype=torch.float32)
                    calibrated_state[name] = {
                        "codes": tt.codes,
                        "scales": perturbed_scale,
                        "shape": tuple(w.shape),
                        "orig_dtype": str(w.dtype),
                        "group_size": 128,
                        "grouping_scheme": "last-dim-rowwise-v1",
                    }
        dst = tmp / "dst_calib_pt"
        report = materialize_calibrated_snapshot(src, dst, cfg, calibrated_state, max_shard_size=1024)
        # placeholder banned: no README.txt alone, must have sharded snapshot
        assert not (dst / "README.txt").exists() or (dst / "model.safetensors.index.json").exists()
        assert (dst / "model.safetensors.index.json").exists()
        shards = list(dst.glob("model-*.safetensors"))
        assert len(shards) == report.shard_count
        assert (dst / "config.json").exists()
        assert (dst / "tokenizer.json").exists()
        # HF loadable via AutoConfig
        from transformers import AutoConfig  # type: ignore

        cfg_loaded = AutoConfig.from_pretrained(str(dst), trust_remote_code=False)
        assert cfg_loaded is not None
        # fingerprint diff for calibrated target
        with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f_src:
            src_t = f_src.get_tensor("model.layers.0.mlp.gate_proj.weight")
        # collect calibrated tensor
        dst_tensors: dict[str, torch.Tensor] = {}
        for sh in shards:
            with safe_open(str(sh), framework="pt", device="cpu") as f:
                for k in f.keys():  # noqa: SIM118
                    dst_tensors[k] = f.get_tensor(k)
        assert "model.layers.0.mlp.gate_proj.weight" in dst_tensors
        assert _tensor_sha(src_t) != _tensor_sha(dst_tensors["model.layers.0.mlp.gate_proj.weight"])
        # excluded preserved exactly
        with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f_src:
            src_emb = f_src.get_tensor("model.embed_tokens.weight")
            src_norm = f_src.get_tensor("model.layers.0.input_layernorm.weight")
        assert torch.equal(src_emb, dst_tensors["model.embed_tokens.weight"])
        assert torch.equal(src_norm, dst_tensors["model.layers.0.input_layernorm.weight"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_materialize_per_group_rowwise() -> None:
    tmp = pathlib.Path.cwd() / f"test_mat_g_{uuid.uuid4().hex[:6]}"
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
        calibrated_state: dict[str, dict] = {}
        with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118
                if "gate_proj" in name or "q_proj" in name:
                    w = f.get_tensor(name)
                    res = quantize_groupwise(w.to(torch.float32), 2)
                    perturbed = res.scales * 1.07
                    calibrated_state[name] = {
                        "codes": res.codes,
                        "scales": perturbed,
                        "shape": tuple(w.shape),
                        "orig_dtype": str(w.dtype),
                        "group_size": 2,
                        "grouping_scheme": "last-dim-rowwise-v1",
                    }
        dst = tmp / "dst_g"
        report = materialize_calibrated_snapshot(src, dst, cfg, calibrated_state, max_shard_size=1024 * 1024)
        assert report.shard_count >= 1
        qj = json.loads((dst / "quantization.json").read_text(encoding="utf-8"))
        assert qj["scale_granularity"] == "per_group"
        # total_groups should be 32 for two [8,4] with G=2
        assert qj["total_groups"] == 32
        # fingerprint diff
        shards = list(dst.glob("model-*.safetensors"))
        dst_tensors: dict[str, torch.Tensor] = {}
        for sh in shards:
            with safe_open(str(sh), framework="pt", device="cpu") as f:
                for k in f.keys():  # noqa: SIM118
                    dst_tensors[k] = f.get_tensor(k)
        with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f_src:
            src_t = f_src.get_tensor("model.layers.0.mlp.gate_proj.weight")
        assert _tensor_sha(src_t) != _tensor_sha(dst_tensors["model.layers.0.mlp.gate_proj.weight"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_materialize_no_whole_shard_copy() -> None:
    """Verify non-target tensors are copied but original shard file is not reused (re-serialized)."""
    tmp = pathlib.Path.cwd() / f"test_mat_nocopy_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        src = _make_small_snapshot(tmp / "src")
        cfg = load_config(cli_overrides={"model.id": str(src), "quantization.scale_granularity": "per_tensor"})
        calibrated_state: dict[str, dict] = {}
        with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f:
            w = f.get_tensor("model.layers.0.mlp.gate_proj.weight")
            tt = quantize_absmean(w.to(torch.float32))
            calibrated_state["model.layers.0.mlp.gate_proj.weight"] = {
                "codes": tt.codes,
                "scales": torch.tensor([tt.scale * 1.1], dtype=torch.float32),
                "shape": tuple(w.shape),
                "orig_dtype": str(w.dtype),
                "group_size": 128,
                "grouping_scheme": "last-dim-rowwise-v1",
            }
        dst = tmp / "dst_nocopy"
        materialize_calibrated_snapshot(src, dst, cfg, calibrated_state, max_shard_size=1024 * 1024)
        # shard file must be newly created, not a hard copy of src shard (different mtime/content)
        src_shard = src / "model.safetensors"
        dst_shards = list(dst.glob("model-*.safetensors"))
        assert dst_shards
        # at least one dst shard differs in bytes from src shard (since gate_proj changed)
        assert any(dst_sh.read_bytes() != src_shard.read_bytes() for dst_sh in dst_shards)
        # ensure index weight_map covers all 4 tensors
        idx = json.loads((dst / "model.safetensors.index.json").read_text(encoding="utf-8"))
        assert len(idx["weight_map"]) == 4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
