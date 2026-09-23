"""CPU tensor fixtures only: atomic publication, packed round-trip and pass semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openternary.cli.main import app
from openternary.config.loader import load_config
from openternary.services.artifacts import atomic_artifact, export_artifact, validate_artifact, write_manifest


def test_atomic_failure_and_existing_destination_preserve_data(tmp_path):
    destination = tmp_path / "artifact"
    with pytest.raises(RuntimeError), atomic_artifact(destination) as staged:
        (staged / "partial.txt").write_text("unfinished")
        raise RuntimeError("injected disk failure")
    assert not destination.exists()
    assert not list(tmp_path.glob(".artifact-*/"))
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("owned by user")
    with pytest.raises(FileExistsError), atomic_artifact(destination):
        pytest.fail("existing artifact was writable")
    assert sentinel.read_text() == "owned by user"


def test_manifest_rejects_changed_missing_and_extra_files(tmp_path):
    (tmp_path / "data.txt").write_text("data")
    write_manifest(tmp_path, format="safetensors", provenance={})
    assert validate_artifact(tmp_path)["quality_acceptance"] == "not_run"
    (tmp_path / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        validate_artifact(tmp_path)
    (tmp_path / "extra.txt").unlink()
    (tmp_path / "data.txt").write_text("edit")
    with pytest.raises(ValueError, match="inventory"):
        validate_artifact(tmp_path)


def tiny_source(path: Path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(-7, 8, dtype=torch.float32).reshape(3, 5),
        "model.layers.0.mlp.down_proj.weight": torch.arange(-7, 8, dtype=torch.bfloat16).reshape(3, 5),
        "model.embed_tokens.weight": torch.arange(10, dtype=torch.float32).reshape(2, 5),
    }
    # Multiple input shards exercise the old first-shard-only bug.
    for index, (name, tensor) in enumerate(tensors.items()):
        safetensors.save_file({name: tensor}, str(path / f"source-{index}.safetensors"))
    return tensors


@pytest.mark.parametrize("granularity", ["per_tensor", "per_group"])
def test_conversion_export_packed_roundtrip(tmp_path, granularity):
    torch = pytest.importorskip("torch")
    from openternary.services.execution import convert_artifact
    from openternary.services.snapshot import SnapshotReader

    source = tmp_path / "source"
    originals = tiny_source(source)
    cfg = load_config(
        cli_overrides={
            "model.id": str(source),
            "quantization.scale_granularity": granularity,
            "quantization.group_size": 2,
        }
    )
    artifact = tmp_path / "converted"
    convert_artifact(source, artifact, cfg)
    manifest = validate_artifact(artifact)
    assert manifest["model_reload"] == "not_run"
    assert manifest["provenance"]["adapter"] == "transformers"
    packed = tmp_path / "packed"
    export_artifact(artifact, packed, "ternary-packed")
    packed_manifest = validate_artifact(packed)
    assert packed_manifest["provenance"]["adapter"] == manifest["provenance"]["adapter"]
    assert packed_manifest["provenance"]["config"] == manifest["provenance"]["config"]
    assert packed_manifest["runtime"]["requires_unpack"]
    assert packed_manifest["runtime"]["native_low_bit"] is False
    restored = tmp_path / "restored"
    export_artifact(packed, restored, "safetensors")
    assert validate_artifact(restored)["provenance"]["backend"] == manifest["provenance"]["backend"]
    with SnapshotReader(artifact) as before, SnapshotReader(restored) as after:
        assert before.keys() == after.keys() == sorted(originals)
        for name in sorted(originals):
            a, b = before.get_tensor(name), after.get_tensor(name)
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        assert torch.equal(before.get_tensor("model.embed_tokens.weight"), originals["model.embed_tokens.weight"])
    invoked = CliRunner().invoke(app, ["artifacts", "validate", str(restored), "--json"])
    assert invoked.exit_code == 0, invoked.output
    assert json.loads(invoked.stdout)["artifact_validation"] == "file_integrity"


def test_packed_rejects_nonternary_tensor():
    torch = pytest.importorskip("torch")
    from openternary.export.packed import encode_tensor

    with pytest.raises(ValueError, match="exactly ternary"):
        encode_tensor(torch.tensor([[1.0, 2.0]]), 2, "per_group")


def test_cli_quantize_and_export_report_file_integrity(tmp_path):
    source = tmp_path / "source"
    tiny_source(source)
    run = tmp_path / "run"
    result = CliRunner().invoke(app, ["quantize", str(source), "--device", "cpu", "--output", str(run), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["execution"] == "completed"
    assert payload["artifact_validation"] == "file_integrity"
    assert payload["quality_acceptance"] == "not_run"
    packed = tmp_path / "packed"
    result = CliRunner().invoke(
        app, ["export", str(run), "--format", "ternary-packed", "--output", str(packed), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["artifact_validation"] == "file_integrity"
    assert validate_artifact(packed)["model_reload"] == "not_run"


def test_noop_content_preserved_clip_and_mixed_precision_applied(tmp_path):
    torch = pytest.importorskip("torch")
    from openternary.services.execution import convert_artifact
    from openternary.services.snapshot import SnapshotReader

    source = tmp_path / "source"
    originals = tiny_source(source)
    cfg = load_config(cli_overrides={"model.id": str(source)})
    reference = convert_artifact(source, tmp_path / "plain", cfg)
    noop = load_config(cli_overrides={"model.id": str(source), "quantization.passes": [{"name": "noop"}]})
    converted = convert_artifact(source, tmp_path / "noop", noop)
    assert reference.content_fingerprint == converted.content_fingerprint
    preserved = "model.layers.0.self_attn.q_proj.weight"
    clipped = load_config(
        cli_overrides={
            "model.id": str(source),
            "quantization.passes": [{"name": "clip", "options": {"max_abs": 1.0}}],
            "quantization.mixed_precision": {preserved: "preserve"},
        }
    )
    convert_artifact(source, tmp_path / "clipped", clipped)
    with SnapshotReader(tmp_path / "clipped") as result:
        assert torch.equal(result.get_tensor(preserved), originals[preserved])
        assert result.get_tensor("model.layers.0.mlp.down_proj.weight").abs().max() <= 1
    with pytest.raises(ValueError, match="already applied"):
        convert_artifact(tmp_path / "clipped", tmp_path / "twice", clipped)
    assert not (tmp_path / "twice").exists()


def test_diffusers_component_planning(tmp_path):
    from openternary.services.planning import build_plan

    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "FluxPipeline",
                "transformer": ["diffusers", "FluxTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "tokenizer": ["transformers", "T5Tokenizer"],
            }
        )
    )
    cfg = load_config(cli_overrides={"model.id": str(pipeline)})
    plan = build_plan(cfg)
    assert {item["role"] for item in plan["components"]} == {"denoiser", "vae", "auxiliary"}
    assert plan["target_count"] is None
    assert plan["execution"] == "not_run"
