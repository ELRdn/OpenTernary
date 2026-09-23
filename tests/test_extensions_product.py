"""Extension, cache and optional integration contracts without real models."""

from __future__ import annotations

import json
import sys
import types

import pytest
from typer.testing import CliRunner

from openternary.cli.main import app
from openternary.services.artifacts import write_manifest
from openternary.services.plugins import discover_plugins, load_plugin


def test_plugins_are_lazy_and_reject_conflicts_and_versions(monkeypatch):
    loaded = []

    class Entry:
        name = "fixture"
        value = "fixture:Plugin"
        dist = None

        def load(self):
            loaded.append(self.name)
            return types.SimpleNamespace(api_version=99)

    entries = [Entry()]
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda **kw: entries if kw.get("group") == "openternary.evaluators.v1" else [],
    )
    assert discover_plugins()[0]["status"] == "discovered"
    assert not loaded
    with pytest.raises(ValueError, match="version mismatch"):
        load_plugin("evaluators", "fixture")
    entries.append(Entry())
    assert all(row["status"] == "conflict" for row in discover_plugins())
    with pytest.raises(ValueError, match="conflicting"):
        load_plugin("evaluators", "fixture")
    assert loaded == ["fixture"]


@pytest.mark.parametrize("kind,name", [("passes", "clip"), ("exporters", "gguf"), ("evaluators", "diffusion")])
def test_plugins_cannot_shadow_builtin_names(monkeypatch, kind, name):
    entry = types.SimpleNamespace(name=name, value="unused:plugin", dist=None)
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda **kw: [entry] if kw.get("group") == f"openternary.{kind}.v1" else [],
    )
    assert discover_plugins()[0]["status"] == "conflict"
    with pytest.raises(ValueError, match="conflicting"):
        load_plugin(kind, name)


def test_cache_reference_and_boundary_guards(tmp_path):
    from openternary.services.cache import remove_artifact

    artifact = tmp_path / "source"
    artifact.mkdir()
    (artifact / "data.txt").write_text("user artifact")
    write_manifest(artifact, format="fixture", provenance={})
    report = remove_artifact(tmp_path, artifact)
    assert report["status"] == "planned" and artifact.exists()
    with pytest.raises(ValueError, match="strictly inside"):
        remove_artifact(tmp_path, tmp_path, execute=True)
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "data.txt").write_text("derived")
    write_manifest(derived, format="fixture", provenance={"source": str(artifact)})
    with pytest.raises(ValueError, match="referenced"):
        remove_artifact(tmp_path, artifact, execute=True)
    assert remove_artifact(tmp_path, derived, execute=True)["status"] == "removed"
    assert artifact.exists() and not derived.exists()


def test_diffusion_dry_run_never_imports_diffusers(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"name": "fixture", "prompts": ["test"], "seeds": [1]}))
    import builtins

    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name in {"diffusers", "torch", "transformers"}:
            raise AssertionError(f"unwanted model import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    result = CliRunner().invoke(
        app, ["evaluate", str(tmp_path / "absent"), "--profile", str(profile), "--dry-run", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["result"]["model_validation"] == "not_run"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["profile.json"]


def test_torchao_api_bridge_with_test_double(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from openternary.backends.torchao import TorchAOBackend
    from openternary.config.loader import load_config
    from openternary.services.artifacts import validate_artifact
    from openternary.services.execution import convert_artifact
    from tests.test_artifact_product import tiny_source

    source = tmp_path / "source"
    tiny_source(source)
    calls = []
    ao = types.ModuleType("torchao")
    ao.__path__ = []
    quantization = types.ModuleType("torchao.quantization")
    granularity = types.ModuleType("torchao.quantization.granularity")

    class Config:
        def __init__(self, **kwargs):
            self.options = kwargs

    def quantize(module, config):
        calls.append((tuple(module.weight.shape), config.options))

    quantization.Int8WeightOnlyConfig = Config
    quantization.quantize_ = quantize
    granularity.PerTensor = lambda: "per_tensor"
    granularity.PerGroup = lambda size: ("per_group", size)
    for name, module in [
        ("torchao", ao),
        ("torchao.quantization", quantization),
        ("torchao.quantization.granularity", granularity),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    original_version = __import__("importlib.metadata", fromlist=["version"]).version
    monkeypatch.setattr(
        "importlib.metadata.version", lambda name: "0.18.0" if name == "torchao" else original_version(name)
    )
    cfg = load_config(
        cli_overrides={
            "model.id": str(source),
            "quantization.backend": "torchao",
            "quantization.scheme": "int8-weight-only",
            "quantization.weight_dtype": "int8",
        }
    )
    output = tmp_path / "artifact"
    report = convert_artifact(source, output, cfg)
    assert len(calls) == 2
    assert all(call[1]["set_inductor_config"] is False for call in calls)
    assert report.quantization_json["native_kernel"] == "unverified"
    assert validate_artifact(output)["model_reload"] == "not_run"
    assert len(list(TorchAOBackend().load_weights(output))) == 3


def test_manifest_traversal_and_packed_reserved_codes(tmp_path):
    from openternary.services.artifacts import member_path

    for name in ["../other", "C:/other", "nested/../../other", "nested\\other", "/other"]:
        with pytest.raises(ValueError):
            member_path(tmp_path, name)
    torch = pytest.importorskip("torch")
    from openternary.export.packed import decode_tensor

    row = {"shape": [1, 1], "dtype": "float32", "granularity": "per_tensor", "group_size": 1}
    with pytest.raises(ValueError, match="reserved"):
        decode_tensor(torch.tensor([0b11000000], dtype=torch.uint8), torch.tensor([1.0]), row)


@pytest.mark.parametrize("invalid_header", [False, True])
def test_gguf_bridge_test_double_atomicity(tmp_path, monkeypatch, invalid_header):
    import struct
    import subprocess
    from pathlib import Path

    from openternary.export.gguf import export_gguf
    from openternary.services.artifacts import validate_artifact
    from tests.test_cli_product import header_fixture

    source = header_fixture(tmp_path / "source")
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("# Test double, never executed", encoding="utf-8")
    calls = []
    original_run = subprocess.run

    def fake_process(command, **kwargs):
        if "--outfile" not in command:
            return original_run(command, **kwargs)
        calls.append(command)
        output = Path(command[command.index("--outfile") + 1])
        output.write_bytes(b"bad" if invalid_header else struct.pack("<4sIQQ", b"GGUF", 3, 1, 0))

    monkeypatch.setattr("openternary.export.gguf.run_process", fake_process)
    output = tmp_path / "export"
    if invalid_header:
        with pytest.raises(ValueError, match="GGUF header"):
            export_gguf(source, output, converter)
        assert not output.exists()
    else:
        result = export_gguf(source, output, converter)
        manifest = validate_artifact(output)
        assert manifest["format"] == "gguf"
        assert manifest["model_reload"] == "not_run"
        assert "header and file hashes only" in manifest["provenance"]["validation_scope"]
        assert manifest == result["manifest"]
    assert len(calls) == 1
    assert not list(tmp_path.glob(".export-*/"))


def test_gguf_bridge_rejects_architecture_before_converter(tmp_path, monkeypatch):
    from openternary.export.gguf import export_gguf
    from tests.test_cli_product import header_fixture

    source = header_fixture(tmp_path / "source", "gemma4")
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("# Test double, never executed")
    monkeypatch.setattr("openternary.export.gguf.run_process", lambda *a, **k: pytest.fail("converter executed"))
    with pytest.raises(ValueError, match="metadata only"):
        export_gguf(source, tmp_path / "export", converter)
    assert not (tmp_path / "export").exists()
