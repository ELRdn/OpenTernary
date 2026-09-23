"""CLI quality evaluation tests."""

import json
import sys
import types
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from openternary.cli.main import app

torch = pytest.importorskip("torch")

runner = CliRunner()


def test_cli_quality_help_exposes_frozen_data_and_split() -> None:
    result = runner.invoke(app, ["quality", "--help"])
    assert result.exit_code == 0
    assert "--data" in result.output
    assert "--split" in result.output
    assert "--max-length" in result.output
    assert "--stride" in result.output


def test_cli_quality_dry_run_validates_dataset_before_claiming_success(tmp_path) -> None:
    dataset = tmp_path / "malformed.json"
    dataset.write_text("{}", encoding="utf-8")
    output = tmp_path / "quality-run"

    result = runner.invoke(
        app,
        ["quality", "--data", str(dataset), "--dry-run", "--output", str(output)],
    )

    assert result.exit_code == 2
    assert "dataset validation failed" in result.output.lower()
    assert not output.exists()


def test_cli_quality_writes_self_contained_quality_artifact(tmp_path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type": "gemma4"}', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"fixture")
    dataset = tmp_path / "quality.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {"id": "fixture", "revision": "v1", "license": "CC0", "source": "fixture"},
                "splits": {
                    "calibration": [{"id": "c", "text": "calibration material only", "language": "en"}],
                    "validation": [
                        {"id": "ve", "text": "independent english validation", "language": "en"},
                        {"id": "vj", "text": "独立した日本語検証文章です", "language": "ja"},
                    ],
                    "test": [
                        {"id": "te", "text": "untouched english final", "language": "en"},
                        {"id": "tj", "text": "未使用の日本語最終文章です", "language": "ja"},
                    ],
                },
                "instruction_cases": {
                    "validation": [{"id": "iv", "prompt": "青", "expected": "青"}],
                    "test": [{"id": "it", "prompt": "赤", "expected": "赤"}],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class Processor:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[ord(character) % 4 for character in text]])}

        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, ids, skip_special_tokens=True):
            return "青"

    class UniformModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 4))

        def generate(self, input_ids, **kwargs):
            return torch.cat((input_ids, torch.tensor([[3]], device=input_ids.device)), dim=-1)

    fake_transformers = types.ModuleType("transformers")

    class AutoProcessor:
        from_pretrained = staticmethod(lambda path, **kwargs: Processor())

    class AutoModel:
        from_pretrained = staticmethod(lambda path, **kwargs: UniformModel())

    fake_transformers.AutoProcessor = AutoProcessor
    fake_transformers.AutoModelForMultimodalLM = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    output = tmp_path / "quality-run"
    result = runner.invoke(
        app,
        [
            "quality",
            str(snapshot),
            "--data",
            str(dataset),
            "--split",
            "validation",
            "--max-length",
            "8",
            "--stride",
            "4",
            "--dtype",
            "fp32",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    quality = json.loads((output / "quality.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert quality["summary"]["general_ppl"] == pytest.approx(4.0)
    assert quality["summary"]["japanese_ppl"] == pytest.approx(4.0)
    assert quality["summary"]["instruction_score"] == pytest.approx(100.0)
    assert quality["scientific_acceptance"] is False
    assert metrics["status"] == "completed"
