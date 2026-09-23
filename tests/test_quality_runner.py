"""Production quality-runner behavior at the public library seam."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_quality_runner_builds_comparable_summary_from_frozen_dataset(tmp_path) -> None:
    from openternary.benchmark.quality_runner import run_quality_benchmark
    from openternary.config.schema import AppConfig

    dataset = tmp_path / "quality-dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {
                    "id": "fixture/general-ja",
                    "revision": "sha256:fixture-v1",
                    "license": "CC0-1.0",
                    "source": "local-test-fixture",
                },
                "splits": {
                    "calibration": [{"id": "cal-1", "text": "training source document", "language": "en"}],
                    "validation": [
                        {"id": "val-en", "text": "english validation sentence", "language": "en"},
                        {"id": "val-ja", "text": "日本語の独立した検証文章です", "language": "ja"},
                    ],
                    "test": [
                        {"id": "test-en", "text": "final untouched english example", "language": "en"},
                        {"id": "test-ja", "text": "最後まで未使用の日本語文章です", "language": "ja"},
                    ],
                },
                "instruction_cases": {
                    "validation": [{"id": "inst-val", "prompt": "青と答えてください", "expected": "青"}],
                    "test": [{"id": "inst-test", "prompt": "赤と答えてください", "expected": "赤"}],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class UniformModel(torch.nn.Module):
        def forward(self, input_ids):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 4))

        def generate(self, input_ids, **kwargs):
            answer = torch.tensor([[3]], device=input_ids.device)
            return torch.cat((input_ids, answer), dim=-1)

    class Processor:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[ord(character) % 4 for character in text]])}

        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, ids, skip_special_tokens=True):
            return "青" if ids == [3] else ""

    report = run_quality_benchmark(
        AppConfig(dtype="fp32", device="cpu"),
        dataset,
        model=UniformModel(),
        processor=Processor(),
        split="validation",
        max_length=8,
        stride=4,
    )

    assert report["summary"] == {
        "general_ppl": pytest.approx(4.0),
        "japanese_ppl": pytest.approx(4.0),
        "instruction_score": pytest.approx(100.0),
        "collapse_count": 0,
    }
    assert report["report_schema_version"] == 2
    assert report["instructions"]["results"][0]["response_text"] == "青"
    assert report["instruction_data_audit"]["disjoint"] is True
    assert report["protocol"]["split"] == "validation"
    assert report["dataset"]["id"] == "fixture/general-ja"
    assert len(report["dataset_fingerprint"]) == 64
    assert report["scientific_acceptance"] is False


def test_quality_runner_loads_snapshot_with_requested_dtype(tmp_path, monkeypatch) -> None:
    from openternary.benchmark.quality_runner import run_quality_benchmark
    from openternary.config.schema import AppConfig

    dataset = tmp_path / "quality-dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {"id": "fixture", "revision": "v1", "license": "CC0", "source": "fixture"},
                "splits": {
                    "calibration": [{"id": "c", "text": "calibration-only material", "language": "en"}],
                    "validation": [
                        {"id": "ve", "text": "independent english validation", "language": "en"},
                        {"id": "vj", "text": "独立した日本語検証文章です", "language": "ja"},
                    ],
                    "test": [
                        {"id": "te", "text": "untouched english final test", "language": "en"},
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
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

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
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(snapshot)
            return Processor()

    class AutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(snapshot)
            assert kwargs["dtype"] is torch.float32
            return UniformModel()

    fake_transformers.AutoProcessor = AutoProcessor
    fake_transformers.AutoModelForMultimodalLM = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    report = run_quality_benchmark(
        AppConfig(dtype="fp32", device="cpu"),
        dataset,
        snapshot_path=snapshot,
        split="validation",
        max_length=8,
        stride=4,
    )
    assert report["summary"]["general_ppl"] == pytest.approx(4.0)
    assert report["model"]["snapshot_path"] == str(snapshot)
    assert report["model"]["actual_dtype"] == "torch.float32"


def test_quality_runner_rejects_instruction_overlap_before_model_use(tmp_path) -> None:
    from openternary.benchmark.quality_runner import run_quality_benchmark
    from openternary.config.schema import AppConfig

    dataset = tmp_path / "overlap.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {"id": "fixture", "revision": "v1", "license": "CC0", "source": "fixture"},
                "splits": {
                    "calibration": [{"id": "c", "text": "calibration document", "language": "en"}],
                    "validation": [
                        {"id": "ve", "text": "english validation document", "language": "en"},
                        {"id": "vj", "text": "日本語検証文章です", "language": "ja"},
                    ],
                    "test": [
                        {"id": "te", "text": "english final document", "language": "en"},
                        {"id": "tj", "text": "日本語最終文章です", "language": "ja"},
                    ],
                },
                "instruction_cases": {
                    "validation": [{"id": "iv", "prompt": "同じ指示です", "expected": "青"}],
                    "test": [{"id": "it", "prompt": "同じ指示です", "expected": "赤"}],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="instruction.*overlap"):
        run_quality_benchmark(
            AppConfig(dtype="fp32", device="cpu"),
            dataset,
            model=object(),
            processor=object(),
            split="validation",
            max_length=8,
            stride=4,
        )


def test_quality_runner_prefers_text_tokenizer_from_multimodal_processor(tmp_path) -> None:
    from openternary.benchmark.quality_runner import run_quality_benchmark
    from openternary.config.schema import AppConfig

    dataset = tmp_path / "quality.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {"id": "fixture", "revision": "v1", "license": "CC0", "source": "fixture"},
                "splits": {
                    "calibration": [{"id": "c", "text": "calibration material", "language": "en"}],
                    "validation": [
                        {"id": "ve", "text": "english validation material", "language": "en"},
                        {"id": "vj", "text": "日本語の検証文章です", "language": "ja"},
                    ],
                    "test": [
                        {"id": "te", "text": "english untouched material", "language": "en"},
                        {"id": "tj", "text": "日本語の最終文章です", "language": "ja"},
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

    class TextTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[ord(character) % 4 for character in text]])}

    class MultimodalProcessor:
        tokenizer = TextTokenizer()

        def __call__(self, text, **kwargs):
            raise AssertionError("outer multimodal processor must not tokenize PPL text")

        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, ids, skip_special_tokens=True):
            return "青"

    class UniformModel(torch.nn.Module):
        def forward(self, input_ids):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 4))

        def generate(self, input_ids, **kwargs):
            return torch.cat((input_ids, torch.tensor([[3]])), dim=-1)

    report = run_quality_benchmark(
        AppConfig(dtype="fp32", device="cpu"),
        dataset,
        model=UniformModel(),
        processor=MultimodalProcessor(),
        split="validation",
        max_length=8,
        stride=4,
    )
    assert report["summary"]["general_ppl"] == pytest.approx(4.0)
