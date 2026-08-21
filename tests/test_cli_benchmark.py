"""CLI benchmark tests — smoke suite v1, mocked model where needed."""

from __future__ import annotations

import json
import pathlib
import shutil
import struct
import sys
import types
import uuid

from typer.testing import CliRunner

from openternary.cli.main import app

runner = CliRunner()


def _write_fake_snapshot(base: pathlib.Path) -> pathlib.Path:
    snapshot = base / "snapshot"
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
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "vocab_size": 262144,
        },
        "vision_config": {"hidden_size": 768, "num_hidden_layers": 16},
        "audio_config": {"hidden_size": 1024, "num_hidden_layers": 12},
    }
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    header = {"model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]}}
    header_json = json.dumps(header).encode("utf-8")
    path = snapshot / "model.safetensors"
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(b"\x00" * 32)
    return snapshot


def test_cli_benchmark_help_shows_suite_and_thinking() -> None:
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--suite" in result.output
    assert "--thinking" in result.output


def test_cli_benchmark_dry_run() -> None:
    tmp = pathlib.Path.cwd() / f"test_bench_cli_dry_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _write_fake_snapshot(tmp)
        result = runner.invoke(app, ["benchmark", str(snapshot), "--dry-run"])
        assert result.exit_code == 0
        assert "dry-run" in result.output.lower()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_benchmark_cache_miss_exit_2() -> None:
    result = runner.invoke(app, ["benchmark", "nonexistent/model-id-xyz-9999"])
    assert result.exit_code == 2
    assert "not found" in result.output.lower() or "snapshot" in result.output.lower()


def test_cli_benchmark_mocked_success() -> None:
    # install fake torch/transformers
    orig_torch = sys.modules.get("torch")
    orig_tr = sys.modules.get("transformers")
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    fake_torch.float16 = "fp16"  # type: ignore[attr-defined]
    fake_torch.float32 = "fp32"  # type: ignore[attr-defined]
    fake_torch.manual_seed = lambda seed: None  # type: ignore[attr-defined]
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, synchronize=lambda: None, manual_seed_all=lambda seed: None
    )  # type: ignore[attr-defined]

    class _InfCtx:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *a: object) -> None:
            return None

    fake_torch.inference_mode = lambda: _InfCtx()  # type: ignore[attr-defined]
    sys.modules["torch"] = fake_torch

    fake_transformers = types.ModuleType("transformers")

    class _FakeTensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def to(self, device: object) -> _FakeTensor:  # type: ignore[no-untyped-def]
            return self

        def tolist(self) -> list[int]:
            return [10, 20]

    class _FakeProcessor:
        def apply_chat_template(self, messages: object, **kwargs: object) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {"input_ids": _FakeTensor((1, 3))}

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:  # type: ignore[no-untyped-def]
            return "decoded"

        def parse_response(self, text: str) -> str:
            return "parsed"

    class _FakeModel:
        def parameters(self):  # type: ignore[no-untyped-def]
            class P:
                dtype = "bf16"
                device = "cpu"

            yield P()

        def eval(self) -> None:
            pass

        def generate(self, **kwargs: object) -> list[_FakeTensor]:  # type: ignore[no-untyped-def]
            return [_FakeTensor((1, 5))]

    class _FakeAP:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> _FakeProcessor:  # type: ignore[no-untyped-def]
            return _FakeProcessor()

    class _FakeAM:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> _FakeModel:  # type: ignore[no-untyped-def]
            return _FakeModel()

    fake_transformers.AutoProcessor = _FakeAP  # type: ignore[attr-defined]
    fake_transformers.AutoModelForMultimodalLM = _FakeAM  # type: ignore[attr-defined]
    sys.modules["transformers"] = fake_transformers

    tmp = pathlib.Path.cwd() / f"test_bench_cli_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _write_fake_snapshot(tmp)
        out = tmp / "out_bench"
        result = runner.invoke(app, ["benchmark", str(snapshot), "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "benchmark.json").exists()
        assert (out / "metrics.json").exists()
        data = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
        assert data["suite"]["name"] == "smoke"
        assert data["suite"]["version"] == "1"
        assert "result_fingerprint" in data
        assert len(data["results"]) == 5
        assert "generated_token_ids" in data["results"][0]
    finally:
        if orig_torch is not None:
            sys.modules["torch"] = orig_torch
        else:
            sys.modules.pop("torch", None)
        if orig_tr is not None:
            sys.modules["transformers"] = orig_tr
        else:
            sys.modules.pop("transformers", None)
        shutil.rmtree(tmp, ignore_errors=True)
