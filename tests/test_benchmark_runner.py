"""Benchmark runner tests — mocked torch/transformers, no 10GB needed."""

from __future__ import annotations

import json
import pathlib
import shutil
import struct
import sys
import types
import uuid


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
    header = {
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]},
    }
    header_json = json.dumps(header).encode("utf-8")
    path = snapshot / "model.safetensors"
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(b"\x00" * 32)
    return snapshot


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def to(self, device: object) -> _FakeTensor:  # type: ignore[no-untyped-def]
        return self

    def tolist(self) -> list[int]:
        # deterministic token ids per call
        return [10, 20, 30]


class _FakeInputs(dict):  # type: ignore[type-arg]
    pass


class _FakeProcessor:
    def apply_chat_template(self, messages: object, **kwargs: object) -> dict[str, object]:  # type: ignore[no-untyped-def]
        # return dict with input_ids tensor-like
        return {"input_ids": _FakeTensor((1, 3)), "attention_mask": _FakeTensor((1, 3))}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:  # type: ignore[no-untyped-def]
        return "decoded: " + ",".join(str(x) for x in ids)

    def parse_response(self, text: str) -> str:
        return text.replace("decoded: ", "parsed: ")


class _FakeModel:
    def __init__(self) -> None:
        import types as _types

        # mimic param for dtype check
        self._dtype = _types.SimpleNamespace()

    def parameters(self):  # type: ignore[no-untyped-def]
        import torch  # type: ignore[import-not-found]

        class P:
            dtype = torch.bfloat16
            device = "cpu"

        yield P()

    def eval(self) -> None:
        pass

    def generate(self, **kwargs: object) -> list[_FakeTensor]:  # type: ignore[no-untyped-def]
        # return [seq] where seq length = input_len(3) + 3 generated
        return [_FakeTensor((1, 6))]


def _install_fake_torch_transformers(monkey_present: bool = True) -> None:  # type: ignore[no-untyped-def]
    # torch fake
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    fake_torch.float16 = "fp16"  # type: ignore[attr-defined]
    fake_torch.float32 = "fp32"  # type: ignore[attr-defined]
    fake_torch.manual_seed = lambda seed: None  # type: ignore[attr-defined]
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, synchronize=lambda: None, manual_seed_all=lambda seed: None
    )  # type: ignore[attr-defined]

    # inference_mode context
    class _InfCtx:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *a: object) -> None:
            return None

    fake_torch.inference_mode = lambda: _InfCtx()  # type: ignore[attr-defined]
    sys.modules["torch"] = fake_torch

    # transformers fake
    fake_transformers = types.ModuleType("transformers")

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


def test_run_benchmark_mocked_end_to_end() -> None:
    from openternary.benchmark.metrics import to_metrics
    from openternary.benchmark.runner import run_benchmark
    from openternary.config.schema import AppConfig

    tmp = pathlib.Path.cwd() / f"test_bench_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    # stash real modules
    orig_torch = sys.modules.get("torch")
    orig_tr = sys.modules.get("transformers")
    try:
        _install_fake_torch_transformers()
        snapshot = _write_fake_snapshot(tmp)
        cfg = AppConfig(
            model={"id": str(snapshot), "revision": None},
            benchmark={
                "suite": "smoke",
                "thinking": False,
                "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 16},
            },
            dtype="bf16",
            device="cpu",
            seed=42,
        )
        result = run_benchmark(cfg, snapshot_path=snapshot, suite="smoke")
        # suite
        assert result["suite"]["name"] == "smoke"
        assert result["suite"]["version"] == "1"
        assert result["suite"]["fingerprint"].startswith("sha256:")
        assert result["result_fingerprint"].startswith("sha256:")
        # thinking/generation
        assert result["thinking"] is False
        assert result["generation"] == {"do_sample": False, "num_beams": 1, "max_new_tokens": 16}
        # results
        assert len(result["results"]) == 5
        for r in result["results"]:
            assert "generated_token_ids" in r
            assert "raw_output_text" in r
            assert "parsed_output_text" in r
            assert r["parsed_output_text"].startswith("parsed:")
            assert "latency_ms" in r
        # warmup
        assert "warmup" in result
        assert "model_load_ms" in result
        # summary
        assert result["summary"]["num_prompts"] == 5
        assert result["summary"]["output_tokens_per_second"] >= 0

        # metrics
        m = to_metrics(result, run_id="abc123")
        assert m["status"] == "completed"
        assert m["suite"] == "smoke"
        assert m["protocol_fingerprint"] == result["suite"]["fingerprint"]
        assert m["result_fingerprint"] == result["result_fingerprint"]
    finally:
        # restore
        if orig_torch is not None:
            sys.modules["torch"] = orig_torch
        else:
            sys.modules.pop("torch", None)
        if orig_tr is not None:
            sys.modules["transformers"] = orig_tr
        else:
            sys.modules.pop("transformers", None)
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_benchmark_rejects_non_smoke_suite() -> None:
    from openternary.benchmark.runner import run_benchmark
    from openternary.config.schema import AppConfig

    tmp = pathlib.Path.cwd() / f"test_bench_rej_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = _write_fake_snapshot(tmp)
        cfg = AppConfig(model={"id": str(snapshot), "revision": None})
        try:
            run_benchmark(cfg, snapshot_path=snapshot, suite="japanese")
            raise AssertionError("should have raised")
        except ValueError as e:
            assert "smoke" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
