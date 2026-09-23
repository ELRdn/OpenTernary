"""Offline, seeded real-library verification. Never loads a pretrained model.

Install the candidate wheel first and run from any cwd with --require-wheel.
Each case runs in a fresh process. All state is under --root; existing case
directories are rejected. Optional dependencies are never installed by this script.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.update(
    HF_HUB_OFFLINE="1",
    HF_DATASETS_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def fixture(root):
    import sentencepiece as spm
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

    torch.manual_seed(42)
    root.mkdir(parents=True)
    corpus = root / "fixture-corpus.txt"
    corpus.write_text(
        "hello artificial fixture world seed forty two\n日本語の人工検証文章です青赤\n" * 20, encoding="utf-8"
    )
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(root / "tokenizer"),
        vocab_size=256,
        model_type="char",
        hard_vocab_limit=False,
        pad_id=3,
        shuffle_input_sentence=False,
        num_threads=1,
    )
    sp = spm.SentencePieceProcessor(model_file=str(root / "tokenizer.model"))
    tokenizer = LlamaTokenizer(
        vocab={sp.id_to_piece(i): i for i in range(sp.vocab_size())}, merges=[], pad_token="<pad>", model_max_length=512
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.save_pretrained(root)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).eval().save_pretrained(root, safe_serialization=True)
    write(root / "openternary-fixture.json", {"seed": 42, "pretrained": False, "family": "llama"})
    return root


def configuration(source, backend="ternary"):
    from openternary.config.schema import AppConfig

    q = {} if backend == "ternary" else {"backend": "torchao", "scheme": "int8-weight-only", "weight_dtype": "int8"}
    return AppConfig.model_validate(
        {
            "model": {"id": str(source), "revision": None, "adapter": "transformers"},
            "device": "cpu",
            "dtype": "fp32",
            "quantization": q,
            "benchmark": {"generation": {"max_new_tokens": 8}, "warmup": False},
        }
    )


def forward(source, config, device, dtype):
    import torch

    from openternary.adapters.runtime import load_model, load_processor

    marker = json.loads((source / "openternary-fixture.json").read_text(encoding="utf-8"))
    assert marker["pretrained"] is False
    from openternary.services.resources import ResourceMeter

    meter = ResourceMeter()
    meter.begin("load", device)
    try:
        model = load_model(config, source, getattr(torch, dtype), {"": device})
        tokenizer = load_processor(config, source)
        tokens = tokenizer("hello world", return_tensors="pt").to(device)
        tokens = {k: v for k, v in tokens.items() if k != "token_type_ids"}
        meter.begin("inference", device)
        with torch.inference_mode():
            logits = model(**tokens).logits.float().cpu()
            generated = model.generate(**tokens, max_new_tokens=8, do_sample=False)
        assert torch.isfinite(logits).all()
        assert str(next(model.parameters()).device) == device
        assert next(model.parameters()).dtype == getattr(torch, dtype)
        return logits, generated.cpu()
    finally:
        meter.finish()
        forward.resources = meter.results


def child_forward(args, source, config, output):
    from openternary.services.process import run_process

    request = output.with_suffix(".json")
    write(request, {"source": str(source), "config": config.model_dump(), "device": args.device, "dtype": args.dtype})
    with output.with_suffix(".log").open("w", encoding="utf-8") as log:
        run_process(
            [sys.executable, str(Path(__file__).resolve()), "--root", str(args.root), "--forward", str(request)],
            timeout=120,
            stdout=log,
        )
    import torch

    return torch.load(output, weights_only=True)


def llm_case(args, folder, backend):
    import torch

    from openternary.services.execution import convert_artifact
    from openternary.services.tensor_validation import validate_tensors

    source = fixture(folder / "source")
    config = configuration(source, backend)
    target = folder / "converted"
    if backend == "ternary":
        from openternary.config.schema import PassConfig

        config.quantization.passes = [PassConfig(name="noop"), PassConfig(name="clip", options={"max_abs": 0.04})]
        config.quantization.mixed_precision = {"model.layers.0.self_attn.q_proj.weight": "preserve"}
    convert_artifact(source, target, config)
    tensor_report = validate_tensors(target)
    if args.device.startswith("cuda"):
        before = child_forward(args, target, config, folder / "first.pt")["logits"]
    else:
        before, _ = forward(target, config, args.device, args.dtype)
    reloaded = child_forward(args, target, config, folder / "reload.pt")
    tolerance = {"rtol": 0, "atol": 0} if args.device == "cpu" else {"rtol": 1e-5, "atol": 1e-6}
    if args.dtype == "bfloat16" and args.device != "cpu":
        tolerance = {"rtol": 0.016, "atol": 0.001}
    torch.testing.assert_close(before, reloaded["logits"], **tolerance)
    if args.device.startswith("cuda"):
        original = child_forward(args, source, config, folder / "original.pt")["logits"]
    else:
        original, _ = forward(source, config, args.device, args.dtype)
    if backend == "torchao":
        # Independently quantize the original model in memory, before serialization.
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        from torchao.quantization.granularity import PerTensor

        from openternary.adapters.registry import select_adapter
        from openternary.adapters.runtime import load_model, load_processor

        model = load_model(config, source, torch.float32, {"": "cpu"})
        adapter = select_adapter(config, source)
        for name, module in model.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                and adapter.classify(name + ".weight", list(module.weight.shape), str(module.weight.dtype)).quantizable
            ):
                quantize_(module, Int8WeightOnlyConfig(granularity=PerTensor(), set_inductor_config=False))
        tokens = load_processor(config, source)("hello world", return_tensors="pt")
        tokens.pop("token_type_ids", None)
        with torch.inference_mode():
            immediate = model(**tokens).logits.float()
        if args.device == "cpu" and args.dtype == "float32":
            torch.testing.assert_close(immediate, before, rtol=0, atol=0)
    else:
        from openternary.services.artifacts import export_artifact
        from openternary.services.snapshot import SnapshotReader

        packed = folder / "packed"
        restored = folder / "restored"
        export_artifact(target, packed, "ternary-packed")
        validate_tensors(packed)
        export_artifact(packed, restored, "safetensors")
        validate_tensors(restored)
        with SnapshotReader(target) as left, SnapshotReader(restored) as right:
            assert left.keys() == right.keys()
            for name in left.keys():  # noqa: SIM118 - SnapshotReader is not a mapping
                a, b = left.get_tensor(name), right.get_tensor(name)
                assert a.shape == b.shape and a.dtype == b.dtype
                assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        with SnapshotReader(source) as left, SnapshotReader(target) as right:
            name = "model.layers.0.self_attn.q_proj.weight"
            assert torch.equal(left.get_tensor(name), right.get_tensor(name))
            from openternary.quant.ternary import dequantize, quantize_absmean

            name = "model.layers.1.self_attn.q_proj.weight"
            expected = dequantize(quantize_absmean(left.get_tensor(name).clamp(-0.04, 0.04)))
            assert torch.equal(expected, right.get_tensor(name))
    return {
        "tensors": tensor_report,
        "reload_bit_equal": bool(torch.equal(before, reloaded["logits"])),
        "reload_max_abs": float((before - reloaded["logits"]).abs().max()),
        "reload_tolerance": tolerance,
        "source_output_mae": float((original - before).abs().mean()),
        "quality_accepted": False,
    }


def diffusion_fixture(root):
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    torch.manual_seed(42)
    tokenizer = CLIPTokenizer(
        vocab={
            "<|startoftext|>": 0,
            "<|endoftext|>": 1,
            **{c: i + 2 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")},
        },
        merges=[],
        model_max_length=16,
    )
    encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
        )
    )
    unet = UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=4,
        norm_num_groups=8,
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(32, 64),
        latent_channels=4,
        layers_per_block=1,
        norm_num_groups=8,
        sample_size=32,
    )
    pipe = StableDiffusionPipeline(
        unet=unet,
        vae=vae,
        text_encoder=encoder,
        tokenizer=tokenizer,
        scheduler=DDIMScheduler(num_train_timesteps=20, steps_offset=1, clip_sample=False),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.save_pretrained(root)
    write(root / "openternary-fixture.json", {"seed": 42, "pretrained": False, "family": "diffusion"})
    return root


def diffusion_case(args, folder):
    from openternary.config.schema import AppConfig
    from openternary.services.diffusion import DiffusionProfile, evaluate_diffusion
    from openternary.services.execution import convert_artifact
    from openternary.services.tensor_validation import validate_tensors

    source = diffusion_fixture(folder / "pipeline")
    config = AppConfig.model_validate(
        {
            "model": {"id": str(source), "revision": None, "adapter": "diffusers", "component": "unet"},
            "device": "cpu",
            "dtype": "fp32",
        }
    )
    converted = folder / "component"
    convert_artifact(source, converted, config)
    tensors = validate_tensors(converted)
    profile = DiffusionProfile(
        name="synthetic-32-v1",
        prompts=["hello"],
        seeds=[42],
        steps=2,
        width=32,
        height=32,
        component="unet",
        warmup=False,
    )
    config.device = "cuda" if args.device.startswith("cuda") else "cpu"
    config.dtype = "bf16" if args.dtype == "bfloat16" else "fp32"
    result = evaluate_diffusion(config, source, profile, folder / "images", converted)
    write(folder / "images" / "evaluation.json", result)
    return {"tensors": tensors, "evaluation": result, "quality_accepted": False}


def quality_search_case(args, folder):
    from openternary.benchmark.quality_runner import run_quality_benchmark
    from openternary.config.schema import QuantizationConfig
    from openternary.services.optimize import QualityProfile, SearchBudget, run_search

    source = fixture(folder / "source")
    config = configuration(source)
    dataset = folder / "quality-data.json"
    write(
        dataset,
        {
            "schema_version": 1,
            "dataset": {
                "id": "artificial-cli-fixture",
                "revision": "seed42-v1",
                "license": "CC0-1.0",
                "source": "locally authored synthetic sentences",
            },
            "splits": {
                "calibration": [{"id": "cal", "text": "calibration fixture material", "language": "en"}],
                "validation": [
                    {"id": "ven", "text": "hello artificial world", "language": "en"},
                    {"id": "vja", "text": "日本語の人工検証文章です", "language": "ja"},
                ],
                "test": [{"id": "test", "text": "final untouched passage", "language": "en"}],
            },
            "instruction_cases": {
                "validation": [{"id": "iv", "prompt": "say hello", "expected": "hello"}],
                "test": [{"id": "it", "prompt": "赤と返してください", "expected": "赤"}],
            },
        },
    )
    baseline = folder / "baseline"
    report = run_quality_benchmark(config, dataset, snapshot_path=source, split="validation", max_length=32, stride=16)
    assert report["report_schema_version"] == 3
    assert report["identity"]["scope"] == "synthetic"
    assert report["resources"]["load_ram_bytes"] > 0
    write(baseline / "quality.json", report)
    profile = QualityProfile(dataset=str(dataset), baseline=str(baseline), max_length=32, stride=16)
    candidates = [QuantizationConfig(scale_granularity="per_group", group_size=32)]
    search = folder / "search"
    result = run_search(
        config, profile, candidates, search, SearchBudget(max_candidates=1, wall_seconds=180), target_vram=4 * 1024**3
    )
    assert result["status"] == "no_feasible_candidate", result
    assert result["candidates"][0]["status"] == "completed", result
    assert result["best"] is None
    resumed = run_search(
        config,
        profile,
        candidates,
        search,
        SearchBudget(max_candidates=1, wall_seconds=180),
        target_vram=4 * 1024**3,
        resume=True,
    )
    assert len(resumed["candidates"]) == 1
    return {"quality": report, "search_status": result["status"], "resume_reused": True, "quality_accepted": False}


def run_case(args):
    import torch

    torch.set_num_threads(2)
    if args.device.startswith("cuda") and args.case in {"runtime", "diffusion"}:
        assert torch.cuda.is_available(), "explicit GPU is unavailable"
        torch.cuda.set_per_process_memory_fraction(
            min(1, 4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        )
    folder = args.root / args.case
    folder.mkdir(parents=True, exist_ok=False)
    if args.case in {"ternary", "torchao"}:
        return llm_case(args, folder, args.case)
    if args.case == "diffusion":
        return diffusion_case(args, folder)
    if args.case == "quality-search":
        return quality_search_case(args, folder)
    if args.case == "runtime":
        from openternary.services.runtime_probe import probe_runtime

        result = probe_runtime("cuda" if args.device.startswith("cuda") else "cpu")
        assert result["status"] == "passed", result
        return result
    raise ValueError(args.case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--case", choices=["runtime", "ternary", "torchao", "diffusion", "quality-search"])
    parser.add_argument("--forward", type=Path)
    parser.add_argument("--require-wheel", action="store_true")
    args = parser.parse_args()
    import openternary

    package = Path(openternary.__file__).resolve()
    if args.require_wheel and (
        not package.is_relative_to(Path(sys.prefix).resolve()) or "site-packages" not in package.parts
    ):
        raise RuntimeError(f"wheel isolation failed: {package}, prefix={sys.prefix}")
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.root / "empty-hf-cache")
    for key in (
        "MIOPEN_USER_DB_PATH",
        "MIOPEN_CUSTOM_CACHE_DIR",
        "PYTORCH_KERNEL_CACHE_PATH",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "AMD_COMGR_CACHE_DIR",
    ):
        cache = args.root / "runtime-cache" / key.lower()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(cache)
    if args.forward:
        import torch

        from openternary.config.schema import AppConfig

        torch.set_num_threads(2)
        request = json.loads(args.forward.read_text(encoding="utf-8"))
        assert Path(request["source"]).resolve().is_relative_to(args.root)
        if request["device"].startswith("cuda"):
            assert torch.cuda.is_available()
            torch.cuda.set_per_process_memory_fraction(
                min(1, 4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
            )
        logits, tokens = forward(
            Path(request["source"]), AppConfig.model_validate(request["config"]), request["device"], request["dtype"]
        )
        torch.save({"logits": logits, "tokens": tokens}, args.forward.with_suffix(".pt"))
        from openternary.services.resources import runtime_environment

        write(
            args.forward.with_suffix(".resources.json"),
            {"resources": forward.resources, "environment": runtime_environment(request["device"])},
        )
        return 0
    if args.case:
        result = run_case(args)
        write(args.root / f"{args.case}-result.json", result)
        return 0
    from filelock import FileLock

    from openternary.services.artifacts import file_hash
    from openternary.services.process import run_process

    rows = []
    # A single GPU validation process owns the entire case sequence, including reload children.
    lock = (
        FileLock(str(args.root.parent / "gpu-validation.lock"))
        if args.device.startswith("cuda")
        else contextlib.nullcontext()
    )
    with lock:
        cases = ["runtime", "ternary", "torchao", "diffusion"]
        if args.device == "cpu":
            cases.append("quality-search")
        for case in cases:
            gpu_used = 0.0
            for path in args.root.parent.glob("*/validation.json"):
                prior = json.loads(path.read_text(encoding="utf-8"))
                if prior.get("device", "").startswith("cuda"):
                    gpu_used += sum(row["seconds"] for row in prior["cases"])
            remaining = 1800 - gpu_used if args.device.startswith("cuda") else 300
            artifact_bytes = sum(p.stat().st_size for p in args.root.parent.rglob("*") if p.is_file())
            if remaining <= 0 or artifact_bytes >= 4 * 1024**3:
                raise RuntimeError("validation budget exhausted (30 GPU minutes or 4 GiB reports)")
            started = time.monotonic()
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--root",
                str(args.root),
                "--case",
                case,
                "--device",
                args.device,
                "--dtype",
                args.dtype,
            ]
            if args.require_wheel:
                command.append("--require-wheel")
            log_path = args.root / f"{case}.log"
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    run_process(command, timeout=min(300, remaining), stdout=log)
                status, reason = "passed", None
            except Exception as exc:
                status, reason = "failed", str(exc)
            rows.append(
                {
                    "case": case,
                    "status": status,
                    "reason": reason,
                    "seconds": time.monotonic() - started,
                    "log": str(log_path),
                    "log_sha256": file_hash(log_path),
                }
            )
            write(
                args.root / "validation.json",
                {
                    "scope": "synthetic_real_libraries",
                    "pretrained": False,
                    "model_quality_acceptance": False,
                    "python": sys.executable,
                    "prefix": sys.prefix,
                    "package": str(package),
                    "device": args.device,
                    "dtype": args.dtype,
                    "cases": rows,
                },
            )
    return int(any(row["status"] != "passed" for row in rows))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None
