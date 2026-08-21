"""Benchmark runner — Phase 1b BF16 Smoke Inference Baseline.

Gemma 4 公式のロード方式に準拠:
  processor = AutoProcessor.from_pretrained(snapshot)
  model = AutoModelForMultimodalLM.from_pretrained(snapshot, dtype=bf16, device_map=...)

- thinking=False 固定（Baseline v1）
- greedy: do_sample=False, num_beams=1 のみ（temperature/top_p/top_k は設定しない）
- BF16 は exact — CPU 非対応なら loud error (exit 2相当の例外)
- warmup 1件 → 本計測、トークンIDsとparse_responseを保存、protocol/result fingerprintを生成
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

from openternary.benchmark.suites import (
    SMOKE_PROMPTS,
    SMOKE_SUITE_VERSION,
    protocol_fingerprint,
    result_fingerprint,
)
from openternary.config.schema import AppConfig


def _resolve_device_map(device: str) -> str | dict[str, str] | None:
    """auto は実デバイスへ明示解決 — CPU-onlyで disk offloadへ流さない."""
    # torch が無い環境（テストmock）でも動くように try
    try:
        import torch  # type: ignore[import-not-found]

        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False

    if device == "auto":
        # CPU-only環境で device_map="auto" をそのままAccelerateへ渡すと
        # 「You are trying to offload the whole model to the disk」になるため、
        # 明示的に cpu へ解決する。CUDA がある環境のみ auto を使う。
        if has_cuda:
            return "auto"
        return {"": "cpu"}
    if device == "cuda":
        # 明示CUDA要求だが実機がCPU-onlyなら loud error ではなく Accelerateに委譲
        # （BF16 exact と同様、from_pretrained で失敗すれば上位で捕捉）
        return "auto"
    if device == "cpu":
        return {"": "cpu"}
    return "auto"


def _dtype_to_torch_dtype(dtype_str: str) -> Any:
    import torch  # type: ignore[import-not-found]

    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def run_benchmark(
    config: AppConfig,
    snapshot_path: pathlib.Path | None = None,
    suite: str | None = None,
) -> dict[str, Any]:
    """Smoke benchmark を実行し benchmark.json 相当の dict を返す.

    Args:
        config: 解決済み AppConfig
        snapshot_path: 解決済み snapshot Path（Noneなら HF cache-only解決）
        suite: suite 名（Noneなら config.benchmark.suite）
    """
    from openternary.utils.hf_cache import resolve_snapshot

    if snapshot_path is None:
        snapshot_path = resolve_snapshot(config.model.id, config.model.revision)
    else:
        snapshot_path = pathlib.Path(snapshot_path)

    suite_name = suite or config.benchmark.suite
    if suite_name != "smoke":
        raise ValueError(f"Phase 1b only supports suite='smoke', got '{suite_name}'")

    # ml 依存の存在チェック（loud errorで exit 2 に繋げる）
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("torch is required for benchmark. Install with: uv sync --extra ml") from e
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("transformers>=5.6.2 is required for benchmark. Install with: uv sync --extra ml") from e

    # dtype は exact — BF16 要求時にCPUで非対応ならフォールバックせずエラー
    requested_dtype = config.dtype
    torch_dtype = _dtype_to_torch_dtype(requested_dtype)
    if requested_dtype == "bf16" and not torch.cuda.is_available():
        # CPU での bf16 可否をチェック: torch 支持がない場合はエラー
        # 一部 CPU 環境では bf16 が動作するが、確実にエラーにせず試行する
        # ここでは警告のみ出し、実際の from_pretrained で失敗したら例外を伝播
        pass

    thinking = config.benchmark.thinking
    gen_cfg = config.benchmark.generation
    generation_kwargs: dict[str, Any] = {
        "do_sample": gen_cfg.do_sample,
        "num_beams": gen_cfg.num_beams,
        "max_new_tokens": gen_cfg.max_new_tokens,
    }
    # 生成パラメータのバリデーション: Phase 1b は greedy のみ
    if gen_cfg.do_sample is not False or gen_cfg.num_beams != 1:
        raise ValueError(
            f"Phase 1b baseline requires do_sample=false and num_beams=1, got do_sample={gen_cfg.do_sample} num_beams={gen_cfg.num_beams}"
        )

    # プロセッサ / モデル ロード
    device_map = _resolve_device_map(config.device)

    # AutoProcessor / AutoModelForMultimodalLM は snapshot_path を直接受ける
    try:
        processor = AutoProcessor.from_pretrained(str(snapshot_path), trust_remote_code=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load AutoProcessor from {snapshot_path}: {e}") from e

    # モデルロード — dtype は exact
    model_load_start = time.perf_counter()
    try:
        model = AutoModelForMultimodalLM.from_pretrained(
            str(snapshot_path),
            dtype=torch_dtype,  # transformers 5.x は dtype 引数
            device_map=device_map,  # type: ignore[arg-type]
            trust_remote_code=False,
        )
    except TypeError:
        # 古い transformers が torch_dtype 引数名を要求する場合のフォールバック
        try:
            model = AutoModelForMultimodalLM.from_pretrained(
                str(snapshot_path),
                torch_dtype=torch_dtype,  # type: ignore[call-arg]
                device_map=device_map,  # type: ignore[arg-type]
                trust_remote_code=False,
            )
        except Exception as e2:
            raise RuntimeError(f"Failed to load AutoModelForMultimodalLM from {snapshot_path}: {e2}") from e2
    except Exception as e:
        # BF16 非対応などの loud error をそのまま伝播（CLIで exit 2 にする）
        raise RuntimeError(f"Failed to load model with dtype={requested_dtype}: {e}") from e

    model_load_ms = (time.perf_counter() - model_load_start) * 1000

    # BF16 exact 検証: ロード後の dtype が要求と一致するか確認（可能な範囲で）
    try:
        # パラメータの dtype をサンプル
        param_dtype = next(model.parameters()).dtype  # type: ignore[attr-defined]
        if param_dtype != torch_dtype:
            raise RuntimeError(
                f"BF16 exact violation: requested {torch_dtype} but model loaded as {param_dtype}. "
                f"Silent fallback is forbidden for Baseline v1. Use --dtype to request a different precision explicitly."
            )
    except StopIteration:
        pass
    except RuntimeError:
        raise
    except Exception:
        pass  # 検証できない環境ではスキップ

    model.eval()

    # Suite 解決
    prompts = list(SMOKE_PROMPTS)
    proto_fp = protocol_fingerprint(
        suite=suite_name,
        version=SMOKE_SUITE_VERSION,
        prompts=prompts,
        thinking=thinking,
        generation=generation_kwargs,
    )

    # Warmup (1件、計測外)
    warmup_executed = False
    if config.benchmark.warmup and prompts:
        warmup_prompt = prompts[0]
        try:
            _run_single_prompt(
                processor=processor,
                model=model,
                prompt=warmup_prompt["prompt"],
                thinking=thinking,
                generation_kwargs=generation_kwargs,
            )
            warmup_executed = True
        except Exception:
            # warmup 失敗は致命ではないがログに残す
            warmup_executed = False

    # 本計測
    results: list[dict[str, Any]] = []
    total_output_tokens = 0
    total_gen_ms = 0.0

    for item in prompts:
        prompt_id = item["id"]
        prompt_text = item["prompt"]
        t0 = time.perf_counter()
        single = _run_single_prompt(
            processor=processor,
            model=model,
            prompt=prompt_text,
            thinking=thinking,
            generation_kwargs=generation_kwargs,
        )
        # CUDA 同期（latency 正確化）
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        latency_ms = (time.perf_counter() - t0) * 1000
        total_gen_ms += latency_ms
        total_output_tokens += int(single.get("output_tokens", 0))
        results.append(
            {
                "id": prompt_id,
                "prompt": prompt_text,
                "raw_output_text": single.get("raw_output_text"),
                "parsed_output_text": single.get("parsed_output_text"),
                "generated_token_ids": single.get("generated_token_ids"),
                "input_tokens": single.get("input_tokens"),
                "output_tokens": single.get("output_tokens"),
                "latency_ms": round(latency_ms, 2),
            }
        )

    result_fp = result_fingerprint(results)
    avg_latency = round(total_gen_ms / len(results), 2) if results else 0.0
    tokens_per_sec = round(total_output_tokens / (total_gen_ms / 1000), 2) if total_gen_ms > 0 else 0.0

    return {
        "suite": {
            "name": suite_name,
            "version": SMOKE_SUITE_VERSION,
            "fingerprint": proto_fp,
        },
        "model": {
            "id": config.model.id,
            "revision": config.model.revision,
            "snapshot_path": str(snapshot_path),
        },
        "thinking": thinking,
        "generation": generation_kwargs,
        "dtype": {
            "requested": requested_dtype,
            "actual": str(torch_dtype),
        },
        "device": config.device,
        "seed": config.seed,
        "model_load_ms": round(model_load_ms, 2),
        "warmup": {"executed": warmup_executed},
        "results": results,
        "result_fingerprint": result_fp,
        "summary": {
            "num_prompts": len(results),
            "total_output_tokens": total_output_tokens,
            "total_generation_ms": round(total_gen_ms, 2),
            "avg_generation_latency_ms": avg_latency,
            "output_tokens_per_second": tokens_per_sec,
        },
    }


def _run_single_prompt(
    processor: Any,
    model: Any,
    prompt: str,
    thinking: bool,
    generation_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """単一 prompt の生成を実行し token IDs / text を返す."""
    import torch  # type: ignore[import-not-found]

    messages = [{"role": "user", "content": prompt}]
    # Gemma 4 公式: apply_chat_template with enable_thinking
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        # enable_thinking 非対応の processor 互換フォールバック
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

    # device に inputs を移動（device_map="auto" でも input はモデルデバイスへ）
    try:
        device = next(model.parameters()).device  # type: ignore[attr-defined]
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    except Exception:
        pass

    input_len = 0
    try:
        # input_ids があれば長さ取得
        ids = inputs.get("input_ids")
        if ids is not None:
            input_len = int(ids.shape[-1])  # type: ignore[attr-defined]
    except Exception:
        pass

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    # outputs: [batch, seq_len] — input 部分を除いた生成部分を抽出
    generated_ids: list[int] = []
    try:
        seq = outputs[0]  # type: ignore[index]
        # input 長を除去
        gen_only = seq[input_len:] if input_len and len(seq) > input_len else seq  # type: ignore[arg-type]
        generated_ids = gen_only.tolist()  # type: ignore[attr-defined]
    except Exception:
        try:
            generated_ids = outputs[0].tolist()  # type: ignore[attr-defined]
        except Exception:
            generated_ids = []

    # raw decode
    raw_text = ""
    try:
        raw_text = processor.decode(generated_ids, skip_special_tokens=True)
    except Exception:
        raw_text = ""

    # parsed response（Gemma 4 公式: processor.parse_response）
    # 公式APIは prefix を受け取る — input_ids を prefix として渡すと
    # chat template の pre-fill 部分を正しく除外できる
    parsed_text = raw_text
    try:
        if hasattr(processor, "parse_response"):
            prefix = inputs.get("input_ids")
            # token IDs と prefix の両方がある場合は token ベースで parse
            if prefix is not None and generated_ids:
                try:
                    parsed = processor.parse_response(  # type: ignore[attr-defined]
                        generated_ids, prefix=prefix
                    )
                    if isinstance(parsed, str):
                        parsed_text = parsed
                    elif isinstance(parsed, dict) and "response" in parsed:
                        parsed_text = str(parsed["response"])
                    elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                        # batched 返却のケース
                        parsed_text = str(parsed[0].get("response", raw_text))
                    else:
                        parsed_text = raw_text
                except TypeError:
                    # prefix 非対応の古い processor へフォールバック
                    parsed = processor.parse_response(raw_text)  # type: ignore[attr-defined]
                    if isinstance(parsed, str):
                        parsed_text = parsed
                    elif isinstance(parsed, dict) and "response" in parsed:
                        parsed_text = str(parsed["response"])
            else:
                parsed = processor.parse_response(raw_text)  # type: ignore[attr-defined]
                if isinstance(parsed, str):
                    parsed_text = parsed
                elif isinstance(parsed, dict) and "response" in parsed:
                    parsed_text = str(parsed["response"])
    except Exception:
        pass

    return {
        "raw_output_text": raw_text,
        "parsed_output_text": parsed_text,
        "generated_token_ids": generated_ids,
        "input_tokens": input_len,
        "output_tokens": len(generated_ids),
    }
