"""Benchmark suites — Phase 1b smoke v1 (BF16 Smoke Inference Baseline)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Suite versioning — bump when prompt set or protocol changes.
SMOKE_SUITE_VERSION = "1"

# 固定 5 件 — 推論できること・挙動差分が見えることを重視（品質スコアは持たない）
# 各 id は benchmark.json / result fingerprint のキーになる
SMOKE_PROMPTS: list[dict[str, str]] = [
    {
        "id": "smoke.en.short",
        "prompt": "Write a one-sentence greeting in English.",
    },
    {
        "id": "smoke.ja.short",
        "prompt": "日本語で一文の挨拶を書いてください。",
    },
    {
        "id": "smoke.ja.summary",
        "prompt": "次の文を日本語で一文で要約してください:「OpenTernaryはGemma 4 E2Bを三値化する研究ツールで、再現可能なCLIパイプラインを構築することを目的としています。」",
    },
    {
        "id": "smoke.reasoning.simple",
        "prompt": "If all Bloops are Razzies and some Razzies are Loppies, is it certain that some Bloops are Loppies? Answer yes/no and explain in one sentence.",
    },
    {
        "id": "smoke.code.python",
        "prompt": "Write a Python function add(a, b) that returns a + b. Reply with code only.",
    },
]


def protocol_fingerprint(
    suite: str = "smoke",
    version: str = SMOKE_SUITE_VERSION,
    prompts: list[dict[str, str]] | None = None,
    thinking: bool = False,
    generation: dict[str, Any] | None = None,
) -> str:
    """Suite protocol の決定的 fingerprint (prompt変更・thinking・generation差分を検出).

    対象: suite name + version + prompt ids/messages + thinking + generation_config
    対象外: latency / model revision / run_id
    """
    if prompts is None:
        prompts = SMOKE_PROMPTS
    if generation is None:
        generation = {"do_sample": False, "num_beams": 1, "max_new_tokens": 64}
    # 正規化: prompt は id 順にソートして id+prompt のみを対象
    normalized_prompts = sorted(
        [{"id": p["id"], "prompt": p["prompt"]} for p in prompts],
        key=lambda x: x["id"],
    )
    payload = {
        "suite": suite,
        "version": version,
        "prompts": normalized_prompts,
        "thinking": thinking,
        "generation": generation,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def result_fingerprint(results: list[dict[str, Any]]) -> str:
    """生成結果の決定的 fingerprint — prompt id + generated_token_ids の hash.

    latency / text decode 差分は含めない。2 回実行で一致すれば推論決定的。
    """
    normalized = sorted(
        [{"id": r["id"], "generated_token_ids": r.get("generated_token_ids", [])} for r in results],
        key=lambda x: x["id"],
    )
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
