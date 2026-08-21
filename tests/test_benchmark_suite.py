"""Benchmark suite v1 tests — protocol/result fingerprint."""

from openternary.benchmark.suites import (
    SMOKE_PROMPTS,
    SMOKE_SUITE_VERSION,
    protocol_fingerprint,
    result_fingerprint,
)


def test_smoke_suite_has_5_prompts() -> None:
    assert SMOKE_SUITE_VERSION == "1"
    assert len(SMOKE_PROMPTS) == 5
    ids = [p["id"] for p in SMOKE_PROMPTS]
    assert ids == [
        "smoke.en.short",
        "smoke.ja.short",
        "smoke.ja.summary",
        "smoke.reasoning.simple",
        "smoke.code.python",
    ]
    for p in SMOKE_PROMPTS:
        assert "prompt" in p and len(p["prompt"]) > 0


def test_protocol_fingerprint_stable() -> None:
    fp1 = protocol_fingerprint()
    fp2 = protocol_fingerprint()
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


def test_protocol_fingerprint_changes_on_thinking_and_generation() -> None:
    base = protocol_fingerprint(thinking=False, generation={"do_sample": False, "num_beams": 1, "max_new_tokens": 64})
    with_think = protocol_fingerprint(
        thinking=True, generation={"do_sample": False, "num_beams": 1, "max_new_tokens": 64}
    )
    assert base != with_think
    other_gen = protocol_fingerprint(
        thinking=False, generation={"do_sample": False, "num_beams": 1, "max_new_tokens": 128}
    )
    assert base != other_gen


def test_result_fingerprint_stable_and_order_independent() -> None:
    r1 = [
        {"id": "smoke.en.short", "generated_token_ids": [1, 2, 3]},
        {"id": "smoke.ja.short", "generated_token_ids": [4, 5]},
    ]
    r2 = [
        {"id": "smoke.ja.short", "generated_token_ids": [4, 5]},
        {"id": "smoke.en.short", "generated_token_ids": [1, 2, 3]},
    ]
    assert result_fingerprint(r1) == result_fingerprint(r2)
    r3 = [{"id": "smoke.en.short", "generated_token_ids": [1, 2, 4]}]
    assert result_fingerprint(r1) != result_fingerprint(r3)
