"""Quality metrics tested against analytic distributions, not model claims."""

import hashlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_perplexity_scores_each_target_once_across_windows():
    from openternary.benchmark.quality import evaluate_perplexity

    class Uniform(torch.nn.Module):
        def forward(self, input_ids):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 4))

    result = evaluate_perplexity(Uniform(), [torch.tensor([0, 1, 2, 3, 0, 1, 2])], max_length=4, stride=2)
    assert result["scored_tokens"] == 6
    assert result["perplexity"] == pytest.approx(4.0)


def test_split_overlap_is_rejected_after_normalization():
    from openternary.benchmark.quality import validate_text_splits

    with pytest.raises(ValueError, match="overlap"):
        validate_text_splits({"calibration": ["Ａ  B"], "validation": ["A B"], "test": ["独立"]})
    result = validate_text_splits({"calibration": ["訓練"], "validation": ["検証"], "test": ["試験"]})
    assert result["disjoint"] is True
    assert len(result["fingerprint"]) == 64


def test_split_near_duplicate_is_rejected_by_minhash():
    from openternary.benchmark.quality import validate_text_splits

    source = "これは独立評価データの近似重複を検出するための十分に長い文章です。" * 20
    near_duplicate = source[:-1] + "！"
    with pytest.raises(ValueError, match="near-duplicate"):
        validate_text_splits(
            {
                "calibration": [source],
                "validation": [near_duplicate],
                "test": ["完全に異なる最終評価文章です。"],
            }
        )


def test_instruction_split_near_duplicate_is_rejected_by_minhash():
    from openternary.benchmark.quality import validate_instruction_splits

    source = "Answer the question using only the supplied context. The independent example is intentionally long. " * 8
    near_duplicate = source[:-1] + "!"
    with pytest.raises(ValueError, match="instruction split near-duplicate"):
        validate_instruction_splits({"validation": [source], "test": [near_duplicate]})

    audit = validate_instruction_splits(
        {
            "validation": ["Answer the astronomy question from this context."],
            "test": ["日本語の歴史資料だけを使って回答してください。"],
        }
    )
    assert audit["disjoint"] is True
    assert audit["near_duplicate"]["method"] == "character-5gram-minhash-v1"


@pytest.mark.parametrize("text, flag", [("", "empty"), ("...!!!", "symbols_only"), ("あ" * 30, "repeated_unit")])
def test_output_collapse_diagnostics(text, flag):
    from openternary.benchmark.quality import output_diagnostics

    assert flag in output_diagnostics(text)["flags"]


def test_normal_japanese_is_not_flagged_as_collapsed():
    from openternary.benchmark.quality import output_diagnostics

    assert output_diagnostics("こんにちは。今日は晴れていますね。")["flags"] == []


def test_quality_report_keeps_language_metrics_and_protocol():
    from openternary.benchmark.quality import evaluate_quality

    class Uniform(torch.nn.Module):
        def forward(self, input_ids):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 4))

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[ord(c) % 4 for c in text]])}

    splits = {
        "calibration": [{"text": "train", "language": "en"}],
        "validation": [{"text": "日本語", "language": "ja"}, {"text": "abcde", "language": "en"}],
        "test": [{"text": "final", "language": "en"}],
    }
    report = evaluate_quality(Uniform(), Tokenizer(), splits, split="validation", max_length=4, stride=2)
    assert report["by_language"]["ja"]["scored_tokens"] == 2
    assert report["by_language"]["en"]["scored_tokens"] == 4
    assert report["overall"]["perplexity"] == pytest.approx(4)
    assert report["scientific_acceptance"] is False


def test_instruction_scoring_requires_matching_case_ids():
    from openternary.benchmark.quality import score_responses

    cases = [{"id": "ja.copy", "expected": "青"}, {"id": "ja.free"}]
    report = score_responses(cases, {"ja.copy": "青", "ja.free": "..."})
    assert report["results"][0]["response_text"] == "青"
    assert report["results"][0]["response_sha256"] == hashlib.sha256("青".encode()).hexdigest()
    assert report["results"][1]["response_text"] == "..."
    assert report["exact_match_scored"] == 1
    assert report["exact_match_accuracy"] == 1.0
    assert report["flagged_outputs"] == 1
    with pytest.raises(ValueError, match="IDs"):
        score_responses(cases, {"ja.copy": "青"})


def test_quality_acceptance_uses_preregistered_composite_and_limits():
    from openternary.benchmark.acceptance import compare_quality_metrics

    control = {
        "general_ppl": 10.0,
        "japanese_ppl": 20.0,
        "instruction_score": 80.0,
        "collapse_count": 0,
    }
    accepted = compare_quality_metrics(
        control,
        {
            "general_ppl": 9.8,
            "japanese_ppl": 20.2,
            "instruction_score": 79.0,
            "collapse_count": 0,
        },
    )
    assert accepted["composite_score"] == pytest.approx(0.0005)
    assert accepted["accepted"] is True

    ppl_regression = compare_quality_metrics(
        control,
        {
            "general_ppl": 10.21,
            "japanese_ppl": 19.0,
            "instruction_score": 82.0,
            "collapse_count": 0,
        },
    )
    assert ppl_regression["accepted"] is False
    assert "general_ppl_regression" in ppl_regression["failed_gates"]

    collapsed = compare_quality_metrics(
        control,
        {
            "general_ppl": 9.0,
            "japanese_ppl": 18.0,
            "instruction_score": 90.0,
            "collapse_count": 1,
        },
    )
    assert collapsed["accepted"] is False
    assert "collapse" in collapsed["failed_gates"]
