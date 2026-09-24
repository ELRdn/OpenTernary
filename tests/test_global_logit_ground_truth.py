"""Ground-truth objective spans for blockwise ternary research."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


def test_next_token_loss_excludes_padding_and_uses_qa_answer_span(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    helper = importlib.import_module("train_gemma4_global_logit_block").ground_truth_next_token_ce
    logits = torch.zeros(1, 5, 6)
    logits[0, 0, 1] = 2
    logits[0, 1, 2] = 3
    logits[0, 2, 0] = -9
    log_probs = logits.log_softmax(dim=-1)
    inputs = {
        "input_ids": torch.tensor([[5, 1, 2, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
    }
    text = {"inputs": inputs, "loss_mask": inputs["attention_mask"]}
    qa = {"inputs": inputs, "loss_mask": torch.tensor([[0, 1, 0, 0, 0]]), "kind": "qa"}
    expected_text = -(log_probs[0, 0, 1] + log_probs[0, 1, 2]) / 2
    assert torch.allclose(helper(log_probs, text), expected_text)
    assert torch.allclose(helper(log_probs, qa), -log_probs[0, 1, 2])
    with pytest.raises(ValueError, match="empty"):
        helper(log_probs, {"inputs": inputs, "loss_mask": torch.zeros_like(inputs["input_ids"])})


def test_qa_first_and_turn_end_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    helper = importlib.import_module("train_gemma4_global_logit_block").ground_truth_next_token_ce
    logits = torch.zeros(1, 6, 6)
    logits[0, 1, 2] = 1
    logits[0, 2, 3] = 2
    logits[0, 3, 4] = 3
    log_probs = logits.log_softmax(dim=-1)
    example = {
        "inputs": {
            "input_ids": torch.tensor([[5, 1, 2, 3, 4, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
        },
        "loss_mask": torch.tensor([[0, 1, 1, 1, 0, 0]]),
        "kind": "qa",
    }
    expected = -(4 * log_probs[0, 1, 2] + 2 * log_probs[0, 2, 3] + log_probs[0, 3, 4]) / 7
    assert torch.allclose(helper(log_probs, example, qa_first_weight=4, qa_end_weight=2), expected)
