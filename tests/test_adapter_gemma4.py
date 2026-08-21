"""Gemma4 adapter policy tests."""

from openternary.adapters.base import default_quantizable_role
from openternary.adapters.gemma4 import Gemma4Adapter


def test_attention_quantizable_when_enabled() -> None:
    adapter = Gemma4Adapter(target_attention=True, target_mlp=True)
    info = adapter.classify("model.layers.0.self_attn.q_proj.weight", [1536, 1536], "BF16")
    assert info.role == "attention.q_proj"
    assert info.quantizable is True


def test_attention_not_quantizable_when_disabled() -> None:
    adapter = Gemma4Adapter(target_attention=False, target_mlp=True)
    info = adapter.classify("model.layers.0.self_attn.q_proj.weight", [1536, 1536], "BF16")
    assert info.quantizable is False
    assert "attention=false" in (info.exclude_reason or "")


def test_mlp_quantizable() -> None:
    adapter = Gemma4Adapter(target_attention=True, target_mlp=True)
    for name in [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]:
        info = adapter.classify(name, [6144, 1536], "BF16")
        assert info.quantizable is True, name
        assert info.role.startswith("mlp.")


def test_excluded_modules() -> None:
    for name, expected_role in [
        ("model.embed_tokens.weight", "embedding"),
        ("model.layers.0.self_attn.q_proj.weight", "attention.q_proj"),  # quantizable
        ("model.layers.0.input_layernorm.weight", "norm"),
        ("lm_head.weight", "lm_head"),
        ("model.layers.0.per_layer_embedding.weight", "ple"),
        ("vision_tower.encoder.layers.0.weight", "vision"),
        ("audio_tower.encoder.layers.0.weight", "audio"),
    ]:
        role, quantizable, _ = default_quantizable_role(name, target_attention=True, target_mlp=True)
        if expected_role in ("embedding", "norm", "lm_head", "ple", "vision", "audio"):
            assert quantizable is False, name
            assert role == expected_role, f"{name} expected {expected_role} got {role}"


def test_unknown_weight_not_quantizable() -> None:
    adapter = Gemma4Adapter()
    info = adapter.classify("model.layers.0.unknown_proj.weight", [1536, 1536], "BF16")
    assert info.quantizable is False
    assert info.role == "other"


def test_architecture_info_extracts_gemma4() -> None:
    adapter = Gemma4Adapter()
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
            "num_hidden_layers": 35,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "vocab_size": 262144,
        },
        "vision_config": {"hidden_size": 768, "num_hidden_layers": 16},
        "audio_config": {"hidden_size": 1024, "num_hidden_layers": 12},
    }
    arch = adapter.architecture_info(config)
    assert arch["architecture"] == "Gemma4ForConditionalGeneration"
    assert arch["text"]["hidden_size"] == 1536
    assert arch["text"]["num_hidden_layers"] == 35
    assert arch["vision"]["num_hidden_layers"] == 16
    assert arch["audio"]["num_hidden_layers"] == 12
