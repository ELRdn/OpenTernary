"""Gemma 4 E2B adapter — Gemma4ForConditionalGeneration向け分類."""

from __future__ import annotations

from openternary.adapters.base import ModelAdapter, TensorInfo, default_quantizable_role
from openternary.utils.safetensors_header import tensor_param_count


class Gemma4Adapter(ModelAdapter):
    """Gemma 4 E2B分類アダプタ."""

    def __init__(self, target_attention: bool = True, target_mlp: bool = True) -> None:
        self.target_attention = target_attention
        self.target_mlp = target_mlp

    @staticmethod
    def supports(model_id: str) -> bool:
        mid = model_id.lower()
        return "gemma" in mid and "4" in mid

    def classify(self, tensor_name: str, shape: list[int], dtype: str) -> TensorInfo:
        role, quantizable, reason = default_quantizable_role(
            tensor_name,
            target_attention=self.target_attention,
            target_mlp=self.target_mlp,
        )
        return TensorInfo(
            name=tensor_name,
            shape=shape,
            dtype=dtype,
            param_count=tensor_param_count(shape),
            role=role,
            quantizable=quantizable,
            exclude_reason=reason,
        )

    def architecture_info(self, config: dict[str, object]) -> dict[str, object]:
        """config.jsonからarchitecture要約を抽出."""
        text_cfg = config.get("text_config", {}) if isinstance(config.get("text_config"), dict) else {}
        vision_cfg = config.get("vision_config", {}) if isinstance(config.get("vision_config"), dict) else {}
        audio_cfg = config.get("audio_config", {}) if isinstance(config.get("audio_config"), dict) else {}
        archs = config.get("architectures")
        arch_name: object = archs[0] if isinstance(archs, list) and len(archs) > 0 else "unknown"  # type: ignore[index]
        return {
            "architecture": arch_name,
            "model_type": config.get("model_type", "unknown"),
            "transformers_version": config.get("transformers_version", "unknown"),
            "dtype": config.get("dtype", "unknown"),
            "text": {
                "model_type": text_cfg.get("model_type", "unknown") if isinstance(text_cfg, dict) else "unknown",
                "hidden_size": text_cfg.get("hidden_size", "unknown") if isinstance(text_cfg, dict) else "unknown",
                "intermediate_size": text_cfg.get("intermediate_size", "unknown")
                if isinstance(text_cfg, dict)
                else "unknown",
                "num_hidden_layers": text_cfg.get("num_hidden_layers", "unknown")
                if isinstance(text_cfg, dict)
                else "unknown",
                "num_attention_heads": text_cfg.get("num_attention_heads", "unknown")
                if isinstance(text_cfg, dict)
                else "unknown",
                "num_key_value_heads": text_cfg.get("num_key_value_heads", "unknown")
                if isinstance(text_cfg, dict)
                else "unknown",
                "vocab_size": text_cfg.get("vocab_size", "unknown") if isinstance(text_cfg, dict) else "unknown",
                "tie_word_embeddings": config.get("tie_word_embeddings", "unknown"),
            },
            "vision": {
                "hidden_size": vision_cfg.get("hidden_size", "unknown") if isinstance(vision_cfg, dict) else "unknown",
                "num_hidden_layers": vision_cfg.get("num_hidden_layers", "unknown")
                if isinstance(vision_cfg, dict)
                else "unknown",
            },
            "audio": {
                "hidden_size": audio_cfg.get("hidden_size", "unknown") if isinstance(audio_cfg, dict) else "unknown",
                "num_hidden_layers": audio_cfg.get("num_hidden_layers", "unknown")
                if isinstance(audio_cfg, dict)
                else "unknown",
            },
            "source": "config.json",
        }


def get_adapter(model_id: str, target_attention: bool = True, target_mlp: bool = True) -> ModelAdapter:
    """model_idから適切なアダプタを返す. 現状Gemma 4のみ."""
    if Gemma4Adapter.supports(model_id):
        return Gemma4Adapter(target_attention=target_attention, target_mlp=target_mlp)
    raise ValueError(
        f"Unsupported model '{model_id}'. Supported: gemma4 family (e.g. google/gemma-4-E2B-it-qat-q4_0-unquantized)"
    )
