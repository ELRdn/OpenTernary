"""Model adapter base — tensor分類契約."""

from __future__ import annotations

import abc
import dataclasses
import re


@dataclasses.dataclass(frozen=True)
class TensorInfo:
    """Safetensors header由来のtensor情報 + 分類結果."""

    name: str
    shape: list[int]
    dtype: str
    param_count: int
    role: str
    quantizable: bool
    exclude_reason: str | None = None


class ModelAdapter(abc.ABC):
    """モデル固有の分類ロジックを担う抽象アダプタ."""

    @abc.abstractmethod
    def classify(self, tensor_name: str, shape: list[int], dtype: str) -> TensorInfo:
        """1 tensorを分類してTensorInfoを返す."""

    @abc.abstractmethod
    def architecture_info(self, config: dict[str, object]) -> dict[str, object]:
        """config.json由来のarchitecture要約を返す."""

    @staticmethod
    def supports(model_id: str) -> bool:
        """このアダプタがmodel_idをサポートするか."""
        raise NotImplementedError


# 共通roleパターン (Gemma 4 E2B用だが他モデルでも再利用可能)
_QUANTIZABLE_ATTENTION_RE = re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$")
_QUANTIZABLE_MLP_RE = re.compile(r"\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)\.weight$")
_EXCLUDE_EMBED_RE = re.compile(r"embed_tokens")
_EXCLUDE_LM_HEAD_RE = re.compile(r"lm_head")
_EXCLUDE_NORM_RE = re.compile(r"norm", re.IGNORECASE)
_EXCLUDE_PLE_RE = re.compile(r"(per_layer_embedding|ple)")
_EXCLUDE_VISION_RE = re.compile(r"vision_tower|vision_model|vision_encoder")
_EXCLUDE_AUDIO_RE = re.compile(r"audio_tower|audio_model|audio_encoder")


def default_quantizable_role(
    tensor_name: str,
    target_attention: bool = True,
    target_mlp: bool = True,
) -> tuple[str, bool, str | None]:
    """tensor_nameから (role, quantizable, exclude_reason) を判定.

    Gemma 4 E2B向けのデフォルト分類. 他アダプタからも呼び出し可能.
    """
    if _EXCLUDE_EMBED_RE.search(tensor_name):
        return ("embedding", False, "high precision policy: embedding")
    if _EXCLUDE_LM_HEAD_RE.search(tensor_name):
        return ("lm_head", False, "high precision policy: lm_head (tie_word_embeddings)")
    if _EXCLUDE_NORM_RE.search(tensor_name):
        return ("norm", False, "high precision policy: norm")
    if _EXCLUDE_PLE_RE.search(tensor_name):
        return ("ple", False, "high precision policy: PLE")
    if _EXCLUDE_VISION_RE.search(tensor_name):
        return ("vision", False, "high precision policy: vision tower")
    if _EXCLUDE_AUDIO_RE.search(tensor_name):
        return ("audio", False, "high precision policy: audio tower")
    # attention
    m = _QUANTIZABLE_ATTENTION_RE.search(tensor_name)
    if m:
        role = f"attention.{m.group(1)}"
        if target_attention:
            return (role, True, None)
        return (role, False, "quantization.target.attention=false")
    m = _QUANTIZABLE_MLP_RE.search(tensor_name)
    if m:
        role = f"mlp.{m.group(1)}"
        if target_mlp:
            return (role, True, None)
        return (role, False, "quantization.target.mlp=false")
    # weightだが上記以外 (例: vision/audio以外のlinear)
    if tensor_name.endswith(".weight"):
        return ("other", False, "not in quantizable policy")
    return ("other", False, "not in quantizable policy")
