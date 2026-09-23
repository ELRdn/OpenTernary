"""Conservative metadata adapters; unknown weights stay unmodified."""

from __future__ import annotations

import math
import re

from openternary.adapters.base import ModelAdapter, TensorInfo, default_quantizable_role


class TransformersAdapter(ModelAdapter):
    name = "transformers"

    def __init__(self, target_attention: bool = True, target_mlp: bool = True) -> None:
        self.target_attention = target_attention
        self.target_mlp = target_mlp

    @staticmethod
    def supports(model_id: str) -> bool:
        return False  # Model IDs alone do not prove an architecture.

    def classify(self, tensor_name: str, shape: list[int], dtype: str) -> TensorInfo:
        role, quantizable, reason = default_quantizable_role(tensor_name, self.target_attention, self.target_mlp)
        if quantizable and (
            len(shape) != 2 or dtype not in {"BF16", "F16", "F32", "torch.bfloat16", "torch.float16", "torch.float32"}
        ):
            quantizable, reason = False, "only floating 2-D Linear weights are supported"
        return TensorInfo(tensor_name, shape, dtype, math.prod(shape), role, quantizable, reason)

    def architecture_info(self, config: dict[str, object]) -> dict[str, object]:
        return {
            "architecture": config.get("architectures", []),
            "model_type": config.get("model_type", "unknown"),
            "adapter": self.name,
            "source": "config.json",
            "model_validation": "pending",
        }


class DiffusersAdapter(TransformersAdapter):
    name = "diffusers"

    def classify(self, tensor_name: str, shape: list[int], dtype: str) -> TensorInfo:
        result = super().classify(tensor_name, shape, dtype)
        if result.quantizable or len(shape) != 2 or "norm" in tensor_name or "embed" in tensor_name:
            return result
        attention = re.search(r"\.(to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj)\.weight$", tensor_name)
        mlp = re.search(r"\.ff\.net\.(0\.proj|2)\.weight$", tensor_name)
        allowed = bool((attention and self.target_attention) or (mlp and self.target_mlp))
        floating = dtype in {"BF16", "F16", "F32", "torch.bfloat16", "torch.float16", "torch.float32"}
        return TensorInfo(
            tensor_name,
            shape,
            dtype,
            math.prod(shape),
            "attention" if attention else "mlp" if mlp else "other",
            allowed and floating,
            None if allowed and floating else "unknown or excluded diffusion tensor",
        )
