"""Pydantic schema for OpenTernary configuration.

FR-06 の再現性要件に準拠: すべての実験定義定数はコードにハードコードせず、
このスキーマ + YAML で表現する。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """モデル識別子."""

    id: str = Field(
        default="google/gemma-4-E2B-it-qat-q4_0-unquantized",
        description="Hugging Face model id or local path",
    )
    revision: str | None = Field(
        default="6befbaca7398925921802abd1f277b495b78b738",
        description="Model revision / commit hash",
    )


class QuantizationTargetConfig(BaseModel):
    """量子化対象モジュール種別."""

    attention: bool = True
    mlp: bool = True


class QuantizationConfig(BaseModel):
    """量子化設定."""

    method: Literal["naive"] = "naive"
    codebook: list[int] = Field(default_factory=lambda: [-1, 0, 1])
    group_size: int = Field(default=128, ge=1, description="Group size for group-wise scaling")
    scale_granularity: Literal["per_tensor", "per_group"] = Field(
        default="per_tensor", description="Scale granularity: per_tensor (Phase2 canonical) or per_group (LD-RW)"
    )
    grouping_scheme: Literal["last-dim-rowwise-v1"] = Field(
        default="last-dim-rowwise-v1", description="Grouping axis definition for per_group"
    )
    target: QuantizationTargetConfig = Field(default_factory=QuantizationTargetConfig)


class CalibrationConfig(BaseModel):
    """較正/再構成設定（Phase 4.1 recon-scale）.

    v1.2 FINAL 仕様に準拠。較正は YAML 駆動（CLI オーバーライドは calibration.enabled のみ想定）。
    window は 4.1 では per-layer のみを許可し、per-block は 4.2 で導入予定のためバリデーションエラーとする。
    """

    enabled: bool = Field(default=False, description="較正を有効化するか")
    method: Literal["recon-scale"] = Field(default="recon-scale", description="較正手法（4.1 は recon-scale のみ）")
    dataset: Literal["synthetic", "wiki-tiny"] = Field(
        default="synthetic", description="較正用データセット（c4-tinyは4.2以降）"
    )
    num_samples: int = Field(default=32, ge=1, le=1024, description="較正サンプル数")
    seq_len: int = Field(default=128, ge=16, le=512, description="較正サンプル系列長")
    steps: int = Field(default=50, ge=1, description="再構成ステップ数")
    lr: float = Field(default=1e-3, gt=0, description="学習率")
    optimizer: Literal["adam", "sgd"] = Field(default="adam", description="オプティマイザ種別")
    loss: Literal["mse", "l1"] = Field(default="mse", description="再構成損失種別")
    window: Literal["per-layer"] = Field(
        default="per-layer", description="較正ウィンドウ（4.1 は per-layer のみ、per-block は 4.2）"
    )
    checkpoint_interval: int = Field(default=10, ge=1, description="チェックポイント保存間隔（ステップ）")
    allow_dataset_fallback: bool = Field(
        default=False, description="データセット取得失敗時に synthetic へのフォールバックを許可するか"
    )
    held_out_ratio: float = Field(default=0.2, ge=0.05, le=0.5, description="ホールドアウト検証用比率")
    seed: int | None = Field(default=None, description="較正専用シード（None なら AppConfig.seed を継承）")


class GenerationConfig(BaseModel):
    """生成パラメータ — Phase 1b は greedy 決定的のみ."""

    do_sample: bool = Field(default=False, description="greedy if false")
    num_beams: int = Field(default=1, ge=1, description="beam count")
    max_new_tokens: int = Field(default=64, ge=1, le=2048, description="max tokens to generate")


class BenchmarkConfig(BaseModel):
    """ベンチマークスイート設定 — Phase 1b は smoke v1 のみ."""

    suite: Literal["smoke"] = Field(default="smoke", description="benchmark suite name")
    # 互換維持: 旧 suites リストが YAML に残っていても受理するが suite を優先
    suites: list[str] = Field(default_factory=lambda: ["smoke"], description="deprecated: use suite")
    thinking: bool = Field(default=False, description="Gemma 4 thinking mode (Phase 1b baseline v1 = false)")
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    warmup: bool = Field(default=True, description="run one warmup prompt before timing")


class AppConfig(BaseModel):
    """アプリケーション全体設定. CLI フラグと YAML の統合結果."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    # 共通オプション（CLI契約）
    seed: int = Field(default=42, description="Random seed for reproducibility")
    device: Literal["auto", "cpu", "cuda"] = Field(default="auto", description="Device preference")
    dtype: Literal["bf16", "fp16", "fp32"] = Field(default="bf16", description="Compute dtype")
    output: str | None = Field(default=None, description="Run output directory override")


# CLI フラグと AppConfig パスの対応（隠れ定数禁止のため明示）
CLI_TO_CONFIG: dict[str, str] = {
    "seed": "seed",
    "device": "device",
    "dtype": "dtype",
    "output": "output",
    "group_size": "quantization.group_size",
    "scale_granularity": "quantization.scale_granularity",
    "grouping_scheme": "quantization.grouping_scheme",
    "suite": "benchmark.suite",
    "thinking": "benchmark.thinking",
}
