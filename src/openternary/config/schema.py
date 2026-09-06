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


class VramConfig(BaseModel):
    """VRAM soft policy — RX7600 8GB default."""

    max_fraction: float = Field(default=0.80, gt=0, le=1.0, description="VRAM max fraction for process")
    reserve_mb: int = Field(default=1024, ge=0, description="VRAM reserve for desktop (MB)")
    min_free_mb: int = Field(default=768, ge=0, description="VRAM min free after allocation (MB)")


class MicrobatchConfig(BaseModel):
    """Microbatch policy."""

    auto: bool = Field(default=True, description="Auto shrink microbatch on OOM")
    min_size: int = Field(default=1, ge=1, description="Minimum microbatch size")


class RuntimeConfig(BaseModel):
    """Runtime backend / VRAM policy — Phase 4.2-G."""

    low_vram: bool = Field(default=True, description="Enable dynamic low-VRAM placement (current module only on GPU)")
    vram: VramConfig = Field(default_factory=VramConfig)
    microbatch: MicrobatchConfig = Field(default_factory=MicrobatchConfig)


class CalibrationConfig(BaseModel):
    """較正/再構成設定（Phase 4.2 recon-threshold 対応）.

    Phase 4.1: method=recon-scale, threshold_enabled=false で完全互換。
    Phase 4.2: method=recon-threshold, threshold_enabled=true で scale+threshold 同時最適化。
    window は 4.2 でも per-layer のみ。
    """

    enabled: bool = Field(default=False, description="較正を有効化するか")
    method: Literal["recon-scale", "recon-threshold"] = Field(
        default="recon-scale", description="較正手法（recon-scale: Phase4.1, recon-threshold: Phase4.2）"
    )
    dataset: Literal["synthetic", "wiki-tiny"] = Field(
        default="synthetic", description="較正用データセット（c4-tinyは4.2以降）"
    )
    num_samples: int = Field(default=32, ge=1, le=1024, description="較正サンプル数")
    seq_len: int = Field(default=128, ge=16, le=512, description="較正サンプル系列長")
    steps: int = Field(default=50, ge=1, description="再構成ステップ数")
    lr: float = Field(default=1e-3, gt=0, description="学習率")
    # Phase 4.2 threshold fields (後方互換: デフォルト無効で 4.1 と同一)
    threshold_enabled: bool = Field(default=False, description="閾値学習を有効化するか（Phase 4.2）")
    threshold_lr: float | None = Field(default=None, description="閾値用学習率（None なら lr を流用）")
    threshold_init_ratio: float = Field(default=0.5, gt=0.0, lt=1.0, description="閾値初期 ratio（0,1）")
    threshold_granularity: Literal["per_tensor", "per_group"] = Field(
        default="per_group", description="閾値粒度（Phase 4.2 は per_group 推奨）"
    )
    threshold_estimator: Literal["clipped-ste"] = Field(
        default="clipped-ste", description="STE 推定器（Phase 4.2 は clipped-ste のみ）"
    )
    threshold_ste_width: float = Field(default=0.1, gt=0, description="clipped STE width")
    threshold_eps: float = Field(default=0.01, gt=0, lt=0.5, description="threshold_ratio 境界 eps")
    optimizer: Literal["adam", "sgd"] = Field(default="adam", description="オプティマイザ種別")
    loss: Literal["mse", "l1"] = Field(default="mse", description="再構成損失種別")
    window: Literal["per-layer"] = Field(default="per-layer", description="較正ウィンドウ（4.1/4.2 は per-layer のみ）")
    checkpoint_interval: int = Field(default=10, ge=1, description="チェックポイント保存間隔（ステップ）")
    allow_dataset_fallback: bool = Field(
        default=False, description="データセット取得失敗時に synthetic へのフォールバックを許可するか"
    )
    held_out_ratio: float = Field(default=0.2, ge=0.05, le=0.5, description="ホールドアウト検証用比率")
    seed: int | None = Field(default=None, description="較正専用シード（None なら AppConfig.seed を継承）")
    # Phase 4.2 warm start
    init_from: str | None = Field(default=None, description="Phase 4.1 成果物からの warm start 元 run dir")


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
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

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
