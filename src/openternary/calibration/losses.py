"""較正用再構成損失 — Phase 4.1.

teacher/student の活性化テンソル間の再構成誤差を計算する。
すべての関数は torch 依存（未導入時は ImportError）。
有限性チェックを徹底し、NaN/Inf を含む入力や結果は ValueError とする。

集約は「平均の平均」ではなく ``total_squared_error / total_elements`` で計算する。
"""

from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    """torch が未導入なら ImportError を送出する."""
    if torch is None:
        raise ImportError("torch is required for calibration losses. Install with: uv sync --extra ml")


def _check_finite(tensor: torch.Tensor, name: str) -> None:  # type: ignore[type-arg]
    """テンソルがすべて有限か検証し、違反時は ValueError."""
    # torch.isfinite は bool テンソルを返すため .all() で集約する
    if not torch.isfinite(tensor).all().item():  # type: ignore[union-attr]
        raise ValueError(f"NaN/Inf detected in {name}")


def _validate_pair(
    teacher: torch.Tensor,  # type: ignore[type-arg]
    student: torch.Tensor,  # type: ignore[type-arg]
) -> None:
    """teacher / student ペアの基本検証.

    - shape 一致
    - 空テンソル禁止
    - 有限性（NaN/Inf 禁止）
    """
    _require_torch()
    if teacher.shape != student.shape:
        raise ValueError(f"shape mismatch: teacher {tuple(teacher.shape)} vs student {tuple(student.shape)}")
    if teacher.numel() == 0 or student.numel() == 0:
        raise ValueError("empty tensor is not allowed for loss computation")
    _check_finite(teacher, "teacher")
    _check_finite(student, "student")


def mse_loss(
    teacher: torch.Tensor,  # type: ignore[type-arg]
    student: torch.Tensor,  # type: ignore[type-arg]
) -> torch.Tensor:  # type: ignore[type-arg]
    """平均二乗誤差（MSE）を計算する.

    Args:
        teacher: 教師モデルの活性化テンソル.
        student: 学生（三値化後）モデルの活性化テンソル. teacher と同 shape であること.

    Returns:
        スカラーテンソル（0 次元）の MSE 損失.

    Raises:
        ImportError: torch 未導入時.
        ValueError: shape 不一致 / 空テンソル / NaN/Inf 入力 / 結果 NaN 時.
    """
    _validate_pair(teacher, student)
    # float32 で差分をとり精度を確保（bf16 等からの誤差蓄積を抑制）
    diff = teacher.to(torch.float32) - student.to(torch.float32)  # type: ignore[union-attr]
    _check_finite(diff, "diff")
    loss = diff.pow(2).mean()
    if not torch.isfinite(loss).all().item():  # type: ignore[union-attr]
        raise ValueError("NaN/Inf detected in mse_loss result")
    return loss


def l1_loss(
    teacher: torch.Tensor,  # type: ignore[type-arg]
    student: torch.Tensor,  # type: ignore[type-arg]
) -> torch.Tensor:  # type: ignore[type-arg]
    """平均絶対誤差（L1 / MAE）を計算する.

    Args:
        teacher: 教師活性化テンソル.
        student: 学生活性化テンソル.

    Returns:
        スカラーテンソル L1 損失.

    Raises:
        ImportError: torch 未導入時.
        ValueError: shape 不一致 / 空テンソル / NaN/Inf 入力 / 結果 NaN 時.
    """
    _validate_pair(teacher, student)
    diff = teacher.to(torch.float32) - student.to(torch.float32)  # type: ignore[union-attr]
    _check_finite(diff, "diff")
    loss = diff.abs().mean()
    if not torch.isfinite(loss).all().item():  # type: ignore[union-attr]
        raise ValueError("NaN/Inf detected in l1_loss result")
    return loss


def huber_loss(
    teacher: torch.Tensor,  # type: ignore[type-arg]
    student: torch.Tensor,  # type: ignore[type-arg]
    delta: float = 1.0,
) -> torch.Tensor:  # type: ignore[type-arg]
    """Huber 損失（Smooth L1）を計算する.

    小さな誤差では二乗、大きな誤差では線形に振る舞う.

    Args:
        teacher: 教師活性化テンソル.
        student: 学生活性化テンソル.
        delta: 二乗と線形の境界閾値. 正の値であること.

    Returns:
        スカラーテンソル Huber 損失.

    Raises:
        ImportError: torch 未導入時.
        ValueError: delta 非正 / shape 不一致 / NaN/Inf 入力 / 結果 NaN 時.
    """
    if delta <= 0:
        raise ValueError(f"delta must be >0, got {delta}")
    _validate_pair(teacher, student)
    diff = teacher.to(torch.float32) - student.to(torch.float32)  # type: ignore[union-attr]
    _check_finite(diff, "diff")
    abs_diff = diff.abs()
    # 二乗領域: 0.5 * diff^2, 線形領域: delta * |diff| - 0.5 * delta^2
    quadratic = 0.5 * diff.pow(2)
    linear = delta * abs_diff - 0.5 * delta * delta
    loss = torch.where(abs_diff <= delta, quadratic, linear).mean()  # type: ignore[union-attr]
    if not torch.isfinite(loss).all().item():  # type: ignore[union-attr]
        raise ValueError("NaN/Inf detected in huber_loss result")
    return loss


def aggregated_mse_loss(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],  # type: ignore[type-arg]
) -> torch.Tensor:  # type: ignore[type-arg]
    """複数レイヤーの MSE を要素数加重で集約する.

    「平均の平均」ではなく ``total_squared_error / total_elements`` で計算する.
    各レイヤーの要素数が異なる場合でも、大きなレイヤーが適切に重み付けされる.

    Args:
        pairs: (teacher, student) ペアの列. 各ペアは同 shape であること.
            少なくとも 1 ペア必要.

    Returns:
        スカラーテンソル：全レイヤー通算の MSE.

    Raises:
        ImportError: torch 未導入時.
        ValueError: pairs 空 / shape 不一致 / NaN/Inf / 結果 NaN 時.
    """
    _require_torch()
    if len(pairs) == 0:
        raise ValueError("pairs must contain at least one (teacher, student) tuple")
    # デバイスと dtype は最初の teacher に合わせる（float32 蓄積だが返却はその device）
    device = pairs[0][0].device
    total_se = torch.tensor(0.0, dtype=torch.float32, device=device)  # type: ignore[union-attr]
    total_elems = 0
    for idx, (teacher, student) in enumerate(pairs):
        if teacher.shape != student.shape:
            raise ValueError(
                f"shape mismatch at index {idx}: teacher {tuple(teacher.shape)} vs student {tuple(student.shape)}"
            )
        if teacher.numel() == 0 or student.numel() == 0:
            raise ValueError(f"empty tensor at index {idx} is not allowed")
        _check_finite(teacher, f"teacher[{idx}]")
        _check_finite(student, f"student[{idx}]")
        diff = teacher.to(torch.float32) - student.to(torch.float32)  # type: ignore[union-attr]
        _check_finite(diff, f"diff[{idx}]")
        total_se = total_se + diff.pow(2).sum()
        total_elems += int(teacher.numel())
        if not torch.isfinite(total_se).all().item():  # type: ignore[union-attr]
            raise ValueError(f"NaN/Inf detected in aggregated total at index {idx}")
    if total_elems == 0:
        raise ValueError("total elements must be >0")
    loss = total_se / float(total_elems)
    if not torch.isfinite(loss).all().item():  # type: ignore[union-attr]
        raise ValueError("NaN/Inf detected in aggregated_mse_loss result")
    return loss


# 後方互換エイリアス（名称揺れ対策）
aggregate_mse_loss = aggregated_mse_loss
global_mse_loss = aggregated_mse_loss


__all__ = [
    "aggregated_mse_loss",
    "aggregate_mse_loss",
    "global_mse_loss",
    "huber_loss",
    "l1_loss",
    "mse_loss",
]
