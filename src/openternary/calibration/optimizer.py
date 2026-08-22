"""較正用スケール最適化ヘルパー — Phase 4.1.

学習可能なスケール ``effective_scale = softplus(raw_scale) + eps`` を提供する。
ゼロスケールグループは正確に 0 を維持し、学習対象外とする。

設計原則:
- ``scale + eps`` の直接加算は禁止。必ず ``softplus`` 経由で正値性を保証する.
- 初期値 ``effective_scale(step0) == orig_scale`` を 1e-6 以内で満たす.
- device / dtype (float32) を保持する.
- per_tensor（1 要素）/ per_group（複数）いずれも扱う.

torch 未導入時は ImportError.
"""

from __future__ import annotations

try:
    import torch
    from torch.nn import Parameter
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Parameter = None  # type: ignore[assignment,misc]


def _require_torch() -> None:
    """torch が未導入なら ImportError を送出する."""
    if torch is None:
        raise ImportError("torch is required for calibration optimizer. Install with: uv sync --extra ml")


def inverse_softplus(
    x: torch.Tensor,  # type: ignore[type-arg]
) -> torch.Tensor:  # type: ignore[type-arg]
    """softplus の逆関数を数値安定に計算する.

    定義: ``softplus(z) = log(1 + exp(z))``
    逆関数: ``inverse_softplus(x) = log(exp(x) - 1)``

    数値安定化:
    - ``x > 20`` では ``exp(x)`` が overflow するため近似 ``inverse ≈ x`` を返す.
    - それ以外は ``log(expm1(x))`` を用いる（``expm1`` は小さい ``x`` で精度が高い）.
    - ``x <= 0`` は数学的に定義域外（``exp(x)-1 <=0``）のため、微小正値でクランプし、
      極めて負の raw 値を返すことで ``softplus(raw)+eps ≈ eps`` とする.

    Args:
        x: 正のターゲット値テンソル（通常は ``orig_scale - eps``）.

    Returns:
        ``softplus(raw) == x`` を満たす ``raw`` テンソル.

    Raises:
        ImportError: torch 未導入時.
        ValueError: NaN/Inf を含む入力時、または結果が NaN の時.
    """
    _require_torch()
    if not torch.isfinite(x).all().item():  # type: ignore[union-attr]
        raise ValueError("NaN/Inf detected in inverse_softplus input")
    # float32 で計算し、device / dtype を保持
    xf = x.to(torch.float32)  # type: ignore[union-attr]
    # 定義域外（<=0）を微小正値でクランプ: expm1(1e-9) ≈ 1e-9, log ≈ -20.7
    # これにより softplus(raw) ≈ 1e-9 となり、最終 effective = 1e-9 + eps ≈ eps
    eps_clamp = torch.tensor(1e-9, dtype=torch.float32, device=xf.device)  # type: ignore[union-attr]
    xf_clamped = torch.clamp(xf, min=eps_clamp)  # type: ignore[union-attr]
    # 大きい x では恒等近似（softplus(x) ≈ x）、小さい x では log(expm1(x))
    # 閾値 20 は exp(20) ≈ 4.8e8 で十分大きく、誤差が無視できる境界
    large_mask = xf_clamped > 20.0
    # expm1(x) = exp(x) - 1 を高精度で計算
    safe = torch.log(torch.expm1(xf_clamped))  # type: ignore[union-attr]
    result = torch.where(large_mask, xf_clamped, safe)  # type: ignore[union-attr]
    if not torch.isfinite(result).all().item():  # type: ignore[union-attr]
        raise ValueError("NaN/Inf detected in inverse_softplus result")
    return result


def build_scale_params(
    orig_scales: torch.Tensor,  # type: ignore[type-arg]
    zero_mask: torch.Tensor | None = None,  # type: ignore[type-arg]
    eps: float = 1e-6,
) -> tuple[Parameter, torch.Tensor]:  # type: ignore[type-arg]
    """学習可能なスケールパラメータを構築する.

    ゼログループ（``orig_scale == 0``）は正確に 0 を維持し、勾配が流れないようにする。
    非ゼログループは ``raw = inverse_softplus(orig - eps)`` で初期化し、
    ``effective = softplus(raw) + eps == orig`` を満たす.

    per_tensor（要素数 1）と per_group（複数）の両方を同じ関数で扱う。
    入力は float32 に正規化し、device を保持する.

    Args:
        orig_scales: 元のスケールテンソル（1-D、float32 推奨）. 形状は任意.
        zero_mask: ゼログループを示す bool テンソル. ``None`` の場合は
            ``orig_scales == 0`` から自動計算する. 明示的に渡す場合は
            ``orig_scales`` と同 shape であること.
        eps: softplus に加算する微小値. デフォルト 1e-6.

    Returns:
        ``(raw_param, zero_mask)`` のタプル.
        - ``raw_param``: ``torch.nn.Parameter``. 非ゼロ要素は inverse_softplus で初期化、
          ゼロ要素は 0 で初期化（effective が 0 になるようマスクされるため値は任意だが
          再現性のため 0 とする）.
        - ``zero_mask``: bool テンソル（``True`` がゼロ固定グループ）.

    Raises:
        ImportError: torch 未導入時.
        ValueError: NaN/Inf 入力 / shape 不一致 / 空テンソル時.
    """
    _require_torch()
    if orig_scales.numel() == 0:
        raise ValueError("orig_scales must not be empty")
    if not torch.isfinite(orig_scales).all().item():  # type: ignore[union-attr]
        raise ValueError("NaN/Inf detected in orig_scales")
    if eps <= 0:
        raise ValueError(f"eps must be >0, got {eps}")

    # dtype / device を保持しつつ float32 に正規化
    orig = orig_scales.to(torch.float32)  # type: ignore[union-attr]
    device = orig.device

    if zero_mask is None:
        zm = orig == 0
    else:
        if zero_mask.shape != orig.shape:
            raise ValueError(f"zero_mask shape {tuple(zero_mask.shape)} != orig_scales shape {tuple(orig.shape)}")
        # bool へ正規化し device を合わせる
        zm = zero_mask.to(dtype=torch.bool, device=device)  # type: ignore[union-attr]
        # 呼び出し元が渡した zero_mask と orig==0 の整合性は強制しないが、
        # orig==0 の位置は必ず True として扱う（安全側）
        zm = torch.logical_or(zm, orig == 0)  # type: ignore[union-attr]

    # 非ゼログループのターゲット: orig - eps
    # ゼログループはターゲット計算から除外（値は使わない）
    target = orig - eps
    # ゼロ位置の target はダミー値（逆関数に渡さないため任意、0 とする）
    target_safe = torch.where(zm, torch.zeros_like(target), target)  # type: ignore[union-attr]
    # 微小正値クランプは inverse_softplus 内でも行うが、ここでも念のため
    raw_values = inverse_softplus(target_safe)
    # ゼロマスク位置の raw は 0 に置換（effective ではマスクされるため学習に影響しない）
    raw_values = torch.where(zm, torch.zeros_like(raw_values), raw_values)  # type: ignore[union-attr]

    # Parameter としてラップ（requires_grad=True）
    raw_param = Parameter(raw_values)  # type: ignore[arg-type]
    # mypy 対策: Parameter は Tensor のサブクラスだが型チェッカ用に明示
    return raw_param, zm


def get_effective_scales(
    raw_param: torch.Tensor,  # type: ignore[type-arg]
    zero_mask: torch.Tensor,  # type: ignore[type-arg]
    eps: float = 1e-6,
) -> torch.Tensor:  # type: ignore[type-arg]
    """有効スケール ``effective = softplus(raw) + eps`` を取得する.

    ゼロマスクが ``True`` の位置は正確に ``0.0`` を返す（学習対象外）.
    それ以外は ``softplus(raw_param) + eps`` を返す.
    ``scale + eps`` の直接加算ではなく、必ず softplus 経由で正値性を保証する.

    Args:
        raw_param: 学習可能な raw パラメータ（``build_scale_params`` の返り値）.
        zero_mask: bool テンソル（``True`` がゼロ固定）.
        eps: 加算する微小値. ``build_scale_params`` と同じ値を使うこと.

    Returns:
        有効スケールテンソル（float32、同 shape）. ゼロ位置は正確に 0.0.

    Raises:
        ImportError: torch 未導入時.
        ValueError: shape 不一致 / NaN/Inf / eps 非正時.
    """
    _require_torch()
    if eps <= 0:
        raise ValueError(f"eps must be >0, got {eps}")
    if raw_param.shape != zero_mask.shape:
        raise ValueError(f"shape mismatch: raw_param {tuple(raw_param.shape)} vs zero_mask {tuple(zero_mask.shape)}")
    if not torch.isfinite(raw_param).all().item():  # type: ignore[union-attr]
        raise ValueError("NaN/Inf detected in raw_param")

    zm = zero_mask.to(dtype=torch.bool, device=raw_param.device)  # type: ignore[union-attr]
    # softplus は数値安定な正値変換（log(1+exp(x))）
    sp = torch.nn.functional.softplus(raw_param.to(torch.float32))  # type: ignore[union-attr]
    effective = sp + eps
    # ゼロ固定位置は正確に 0
    result = torch.where(zm, torch.zeros_like(effective), effective)  # type: ignore[union-attr]
    if not torch.isfinite(result).all().item():  # type: ignore[union-attr]
        # ゼロマスク位置は 0 なので有限、それ以外で Inf/NaN があれば異常
        raise ValueError("NaN/Inf detected in get_effective_scales result")
    return result


__all__ = [
    "build_scale_params",
    "get_effective_scales",
    "inverse_softplus",
]
