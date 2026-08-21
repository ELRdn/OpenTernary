"""ML dependencies smoke — Pillow/torchvision required for Gemma4Processor."""

from __future__ import annotations


def test_gemma4_runtime_dependencies_available() -> None:
    """Pillow / torchvision が ml extra で導入されていることを検証.

    Gemma4Processor は PIL と torchvision に依存し、欠落すると
    「Gemma4Processor requires the PIL library」で失敗する。
    このテストは実モデルDL不要で import 可否のみを担保する。
    """
    # torch は optional だが ml extra では必須
    import torch  # noqa: F401  # type: ignore[import-not-found]
    import torchvision  # noqa: F401  # type: ignore[import-not-found]
    from PIL import Image  # noqa: F401  # type: ignore[import-not-found]
    from transformers import Gemma4Processor  # noqa: F401  # type: ignore[import-not-found]

    assert Image is not None
    assert torchvision is not None
    assert Gemma4Processor is not None
