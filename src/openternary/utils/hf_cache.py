"""HF cache-only snapshot解決.

- ローカルpathが存在すればそれを優先
- HF repo IDなら huggingface_hub.snapshot_download(local_files_only=True) でcacheのみ解決
- ネットワークDLは行わない
"""

from __future__ import annotations

import pathlib


def resolve_snapshot(
    model_id_or_path: str,
    revision: str | None = None,
) -> pathlib.Path:
    """model_id_or_pathをsnapshot Pathに解決.

    Args:
        model_id_or_path: ローカルsnapshot path または HF repo ID (例 google/gemma-4-E2B-it-qat-q4_0-unquantized)
        revision: HF revision (commit hash). repo ID時のみ使用

    Returns:
        snapshot directory Path (config.json, model.safetensorsを含む)

    Raises:
        FileNotFoundError: 見つからない場合
        ImportError: HF ID解決にhuggingface_hubが必要だが未導入
    """
    # 1) ローカルpath優先
    p = pathlib.Path(model_id_or_path)
    # 環境により HF cacheがsymlink構成のため、存在判定は厳密に
    if p.exists():
        # snapshot dirかファイルかを判定: dirならそのまま、fileなら親
        if p.is_dir():
            _validate_snapshot(p)
            return p
        # 単一ファイル指定なら親dirを返す (稀)
        parent = p.parent
        if parent.exists():
            _validate_snapshot(parent)
            return parent
        return p

    # 2) HF repo IDとしてcache-only解決
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            f"huggingface-hub is required to resolve HF repo ID '{model_id_or_path}'. "
            f"Install with: uv sync  (huggingface-hub is a core dependency) or pass a local snapshot path."
        ) from e

    try:
        snapshot_path_str: str = snapshot_download(
            repo_id=model_id_or_path,
            revision=revision,
            local_files_only=True,
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Local snapshot not found for '{model_id_or_path}'"
            + (f"@{revision}" if revision else "")
            + f". Run: hf download {model_id_or_path}"
            + (f" --revision {revision}" if revision else "")
            + f"  Original error: {e}"
        ) from e

    snapshot_path = pathlib.Path(snapshot_path_str)
    _validate_snapshot(snapshot_path)
    return snapshot_path


def _validate_snapshot(snapshot: pathlib.Path) -> None:
    """snapshotに必要なファイルがあるか検証."""
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Snapshot path is not a directory: {snapshot}")
    # config.json と model.safetensors のいずれかの存在をチェック (safetensorsは複数形も許容)
    has_config = (snapshot / "config.json").exists()
    has_safetensors = (snapshot / "model.safetensors").exists() or any(snapshot.glob("*.safetensors"))
    if not has_config:
        raise FileNotFoundError(f"Snapshot missing config.json: {snapshot}")
    if not has_safetensors:
        raise FileNotFoundError(f"Snapshot missing model.safetensors: {snapshot}")
