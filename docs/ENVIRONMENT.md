# Environment — uv と pip の使い分け

OpenTernary は **uv を推奨**しつつ、pip フォールバックも保証する。

## 推奨: uv（再現性重視）

```bash
git clone <repo>
cd OpenTernary
uv sync
uv run openternary --help
uv run --frozen pytest
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy src/openternary
```

- `.python-version` に `3.12` を pin（`uv python pin 3.12` で変更）
- `requires-python = ">=3.11"` は広め、CI は 3.11/3.12 マトリクスで保証
- `uv.lock` は Python 依存の正本。`1 experiment = 1 directory = 1 reproducibility unit` の一部として `environment.json` とセットで保存
- `uv run` は `.venv` の activate を不要にする（DSH/Codex 並行開発で有効）

### よくある操作

```bash
uv add <package>          # 依存追加（pyproject.toml を手動編集しない）
uv sync --locked          # CI 用、lock の整合性検証
uv python pin 3.12        # Python バージョン変更（メンテ時のみ）
uv cache clean            # キャッシュ修復
```

## Fallback: pip

uv が使えない環境でも動作する（PEP 517 準拠）:

```bash
pip install -e .[dev]  # dev は dependency-groups のため pip では別途 pip install pytest ruff mypy が必要
pip install -e .
openternary --help
pytest
```

> Note: `dependency-groups` は pip では直接読まれないため、dev 依存は手動で入れる。研究再現性が必要なら uv を使う。

## トラブルシューティング

### Windows で `uv cache` の `.lock` / `sdists-v9\.git` がアクセス拒否

```powershell
uv cache clean
# ダメなら
Remove-Item -Force $env:LOCALAPPDATA\uv\cache\.lock
Remove-Item -Recurse -Force $env:LOCALAPPDATA\uv\cache\sdists-v9\.git
# 最終手段: プロジェクト固有キャッシュ
$env:UV_CACHE_DIR = "C:\path\to\OpenTernary\.uv-cache"
uv sync
```

`.uv-cache` を常設しない理由: 各 repo で wheel を再 DL することになり uv の利点を減らすため。プロジェクト固有キャッシュは fallback のみ。

### uv バージョン齟齬

`pyproject.toml` の `[tool.uv] required-version = "==0.12.5"` に合わない uv で実行するとエラーになる。正しいバージョン（現在 0.12.5）を `astral-sh/setup-uv` などで用意する。

### torch / ML 依存 (Phase 1)

Phase 1 では以下で確定:

```toml
[project]
dependencies = [ ..., "huggingface-hub>=0.23" ]  # header-onlyでもHF ID cache-only解決

[project.optional-dependencies]
ml = ["torch>=2.5", "safetensors>=0.4"]  # --load-weightsのみ
```

* `uv sync` → `huggingface-hub` は入るが `torch` は入らない。`openternary inspect` は header-onlyでGPU不要・payload非ロード。
* `uv sync --extra ml` → `torch` + `safetensors` が入り `--load-weights` でtensor単位streaming統計。
* `transformers`/`accelerate` は Phase 1b/2で追加 (モデルinstantiateが必要になってから)。
* CPU/CUDA/ROCm の index 切替はPhase 2で設計。Phase 1の `uv sync` は数GBを強制しない。

## CI

```yaml
- uses: astral-sh/setup-uv@<sha> # v9.0.0
  with: { python-version: ${{ matrix.python-version }}, enable-cache: true }
- run: uv sync --locked
- run: uv run --frozen ruff check .
- run: uv run --frozen mypy src/openternary
- run: uv run --frozen pytest -q
```
