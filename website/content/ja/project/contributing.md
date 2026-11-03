---
description: "開発環境の用意、CIと同じチェック、ドキュメントの編集、コードや研究のコントリビュートの方法。"
---

# コントリビュート

OpenTernaryは研究プロジェクトで、**機能の数より再現性を優先します**。コントリビュートは、プロジェクトのライセンス
（Apache-2.0、第5条）のもとで受け付けます。詳しい指針は
[CONTRIBUTING.md](https://github.com/ELRdn/OpenTernary/blob/main/CONTRIBUTING.md)にあります。

## 開発環境 {#development-setup}

```bash
git clone https://github.com/ELRdn/OpenTernary.git
cd OpenTernary
uv sync                                   # 本体と開発ツール（uv.lockどおり）
uv sync --extra ml --extra calibration    # 変換・評価・校正を扱うとき
```

リポジトリはuvのバージョン（`pyproject.toml` の `[tool.uv] required-version`）を固定しているのでそれを使い、
`.python-version` はPython 3.12に固定しています。`uv.lock` は手で編集せず、`uv add` / `uv lock` を使ってください。

## CIと同じチェック {#the-checks-ci-runs}

プルリクエストを出す前に実行してください:

```bash
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv run --frozen mypy src/openternary
uv run --frozen pytest
```

CIでは、さらに次のものも実行します:

- **本体だけ**のジョブ: PyTorchなしで成り立つべき約束事
- `python scripts/validate_cli_offline.py`: 空のHugging Faceキャッシュ、GPUなし、オフラインの設定での全試験（ログは
  `.tmp/`）
- `scripts/validate_cli_real.py --require-wheel`: 導入したwheelを、本物のライブラリとシード固定の人工モデルで、
  新しいプロセスで試験（LinuxとWindows）
- `scripts/validate_cli_plugins.py`: 衝突やAPIの不一致を含む、本物のentry pointのplugin
- uvを使わない `pip install -e .`（pipでの導入が動くことの確認）

## ドキュメントの編集 {#working-on-the-documentation}

このサイトは、`website/` と `mkdocs.yml` から [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
で作っています:

```bash
uv sync --group docs
uv run --frozen mkdocs serve          # http://127.0.0.1:8000（保存すると再読込）
uv run --frozen mkdocs build --strict # CIが公開するもの。警告や壊れたリンクがあれば失敗
```

- 英語のページは `website/content/en/`、日本語のページは `website/content/ja/` にあり、**ファイル名は同じ**です。
  同じプルリクエストで両方を変えてください。
- CLIのコマンド・オプション・設定のキーがどちらかの言語のリファレンスから漏れていたり、2つの言語でページの構成が
  違ったりすると、`tests/test_docs.py` が失敗します。
- `website/content/assets/images/` のTUIのスクリーンショットは、実際の画面からTextualで書き出したSVGです。手で
  編集せず、作り直してください。
- `main` にpushすると、GitHub Pagesにサイトが公開されます（`.github/workflows/docs.yml`）。

## プルリクエスト {#pull-requests}

プルリクエストには、何の問題を解決するか、なぜその方法がよいかを書き、試験、ベンチマークへの影響（あれば）、
互換性への影響、インターフェースを変えた場合はドキュメントと設定の更新を含めてください。関係のないリファクタリングは
避けてください。

**研究のコントリビュート**（新しい量子化や校正の方式）には、元の文献や参照先、実装した考え方そのもの、元との違い、
基準、設定、評価のデータ、結果、限界を含めてください。都合のよいプロンプト1つを根拠に、方式が優れていると示すことは
しません。また、元と照らし合わせて確認していない論文やフレームワークを「再現した」とは書きません。

明示的に小さなfixtureとして定めたもの以外の、モデルの重み、大きなチェックポイント、校正やベンチマークのキャッシュ、
packedのモデルのバイナリは**コミットしないでください**。

## 不具合の報告 {#reporting-bugs}

[issue](https://github.com/ELRdn/OpenTernary/issues)に次の内容を書いてください:

```text
OpenTernaryのバージョン（openternary --version）:
モデルとrevision:
コマンドと設定:
ハードウェア・OS・Python:
`openternary doctor --json` の出力:
エラーのパネル全体:
最小の再現手順:
```

## リリース {#releasing}

メンテナーは、PyPIのTrusted Publishingでリリースします。
[RELEASING.md](https://github.com/ELRdn/OpenTernary/blob/main/RELEASING.md)を参照してください。
