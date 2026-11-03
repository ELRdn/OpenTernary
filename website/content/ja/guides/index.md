---
description: "OpenTernaryの作業ごとのガイド。"
---

# ガイド

各ガイドは、1つの作業を最初から最後まで、実際のモデルを想定して説明します。OpenTernaryを初めて使う場合は、
先に[はじめての実行](../getting-started/first-run.md)を試してください。小さなモデルで全ステップを確認できます。

## 基本の流れ {#the-main-workflow}

| 手順 | ガイド | コマンド |
|---|---|---|
| 1 | [調査と計画](inspect-and-plan.md): モデルを用意し、量子化できる部分を確認する | `inspect`・`plan` |
| 2 | [量子化](quantize.md): 三値のsnapshotを作る | `quantize` |
| 3 | [校正](calibrate.md): 品質の回復を試みる | `calibrate` |
| 4 | [評価と比較](evaluate-and-compare.md): 測定して判定する | `benchmark`・`quality`・`compare` |
| 5 | [書き出し](export.md): packed化・復元・GGUF | `export` |

## さらに進んだ使い方 {#beyond-the-basics}

| ガイド | 用途 |
|---|---|
| [Diffusionモデル](diffusion.md) | Diffusersのpipelineのcomponentを量子化し、画像で評価する |
| [制約付き探索](optimize.md) | 時間・VRAM・ディスクの予算内で、複数の設定を試す |
| [成果物の管理](artifacts.md) | 成果物の検査、一覧、安全な削除 |
| [自動化とスクリプト](automation.md) | `--json`・進捗イベント・終了コード・CI |
| [トラブルシューティング](troubleshooting.md) | よくあるエラーとその意味 |
