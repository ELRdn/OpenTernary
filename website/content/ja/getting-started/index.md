---
description: "何もない状態から、OpenTernaryを入れて最初の三値モデルを作るまでの最短経路。"
---

# はじめに

この章では、何もない状態から、量子化・書き出し・評価まで済んだモデルを作るところまで進みます。

<div class="grid cards" markdown>

-   **1. [インストール](installation.md)**

    ---

    `openternary` コマンドを入れ、`openternary setup` にGPUに合うPyTorchを選ばせ、`openternary doctor` で
    確認します。更新やソースからの導入もここで説明します。

-   **2. [はじめての実行](first-run.md)**

    ---

    CPUで小さなモデルを作り、調査・量子化・packed化・復元・検査・評価・比較の全工程を1分ほどで動かします。
    ダウンロードもGPUも不要です。

-   **3. [全画面インターフェース](tui.md)**

    ---

    同じコマンドをTUIから使います。`❯` のプロンプトからコマンドを選び、計画を確認し、ジョブの進捗とETAを
    リアルタイムで見られます。

</div>

## 必要なもの {#what-you-need}

| | 最低限 | 補足 |
|---|---|---|
| Python | 3.11 | 3.12を推奨。リリースは主に3.12で検証しています。WindowsのROCmビルドは3.12と3.13にしかありません。 |
| OS | Windows・Linux・macOS | WSLでも動きます。 |
| [uv](https://docs.astral.sh/uv/) | 最近のもの | `openternary setup` と `update` に必要です。`pip` でも入れられますが、`setup` は内部でuvを使います。 |
| GPU | 任意 | NVIDIA（CUDA）・AMD（ROCm）・Apple silicon（MPS）。量子化はCPUで行います。GPUがあると評価と校正が速くなります。 |
| ディスク | 小さなモデルなら数GB | 三値のfake-quant snapshotは元のモデルと同じ大きさです。校正では活性値のキャッシュとチェックポイントが増えます。 |

!!! tip "CLIに慣れている方へ"

    どのコマンドも `--help` に対応しています。全オプションは[CLIリファレンス](../reference/cli.md)に、`--json`・
    進捗イベント・終了コードは[自動化](../guides/automation.md)にまとめています。
