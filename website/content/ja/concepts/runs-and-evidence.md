---
description: "実行フォルダ、4つの別々の判定（実行・成果物の検査・モデルの再読込・品質の受入）、品質Gate、データのsplit、状態の記号。"
---

# 実行結果と証拠

OpenTernaryは1つの原則に沿って作られています。**数字の価値は、それをどう出したかの記録で決まる。** この
ページでは、その記録の残し方と読み方を説明します。

## 1つの実験に1つのフォルダ {#one-experiment-one-directory}

実際に処理を行うコマンド（`inspect`・`quantize`・`calibrate`・`benchmark`・`quality`・`evaluate`・`compare`・
`optimize`）は、どれも**実行フォルダ（run）**を作ります:

```text
runs/2026-11-03_tiny-llama-naive-g128/
├── config.yaml        # すべて解決済みの設定: CLIのフラグ + YAML + 既定値
├── environment.json   # OpenTernaryのバージョン、gitのcommit、Python、パッケージ、OS、CPU/GPU、メモリ、シード
├── model.json         # モデルIDとrevision
├── metrics.json       # 状態（"completed"・"failed" など）と主な数値
├── <結果>.json        # quantization.json・quality.json・compare.json など（コマンドごとに1つ）
├── logs/openternary.log
└── artifacts/         # モデルのsnapshot、チェックポイント、キャッシュ
```

- **名前:** `--output` を指定しなければ `runs/<日付>_<モデル>-<方式>-g<グループサイズ>` になります。すでに
  あれば `-001`・`-002` … を付けるので、上書きは起こりません。`--output` を指定した場合はそのパスを使い、
  同じ規則で番号を付けます。
- **原子的な書き込み:** JSONは一時ファイルに書いてから名前を変えるので、中断しても書きかけの `metrics.json` が
  残ることはありません。
- **ロック:** 実行中のコマンドは `.execution.lock` を持ちます。同じrunを再開・確定しようとする2つ目のプロセスは
  拒否され、TUIからの削除も拒否されます。
- **失敗も記録します。** runを作ったあとでコマンドが失敗すると、`metrics.json` に `status: failed`（中断なら
  `interrupted`、終了コード130）、エラーの文言、終了コードが入ります。一度完了したrunで、後の再試行が失敗した
  場合は、前の結果を残し、今回の失敗を `last_failure.json` に書きます。

全ファイルは[実行フォルダのリファレンス](../reference/run-directory.md)にあります。

!!! tip "再現性の単位"

    `config.yaml` ＋ `environment.json` ＋ `model.json`（revision付き）＋ソースのチェックアウトの `uv.lock` が
    あれば、runを再現できるように作っています。報告する数字には、これらを必ず添えてください。

## 4つの問いに4つの答え {#four-questions-four-answers}

「うまくいった」には、いくつもの意味があります。OpenTernaryはそれぞれの問いに別々に答え、ある答えで別の答えを
代用することはありません:

| 問い | 項目 | 取りうる値 | それでは*分からない*こと |
|---|---|---|---|
| コマンドは最後まで動いたか | `execution` | `planned`・`completed`・`failed`・`interrupted` | 結果がよいかどうか |
| ファイルは壊れていないか | `artifact_validation` | `not_run`・`file_integrity`・`tensor_integrity` | モデルが読み込めるか |
| モデルを読み込み直せるか | `model_reload`（manifest内） | `not_run` など | 正確かどうか |
| 品質は受け入れられるか | `quality_acceptance` | `not_run`、または `{"accepted": …, "basis": …}` | 測った手順の外のこと |

`--json` の結果には、1つ目・2つ目・4つ目が最上位にあります。量子化が終わった状態は `execution: completed`、
`quality_acceptance: not_run` で、**これが量子化の正常で正直な結果です。** 品質は、測定を比べて初めて分かります。

!!! warning "`completed` は成功ではありません"

    表やTUIでは、`completed` は中立の `●` で表示し、緑のチェックにはしません。`✔` が付くのは、品質の判定が
    受け入れられたときだけです。

## 比較: 観測と受入 {#comparing-observation-vs-acceptance}

`compare` はまず、2つのrunが比べられるものか（手順・データセット・品質のschema・モデルのrevision・dtype・
デバイスが同じか）を確認し、違えば拒否します（終了コード2）。

**ベンチマークの比較は観測です。** smokeベンチマークは、各プロンプトで生成したトークンを正確に記録します。
`compare` は、出力が同じか（`same` / `different`）を、全体とプロンプトごとに報告します。量子化したモデルの出力が
変わるのは当然で、失敗ではありません。それが*悪くなった*かどうかは品質の問題です。

**品質の比較は、事前に決めた固定のGate**（`balanced-quality-gate-v1`）で、受け入れるか拒否するかを判定できます。
候補 \(c\) と、それに対応する基準 \(b\) について:

$$
\Delta_{\text{gen}} = \frac{\text{PPL}^{\text{en}}_c - \text{PPL}^{\text{en}}_b}{\text{PPL}^{\text{en}}_b}, \quad
\Delta_{\text{ja}} = \frac{\text{PPL}^{\text{ja}}_c - \text{PPL}^{\text{ja}}_b}{\text{PPL}^{\text{ja}}_b}, \quad
\Delta_{\text{inst}} = \text{score}_c - \text{score}_b
$$

$$
\text{composite} = 0.35\,(-\Delta_{\text{gen}}) + 0.35\,(-\Delta_{\text{ja}}) + 0.30\,\frac{\Delta_{\text{inst}}}{100}
$$

次のすべてを満たしたときだけ、候補は**受け入れ**られます:

| Gate | 条件 |
|---|---|
| composite | \(> 0\) |
| 一般（英語）のパープレキシティ | 上昇が2%以内 |
| 日本語のパープレキシティ | 上昇が2%以内 |
| 指示スコア | 低下が2ポイント以内 |
| 崩壊 | 崩壊したケースが0 |

このGateは**相対的**です。候補は、自分の基準と比べて判定されます。また、証拠が揃っているときにだけ適用します:

- 両方のレポートが品質レポートの**schema 3**である
- 元モデルの同一性が確定していて、重みとトークナイザ・processorの内容のfingerprintが同じである
- 測定環境（デバイス・dtype・ライブラリのバージョン・OS）が同じである
- モデルが本物である: 人工のものとして印の付いたモデル（`openternary-fixture.json`）は決して受け入れない

それ以外の場合、判定は `accepted: false`、`status: insufficient_evidence` になります。
`compare --require-acceptance` を付けると、受け入れられなかった場合はすべて終了コード**6**になります。閾値は
コードと研究計画で固定されています。変えるのは研究上の判断であって、設定の項目ではありません。

## データのsplit {#data-splits}

品質のデータセットには、重ならない3つのsplitがあります:

| split | 用途 |
|---|---|
| `calibration` | 合わせ込み（校正のデータ。評価には使わない） |
| `validation` | 候補の選択（`quality` の既定） |
| `test` | 選んだ候補の最終的な、1回だけの測定 |

OpenTernaryは、同じ文章が2つのsplitに現れないことを、正規化後の完全一致と、ほぼ重複（文字5-gramのMinHashで
類似度0.9以上）の両方で確認し、重なりのあるデータセットを拒否します。`optimize` がtestのsplitで候補を選ぶことは
ありません。

## 分からないことは0ではない {#unknown-is-not-zero}

測っていないものは、結果に `0` ではなく `null` と書きます。CPUでのrunのVRAMのピーク、測定条件が違う場合の速度、
量子化しただけのモデルの品質は、どれも `null` か `not_run` です。速度の順位は、測定条件が同じ場合にだけ出します。

## 状態の記号 {#status-symbols}

人間向けの出力では、記号と語を組み合わせて表示します。色は記号にだけ付けるので、コピーしても、色の見分けが
難しくても意味が伝わります。

| 記号 | 意味 | 例 |
|---|---|---|
| `✔` | 受入・合格 | 品質の受入、デバイスの試験の合格 |
| `●` | 導入済み・完了 | パッケージが入っている、コマンドが最後まで動いた |
| `◐` | 実装済みだが未検証 | 変換は実装済みで、実モデルでは未検証 |
| `▲` | 実験的・注意 | backendの品質は実験的、更新がある |
| `○` | 未実行・未導入 | 品質は未評価、任意のパッケージがない |
| `–` | 非対応 | ternary backendのnative低bit実行 |
| `✖` | 失敗・拒否 | 失敗したコマンド、拒否された品質の比較 |
