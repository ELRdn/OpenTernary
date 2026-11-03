---
description: "--jsonの結果、終了コード、進捗イベント（JSONL）とすべての段階、人間向けの出力の決まり。"
---

# 出力・終了コード・イベント

OpenTernaryの機械向けの約束事です。バージョンが付いていて、スクリプトと同じようにTUIもこれに依存しています。
使い方は[自動化](../guides/automation.md)で説明しています。

## `--json` の結果 {#the-json-envelope}

`--json` を付けると、stdoutにはJSONが1つだけ、1行で出ます。人間向けの出力はstderrに出ます。

```json
{
  "schema_version": 1,
  "command": "plan",
  "exit_code": 0,
  "execution": "planned",
  "artifact_validation": "not_run",
  "quality_acceptance": "not_run",
  "result": {},
  "error": null
}
```

| 項目 | 型 | 値 |
|---|---|---|
| `schema_version` | int | `1` |
| `command` | 文字列 | コマンド（例: `quantize`・`cache remove`） |
| `exit_code` | int | プロセスの終了コード（下記） |
| `execution` | 文字列 | `planned`（計画・dry-runの成功）、`completed`（終了コード0、または6: 実行されたが判定が否定的）、`interrupted`（130）、`failed`（それ以外） |
| `artifact_validation` | 文字列 | `not_run`・`file_integrity`・`tensor_integrity` |
| `quality_acceptance` | 文字列またはobject | `not_run`、`compare` では `{"accepted": false, "basis": "validated quality reports only"}` のようなobject |
| `result` | object | コマンドごとの内容: パス、run ID、指標、レポート |
| `error` | 文字列またはnull | 成功なら `null`、失敗ならエラーの文言（エラーのパネルと同じ文） |

`execution: completed` が品質の受入を意味することはなく、`artifact_validation: file_integrity` がモデルを読み込める
ことを意味することもありません。内容は表示言語に左右されません。

## 終了コード {#exit-codes}

| コード | 意味 |
|---|---|
| `0` | 操作または計画が成功 |
| `1` | 想定外の実行時・入出力のエラー |
| `2` | 設定・引数・入力の誤り、または不在 |
| `3` | 非対応の機能、またはpluginのAPIの不一致 |
| `4` | 必要な依存が入っていない |
| `5` | 必要なハードウェア・デバイスがない |
| `6` | 明示した受入条件を満たさない、または適格な候補がない |
| `7` | 探索の予算が尽きた |
| `130` | ユーザーによる中断 |

## 進捗イベント（JSONL） {#progress-events-jsonl}

`--events-jsonl FILE` は、新しいファイルに1行1つのJSONを書きます:

```json
{"schema_version": 1, "sequence": 3, "command": "quantize", "run_id": "e67150d8",
 "stage": "convert.tensors", "completed": 0, "total": 21, "unit": "tensor",
 "elapsed_s": 1.776, "detail": {"tensor": "lm_head.weight"}}
```

| 項目 | 型 | 意味 |
|---|---|---|
| `schema_version` | int | `1` |
| `sequence` | int | 1つの操作の中で、イベントごとに1ずつ増える |
| `command` | 文字列 | コマンド |
| `run_id` | 文字列またはnull | runができるまでは `null` |
| `stage` | 文字列 | 下の表を参照 |
| `completed`・`total` | intまたはnull | 段階の中での進捗。不明なら `null` |
| `unit` | 文字列またはnull | 数えている単位（`tensor`・`step`・`batch` など） |
| `elapsed_s` | 数値 | 操作の開始からの秒数 |
| `detail` | object | 段階ごとの追加の項目。常にobject。有限でない数値は `null` |

細かい段階は、**段階ごとに毎秒5行まで**に間引きます。段階の最初の行と、`completed == total` の行は必ず書きます。
ファイルは事前に存在していてはならず、`--events-jsonl` は `--dry-run` と併用できません。

### 段階 {#stages}

| 段階 | 単位 | `detail` | 出すもの |
|---|---|---|---|
| `started` / `completed` / `failed` / `interrupted` | | | すべてのコマンド |
| `run_created` | | | 実行フォルダを作ったとき |
| `phase` | | `name`・`device` | 資源の計測の区切り（load・inferenceなど） |
| `convert.tensors` | tensor | `tensor` | `quantize`、探索の候補 |
| `convert.flush` | | `shard`・`tensors` | shardをディスクに書き始めた |
| `convert.hash` | shard | | 書いたshardのハッシュ計算 |
| `calibrate.dataset` | | `dataset`・`samples`・`seq_len` | 校正の文章とトークナイザの準備（初回はダウンロードを含む） |
| `calibrate.load_teacher` | | `snapshot` | 元のモデルの読み込み |
| `calibrate.cache` | | `status`（`capturing` / `reused`）・`source` | 活性値のキャッシュ |
| `capture.batches` | batch | `split`（`train` / `heldout`） | teacherの入出力の記録 |
| `calibrate.prepare` | | `modules` | キャッシュの記録、teacherの解放、モジュールの状態の準備 |
| `calibrate.modules` | module | `module`、完了時は `last_loss` | 対象の各層の開始と終了 |
| `calibrate.steps` | step | `module`・`module_index`・`modules`・`loss` | 層の中の最適化のstep |
| `calibrate.checkpoint` | | `checkpoint`・`cursor` | チェックポイントを保存した |
| `calibrate.evaluate` | module | `module` | 学習用・ホールドアウトの再構成誤差 |
| `calibrate.materialize` | | `destination` | 校正済みsnapshotの書き出し |
| `materialize.tensors` | tensor | `tensor` | …テンソルごと |
| `materialize.flush` | | `shard`・`tensors` | …shardを書き始めた |
| `materialize.hash` | shard | | …ハッシュ計算 |
| `calibrate.report` | | | 指標と `calibration.json` の作成 |
| `benchmark.prompts` | prompt | `prompt` | `benchmark` |
| `quality.perplexity` | language | `language` | `quality` |
| `quality.cases` | case | `case` | `quality` |
| `export.tensors` | tensor | `tensor` | packedへの書き出しと復元 |
| `export.converter` | | `status`・`converter` | 外部のGGUFのconverter |
| `validate.tensors` | tensor | `tensor` | `artifacts validate --level tensors` |
| `candidate_started` | | | `optimize`: 候補の開始 |
| `search.candidates` | candidate | `status`・`folder` | `optimize`: 候補の終了 |
| `diffusion.prompts` | prompt | | diffusionの評価 |

探索の候補は別のプロセスで動くので、候補内部の `convert.*` のイベントは親のファイルには出ません。

`schema_version: 1` のまま、項目や段階が追加されることがあります。既存の項目の意味は変わりません。読む側は、
知らないものを無視してください。

## 人間向けの出力 {#human-readable-output}

`--json` がないとき:

- `doctor`・`backends`・`plan`・`plugins`・`artifacts validate`・`cache` と、各ワークフローのコマンドの結果は
  表で表示します。
- 状態は記号と語で表し、色は記号にだけ付けます。[状態の記号](../concepts/runs-and-evidence.md#status-symbols)を
  参照してください。
- エラーは**1回だけ**、stderrにパネルで表示します。何が失敗したか、詳細、あればヒント、終了コードです。
- 言語は `OPENTERNARY_LANG=en|ja`、指定がなければOSの言語です。ターミナルの文字コードで日本語を表示できなければ
  英語になります。翻訳するのはhelpの説明、パネルの見出し、表の見出し、エラーの見出しで、`--json` の内容は翻訳
  しません。
- `NO_COLOR` で色を消せます。
