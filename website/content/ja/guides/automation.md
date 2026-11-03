---
description: "スクリプトやCIからOpenTernaryを使う: --jsonの結果、stdoutとstderrの分離、進捗イベント、終了コード、対話できない環境での動作。"
---

# 自動化とスクリプト

OpenTernaryのコマンドはすべて、スクリプト・CI・ほかのプログラムから操作できるように作られています。TUI自身も、
これらの約束事を使う側の1つです。仕様の全体は[出力のリファレンス](../reference/output.md)にあり、このガイドでは
使い方を説明します。

## `--json`: stdoutに結果を1つ {#-json-one-result-on-stdout}

```bash
openternary quantize tiny-llama --device cpu --output runs/ev --json > result.json
```

`--json` を付けると、**stdoutには1行だけ**、バージョン付きのJSONが出ます。人間向けの出力（進捗・表・ログ・
エラー）はすべて**stderr**に出ます。ターミナルでは両方が見えるので2回出たように見えますが、リダイレクトすると
JSONだけが保存されます:

```json
{"schema_version": 1, "command": "quantize", "exit_code": 0,
 "execution": "completed", "artifact_validation": "file_integrity", "quality_acceptance": "not_run",
 "result": {"run_dir": "…/runs/ev", "run_id": "e67150d8", "artifact": "…/runs/ev/artifacts/snapshot", …},
 "error": null}
```

- `exit_code` はプロセスの終了コードと同じです。
- `execution`・`artifact_validation`・`quality_acceptance` は[別々の判定](../concepts/runs-and-evidence.md#four-questions-four-answers)です。
- `result` の中身はコマンドごとに違います。
- `error` は成功なら `null`、失敗ならエラーの文言です。

JSONの内容は、表示言語（`OPENTERNARY_LANG`）に左右されません。

```bash
# シェル: jqで項目を読む
openternary plan ./my-model --json | jq -r '.result.targets | map(select(.quantizable)) | length'
```

```powershell
# PowerShell
$r = openternary doctor --json | ConvertFrom-Json
$r.result.dependencies
```

`--quiet` は進捗の表示を止めますが、結果とエラーは出します。

## 終了コード {#exit-codes}

| コード | 意味 | よくある原因 |
|---|---|---|
| `0` | 成功（操作または計画が完了） | |
| `1` | 想定外の実行時・入出力のエラー | 不具合、ディスクの空き不足 |
| `2` | 設定・引数・入力の誤り | 設定の未知のキー、ファイルがない、`compare` で条件の違うrun |
| `3` | 非対応の機能 | `quantize --device cuda`、APIのバージョンが違うplugin |
| `4` | 必要な依存が入っていない | PyTorchなしでの量子化 |
| `5` | 必要なハードウェア・デバイスがない | `setup` のGPUの試験に失敗 |
| `6` | 明示した受入条件を満たさない／適格な候補がない | `compare --require-acceptance`、`optimize` |
| `7` | 探索の予算が尽きた | `optimize` |
| `130` | 中断 | ++ctrl+c++、TUIでの中止 |

終了コード `6` でも、コマンドは**実行されています**。`execution` は `completed` で、判定が否定的だったという意味です。

## 進捗イベント: `--events-jsonl` {#progress-events-events-jsonl}

```bash
openternary quantize tiny-llama --device cpu --output runs/ev --json --events-jsonl events.jsonl
```

ファイルの1行が1つのイベントです:

```json
{"schema_version": 1, "sequence": 1, "command": "quantize", "run_id": null, "stage": "started", "completed": null, "total": null, "unit": null, "elapsed_s": 0.0, "detail": {}}
{"schema_version": 1, "sequence": 2, "command": "quantize", "run_id": "e67150d8", "stage": "run_created", "completed": null, "total": null, "unit": null, "elapsed_s": 0.004, "detail": {}}
{"schema_version": 1, "sequence": 3, "command": "quantize", "run_id": "e67150d8", "stage": "convert.tensors", "completed": 0, "total": 21, "unit": "tensor", "elapsed_s": 1.776, "detail": {"tensor": "lm_head.weight"}}
…
{"schema_version": 1, "sequence": 8, "command": "quantize", "run_id": "e67150d8", "stage": "completed", "completed": null, "total": null, "unit": null, "elapsed_s": 1.842, "detail": {}}
```

- ファイルは**まだ存在しない**ものでなければなりません。2つの実行が誤って同じファイルに書くのを防ぐためです。
  親のフォルダは必要なら作られます。
- dry-runは何も書かないので、`--events-jsonl` と `--dry-run` は併用できません。
- `sequence` はイベントごとに1ずつ増えます。不明な総量や単位は `null` です。
- 細かい段階は、段階ごとに毎秒5行までに間引きます。ただし、最初の行と `completed == total` の行は必ず書きます。

すべての段階と、その単位・detailの項目は、[進捗イベントのリファレンス](../reference/output.md#progress-events-jsonl)
にあります。長い実行を別のターミナルから追うには:

```bash
tail -f events.jsonl | jq -c '{stage, completed, total, detail}'
```

## 対話できない環境での動作 {#non-interactive-behaviour}

OpenTernaryは、答える人がいないのに入力を待つことはありません:

- **インストールには `--yes` が必要です。** stdinがターミナルでないときや `--json` 付きのとき、`setup` と `update`
  は `--yes` がなければ何も入れません（終了コード `2`）。
- **パイプではTUIを開きません。** 引数なしの `openternary` は、stdinとstdoutの両方がターミナルで、`CI` が設定されて
  いないときだけTUIを開きます。それ以外ではhelpを表示して `2` で終わります。`OPENTERNARY_NO_TUI=1` で、どこでも
  この動作にできます。
- **予期しない通信をしません。** 自動で行う通信は、対話できる `doctor` やTUIでの、1日1回のPyPIのバージョン確認
  だけです。`OPENTERNARY_NO_UPDATE_CHECK=1` で止められます。モデルをダウンロードすることはありません。

## 例: GitHub Actions {#example-github-actions}

モデルを使わない確認は、GPUもPyTorchもモデルのダウンロードもなしに、どのrunnerでも動きます:

```yaml title=".github/workflows/openternary.yml"
jobs:
  plan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v9
      - run: uv tool install openternary --python 3.12
      - run: openternary doctor --json > doctor.json
      - run: openternary calibrate --config configs/my-calibration.yaml --dry-run --json
```

GPUのあるself-hosted runnerのジョブでは、本番の流れを動かし、品質で止めることもできます:

```bash
set -e
openternary quantize "$MODEL" --config quant.yaml --output runs/candidate --json > /dev/null
openternary quality runs/candidate/artifacts/snapshot --data quality.json --device cuda --output runs/candidate-q --json > /dev/null
openternary compare runs/baseline-q runs/candidate-q --require-acceptance --output runs/gate --json > gate.json
# 候補が受け入れられなければ、ここで終了コード6になりジョブが失敗する
```

## Pythonから {#from-python}

同じ機能を、子プロセスなしで使えます。[Python API](../reference/python-api.md)を参照してください:

```python
from openternary.api import load_config, build_plan

config = load_config(cli_overrides={"model.id": "./my-model"})
plan = build_plan(config)
print(sum(t["quantizable"] for t in plan["targets"]), "tensors to quantize")
```
