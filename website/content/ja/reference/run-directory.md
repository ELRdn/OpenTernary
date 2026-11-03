---
description: "OpenTernaryが実行フォルダと成果物に書くすべてのファイルと、その中身。"
---

# 実行フォルダ

## 構成 {#layout}

```text
runs/<名前>/
├── config.yaml            常に     すべて解決済みの設定
├── environment.json       常に     バージョン、gitのcommit、Python、OS、ハードウェア、シード、設定
├── model.json             常に     {"id": …, "revision": …}
├── metrics.json           常に     状態、run_id、主な結果
├── last_failure.json      場合による  一度完了したrunの、失敗した再試行
├── .execution.lock        実行中   runを担当するプロセスが持つ
├── logs/
│   └── openternary.log
├── artifacts/             モデルのsnapshot、チェックポイント、キャッシュ
└── <結果のファイル>       コマンドごとに1つ（下記）
```

### 名前の付け方 {#naming}

| `--output` | フォルダ |
|---|---|
| 指定なし | `runs/<YYYY-MM-DD>_<モデル名>-<方式>-g<グループサイズ>`（例: `runs/2026-11-03_tiny-llama-naive-g128`） |
| 指定あり | そのパス |

フォルダがすでにあれば、`-001`・`-002` … を付けます。既存のフォルダにrunを書くことはありません。例外は復旧の
コマンド（`calibrate --resume`・`--finalize`・`--materialize-only`）で、これらはあえて `--output` を使い回します。
複数のプロセスが同時に始めても名前が衝突しないよう、`runs/.run.lock` というロックファイルを使います。

## 共通のファイル {#common-files}

### `config.yaml` {#configyaml}

実際に使った設定で、フラグ・YAML・既定値を合わせたものです。すべてのキーを含むので、`--config` に渡せばrunを
繰り返せます。[設定](configuration.md)を参照してください。

### `environment.json` {#environmentjson}

| キー | |
|---|---|
| `run_id` | 16進数8文字。runごとに一意 |
| `timestamp` | 開始時刻（UTC、ISO 8601） |
| `openternary_version` | 例: `0.1.0a2` |
| `python_version`・`python_executable` | 実行したインタープリタ |
| `uv_lock_hash` | 作業フォルダの `uv.lock` のSHA-256の先頭16桁、なければ `unknown` |
| `git_commit`・`git_branch`・`code_provenance` | gitのチェックアウトから実行した場合のソースのrevision |
| `seed`・`PYTHONHASHSEED` | シード |
| `config` | 設定（`config.yaml` と同じ） |
| `hardware` | プラットフォーム、プロセッサ、メモリ（GB）、GPUの名前、CUDA/ROCmが使えるか |

### `metrics.json` {#metricsjson}

常に `status` と `run_id` があります。

| `status` | 意味 |
|---|---|
| `not_run` | 作っただけで、まだ何も終わっていない |
| `completed` | コマンドが最後まで動いた（品質の判定ではない） |
| `failed` | 失敗した。`error_message` と `exit_code` が加わる |
| `interrupted` | 中断された（終了コード130） |

完了したrunには、そのコマンドの主な数値（量子化の要約や校正の損失など）が加わります。一度完了したrunを再試行して
失敗した場合、`metrics.json` は前の結果を残し、今回の失敗は `last_failure.json`（失敗した `metrics.json` と同じ項目）
に書きます。

## コマンドごとの結果のファイル {#result-files-by-command}

| コマンド | 結果のファイル | 主な内容 |
|---|---|---|
| `inspect` | `inspection.json` | アダプタ、構造、全テンソル（`name`・`shape`・`dtype`・`param_count`・`role`・`quantizable`・`exclude_reason`）、要約、メモリの見積もり、fingerprint |
| `quantize` | `quantization.json` | 設定、テンソルごとの統計、グループの総数、contentのfingerprint、ビットの集計（`packed_estimate`） |
| `calibrate` | `calibration.json` | 方式と設定、層ごとの損失、学習用・ホールドアウトの結果、汚染の確認、VRAMのピーク、データの来歴 |
| `benchmark` | `benchmark.json` | suiteとprotocolのfingerprint、プロンプトごとのトークンと文章、結果のfingerprint、レイテンシ、トークン毎秒、資源 |
| `quality` | `quality.json` | レポートschema 3: 同一性、手順、データセットのfingerprintと監査、言語ごとのパープレキシティ、指示の結果、`summary`、資源、測定環境 |
| `evaluate` | `evaluation.json` | profile、画像（ハッシュ・時間・画素の統計）、`human_review_required` |
| `compare` | `compare.json`・`acceptance.json` | 一致の確認、プロンプトごとの差分または品質の変化、判定 |
| `optimize` | `search.json`（＋journal、候補のフォルダ） | すべての候補の測定値と適格性、結果 |

`quality.json` の `summary` には、Gateが使う4つの数値、`general_ppl`・`japanese_ppl`・`instruction_score`・
`collapse_count` があります。

## 成果物 {#artifacts}

成果物は、manifest付きのモデルのフォルダです。runの中では `artifacts/snapshot`（`quantize`）または
`artifacts/calibrated_snapshot`（`calibrate`）にあります。`export` の出力は、runではなく独立した成果物のフォルダです。

```text
<成果物>/
├── openternary-manifest.json     形式、schema、来歴、全ファイルのSHA-256
├── config.json・トークナイザのファイル・chat template・generation_config.json   元からコピー
├── openternary-fixture.json      人工のテストモデルにだけある。品質の証拠としては決して受け入れない
│
│   safetensorsのsnapshot:
├── model-00001-of-0000N.safetensors
├── model.safetensors.index.json
├── quantization.json
│
│   ternary-packed:
├── packed.json                   符号・scale・形・詰め物の配置
└── tensors/000000.safetensors …  テンソルごとに1ファイル
```

manifestには、`schema_version`・`format`・`execution`・`artifact_validation`・`model_reload`・`quality_acceptance`、
元とそのインターフェース（トークナイザとprocessorのファイル）、設定、backend、アダプタ、component、pass、コードの
ハッシュ、全ファイルのハッシュと大きさ、それまでの変換の履歴が記録されます。`openternary artifacts validate` は、
成果物をこれと照合します。

## 校正で増えるもの {#calibration-extras}

```text
artifacts/
├── calibrated_snapshot/   結果
├── checkpoint/            module_00000.pt など: 再開用の状態（ハッシュのmanifest付き）
├── calibration_state/     層ごとの学習済みパラメータ（safetensors）
├── activation_cache/      記録したteacherの活性値（activation_cache_train/・activation_cache_held/ も）
└── threshold_metrics.jsonl
```

活性値のキャッシュはモデルよりずっと大きくなることがあります。runを再開したり、キャッシュを再利用したりする必要が
なくなったら削除して構いません。
