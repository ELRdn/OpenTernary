---
description: "層ごとの校正を実行し、方式を選び、中断から再開し、チェックポイントからmaterializeし、runの記録を復旧する。"
---

# 校正

`calibrate` は、三値化で失われた品質の回復を試みます。各層のパラメータを調整し、その出力を元のモデルの出力に
合わせます。仕組みと、期待できる結果については、先に[校正の考え方](../concepts/calibration.md)を読んでください。

## 最小の設定 {#a-minimal-configuration}

校正はYAMLで設定します。次の小さな設定ファイルは、[はじめての実行](../getting-started/first-run.md)の小さな
モデルと人工データを使った、仕組みの確認用です:

```yaml title="calib-smoke.yaml"
model:
  id: tiny-llama          # フォルダ、またはキャッシュにあるHugging FaceのID
  revision: null
quantization:
  group_size: 64
  scale_granularity: per_group
calibration:
  enabled: true
  method: recon-scale
  dataset: synthetic      # 仕組みの確認用。研究の証拠にはならない
  num_samples: 8
  seq_len: 32
  steps: 5
  checkpoint_interval: 5
device: cpu
dtype: fp32
```

```bash
openternary calibrate --config calib-smoke.yaml --dry-run          # 確認。何も作らない
openternary calibrate --config calib-smoke.yaml --output runs/calib-smoke
```

数秒で終わり、次のように表示されます:

```text
    Calibrate — tiny-llama
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric          ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Dataset         │ synthetic │
│ Steps           │ 5         │
│ Initial loss    │ 0.009966  │
│ Final loss      │ 0.009901  │
│ Held-out before │ 0.009592  │
│ Held-out after  │ 0.009533  │
│ Scale delta     │ 0.000074  │
│ Contamination   │ False     │
└─────────────────┴───────────┘
```

`Initial loss`・`Final loss` は学習の再構成損失、`Held-out before/after` は学習に使わなかったサンプルでの同じ
損失です。`Contamination: False` は、学習用とホールドアウトのサンプルが重なっていないことを確認したという意味です。

## Gemma 4 E2Bでの本番の校正 {#real-calibration-on-gemma-4-e2b}

研究の設定では、WikiTextと固定したGemmaのrevisionを使います。`datasets` パッケージ（`openternary setup` が
入れます）が必要で、WikiTextは初回に自動でダウンロードされます。すぐに使えるファイルがリポジトリの
[`configs/`](https://github.com/ELRdn/OpenTernary/tree/main/configs) にあります:

| ファイル | 用途 |
|---|---|
| [`gemma4-e2b-calibration.yaml`](https://github.com/ELRdn/OpenTernary/blob/main/configs/gemma4-e2b-calibration.yaml) | Phase 4.1の受入: `recon-scale`、G128、`wiki-tiny`、32サンプル×128トークン、層ごとに50 step |
| [`gemma4-e2b-calibration-smoke.yaml`](https://github.com/ELRdn/OpenTernary/blob/main/configs/gemma4-e2b-calibration-smoke.yaml) | 同じモデルと対象で8×64トークン、2 step。仕組みと進捗表示の確認用で、品質の確認ではない |
| [`gemma4-e2b-threshold.yaml`](https://github.com/ELRdn/OpenTernary/blob/main/configs/gemma4-e2b-threshold.yaml) | `recon-threshold`: 閾値の学習 |

```bash
openternary calibrate --config configs/gemma4-e2b-calibration.yaml --device cuda --output runs/e2b-recon-scale
```

`--device cuda` はNVIDIAでもAMD（ROCm）でも使えます。

### preflight {#preflight}

研究対象にGPUを何時間も使う前に、確認しておきましょう:

```bash
openternary calibrate --config configs/gemma4-e2b-calibration.yaml --preflight --output runs/e2b-preflight
```

`--preflight` は重い読み込みをしません。元モデルが固定したrevisionであること、重みがあって記録したSHA-256と
一致すること、正準の対象層の集合（Gemma 4 E2Bでは205）が見つかること、手順のファイルと資源の予算が揃っている
ことを確認します。`preflight.json` を書き、合格なら `0`、不合格なら失敗した項目を示して `2` で終わります。

!!! note

    preflightは**正準の研究対象**（固定したrevisionのGemma 4 E2B）と照合します。それ以外のモデルでは、
    `revision_pinned` や `canonical_target_count` などが不合格になりますが、これは想定どおりです。ほかの
    モデルの確認には `--dry-run` を使ってください。

## 方式を選ぶ {#choosing-a-method}

`calibration.method` を設定します（[3つの方式](../concepts/calibration.md#the-three-methods)を参照）:

=== "recon-scale"

    ```yaml
    calibration:
      enabled: true
      method: recon-scale
    ```

=== "recon-threshold"

    ```yaml
    calibration:
      enabled: true
      method: recon-threshold
      threshold_enabled: true          # recon-threshold では必須
      threshold_granularity: per_group
      threshold_init_ratio: 0.5
      threshold_lr: null               # null なら lr を使う
    ```

=== "recon-soft-to-hard"

    ```yaml
    calibration:
      enabled: true
      method: recon-soft-to-hard
      temperature_schedule: cosine     # linear | cosine | exponential
      temperature_start: 1.0
      temperature_end: 0.05
      hard_fraction: 0.1               # 最後の10%のstepはhardな符号
    ```

以前のrunで学習したscaleから始めるには `--init-from runs/<以前のrun>` を使います。`--resume` とは併用できません。

## 進捗 {#progress}

校正は詳しい進捗を報告します。データセットの準備、teacherの読み込み、バッチごとの活性値の記録、各層のstepと
現在の損失、チェックポイント、ホールドアウトの評価、materializeです。ターミナルではリアルタイムで表示され、
`--events-jsonl FILE` を付けると[JSON lines](../reference/output.md#progress-events-jsonl)としても書き出され、
TUIのジョブタブでは進捗バーとETAで表示されます。

## 中断と復旧 {#interruptions-and-recovery}

長い実行は中断されるものです。OpenTernaryは、失うものができるだけ少なくなるように作られています。

| 状況 | コマンド |
|---|---|
| 最新のチェックポイントから続ける | `openternary calibrate --config X.yaml --output runs/R --resume` |
| 指定したチェックポイントから続ける | `… --output runs/R --resume-from <チェックポイント>`（`runs/R/artifacts/checkpoint/` 内のファイル） |
| 学習を飛ばし、最新のチェックポイントからsnapshotを書き出す | `… --output runs/R --materialize-only` |
| 最適化をせず、保存済みの状態から `calibration.json`・`metrics.json` を作り直す | `… --output runs/R --finalize` |

- 再開は既存の実行フォルダを使い、`R-001` を作りません。
- チェックポイントは、層ごとの状態を、ファイルの大きさとSHA-256のmanifestとともに保存します。壊れていたり
  一致しなかったりするチェックポイントでは止まります（`Integrity gate FAILED — fail closed, no files written`）。
- 1つのrunを同時に扱えるプロセスは1つだけです。2つ目には `run is already in use` が出ます。
- ++ctrl+c++ やTUIの*中止*は安全に止まり、終了コード `130` で終わり、中断がrunに記録されます。

## 出力 {#output}

```text
runs/R/
├── calibration.json            # 方式、設定、層ごとの損失、ホールドアウトの結果、VRAMのピーク、来歴
├── metrics.json                # 状態と主な損失
└── artifacts/
    ├── calibrated_snapshot/    # 結果: hardな三値のfake-quant snapshot
    ├── checkpoint/             # 再開用の状態（module_00000.pt など）
    ├── calibration_state/      # 層ごとの学習済みパラメータ
    ├── activation_cache*/      # 記録したteacherの活性値（再利用でき、大きくなることがある）
    └── threshold_metrics.jsonl
```

`artifacts/calibrated_snapshot` は、量子化したsnapshotとまったく同じように評価できます:

```bash
openternary quality runs/R/artifacts/calibrated_snapshot --data quality.json --device cuda --output runs/R-quality
```

!!! warning "ディスクの容量"

    活性値のキャッシュとチェックポイントは、モデルの何倍にもなることがあります。runが終わり、再開やキャッシュの
    再利用をしなくなったら、`activation_cache*` を削除してください。
