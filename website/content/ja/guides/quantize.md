---
description: "テンソル単位またはグループ単位のscale、層の除外、pass、TorchAO backendを使って、モデルを三値のfake-quant snapshotに量子化する。"
---

# 量子化

`quantize` は、モデルの線形層を三値に変換し、**fake-quantのsnapshot**を書き出します。量子化した重みがscale
ごとに3値しか取らない、ふつうのHugging Faceのモデルフォルダです。読み込み・ベンチマーク・評価・書き出しが
できます。重みに何が起きるかは[三値量子化](../concepts/ternary-quantization.md)で説明しています。

## 基本のコマンド {#the-basic-command}

```bash
openternary quantize google/gemma-4-E2B-it-qat-q4_0-unquantized \
    --scale-granularity per_group --group-size 128 \
    --output runs/e2b-g128
```

結果は `runs/e2b-g128/artifacts/snapshot` にあり、その横にrunの記録（`config.yaml`・`quantization.json` など）が
あります。

必ず先に確認しましょう。`--dry-run` はすべてを解決・検査し、何も作りません:

```bash
openternary quantize google/gemma-4-E2B-it-qat-q4_0-unquantized --scale-granularity per_group --group-size 128 --dry-run
```

## scaleの選び方 {#choosing-scales}

| オプション | 既定 | 効果 |
|---|---|---|
| `--scale-granularity per_tensor` | ✔ | テンソルごとに1つのscale。追加コストは最小、忠実さは最低 |
| `--scale-granularity per_group` | | 各行の `--group-size` 個ごとに1つのscale |
| `--group-size N` | 128 | 1グループの値の数（`per_group` のとき）。小さいほど忠実だが、保存するscaleが増える |
| `--grouping-scheme last-dim-rowwise-v1` | ✔ | 唯一のグループの作り方。[グループ化](../concepts/ternary-quantization.md#per-tensor-or-per-group-scales)を参照 |

OpenTernaryの研究で使っている構成はグループ単位のG128です。FP32のscaleのために、重み1つあたり0.25bitが
追加でかかります。

## 設定ファイルを使う {#using-a-config-file}

フラグで渡せるものはすべてYAMLにも書けます。フラグのないオプションも書けます:

```yaml title="my-quant.yaml"
model:
  id: google/gemma-4-E2B-it-qat-q4_0-unquantized
  revision: 6befbaca7398925921802abd1f277b495b78b738
quantization:
  backend: ternary
  scheme: absmean
  scale_granularity: per_group
  group_size: 128
  target:
    attention: true
    mlp: true
  mixed_precision:                       # これらのテンソルは元の精度のまま
    model.language_model.layers.0.self_attn.q_proj.weight: preserve
  passes:                                # 量子化の前に順に適用
    - name: clip
      options: {max_abs: 2.0}
device: cpu
```

```bash
openternary quantize --config my-quant.yaml --output runs/e2b-g128-keep-l0q
```

フラグはファイルより優先され、ファイルは既定値より優先されます。解決した結果は、runの `config.yaml` として
保存されます。コマンドラインで渡したモデルは `model.id` を置き換え、新しいモデルIDが別のモデルの固定revisionを
引き継ぐことはありません。各キーの説明は[設定のリファレンス](../reference/configuration.md)にあります。

!!! tip "テンソル名の調べ方"

    `mixed_precision` のキーは正確なテンソル名でなければなりません。`inspection.json` から調べてください
    （[調査と計画](inspect-and-plan.md#inspect-what-is-in-the-model)）。どれにも当てはまらない名前はエラーになります。

## デバイスとdtype {#device-and-dtype}

snapshotの変換は、**常にCPUで**テンソルを1つずつ行います:

- `--device cpu` か `auto`（既定）を受け付けます。`--device cuda` は、黙ってCPUを使うのではなく、終了コード
  `3` で拒否します。
- snapshotは元のモデルの浮動小数点のdtypeを保ちます（BF16のモデルならBF16のsnapshot）。そのため `quantize` の
  `--dtype` は終了コード `2` で拒否します。モデルを*動かす*ときのdtypeは、ベンチマークや評価のときに選びます。

メモリの使用量には上限があり、おおよそ最大のテンソル1つと出力のshard 1つ（最大512MiB）分で、モデル全体の
大きさではありません。ディスクは、snapshotのために元のモデルとほぼ同じ容量を使います。

## TorchAO backend {#the-torchao-backend}

確立された8bitの方式と比べるために、`torchao==0.18.0` を入れた環境で:

```bash
openternary quantize ./my-model --backend torchao --scheme int8-weight-only --weight-dtype int8 --output runs/my-int8
```

TorchAOの成果物は、同じバージョンのTorchAOでしか読み込み直せず、safetensorsへの書き出しもできません。

## runに記録されるもの {#what-the-run-records}

`quantization.json` には、たとえば次のものがあります:

- 実行した設定と、テンソルを分類したアダプタ
- テンソルごとの詳細: 形、dtype、scaleの統計、グループの数、scaleのfingerprint（scaleの配列そのものはJSONに
  保存しない）
- snapshotの**contentのfingerprint**（テンソル名・形・dtype・バイト列のハッシュ）
- ビットの集計: 理想・packed・scaleの追加分を分けて記録（[ビットを正直に数える](../concepts/ternary-quantization.md#counting-bits-honestly)）

snapshot自体には、全ファイルのSHA-256を記した `openternary-manifest.json` が含まれるので、いつでも
`openternary artifacts validate` で確認できます。

## 量子化のあとで {#after-quantizing}

- 測る: [評価と比較](evaluate-and-compare.md)。素朴な三値化では、品質は大きく落ちると考えてください。
- 品質の回復を試す: [校正](calibrate.md)
- 実際のpackedの大きさを見る: [書き出し](export.md)

## 古いバージョンからの移行 {#migrating-from-older-versions}

| 以前 | 現在 |
|---|---|
| `quantize --method naive` | `--backend ternary --scheme absmean`（YAMLの `method: naive` は引き続き使える） |
| `quantize --dtype bf16` | 拒否。snapshotは元のdtypeを保つ |
