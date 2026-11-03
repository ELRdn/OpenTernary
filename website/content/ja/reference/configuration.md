---
description: "OpenTernaryのYAML設定のすべてのキー（型・既定値・取りうる値）と、フラグによる上書きの規則。"
---

# 設定

`--config` を受け取るコマンドは、YAMLのファイルを読みます。実験を決めるものはすべてここに書けるので、runに保存
される `config.yaml` で、その実験を完全に説明できます。

## 規則 {#rules}

- **優先順位:** コマンドラインのフラグ ＞ YAMLのファイル ＞ 組み込みの既定値
- **厳格:** 未知のキー、有限でない数値（`.nan`・`.inf`）、範囲外の値、矛盾する設定はエラー（終了コード `2`）です。
  黙って無視されるものはありません。
- **解決した設定を保存します。** すべてのrunに、書かなかった既定値も含めて `config.yaml` として保存されます。
- **モデルのrevisionは引き継ぎません。** 既定以外のモデルID（コマンドラインまたは `model.id`）を指定すると、
  `model.revision` は既定のGemmaのものを引き継がず、自分で設定しなければ `null` になります。
- どのコマンドでも、`--dry-run` で何も実行せずに解決した設定を表示できます。

フラグとキーの対応:

| フラグ | キー |
|---|---|
| `MODEL` 引数 | `model.id` |
| `--seed` | `seed` |
| `--device` | `device` |
| `--dtype` | `dtype` |
| `--output` | `output` |
| `--group-size` | `quantization.group_size` |
| `--scale-granularity` | `quantization.scale_granularity` |
| `--grouping-scheme` | `quantization.grouping_scheme` |
| `--backend`・`--scheme`・`--weight-dtype` | `quantization.backend`・`.scheme`・`.weight_dtype` |
| `--component` | `model.component` |
| `--suite` | `benchmark.suite` |
| `--thinking` / `--no-thinking` | `benchmark.thinking` |

## 全体の例 {#a-complete-example}

```yaml
schema_version: 1
model:
  id: google/gemma-4-E2B-it-qat-q4_0-unquantized
  revision: 6befbaca7398925921802abd1f277b495b78b738
  adapter: auto
quantization:
  backend: ternary
  scheme: absmean
  weight_dtype: ternary
  scale_granularity: per_group
  group_size: 128
  target: {attention: true, mlp: true}
  mixed_precision: {}
  passes: []
calibration:
  enabled: true
  method: recon-scale
  dataset: wiki-tiny
  num_samples: 32
  seq_len: 128
  steps: 50
  lr: 0.001
  optimizer: adam
  loss: mse
  checkpoint_interval: 10
  held_out_ratio: 0.2
benchmark:
  suite: smoke
  thinking: false
  generation: {do_sample: false, num_beams: 1, max_new_tokens: 64}
  warmup: true
runtime:
  low_vram: true
  vram: {max_fraction: 0.8, reserve_mb: 1024, min_free_mb: 768}
  microbatch: {auto: true, min_size: 1}
seed: 42
device: cuda
dtype: bf16
```

## 最上位 {#top-level}

| キー | 型 | 既定 | 説明 |
|---|---|---|---|
| `schema_version` | `1` | `1` | 設定のschemaのバージョン。`1` だけ |
| `seed` | int | `42` | Python・NumPy・PyTorchのシード |
| `device` | `auto` \| `cpu` \| `cuda` | `auto` | モデルを動かすデバイス。ROCmのAMDのGPUも `cuda` |
| `dtype` | `bf16` \| `fp16` \| `fp32` | `bf16` | モデルを動かすdtype（ベンチマーク・品質・校正）。snapshotは元のdtypeを保つ |
| `output` | 文字列 \| null | `null` | 実行フォルダ。`null` なら `runs/<日付>_<モデル>-<方式>-g<グループサイズ>` |

## `model` {#model}

| キー | 型 | 既定 | 説明 |
|---|---|---|---|
| `model.id` | 文字列 | `google/gemma-4-E2B-it-qat-q4_0-unquantized` | ローカルのフォルダ、またはローカルキャッシュにあるHugging FaceのID |
| `model.revision` | 文字列 \| null | 既定のIDなら固定したGemmaのcommit、それ以外は `null` | 使うHugging Faceのcommit。再現性のために固定する |
| `model.adapter` | `auto` \| `gemma4` \| `transformers` \| `diffusers` | `auto` | モデルを分類・読み込むアダプタ。`auto` はファイルから判断 |
| `model.component` | 文字列 \| null | `null` | 対象にするDiffusers pipelineのcomponent（例: `transformer`） |

## `quantization` {#quantization}

| キー | 型 | 既定 | 説明 |
|---|---|---|---|
| `quantization.backend` | 文字列 | `ternary` | `ternary`・`torchao`・導入済みのplugin |
| `quantization.scheme` | 文字列 | `absmean` | backendの方式。ternaryは `absmean`、TorchAOは `int8-weight-only` |
| `quantization.weight_dtype` | 文字列 | `ternary` | 量子化した重みの表現: `ternary`・`int8` |
| `quantization.activation_dtype` | `preserve` | `preserve` | 活性値は量子化しない |
| `quantization.scale_granularity` | `per_tensor` \| `per_group` | `per_tensor` | テンソルごと、またはグループごとに1つのscale |
| `quantization.group_size` | int ≥ 1 | `128` | `per_group` のときの1グループの値の数 |
| `quantization.grouping_scheme` | `last-dim-rowwise-v1` | `last-dim-rowwise-v1` | グループの作り方: 最後の次元の各行で連続する値 |
| `quantization.target.attention` | bool | `true` | attentionの射影を量子化する |
| `quantization.target.mlp` | bool | `true` | MLPの射影を量子化する |
| `quantization.mixed_precision` | テンソル名 → `preserve` \| `ternary` の対応 | `{}` | テンソルごとの上書き。名前は量子化の対象でなければならない（[調べ方](../guides/inspect-and-plan.md#inspect-what-is-in-the-model)） |
| `quantization.passes` | `{name, options}` のリスト | `[]` | 量子化の前に順に適用するpass: `clip`（`options: {max_abs: <0より大きい有限の数>}`）、`noop`（optionsなし）、plugin |
| `quantization.backend_options` | 文字列 → 文字列/int/float/bool の対応 | `{}` | backendへの追加のオプション |
| `quantization.method` | `naive` \| null | `naive` | ternaryのAbsMeanの旧称。ternaryのみ |
| `quantization.codebook` | intのリスト \| null | `[-1, 0, 1]` | 旧来の項目。ternaryでは `[-1, 0, 1]` でなければならない |

## `calibration` {#calibration}

`calibrate` が使います。各設定の働きは[校正](../concepts/calibration.md)を参照してください。

| キー | 型 | 既定 | 説明 |
|---|---|---|---|
| `calibration.enabled` | bool | `false` | `calibrate` が自動で有効にする。明示的に `false` と書いて `calibrate` を実行するとエラー |
| `calibration.method` | `recon-scale` \| `recon-threshold` \| `recon-soft-to-hard` | `recon-scale` | 学習するもの |
| `calibration.window` | `per-layer` | `per-layer` | 再構成の範囲。層ごとだけ |
| `calibration.dataset` | `synthetic` \| `wiki-tiny` | `synthetic` | 校正の文章。`synthetic` は仕組みの確認用 |
| `calibration.dataset_revision` | 文字列（40文字以上） | `b08601e04326c79dfdd32d625aee71d232d685c3` | `wiki-tiny` に使う `Salesforce/wikitext` のcommit |
| `calibration.allow_dataset_fallback` | bool | `false` | `wiki-tiny` を読めないときに `synthetic` に切り替えることを許す |
| `calibration.num_samples` | int 1〜1024 | `32` | 校正の系列の数 |
| `calibration.seq_len` | int 16〜512 | `128` | 系列あたりのトークン数 |
| `calibration.held_out_ratio` | float 0.05〜0.5 | `0.2` | 最適化に使わないサンプルの割合 |
| `calibration.steps` | int ≥ 1 | `50` | 層ごとの最適化のstep数 |
| `calibration.lr` | float > 0 | `0.001` | 学習率 |
| `calibration.optimizer` | `adam` \| `sgd` | `adam` | 最適化手法 |
| `calibration.loss` | `mse` \| `l1` | `mse` | 再構成の損失 |
| `calibration.checkpoint_interval` | int ≥ 1 | `10` | このstep数ごとにチェックポイントを保存 |
| `calibration.seed` | int \| null | `null` | 校正のシード。`null` なら最上位の `seed` |
| `calibration.init_from` | 文字列 \| null | `null` | 値を引き継ぐ以前の校正のrun（`--init-from` でも可） |
| `calibration.threshold_enabled` | bool | `false` | `method` が `recon-threshold` のときに限り `true` |
| `calibration.threshold_lr` | float > 0 \| null | `null` | 閾値の学習率。`null` なら `lr` |
| `calibration.threshold_init_ratio` | float (0, 1) | `0.5` | 閾値の初期値（scaleに対する比） |
| `calibration.threshold_granularity` | `per_tensor` \| `per_group` | `per_group` | テンソルごと、またはグループごとに1つの閾値 |
| `calibration.threshold_estimator` | `clipped-ste` | `clipped-ste` | hardな閾値の勾配の推定法 |
| `calibration.threshold_ste_width` | float > 0 | `0.1` | clipped straight-throughの窓の幅 |
| `calibration.threshold_eps` | float (0, 0.5) | `0.01` | 閾値の比を0と1から離しておく幅 |
| `calibration.temperature_schedule` | `linear` \| `cosine` \| `exponential` | `linear` | soft-to-hardの温度のスケジュール |
| `calibration.temperature_start` | float > 0 | `1.0` | 最初の温度。`temperature_end` 以上 |
| `calibration.temperature_end` | float > 0 | `0.05` | softの最後の温度 |
| `calibration.hard_fraction` | float [0, 1) | `0.1` | 最後にhardな符号を使うstepの割合 |
| `calibration.zero_logit_bias` | float | `0.0` | softの経路で符号0に寄せる、任意の固定のバイアス。最後のhard化では使わない |

## `benchmark` {#benchmark}

| キー | 型 | 既定 | 説明 |
|---|---|---|---|
| `benchmark.suite` | `smoke` | `smoke` | ベンチマークのsuite |
| `benchmark.suites` | リスト | `["smoke"]` | 非推奨の別名。書く場合は `[suite]` と一致すること |
| `benchmark.thinking` | bool | `false` | Gemma 4のthinkingモード |
| `benchmark.warmup` | bool | `true` | 計測の前にウォームアップのプロンプトを1つ実行 |
| `benchmark.generation.do_sample` | bool | `false` | サンプリング。`false` はgreedy |
| `benchmark.generation.num_beams` | int ≥ 1 | `1` | ビームの数 |
| `benchmark.generation.max_new_tokens` | int 1〜2048 | `64` | プロンプトごとに生成する最大トークン数 |

どれかを変えるとベンチマークのprotocolのfingerprintが変わるので、設定の違うrunどうしは比べられません。

## `runtime` {#runtime}

GPUで校正するときのメモリの方針です。

| キー | 型 | 既定 | 説明 |
|---|---|---|---|
| `runtime.low_vram` | bool | `true` | 最適化中の層だけをGPUに置く |
| `runtime.vram.max_fraction` | float (0, 1] | `0.8` | GPUのメモリを使う割合の上限 |
| `runtime.vram.reserve_mb` | int ≥ 0 | `1024` | デスクトップやほかのプログラムのために空けておくメモリ |
| `runtime.vram.min_free_mb` | int ≥ 0 | `768` | 空きがこれを下回ったら確保をやめる |
| `runtime.microbatch.auto` | bool | `true` | メモリ不足のエラーのあと、小さなマイクロバッチでやり直す |
| `runtime.microbatch.min_size` | int ≥ 1 | `1` | マイクロバッチの最小 |

## すぐに使えるファイル {#ready-made-files}

リポジトリの[`configs/`](https://github.com/ELRdn/OpenTernary/tree/main/configs)に、研究で使った設定があります。
PyPIのパッケージには含まれていないので、GitHubからダウンロードするか、リポジトリをcloneしてください。

| ファイル | 内容 |
|---|---|
| `gemma4-e2b.yaml` | Gemma 4 E2BのBF16のsmokeの基準（revision固定） |
| `gemma4-e2b-calibration.yaml` | `recon-scale`、G128、`wiki-tiny`、32×128、50 step |
| `gemma4-e2b-calibration-smoke.yaml` | 同じものを8×64・2 stepにした、仕組みの確認用 |
| `gemma4-e2b-threshold.yaml` | `recon-threshold` |
| `gate-real-rocm-5.yaml` | 5モジュールのAMD GPUのGateの設定 |
