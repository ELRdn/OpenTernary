---
description: OpenTernaryのすべてのコマンドとオプション（既定値・注意点・例）。
---

# CLIリファレンス

OpenTernary 0.1.0a2 のすべてのコマンドとオプション。`openternary <コマンド> --help` でも同じオプションが表示され、
TUIのフォームも同じ定義から作られる。

```text
openternary [--version] [--help] <コマンド> [引数] [オプション]
```

| 全体のオプション | |
|---|---|
| `--version` / `-V` | バージョン（ソースからの導入ではgitのcommitも）を表示して終了 |
| `--help` | helpを表示して終了 |
| `--install-completion` / `--show-completion` | シェル補完を入れる・表示する |

コマンドなしで対話できる端末から実行すると、`openternary` は[TUI](../getting-started/tui.md)を開く。それ以外では
helpを表示して終了コード `2` で終わる。

## すべてのコマンドに共通のオプション {#options-on-every-command}

（`ui` 以外の）すべてのコマンドで使える:

| オプション | |
|---|---|
| `--json` | バージョン付きのJSON結果を1つだけstdoutに出す。人間向けの表示はstderrへ。[出力](output.md)を参照 |
| `--quiet` | 進捗の表示を止める。結果とエラーは表示する |
| `--events-jsonl FILE` | 進捗イベントを**新しい**ファイルに書く（フォルダは作られる）。`--dry-run` とは併用不可 |

モデルの引数（`[MODEL]`）は、設定ファイルの `model.id` で与えられる場合は省略できる。


## ワークフロー {#workflow}

### `inspect`

モデル構造と量子化対象を調べる（既定ではヘッダのみ読む）。

| 引数・オプション | 説明 |
|---|---|
| `[MODEL]` | モデル。ローカルのフォルダ、またはローカルのHugging Faceキャッシュにあるモデル ID |
| `--config` / `-c` | YAMLの設定ファイル。フラグの値が優先される |
| `--output` / `-o` | 作成する実行（または成果物）フォルダ。既定は `runs/<日付>_<slug>` |
| `--seed` | 乱数シード（既定 42） |
| `--device` | `auto`・`cpu`・`cuda`（AMD ROCmも `cuda`） |
| `--dtype` | 計算のdtype: `bf16`（既定）・`fp16`・`fp32` |
| `--load-weights` | 重みの値も読み、分布の統計を出す（PyTorchが必要） |
| `--dry-run` | すべてを解決・検査して計画を表示する。何も作らない |
| `--verbose` / `-v` | 詳しいログを出す |

`--load-weights` を付けない限り、safetensorsのヘッダと `config.json` だけを読む。新しいrunに `inspection.json` を書く。ガイド: [調査と計画](../guides/inspect-and-plan.md)。

```bash
openternary inspect google/gemma-4-E2B-it-qat-q4_0-unquantized
```

### `plan`

ダウンロード・テンソル読込・書込なしで量子化の計画を確認する。

| 引数・オプション | 説明 |
|---|---|
| `[MODEL]` | モデル。ローカルのフォルダ、またはローカルのHugging Faceキャッシュにあるモデル ID |
| `--config` / `-c` | YAMLの設定ファイル。フラグの値が優先される |
| `--backend` | 量子化backend: `ternary`（既定）・`torchao`・plugin |
| `--scheme` | backendの方式: `absmean`（ternary）・`int8-weight-only`（torchao） |
| `--weight-dtype` | 量子化後の重みの表現: `ternary`・`int8` |
| `--component` | 対象にするDiffusers pipelineのcomponent（例: `transformer`） |

runフォルダを作らず、何も書かない。グループ単位の設定は `--config` で渡す（または `quantize --dry-run`）。オフラインで分からない項目は不明として返す。

```bash
openternary plan ./my-model --backend ternary --scheme absmean --json
```

### `quantize`

モデルを三値fake-quantのsnapshotに量子化する。

| 引数・オプション | 説明 |
|---|---|
| `[MODEL]` | モデル。ローカルのフォルダ、またはローカルのHugging Faceキャッシュにあるモデル ID |
| `--config` / `-c` | YAMLの設定ファイル。フラグの値が優先される |
| `--output` / `-o` | 作成する実行（または成果物）フォルダ。既定は `runs/<日付>_<slug>` |
| `--seed` | 乱数シード（既定 42） |
| `--device` | `cpu` か `auto`。`cuda` は拒否（変換はCPUで行う） |
| `--dtype` | `quantize` では拒否される。snapshotは元のdtypeを保つ |
| `--group-size` | `per_group` のときの1グループの値の数（既定 128） |
| `--scale-granularity` | `per_tensor`（既定）か `per_group` |
| `--grouping-scheme` | グループの定義: `last-dim-rowwise-v1`（唯一） |
| `--backend` | 量子化backend: `ternary`（既定）・`torchao`・plugin |
| `--scheme` | backendの方式: `absmean`（ternary）・`int8-weight-only`（torchao） |
| `--weight-dtype` | 量子化後の重みの表現: `ternary`・`int8` |
| `--component` | 対象にするDiffusers pipelineのcomponent（例: `transformer`） |
| `--dry-run` | すべてを解決・検査して計画を表示する。何も作らない |
| `--verbose` / `-v` | 詳しいログを出す |

`runs/<run>/artifacts/snapshot` と `quantization.json` を書く。変換はCPUで行い、元のdtypeを保つ。ガイド: [量子化](../guides/quantize.md)。

```bash
openternary quantize ./my-model --scale-granularity per_group --group-size 128 --output runs/g128
```

### `calibrate`

三値のscale・閾値を層ごとに校正する（recon-scale）。

| 引数・オプション | 説明 |
|---|---|
| `[MODEL]` | モデル。ローカルのフォルダ、またはローカルのHugging Faceキャッシュにあるモデル ID |
| `--config` / `-c` | YAMLの設定ファイル。フラグの値が優先される |
| `--output` / `-o` | 作成する実行（または成果物）フォルダ。既定は `runs/<日付>_<slug>` |
| `--seed` | 乱数シード（既定 42） |
| `--device` | `auto`・`cpu`・`cuda`（AMD ROCmも `cuda`） |
| `--dtype` | 計算のdtype: `bf16`（既定）・`fp16`・`fp32` |
| `--dry-run` | すべてを解決・検査して計画を表示する。何も作らない |
| `--verbose` / `-v` | 詳しいログを出す |
| `--resume` | `--output` の最新のチェックポイントから再開する |
| `--resume-from` | 指定したチェックポイントから再開する |
| `--materialize-only` | 学習せず、チェックポイントから校正済みsnapshotを書き出す |
| `--init-from` | 以前の校正runの値から始める（`--resume` とは併用不可） |
| `--finalize` | 最適化をせず、保存済みの状態から `calibration.json`・`metrics.json` を作り直す |
| `--preflight` | 元モデル・正準の対象・環境・予算だけを検査する。`--output` が必要 |

`calibration` セクションのある設定が必要（`enabled` は自動で有効になる）。`--finalize`・`--materialize-only`・`--resume`・`--preflight` は既存の `--output` に対して動く。ガイド: [校正](../guides/calibrate.md)。

```bash
openternary calibrate --config calib.yaml --output runs/calib
openternary calibrate --config calib.yaml --output runs/calib --resume
```

### `export`

成果物を別形式で書き出す（元は変更しない）。

| 引数・オプション | 説明 |
|---|---|
| `RUN` | 書き出す実行または成果物のフォルダ（変更されない） |
| `--format` | `safetensors`（既定）・`ternary-packed`・`torchao`・`gguf` |
| `--output` / `-o` | 書き出す新しいフォルダ。既存は不可 |
| `--converter` | GGUF: llama.cpp の `convert_hf_to_gguf.py` のパス（`gguf` では必須） |
| `--output-dtype` | GGUFの出力: `f16`（既定）・`bf16`・`f32` |
| `--converter-python` | GGUF: converterを実行するPython（隔離した環境） |
| `--timeout-seconds` | GGUF: この秒数でconverterを止める（既定 3600） |
| `--dry-run` | すべてを解決・検査して計画を表示する。何も作らない |

元を変更せず、書き出し先を上書きしない。manifest付きで原子的に書く。ガイド: [書き出し](../guides/export.md)。

```bash
openternary export runs/g128 --format ternary-packed --output runs/g128-packed
```

## 評価・比較 {#evaluate-and-compare}

### `benchmark`

推論のsmokeベンチマークを実行する。

| 引数・オプション | 説明 |
|---|---|
| `[MODEL]` | モデル。ローカルのフォルダ、またはローカルのHugging Faceキャッシュにあるモデル ID |
| `--config` / `-c` | YAMLの設定ファイル。フラグの値が優先される |
| `--output` / `-o` | 作成する実行（または成果物）フォルダ。既定は `runs/<日付>_<slug>` |
| `--seed` | 乱数シード（既定 42） |
| `--device` | `auto`・`cpu`・`cuda`（AMD ROCmも `cuda`） |
| `--dtype` | 計算のdtype: `bf16`（既定）・`fp16`・`fp32` |
| `--suite` | ベンチマークsuite: `smoke`（唯一） |
| `--thinking` / `--no-thinking` | Gemma 4 のthinkingモード（既定はオフ） |
| `--dry-run` | すべてを解決・検査して計画を表示する。何も作らない |
| `--verbose` / `-v` | 詳しいログを出す |

固定の5プロンプトをgreedyで生成する。protocolと結果のfingerprint、トークン、レイテンシ、資源を記録する。ガイド: [評価と比較](../guides/evaluate-and-compare.md#benchmark-the-smoke-suite)。

```bash
openternary benchmark runs/g128/artifacts/snapshot --device cuda --dtype bf16 --output runs/g128-bench
```

### `quality`

固定した英日パープレキシティと指示ケースで品質を評価する。

| 引数・オプション | 説明 |
|---|---|
| `[MODEL]` | モデル。ローカルのフォルダ、またはローカルのHugging Faceキャッシュにあるモデル ID |
| `--data` | 固定した品質データセットのJSON（必須） |
| `--split` | `validation`（既定）か `test` |
| `--max-length` | パープレキシティの文脈長（既定 512） |
| `--stride` | パープレキシティのスライド幅（既定 256） |
| `--config` / `-c` | YAMLの設定ファイル。フラグの値が優先される |
| `--output` / `-o` | 作成する実行（または成果物）フォルダ。既定は `runs/<日付>_<slug>` |
| `--seed` | 乱数シード（既定 42） |
| `--device` | `auto`・`cpu`・`cuda`（AMD ROCmも `cuda`） |
| `--dtype` | 計算のdtype: `bf16`（既定）・`fp16`・`fp32` |
| `--dry-run` | モデルを読まずにデータセットを検査する |
| `--verbose` / `-v` | 詳しいログを出す |

固定データセットの1つのsplitで、英語・日本語のパープレキシティと完全一致の指示ケースを評価する。`quality.json`（レポートschema 3）を書く。ガイド: [quality](../guides/evaluate-and-compare.md#quality-perplexity-and-instructions)。

```bash
openternary quality runs/g128/artifacts/snapshot --data quality.json --device cuda --output runs/g128-q
```

### `evaluate`

固定profileで評価する（Diffusion）。画像の代理指標は人の受入を意味しない。

| 引数・オプション | 説明 |
|---|---|
| `MODEL` | Diffusers pipelineのフォルダ |
| `--profile` | 評価profileのJSON（必須） **（必須）** |
| `--config` / `-c` | YAMLの設定ファイル。フラグの値が優先される |
| `--output` / `-o` | 作成する実行（または成果物）フォルダ。既定は `runs/<日付>_<slug>` |
| `--component-artifact` | 差し替える量子化済みcomponent（safetensors） |
| `--evaluator` | evaluator: `diffusion`（既定）かplugin |
| `--dry-run` | すべてを解決・検査して計画を表示する。何も作らない |

固定profileで画像を生成し、ハッシュ・時間・画素統計を記録する。結果は常に人の確認が必要。ガイド: [Diffusionモデル](../guides/diffusion.md)。

```bash
openternary evaluate ./pipeline --profile profile.json --component-artifact runs/t/artifacts/snapshot
```

### `compare`

同条件のrunを比較する（受入を要求しない限り観測のみ）。

| 引数・オプション | 説明 |
|---|---|
| `[BASELINE]` | 基準の実行フォルダ |
| `[QUANTIZED]` | 候補の実行フォルダ |
| `--config` / `-c` | YAMLの設定ファイル。フラグの値が優先される |
| `--output` / `-o` | 作成する実行（または成果物）フォルダ。既定は `runs/<日付>_<slug>` |
| `--seed` | 乱数シード（既定 42） |
| `--device` | `auto`・`cpu`・`cuda`（AMD ROCmも `cuda`） |
| `--dtype` | 計算のdtype: `bf16`（既定）・`fp16`・`fp32` |
| `--candidate` | 同じ基準と比べる追加の候補（複数指定可） |
| `--require-acceptance` | 品質Gateが受け入れなければ終了コード6 |
| `--dry-run` | すべてを解決・検査して計画を表示する。何も作らない |
| `--verbose` / `-v` | 詳しいログを出す |

条件の違うrunは拒否する（終了コード2）。benchmarkのrunは観測のみ。qualityのrunは、証拠が揃っていれば品質Gateで判定する。`compare.json` と `acceptance.json` を書く。

```bash
openternary compare runs/base-q runs/g128-q --require-acceptance --output runs/gate
```

### `optimize`

予算内で明示した候補を探索する（testでは選ばない）。

| 引数・オプション | 説明 |
|---|---|
| `MODEL` | 探索するモデル |
| `--search-space` | 量子化設定のJSONリスト（必須） **（必須）** |
| `--quality-profile` | 検証profileのJSON（必須） **（必須）** |
| `--target-vram` | 推論VRAMの上限。例 `12GiB`（必須） **（必須）** |
| `--output` / `-o` | 探索のフォルダ（journal・候補・`search.json`） |
| `--config` / `-c` | YAMLの設定ファイル。フラグの値が優先される |
| `--objective` | `speed`（既定。最小レイテンシ）か `size`（最小の成果物） |
| `--max-candidates` | 候補の最大数、1〜256（既定 4） |
| `--wall-seconds` | 経過時間の予算・秒（既定 3600） |
| `--gpu-seconds` | GPU時間の予算・秒（既定 3600） |
| `--max-artifact-bytes` | 候補の成果物のディスク予算（既定 `20GiB`） |
| `--resume` / `--no-resume` | 同じ入力で中断した探索を再開する |
| `--dry-run` | すべてを解決・検査して計画を表示する。何も作らない |

候補ごとに新しいworkerで評価する。適格な候補がなければ終了コード `6`、予算切れは `7`。どちらの場合も最良候補は公開しない。ガイド: [制約付き探索](../guides/optimize.md)。

```bash
openternary optimize ./my-model --search-space space.json --quality-profile profile.json --target-vram 12GiB --dry-run
```

## 環境・成果物 {#environment-and-artifacts}

### `ui`

全画面のインターフェースを開く（ターミナルで引数なしの openternary でも開く）。

| 引数・オプション | 説明 |
|---|---|
| `--runs` | 実行フォルダが入っているフォルダ（既定 `runs`） |

対話できる端末でだけ開き、それ以外では終了コード2。引数なしの `openternary` も同じ。ガイド: [全画面インターフェース](../getting-started/tui.md)。

### `doctor`

導入済みパッケージとbackendを表示する（インストールやモデル実行はしない）。

| 引数・オプション | 説明 |
|---|---|
| `--probe-runtime` | デバイスでFP32・BF16の小さな試験を実行する（モデルは読まない） |
| `--device` | `--probe-runtime` のデバイス: `cpu`（既定）か `cuda` |

インストールもモデルの読込もしない。対話端末では、更新の確認を1日1回までPyPIに問い合わせる。

```bash
openternary doctor --probe-runtime --device cuda
```

### `setup`

GPUを検出し、合うPyTorch（CUDA / ROCm / MPS / CPU）をこの環境に入れる。

| 引数・オプション | 説明 |
|---|---|
| `--backend` | `auto`（既定）・`cuda`・`rocm`・`mps`・`cpu` |
| `--torch` | PyTorchのバージョンを指定。例 `2.13.0`・`2.13.0+rocm10.0.0` |
| `--variant` | ビルドの種類を指定。例 `cu130`・`rocm10.0.0`・`cpu` |
| `--amd-device` | AMDのカーネル: `auto`（このGPU用があれば）・`all`・gfxターゲット |
| `--list` | 検出したハードウェアと、入るビルドをすべて表示する |
| `--restore` | 前回の `setup` が記録したビルドをそのまま入れ直す |
| `--profile` | `--restore` と併用: この記録ファイルを読む |
| `--pre` | プレリリースのビルドも含める |
| `--yes` / `-y` | 確認せずに実行する（端末以外と `--json` では必須） |
| `--dry-run` | すべてを解決・検査して計画を表示する。何も作らない |

`openternary` を動かしている環境に入れ、その後デバイスで試験する（失敗は終了コード `5`）。uvが必要。ガイド: [インストール](../getting-started/installation.md#3-install-pytorch-for-your-hardware-openternary-setup)。

```bash
openternary setup --list
openternary setup --backend rocm --amd-device gfx1201
```

### `update`

PyPIで新しい版を確認して更新し、setupで選んだPyTorchの構成を入れ直す。

| 引数・オプション | 説明 |
|---|---|
| `--check` | PyPIで新しい版を確認するだけ |
| `--pre` | プレリリースも対象にする（既定: 導入済みがプレリリースなら対象） |
| `--yes` / `-y` | 確認せずに実行する（端末以外と `--json` では必須） |
| `--dry-run` | すべてを解決・検査して計画を表示する。何も作らない |

導入方法に合わせて更新し、`setup` が記録したPyTorchを入れ直す。Windowsのuv tool導入では、実行せずに手順を表示する。

```bash
openternary update --check
```

### `plugins`

導入済みの拡張を一覧する（コードは実行しない）。

導入済みのpluginを、コードを読み込まずにentry pointの情報から一覧する。[Plugin](plugins.md) を参照。

### `cache-info`

キャッシュ設定を表示する。

uvとHugging Faceのキャッシュの場所を表示する。

### `backends list`

backendと、その導入・対応状況を一覧する。

backendごとに、導入状況・品質の成熟度・検証済みの範囲（変換、モデル再読込、native低bit実行）を示す。

### `backends info`

1つのbackendの導入・対応状況の詳細を表示する。

| 引数・オプション | 説明 |
|---|---|
| `NAME` | backend名。例 `ternary`・`torchao` |

```bash
openternary backends info torchao --json
```

### `artifacts validate`

artifactのmanifestとファイルの整合性を検査する（モデルの品質ではない）。

| 引数・オプション | 説明 |
|---|---|
| `PATH` | 成果物のフォルダ |
| `--level` | `files`（既定。manifestとファイルのハッシュ）か `tensors`（全tensorも開く） |

成果物を変更しない。`file_integrity`・`tensor_integrity` は、モデルが読めることや正確であることを意味しない。ガイド: [成果物の管理](../guides/artifacts.md)。

### `cache list`

キャッシュの場所にある管理対象のartifactを一覧する。

| 引数・オプション | 説明 |
|---|---|
| `ROOT` | 成果物を探すフォルダ |

```bash
openternary cache list runs --json
```

### `cache remove`

参照されていないことを確かめてから管理対象のartifactを1つ削除する（--executeで実行）。

| 引数・オプション | 説明 |
|---|---|
| `TARGET` | 削除する成果物 |
| `--root` | 参照を調べる範囲のフォルダ。対象はこの中にあること（必須） **（必須）** |
| `--execute` | 実際に削除する。付けなければ検査と報告だけ |
| `--dry-run` | 検査だけ（`--execute` なしの既定と同じ） |

`--root` 内でまだ参照されている成果物、root自体、rootの外は拒否する。rootの外からの参照は `unknown` と表示する。

```bash
openternary cache remove runs/old-export --root runs            # 検査
openternary cache remove runs/old-export --root runs --execute  # 削除
```
