---
description: "openternaryコマンドを入れ、openternary setupでGPUに合うPyTorchを用意し、最新に保つ方法。"
---

# インストール

導入方法は3つあります。ほとんどの場合は1つ目で十分です。

| 導入方法 | 用途 | コマンド |
|---|---|---|
| **uv tool**（推奨） | どのフォルダからでもOpenTernaryを使う | `uv tool install openternary --python 3.12` |
| **プロジェクトのvenv** | Pythonのコードやノートブックから使う | `uv pip install openternary` |
| **ソースから** | OpenTernary自体を開発する | `git clone` + `uv sync` |

## 1. uvを入れる {#1-install-uv}

OpenTernaryは、自分自身とPyTorchのインストールに[uv](https://docs.astral.sh/uv/)を使います。まだ入っていなければ:

=== "Windows（PowerShell）"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "macOS / Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

入れたあとは、`uv` が `PATH` に入るようにターミナルを開き直してください。

## 2. `openternary` コマンドを入れる {#2-install-the-openternary-command}

```bash
uv tool install openternary --python 3.12
```

専用の環境が作られ、`openternary` が `PATH` に入ります。コマンドが見つからない場合は、`uv tool update-shell` を
一度実行してからターミナルを開き直してください。

!!! info "なぜ `--python 3.12` を付けるのか"

    付けないと、uvはPCにある最新のPythonを選ぶことがあります。OpenTernaryの検証は3.11と3.12で行っており、
    PyTorchのビルドによっては特定のPythonにしかありません。たとえばAMDのWindows向けROCmのwheelは、
    Python 3.12と3.13用だけです。

現在のリリースはプレリリース（`0.1.0a2`）です。安定版がまだないので、uvはそのままプレリリースを入れます。
安定版が出たあとも続けてプレリリースを使いたい場合は、`uv tool install openternary --prerelease allow` を使います。

動作を確認します:

```console
$ openternary --version
openternary 0.1.0a2
```

この時点で入っているのは**本体（core）**です。help・量子化の計画・モデルのヘッダの調査・環境の表示ができます。
PyTorchは含まれていないので、次の手順で入れます。

## 3. ハードウェアに合うPyTorchを入れる: `openternary setup` {#3-install-pytorch-for-your-hardware-openternary-setup}

PyTorchには、NVIDIA（CUDA）・AMD（ROCm）・Apple silicon（MPS）・CPUのみ、と種類の違うビルドがあり、どれが
合うかはGPU・ドライバ・OS・Pythonで決まります。`openternary setup` がこれを判断します:

```bash
openternary setup --list      # 検出したハードウェアと、入れられるビルドの一覧
openternary setup --dry-run   # おすすめの計画を表示（何も入れない）
openternary setup             # 確認のうえ入れ、GPUで動作を試験する
```

`setup` が行うこと:

1. **GPUを検出します。**

    | ハードウェア | 検出方法 |
    |---|---|
    | NVIDIA | `nvidia-smi`（ドライバが対応するCUDAの上限も読む） |
    | WindowsのAMD | `Win32_VideoController` から得たGPUのPCIデバイスIDを、gfxターゲットに対応付ける |
    | Linux / WSLのAMD | `rocminfo`、ROCmがなければ `/sys` |
    | Apple silicon | macOSのMPS |

    AMDのGPUを判別できない場合は、ターゲットを指定します: `OPENTERNARY_GPU_ARCH=gfx1201 openternary setup`

2. **いま公開されているビルドを一覧します。** [PyTorchの配布サイト](https://download.pytorch.org/whl/)
    （CUDA・ROCm・CPU）、AMD（Windows・Linux向けROCm）、PyPIの一覧をその場で読みます。内蔵のバージョン表は
    持たないので、古くなることがありません。*このPC*のPythonとOSに入るwheelだけを候補にします。

3. **1つを推奨します。** 同じ配布元に対応する `torchvision` があるもの、プレリリースより安定版、ドライバで
    動くもの（NVIDIAならドライバの上限以下のCUDA）を優先します。このPCですでに検証済みのビルドがあれば、
    それを優先します。

4. **`openternary` を動かしている環境に入れます。** OpenTernaryに必要な機械学習のパッケージ（`transformers`・
    `safetensors`・`accelerate`・`datasets` など）も一緒に、`uv pip install --python <その環境>` で入れます。

5. **実際のデバイスで試験します。** 新しいプロセスで、GPU上のFP32とBF16の行列積を実行します。結果と正確な
    ビルドは、その環境の `openternary-torch.json` に記録されます。試験に失敗すると終了コード `5` で終わります。

よく使うオプション:

```bash
openternary setup --backend rocm            # backendを指定: auto | cuda | rocm | mps | cpu
openternary setup --torch 2.13.0            # バージョンを指定
openternary setup --variant cu130           # ビルドの種類を指定（cu130・rocm10.0.0・cpu など）
openternary setup --amd-device gfx1201      # AMD: このGPU用のカーネルだけ（ダウンロードが小さい）
openternary setup --pre                     # プレリリースのビルドも含める
openternary setup --restore                 # 前回setupが記録したビルドをそのまま入れ直す
```

AMDでは、`--amd-device auto`（既定）なら、AMDがそのGPU用のパッケージを公開している場合はそのカーネルだけを、
そうでなければ全カーネルをダウンロードします。

!!! warning "`uv sync` などでPyTorchが置き換わることがあります"

    何かが `torch` をPyPIから入れ直すと、GPU用のビルドが知らないうちに失われます。`openternary doctor` は、
    入っているtorchと `setup` の記録を比べて知らせます。`openternary setup --restore` で記録どおりのビルドに
    戻せます。

### 確認できない環境で {#without-a-terminal-prompt}

スクリプト・CI・`--json` 付きでは、`--yes` がない限り `setup` は何も入れず、終了コード `2` で終わります:

```bash
openternary setup --yes --json
```

## 4. 環境を確認する: `openternary doctor` {#4-check-your-environment-openternary-doctor}

```bash
openternary doctor                                # パッケージ・backend・PyTorchのビルド
openternary doctor --probe-runtime --device cuda  # GPUでFP32・BF16の小さな試験も行う
```

`doctor` は何もインストールせず、モデルも読み込みません。`--probe-runtime` がなければパッケージの情報を読む
だけです。ROCmでもPyTorchはGPUを `cuda` と呼ぶので、AMDのGPUでも `--device cuda` を使います。

`setup` を使わずにCPU版のPyTorch 2.15.0を入れたLinuxでの実際の出力（`OPENTERNARY_LANG=ja`）:

```text
╭─ 環境 ────────────────────────────────────────────────────────────────────────────╮
│ Python 3.12.3  ·  Linux-6.18.44-fc-v230-x86_64-with-glibc2.39                     │
│                                                                                   │
│ 依存パッケージ                                                                    │
│ パッケージ     バージョン                                                         │
│ ──────────────────────────────                                                    │
│ openternary    0.1.0a2                                                            │
│ torch          2.15.0                                                             │
│ safetensors    0.8.0                                                              │
│ transformers   5.22.1                                                             │
│ diffusers      ○ not installed                                                    │
│ torchao        ○ not installed                                                    │
│ datasets       ○ not installed                                                    │
│ バックエンド                                                                      │
│ 項目              ternary                          torchao                        │
│ ───────────────────────────────────────────────────────────────────────────────── │
│ 導入              ● installed                      ○ not installed                │
│ 品質              ▲ experimental                   ▲ experimental                 │
│ 変換              ◐ implemented model unverified   ◐ implemented model unverified │
│ モデル再読込      ◐ unverified                     ◐ unverified                   │
│ ネイティブ低bit   – unsupported                    ◐ unverified                   │
│ 形式              safetensors, ternary-packed      torchao                        │
│ バージョン        0.1.0a2                          必要: 0.18.0                   │
│ ハードウェア: 未確認 — openternary doctor --probe-runtime                         │
│ モデル検証: ○ not run                                                             │
│ PyTorch: 2.15.0  (openternary setup では未設定)                                   │
╰───────────────────────────────────────────────────────────────────────────────────╯
```

見出しは日本語になりますが、状態を表す語（`installed`・`not run` など）は、`--json` の値と同じ英語のままです。
注目するのは最後の行です:

| `PyTorch:` の行 | 意味 |
|---|---|
| `2.15.0  (openternary setup では未設定)` | PyTorchは入っているが `setup` が入れたものではないので、GPUで動くかの記録がない |
| `2.14.0+rocm10.1.0  · rocm · amd  ✔ passed` | `setup` がこのビルドを入れ、GPUでの試験に合格した |
| `▲ openternary setup で入れた構成（…）と違います。` | 記録したビルドが別のものに置き換わっている。`openternary setup --restore` を実行する |

状態の記号の意味は[実行結果と証拠](../concepts/runs-and-evidence.md#status-symbols)を参照してください。

## 追加の機能（extras） {#optional-extras}

| extra | 追加されるもの | 用途 |
|---|---|---|
| `ml` | `torch`・`safetensors`・`transformers`・`accelerate`・`pillow`・`torchvision` | 量子化・ベンチマーク・品質評価（`setup` が入れる） |
| `calibration` | `datasets` | `wiki-tiny` での校正（`setup` が入れる） |
| `torchao` | `torchao==0.18.0` | TorchAOのINT8 backend |
| `diffusion` | `diffusers==0.40.0` | Diffusersのpipelineと `evaluate` |

uv toolの導入にextraを追加するには、extra付きで入れ直してから、もう一度 `setup` を実行します。入れ直すと環境が
作り直され、GPU用のビルドが消えるためです:

```bash
uv tool install "openternary[diffusion,torchao]" --python 3.12 --reinstall
openternary setup
```

## 更新: `openternary update` {#updating-openternary-update}

```bash
openternary update --check     # PyPIに新しい版があるか確認するだけ
openternary update --dry-run   # 実行する手順を表示
openternary update             # 更新し、setupが記録したPyTorchを入れ直す
```

- ネットワークへの問い合わせは、PyPIのJSON API（`pypi.org/pypi/openternary/json`）だけです。結果は1日キャッシュします。
- 手順は導入方法で変わります。uv toolなら `uv tool upgrade openternary`、venvなら
  `uv pip install --upgrade openternary`。ソースからのeditable導入は変更せず、`git pull` を案内します。
- uv toolの更新では環境が作り直され、GPU用のビルドが消えます。そのため `update` は `setup` の記録を先に退避し、
  更新後に `openternary setup --restore --profile <退避先> --yes` を実行します。
- **Windowsでは**、実行中のプログラムが自分自身を置き換えられません。uv toolで入れている場合、`update` は
  実行する代わりに、新しいターミナルで実行するコマンドを表示します:

    ```powershell
    uv tool upgrade openternary
    openternary setup --restore
    ```

- `openternary doctor` は、対話できるターミナルでだけ、1日1回までPyPIを確認し、新しい版があれば知らせます。
  `OPENTERNARY_NO_UPDATE_CHECK=1` で無効にできます。

## プロジェクトの環境に入れる {#installing-into-a-project-environment}

自分のコードやノートブックから `import openternary` したい場合:

```bash
uv venv --python 3.12
uv pip install openternary
uv run openternary setup
```

`setup` は、実行した環境（ここではプロジェクトの `.venv`）にPyTorchを入れます。通常の `pip` でも
`pip install openternary` で同じように入りますが、`setup` と `update` には `PATH` 上のuvが必要です。

## ソースから入れる {#installing-from-source}

OpenTernary自体を開発する場合です。リポジトリはuvのバージョンを固定している（`pyproject.toml` の
`required-version`）ので、そのバージョンのuvを使います。

```bash
git clone https://github.com/ELRdn/OpenTernary.git
cd OpenTernary
uv sync                                  # 本体と開発ツール（uv.lockどおり）
uv run openternary --help
uv sync --extra ml --extra calibration   # 機械学習の依存を追加（PyPIのCPU/CUDA版torch）
uv run openternary setup                 # またはGPUに合うビルドに置き換える
```

チェックアウトに追従する `openternary` コマンドを、どのフォルダからでも使えるようにするには:

```bash
uv tool install --editable . --python 3.12
uv tool update-shell     # 初回のみ。そのあとターミナルを開き直す
```

`--editable` なので、`git pull` した変更はすぐ反映されます。依存が増えた更新のあとは
`uv tool install --editable . --reinstall` を実行し、環境が作り直されるので `openternary setup --restore` も
実行します。

開発時のチェックは[コントリビュート](../project/contributing.md)で説明しています。

## アンインストール {#uninstalling}

```bash
uv tool uninstall openternary
```

コマンドとその環境（`setup` が入れたPyTorchを含む）が削除されます。実行結果やモデルには触れません。ほかに
削除してよいものとして、TUIのジョブの記録（[環境変数](../reference/environment-variables.md#files-openternary-writes-outside-your-project)を参照）と、
モデルをダウンロードした場合はHugging Faceのキャッシュがあります。
