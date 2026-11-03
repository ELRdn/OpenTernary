---
description: "OpenTernaryが読むすべての環境変数と、プロジェクトの外に書くすべてのファイル。"
---

# 環境変数

## OpenTernary {#openternary}

| 変数 | 値 | 効果 |
|---|---|---|
| `OPENTERNARY_LANG` | `en`・`ja` | help・表・エラーの表示言語。既定はOSのロケール。ターミナルが日本語を扱えなければ `en`。`--json` には影響しない |
| `OPENTERNARY_NO_TUI` | 空でない任意の値 | 引数なしの `openternary` がTUIを開かずhelpを表示する。`openternary ui` は拒否する |
| `OPENTERNARY_NO_UPDATE_CHECK` | 空でない任意の値 | PyPIの新しい版を自動で確認しない（`doctor` とTUI）。`update --check` は使える |
| `OPENTERNARY_STATE_DIR` | フォルダ | TUIがジョブの記録を置く場所。既定は[下記](#files-openternary-writes-outside-your-project) |
| `OPENTERNARY_GPU_ARCH` | 例: `gfx1201` | `setup` が判別できないAMDのGPUのgfxターゲットを伝える |
| `OPENTERNARY_INDEX_PYTORCH` | URL | `setup` が読むPyTorchのwheelの一覧を差し替える（既定 `https://download.pytorch.org/whl`） |
| `OPENTERNARY_INDEX_AMD` | URL | AMDのROCmのwheelの一覧を差し替える（既定 `https://stable.repo.amd.com/rocm/whl-next`） |
| `OPENTERNARY_INDEX_PYPI` | URL | `setup` が使うPyPIの一覧を差し替える（既定 `https://pypi.org/simple`） |
| `OPENTERNARY_REDUCED_MOTION` | 空でない任意の値 | TUIと起動画面の、回転するシンボルを止める |
| `OPENTERNARY_SPIN` | `cw`・`ccw` | 回転するシンボルの向き（既定は時計回り） |
| `OPENTERNARY_BREAK_IS_INTERRUPT` | `1` | 内部用: WindowsでTUIが子プロセスに設定し、Ctrl+Breakを割り込み（終了コード130）として扱わせる |
| `OPENTERNARY_VALIDATION_ROOT` | フォルダ | `scripts/validate_cli_offline.py` がログを書く場所（既定 `.tmp`） |

## OpenTernaryが従う一般的な変数 {#standard-variables-openternary-respects}

| 変数 | 効果 |
|---|---|
| `NO_COLOR` | 出力に色を付けない |
| `NO_MOTION` | `OPENTERNARY_REDUCED_MOTION` と同じ |
| `CI` | 設定されていれば、引数なしの `openternary` はTUIを開かない |
| `TERM=dumb` | 同じくTUIを開かない |
| `HF_HOME`・`HF_HUB_CACHE`・`HF_HUB_OFFLINE` など | モデルのキャッシュの場所はHugging Face自身の変数で決まる。OpenTernaryはそのキャッシュを読むだけ |
| `UV_CACHE_DIR` などの `UV_*` | `setup` と `update` が実行するuvにそのまま渡る |
| `LOCALAPPDATA`（Windows）・`XDG_STATE_HOME`・`XDG_CACHE_HOME` | 下記のファイルの置き場所の基準 |

## 研究・デバッグ用のスイッチ {#research-and-debugging-switches}

校正の研究とそのGateの試験のためのものです。結果が証拠として使えなくなる形で動作を変えるので、ふだんは設定しないで
ください。

| 変数 | 効果 |
|---|---|
| `OT_LIMIT_MODULES` | 対象の層のうち最初のN個だけを校正する |
| `OT_SKIP_MATERIALIZE` | 校正済みsnapshotの書き出しを飛ばす |
| `OT_FORCE_ROCM`・`OT_FORCE_GPU` | デバイスのコードの経路を試すため、GPUのbackendを模擬する |
| `OT_FORCE_LOW_BUDGET` | マイクロバッチの縮小を試すため、VRAMの予算を極端に小さくする |

## プロジェクトの外に書くファイル {#files-openternary-writes-outside-your-project}

| 内容 | Windows | Linux / macOS |
|---|---|---|
| 更新確認のキャッシュ（1日） | `%LOCALAPPDATA%\openternary\update-check.json` | `$XDG_CACHE_HOME/openternary/update-check.json`（既定 `~/.cache/…`） |
| TUIのジョブの記録（新しい200件を保持） | `%LOCALAPPDATA%\openternary\jobs\` | `$XDG_STATE_HOME/openternary/jobs/`（既定 `~/.local/state/…`） |
| PyTorchの環境構築の記録 | `openternary` を動かしている環境の中の `openternary-torch.json` | 同じ |

それ以外（run・成果物・イベントのファイル）は、指定した場所（既定では今いるフォルダの `runs/` の下）に書きます。
