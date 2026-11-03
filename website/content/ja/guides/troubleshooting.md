---
description: "OpenTernaryのよくあるエラーと対処を、インストール・モデル・実行・評価・Windowsに分けてまとめています。"
---

# トラブルシューティング

エラーは**1回だけ**、stderrにパネルとして表示されます。何が失敗したか、詳細、あればヒント、終了コードが入ります:

```text
╭─ ✖ 非対応の機能 ─────────────────────────────────────────────────────────────╮
│ Quantize error                                                               │
│ ternary snapshot conversion runs on CPU; choose --device cpu or auto         │
╰─────────────────────────────────────────────────────────────── 終了コード 3 ─╯
```

パネルの見出しは表示言語に合わせて翻訳されますが、詳細の文言は英語のままです。終了コードでエラーの種類が分かります
（[終了コード](automation.md#exit-codes)を参照）。`--json` 付きなら、同じ文言が `error` の項目に入ります。以下で、
表示された文言を探してください。

## インストールと環境 {#installation-and-environment}

??? question "`uv tool install` のあとで `openternary: command not found`"

    uvのツールのフォルダが、まだ `PATH` に入っていません。`uv tool update-shell` を一度実行し、ターミナルを
    開き直してください。`uv tool dir --bin` で、コマンドが入った場所が分かります。

??? question "`torch is required …`（終了コード4）"

    `openternary` を動かしている環境にPyTorchがありません。`openternary setup` を実行してください。メッセージの
    ヒントにある `uv sync --extra ml` は、ソースのチェックアウト向けです。`uv tool` で入れた場合は
    `openternary setup` が正しいコマンドです。

??? question "`doctor` に「openternary setup で入れた構成（…）と違います」と出る"

    何かが `torch` を入れ直しました。よくあるのは、`uv sync`、別のパッケージの `pip install`、ツールの入れ直しです。
    `openternary setup --restore` で、記録したGPU用のビルドに戻せます。

??? question "`setup` がAMDのGPUを認識しない"

    gfxのターゲットを指定してください: `OPENTERNARY_GPU_ARCH=gfx1201 openternary setup`（PowerShellなら
    `$env:OPENTERNARY_GPU_ARCH="gfx1201"; openternary setup`）。Linuxなら `rocminfo`、またはAMDの資料で、
    カードのターゲットを確認できます。

??? question "`setup` に自分のPython向けのビルドがない"

    ビルドによっては特定のPythonにしかありません（たとえばWindowsのROCmは3.12と3.13だけ）。`setup` は、ビルドの
    あるPythonを示します。そのどれかで入れ直してください: `uv tool install openternary --python 3.12 --reinstall`

??? question "`setup` が終了コード5で終わる"

    ビルドは入りましたが、GPUでの試験（実デバイスでのFP32・BF16の行列積）に失敗しました。GPUのドライバを確認するか、
    別のビルドを試す（`openternary setup --list` のあと `--torch` / `--variant`）か、`--backend cpu` を使ってください。

??? question "スクリプトの中で `setup` / `update` が終了コード2で終わる"

    対話できるターミナルの外や `--json` 付きでは、何かを入れるのに `--yes` が必要です。

??? question "ソースのチェックアウトで、uvがバージョンについてエラーを出す"

    リポジトリは `pyproject.toml` の `required-version` でuvのバージョンを固定しています。そのバージョンのuvを
    入れてください（CIでは `astral-sh/setup-uv`）。

## モデルとファイル {#models-and-files}

??? question "`Local snapshot not found for '<モデルID>'`"

    OpenTernaryはモデルをダウンロードしません。先にモデルをダウンロードしてください。メッセージに正確なコマンド
    （例: `hf download google/gemma-4-E2B-it-qat-q4_0-unquantized --revision 6bef…`）が入っているので、
    `uvx --from huggingface_hub hf download …` として実行します。ゲート付きのモデルは、先にHugging Faceで
    利用規約に同意し、`hf auth login` を実行してください。

??? question "`Snapshot missing config.json` / `missing model.safetensors`"

    指定したフォルダがモデルのsnapshotではありません。`config.json` と `*.safetensors` を直接含むフォルダを指定して
    ください。OpenTernaryのrunなら `runs/<run>/artifacts/snapshot`（`export` ではrunのフォルダでも可）です。

??? question "`mixed_precision key is not a supported target: …`（終了コード2）"

    そのテンソル名は存在しないか、量子化の対象ではありません。`inspection.json` から正確な名前をコピーしてください
    （[調査と計画](inspect-and-plan.md#inspect-what-is-in-the-model)）。

??? question "正しそうなキーで設定のエラーになる"

    設定ファイルは厳格で、未知のキーは無視されずエラーになります。綴りとセクションを
    [設定のリファレンス](../reference/configuration.md)と照らし合わせてください。`benchmark.suites`（複数形）は
    非推奨で、`benchmark.suite` と一致していなければなりません。

## コマンドの実行 {#running-commands}

??? question "`quantize preserves source dtype; use --weight-dtype …`（終了コード2）"

    `quantize` は `--dtype` を受け付けません。snapshotは元のdtypeを保ちます。モデルを動かすときのdtypeは、
    `benchmark` や評価のときに選んでください。

??? question "`ternary snapshot conversion runs on CPU`（終了コード3）"

    `quantize` では `--device cpu` を使うか、`auto` のままにしてください。GPUは評価と校正で使います。

??? question "`run is already in use`"

    別のプロセスがrunの `.execution.lock` を持っています。まだ動いている校正や、二重に始めた再開などです。ロックは、
    落ちた場合も含めて、プロセスが終われば自動で解放されます。

??? question "`events file already exists; pass a new path to --events-jsonl`"

    イベントのファイルには追記しません。実行ごとに新しいファイル名を使ってください。

??? question "`--events-jsonl cannot be used with --dry-run`"

    dry-runはイベントも含めて何も書きません。どちらかを外してください。

??? question "`--preflight` が `revision_pinned: False`・`canonical_target_count: False` などで失敗する"

    preflightは正準の研究対象（固定したrevisionのGemma 4 E2B）と照合します。ほかのモデルでは、これらは設計上
    不合格になります。代わりに `calibrate --dry-run` を使ってください。

??? question "校正でGPUのメモリが足りなくなる"

    `runtime.low_vram: true` と `runtime.microbatch.auto: true`（どちらも既定）のままにしておけば、小さな
    マイクロバッチでやり直します。ほかのプログラムがGPUを使うなら、`runtime.vram.max_fraction` を下げてください。
    `calibration.seq_len` や `num_samples` を減らすこともできます。そのあと `--resume` で再開すれば、終わった分は
    失われません。

??? question "`--json` で出力が2回表示される"

    ターミナルでは、stdout（JSONの行）とstderr（人間向けの表示）が一緒に見えます。stdoutをリダイレクトする
    （`> result.json`）と、JSONだけが保存されます。

## 評価と比較 {#evaluation-and-comparison}

??? question "`quality requires --data with a frozen dataset JSON`"

    `quality` にはデータセットのファイルが必要です。[品質データセット](evaluate-and-compare.md#quality-datasets)を
    参照してください。

??? question "`split overlap` / `split near-duplicate` / `duplicate texts`"

    データセットの2つのsplitに同じ、またはほぼ同じ文章があるか、1つのsplitに同じ文章が2回あります。評価が汚染される
    ので拒否します。該当するレコードを削除するか、置き換えてください。

??? question "`compare` で `quality … mismatch` と出る"

    2つのrunの測り方（データセット・手順・dtype・デバイス・元モデル・環境）が違うので、比べても意味がありません。
    どちらかを、設定を揃えて測り直してください。

??? question "`compare` で `insufficient_evidence` と出る"

    証拠が揃っていないので、Gateを適用しませんでした。古いレポートのschema、特定できなかった元モデル、違う環境、
    人工のモデルなどが原因です。どれかは `acceptance.json` の `reason` に書かれています。

??? question "量子化したモデルの出力が意味をなさない"

    16bitで学習したモデルを素朴に三値化すると、そうなるのが想定どおりです。[現在の状況](../project/status.md)を
    参照してください。品質を取り戻す研究の方向が校正です。

## Windows {#windows}

??? question "日本語が `?` になる、または出力が英語になる"

    コンソールの文字コード（cp1252など）が日本語を表示できない場合、OpenTernaryは文字化けを出す代わりに英語に
    切り替えます。Windows Terminalを使うか、コンソールをUTF-8にするか、`OPENTERNARY_LANG=en|ja` で言語を明示して
    ください。

??? question "`update` が更新せずにコマンドを表示する"

    実行中の `openternary.exe` は自分自身を置き換えられません。OpenTernaryを閉じ、新しいターミナルで表示された
    コマンド（`uv tool upgrade openternary`、続いて `openternary setup --restore`）を実行してください。

??? question "ターミナルを閉じたらTUIのジョブが止まった"

    Windowsでは、ジョブはターミナルのコンソールで動いていて、そのウィンドウを閉じると止まります。終わるまで
    ウィンドウは閉じずに、最小化しておいてください。

??? question "uv: キャッシュ（`.lock`・`sdists-v9\.git`）へのアクセスが拒否される"

    ```powershell
    uv cache clean
    # それでもだめなら:
    Remove-Item -Force $env:LOCALAPPDATA\uv\cache\.lock
    Remove-Item -Recurse -Force $env:LOCALAPPDATA\uv\cache\sdists-v9\.git
    ```

## それでも解決しないとき {#still-stuck}

[GitHub](https://github.com/ELRdn/OpenTernary/issues)にissueを立ててください。実行したコマンド、エラーのパネル
全体、`openternary doctor --json` の出力、runの場合はその `config.yaml` と `metrics.json` を添えてください。
