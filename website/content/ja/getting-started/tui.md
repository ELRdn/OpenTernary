---
description: "OpenTernaryの全画面インターフェース: ❯のコマンドプロンプト、実行結果の一覧、進捗を表示するバックグラウンドのジョブ、比較・診断・更新。"
---

# 全画面インターフェース

ターミナルで引数なしの `openternary`、または `openternary ui` を実行するとTUIが開きます。実行結果・ジョブの
リアルタイムの進捗・比較・環境の診断・更新を1つの画面にまとめたもので、CLIと同じコマンドを実行します。

<figure class="ot-screenshot" markdown>
![ホームタブ: バージョン、PyTorchの状態、実行結果の件数、最後の実行、ヒント](../assets/images/tui-home-ja.svg)
</figure>

```bash
openternary                    # ターミナルで引数なし
openternary ui --runs results  # 明示して開く。実行結果のフォルダを変える場合（既定は runs）
```

!!! info "開く条件"

    TUIが開くのは対話できるターミナル（stdinとstdoutの両方がTTY）のときだけです。パイプ、CI（`CI` 環境変数が
    ある）、`TERM=dumb`、`OPENTERNARY_NO_TUI=1` のときは、引数なしの `openternary` は以前と同じくhelpを表示して
    終了コード `2` で終わります。`openternary ui` は理由を表示して終了コード `2` で終わります。スクリプトが
    全画面のプログラムに捕まることはありません。

## すべては `❯` のプロンプトから {#everything-starts-at-the-prompt}

ボタンはありません。画面下の入力欄は、コーディング用エージェントのコマンドラインと同じように使います。`/` を
打つとコマンドの一覧が出て、続けて文字を打つと絞り込まれます。

<figure class="ot-screenshot" markdown>
![/ を打つと、すべてのコマンドが説明付きで一覧される](../assets/images/tui-commands-ja.svg)
</figure>

<figure class="ot-screenshot" markdown>
![/qu と打つと一覧が絞り込まれる: /quantize、/quality、/quit、続いて名前や説明に qu を含むもの](../assets/images/tui-filter-ja.svg)
</figure>

| キー | プロンプトでの動作 |
|---|---|
| ++slash++ | コマンドの一覧を開く（どこからでも） |
| 文字 | 絞り込み。名前が入力で始まるもの、名前に含むもの、説明に含むものの順 |
| ++up++ ++down++ | 一覧の中を移動。一覧が閉じているときは、表示中のタブの表を移動 |
| ++enter++ | 選択中のコマンドを実行 |
| ++tab++ | 選択中のコマンドを補完し、続けて引数を入力できるようにする |
| ++esc++ | 一覧を閉じる。または入力を消す |
| ++ctrl+q++ | 終了（`/quit` でも可） |

### コマンド {#commands}

| コマンド | 内容 |
|---|---|
| `/new` | 一覧からコマンドを選び、そのフォームを開く |
| `/quantize`・`/calibrate`・`/benchmark` など | そのコマンドのフォームを開く。引数を続けると、フォームを飛ばして計画画面へ: `/quantize tiny-llama --device cpu` |
| `/compare` | 実行結果から基準、続いて候補を選び、比較のフォームを開く |
| `/home`・`/runs`・`/jobs`・`/doctor`・`/update` | そのタブを表示（数字キー ++1++〜++6++ でも切り替え可） |
| `/baseline`・`/candidate` | **実行結果**タブで選択中のrunを、`/compare` の基準・候補に指定 |
| `/cancel` | 選択中のジョブを中止 |
| `/delete` | 選択中のrunを削除（名前の入力で確認） |
| `/probe` | `doctor --probe-runtime`: デバイスでPyTorchを試験 |
| `/setup` | このPCにPyTorchを入れる（`openternary setup`） |
| `/upgrade` | `openternary update` を実行 |
| `/check` | PyPIの新しい版を今すぐ確認 |
| `/refresh` | 実行結果・ジョブ・診断を再読込 |
| `/cache:remove`・`/artifacts:validate` など | グループのコマンドは `:` でつなぐ |

`/doctor` と `/update` はタブを表示します。同じ名前のコマンドを実行するのは `/probe` と `/upgrade` です。

## ジョブを始める: フォーム → 計画 → 開始 {#starting-a-job-form-plan-start}

1. **フォーム。** コマンドを選ぶと、CLIの定義から自動生成したフォームが開きます。そのため、
    `openternary <コマンド> --help` と常に同じオプションが並びます。++enter++ で次の欄に移ります。
    `--json`・`--events-jsonl`・`--quiet`・`--dry-run`・`--yes` はTUIが扱うので表示されません。++ctrl+s++ で
    確認に進みます。

2. **計画。** 計画画面には、実行するコマンド行と、そのコマンド自身の `--dry-run` の出力が表示されます。
    dry-runが成功するまで**開始**は選べません（薄く表示され、`❯` も止まりません）。dry-runが失敗した場合は、
    理由が表示され、開始できません。

    <figure class="ot-screenshot" markdown>
    ![計画画面: コマンド行と、--dry-run で解決された設定](../assets/images/tui-plan-ja.svg)
    </figure>

3. **開始。** ジョブがバックグラウンドで動きます。`setup` と `update` では、この画面で開始を選ぶことが
    `--yes` の代わりになります。

計画・削除・中止・終了の確認画面も同じ操作です。++up++ ++down++ で `❯` を動かし、++enter++ で決定します。

## ジョブ: リアルタイムの進捗 {#jobs-live-progress}

<figure class="ot-screenshot" markdown>
![ジョブタブ: 完了したquantizeのジョブの段階、進捗バー、ログの末尾](../assets/images/tui-jobs-ja.svg)
</figure>

各ジョブは、ふつうのCLIを**子プロセス**として実行したものです:

```text
python -m openternary <コマンド> … --json --events-jsonl <ジョブのフォルダ>/events.jsonl
```

TUIは、ジョブが書く[進捗イベント](../reference/output.md#progress-events-jsonl)を読み、段階ごとに進捗バー・速度・
ETAと、ジョブのログの末尾を表示します。処理は別のプロセスで行われるので:

- 結果・終了コード・JSON・run lock・`last_failure.json` は、スクリプトから実行した場合とまったく同じです
- ジョブがメモリ不足で落ちても、TUIは巻き込まれません
- Linux・macOSではTUIを閉じてもジョブは止まらず、開き直すと監視を再開します。Windowsでは、ターミナルの
  ウィンドウを閉じると、そのコンソールで動いているジョブも止まります

| キー（ジョブの表を選択中） | |
|---|---|
| ++n++ | 新しいジョブ |
| ++k++ | 選択中のジョブを中止 |

**中止**すると、方法を聞かれます。*中止*は割り込みを送ります（Linux・macOSはプロセスグループへSIGINT、Windowsは
Ctrl+Break）。コマンドは安全に止まり、終了コード `130` で終わり、失敗がrunに記録されます。*強制停止*は
プロセスツリーをすぐに終了させ、何も記録されません。

ジョブの記録は、Windowsでは `%LOCALAPPDATA%\openternary\jobs`、それ以外では `$XDG_STATE_HOME/openternary/jobs`
（既定は `~/.local/state/openternary/jobs`）に保存されます。`OPENTERNARY_STATE_DIR` で場所を変えられます。
新しいものから200件まで残ります。

## 実行結果 {#runs}

**実行結果**タブには、`runs/`（または `--runs` で指定したフォルダ）にあるrunが、新しい順に、状態・品質の判定・
種類・名前とともに並びます。右側には、選択中のrunの詳細とファイルが表示されます。

| キー（実行結果の表を選択中） | |
|---|---|
| ++b++ / ++c++ | 比較の基準・候補にする |
| ++x++ | runを削除 |
| ++r++ | 再読込 |

削除するには、**runの名前を入力**して確認します。削除できるのは、実行結果のフォルダの直下にあって、自身の
`metrics.json` と `config.yaml` を持つrunだけです。別のプロセスが使っている（`.execution.lock` を持っている）
runは拒否されます。`cache remove --execute` も、同じように対象の名前の入力を求めます。

## そのほかのタブ {#other-tabs}

- **比較**: 選んだ基準と候補を表示します。`/compare` で一覧から選べます。
- **診断**: `openternary doctor` と同じ内容です。`/probe` でデバイスの試験を行います。
- **更新**: 導入済みと最新のバージョンです。`/check` で今すぐPyPIを確認し、`/upgrade` で `openternary update` を
  実行します。

## 言語・色・動き {#language-colour-and-motion}

- 表示言語は `OPENTERNARY_LANG=en|ja` に従い、指定がなければOSの言語になります。
- `OPENTERNARY_REDUCED_MOTION=1` または `NO_MOTION=1` で、回転するシンボルが止まります。
- `OPENTERNARY_NO_UPDATE_CHECK=1` で更新の確認をしなくなります（通常は1日1回まで）。
