# OpenTernary TUI / UX 設計

2026-11-03 / status: draft (P1〜P5実装済み。0.1.0a1をPyPIで公開済み。P5のTUIはオーナー実機での確認待ち)

## 目的

既存CLIの機械契約（`--json`・終了コード・events JSONL）を1バイトも変えずに、人間向けの体験をRich/Textualで高品質化する。`AGENTS.md`の「GUIを主開発対象にしない」に対し、本件は**主開発ではなくUI/UX改善として例外扱い**（2026-11-01 オーナー承認）。

## 決定事項ログ

| # | 項目 | 決定 |
|---|---|---|
| D1 | 優先度 | UI/UX改善として例外扱い |
| D2 | 起動 | TTYで引数なし`openternary` → TUI。非TTYは従来どおりhelp。`openternary ui`も提供 |
| D3 | 大文字別名 | 追加しない（Windowsは大文字小文字を区別しない） |
| D4 | Textual | 必須依存。ただし遅延import（`--json`経路では読み込まない） |
| D5 | 配布 | **PyPIを正（source of truth）**とする（`uv tool install openternary`）。npmは後から薄い入口として追加（下記O1比較） |
| D6 | 版管理 | これからタグ運用を始める（SemVer、`vX.Y.Z`） |
| D7 | ML依存 | `update`はtorch等に触れない。再同期が必要なら案内のみ |
| D8 | 更新確認 | 既定で有効。ネットワークは「確認」にだけ使う。キャッシュ付き、失敗しても起動を妨げない |
| D9 | TUI操作範囲 | 全操作（実行起動・比較・削除含む）。実行前に計画確認、削除は文字入力で確認 |
| D10 | 言語 | 英語・日本語の両対応 |
| D11 | ロゴ | `OPEN-TERNARY`をHermes風の3D AA（ANSI Shadow系） |
| D12 | アイコン | 三つ巴をモチーフ。3つの巴＝三値 {-1, 0, +1}。**巴ごとに色を塗り分ける** |
| D13 | 回転方向 | 頭が進行方向（頭が先、尾が後ろに流れる） |
| D14 | アイコン配色 | 公式SVGの配色を使う: +1 = amber/orange、0 = mint/cyan、-1 = blue/violet、外周リングは連続した虹色（`docs/brand/`） |
| D15 | UIパレット | ラボのアイコン（金・銅）には合わせない。**Yoru（夜）に決定** |
| D16 | シンボル | **オーナー作成の公式SVG v1.0**（`docs/brand/openternary-v1.0.svg`）を採用。ストック画像を参考に作った組紐（A）・渦（B）の再構成は破棄 |
| D17 | アニメーション | **時計回り**（頭が先に進む向き。2026-11-03 オーナー決定）。`OPENTERNARY_SPIN=ccw`・`--ccw`・`mark(..., clockwise=False)`で反転。資産は反時計回りで格納し、時計回りは逆順再生。48コマ・12fps＝1周4秒。非TTYと`OPENTERNARY_REDUCED_MOTION`/`NO_MOTION`では静止 |
| D18 | 環境構築 | `openternary setup`を追加（P4より先）。GPUを自動検出し、CUDA / ROCm / MPS / CPUのPyTorchを**`openternary`を実行中の環境**に入れる（2026-11-03 オーナー決定）。D7（MLに触れない）はこの明示コマンドに限って撤回 |
| D19 | 対応範囲 | Windows（CPU/NVIDIA/AMD）、Linux（CPU/CUDA/ROCm）、WSL、macOS（MPS/CPU）すべて |
| D20 | バージョン選択 | オーナーによる事前検証は行わない。配布元から**その場で**一覧を取得し、互換のあるものを全て提示。印は「おすすめ」「このPCで自動確認済み（インストール後のデバイス検査の記録）」 |
| D21 | 構成の記録 | `<環境>/openternary-torch.json`。`doctor`がずれを警告、`setup --restore`で復元、P4の`update`は更新後にこれを再適用する |
| D22 | TUIの実行方式 | ジョブは通常のCLIを子プロセスで起動（`--json --events-jsonl`）。TUIは描画と監視だけ。フォームはCLI定義から自動生成、開始前に`--dry-run`の計画を表示（失敗なら開始不可）。中止は割り込み→終了コード130、2回目で強制停止 |
| D23 | 起動の入口 | console scriptを`openternary.cli.main:run`へ変更（引数なし＋TTYのときだけTUI。Typerの`app`と`--json`経路は不変で、Textualを読み込まない） |

### 未決事項

- ~~O4 回転方向~~: 解消（D17、時計回り）。
- ~~O3 参考画像の出典~~: 解消。オーナー作成のオリジナルSVGに置き換え、ストック由来の再構成（`scripts/design/fit/`、比較画像5・6）はリポジトリから削除した。ただし、このブランチの過去のコミットには残っているので、mainへは squash merge で取り込む。

## O1: npm と PyPI の比較

事実（2026-11-01確認）: `openternary`はnpm・PyPIとも未登録（registry 404）。`open-ternary`もnpm未登録。

| 観点 | PyPI（`uv tool install openternary`） | npm（`npm i -g openternary`） |
|---|---|---|
| 本体の格納 | wheelをそのまま配布 | npmにはPython wheelを載せられない。Nodeラッパーが初回起動時にuv/Pythonを用意し、wheelを**別の場所（PyPIかGitHub Releases）から**取得する二段構成になる |
| 利用者の前提 | Python（またはuv単体） | Node.js + Python/uv |
| 依存解決 | uvが`[ml]`などのextrasを扱う | ラッパーがuvを呼ぶので結局同じ処理 + Node層の保守 |
| 更新確認 | `https://pypi.org/pypi/openternary/json` | `https://registry.npmjs.org/openternary/latest` |
| ROCm torch保護 | `uv tool upgrade`は対象toolの環境だけを更新。ML環境は別に保つ設計が必要 | 同じ問題 + ラッパーが環境を作り直す危険 |

推奨: **PyPIを正（source of truth）とし、必要ならnpmは薄い入口として後から追加する。** npmだけでは完結せず、どちらにせよwheelの配布先が必要になる。

## アーキテクチャ

```
openternary <subcommand> [--json]   ← 既存。契約は不変（tests/test_cli_json_contract.py で固定）
openternary / openternary ui        ← Textual App（TTYのみ）
   └─ 実行は子プロセス:  python -m openternary <cmd> --json --events-jsonl <file>
        - 既存契約をそのまま消費。GPU OOMなどでTUIが巻き込まれない
        - Ctrl+C → 子プロセスへ伝播（exit 130 / run lock / last_failure.json は既存のまま）
openternary/ui/
   brand.py   パレット・3Dワードマーク・三つ巴マーク（実装済み・草案）
   theme.py   状態バッジ・意味的スタイル（P3）
   i18n.py    en/ja メッセージカタログ（P3）
   app/       Textual 画面群（P5）
```

### 状態表示の原則（正直さ）

`execution=completed`は品質合格を意味しない（`docs/CLI.md`）。UIもこれを守る。

| 状態 | 記号 | 色 |
|---|---|---|
| accepted（品質受入済み） | ✔ | sage |
| completed（処理完了・品質未評価） | ● | champagne（緑にしない） |
| running | ◐ | gold |
| planned | ○ | ash |
| failed / interrupted | ✖ | copper |

色だけに頼らず、必ず記号と文字列を併記する。

## ブランド（確定: 公式SVG v1.0・Yoru）

実装: `src/openternary/ui/brand.py`（パレット・ワードマーク・マーク読込）、`ui/splash.py`（回転デモ: `python -m openternary.ui.splash`）、`ui/assets/tomoe.otmk`（`scripts/design/bake_mark.py`が公式SVGをChromiumで正確にラスタライズして生成。16/24/30桁 × 48コマ、38KB、標準ライブラリだけで読込）。資産の拡張子は`.otmk`。`.bin`は`.gitignore`とsdistの除外対象（モデル重み用）になっているため使わない。

以下は選定時の比較記録。

画像: `docs/design/candidates/`（生成: `scripts/design/`、要 numpy・scipy・pillow）

### ワードマーク
ANSI Shadow系の3D文字。面（`█`）にアイコンと同じ知覚的虹色グラデーション、ストローク（`╔═╝`）は影としてborder色。100桁未満では1行版。

### 三つ巴（参考画像から4スタイル）

| 候補 | 特徴 | ターミナル実寸での判読性（観測） |
|---|---|---|
| A 組紐 | 3本の輪が上下に編まれる | 30桁で構造は読めるが煩雑。16桁では潰れる |
| B 渦 | 頭が中心、尾が渦状の帯 | 帯の隙間が点字ドット未満で消える。採用時は帯の幅・隙間を作り直す必要あり |
| C 大頭・細線 | 大きな頭、面を細線で分割（参考は二つ巴→三つに変更） | 細線は無調整だと消える。ヒンティング後は16桁でも判読可 |
| D 風車 | 中心の隙間が風車状に開く | ヒンティング後は16桁でも最も明瞭 |

**発見（観測）**: 点字ドット1個の間隔は30桁で約0.033（単位円基準）。デザイン上の細線（0.032）はこれ未満なので描画されない。表示サイズごとに隙間を最低1.6ドットへ広げる「光学サイズ調整（ヒンティング）」を行う。

### 虹色
OKLCHで明度・彩度を固定し、色相だけを回す（HSVの虹と違い、黄色だけ眩しい・青だけ暗いといった明るさのムラが出ない）。巴kは色相 `25° + 120k` から `+120°` を担当し、頭→尾の位置は形の内側を通る測地距離で測る。Google/Geminiのブランド色そのものは使わない（他社ブランドとの混同を避けるため）。ライトテーマでは明度を下げる（L 0.74 → 0.60）。

### UIパレット候補
虹色アイコンを主役にするため、UIの地は無彩色寄りにする。値はOKLCHで設計し、検証は機械的に行った。

| 案 | mode | bg / text / accent | コントラスト text / muted / 状態色の最小値（WCAG） | 成功↔失敗の色差 ΔE（通常 / 1型 / 2型 / 3型色覚, Machado 2009） |
|---|---|---|---|---|
| Sumi 墨 | dark | `#0e0f12` / `#e5e8ec` / `#97c9fd` | 15.59 / 6.14 / 6.69 | 32.2 / 27.7 / 18.6 / 38.1 |
| Yoru 夜 | dark | `#090c1c` / `#e9ebf3` / `#b89ffe` | 16.32 / 7.21 / 6.78 | 32.2 / 27.7 / 18.6 / 38.1 |
| Shiro 白 | light | `#f8f7f2` / `#1c1a17` / `#5550c0` | 16.19 / 5.6 / 5.15 | 27.1 / 10.4 / 16.8 / 31.0 |

- 基準: 本文4.5以上（AA）。全案・全色が満たす。
- 初版では成功=緑・失敗=赤で、2型色覚のΔEが3.3（ほぼ同じ色）だった。成功を青緑（hue 195）、失敗を朱（hue 32）にし、さらに明度差をつけて改善した。
- 状態は色だけで区別しない（✔ ● ◐ ✖ ▲ を必ず併記）。
- Textualのテーマ機能でダーク/ライトを切り替える。端末の背景色はOSC 11で問い合わせ、自動選択を試みる（未検証）。

## 作業計画と終了条件

| Phase | 内容 | 終了条件 |
|---|---|---|
| P1 ✅ | `--json`契約の固定（14プローブ）、ブランド部品 | golden一致・CI登録 |
| P2 ✅ | 長時間処理の構造化イベント（追加のみ、schema_version 1のまま）。一覧は`docs/CLI.md`の「JSON・終了コード」 | オーナー実機のsmoke runで欠落・順序乱れなし（下記）。ログとの時刻突合は`logs/`未入手のため未実施 |
| P3 ✅（オーナー確認待ち） | theme（記号のみ着色）・i18n（`OPENTERNARY_LANG`）・helpの3パネル化と内部用語除去・`doctor`/`backends`/`plan`/`plugins`の表表示・エラーの1回パネル化 | golden差分ゼロ（確認済み）、見た目をオーナー承認 |
| P4 ✅（初回リリース待ち） | Apache-2.0、`0.1.0a1`、`release.yml`（Trusted Publishing、タグでPyPI・手動でTestPyPI）、`openternary update`（PyPI確認・1日キャッシュ・導入方法別の更新・setup構成の退避と復元・Windowsは手順表示）、`doctor`の更新通知 | タグ`v0.1.0a1`でPyPI公開、`update --check`が0.1.0a1を最新と表示 |
| P5 ✅（オーナー確認待ち） | Textual App（Home / Runs / Jobs / Compare / Doctor / Update）、`services/jobs.py`（子プロセスのジョブ・イベント追跡・中止）、`services/runs.py`（一覧・削除の安全装置）、`openternary ui`と引数なし起動 | Pilotテスト・実ptyでの起動確認（済）、オーナー実機での長時間ジョブ監視 |
| P6 | `docs/CLI.md`更新、移行表、Windows CI | DoD確認 |

## 実機計測（P2、2026-11-03）

オーナーのWindows PC（CPU実行）で`configs/gemma4-e2b-calibration-smoke.yaml`（205モジュール × 2ステップ、8サンプル × 64トークン）を実行した`--events-jsonl`の集計。

| 段階 | 所要 | 備考 |
|---|---|---|
| 開始〜データセット準備 | 11.7秒 | wikitext初回ダウンロードを含む。当時はイベントがなく無言 → `calibrate.dataset`を追加 |
| 教師モデル読込 | 8.6秒 | |
| 活性値収集 | 6.3秒 | train 7バッチ・heldout 1バッチ |
| 収集後の準備 | 15.2秒 | 無言 → `calibrate.prepare`を追加 |
| モジュール最適化 | 150.7秒 | 1モジュールあたり中央値0.45秒・最大1.33秒（後半層のMLPが重い） |
| 評価 | 26.8秒 | |
| 書き出し | 145.0秒 + hash 6.9秒 | `embed_tokens_per_layer`の処理とshard書込で最大31秒無言 → `materialize.flush`を追加 |
| 結果の作成 | 10.3秒 | 無言 → `calibrate.report`を追加 |
| 合計 | 381.8秒 | オーナー体感の約8分との差は、コマンド開始前（`uv run`・Python・torchの起動）と推測 |

- 1394行・416KB。sequenceと`elapsed_s`は単調増加、全行のキー構成が同一、`completed == total`の最終行は全段階で出ている。
- 書き出し中のRSSはコンソールログで約18.6GB。メモリの少ないPCでは書き出しが最も厳しい段階になりうる。
- 本番設定（50ステップ・32サンプル × 128トークン）の所要は、モジュール最適化が計算量にほぼ比例すると仮定して、このPCのCPUで**おおよそ5〜7時間（推測、未計測）**。

## setupの実機確認（2026-11-03、オーナーPC）

Windows 11 / Python 3.12（uv tool環境） / AMD Radeon RX 9070 XT（ドライバ 32.0.32015.7029）＋Ryzen内蔵GPU。

- 検出: RX 9070 XTは`DEV_7550`から`gfx1201`、内蔵GPUは`gfx1036`（Raphael）。単体GPUを先に表示。
- 配布元: AMD `stable.repo.amd.com/rocm/whl-next`から、torch 2.11.0〜2.14.0（rocm10.0.0 / 10.1.0）の一覧を取得。cp312のwin_amd64のみ。
- インストール: `torch[device-all]==2.14.0+rocm10.1.0`＋`torchvision==0.29.0a0+rocm10.1.0`。66パッケージで約6分。`device-all`は全GPU系統のカーネルを入れていた → 既定を`device-<gfx>`（AMDが個別パッケージ`amd-torch-device-<gfx>`を公開している場合）に変更。
- 動作確認: 実デバイスでFP32・BF16の行列積に合格（`device_name: AMD Radeon RX 9070 XT`）。`doctor`に`PyTorch: 2.14.0+rocm10.1.0 · rocm · amd ✔ passed`。
- 判明して直したこと: 再実行時に確認済みの2.14.0を未確認の2.13.0へ置き換える提案をしていた → このPCで確認済みの版を最優先にし、入っている版が確認済みなら何もしない。
- 未確認: NVIDIA（Windows/Linux）、Linux/WSLのROCm、macOS（MPS）。いずれも実機での確認はまだない。

## 既知の事実（P1で判明）

- `backends list`/`info`、`cache list`の`--json`では`command`が`list`/`info`になる（グループ名を含まない）。現行の契約として固定し、変更しない。
- `NO_COLOR`では色を消し、太字などは残す（no-color.orgの仕様どおり）。パイプ出力では装飾なし。
- 起動時のimport時間: CLI全体で約0.26秒、`textual.app`の追加分は約0.2秒（`-X importtime`）。

## P5 実装と検証（2026-11-03）

構成: `ui/app/app.py`（画面）、`ui/app/dialogs.py`（ジョブのフォーム・計画・名前入力での削除・終了確認）、`ui/app/catalog.py`（CLI定義→フォーム）、`services/jobs.py`、`services/runs.py`。

- 実行方式（D22）: TUIのプロセスでは重い処理をしない。ジョブごとに`job.json`・`events.jsonl`・`stdout.json`（最後の行がJSON封筒）・`stderr.log`を残し、TUIを開き直しても過去と実行中のジョブを表示できる。別セッションで起動したジョブは、pidと起動時刻で同一プロセスか確かめる（pidの再利用対策）。
- 進捗: `events.jsonl`を追記中に読み、未完成の最終行は次回に回す。段階ごとに完了数/総数・単位・詳細（lossなど）を表示し、ETAはその段階の平均速度から出す（測れるまでは表示しない）。
- 正直さ: 実行の完了（`completed`、●）と品質の受入（`accepted`、✔）を分けて表示する。比較画面にも「結果は観察であり、品質ゲート合格だけが受入」と明記した。
- 中止: POSIXは`start_new_session`で別グループにし、`SIGINT`をグループへ送る。WindowsはCREATE_NEW_PROCESS_GROUPで起動してCtrl+Breakを送り、子（`OPENTERNARY_BREAK_IS_INTERRUPT=1`のときだけ）がSIGBREAKを`KeyboardInterrupt`に変える。どちらもCLIの通常の中断処理（終了コード130・失敗の記録）に乗る。
- 削除: 実行結果フォルダは`runs/`直下で`metrics.json`と`config.yaml`を持つものだけ。別プロセスが`.execution.lock`を持っていれば拒否。名前を正確に入力するまで削除ボタンは押せない。

検証（このセッション、Linux）:

| 確認 | 結果 |
|---|---|
| `tests/test_jobs.py`（10件）: 部分行の読み込み、ETA、完了・中止（130）・強制停止、別セッションの記録、実CLIの子プロセスで`doctor`の封筒とイベント、dry-runの計画、run一覧、削除の安全装置（ロック保持中は拒否） | 全件合格 |
| `tests/test_tui.py`（12件）: 全コマンドのフォーム生成と隠す引数、引数の組み立て、数字キーとフォーカス、名前入力なしでは削除不可、フォーム→計画→開始でジョブ起動、dry-run失敗時は開始不可、実行中の終了確認、非TTYでは従来のhelp・`ui`は終了コード2、`doctor --json`でTextualを読み込まない | 全件合格 |
| wheelを作って入れ、実際の疑似端末（pty）で引数なしの`openternary`を起動 | 画面を描画し、`q`で終了コード0 |
| 画面の目視（130×40、日本語） | ホーム・実行結果・ジョブ（進捗とETA、中止→interrupted/130）・比較・診断・更新・フォーム・計画画面を確認 |

未確認: Windows実機でのCtrl+Breakによる中止と、長時間ジョブ（本番のcalibrate）の監視。WindowsはCIで同じテストが走る。
