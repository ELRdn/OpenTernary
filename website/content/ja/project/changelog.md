---
description: "OpenTernaryの各リリースでの変更。"
---

# 変更履歴

バージョンは[PEP 440](https://peps.python.org/pep-0440/)に従います。プレリリース（`aN`）では、バージョンの間で動作が
変わることがあります。`--json` の結果、終了コード、進捗イベントのschemaはバージョン付きの約束事で、変更があれば
ここに書きます。

## 0.1.0a2 — 2026-11-03 {#010a2-2026-11-03}

**全画面インターフェース。**

- **新機能: ターミナルで引数なしの `openternary` を実行するとTUIが開きます**（`openternary ui` でも可）。
  ホーム・実行結果・ジョブ・比較・診断・更新のタブがあります。
- **選択式の操作**: 画面下の `❯` のプロンプトで `/` を打つと、コマンドが説明付きで一覧されます。文字を打って
  絞り込み、↑↓で選び、Enterで実行、Tabで補完します。確認画面も `❯` のメニューで、ボタンはありません。
- **バックグラウンドのジョブ**: 各ジョブはふつうのCLIの子プロセスで、段階ごとの進捗・速度・ETAと、ログの末尾を
  リアルタイムで表示します。中止は安全な割り込み（終了コード130）、強制停止はプロセスツリーを終了します。TUIを
  閉じてもジョブは続き（Linux・macOS）、開き直すと監視を再開します。
- **CLIの定義から作るフォーム**と、そのコマンド自身の `--dry-run` の出力を見せる**計画画面**。dry-runが成功するまで
  *開始*は選べません。
- **安全な削除**: runの削除と `cache remove --execute` には、対象の名前の入力が必要です。使用中のrunは拒否します。
- 入口が `openternary.cli.main:run` になりました。`openternary.cli.main:app` も引き続き使えます。
- ターミナルがない場合（パイプ・`CI`・`TERM=dumb`・`OPENTERNARY_NO_TUI=1`）、引数なしの `openternary` は以前と
  同じくhelpを表示して終了コード2で終わります。`--json` の処理はTUIのライブラリを読み込みません。
- 修正: editable導入で、古いパッケージ情報ではなくソースのバージョンを表示するようにしました。
- 新しい依存: `textual>=8.2`

## 0.1.0a1 — 2026-11-03 {#010a1-2026-11-03}

**最初のPyPIリリース。**

- **`openternary setup`**: NVIDIA・AMD・AppleのGPUを検出し、PyTorch・AMD・PyPIの一覧からその場でビルドを列挙して
  推奨し、機械学習の依存と一緒に実行中の環境に入れ、デバイスで試験します。`--restore` で記録したビルドを入れ直し
  ます。このPCで検証済みのビルドを優先し、AMDではダウンロードを1つのGPUのカーネルに絞れます。
- **`openternary update`**: PyPIを確認し（1日キャッシュ）、導入方法に合わせて更新し、そのあとPyTorchのビルドを
  戻します。Windowsのuv tool導入では、実行するコマンドを表示します。
- **読みやすい出力**: 人間向けには生のJSONではなく表、状態の記号、失敗ごとに1つのエラーのパネル、英語と日本語の
  help（`OPENTERNARY_LANG`）。
- **進捗イベント**（`--events-jsonl`）: `unit`・`elapsed_s`・`detail` の項目と、校正・記録・materialize・ベンチ
  マーク・品質・書き出し・検査・探索の段階を追加。`schema_version: 1` のままです。
- **ライセンス**: Apache-2.0。NOTICEのファイルと、名称・ロゴの商標の方針。
- **ブランド**: 正式な三つ巴のシンボルと「Yoru」の配色。

### 0.1.0a1より前 {#before-010a1}

最初のリリースより前（2026年8〜9月）に、研究の流れとCLIの製品を作りました:

- Phase 0〜3: 調査、BF16のsmokeの基準、2bitのpackedの集計を備えたAbsMeanの三値の中核、サイズ上限付きのshardでの
  モデル全体のfake量子化
- Phase 4.1/4.2: scaleと閾値を学習する層ごとの校正、チェックポイント、再開、低VRAMのruntime、確定の復旧
- CLI-0〜CLI-8: backend（ternary・TorchAO）、アダプタ（Gemma 4・Transformers・Diffusers）、pass、manifest付きの
  原子的な書き出し、packed形式、GGUFへの橋渡し、品質レポートschema 3と品質Gate、制約付き探索、plugin、
  バージョン付きのJSONの約束事と終了コード

この期間の移行の注意は、各ガイドの「古いバージョンからの移行」と
[docs/CLI.md](https://github.com/ELRdn/OpenTernary/blob/main/docs/CLI.md)にあります。
