# CLI実装・検証状況

更新: 2026-09-23。学習済みモデルを使わず、CLI・実ライブラリ・人工ネットワークの保存後再読込まで検証した。

## 現在の判定

**今回の計画を、Windows／WSLのCPU wheel、WindowsのRX 9070 XT FP32・BF16、人工LlamaのGGUF CPU実行まで完了。** CLI-0〜8はこの限定されたfixture範囲で接続を確認できた。ロードマップ全体の安定版Gateや実モデル品質合格ではない。

学習済みモデルの取得・ロード・推論・校正は未実施。既存の研究run・モデルcache・`.venv`／`.venv-rocm`のpackageは保持。研究source、設定、pyproject、uv.lockの計20ファイルは今回開始時のhashと一致した。この検証記録はcommit・push前に採取したもの。公開releaseは未実施。

## 環境別の対応表

| 対象 | Windows CPU | WSL Ubuntu 24.04 CPU | Windows RX 9070 XT |
|---|---|---|---|
| 人工Llama Ternary、pass、層別preserve、packed | 実ライブラリ検証済み | 実ライブラリ検証済み | FP32/BF16 forward・別process再読込済み |
| TorchAO 0.18.0 INT8変換・保存・再読込 | 実ライブラリ検証済み | 実ライブラリ検証済み | FP32/BF16 forward・別process再読込済み |
| Diffusers 0.40.0、UNet差し替え、32×32・2 step | 実ライブラリ検証済み | 実ライブラリ検証済み | FP32/BF16生成・有限値・保存済み |
| schema 3評価・実探索worker・再開 | 人工fixtureで検証済み | 人工fixtureで検証済み | 演算・資源測定まで。品質探索はCPUで検証 |
| 固定版GGUF | 隔離converterとreaderで人工Llama FP32検証済み | 同版CPU runtimeで最大8 token生成済み | GGUF GPU実行は未検証 |
| wheel導入・実entry point plugin | 検証済み | 検証済み | 複製ROCm環境にwheel導入済み |
| 学習済みモデル／画像品質 | 実モデル未検証 | 実モデル未検証 | 実モデル未検証 |
| native packed ternary kernel | 非対応 | 非対応 | 非対応 |
| native低bit kernelの認定 | 未認定 | 未認定 | GPU実行と区別し未認定 |

WSLの検証環境はCPU版Torch。WSLでのROCm試験は環境構成の対象外で未検証。Gemma/Mistral/Qwen等の実モデル対応へ結果を広げない。

## ロードマップとの対応

| Phase | 今回確認した内容 | 残るGate |
|---|---|---|
| CLI-0/1 | 除外6ファイル復帰、cwd非依存、実process排他・中断回収、容量不足からの復旧、明示runtime probe | 実モデルの既存経路回帰 |
| CLI-2/3 | 人工Llama、実TorchAO、人工Diffusers pipelineの変換と再読込 | architecture・backendごとの実モデル品質・資源 |
| CLI-4 | noop/clip順序、保存後のclip結果、preserve層のbyte一致 | 新しいpass研究は対象外 |
| CLI-5 | 全tensor inventory、shape/dtype/有限値、packed全重みbyte一致、隔離GGUF converter、実reader/CPU生成 | native packed実行は非対応。Llama FP32以外のGGUF認定は未実施 |
| CLI-6 | quality schema 3、path非依存source/interface、実device/環境、load/推論のRAM/VRAM/時間 | 実モデルでの品質・性能受入、画像の人間評価 |
| CLI-7 | 制御fixtureの成功選択・全候補不合格・予算・再開・改変拒否、実人工モデルworkerと再開 | 実モデルの有用な候補選択とVRAM/速度制約 |
| CLI-8 | Windows/WSLのinstalled wheel、実plugin衝突/API拒否、CIへの共通script接続、license棚卸し | 各commitのGitHub CI合格、license TBDの決定、署名・公開 |

## 最終検証

- Windows CPU offline: **326 passed / 2 skipped / 0 failed**（全328件、ファイル一括除外なし）。Ruff/format全139ファイル、mypy全93ファイルが通過。
- 上記2 skipはROCm必須の既存人工teacherテスト。別途、複製ROCm環境のinstalled wheelで **2 passed**。
- Windows／WSL installed wheel: runtime、Ternary、TorchAO、Diffusion、quality/searchの**各5項目通過**。sourceの`src`ではなく各環境のsite-packagesを確認。
- Windows GPU: FP32・BF16それぞれruntime、Ternary、TorchAO、Diffusionの**各4項目通過**。
- Windows／WSLの実entry point: 正常導入、予約名衝突、同名重複、API不一致を通過。テスト用pluginは検証後に除去。
- WSLのprocess/tensor/schema契約テストも**13 passed**。成功する制御workerと人工ネットワークの品質不合格は別証拠。
- packedは値・dtype・shape・byte一致。TorchAOはper_tensor INT8を検証し、CPUの量子化直後対保存後出力は厳密一致。GPU FP32の別process出力差は最大約3.58e-7、記録済み許容誤差rtol=1e-5/atol=1e-6内。BF16再読込は今回厳密一致。これは品質閾値の変更ではない。
- 計測GPU VRAM（Torch allocator）の最大値は約84.6MiB。GPU関連試験の壁時計合計は失敗試行・起動時間を含め約650秒、30分上限内。ドライバの総常駐量や速度優劣の認定ではない。
- 実装検証終了時の追加領域は約11.3GiB、人工成果物は4GiB未満。値・log/hashは[証拠記録](CLI_VALIDATION_EVIDENCE.md)。push後のGitHub CI結果は[Actions](https://github.com/ELRdn/OpenTernary/actions/workflows/ci.yml)でcommitごとに確認する。

### push後のCI修正

最初の[GitHub CI実行](https://github.com/ELRdn/OpenTernary/actions/runs/35824126877)ではWindows/Linuxの実ライブラリwheel試験が通過した。一方、Torchなし環境の型推論、Windows cp1252でのヘルプ出力、ANSI装飾付きヘルプの文字列検査が失敗したため、bytes戻り値の明示・ヘルプ文言・装飾を除いた同一項目検査を修正した。cp1252の別process回帰テスト2件を追加し、ローカルで関連31件とcore環境のmypyを通した。上記326件の記録は追加前のもので、最新の全件結果は修正commitのCIを参照する。

## 見つかって修正した不具合

1. Diffusers componentの出力shard/index名がTransformers規約だったため、Diffusersから読めなかった。標準Diffusers名にして、読み戻した全stateが保存tensorと一致することも確認する。
2. TorchAO loaderの保存dtypeが推論dtype指定を上書きしていた。tensor subclassを保ったまま指定dtypeへ変換し、BF16実行で確認した。
3. GGUF converterと探索workerのプロセス回収を共通化し、タイムアウト・中断で子孫を回収する。
4. 旧quality schemaの受入を止め、source/tokenizer/runtimeが一致するschema 3だけを受入候補にする。人工fixture・由来未解決は合格しない。評価APIのseed固定とseed/warmup比較も実施する。研究の数式・閾値・splitは維持した。

## 次の実モデルGate

対象モデルと環境を固定し、同じ検証scriptの受入項目を実モデルへ適用する。その段階で品質・実メモリ・速度・互換性を評価する。今回の結果だけで、Gemmaの研究品質回復や任意Diffusionモデルの対応を宣言しない。
