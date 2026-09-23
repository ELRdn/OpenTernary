# CLI実装・検証状況

更新: 2026-09-24。人工fixtureに加え、Gemma 4 E2Bと小型pretrained DiffusersをRX 9070 XTで保存後再読込まで検証した。

## 現在の判定

**CLI-0〜8の製品実装と、固定したpretrainedモデルでの実挙動検証まで完了。** Windows／WSLのCPU wheel、WindowsのRX 9070 XT FP32・BF16、人工LlamaのGGUF CPU実行に加え、Gemma 4 E2BのBF16・Ternary・TorchAO INT8、pretrained tiny Stable DiffusionのUNet差し替えを確認した。

Gemma 4のTernary経路は変換・保存・packed round-trip・別process推論まで成功したが、品質Gateは不合格。TorchAO 0.18.0 INT8 weight-onlyは同じvalidation protocolで品質Gateに合格した。単一validation評価なので研究上の最終結論にはしない。TorchAOのGPU演算は `to → mm → mul` を観測し、native INT8 matmulとは認定しない。既存の研究run・モデルcache・`.venv`／`.venv-rocm`は保持。公開releaseはproject licenseの権利者判断待ち。

## 環境別の対応表

| 対象 | Windows CPU | WSL Ubuntu 24.04 CPU | Windows RX 9070 XT |
|---|---|---|---|
| 人工Llama Ternary、pass、層別preserve、packed | 実ライブラリ検証済み | 実ライブラリ検証済み | FP32/BF16 forward・別process再読込済み |
| TorchAO 0.18.0 INT8変換・保存・再読込 | 実ライブラリ検証済み | 実ライブラリ検証済み | FP32/BF16 forward・別process再読込済み |
| Diffusers 0.40.0、UNet差し替え、32×32・2 step | 実ライブラリ検証済み | 実ライブラリ検証済み | FP32/BF16生成・有限値・保存済み |
| schema 3評価・実探索worker・再開 | 人工fixtureで検証済み | 人工fixtureで検証済み | 演算・資源測定まで。品質探索はCPUで検証 |
| 固定版GGUF | 隔離converterとreaderで人工Llama FP32検証済み | 同版CPU runtimeで最大8 token生成済み | GGUF GPU実行は未検証 |
| wheel導入・実entry point plugin | 検証済み | 検証済み | 複製ROCm環境にwheel導入済み |
| Gemma 4 E2B pretrained | header・変換は可能 | 未検証 | BF16/Ternary/TorchAOの別process推論・品質比較済み |
| pretrained Diffusers | 取得・safetensors正規保存済み | 未検証 | UNet三値化・差し替え・32×32生成済み。人間品質は未受入 |
| native packed ternary kernel | 非対応 | 非対応 | 非対応 |
| native低bit kernelの認定 | 未認定 | 未認定 | TorchAOは通常matmulを観測。native INT8/ternaryは未認定 |

WSLの検証環境はCPU版Torch。WSLでのROCm試験は環境構成の対象外で未検証。Gemma/Mistral/Qwen等の実モデル対応へ結果を広げない。

## ロードマップとの対応

| Phase | 今回確認した内容 | 残るGate |
|---|---|---|
| CLI-0/1 | 除外6ファイル復帰、cwd非依存、実process排他・中断回収、容量不足からの復旧、明示runtime probe | 実モデルの既存経路回帰 |
| CLI-2/3 | 人工Llama、Gemma 4、実TorchAO、人工／pretrained Diffusersの変換と再読込 | 追加architecture・native kernelは個別認定が必要 |
| CLI-4 | noop/clip順序、保存後のclip結果、preserve層のbyte一致 | 新しいpass研究は対象外 |
| CLI-5 | 全tensor inventory、shape/dtype/有限値、Gemma 4のpacked 10,208,596,934 byte完全一致、隔離GGUF converter、実reader/CPU生成 | native packed実行は非対応。Llama FP32以外のGGUF認定は未実施 |
| CLI-6 | quality schema 3、同一validationでGemma BF16/Ternary/TorchAO比較、実RAM/VRAM/時間 | 画像の人間評価と別datasetでの再現は未実施 |
| CLI-7 | 制御fixtureの成功・不合格・予算・再開・改変拒否。実Gemma workerの容量停止・resume・回収 | 実Gemmaの成功候補は通常経路で実測済み。探索内の重複8GB生成は省略 |
| CLI-8 | Windows/WSLのinstalled wheel、実plugin衝突/API拒否、CIへの共通script接続、license棚卸し | 各commitのGitHub CI合格、license TBDの決定、署名・公開 |

## 最終検証

- Windows CPU offline: **330 passed / 2 skipped / 0 failed**（全332件、ファイル一括除外なし）。Ruff/format全139対象ファイル、mypy全93 source fileが通過。追加pretrained harnessも単体Ruff/format通過。
- 上記2 skipはROCm必須の既存人工teacherテスト。別途、複製ROCm環境のinstalled wheelで **2 passed**。
- Windows／WSL installed wheel: runtime、Ternary、TorchAO、Diffusion、quality/searchの**各5項目通過**。sourceの`src`ではなく各環境のsite-packagesを確認。
- Windows GPU: FP32・BF16それぞれruntime、Ternary、TorchAO、Diffusionの**各4項目通過**。
- Windows／WSLの実entry point: 正常導入、予約名衝突、同名重複、API不一致を通過。テスト用pluginは検証後に除去。
- WSLのprocess/tensor/schema契約テストも**13 passed**。成功する制御workerと人工ネットワークの品質不合格は別証拠。
- packedは値・dtype・shape・byte一致。TorchAOはper_tensor INT8を検証し、CPUの量子化直後対保存後出力は厳密一致。GPU FP32の別process出力差は最大約3.58e-7、記録済み許容誤差rtol=1e-5/atol=1e-6内。BF16再読込は今回厳密一致。これは品質閾値の変更ではない。
- 計測GPU VRAM（Torch allocator）の最大値は約84.6MiB。GPU関連試験の壁時計合計は失敗試行・起動時間を含め約650秒、30分上限内。ドライバの総常駐量や速度優劣の認定ではない。
- 実装検証終了時の追加領域は約11.3GiB、人工成果物は4GiB未満。値・log/hashは[証拠記録](CLI_VALIDATION_EVIDENCE.md)。push後のGitHub CI結果は[Actions](https://github.com/ELRdn/OpenTernary/actions/workflows/ci.yml)でcommitごとに確認する。

### push後のCI修正

最初の[GitHub CI実行](https://github.com/ELRdn/OpenTernary/actions/runs/35824126877)ではWindows/Linuxの実ライブラリwheel試験が通過した。一方、Torchなし環境の型推論、Windows cp1252でのヘルプ出力、ANSI装飾付きヘルプの文字列検査が失敗したため、bytes戻り値の明示・ヘルプ文言・装飾を除いた同一項目検査を修正した。cp1252の別process回帰テスト2件を追加した。2026-09-24のローカル最終結果は上記332件。修正を含むcommit `eb68da566658cb5abdc8dcfbf0771d889a67bf16` の[最終GitHub CI](https://github.com/ELRdn/OpenTernary/actions/runs/35881121188)は、Windows/Linuxのcore、ML fixture、実ライブラリwheel、pip fallbackの全10 jobが成功した。

## 見つかって修正した不具合

1. Diffusers componentの出力shard/index名がTransformers規約だったため、Diffusersから読めなかった。標準Diffusers名にして、読み戻した全stateが保存tensorと一致することも確認する。
2. TorchAO loaderの保存dtypeが推論dtype指定を上書きしていた。tensor subclassを保ったまま指定dtypeへ変換し、BF16実行で確認した。
3. GGUF converterと探索workerのプロセス回収を共通化し、タイムアウト・中断で子孫を回収する。
4. 旧quality schemaの受入を止め、source/tokenizer/runtimeが一致するschema 3だけを受入候補にする。人工fixture・由来未解決は合格しない。評価APIのseed固定とseed/warmup比較も実施する。研究の数式・閾値・splitは維持した。

## pretrainedモデルの結果

| 経路 | 実行・artifact | 品質／資源の観測 |
|---|---|---|
| Gemma 4 BF16 | 5 prompt、320 token、別process GPU推論 | 6.12 tok/s、推論VRAM約10.73GB。general PPL 1214.61、日本語2229.70、instruction 62.5、collapse 0 |
| Gemma 4 Ternary G128 | 1951 tensor、205対象、12 shard。packed→復元後も全10,208,596,934 byte一致 | 6.63 tok/s、推論VRAM約10.30GB。PPL悪化・instruction 0・collapse 8でGate不合格 |
| Gemma 4 TorchAO INT8 | 1951 tensor、7.83GiB、別process GPU推論 | 18.12 tok/s、推論VRAM約8.51GB。general PPL 1201.52、日本語2078.28、instruction 62.5、collapse 0、Gate合格 |
| pretrained tiny Stable Diffusion | UNet 304 tensor中60対象、全tensor reload、BF16 GPU生成 | 32×32、2 step、finite pixel、PNG保存。人間品質は `human_review_required` |

TorchAOの数値はこの1回の固定validation観測であり、速度比較の統計的受入や研究上の一般化ではない。Ternary品質の回復、新しいkernel、追加architectureは研究・個別backendの次段階。CLI製品として残る公開ブロッカーはproject license `TBD` だけで、権利者決定なしにstable releaseへ進めない。
