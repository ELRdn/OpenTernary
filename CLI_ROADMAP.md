# OpenTernary CLI Product Roadmap

OpenTernaryを、既存Ternaryを組み込みbackendとして維持しながら、LLM・DiT・Diffusionの量子化を共通操作で扱えるCLIへ拡張する。まず実行結果の信頼性を固め、その上にbackend、モデル、変換処理、評価、export、自動探索を順に載せる。

> **Status: IMPLEMENTED / PRETRAINED VALIDATED / 2026-09-24** — CLI-0〜CLI-8の製品機能は実装済み。人工fixture、Windows/WSL wheel、固定版GGUFに加え、Gemma 4 E2BのBF16・Ternary・TorchAO INT8と、小型pretrained Diffusers pipelineをRX 9070 XTで検証した。Ternaryは実行できるが品質Gate不合格。TorchAO INT8 weight-onlyは保存・別process再読込・品質Gate合格まで確認したが、観測演算はdequantize後の通常matmulでありnative INT8 kernelとは認定しない。環境別結果・制限・hash付き証拠は[実装・検証状況](docs/CLI_IMPLEMENTATION_STATUS.md)と[検証記録](docs/CLI_VALIDATION_EVIDENCE.md)、操作仕様は[CLIガイド](docs/CLI.md)を参照。native packed ternary runtimeとproject license決定は実装範囲外、安定版公開はlicense決定待ち。
>
> **正本:** CLI製品開発は本書、研究の進行は[ROADMAP.md](ROADMAP.md)、研究予算・比較条件・品質受入は[P0–P7研究計画](docs/plans/p0-p7-research-acceptance.md)で管理する。研究のPhase/P番号と本書のCLI番号は別体系。

## 1. CLIの完成とモデル品質の合格を分ける

本書が扱うのは、設定の検証、実行の制御、対応可否の判定、成果物の保存、測定・比較、他ツールからの利用である。新しいternary数学、Gemmaの品質回復、論文再現、評価データの変更は研究側で扱う。

既存の品質ゲートが不合格でも、CLIのエラー処理やbackend境界は改善できる。一方、backend統合やexportの成功を、そのモデルの品質合格として扱わない。次の3状態を別々に記録する。

| 状態 | 判定する内容 |
|---|---|
| execution | 指定した処理が完了したか |
| artifact validation | ファイル・重み・メタデータが整合し、対応するloaderで読めるか |
| quality acceptance | 指定した比較条件と品質基準を満たしたか |

Electron/DesktopはCLI-8以降の別計画とする。GUIは同じサービスAPIまたはCLIのJSON/JSONLを利用し、量子化・モデル読込・評価ロジックを持たない。

## 2. 出発点は現在の作業ツリーで固定する

以下は**実装前の2026-09-22監査記録**。現在のコマンド数・検証件数・export対応は上記の実装状況を正本とする。

参照案はGitHub mainの`31564ae`を対象としていた。本書は同じタスク内の2026-09-22監査と、現在のソースを基準に更新した。

| 項目 | 監査時点の状態 |
|---|---|
| Repository / version | `ELRdn/OpenTernary` / `0.1.0a0` |
| Branch / HEAD | `feat/calib-threshold` / `d37ace6c3e75291048a7bd12534618fe7b6d4059` |
| 作業ツリー | 本書追加前に変更済み27ファイル、未追跡25項目。HEADだけでは現状を再現できない |
| 実装済みコマンド | `inspect / quantize / calibrate / benchmark / quality / compare / cache-info` |
| 未実装 | `export`。Backend Registry、汎用モデル層、pass基盤、AutoQuantも未実装 |
| モデル・表現 | Gemma 4 E2B、正準205対象。保存物は浮動小数点のfake-quant snapshot |
| コードの集中 | `cli/main.py` 1,233行、`calibration/runner.py` 3,157行。個別CLIファイルに旧スタブが残る |
| ローカル検証 | 267テストを確認。隔離実行266件とcwd依存1件の再実行が通過。mypyは56ファイル通過 |
| 品質上の負債 | `ruff check src tests`は10件、formatは2ファイル未通過。最新GitHub CIもRuffで失敗 |
| モデル品質 | 保存済みvalidationの三値化6候補は全て不合格。研究P4は再設計が必要 |

実装の入口は[CLI](src/openternary/cli/main.py)、[設定](src/openternary/config/schema.py)、[run管理](src/openternary/experiment/run.py)、[比較](src/openternary/experiment/compare.py)。研究の証拠は[foundation status](docs/PHASE4_FOUNDATION_STATUS.md)を参照する。数値はこの監査時点の記録であり、今後の変更後は該当範囲を再検証する。

## 3. CLI-0からCLI-8までを段階的に進める

| Phase | 目的 | 依存 | 完了を示す証拠 |
|---|---|---|---|
| CLI-0 | 誤成功・設定不一致を直し、既存経路を固定 | 現状監査 | 異常系・並行実行・既存互換性の回帰テスト |
| CLI-1 | Backend・capability・成果物・JSONの共通仕様 | CLI-0 | Ternaryを共通API経由で実行し、重みの同一性を確認 |
| CLI-2 | モデル依存をadapterへ移す | CLI-1 | Gemma以外のLLMとDiffusers構成を区別して検査 |
| CLI-3 | 非Ternary backendを接続する | CLI-1/2、CLI-5a | 対応実機で変換・保存・再読込・実行 |
| CLI-4 | 量子化前後の変換をpassとして構成する | CLI-1/2、対象backend | 適用条件・順序・再読込後の整合性を検証 |
| CLI-5 | 成果物を独立したexporterで保存する | 5aはCLI-1、5b/5cは対応backend | 内容検証、round-trip、対象runtimeでの読込 |
| CLI-6 | 共通評価と複数run比較を整える | CLI-1/2/3/5a | 同条件比較、不一致拒否、測定根拠を保存 |
| CLI-7 | 制約付きAutoQuantとmixed precision探索 | CLI-3/4/5/6 | 予算内探索、再開、実測制約判定 |
| CLI-8 | 配布・plugin・互換性を安定させる | CLI-0〜7 | Windows/Linuxの導入試験と公開候補検証 |

番号は機能群を表す。CLI-5aの成果物保存はCLI-3より先に作り、JSONやcapabilityもCLI-1で導入する。後半にまとめて追加しない。

途中成果は、A「既存CLIの信頼性」= CLI-0、B「複数backendの手動変換」= CLI-1/2/3/5a/6のLLM経路、C「汎用CLI製品」= 全Phaseの3段階で判定する。Bで手動利用を検証できる状態を作り、CではDiffusion、pass、自動探索、配布まで含める。

## CLI-0 — 成功・失敗と実際の処理を一致させる

最初に修正するのは、実験結果を誤読させる経路である。既知の不具合をgolden testで正しい挙動として固定しない。

| Task | 修正対象 | Acceptance |
|---|---|---|
| C0-01 | モデル不在時の合成実験への移行 | 通常の`calibrate`はモデル不在で失敗。合成fixtureはテスト専用入口に分離し、実モデル完了を出さない |
| C0-02 | 設定キー・値・指定内容の不一致 | 未知キーと未対応codebookを拒否。`enabled`、method/threshold、dtype/device、low-VRAMの意味を定義し、表示・記録・実処理を一致させる |
| C0-03 | runディレクトリの競合 | 正規化した出力先の選択と予約を同じロック内で行う。異なるcwdからの実行や同一runへの同時resumeでも、上書き・混在を起こさない |
| C0-04 | compareの誤成功と不完全な失敗記録 | 入力なし・破損・片側のみの品質入力を拒否し、run作成後の失敗は`failed`と理由を保存 |
| C0-05 | 証拠不足のquality report | schemaなしの旧reportは閲覧できても受入判定へ昇格させない。現行schemaの整合性検査を維持 |
| C0-06 | 予算・容量の意味 | preflightの観測値と合否条件を分離。G128合計容量にgroup scaleを含め、理論値・packed見積もり・実ファイル容量を別表示 |
| C0-07 | 巨大CLIと旧スタブ | 各コマンドと共通処理へ分割。`openternary.cli.main:app`は互換入口として維持し、研究数学を変更しない |
| C0-08 | ドキュメントとCI | 存在するflag・suite・snapshotパスで例を検証。ルートとdocs内の重複文書は正本を定め、旧版から誘導する。coreのみのCIとML依存ありのCIを分け、Lint/format/type/testを通す |

通常の`--dry-run`は設定・計画の検証に限定し、run作成、download、モデル実体化、最適化を行わない。ローカルで判断できない項目は`unknown`と理由を返す。実データ・実機確認は明示的なpreflight/doctorへ分離する。

既存の正しい変換結果には、tensorの値・dtype・shapeと内容fingerprintの回帰試験を置く。C0-01〜06の修正後に比較基準を固定し、CLI分割とCLI-1の共通API移行で内容を保持する。時刻やrun IDを含むJSON全文、Richの装飾、shard境界の変更まで無条件に一致させない。エラー仕様を変更する場合は移行表に理由と旧挙動を残す。

**Gate:** C0-01〜08の失敗経路を含むテスト、既存の影響範囲、Ruff、format、mypyが通過する。最初のPRではC0-01だけを扱い、その修正とCLI分割を混在させない。

## CLI-1 — Backendと外部利用の仕様を先に定める

CLIは引数と表示を担当し、処理はPythonのサービスAPIに委譲する。既存Ternaryは薄いadapterで包み、最初から数学やcheckpointを大移動しない。

```text
CLI / 将来のDesktop
        ↓
Application services（plan / execute / evaluate / export）
        ↓
Model Adapter × Backend × Pass Graph × Exporter
        ↓
Artifact manifest / Run record / JSON events
```

| Contract | 責任 |
|---|---|
| ModelAdapter | モデル・componentの識別、読込、role分類、tokenizer等の付属情報 |
| QuantizationBackend | capability照会、計画、準備、校正、変換、backend固有状態の符号化・復元 |
| Pass | 適用条件、前後関係、変換、追加状態、runtime/export要件 |
| Artifact / Exporter | 共通manifest、原子的保存、形式変換、整合性と読込の検証 |
| Evaluator / Comparator | 測定条件、結果、比較可否、明示的な品質ゲート |

Backend Registryへまず`TernaryBackend`を登録する。新しいpackageをimportしなくても`backends list/info`で導入状態と制約を見られるようにする。Ternaryの品質状態は`experimental`として記録する。

capabilityはbackend名だけで決めない。`backend version × scheme × model/component × OS × hardware/runtime × operation`を照合し、変換可能、保存可能、再読込可能、native kernelで実行可能を分ける。未導入・未検証・非対応を別状態にする。CPUへ戻した実行をGPU成功として表示しない。

この段階で`plan`、読み取り専用の`doctor`、JSON出力の最小版を用意する。量子化の構成は`scheme`、重み・活性化の精度は`weight_dtype / activation_dtype`、出力コンテナは`format`で指定する。GGUFは出力コンテナとして扱う。本書の例は設計上の概要であり、現在の必須引数・制限は[CLIガイド](docs/CLI.md)で確認する。

```text
openternary backends list --json
openternary plan MODEL --backend ternary --scheme absmean --json
openternary doctor --json
```

共通manifestはsourceの重み・config・tokenizer・processor・chat templateの識別、backend/version、正規化済み設定、対象と除外理由、pass graph、runtime要件、ファイルhashを保持する。runにはコードのcommit、変更状態と対象ソースhash、依存version、実際のdtype/device、評価データとprotocolを記録する。秘密情報と無関係なファイルは収集しない。

**Gate:** 既存コマンドがサービスとRegistryを経由し、同じ条件のTernary重みは移行前と一致する。CLI・Python APIで同じ検証が働き、optional依存なしでもhelp、設定検証、backend一覧が動く。

## CLI-2 — モデル固有処理をadapterに閉じ込める

Gemma 4の正準205対象は既存adapterの回帰基準として保持する。CLIやwriterにある205、hidden size 1536、教師約9.5 GBなどの固定表示は、inspection結果から導く。

まずTransformersの一般的なLinear、Embedding、Norm、Attention/MLP projectionをroleへ変換する。未知の重み名を推測で量子化せず、理由付きで未対応とする。Gemma以外の最初の実モデルは、このPhaseの開始時にlicense・容量・実行環境を確認して固定する。

次にDiffusersのpipelineを、denoiser/DiT、text encoder、VAEなどのcomponentとして識別する。componentごとに量子化対象を選び、モデルの一部を認識できたことと、pipeline全体を変換・保存できることを分ける。

**Gate:** Gemmaの既存対象一覧を保持し、追加LLMのfixtureと、Diffusersの小規模pipelineを検査できる。CLI coreは`Gemma4Adapter`や特定のTransformers loaderを直接呼ばない。実モデル対応は別途、固定したモデルでread/convert/reloadの証拠を残す。

## CLI-3 — 実機で確認した非Ternary backendを追加する

最初の候補はTorchAO。INT8やFP8などの名称だけで採用せず、最初に1つのschemeと実行環境を絞る。下表は統合候補であり、現在の対応表ではない。

| 候補 | 予定する用途 | 統合前の確認 |
|---|---|---|
| TorchAO | 最初の非Ternary変換 | 固定versionのAPI、scheme、kernel、save/loadの対応 |
| SDNQ | Diffusion/DiT向け候補 | 対応component、補正状態、保存形式、実機kernel |
| AutoRound | 校正を使う低bit PTQ候補 | データ契約、対応モデル、出力loader、依存競合 |
| ModelOpt | NVIDIA向け追加候補 | GPU世代、CUDA、scheme、配布・runtime要件 |

FP8、INT8、INT4、MXFP4、NVFP4は、形式の詳細、weight/activation精度、group設定を明示する。全backendが全形式・全GPUで動くとは仮定しない。Windows/ROCmでの導入可否と、実kernelの動作を別々に確認する。

依存は`openternary[torchao]`等へ分離する。互換性のないPyTorch/runtimeを要求するbackendは、別環境のworkerで扱う判断を含める。既存`.venv-rocm`を統合試験で置き換えない。外部実装の機能・license・APIは統合時点の公式資料で再確認する。

**Gate:** Ternaryと非Ternaryの最低2backendを同じコマンド・manifestで利用できる。各対応行にOS/GPU/runtime/version、変換、保存、新規プロセスからの再読込、実行の証拠がある。Diffusion向けbackendを最低1つ確認するまで汎用CLI全体の完成とはしない。

## CLI-4 — 回転や補正を順序付きpassとして扱う

`INT8 + ConvRot`のような構成を、量子化方式と変換処理の組として記録する。候補はHadamard/rotation、ConvRot、SmoothQuant、clipping、SVD correction、sensitivity analysis。列挙しただけで実装・互換性を認定しない。

passは前処理・後処理・解析を区別し、必要な校正データ、対象shape、前後関係、追加パラメータ、runtime側の演算変更を宣言する。weightだけを回転してactivationや下流演算との対応を失う構成は拒否する。backend内蔵変換との二重適用も検出する。

`plan`は順序、対象、除外、必要なデータ・依存、保存時の表現を示す。実行できないgraphには理由を返す。

**Gate:** no-op passで既存内容を保持し、最低1つの実passについて変換前後の意味、保存・再読込後の整合性を確認する。未対応のbackend/pass/exporterの組は実行前に拒否する。新しい回転手法の研究は研究ロードマップへ戻す。

## CLI-5 — 保存・packed表現・runtime連携を分けて完成させる

manifestとExporter APIはCLI-1で定義する。quantizeは変換済み状態を作り、exportはその状態を指定形式で永続化する。既存の`quantize --output`は、サービス内で既定exporterを呼ぶ互換操作として維持できる。

| 段階 | 成果物 | Gate |
|---|---|---|
| CLI-5a | 現行fake-quant/safetensorsとbackend固有状態 | source非破壊、原子的公開、全ファイルhash、新規プロセスでreload |
| CLI-5b | true packed ternary | 2bit code、group scale、shape、対象外重みを保存。codebook・bit配置・padding・scale dtype・形式versionを記録し、pack/unpackと再構築したtensorの一致を確認 |
| CLI-5c | GGUF等のruntime向けbridge | 対応するarchitecture・表現・変換toolを固定し、実際の対象runtimeでロード・実行 |

2bitの保存ができても、packedのまま高速推論できるとは限らない。通常精度へ展開して実行する場合は、そのRAM/VRAMを実測して示す。GGUFへの変換で再量子化が必要なら、それを新たな派生成果物として記録する。任意のTernary、FP8、DiffusionをGGUFへ損失なく移せるとは約束しない。

```text
openternary export RUN --format safetensors
openternary export RUN --format ternary-packed
openternary export RUN --format gguf
```

公開するformatは対応組合せが確認できたものに限定する。未対応formatは空runを作って成功させず、変換開始前に拒否する。hashとpayloadの同一性、container file hash、推論結果の許容誤差はそれぞれ別基準で検証する。

## CLI-6 — 同じ条件の測定だけを比較する

既存`benchmark`のsmoke、`quality`のPPL・指示回答・崩壊検出を残し、共通Evaluatorへ委譲する。`quality`の改名や吸収を行う場合も互換入口を先に用意する。

LLMではtokenizer、chat template、文脈長、stride、生成条件を、Diffusionではpipeline/component、prompt、seed、scheduler、steps、解像度を固定する。速度比較はhardware/runtime、warmup、同期、繰り返し条件も揃える。異なる条件の並列表は作れても、公平な優劣や品質合格には使わない。

`compare`は2 run互換を保ったまま複数runへ拡張する。実ディスク容量、load時と推論時のRAM/VRAM、latency、throughput、品質を別項目で表示し、未測定値を0にしない。fake quantの理論圧縮率を実圧縮率として扱わない。

品質測定用のprofileとgateはversion・hash付きで保存する。既存のbalanced gateはLLM研究profileに保持し、画像のproxy指標へ流用しない。画像proxyでの通過だけを人間による品質受入としない。

**Gate:** source/interface、dataset/split、protocol、実際のdtype/deviceなどの整合性を検証する。欠測・破損・旧schemaでの不正合格を拒否し、複数backendの比較を同じ形式で保存できる。

## CLI-7 — 実測制約と予算の中でAutoQuantを行う

自動探索は、手動で変換・export・測定できる候補だけを対象にする。最初は限定した探索集合で成立させ、その後に層別mixed precisionへ進む。

```text
openternary optimize MODEL --target-vram 12GiB --quality-profile PROFILE --objective speed
```

品質低下「1%」のような曖昧な指定は避け、profileで比較元、指標の方向、相対変化かポイント差か、許容値を定める。探索にはvalidationを使い、最終testを候補選択へ流用しない。研究profileを使う場合は、既存の研究予算と品質ゲートを引き継ぐ。

処理はinspect → capability →候補生成 → quantize/pass →保存・reload →測定 →制約判定の順。全候補に成功を求めず、OOM・未対応・品質不合格・中断を理由付きで残す。制約を満たす候補がなければ`no_feasible_candidate`を返し、失敗候補をbestとして公開しない。

GPU時間・wall time・候補数・active artifact容量を制限し、新規候補を始める前にも残量を確認する。resumeは完了済み候補を重複実行せず、失敗途中の出力をcache hitにしない。cache keyにはsource/interface、設定、backend/version、pass graph、データ、seed、評価条件、runtimeを含める。

**Gate:** 小規模fixtureで探索・予算停止・再開・候補なしを再現できる。代表実モデルでは、保存後の候補を測定し、実VRAM・品質・速度の条件を満たすことを確認する。元モデル・既存研究run・共有cacheを自動削除しない。

## CLI-8 — 配布と拡張の互換性を維持する

Windows/Linuxでcore-only、ML、選択したoptional backendの導入試験を分離する。ハードウェアがないCIでは実kernelの合格を主張せず、release matrixに実機検証の環境・日付・versionを添える。

CLI-1の仕様に対して後方互換・schema migration・非推奨期間を定め、別cwdからの実行、空白・日本語path、Ctrl+C、中断後の再開、ディスク不足も検証する。第三者backend/pass/exporter/evaluatorはversion付きPython entry pointで登録できるようにし、重複名やAPI不整合を起動時に識別する。pluginの自動downloadや自動実行許可は行わない。

`doctor`は診断のみとし、install・環境再構築を勝手に行わない。`cache`は状態と参照関係を表示し、削除は明示的な対象・dry-run・実行操作に分ける。

**Gate:** clean環境でinstall → inspect → plan → quantize →export →reload →benchmark/quality →compareが通り、対応表と同じ範囲で動く。project licenseのTBD、外部backendのlicense・帰属、配布方法を解消してから安定版公開を判定する。

## 4. JSON・終了コード・設定移行はCLI-1で仕様化する

以下は設計上の仕様。2026-09-23にschema version 1を実装した。現行の終了コード・互換性表は[CLIガイド](docs/CLI.md)を参照する。

| 終了コード | 意味 |
|---|---|
| 0 | 指定操作が完了。品質合格とは別 |
| 1 | 実行時の失敗 |
| 2 | 入力・設定が不正 |
| 3 | capabilityが非対応 |
| 4 | 必須依存が未導入 |
| 5 | 指定hardware/runtimeが利用不可 |
| 6 | 明示的に要求した品質・制約ゲートが不合格、または有効な候補なし |
| 7 | 資源予算の上限で停止 |
| 130 | ユーザーによる中断 |

通常のcompareは有効な比較レポートを作れれば0、`--require-acceptance`を指定した場合は不合格・証拠不足で6とする案。どちらの場合もJSONの合格状態を省略しない。

`--json`ではstdoutにversion付きJSONを1件だけ出す。人間向けログと警告はstderrへ、進捗は`--events-jsonl FILE`などの明示した別streamへ出す。`--quiet`でも結果と失敗理由は消さない。eventにはrun ID、順序番号、stage、完了量と総量を含め、総量不明なら進捗率を捏造しない。

設定はschema versionを持ち、`backend / scheme / weight_dtype / activation_dtype / group_size / target / backend_options / passes`へ正規化する。旧`method: naive`はTernary/AbsMeanへ移行し、旧来のper-tensor既定値を勝手にper-groupへ変えない。既知の旧キーは明示的に移行し、未知キーは拒否する。backend固有設定にもversion付き検証を適用する。

## 5. 小さいPRで順に検証する

| PR | Phase | 実装単位と証明 |
|---|---|---|
| PR-CLI-01 | 0 | モデル不在時の実モデル校正を拒否し、合成fixtureをテスト入口へ分離する |
| PR-CLI-02 | 0 | 設定の誤記・矛盾・未対応値を拒否し、dtype/deviceの実処理を検証する |
| PR-CLI-03 | 0 | run予約と失敗記録を直し、2プロセス競合・失敗注入で証明する |
| PR-CLI-04a | 0 | compareの入力・旧schema・失敗記録を回帰試験で固定する |
| PR-CLI-04b | 0 | group scaleを含む容量計算と表示を回帰試験で固定する |
| PR-CLI-05 | 0 | CLI分割、旧スタブ整理、実例更新、core/ML CIを整える |
| PR-CLI-06 | 1 | Backend RegistryとTernary adapterを導入し、重みの内容一致を確認する |
| PR-CLI-07 | 1 | capability、plan/doctor、JSON、設定移行、manifest仕様を確定する。必要なら仕様と実装を分割 |
| PR-CLI-08 | 5a | 既存fake-quant exporterを共通保存層へ接続し、reloadを検証する |
| PR-CLI-09 | 2 | 汎用LLM adapterを追加し、Gemmaとの共通経路を確認する |
| PR-CLI-10 | 3 | TorchAOの1scheme・1環境を変換からreloadまで接続する |
| PR-CLI-11 | 6 | LLMの共通Evaluator、複数run比較、実測資源指標を整える |
| PR-CLI-12 | 2/3/6 | Diffusers adapter、最初のDiffusion backend、評価profileを別PRで順次追加する |
| PR-CLI-13 | 4 | pass graphと最初の実passを別PRで検証する |
| PR-CLI-14 | 5b/5c | packed ternary、対応runtime向けbridgeをそれぞれ独立して検証する |
| PR-CLI-15 | 7 | 限定候補探索、予算/cache/resume、mixed precisionを別PRで順次追加する |
| PR-CLI-16 | 8 | plugin互換性、配布試験、文書・license・release matrixを仕上げる |

PR-CLI-12以降は作業群であり、一括PRの指示ではない。各PRは「機能Xを実装し、検証Yで示す」単位へ分割する。日程・費用はbackendの互換性調査後に見積もり、未調査の対応を期限付きで約束しない。

次の実装対象はPR-CLI-01。model不在時の終了コード、`metrics.json`、実モデル完了を表示しないこと、合成fixtureと正常な校正入口が維持されることを確認する。量子化数学、研究用データ、品質閾値、依存環境はこのPRに含めない。

## 6. 完成は証拠付きの対応範囲で判定する

- [x] 誤成功・設定無視・run混在・不正な品質合格を防ぐ。
- [x] Model Adapter、Backend、Pass、Exporter、Evaluatorの責任が分かれている。
- [x] Ternaryと非Ternary、LLMとDiffusionについて、固定した対応組合せの実行証拠がある。
- [x] fake quant、packed保存、native低bit推論の対応範囲を区別している。
- [x] 保存・別プロセスでのreload・hash検証・runtime要件が揃っている。
- [x] JSON/JSONL、終了コード、config/report/manifestのversionと移行方針がある。
- [x] 同条件の複数run比較と、合格を要求する自動処理ができる。
- [x] optimizeの予算停止、再開、cache、有効候補なし、mixed precisionを検証した。実Gemmaでは容量停止とworker回収、成功候補は同じ通常経路の実測で確認した。
- [ ] Windows/Linuxの導入CI、実機matrix、plugin互換性は揃った。project licenseは権利者判断が必要なためTBDで、安定版公開Gateだけ未完了。
- [x] CLI coreにGUI固有コードがなく、Desktopが公開仕様から利用できる。

gateを閉じる際は、対象commitと変更状態、コマンド・条件、テスト結果、成果物の場所・hash、実行環境、制約、残る未検証事項を記録する。CLIリリースの合格記録からモデル品質の合格を推測させない。

固定した検証範囲は、Gemma 4 E2B revision `6befbaca7398925921802abd1f277b495b78b738`、TorchAO 0.18.0 INT8 weight-only、Diffusers 0.40.0のpretrained tiny Stable Diffusion UNet、llama.cpp `e6ab7c1a41054a888ada952eab4c886444c2f5ad`の人工Llama FP32 GGUFである。任意architecture、native ternary/INT8 kernel、画像の人間品質、署名済み安定版へ結果を広げない。

CLI製品ロードマップの実装作業は完了扱いとする。残る2点は通常のCLI修正では閉じられない境界である。native packed ternary runtimeは新しいkernel/runtime研究が必要で、今回の製品範囲外。project licenseは著作権者の決定が必要で、あーしが独断で設定できない。これらを解消するまでstable releaseは行わない。
