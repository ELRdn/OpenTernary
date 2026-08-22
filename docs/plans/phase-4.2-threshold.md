# Phase 4.2 — Learnable Thresholds 導入計画

> **Status:** Planning (Draft 2026-08-22) — 実装前レビュー用
> **Owner:** DSH (Lead) / Codex (実装) / Scarlet (技術実装)
> **Predecessor:** Phase 4.1 ✅ CLOSED (recon-scale, 141 tests, tiny fixture 1.583→1.576)

---

## 0. 要旨

Phase 4.1 で `scale` のみを `softplus(raw)+eps` + `inverse_softplus` 初期化で学習し、layer-local `codes*scale→F.linear` 再構成が `tiny fixture` と `heldout` 改善、`fingerprint` 差、`checkpoint/resume/materialize` を bounded-memory で達成した。

**Phase 4.2** では閾値 `threshold` を学習対象に加える。Phase 2/4.1 の暗黙閾値 `Q=clamp(round(W/scale))` ＝ `±0.5*scale` を、`t ∈ (0,1)` の連続パラメータで一般化し、`per_tensor / per_group` で `codes` 自体が最適化中に変化するパスを確立する。STE (Straight-Through Estimator) と `sigmoid(raw_threshold)` で微分可能にし、`scale` 学習との同時最適化で Phase 4.1 からの追加改善を証明する。

本計画は実装を含まない。レビュー承認後に Phase 4.2 実装フェーズへ移行する。

---

## 1. 背景とスコープ

### 1.1 現状 (ROADMAP/ARCHITECTURE 準拠)

- `src/openternary/quant/ternary.py`: `quantize_absmean` は `scale=mean(abs(W))` のみ、`0.5*scale` はハードコード（`round` 由来）。
- `src/openternary/quant/grouping.py`: LD-RW で同じ `0.5*scale_g` 暗黙。
- `src/openternary/calibration/optimizer.py`: `build_scale_params / get_effective_scales` のみ。`codes` は固定。
- `src/openternary/calibration/runner.py`: `codes` 固定 × `scale` 学習。`calibrated_state` は `codes+scales` を materialize。
- `config/schema.py`: `CalibrationConfig` は `method=recon-scale` / `window=per-layer` のみ。

### 1.2 ゴール

Research Question (RESEARCH_PLAN H3/H4): 閾値の per-group 最適化は `scale` 単独より held-out 再構成損失と下流 smoke 挙動で追加利得を与えるか？

### 1.3 やること / やらないこと

**やる:**
- 閾値の数学的定義とパラメタ化の確定
- threshold 学習の differentiable 実装（STE）
- scale + threshold 同時最適化の calibration パス
- bounded sharded materialize の閾値対応
- tiny fixture での scale-only 比追加改善の証明

**やらない (defer):**
- 4.3 Soft-to-hard temperature annealing / 4.4 per-block window / 4.5 CAT-Q 等
- PLE/embed/lm_head の三値化（Phase 6）
- packed 2-bit への閾値メタデータ埋め込み（Phase 7）
- GUI / MoE / 大規模スケール

---

## 2. 技術設計

### 2.1 閾値の定義

現行 `round(W/scale)` は閾値 `±0.5*scale` と等価。Phase 4.2 では:

```
threshold_ratio = sigmoid(raw_threshold) ∈ (0,1)   # 初期 0.5
effective_threshold = threshold_ratio * effective_scale

Q = -1  if W < -effective_threshold
     0  if |W| ≤ effective_threshold
    +1  if W >  effective_threshold
```

- 対称閾値1つで開始（非対称 t_low/t_high は 4.2.1 拡張候補）。
- per_tensor: 1 threshold / tensor、per_group: 1 threshold / group（scales と同数、interleaved 順序は grouping.py に準拠）。
- ゼログループ (scale==0) は閾値も学習対象外。

### 2.2 微分可能性 (STE)

閾値は Q へのハード分岐を伴うため STE で近似:

```python
codes_hard = where(W < -thr, -1, where(W > thr, +1, 0))
# backward: STE — upstream grad を素通し
```

Phase 4.2 では最小の STE (`forward=hard`, `backward=identity` via `torch.autograd.Function`) で開始。4.3 で temperature annealing を導入。

**codes 再計算の頻度:** 各 forward で W (固定) と effective_threshold から codes を再計算。W は保持せず最終 codes を保存する方針で bounded を維持。

### 2.3 パラメタ化

`optimizer.py` に `build_threshold_params / get_effective_thresholds` を追加:

- `raw_threshold` は `inverse_sigmoid(0.5) = 0` で初期化 → `sigmoid(0)=0.5 == orig`。
- `effective_threshold_ratio = sigmoid(raw_threshold)`。
- ゼロマスク位置は学習対象外。

### 2.4 Config 拡張

`CalibrationConfig` に追加 (後方互換、デフォルトで 4.1 挙動):

```python
threshold_enabled: bool = False
threshold_lr: float | None = None  # None なら calibration.lr を流用
threshold_init: float = 0.5        # (0,1)
method: Literal["recon-scale", "recon-scale-threshold"] = "recon-scale"
```

window は引き続き per-layer のみ。

### 2.5 Runner 変更点

- per_module に `raw_threshold` を追加。`all_raw_params` に threshold params を含む。
- loss 計算ループ内で `eff_scale` と `eff_thr_ratio` から codes を STE で再計算 → `w_hat = codes * eff_scale`。
- checkpoint に `raw_thresholds` を追加。`_run_materialize_only` は thresholds も復元。
- `calibration.json` に `threshold_fingerprint_before/after`, `mean_abs_threshold_delta` を追記。

### 2.6 Materialize 変更点

- `calibrated_state` に thresholds を含めるか、codes を threshold から再計算する分岐を追加。
- 巨大テンソル (4.6GB) の codes 再計算は W を再ロードせず calibrated_state[codes] を直接使用（最終 codes を保存）。
- `quantization.json` の per_tensor エントリに `threshold_stats / threshold_fingerprint` を追加。

### 2.7 Quant モジュール拡張

- `ternary.py` / `grouping.py` に `quantize_with_threshold` ヘルパを追加（threshold_ratio 引数、デフォルト 0.5 で既存と互換）。

---

## 3. タスク分解

| ID | Task | Files | Do not change | 検証 |
|---|---|---|---|---|
| T4.2-1 | 閾値パラメタ化ヘルパ | `src/openternary/calibration/optimizer.py` | scale 既存ロジック | pytest 拡張: sigmoid(0)=0.5, zero_mask 固定 |
| T4.2-2 | Threshold-aware quantize | `src/openternary/quant/ternary.py`, `grouping.py` | round ties-to-even | threshold=0.5 で既存と一致, 0.3/0.7 で zero_ratio 単調性 |
| T4.2-3 | STE 最小実装 | `src/openternary/calibration/ste.py` (新規) | torch なし環境 | hard forward, backward identity |
| T4.2-4 | Config 拡張 | `src/openternary/config/schema.py` | window 制約 | --dry-run で threshold_enabled 表示 |
| T4.2-5 | Runner 同時最適化 | `src/openternary/calibration/runner.py` | materialize_only 既存パス | tiny fixture で loss 改善 + checkpoint threshold |
| T4.2-6 | Materialize 閾値対応 | `src/openternary/quant/fake_quant.py` | convert_snapshot bounded 性 | threshold=0.3 で content_fp 差異, sharded roundtrip |
| T4.2-7 | CLI 表示 | `src/openternary/cli/main.py` | inspect/quantize/benchmark/compare | --dry-run に Threshold 行 |
| T4.2-8 | ドキュメント | `ROADMAP.md`, `ARCHITECTURE.md`, `EXPERIMENT_LOG.md` | Phase 4.1 記述の削除なし | Gate 章の更新 |

依存順: T4.2-1 → T4.2-2 → T4.2-3 → T4.2-5 → T4.2-6、T4.2-4 は並行可。

---

## 4. 受け入れゲート (G4.2-1〜G4.2-12)

Phase 4.1 の G4-1〜12 を継承し閾値版を追加:

```
G4.2-1  dry-run は filesystem 変更なし、threshold_enabled/threshold_init/cache estimate を表示
G4.2-2  tiny fixture (4 samples,5 steps, CPU <5分) で threshold_enabled=true が scale-only より best_loss ≤ 同等 かつ finite
G4.2-3  threshold_enabled=false 時に 4.1 と完全互換（loss_history / fingerprint が 4.1 と一致）
G4.2-4  checkpoint に raw_thresholds が保存され、--resume で復元され loss が継続
G4.2-5  held-out loss が scale-only 比で改善 or 同等、かつ train/held/smoke の hash isdisjoint 汚染なし
G4.2-6  ruff/mypy/pytest green（新規 tests 含め 150+）
G4.2-7  docs + fingerprints + dataset provenance 保存、threshold_fingerprint_before != after かつ mean_abs_threshold_delta >1e-9 (enabled時)
G4.2-8  trainable params == scales count + thresholds count（非ゼロのみ）、ゼログループは exact zero 維持
G4.2-9  step0 の effective_threshold_ratio == 0.5、threshold=0.5 時に naive と codes 一致
G4.2-10 ActivationCache / materialize は bounded streaming 維持、4.6GB embed で OOM なし
G4.2-11 materialize した calibrated_snapshot が threshold 反映で content_fingerprint が scale-only と差異、かつ AutoModel load + smoke PASS
G4.2-12 --materialize-only が threshold チェックポイントから正しく再現
```

---

## 5. 検証計画

### 5.1 自動テスト

```bash
uv sync --locked
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv run --frozen mypy src/openternary
uv run --frozen pytest -q -p no:cacheprovider
```

### 5.2 Tiny fixture 手動受け入れ

```bash
# scale-only ベースライン
echo "calibration:\n  enabled: true\n  method: recon-scale\n  dataset: synthetic\n  num_samples: 4\n  seq_len: 16\n  steps: 5\n  checkpoint_interval: 2" > /tmp/calib-scale.yaml
uv run --frozen openternary calibrate --config /tmp/calib-scale.yaml --output runs/calib-scale

# threshold 有効
echo "calibration:\n  enabled: true\n  method: recon-scale-threshold\n  threshold_enabled: true\n  dataset: synthetic\n  num_samples: 4\n  seq_len: 16\n  steps: 5\n  checkpoint_interval: 2" > /tmp/calib-thr.yaml
uv run --frozen openternary calibrate --config /tmp/calib-thr.yaml --output runs/calib-thr
```

---

## 6. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| STE 勾配不安定 | loss NaN | NaN loud、threshold を (0.01,0.99) に clamp、lr 分離 |
| threshold 飽和 | zero_ratio 崩壊 | 初期 0.5、lr 小、mean_abs_threshold_delta 監視 |
| codes 再計算でメモリ増 | OOM | W は保持せず最終 codes を保存 |
| 旧 checkpoint 互換性破損 | --resume 失敗 | version 追加、旧は threshold なしとしてロード |
| per_group 14M パラメタ | VRAM 増 | 56MB で許容、将来 per_row 共有は 4.2.1 で検討 |

---

## 7. 実装順序とブランチ戦略

1. `feat/calib-threshold` ブランチを main から作成。
2. T4.2-1〜4 を Codex へ委譲（各タスクは "Implement X and prove with Y test" 形式）。
3. DSH が統合 → ruff/mypy/pytest → tiny fixture → EXPERIMENT_LOG.md 更新 → PR。

---

## 8. 次の Phase との境界

- 4.3 Soft-to-hard で temperature 導入、4.4 per-block は window 拡張を伴うため Phase 4.2 では window=per-layer を厳守。
- Phase 5 (Japanese Retention) の前に scale+threshold の日本語 smoke.ja.* 観測を compare で記録。

---

## 9. 承認後の即時アクション

- [ ] ROADMAP.md の Phase 4.2 を IN_PROGRESS に更新
- [ ] Codex へ T4.2-1 を Handoff Template で委譲
