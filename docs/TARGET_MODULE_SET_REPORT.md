# Target Module Set Report — P0 Canonical Fix (205 vs 276)

## 结论
- **Phase 4.1 real run: 205 modules** — 検査 (`inspect`) の `quantizable 205` と一致
- **Phase 4.2 Gate A/C (修正前): 276 modules** — `per_layer` 71個が混入
- **原因:** `calibration/runner.py` が `model.named_modules()` の `Linear` を `lm_head/embed/vision/audio` だけで除外し、`per_layer_input_gate / per_layer_projection / per_layer_model_projection` を quantizable として含めていた
- **Canonical:** `205` (attention q/k/v/o + mlp gate/up/down のみ) を固定。`per_layer` は PLE/high-precision ポリシーで除外。`adapters/base.py` の `default_quantizable_role` と同一基準。

## 集合Diff (276 - 205 = 71)

```
model.language_model.layers.{0..34}.per_layer_input_gate   (35)
model.language_model.layers.{0..34}.per_layer_projection    (35)
model.language_model.per_layer_model_projection             (1)
合計 71
```

- これらは `vision_tower` ではないが、`per_layer` を含むため本来 `quantizable=false / role=other / exclude_reason=not in quantizable policy`。
- `adapters/base.py` の `_QUANTIZABLE_ATTENTION_RE` / `_QUANTIZABLE_MLP_RE` のみに一致するものが quantizable。

## 検証

```python
current runner (修正前): 276 = all Linear - lm_head/embed/vision/audio
canonical (修正後):      205 = 276 - 71 per_layer - norm - 非 attention/mlp
```

- `inspect` 205 と一致、Gate で `all Linear 526 → current 276 → canonical 205 → diff 71 per_layer` を確認
- 修正後の runner で `target_module_count 205` を再確認（`AutoModelForImageTextToText` 実機ロードで数え直し PASS）

## 修正内容

`src/openternary/calibration/runner.py`

- `per_layer` を明示除外 (`if "per_layer" in name: continue`)
- `norm` を除外
- `attention/mlp` のみを `re` でフィルタ（`adapters/base.py` と同一）
- fallback 分岐も同様に修正
- コメントに「canonical 205」を明記（研究比較で scale-only vs threshold 以外を変えないため）

`src/openternary/quant/fake_quant.py` の MemoryError 対応（streaming hash）は本P0とは別件だが同タイミングで修正済み。

## Cache Provenance への影響

- `205 vs 276` は `fingerprint.json` の `target_module_names / count / fingerprint` が一致しないため、`Activation Cache: INVALID (fingerprint mismatch)` が**正しい**挙動（以前の Gate C で 205→276 を比較すれば INVALID、276→276 なら REUSED が正しかった）
- 修正後は `205 vs 205` で `REUSED from --init-from` が期待通り。Gate C の `5 vs 5` 縮小版では実際に `REUSED` を確認済み（`fingerprint 7c6dfe...` 一致、vision除外 True）
- 今後の 50-step 本番は `205` 固定で `Phase 4.1 scale-only` と `Phase 4.2 threshold` を純粋比較可能

## 次のアクション

- 50-step 本番は本fix適用後の `205` で開始（Gate完了まで開始しない指示を遵守）
- 既存の `276` で作られた Gate A/C の snapshot は破棄し、`205` で再作成すること（互換性なし）
