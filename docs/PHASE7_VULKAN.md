# Phase 7 — Vulkan Packed Ternary Inference (Future Candidate)

> **Status: non-scope for Phase 4.2-G**

## 位置づけ
- Phase 4.2-G の GPU 対象は **calibration reconstruction optimization (205-module, current-module resident)** のみ
- Vulkan backend は Phase 4.2 では non-scope

## 将来候補 (Phase 7)
- Packed ternary inference runtime として Vulkan を検討
- 候補: `2-bit pack (00=-1,01=0,10=+1,11=reserved)` を Vulkan compute shader で dequantize → matmul
- 対象は `calibrated_snapshot` の `codes + scales` を `packing.py` 経由で `packed` にした artifact

## 非スコープ理由
- Phase 4.2-G は `reconstruction` の高速化と `dynamic VRAM` の堅牢性が最優先
- Teacher full GPU load や Vulkan inference は別フェーズで検証

## 記録のみ
- 本ドキュメントは将来の Phase 7 検討用に Vulkan を候補として残す
- 実装時は `src/openternary/quant/packing.py` の `pack/unpack` 契約と `grouping` を再利用する
