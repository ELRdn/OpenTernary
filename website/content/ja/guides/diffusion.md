---
description: "Diffusersのpipelineの1つのcomponentを量子化し、固定のprofileで画像生成を評価する。"
---

# Diffusionモデル

OpenTernaryは、[Diffusers](https://huggingface.co/docs/diffusers)のpipelineのcomponent（transformerやUNetなど）を
量子化し、比較のために固定の手順で画像を生成することもできます。先にextraを入れます:

```bash
uv tool install "openternary[diffusion]" --python 3.12 --reinstall   # diffusers==0.40.0
openternary setup                                                    # 入れ直しでGPU用のビルドが消えたため
```

## componentを量子化する {#quantize-a-component}

pipelineのフォルダには、`model_index.json` とcomponentごとのサブフォルダがあります。componentは明示して選びます:

```bash
openternary plan ./my-pipeline --component transformer
openternary quantize ./my-pipeline --component transformer --device cpu --output runs/pipe-transformer
```

変換するのは、判別できる**2次元の線形層の重み**だけです。畳み込みの重みとVAEはそのまま残ります。pipeline全体を
低bitにするものではありません。出力のshardにはDiffusersの標準の名前を使うので、Diffusersでcomponentを読み込み
直せます。

## 固定のprofileで評価する {#evaluate-with-a-fixed-profile}

画像の品質を判断するのは人なので、`evaluate` は生成を再現可能にして証拠を記録しますが、画像を「よい」と判定する
ことはありません。手順はprofileに書きます:

```json title="diffusion-profile.json"
{
  "schema_version": 1,
  "family": "diffusion",
  "name": "fixed-prompt-v1",
  "prompts": ["A red cube on a white table"],
  "seeds": [42],
  "steps": 20,
  "width": 512,
  "height": 512,
  "guidance_scale": 3.5,
  "warmup": true
}
```

```bash
openternary evaluate ./my-pipeline --profile diffusion-profile.json --dry-run      # profileを確認
openternary evaluate ./my-pipeline --profile diffusion-profile.json --output runs/pipe-original
```

量子化したcomponentに差し替えて評価するには、profileに `"component": "transformer"` を加え、成果物を渡します:

```bash
openternary evaluate ./my-pipeline --profile diffusion-profile.json \
    --component-artifact runs/pipe-transformer/artifacts/snapshot --output runs/pipe-ternary
```

差し替えるものは、今のところsafetensorsでなければなりません。

## 記録されるもの {#what-is-recorded}

`evaluation.json` には、プロンプト・シード・step数・解像度・pipelineのschedulerの設定と、画像ごとのハッシュ・
生成時間・画素の統計が保存されます。比較のときには、これらの設定が一致するかを確認します。

結果には必ず `human_review_required` が付きます。画素の統計やハッシュで分かるのは画像が*変わった*ことで、悪く
なったかどうかではありません。そのため、OpenTernaryがdiffusionの結果を品質合格とすることはありません。

## 検証済みの範囲 {#what-has-been-verified}

32×32ピクセル・2 stepの人工のStable Diffusionのpipelineで、UNetを差し替えた変換と有限値の画像の生成を、
Windows・WSLのCPUと、Radeon RX 9070 XTのFP32・BF16で確認しています。学習済みの実際のpipelineと、画像の品質は
評価していません。[現在の状況](../project/status.md)を参照してください。
