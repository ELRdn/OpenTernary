---
description: "OpenTernary's licence (Apache-2.0), the NOTICE requirement, the trademark policy for the name and logo, and the licences of models and data you use with it."
---

# Licence and trademarks

## Code: Apache License 2.0

The source code of OpenTernary is licensed under the [Apache License 2.0](https://github.com/ELRdn/OpenTernary/blob/main/LICENSE).
Copyright 2026 The OpenTernary Authors.

In short, you may use, modify and redistribute the code, including commercially, provided that:

- you keep the **[NOTICE](https://github.com/ELRdn/OpenTernary/blob/main/NOTICE)** file, which names this project
  as the origin, in redistributions and derivative works, forks included (Section 4(d));
- you mark the files you changed (Section 4(b));
- you include the licence.

Contributions are accepted under the same licence (Section 5). This summary is not legal advice. The licence text
is what applies.

## Name and logo: not licensed

The Apache licence does **not** grant rights to the name **OpenTernary**, the `OPEN-TERNARY` wordmark or the
three-tomoe symbol (Section 6). The details are in [TRADEMARKS.md](https://github.com/ELRdn/OpenTernary/blob/main/TRADEMARKS.md):

| Allowed without asking | Not allowed without written permission |
|---|---|
| redistributing unmodified releases, which contain the marks | naming a fork, product or service "OpenTernary" or something confusingly similar |
| referring to the project: "based on OpenTernary", "a fork of OpenTernary", "works with OpenTernary" | using the logo or wordmark as the identity of a fork, product or service |
| showing the unmodified logo next to such a reference, without suggesting endorsement | suggesting a modified version is official or endorsed |

A public fork should use its own name and logo. Change `src/openternary/ui/brand.py` and the assets, and keep
the NOTICE file.

## Models and data you use

OpenTernary does not include any model weights or datasets. Models you quantize and datasets you evaluate on keep
their own licences, and a quantized model is a derivative of the original. In particular:

- **Gemma** models are subject to Google's Gemma terms, which you accept on Hugging Face before downloading.
- The research quality dataset is built from **WikiText** (CC BY-SA 3.0 / GFDL), **SQuAD** (CC BY-SA 4.0) and
  **JGLUE/JSQuAD** (CC BY 4.0). Those terms apply to the built file.
- `wiki-tiny` calibration downloads WikiText from Hugging Face.

Check the licence of every model and dataset before you publish anything derived from it.
