"""Create a tiny, randomly initialised Llama model for trying OpenTernary offline.

The weights are random: the model produces gibberish. It exists only so that every
command in the tutorial can run on a laptop CPU in seconds, without downloads.
"""

import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

out = Path(sys.argv[1] if len(sys.argv) > 1 else "tiny-llama")
out.mkdir(parents=True, exist_ok=False)
torch.manual_seed(0)

# A character-level tokenizer: every printable ASCII character is one token.
specials = ["<unk>", "<s>", "</s>", "<pad>"]
vocab = {token: i for i, token in enumerate(specials + [chr(c) for c in range(32, 127)])}
backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>"
)
tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
tokenizer.save_pretrained(out)

config = LlamaConfig(
    vocab_size=len(vocab),
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=3,
    tie_word_embeddings=False,
)
LlamaForCausalLM(config).save_pretrained(out)
# Marks the model as synthetic: OpenTernary will never report quality acceptance for it.
(out / "openternary-fixture.json").write_text(json.dumps({"pretrained": False, "family": "llama", "seed": 0}))
print(f"wrote {out}")
