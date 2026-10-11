"""Shared pieces for the literature-rule scripts: prompt format, rule parsing, and a tiny
word-level tokenizer so the pipeline can be tested without downloading a model."""

from __future__ import annotations

import json
import os
import re

import torch

RULE_FIELDS = ("cell_type", "signal", "direction", "behavior")
DEFAULT_KEYWORDS = (
    "il-4", "il-6", "il-10", "egf", "csf", "tgf", "pd-1", "pd-l1", "ifn", "tnf", "vegf", "cxcl12", "ccl17",
    "macrophage", "fibroblast", "t cell", "th2", "tumor cell", "tumour cell", "carcinoma", "hypoxia", "oxygen", "lactate",
)


def make_prompt(sentence: str) -> str:
    return f"Sentence: {sentence.strip()}\nRule:"


def parse_rule(text: str) -> dict:
    """'macrophage, IL-4, increases, EGF secretion' -> dict; unparsable text is kept raw."""
    line = text.strip().split("\n")[0].strip()
    parts = [p.strip() for p in line.split(",")]
    out = {"rule_raw": line, "parsed": len(parts) == 4}
    if out["parsed"]:
        out.update(dict(zip(RULE_FIELDS, parts)))
    return out


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class WordTokenizer:
    """Whitespace/punctuation tokenizer with the few methods the scripts need.
    Only used with `--model-name tiny`; real runs use `transformers.AutoTokenizer`."""

    _split = re.compile(r"\w+|[^\w\s]")

    def __init__(self, vocab: list[str]):
        self.itos = vocab
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id = 0, 1, 2, 3
        self.padding_side = "right"

    @classmethod
    def build(cls, texts: list[str]) -> "WordTokenizer":
        words = sorted({w.lower() for t in texts for w in cls._split.findall(t)})
        return cls(["<pad>", "<bos>", "<eos>", "<unk>"] + words)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path) as f:
            return cls(json.load(f))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.stoi.get(w.lower(), self.unk_token_id) for w in self._split.findall(text)]
        return [self.bos_token_id] + ids if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        toks = [self.itos[int(i)] for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if not (t.startswith("<") and t.endswith(">"))]
        return " ".join(toks)

    def __len__(self) -> int:
        return len(self.itos)

    def __call__(self, texts: list[str], padding: bool = True, return_tensors: str = "pt") -> dict:
        """Mimics the HF tokenizer call used for batched generation (honours padding_side)."""
        enc = [self.encode(t, add_special_tokens=True) for t in texts]
        width = max(len(e) for e in enc)
        ids, mask = [], []
        for e in enc:
            pad = [self.pad_token_id] * (width - len(e))
            if self.padding_side == "left":
                ids.append(pad + e)
                mask.append([0] * len(pad) + [1] * len(e))
            else:
                ids.append(e + pad)
                mask.append([1] * len(e) + [0] * len(pad))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def load_tokenizer(model_name: str, model_dir: str | None = None):
    if model_name == "tiny":
        return WordTokenizer.load(os.path.join(model_dir, "vocab.json"))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir or model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
