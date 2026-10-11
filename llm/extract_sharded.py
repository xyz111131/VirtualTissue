"""Rule extraction over an article corpus, sharded across GPUs (Sec. 4.3b, row "Rules from
the literature", extraction step over the 10^4–10^5-article PMC corpus).

Inference needs no gradient synchronisation, so this is embarrassingly parallel: every
process loads the fine-tuned model on its own GPU and takes every W-th article
(`huangtests.shard_indices`). Either launcher works:

    sbatch --array=0-15 slurm/llm_extract_array.sbatch                       # one GPU per array task
    torchrun --standalone --nproc_per_node=8 llm/extract_sharded.py ...      # one GPU per rank, one node

Each shard writes <out>/rules_shard<k>.jsonl (one candidate rule per matched sentence) and
<out>/manifest_shard<k>.json (which articles it covered). `--engine vllm` swaps the
Hugging Face generate loop for a vLLM engine on the same GPU (and `--tp N` lets one shard
span N GPUs with tensor parallelism if a model does not fit on one).

Offline test:  python llm/extract_sharded.py --model-name tiny --model-dir runs/llm_lora/model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from common import DEFAULT_KEYWORDS, load_tokenizer, make_prompt, parse_rule, read_jsonl, split_sentences  # noqa: E402
from huangtests import shard_indices  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--articles", default=os.path.join(_HERE, "data", "articles_toy.jsonl"))
    p.add_argument("--model-name", default="tiny", help="'tiny' or the HF id of the base model")
    p.add_argument("--model-dir", default="runs/llm_lora/model", help="merged tiny model dir (with vocab.json)")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir from lora_finetune.py (non-tiny models)")
    p.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    p.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--keywords", nargs="*", default=list(DEFAULT_KEYWORDS))
    p.add_argument("--out", default="runs/llm_extract")
    return p.parse_args()


def load_hf_model(args: argparse.Namespace, device: torch.device):
    from transformers import AutoModelForCausalLM

    if args.model_name == "tiny":
        tok = load_tokenizer("tiny", args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    else:
        tok = load_tokenizer(args.model_name)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    tok.padding_side = "left"          # batched generation continues from the last real token
    return model.to(device).eval(), tok


@torch.no_grad()
def generate_hf(model, tok, prompts: list[str], args: argparse.Namespace, device: torch.device) -> list[str]:
    outs = []
    for i in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[i:i + args.batch_size], padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.decode(row, skip_special_tokens=True) for row in new)
    return outs


def generate_vllm(prompts: list[str], args: argparse.Namespace) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model_name, tensor_parallel_size=args.tp,
              enable_lora=bool(args.adapter), dtype="bfloat16")
    params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0, stop=["\n"])
    lora_req = None
    if args.adapter:
        from vllm.lora.request import LoRARequest

        lora_req = LoRARequest("rules", 1, args.adapter)
    return [o.outputs[0].text for o in llm.generate(prompts, params, lora_request=lora_req)]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    articles = read_jsonl(args.articles)
    mine, shard, n_shards = shard_indices(len(articles))
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"[extract] shard {shard}/{n_shards} on {dev_name}: {len(mine)} of {len(articles)} articles", flush=True)

    # 1. candidate sentences: those mentioning a signal or cell type of interest
    jobs = []
    for i in mine:
        for s in split_sentences(articles[i]["text"]):
            if any(k in s.lower() for k in args.keywords):
                jobs.append((articles[i]["id"], s))
    prompts = [make_prompt(s) for _, s in jobs]

    # 2. generation on this shard's GPU
    t0 = time.time()
    if args.engine == "vllm":
        texts = generate_vllm(prompts, args)
    else:
        model, tok = load_hf_model(args, device)
        texts = generate_hf(model, tok, prompts, args, device)
    gen_time = time.time() - t0

    # 3. write candidate rules and a manifest of what this shard covered
    os.makedirs(args.out, exist_ok=True)
    n_parsed = 0
    with open(os.path.join(args.out, f"rules_shard{shard}.jsonl"), "w") as f:
        for (aid, sentence), text in zip(jobs, texts):
            rec = {"article_id": aid, "sentence": sentence, **parse_rule(text)}
            n_parsed += rec["parsed"]
            f.write(json.dumps(rec) + "\n")
    manifest = {"shard": shard, "n_shards": n_shards, "device": dev_name, "engine": args.engine,
                "article_ids": [articles[i]["id"] for i in mine], "n_sentences": len(jobs),
                "n_parsed": n_parsed, "generation_time_s": round(gen_time, 2),
                "sentences_per_s": round(len(jobs) / max(gen_time, 1e-9), 2)}
    with open(os.path.join(args.out, f"manifest_shard{shard}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[extract] shard {shard}: {len(jobs)} sentences, {n_parsed} parsed as 4-field rules, "
          f"{gen_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
