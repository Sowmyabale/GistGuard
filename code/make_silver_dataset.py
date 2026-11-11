"""
make_silver_dataset.py — Step 1 of Option B (LoRA SFT for Layer 3).

Generates validity-filtered gists for the TRAIN split (gold scam type + gold
dominant tactic + victim profile from victim_config) using the grounded v3
generator. Only gists passing the automated QC become SILVER training data.

Output:
    <output>.silver.jsonl   -> rows: {conversation_id, scam_type, tactic,
                                     trust, tech, awareness, conv_text,
                                     warning, action}   (SFT input)
    <output>.rejected.jsonl -> gists that failed QC (for inspection)

"""

import os
import json
import time
import argparse

from config import SEED
from dataset_utils import load_split
from generate_gists_v3 import (
    build_chat, parse_output, is_valid, profile_nums, load_conversations,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    train_df = load_split(args.dataset_dir, args.split)
    if args.limit:
        train_df = train_df.head(args.limit)
    print(f"{len(train_df)} conversations in split '{args.split}'")

    convs = load_conversations(args.dataset_dir, args.split)
    print(f"{len(convs)} conversations loaded with victim_config")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map="cuda")
    model.eval()
    print(f"Model loaded: {args.model}")

    silver_path = args.output.replace(".jsonl", ".silver.jsonl")
    rej_path = args.output.replace(".jsonl", ".rejected.jsonl")
    done_ids = set()
    if args.resume and os.path.exists(silver_path):
        for l in open(silver_path, encoding="utf-8"):
            done_ids.add(json.loads(l)["conversation_id"])
    if args.resume and os.path.exists(rej_path):
        for l in open(rej_path, encoding="utf-8"):
            done_ids.add(json.loads(l)["conversation_id"])
    print(f"Resume: {len(done_ids)} already processed")

    todo = [r for _, r in train_df.iterrows() if r["conversation_id"] not in done_ids]
    print(f"To process: {len(todo)}")

    fs = open(silver_path, "a", encoding="utf-8")
    fr = open(rej_path, "a", encoding="utf-8")
    B = args.batch_size
    t0 = time.time()

    for i in range(0, len(todo), B):
        batch = todo[i: i + B]
        texts, metas = [], []
        for row in batch:
            cid = row["conversation_id"]
            conv_text, vc = convs.get(cid, ("", {}))
            trust, tech, aw = profile_nums(vc)
            msgs = build_chat(conv_text, row["scam_type"], row["tactic"],
                              trust, tech, aw, strict=False)
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
            metas.append({"conversation_id": cid, "scam_type": row["scam_type"],
                          "tactic": row["tactic"], "trust": trust, "tech": tech,
                          "awareness": aw, "conv_text": conv_text})

        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=3072).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=110, do_sample=True,
                                 temperature=0.3, top_p=0.9,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

        # QC + one strict retry per batch
        parsed = [parse_output(g) for g in gen]
        retry_idx = [j for j, (w, a) in enumerate(parsed) if not is_valid(w, a)]
        retries = {}
        if retry_idx:
            rtexts = []
            for j in retry_idx:
                m = metas[j]
                msgs = build_chat(m["conv_text"], m["scam_type"], m["tactic"],
                                  m["trust"], m["tech"], m["awareness"], strict=True)
                rtexts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
            enc = tokenizer(rtexts, return_tensors="pt", padding=True, truncation=True,
                            max_length=3072).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=110, do_sample=True,
                                     temperature=0.3, top_p=0.9,
                                     pad_token_id=tokenizer.pad_token_id)
            rgen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for j, g in zip(retry_idx, rgen):
                retries[j] = parse_output(g)

        for j, meta in enumerate(metas):
            w, a = retries.get(j, parsed[j])
            meta.update({"warning": w, "action": a,
                         "valid": is_valid(w, a), "retried": j in retries})
            (fs if meta["valid"] else fr).write(
                json.dumps(meta, ensure_ascii=False) + "\n")
        fs.flush(); fr.flush()

        if (i // B) % 20 == 0:
            n = i + len(batch)
            print(f"  [{n}/{len(todo)} done, {time.time()-t0:.0f}s]")

    fs.close(); fr.close()
    ns = sum(1 for _ in open(silver_path, encoding="utf-8"))
    nr = sum(1 for _ in open(rej_path, encoding="utf-8")) if os.path.exists(rej_path) else 0
    print(f"\n=== SILVER DATASET ===")
    print(f"valid (silver): {ns} | rejected: {nr} | "
          f"acceptance: {100*ns/max(ns+nr,1):.1f}%")
    print(f"saved -> {silver_path}")

if __name__ == "__main__":
    main()
