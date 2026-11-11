"""
generate_gists_diverse.py — v5: diversity-aware inference for the LoRA gist generator.

Same adapter + same short deployment prompt as generate_gists_lora.py (no few-shot,
no rules — preserves the "prompt-free capability" ablation claim). The ONLY change
is decoding + selection:

  1. Greedy candidate first (same as v4b).
  2. QC check: is_valid(w, a) AND Flesch-Kincaid grade <= --max_fk.
  3. Diversity check: bigram Jaccard overlap against warnings ALREADY ACCEPTED in
     the same predicted scam class must be <= --max_sim.
  4. If either fails -> resample up to --resample rounds (temperature 0.9, top_p 0.95).
  5. Fallback: best-scoring sampled candidate (lowest sim, must still pass QC);
     if none passes QC, keep the greedy candidate.

Supports --resume (skips conversation_ids already in the output file).

"""

import os
import json
import time
import argparse
from collections import defaultdict

from generate_gists_v3 import parse_output, is_valid, profile_nums, load_conversations
from train_gist_lora import short_user_message
from extract_evidence import evidence_card_fields, format_card
from eval_gist_quality import fk_grade

def bigrams(text):
    toks = text.lower().split()
    if len(toks) > 1:
        return set(zip(toks, toks[1:]))
    return {("<single>", toks[0])} if toks else {("<empty>", "")}

def jaccard(a, b):
    inter = len(a & b)
    return inter / max(1, len(a | b))

def max_overlap(warn_bigrams, accepted):
    if not accepted:
        return 0.0
    return max(jaccard(warn_bigrams, ab) for ab in accepted)

def qc_ok(w, a, max_fk):
    return bool(w) and bool(a) and is_valid(w, a) and fk_grade(w) <= max_fk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_fk", type=float, default=9.0)
    ap.add_argument("--max_sim", type=float, default=0.5)
    ap.add_argument("--resample", type=int, default=3)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    records = [json.loads(l) for l in open(args.predictions, encoding="utf-8")]
    if args.limit:
        records = records[: args.limit]
    convs = load_conversations(args.dataset_dir, args.split)
    print(f"{len(records)} prediction records | {len(convs)} conversations")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                                device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    print("LoRA adapter loaded (diverse mode)")

    SYSTEM = ("You are a protective assistant for elderly people. "
              "You write very short, plain-language, actionable scam warnings.")

    done = {}
    if args.resume and os.path.exists(args.output):
        for r in json.load(open(args.output, encoding="utf-8")):
            done[r["conversation_id"]] = r
    results = list(done.values())
    todo = [r for r in records if r["conversation_id"] not in done]
    print(f"To generate: {len(todo)}")

    # rebuild accepted-warning bigrams per predicted class from resumed results
    accepted = defaultdict(list)
    for r in results:
        accepted[r.get("pred_scam", "?")].append(bigrams(r.get("warning", "")))

    def gen(batch, sample):
        texts = []
        for rec in batch:
            cid = rec["conversation_id"]
            conv_text, vc = convs.get(cid, ("", {}))
            trust, tech, aw = profile_nums(vc)
            scam = rec.get("pred_scam") or rec.get("scam_type", "bank")
            fields = evidence_card_fields(conv_text, scam)
            pseudo = {"conv_text": conv_text,
                      "scam_type": scam,
                      "tactic": rec.get("pred_tactic", "authority_impersonation"),
                      "trust": trust, "tech": tech, "awareness": aw,
                      "evidence_card": format_card(fields)}
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": short_user_message(pseudo)}]
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=3072).to("cuda")
        kwargs = dict(max_new_tokens=110, pad_token_id=tokenizer.pad_token_id)
        if sample:
            kwargs.update(do_sample=True, temperature=0.9, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, **kwargs)
        gen_text = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        return [parse_output(g) for g in gen_text]

    B = args.batch_size
    t0 = time.time()
    n_resampled = 0
    for i in range(0, len(todo), B):
        batch = todo[i: i + B]
        # slot state: meta -> list of (w, a, sim, pass_qc) candidates
        slots = []
        for rec, (w, a) in zip(batch, gen(batch, sample=False)):
            sim = max_overlap(bigrams(w), accepted[rec.get("pred_scam") or rec.get("scam_type", "bank")])
            slots.append([rec, [(w, a, sim, qc_ok(w, a, args.max_fk))]])

        for rnd in range(args.resample):
            retry = [rec for rec, cands in slots if not any(pq and sm <= args.max_sim for _, _, sm, pq in cands)]
            if not retry:
                break
            for rec, (w, a) in zip(retry, gen(retry, sample=True)):
                sim = max_overlap(bigrams(w), accepted[rec.get("pred_scam") or rec.get("scam_type", "bank")])
                for slot_rec, cands in slots:
                    if slot_rec is rec:
                        cands.append((w, a, sim, qc_ok(w, a, args.max_fk)))
                        break
            n_resampled += len(retry)

        # pick per slot: first candidate passing QC + sim; else lowest-sim QC-passing; else greedy
        for rec, cands in slots:
            scam = rec.get("pred_scam") or rec.get("scam_type", "bank")
            chosen = None
            for w, a, sim, pq in cands:
                if pq and sim <= args.max_sim:
                    chosen = (w, a, sim)
                    break
            if chosen is None:
                ok_cands = [(w, a, sim) for w, a, sim, pq in cands if pq]
                if ok_cands:
                    chosen = min(ok_cands, key=lambda x: x[2])
                else:
                    w, a = cands[0][0], cands[0][1]
                    chosen = (w, a, cands[0][2])
            w, a, sim = chosen
            row = {"conversation_id": rec["conversation_id"],
                   "true_scam": rec.get("true_scam"), "pred_scam": scam,
                   "pred_tactic": rec.get("pred_tactic", "authority_impersonation"),
                   "warning": w, "action": a, "valid": is_valid(w, a),
                   "sim": round(sim, 3), "variant": "v5_diverse"}
            results.append(row)
            accepted[scam].append(bigrams(w))

        if (i // B) % 10 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [{len(results)}/{len(todo)} done, {time.time()-t0:.0f}s, "
                  f"{n_resampled} resamples]", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n = len(results)
    valid = sum(r["valid"] for r in results)
    avg_words = sum(len(r["warning"].split()) for r in results) / max(n, 1)
    uniq = len({r["warning"] for r in results}) / max(n, 1)
    print(f"\n=== V5 DIVERSE GIST SUMMARY ===")
    print(f"Total: {n} | valid: {valid} ({100*valid/max(n,1):.1f}%)")
    print(f"Avg warning length: {avg_words:.1f} words | unique: {100*uniq:.1f}%")
    print("\nSample outputs:")
    for r in results[:3]:
        print(json.dumps({k: r[k] for k in ("true_scam", "pred_scam", "warning",
                                            "action", "sim")},
                         indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
