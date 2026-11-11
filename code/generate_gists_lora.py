"""
generate_gists_lora.py — Step 3 of Option B: inference with the LoRA-trained gist
generator. Uses the SAME short deployment prompt as training (no few-shot, no
rule list, no retry loop) — this is the "prompt-free capability" evidence for
the paper: ablation of prompting vs fine-tuning.

"""

import os
import json
import time
import argparse

from generate_gists_v3 import parse_output, is_valid, profile_nums, load_conversations
from train_gist_lora import short_user_message  # same short prompt as training
from extract_evidence import evidence_card_fields, format_card

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--profile_source", default="supplied",
                    choices=["supplied", "predicted", "absent"],
                    help="supplied = dataset victim_config (current runs); "
                         "predicted = Layer-2 profile saved in --predictions; "
                         "absent = state that no profile is available")
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
    print("LoRA adapter loaded")

    SYSTEM = ("You are a protective assistant for elderly people. "
              "You write very short, plain-language, actionable scam warnings.")

    done = {}
    if args.resume and os.path.exists(args.output):
        for r in json.load(open(args.output, encoding="utf-8")):
            done[r["conversation_id"]] = r
    results = list(done.values())
    todo = [r for r in records if r["conversation_id"] not in done]
    print(f"To generate: {len(todo)}")

    B = args.batch_size
    t0 = time.time()
    for i in range(0, len(todo), B):
        batch = todo[i: i + B]
        texts, metas = [], []
        for rec in batch:
            cid = rec["conversation_id"]
            conv_text, vc = convs.get(cid, ("", {}))
            if args.profile_source == "supplied":
                trust, tech, aw = profile_nums(vc)
            elif args.profile_source == "predicted":
                prof = rec.get("profile") or {}
                trust = prof.get("trust_label", 1)
                tech = prof.get("tech_label", 1)
                aw = prof.get("awareness_label", 1)
            else:  # absent
                trust = tech = aw = None
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
            metas.append({"conversation_id": cid, "true_scam": rec.get("true_scam"),
                          "pred_scam": pseudo["scam_type"], "pred_tactic": pseudo["tactic"]})

        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=3072).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=110, do_sample=False,  # greedy
                                 pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        for meta, g in zip(metas, gen):
            w, a = parse_output(g)
            meta.update({"warning": w, "action": a, "valid": is_valid(w, a)})
            results.append(meta)
        if (i // B) % 10 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [{len(results)}/{len(todo)} done, {time.time()-t0:.0f}s]")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n = len(results)
    valid = sum(r["valid"] for r in results)
    avg_words = sum(len(r["warning"].split()) for r in results) / max(n, 1)
    print(f"\n=== LoRA GIST GENERATION SUMMARY ===")
    print(f"Total: {n} | valid: {valid} ({100*valid/max(n,1):.1f}%)")
    print(f"Avg warning length: {avg_words:.1f} words")
    print("\nSample outputs (trained model, short prompt only):")
    for r in results[:3]:
        print(json.dumps({k: r[k] for k in ("true_scam", "pred_scam", "warning", "action")},
                         indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
