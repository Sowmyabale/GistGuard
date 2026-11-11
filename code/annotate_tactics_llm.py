"""
annotate_tactics_llm.py — the "REAL FIX" for tactic supervision (advisor discussion, 18 Sep).

COVA-X ships no per-turn tactic annotations; the v1.5/v3 tactic labels were
derived from a fixed scam-type -> dominant-tactic mapping (label leakage).
This script uses Llama-3.1-8B to annotate EVERY scammer turn of EVERY
conversation with one of the 6-tactic taxonomy (or "none"), producing:

    results_v1rerun/tactic_annotations_<split>.jsonl

Rows:
    {conversation_id, scam_type,
     turn_tactics: [{turn, tactic}],
     dominant_tactic,           # mode of non-"none" annotations
     mapping_tactic,            # the old SCAM_TYPE_TO_TACTIC value (for agreement stats)
     agrees_with_mapping}       # bool

The summary prints the dataset-wide tactic distribution and the
agreement rate with the old mapping — both go into the paper.

"""

import os
import re
import json
import glob
import time
import argparse
from collections import Counter

from config import SCAM_TYPES, TACTICS, SCAM_TYPE_TO_TACTIC
from dataset_utils import conversation_to_text

TACTIC_DEFS = {
    "authority_impersonation": "caller pretends to be an authority: bank, IRS, government, police, Medicare, company official",
    "fear_threat": "caller uses fear, threats, or fake emergencies: arrest, lawsuit, kidnapping, account closed",
    "isolation_secrecy": "caller tells the victim to keep it secret, not tell family, or stay on the line",
    "over_familiarity": "caller pretends to be family, a friend, or a romantic partner to build false trust",
    "reciprocity_gift": "caller promises a prize, lottery win, refund, or gift that requires a fee or action first",
    "social_proof": "caller claims other people joined, made money, or approved it: fake reviews, testimonials, groups",
}

def scammer_turns(conv: dict):
    """[(turn_number, content)] for caller/scammer turns."""
    out = []
    for t in conv.get("turns", []):
        role = t.get("role", "").lower()
        if role.startswith(("caller", "scammer", "attacker")):
            out.append((t.get("turn_number", len(out) + 1), t.get("content", "")))
    return out

def build_prompt(turns):
    lines = "\n".join(f'Turn {n}: "{c}"' for n, c in turns)
    defs = "\n".join(f'- {name}: {TACTIC_DEFS[name]}' for name in TACTICS)
    return (
        "You are an expert annotator of manipulation tactics in scam calls.\n"
        "Label EACH numbered caller turn below with exactly one tactic from this list "
        '(or "none" if the turn contains no manipulation):\n'
        f"{defs}\n\n"
        f"Caller turns:\n{lines}\n\n"
        'Reply with ONLY a JSON object in this exact shape:\n'
        '{"annotations": [{"turn": <turn number>, "tactic": "<tactic name or none>"}, ...]}\n'
        "One entry per caller turn, in order. No explanation."
    )

SYSTEM = ("You are an expert annotator of manipulation tactics in scam calls. "
          "You reply with strict JSON only.")

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_annotation(text: str, expected_turns):
    """-> {turn_number: tactic} (missing turns filled with 'none')."""
    m = JSON_RE.search(text or "")
    result = {n: "none" for n, _ in expected_turns}
    if not m:
        return result
    try:
        data = json.loads(m.group(0))
        for item in data.get("annotations", []):
            t, tac = item.get("turn"), str(item.get("tactic", "none")).lower().strip()
            tac = tac.replace(" ", "_").replace("-", "_")
            if tac not in TACTICS:
                tac = "none"
            if t in result:
                result[t] = tac
    except Exception:
        pass
    return result

def dominant(tactics):
    c = Counter(t for t in tactics if t != "none")
    return c.most_common(1)[0][0] if c else "none"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=2048,
                    help="prompt token cap (lower = less KV-cache memory)")
    ap.add_argument("--max_new_tokens", type=int, default=320,
                    help="annotation JSON is short; 320 is plenty")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    # ---- load conversations ----
    items = []  # (conversation_id, scam_type, turns)
    for split in args.splits:
        split_dir = os.path.join(args.dataset_dir, split)
        for fpath in sorted(glob.glob(os.path.join(split_dir, "*", "*.json"))):
            with open(fpath, encoding="utf-8") as f:
                conv = json.load(f)
            turns = scammer_turns(conv)
            if not turns:
                continue
            items.append((conv.get("conversation_id", os.path.basename(fpath)),
                          conv.get("scam_type", "unknown"), turns))
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} conversations with scammer turns")

    # ---- model ----
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map="cuda")
    model.eval()
    print(f"Model loaded: {args.model}")

    # ---- resume ----
    done = set()
    fout = open(args.output, "a", encoding="utf-8")
    if args.resume and os.path.exists(args.output):
        for l in open(args.output, encoding="utf-8"):
            try:
                done.add(json.loads(l)["conversation_id"])
            except Exception:
                pass
    todo = [it for it in items if it[0] not in done]
    print(f"Resume: {len(done)} done | To annotate: {len(todo)}")

    B = args.batch_size
    t0 = time.time()
    dist = Counter()
    agree = agree_n = 0

    for i in range(0, len(todo), B):
        batch = todo[i: i + B]
        texts = []
        for cid, st, turns in batch:
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_prompt(turns)}]
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_length).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

        for (cid, st, turns), g in zip(batch, gen):
            ann = parse_annotation(g, turns)
            turn_tactics = [{"turn": n, "tactic": ann[n]} for n, _ in turns]
            dom = dominant(ann[n] for n, _ in turns)
            mapping = SCAM_TYPE_TO_TACTIC.get(st, "none")
            row = {"conversation_id": cid, "scam_type": st,
                   "turn_tactics": turn_tactics,
                   "dominant_tactic": dom,
                   "mapping_tactic": mapping,
                   "agrees_with_mapping": dom == mapping}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            dist[dom] += 1
            agree_n += 1
            agree += int(row["agrees_with_mapping"])
        fout.flush()
        if (i // B) % 20 == 0:
            print(f"  [{i + len(batch)}/{len(todo)} done, {time.time() - t0:.0f}s]")

    fout.close()
    n = len(done) + len(todo)
    print("\n=== TACTIC ANNOTATION SUMMARY ===")
    print(f"Total annotated: {n}")
    print("Dominant-tactic distribution:", dict(dist.most_common()))
    print(f"Agreement with old scam-type mapping: {agree}/{agree_n} "
          f"({100 * agree / max(agree_n, 1):.1f}%)")
    print(f"Saved -> {args.output}")

if __name__ == "__main__":
    main()
