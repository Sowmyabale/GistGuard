"""
generate_gists_v3.py — Layer 3 V3: GROUNDED gist generation with quality control.

Why v3 (vs the v1.5 gist run):
    The old prompt gave the LLM only (tactic, emotion) — no conversation, no scam
    type — so it produced generic filler ("Bottom-line meanings (bottom lines) help
    you understand..."). This script GROUNDS every gist in:
        (a) the actual conversation excerpt,
        (b) the predicted scam type (plain words),
        (c) the detected manipulation tactic,
        (d) the victim's ToM profile from victim_config.
    Plus: few-shot examples, anti-meta-word instruction, decoding controls, and an
    automatic quality check with one strict retry.

"""

import os
import re
import json
import glob
import time
import argparse

from config import TRUST_MAP, TECH_MAP, AWARENESS_MAP

def profile_nums(vc: dict):
    """victim_config raw strings -> (trust, tech, awareness) ints via config maps."""
    return (
        TRUST_MAP.get(vc.get("trust_level", "moderately_trusting"), 1),
        TECH_MAP.get(vc.get("tech_savviness", "moderate"), 1),
        AWARENESS_MAP.get(vc.get("scam_awareness", "vaguely_aware"), 1),
    )

SCAM_PLAIN = {
    "bank": "a fake bank",
    "government_impersonation": "a fake government agency",
    "grandparent": "someone pretending to be your grandchild",
    "medicare": "a fake Medicare or health-insurance agent",
    "romance": "a fake online romantic partner",
    "investment": "a fake investment advisor",
    "lottery": "a fake lottery or prize office",
    "virtual_kidnapping": "a fake kidnapper",
}

PROFILE_WORDS = {
    "trust": {0: "skeptical of strangers", 1: "somewhat trusting", 2: "very trusting"},
    "tech": {0: "not comfortable with technology", 1: "somewhat comfortable with technology",
             2: "comfortable with technology"},
    "awareness": {0: "not aware of common scams", 1: "vaguely aware of scams",
                  2: "aware of common scams"},
}

# words that indicate the model is talking ABOUT the task instead of doing it
META_WORDS = ("gist", "bottom-line", "bottom line", "meaning of", "scam tactic",
              "the format", "as requested", "here is")

def build_user_message(conv_text: str, scam_type: str, tactic: str,
                       trust: int, tech: int, awareness: int,
                       strict: bool = False, max_chars: int = 1200) -> str:
    excerpt = conv_text[:max_chars].strip()
    if len(conv_text) > max_chars:
        excerpt += " ..."
    scam_words = SCAM_PLAIN.get(scam_type, f"a {scam_type.replace('_', ' ')} scam")
    profile = (f"They are {PROFILE_WORDS['trust'].get(trust, 'somewhat trusting')}, "
               f"{PROFILE_WORDS['tech'].get(tech, 'somewhat comfortable with technology')}, and "
               f"{PROFILE_WORDS['awareness'].get(awareness, 'vaguely aware of scams')}.")

    rules = (
        "Rules:\n"
        "- Write ONLY the warning and the action. Never explain your task.\n"
        "- Never use the words: gist, bottom-line, scam tactic, format.\n"
        "- Two sentences maximum for the warning. Plain, calm words an 80-year-old understands.\n"
        "- The action must be one concrete thing they can do right now.\n"
    )
    if strict:
        rules += (
            "- Your previous reply broke the rules. This time output ONLY the two labeled lines, "
            "nothing else.\n"
        )

    return (
        "You are protecting an elderly person from a phone or text scam.\n\n"
        f"Scam type: this is {scam_words}.\n"
        f"Manipulation tactic used: {tactic.replace('_', ' ')}.\n"
        f"Victim profile: {profile}\n\n"
        "Conversation so far:\n"
        f"\"\"\"\n{excerpt}\n\"\"\"\n\n"
        f"{rules}\n"
        "Reply in EXACTLY this format:\n"
        "WARNING: <one or two short sentences about what the scammer is really doing>\n"
        "ACTION: <one short concrete instruction>"
    )

FEW_SHOT = [
    {
        "scam": "bank", "tactic": "authority_impersonation",
        "profile": "They are very trusting, not comfortable with technology, and not aware of common scams.",
        "conv": ("caller: This is the fraud department of your bank. Your account has been compromised. "
                 "I need you to verify your card number now.\n"
                 "victim: Oh no. Is my money safe?"),
        "warning": ("WARNING: This is not your bank. Real banks never ask for your full card number "
                    "on a call you did not start. They are trying to steal your account."),
        "action": "ACTION: Hang up and call the number on the back of your bank card.",
    },
    {
        "scam": "grandparent", "tactic": "over_familiarity",
        "profile": "They are very trusting, somewhat comfortable with technology, and vaguely aware of scams.",
        "conv": ("caller: Grandma, it's me! I'm in trouble and I need money for bail. Please don't tell mom.\n"
                 "victim: Oh my goodness, are you okay?"),
        "warning": ("WARNING: Someone is pretending to be your grandchild. Real family in trouble will let "
                    "you call their parents. Asking for secrecy is the trick."),
        "action": "ACTION: Hang up and call your grandchild's real phone number to check.",
    },
]

def build_chat(conv_text, scam_type, tactic, trust, tech, awareness, strict=False):
    system = ("You are a protective assistant for elderly people. You write very short, "
              "plain-language, actionable scam warnings.")
    messages = [{"role": "system", "content": system}]
    # few-shot examples in the SAME format as the target turn (grounded, labeled)
    for ex in FEW_SHOT:
        messages.append({"role": "user", "content": build_user_message(
            ex["conv"], ex["scam"], ex["tactic"],
            trust=2, tech=0, awareness=0, strict=False, max_chars=400)})
        messages.append({"role": "assistant", "content": f"{ex['warning']}\n{ex['action']}"})
    messages.append({"role": "user", "content": build_user_message(
        conv_text, scam_type, tactic, trust, tech, awareness, strict=strict)})
    return messages

def parse_output(text: str):
    warning, action = "", ""
    for line in text.splitlines():
        u = line.upper()
        if u.startswith("WARNING:"):
            warning = line.split(":", 1)[1].strip()
        elif u.startswith("ACTION:"):
            action = line.split(":", 1)[1].strip()
    if not warning:  # fall back to first non-empty line
        for line in text.strip().splitlines():
            if line.strip():
                warning = line.strip()
                break
    return warning, action

def is_valid(warning: str, action: str) -> bool:
    if not warning or not action:
        return False
    w = warning.lower()
    if any(m in w for m in META_WORDS):
        return False
    nw = len(warning.split())
    if not (5 <= nw <= 45):
        return False
    if len(action.split()) > 25:
        return False
    if not re.search(r"\b(hang up|call|do not|don't|never|stop|check|verify|ignore|block|report|refuse|say no)\b",
                     action.lower()):
        return False
    return True

def load_conversations(dataset_dir: str, split: str):
    """conversation_id -> (conversation_text, victim_config)"""
    convs = {}
    split_dir = os.path.join(dataset_dir, split)
    for fpath in glob.glob(os.path.join(split_dir, "*", "*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            conv = json.load(f)
        parts = [f"{t.get('role','?')}: {t.get('content','')}" for t in conv.get("turns", [])]
        convs[conv.get("conversation_id", os.path.basename(fpath))] = (
            "\n".join(parts), conv.get("victim_config", {}) or {})
    return convs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="test_predictions.jsonl from evaluate_test.py")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    records = [json.loads(l) for l in open(args.predictions, encoding="utf-8")]
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} prediction records")

    convs = load_conversations(args.dataset_dir, args.split)
    print(f"{len(convs)} conversations loaded from split '{args.split}'")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map="cuda")
    model.eval()
    print(f"Model loaded: {args.model}")

    done = {}
    if args.resume and os.path.exists(args.output):
        for r in json.load(open(args.output, encoding="utf-8")):
            done[r["conversation_id"]] = r
        print(f"Resuming: {len(done)} already generated")

    results = list(done.values())
    todo = [r for r in records if r["conversation_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"To generate: {len(todo)}")

    def gen_batch(batch_recs, strict_flags):
        texts = []
        for rec, strict in zip(batch_recs, strict_flags):
            cid = rec["conversation_id"]
            conv_text, vc = convs.get(cid, ("", {}))
            scam = rec.get("pred_scam") or rec.get("scam_type", "bank")
            tactic = rec.get("pred_tactic", "authority_impersonation")
            trust, tech, aw = profile_nums(vc)
            msgs = build_chat(conv_text, scam, tactic, trust, tech, aw, strict=strict)
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=3072).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=110, do_sample=True,
                                 temperature=0.3, top_p=0.9,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        return [parse_output(g) for g in gen]

    t0 = time.time()
    B = args.batch_size
    for i in range(0, len(todo), B):
        batch = todo[i: i + B]
        parsed = gen_batch(batch, [False] * len(batch))
        # quality gate: one strict retry for failures
        retry_idx = [j for j, (w, a) in enumerate(parsed) if not is_valid(w, a)]
        retries = {}
        if retry_idx:
            rp = gen_batch([batch[j] for j in retry_idx], [True] * len(retry_idx))
            for j, (w, a) in zip(retry_idx, rp):
                retries[j] = (w, a)

        for j, rec in enumerate(batch):
            w, a = retries.get(j, parsed[j])
            results.append({
                "conversation_id": rec["conversation_id"],
                "true_scam": rec.get("true_scam"),
                "pred_scam": rec.get("pred_scam"),
                "pred_tactic": rec.get("pred_tactic"),
                "warning": w,
                "action": a,
                "valid": is_valid(w, a),
                "retried": j in retries,
            })
        n_done = len(results)
        if (i // B) % 10 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [{n_done}/{len(todo)} done, {time.time()-t0:.0f}s]")
            # live validity stats
            recent = results[-len(batch):]
            print(f"    valid {sum(r['valid'] for r in recent)}/{len(recent)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- summary ----
    n = len(results)
    valid = sum(r["valid"] for r in results)
    avg_words = sum(len(r["warning"].split()) for r in results) / max(n, 1)
    print(f"\n=== V3 GIST GENERATION SUMMARY ===")
    print(f"Total: {n} | valid: {valid} ({100*valid/max(n,1):.1f}%) | "
          f"retried: {sum(r['retried'] for r in results)}")
    print(f"Avg warning length: {avg_words:.1f} words")
    print("\nSample outputs:")
    for r in results[:3]:
        print(json.dumps({k: r[k] for k in
                          ("true_scam", "pred_scam", "warning", "action", "valid")},
                         indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
