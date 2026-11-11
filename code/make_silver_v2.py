"""
make_silver_v2.py — Grounded silver data for LoRA v4 (personalized gists).

vs make_silver_dataset.py (silver v1):
    + EVIDENCE CARD injected into the prompt (claimed identity, requested items,
      payment channel, pressure) — the warning is FORCED to name what the caller
      asked for and who they claimed to be.
    + REAL tactic labels from annotate_tactics_llm.py when available (falls back
      to the scam-type mapping otherwise).
    + QC v2: format validity (is_valid) AND ask-coverage AND identity-coverage
      (extract_evidence). One strict retry per failure.

Output rows (superset of silver v1 — backward compatible with train_gist_lora):
    {conversation_id, scam_type, tactic, trust, tech, awareness, conv_text,
     warning, action, evidence_card, claimed_identity, requested_items,
     valid, retried, ask_cov, id_cov}

"""

import os
import json
import time
import argparse

from dataset_utils import load_split
from generate_gists_v3 import (
    parse_output, is_valid, profile_nums, load_conversations,
    SCAM_PLAIN, PROFILE_WORDS,
)
from extract_evidence import (
    evidence_card_fields, format_card, ask_coverage, identity_coverage,
)

SYSTEM = ("You are a protective assistant for elderly people. You write very short, "
          "plain-language, actionable scam warnings.")

def build_user_message_v2(conv_text, scam_type, tactic, trust, tech, awareness,
                          card_fields, strict=False, max_chars=1200):
    excerpt = conv_text[:max_chars].strip()
    if len(conv_text) > max_chars:
        excerpt += " ..."
    scam_words = SCAM_PLAIN.get(scam_type, f"a {scam_type.replace('_', ' ')} scam")
    profile = (f"They are {PROFILE_WORDS['trust'].get(trust, 'somewhat trusting')}, "
               f"{PROFILE_WORDS['tech'].get(tech, 'somewhat comfortable with technology')}, and "
               f"{PROFILE_WORDS['awareness'].get(awareness, 'vaguely aware of scams')}.")
    card = format_card(card_fields)

    rules = (
        "Rules:\n"
        "- Write ONLY the warning and the action. Never explain your task.\n"
        "- Never use the words: gist, bottom-line, scam tactic, format.\n"
        "- The warning MUST say who the caller claims to be and what they ask for, "
        "using the evidence below. Mention the requested item explicitly.\n"
        "- Two sentences maximum for the warning. Plain, calm words an 80-year-old understands.\n"
        "- The action must be one concrete thing they can do right now.\n"
    )
    if strict:
        rules += ("- Your previous reply broke the rules. This time name the requested item "
                  "and the claimed identity from the evidence, and output ONLY the two "
                  "labeled lines.\n")

    return (
        "You are protecting an elderly person from a phone or text scam.\n\n"
        f"Scam type: this is {scam_words}.\n"
        f"Manipulation tactic used: {tactic.replace('_', ' ')}.\n"
        f"Victim profile: {profile}\n\n"
        f"Evidence from the conversation:\n{card}\n\n"
        "Conversation so far:\n"
        f"\"\"\"\n{excerpt}\n\"\"\"\n\n"
        f"{rules}\n"
        "Reply in EXACTLY this format:\n"
        "WARNING: <one or two short sentences about what the scammer is really doing>\n"
        "ACTION: <one short concrete instruction>"
    )

# Few-shot examples with their own evidence cards (grounded demonstrations)
FEW_SHOT_V2 = [
    {
        "user": build_user_message_v2(
            conv_text=("caller: This is the fraud department of Chase bank. Your account has been "
                       "compromised. I need you to verify your card number and the CVV code now.\n"
                       "victim: Oh no. Is my money safe?"),
            scam_type="bank", tactic="authority_impersonation",
            trust=2, tech=0, awareness=0, max_chars=400,
            card_fields={"claimed_identity": "chase fraud department",
                         "requested_items": ["card number / full card", "CVV / security code"],
                         "payment_channels": [],
                         "pressure": []}),
        "assistant": ("WARNING: This caller says they are from the Chase fraud department, but they are "
                      "not. They are asking for your card number and CVV code — real banks never ask "
                      "for those on a call you did not start.\n"
                      "ACTION: Hang up and call the number on the back of your bank card."),
    },
    {
        "user": build_user_message_v2(
            conv_text=("caller: This is the IRS. You owe back taxes and there is a warrant for your "
                       "arrest. Pay the balance today with gift cards to clear this.\n"
                       "victim: Arrest? What do I do?"),
            scam_type="government_impersonation", tactic="fear_threat",
            trust=1, tech=0, awareness=1, max_chars=400,
            card_fields={"claimed_identity": "irs",
                         "requested_items": ["gift cards"],
                         "payment_channels": ["gift cards"],
                         "pressure": ["threat"]}),
        "assistant": ("WARNING: This is not the IRS. The IRS never calls about arrest warrants and "
                      "never accepts gift cards — asking you to pay with gift cards is how they "
                      "steal your money.\n"
                      "ACTION: Hang up and do not buy any gift cards. If you are worried, call the "
                      "real IRS at 1-800-829-1040."),
    },
]

def build_chat_v2(conv_text, scam_type, tactic, trust, tech, awareness,
                  card_fields, strict=False):
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in FEW_SHOT_V2:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": build_user_message_v2(
        conv_text, scam_type, tactic, trust, tech, awareness,
        card_fields, strict=strict)})
    return messages

def is_valid_v2(warning, action, card_fields):
    if not is_valid(warning, action):
        return False
    if not ask_coverage(warning, card_fields.get("requested_items")):
        return False
    if not identity_coverage(warning, card_fields.get("claimed_identity", "")):
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--annotations", default=None,
                    help="tactic_annotations.jsonl from annotate_tactics_llm.py")
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

    real_tactics = {}
    if args.annotations and os.path.exists(args.annotations):
        for l in open(args.annotations, encoding="utf-8"):
            r = json.loads(l)
            if r.get("dominant_tactic") and r["dominant_tactic"] != "none":
                real_tactics[r["conversation_id"]] = r["dominant_tactic"]
        print(f"Real tactic labels loaded: {len(real_tactics)}")

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
    for path_ in (silver_path, rej_path):
        if args.resume and os.path.exists(path_):
            for l in open(path_, encoding="utf-8"):
                try:
                    done_ids.add(json.loads(l)["conversation_id"])
                except Exception:
                    pass
    todo = [r for _, r in train_df.iterrows() if r["conversation_id"] not in done_ids]
    print(f"Resume: {len(done_ids)} done | To process: {len(todo)}")

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
            tactic = real_tactics.get(cid) or row["tactic"]
            fields = evidence_card_fields(conv_text, row["scam_type"])
            msgs = build_chat_v2(conv_text, row["scam_type"], tactic,
                                 trust, tech, aw, fields, strict=False)
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
            metas.append({
                "conversation_id": cid, "scam_type": row["scam_type"],
                "tactic": tactic, "trust": trust, "tech": tech, "awareness": aw,
                "conv_text": conv_text, "evidence_card": format_card(fields),
                "claimed_identity": fields["claimed_identity"],
                "requested_items": fields["requested_items"],
            })

        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=3072).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=120, do_sample=True,
                                 temperature=0.5, top_p=0.9,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

        parsed = [parse_output(g) for g in gen]
        retry_idx = [j for j, (w, a) in enumerate(parsed)
                     if not is_valid_v2(w, a, evidence_card_fields(
                         metas[j]["conv_text"], metas[j]["scam_type"]))]
        retries = {}
        if retry_idx:
            rtexts = []
            for j in retry_idx:
                m = metas[j]
                fields = evidence_card_fields(m["conv_text"], m["scam_type"])
                msgs = build_chat_v2(m["conv_text"], m["scam_type"], m["tactic"],
                                     m["trust"], m["tech"], m["awareness"],
                                     fields, strict=True)
                rtexts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
            enc = tokenizer(rtexts, return_tensors="pt", padding=True, truncation=True,
                            max_length=3072).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=120, do_sample=True,
                                     temperature=0.4, top_p=0.9,
                                     pad_token_id=tokenizer.pad_token_id)
            rgen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for j, g in zip(retry_idx, rgen):
                retries[j] = parse_output(g)

        for j, meta in enumerate(metas):
            w, a = retries.get(j, parsed[j])
            fields = evidence_card_fields(meta["conv_text"], meta["scam_type"])
            meta.update({
                "warning": w, "action": a,
                "valid": is_valid_v2(w, a, fields), "retried": j in retries,
                "ask_cov": ask_coverage(w, fields["requested_items"]),
                "id_cov": identity_coverage(w, fields["claimed_identity"]),
            })
            (fs if meta["valid"] else fr).write(
                json.dumps(meta, ensure_ascii=False) + "\n")
        fs.flush(); fr.flush()
        if (i // B) % 20 == 0:
            print(f"  [{i + len(batch)}/{len(todo)} done, {time.time()-t0:.0f}s]")

    fs.close(); fr.close()
    ns = sum(1 for _ in open(silver_path, encoding="utf-8"))
    nr = sum(1 for _ in open(rej_path, encoding="utf-8")) if os.path.exists(rej_path) else 0
    print(f"\n=== SILVER V2 SUMMARY ===")
    print(f"valid (silver): {ns} | rejected: {nr} | acceptance: {100*ns/max(ns+nr,1):.1f}%")
    print(f"saved -> {silver_path}")

if __name__ == "__main__":
    main()
