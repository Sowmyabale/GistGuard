"""
train_gist_lora.py — Step 2 of Option B: LoRA SFT of the gist generator.

Trains Llama-3.1-8B (4-bit) + LoRA adapters on the silver dataset, with
COMPLETION-ONLY loss (prompt tokens are masked — the model learns to produce
WARNING/ACTION, not to echo the prompt).

Deployment prompt is intentionally SHORT (this same short prompt is used at
inference — the capability comes from training, not prompt engineering).

"""

import os
import json
import argparse
import random

from config import SEED
from generate_gists_v3 import SCAM_PLAIN, PROFILE_WORDS

def short_user_message(rec, max_chars: int = 1200) -> str:
    """Deployment-style SHORT prompt (no few-shot, no rule list).
    If the record carries an evidence_card (silver v2), it is injected so the
    model grounds the warning in the conversation facts.
    If rec['trust'] is None, the victim profile is reported as unavailable
    (profile-ablation runs)."""
    excerpt = rec["conv_text"][:max_chars].strip()
    if len(rec["conv_text"]) > max_chars:
        excerpt += " ..."
    scam_words = SCAM_PLAIN.get(rec["scam_type"], f"a {rec['scam_type'].replace('_', ' ')} scam")
    if rec.get("trust") is None:
        profile = "the victim profile is not available"
    else:
        profile = (f"the victim is {PROFILE_WORDS['trust'].get(rec['trust'], 'somewhat trusting')}, "
                   f"{PROFILE_WORDS['tech'].get(rec['tech'], 'somewhat comfortable with technology')}, and "
                   f"{PROFILE_WORDS['awareness'].get(rec['awareness'], 'vaguely aware of scams')}")
    card = rec.get("evidence_card")
    card_block = (f"Evidence from the conversation:\n{card}\n\n") if card else ""
    return (
        f"This conversation is {scam_words} using the tactic "
        f"{rec['tactic'].replace('_', ' ')}; {profile}.\n\n"
        f"{card_block}"
        f"Conversation:\n\"\"\"\n{excerpt}\n\"\"\"\n\n"
        "Respond with the WARNING and ACTION."
    )

def target_text(rec) -> str:
    return f"WARNING: {rec['warning']}\nACTION: {rec['action']}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
        Trainer, TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

    rows = [json.loads(l) for l in open(args.silver, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    random.shuffle(rows)
    print(f"Silver examples: {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    SYSTEM = ("You are a protective assistant for elderly people. "
              "You write very short, plain-language, actionable scam warnings.")

    def encode(rec):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": short_user_message(rec)}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        full = prompt + target_text(rec) + "<|eot_id|>"
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        # completion-only masking; if prefix misaligns (BPE edge), train on full seq
        labels = list(f_ids)
        if f_ids[: len(p_ids)] == p_ids:
            labels[: len(p_ids)] = [-100] * len(p_ids)
        else:
            print("WARN: prompt not a token-prefix; training on full sequence")
        return {"input_ids": f_ids[: args.max_len], "labels": labels[: args.max_len]}

    encoded = [encode(r) for r in rows]

    class Collator:
        def __call__(self, feats):
            maxlen = max(len(f["input_ids"]) for f in feats)
            pad = tokenizer.pad_token_id
            input_ids, labels, attn = [], [], []
            for f in feats:
                ids = f["input_ids"]; labs = f["labels"]
                padlen = maxlen - len(ids)
                input_ids.append(ids + [pad] * padlen)
                labels.append(labs + [-100] * padlen)
                attn.append([1] * len(ids) + [0] * padlen)
            return {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn),
            }

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="cuda")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir + "/trainer",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
        seed=SEED,
        gradient_checkpointing=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=encoded,
                      data_collator=Collator())
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved LoRA adapter -> {args.output_dir}")

if __name__ == "__main__":
    main()
