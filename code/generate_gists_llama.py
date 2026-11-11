"""
generate_gists_llama.py — Layer 3 RERUN: Llama-3.1-8B zero-shot gist generation

Reads test predictions (test_predictions.jsonl produced by evaluate_test.py),
builds the SAME zero-shot gist prompt validated in the v2 run, and generates
GIST + ACTION for each conversation. Checkpoint/resume supported.

"""

import os
import json
import time
import argparse

EMOTION_BY_PROFILE = {
    # (trust_label, awareness_label) -> emotional state wording for the prompt
    (2, 0): "frightened and trusting",
    (2, 1): "worried but trusting",
    (2, 2): "trusting but alert",
    (1, 0): "uncertain",
    (1, 1): "somewhat concerned",
    (1, 2): "cautious",
    (0, 0): "skeptical",
    (0, 1): "skeptical",
    (0, 2): "skeptical and alert",
}

MAX_WORDS = 30  # gist complexity target (simplest tier for high susceptibility)

def build_prompt(tactic: str, emotion: str) -> str:
    system_msg = ("You are a protective assistant that helps elderly people avoid scams. "
                  "You write very short, plain-language, actionable warnings.")
    user_msg = (
        f"An elderly person is receiving a scam attempt that uses the tactic: "
        f"{tactic.replace('_', ' ')}.\n"
        f"The person is currently feeling: {emotion}.\n\n"
        f"Give them the BOTTOM-LINE MEANING of what is happening, in plain, calm, "
        f"simple language — no technical terms, no jargon, no lecturing. "
        f"Focus on what the scammer is really trying to do.\n"
        f"Then give one concrete action item.\n\n"
        f"Reply in EXACTLY this format:\n"
        f"GIST: <one or two short sentences, under {MAX_WORDS} words>\n"
        f"ACTION: <one short, concrete instruction>"
    )
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def parse_output(text: str):
    gist, action = "", ""
    for line in text.splitlines():
        if line.upper().startswith("GIST:"):
            gist = line.split(":", 1)[1].strip()
        elif line.upper().startswith("ACTION:"):
            action = line.split(":", 1)[1].strip()
    if not gist:
        gist = text.strip().splitlines()[0] if text.strip() else ""
    return gist, action

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        records = records[:args.limit]

    done = {}
    if args.resume and os.path.exists(args.output):
        for r in json.load(open(args.output, encoding="utf-8")):
            done[r["conversation_id"]] = r
        print(f"Resuming: {len(done)} already generated")

    remaining = [r for r in records if r["conversation_id"] not in done]
    print(f"Total={len(records)} | remaining={len(remaining)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 4-bit quantization via BitsAndBytesConfig (transformers 5.x API; same as validated Sept-16 run)
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    model.eval()

    results = list(done.values())
    bs = args.batch_size
    t0 = time.time()
    for i in range(0, len(remaining), bs):
        batch = remaining[i:i + bs]
        prompts = []
        for rec in batch:
            prof = rec.get("profile", {})
            emotion = EMOTION_BY_PROFILE.get(
                (prof.get("trust_label", 1), prof.get("awareness_label", 1)),
                "somewhat concerned",
            )
            prompts.append(build_prompt(rec["pred_tactic"], emotion))

        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, rec in enumerate(batch):
            text = tokenizer.decode(out[j][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            gist, action = parse_output(text)
            results.append({
                "conversation_id": rec["conversation_id"],
                "scam_type": rec["scam_type"],
                "pred_scam": rec["pred_scam"],
                "pred_tactic": rec["pred_tactic"],
                "scam_correct": rec["scam_correct"],
                "profile": rec.get("profile", {}),
                "llama_gist": gist,
                "llama_action": action,
            })

        n_done = len(results)
        if (i // bs + 1) % 10 == 0 or i + bs >= len(remaining):
            json.dump(results, open(args.output, "w", encoding="utf-8"), indent=2)
            el = time.time() - t0
            print(f"  {n_done}/{len(records)} done | {el/60:.1f} min elapsed")

    json.dump(results, open(args.output, "w", encoding="utf-8"), indent=2)
    lens = [len(r["llama_gist"].split()) for r in results if r["llama_gist"]]
    print(f"\nDone. {len(results)} gists | avg {sum(lens)/max(len(lens),1):.1f} words")
    print(f"Saved -> {args.output}")

if __name__ == "__main__":
    main()
