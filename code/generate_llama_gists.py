"""
Generate Llama-3 gists for a stratified sample of evaluated conversations
==========================================================================
Reuses stored Layer 1 / Layer 2 outputs from run_evaluation.py results
(v2_eval_<split>.json), so ONLY Layer 3 (gist generation) runs on the GPU.

Usage:
    python generate_llama_gists.py \
        --eval_json results/v2_eval_test.json \
        --n_per_type 60 \
        --output results/llama_gists_test.json

    # Resume after interruption:
    python generate_llama_gists.py ... --resume
"""

import json
import os
import time
import argparse
import random
from collections import defaultdict

from layer3_gist_generator import GistGenerator

def stratified_sample(records, n_per_type, seed=42):
    """Pick n_per_type detected conversations per scam type, fixed seed."""
    by_type = defaultdict(list)
    for idx, r in enumerate(records):
        if r.get("predicted_detected") and r.get("predicted_tactic", "none") != "none":
            by_type[r["true_type"]].append(idx)
    rng = random.Random(seed)
    picked = []
    for t, idxs in sorted(by_type.items()):
        rng.shuffle(idxs)
        picked.extend(idxs[:n_per_type])
    return sorted(picked)

def main():
    parser = argparse.ArgumentParser(description="Llama gist generation on evaluated sample")
    parser.add_argument("--eval_json", type=str, default="results/v2_eval_test.json")
    parser.add_argument("--n_per_type", type=int, default=60)
    parser.add_argument("--output", type=str, default="results/llama_gists_test.json")
    parser.add_argument("--model", type=str, default="llama",
                        help="'llama' (default) or 'mistral'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=25)
    args = parser.parse_args()

    with open(args.eval_json, 'r', encoding='utf-8') as f:
        records = json.load(f)["per_conversation"]

    sample_idx = stratified_sample(records, args.n_per_type, args.seed)
    print(f"Sampled {len(sample_idx)} conversations "
          f"({args.n_per_type} per type x {len(set(records[i]['true_type'] for i in sample_idx))} types)")

    # Resume support
    done = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for entry in json.load(f):
                done[entry["file"]] = entry
        print(f"Resuming: {len(done)} already generated")

    model_name = "Meta-Llama-3.1-8B" if args.model == "llama" else args.model
    print(f"Loading gist generator: {model_name} (4-bit)")
    gen = GistGenerator(model_name=model_name)
    if gen.model is None:
        raise RuntimeError("Model failed to load — check bitsandbytes/accelerate installation")

    results = list(done.values())
    t_start = time.time()
    new_count = 0

    for n, idx in enumerate(sample_idx, 1):
        r = records[idx]
        if r["file"] in done:
            continue

        layer1 = {"primary_tactic": r["predicted_tactic"], "detected": True,
                  "confidence": r.get("predicted_confidence", 0.0)}
        tom = {"victim_tom": {"emotional_state": r.get("victim_emotion", "neutral"),
                              "cognitive_load": "medium"},
               "scammer_tom": {}}

        t0 = time.time()
        out = gen.generate(layer1, tom, tech_comfort=2)
        dt = time.time() - t0

        results.append({
            "file": r["file"],
            "true_type": r["true_type"],
            "true_tactic": r["true_tactic"],
            "predicted_tactic": r["predicted_tactic"],
            "predicted_confidence": r.get("predicted_confidence", 0.0),
            "victim_emotion": r.get("victim_emotion", "neutral"),
            "victim_susceptibility": r.get("victim_susceptibility", None),
            "rule_gist": r.get("gist", ""),
            "rule_action": r.get("action", ""),
            "llama_gist": out["gist"],
            "llama_action": out["action_item"],
            "method": out.get("method", "unknown"),
            "gen_time_sec": round(dt, 2),
        })
        new_count += 1

        if new_count % 5 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / new_count * (len(sample_idx) - len(done) - new_count)
            print(f"  [{len(results)}/{len(sample_idx)}] last: {dt:.1f}s | ETA {eta/60:.1f} min")

        if new_count % args.save_every == 0:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"    [Saved {len(results)} -> {args.output}]")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDONE. {len(results)} conversations -> {args.output}")

    # Quick stats
    lens = [len(e["llama_gist"].split()) for e in results]
    print(f"Avg gist length: {sum(lens)/len(lens):.1f} words")
    print("\n--- Sample outputs ---")
    for e in results[:4]:
        print(f"\n[{e['true_type']} | pred: {e['predicted_tactic']}]")
        print(f"  RULE : {e['rule_gist'][:110]}")
        print(f"  LLAMA: {e['llama_gist'][:110]}")
        print(f"  ACTION: {e['llama_action'][:110]}")

if __name__ == "__main__":
    main()

