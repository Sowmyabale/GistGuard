"""
GXAI-Elderly v2: Full Evaluation Script (with checkpointing + fast mode)
========================================================================

Usage:
    # Fast mode (rule-based gists, no neural model, ~20 min for all splits)
    python run_evaluation.py --fast --split all --dataset_dir ... --output_dir results

    # Neural mode (DialoGPT gists, very slow, ~days)
    python run_evaluation.py --split all --dataset_dir ... --output_dir results

    # Resume from checkpoint
    python run_evaluation.py --fast --split train --dataset_dir ... --output_dir results --resume
"""

import json
import os
import glob
import time
import argparse
from collections import Counter, defaultdict
from datetime import datetime

from config import MANIPULATION_TACTICS, DECEPTION_MARKERS
from layer1_manipulation_detector import ManipulationDetector
from layer2_tom_inference import ToMInferenceEngine, SignUpProfile
from layer3_gist_generator import GistGenerator
from pipeline import GXAIPipeline

SCAM_TYPE_TO_TACTIC = {
    "bank": "authority_impersonation",
    "government_impersonation": "authority_impersonation",
    "grandparent": "over_familiarity",
    "medicare": "authority_impersonation",
    "romance": "over_familiarity",
    "investment": "social_proof",
    "lottery": "reciprocity_gift",
    "virtual_kidnapping": "fear_threat",
}

def get_checkpoint_path(output_dir: str, split_name: str) -> str:
    return os.path.join(output_dir, f"checkpoint_{split_name}.json")

def save_checkpoint(output_dir: str, split_name: str, results: list, metrics: dict):
    path = get_checkpoint_path(output_dir, split_name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"results": results, "metrics": metrics}, f, indent=2)
    print(f"    [Checkpoint saved: {len(results)} processed] -> {path}")

def load_checkpoint(output_dir: str, split_name: str):
    path = get_checkpoint_path(output_dir, split_name)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"    [Resuming from checkpoint: {len(data['results'])} already processed]")
        return data["results"], len(data["results"])
    return [], 0

def load_split(split_dir: str):
    conversations = []
    scam_dirs = glob.glob(os.path.join(split_dir, "*"))
    for scam_dir in scam_dirs:
        if not os.path.isdir(scam_dir):
            continue
        scam_type = os.path.basename(scam_dir)
        for jf in glob.glob(os.path.join(scam_dir, "*.json")):
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                conversations.append((conv, scam_type, os.path.basename(jf)))
            except Exception as e:
                print(f"  Warning: Could not load {jf}: {e}")
    return conversations

def evaluate_split(pipeline, conversations, split_name: str, profile: SignUpProfile,
                   output_dir: str, checkpoint_every: int = 100, resume: bool = False):
    print(f"\n{'='*70}")
    print(f"EVALUATING {split_name.upper()} SPLIT ({len(conversations)} conversations)")
    print(f"{'='*70}")

    # Load checkpoint if resuming
    results, skip_count = load_checkpoint(output_dir, split_name) if resume else ([], 0)

    correct_detection = 0
    correct_tactic_top1 = 0
    correct_tactic_top3 = 0
    total = 0
    tactic_confusion = defaultdict(lambda: Counter())
    timing = []

    for i, (conv, true_type, fname) in enumerate(conversations, 1):
        # Skip already processed
        if i <= skip_count:
            continue

        true_tactic = SCAM_TYPE_TO_TACTIC.get(true_type, "unknown")

        t0 = time.time()
        result = pipeline.process_conversation(conv, profile)
        t1 = time.time()
        timing.append(t1 - t0)

        l1 = result["layer1_manipulation"]
        l2 = result["layer2_tom"]
        l3 = result["layer3_gist"]

        pred_detected = l1["detected"]
        pred_tactic = l1["primary_tactic"]
        top_tactics = list(l1.get("all_scores", {}).keys())[:3]

        if pred_detected:
            correct_detection += 1
        if true_tactic == pred_tactic:
            correct_tactic_top1 += 1
        if true_tactic in top_tactics or true_tactic == pred_tactic:
            correct_tactic_top3 += 1

        total += 1
        tactic_confusion[true_tactic][pred_tactic] += 1

        results.append({
            "file": fname,
            "true_type": true_type,
            "true_tactic": true_tactic,
            "predicted_detected": pred_detected,
            "predicted_tactic": pred_tactic,
            "predicted_confidence": l1.get("confidence", 0.0),
            "top_3_tactics": top_tactics,
            "victim_susceptibility": l2["victim_tom"]["susceptibility"],
            "victim_emotion": l2["victim_tom"]["emotional_state"],
            "gist": l3["gist"],
            "action": l3["action_item"],
            "processing_time_sec": round(t1 - t0, 4),
        })

        if i % checkpoint_every == 0:
            print(f"  Processed {i}/{len(conversations)}...")
            save_checkpoint(output_dir, split_name, results, {})

    # Metrics — computed from the FULL results list (includes resumed entries)
    total = len(results)
    correct_detection = sum(1 for r in results if r.get("predicted_detected", False))
    correct_tactic_top1 = sum(1 for r in results if r.get("true_tactic") == r.get("predicted_tactic"))
    correct_tactic_top3 = sum(
        1 for r in results
        if r.get("true_tactic") in r.get("top_3_tactics", []) or r.get("true_tactic") == r.get("predicted_tactic")
    )
    tactic_confusion = defaultdict(lambda: Counter())
    for r in results:
        tactic_confusion[r.get("true_tactic", "unknown")][r.get("predicted_tactic", "none")] += 1
    all_times = [r.get("processing_time_sec", 0.0) for r in results]
    avg_time = sum(all_times) / len(all_times) if all_times else 0

    detection_rate = correct_detection / total if total else 0
    top1_accuracy = correct_tactic_top1 / total if total else 0
    top3_accuracy = correct_tactic_top3 / total if total else 0

    print(f"\n--- {split_name.upper()} RESULTS ---")
    print(f"  Total conversations: {total}")
    print(f"  Detection rate:      {detection_rate:.2%} ({correct_detection}/{total})")
    print(f"  Top-1 tactic accuracy: {top1_accuracy:.2%} ({correct_tactic_top1}/{total})")
    print(f"  Top-3 tactic accuracy: {top3_accuracy:.2%} ({correct_tactic_top3}/{total})")
    print(f"  Avg processing time: {avg_time:.3f}s per conversation")

    print(f"\n  Tactic confusion (true -> predicted):")
    for true_t, preds in sorted(tactic_confusion.items()):
        print(f"    {true_t:28s}: {dict(preds)}")

    return {
        "split": split_name,
        "metrics": {
            "total": total,
            "detection_rate": round(detection_rate, 4),
            "top1_tactic_accuracy": round(top1_accuracy, 4),
            "top3_tactic_accuracy": round(top3_accuracy, 4),
            "avg_time_sec": round(avg_time, 4),
        },
        "per_conversation": results,
    }

def print_samples(results_data, n=5):
    print("\n" + "=" * 70)
    print(f"SAMPLE GIST OUTPUTS (n={n})")
    print("=" * 70)
    detected = [r for r in results_data["per_conversation"] if r.get("predicted_detected", True)]
    import random
    sample = random.sample(detected, min(n, len(detected)))
    for i, r in enumerate(sample, 1):
        print(f"\n--- Sample {i} ---")
        print(f"  True type:   {r['true_type']}")
        print(f"  True tactic: {r['true_tactic']}")
        print(f"  Predicted:   {r['predicted_tactic']} (conf: {r['predicted_confidence']})")
        print(f"  Victim susceptibility: {r['victim_susceptibility']}")
        print(f"  Gist: {r['gist']}")
        print(f"  Action: {r['action']}")

def main():
    parser = argparse.ArgumentParser(description="GXAI-Elderly v2 Evaluation")
    parser.add_argument("--dataset_dir", type=str,
                        default="dataset/COVA-X Dataset/COVA-X_Split")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"],
                        default="all")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--fast", action="store_true",
                        help="Skip neural model loading; use rule-based gists (much faster)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save checkpoint every N conversations")
    args = parser.parse_args()

    profile = SignUpProfile(
        age_group="70-79", living_situation="alone", tech_comfort=2,
        prior_scam="no", trust_tendency=4, urgency_response=4, financial_control="self",
    )

    print("Initializing v2 pipeline...")
    if args.fast:
        print("  [FAST MODE: Rule-based gist generation, no neural model loaded]")
        pipeline = GXAIPipeline(layer3_model=None)
    else:
        pipeline = GXAIPipeline(layer3_model="microsoft/DialoGPT-medium")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    splits_to_run = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split_name in splits_to_run:
        split_dir = os.path.join(args.dataset_dir, split_name)
        if not os.path.exists(split_dir):
            print(f"WARNING: {split_dir} not found, skipping {split_name}")
            continue

        conversations = load_split(split_dir)
        if not conversations:
            print(f"WARNING: No conversations found in {split_dir}")
            continue

        results = evaluate_split(
            pipeline, conversations, split_name, profile,
            output_dir=args.output_dir,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
        )
        all_results[split_name] = results

        # Save final results
        output_path = os.path.join(args.output_dir, f"v2_eval_{split_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved results to: {output_path}")

        # Remove checkpoint since split is complete
        ckpt_path = get_checkpoint_path(args.output_dir, split_name)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"  Removed checkpoint: {ckpt_path}")

        if args.samples > 0:
            print_samples(results, n=args.samples)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "splits_evaluated": splits_to_run,
        "summary_metrics": {s: all_results[s]["metrics"] for s in all_results},
    }
    summary_path = os.path.join(args.output_dir, "v2_eval_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved summary to: {summary_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

