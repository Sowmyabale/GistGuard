"""
GXAI-Elderly v2: Evaluation Script
===================================
Evaluates the v2 framework on:
  1. COVA-X dataset (known scam types)
  2. YouTube transcripts (real-world, unlabeled)

Outputs:
  - detection accuracy on COVA-X
  - tactic distribution
  - ToM inference statistics
  - sample gist outputs
  - JSON results file
"""

import json
import os
import sys
import glob
import random
from collections import Counter, defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v2_framework.config import MANIPULATION_TACTICS, DECEPTION_MARKERS, RESULTS_DIR
from v2_framework.layer1_manipulation_detector import ManipulationDetector
from v2_framework.layer2_tom_inference import ToMInferenceEngine, SignUpProfile
from v2_framework.layer3_gist_generator import GistGenerator
from v2_pipeline import GXAIPipeline

def load_cova_conversations(dataset_dir: str, max_per_type: int = 20):
    """
    Load COVA-X conversations.
    Returns list of (conversation, true_tactic) tuples.
    """
    conversations = []
    scam_dirs = glob.glob(os.path.join(dataset_dir, "*"))

    for scam_dir in scam_dirs:
        scam_type = os.path.basename(scam_dir)
        if not os.path.isdir(scam_dir):
            continue

        json_files = glob.glob(os.path.join(scam_dir, "*.json"))
        random.shuffle(json_files)

        for jf in json_files[:max_per_type]:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                conversations.append((conv, scam_type))
            except Exception as e:
                print(f"Warning: Could not load {jf}: {e}")

    return conversations

def map_scam_type_to_tactic(scam_type: str) -> str:
    """Map COVA-X scam type to manipulation tactic for evaluation."""
    mapping = {
        "bank": "authority_impersonation",
        "government_impersonation": "authority_impersonation",
        "grandparent": "over_familiarity",
        "medicare": "authority_impersonation",
        "romance": "over_familiarity",
        "investment": "social_proof",
        "lottery": "reciprocity_gift",
        "virtual_kidnapping": "fear_threat",
    }
    return mapping.get(scam_type, "unknown")

def evaluate_on_cova(pipeline: GXAIPipeline, dataset_dir: str, max_per_type: int = 20):
    """Evaluate v2 pipeline on COVA-X dataset."""
    print("\n" + "=" * 70)
    print("EVALUATING ON COVA-X DATASET")
    print("=" * 70)

    conversations = load_cova_conversations(dataset_dir, max_per_type)
    print(f"Loaded {len(conversations)} conversations")

    # Default profile for evaluation
    profile = SignUpProfile(
        age_group="70-79",
        living_situation="alone",
        tech_comfort=2,
        prior_scam="no",
        trust_tendency=4,
        urgency_response=4,
        financial_control="self",
    )

    results = []
    correct_detection = 0
    correct_tactic = 0
    total = 0

    tactic_confusion = defaultdict(lambda: Counter())

    for conv, true_type in tqdm(conversations, desc="Processing COVA-X"):
        true_tactic = map_scam_type_to_tactic(true_type)

        result = pipeline.process_conversation(conv, profile)
        l1 = result["layer1_manipulation"]
        l2 = result["layer2_tom"]
        l3 = result["layer3_gist"]

        pred_detected = l1["detected"]
        pred_tactic = l1["primary_tactic"]

        # Check detection
        if pred_detected:
            correct_detection += 1

        # Check tactic match (loose: top-3)
        top_tactics = list(l1["all_scores"].keys())[:3]
        if true_tactic in top_tactics or true_tactic == pred_tactic:
            correct_tactic += 1

        total += 1

        tactic_confusion[true_tactic][pred_tactic] += 1

        results.append({
            "conversation_id": conv.get("conversation_id", ""),
            "true_type": true_type,
            "true_tactic": true_tactic,
            "predicted_detected": pred_detected,
            "predicted_tactic": pred_tactic,
            "predicted_confidence": l1["confidence"],
            "victim_susceptibility": l2["victim_tom"]["susceptibility"],
            "victim_emotion": l2["victim_tom"]["emotional_state"],
            "gist": l3["gist"],
            "action": l3["action_item"],
        })

    # Metrics
    detection_rate = correct_detection / total if total else 0
    tactic_accuracy = correct_tactic / total if total else 0

    print(f"\n--- COVA-X RESULTS ---")
    print(f"  Total conversations: {total}")
    print(f"  Detection rate: {detection_rate:.2%} ({correct_detection}/{total})")
    print(f"  Top-3 tactic accuracy: {tactic_accuracy:.2%} ({correct_tactic}/{total})")

    print(f"\n  Tactic confusion (true -> predicted):")
    for true_t, preds in tactic_confusion.items():
        print(f"    {true_t}: {dict(preds)}")

    # Save results
    output_path = os.path.join(RESULTS_DIR, "v2_cova_evaluation.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": {
                "total": total,
                "detection_rate": detection_rate,
                "tactic_accuracy": tactic_accuracy,
            },
            "per_conversation": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {output_path}")

    return results

def evaluate_on_youtube(pipeline: GXAIPipeline, youtube_csv: str, max_samples: int = 50):
    """Evaluate on YouTube transcripts."""
    print("\n" + "=" * 70)
    print("EVALUATING ON YOUTUBE TRANSCRIPTS")
    print("=" * 70)

    import pandas as pd
    df = pd.read_csv(youtube_csv)
    print(f"Loaded {len(df)} YouTube transcripts")

    sample = df.sample(min(max_samples, len(df)), random_state=42)

    profile = SignUpProfile(
        age_group="70-79",
        living_situation="alone",
        tech_comfort=2,
        prior_scam="no",
        trust_tendency=4,
        urgency_response=4,
        financial_control="self",
    )

    results = []
    detected_count = 0

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Processing YouTube"):
        text = str(row.get("Transcript", row.get("transcript", "")))
        if len(text) < 20:
            continue

        result = pipeline.process_text(text, profile)
        l1 = result["layer1_manipulation"]

        if l1["detected"]:
            detected_count += 1

        results.append({
            "youtube_id": row.get("Scam_ID", row.get("Video_ID", "unknown")),
            "text_preview": text[:200],
            "detected": l1["detected"],
            "primary_tactic": l1["primary_tactic"],
            "confidence": l1["confidence"],
            "victim_susceptibility": result["layer2_tom"]["victim_tom"]["susceptibility"],
            "gist": result["layer3_gist"]["gist"],
        })

    detection_rate = detected_count / len(results) if results else 0

    print(f"\n--- YOUTUBE RESULTS ---")
    print(f"  Total samples: {len(results)}")
    print(f"  Detected as scam: {detection_rate:.2%} ({detected_count}/{len(results)})")

    # Tactic distribution
    tactics = [r["primary_tactic"] for r in results if r["detected"]]
    tactic_dist = Counter(tactics)
    print(f"\n  Detected tactic distribution:")
    for tactic, count in tactic_dist.most_common():
        print(f"    {tactic}: {count}")

    # Save results
    output_path = os.path.join(RESULTS_DIR, "v2_youtube_evaluation.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": {
                "total": len(results),
                "detection_rate": detection_rate,
            },
            "tactic_distribution": dict(tactic_dist),
            "per_transcript": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {output_path}")

    return results

def print_sample_gists(results, n=5):
    """Print n sample gist outputs for manual inspection."""
    print("\n" + "=" * 70)
    print(f"SAMPLE GIST OUTPUTS (n={n})")
    print("=" * 70)

    detected = [r for r in results if r.get("detected", True)]
    sample = random.sample(detected, min(n, len(detected)))

    for i, r in enumerate(sample, 1):
        print(f"\n--- Sample {i} ---")
        print(f"  True type: {r.get('true_type', 'N/A')}")
        print(f"  Predicted tactic: {r.get('predicted_tactic', r.get('primary_tactic', 'N/A'))}")
        print(f"  Victim susceptibility: {r.get('victim_susceptibility', 'N/A')}")
        print(f"  Gist: {r['gist']}")
        print(f"  Action: {r['action']}")

def main():
    dataset_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset", "COVA-X Dataset", "COVA-X_Dataset"
    )
    youtube_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset", "Youtube transcripts", "Labeled_ScamCallTranscripts_V1.csv"
    )

    print("Initializing v2 pipeline...")
    pipeline = GXAIPipeline()

    # Evaluate on COVA-X
    cova_results = []
    if os.path.exists(dataset_dir):
        cova_results = evaluate_on_cova(pipeline, dataset_dir, max_per_type=20)
        print_sample_gists(cova_results, n=5)
    else:
        print(f"COVA-X dataset not found at {dataset_dir}")

    # Evaluate on YouTube
    youtube_results = []
    if os.path.exists(youtube_csv):
        youtube_results = evaluate_on_youtube(pipeline, youtube_csv, max_samples=50)
    else:
        print(f"YouTube transcripts not found at {youtube_csv}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
