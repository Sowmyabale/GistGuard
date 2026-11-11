"""
dataset_utils.py — Load COVA-X conversations from pre-made split folders.

Expected folder layout (same as split_cova_dataset.py output):
    <dataset_dir>/train/<scam_type>/*.json
    <dataset_dir>/val/<scam_type>/*.json
    <dataset_dir>/test/<scam_type>/*.json

Each JSON: {conversation_id, scam_type, victim_config, attacker_config, turns:[{role, content, turn_number}]}
"""

import os
import glob
import json
import pandas as pd

from config import (
    LABEL_MAP, SCAM_TYPE_TO_TACTIC, TACTIC_MAP,
    TRUST_MAP, TECH_MAP, AWARENESS_MAP,
)

def conversation_to_text(conv: dict) -> str:
    """Join turns into the same 'role: content' text format used in v1."""
    parts = []
    for turn in conv.get("turns", []):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)

def victim_turns_text(conv: dict) -> str:
    """Victim-side turns only (used by the Layer 2 profiler)."""
    parts = []
    for turn in conv.get("turns", []):
        if turn.get("role", "").lower().startswith("victim"):
            parts.append(turn.get("content", ""))
    return " ".join(parts)

def load_split(dataset_dir: str, split: str) -> pd.DataFrame:
    """Load one split into a DataFrame with labels for all three layers."""
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    rows = []
    for scam_type in sorted(os.listdir(split_dir)):
        type_dir = os.path.join(split_dir, scam_type)
        if not os.path.isdir(type_dir):
            continue
        for fpath in sorted(glob.glob(os.path.join(type_dir, "*.json"))):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    conv = json.load(f)
            except Exception as e:
                print(f"  [WARN] skipping {fpath}: {e}")
                continue

            st = conv.get("scam_type", scam_type)
            vc = conv.get("victim_config", {}) or {}

            trust_raw = vc.get("trust_level", "moderately_trusting")
            tech_raw = vc.get("tech_savviness", "moderate")
            aw_raw = vc.get("scam_awareness", "vaguely_aware")

            rows.append({
                "file_path": fpath,
                "conversation_id": conv.get("conversation_id", os.path.basename(fpath)),
                "scam_type": st,
                "label": LABEL_MAP.get(st, -1),
                "tactic": SCAM_TYPE_TO_TACTIC.get(st, "authority_impersonation"),
                "tactic_label": TACTIC_MAP.get(SCAM_TYPE_TO_TACTIC.get(st, "authority_impersonation"), -1),
                "text": conversation_to_text(conv),
                "victim_text": victim_turns_text(conv),
                "trust_label": TRUST_MAP.get(trust_raw, 1),
                "tech_label": TECH_MAP.get(tech_raw, 1),
                "awareness_label": AWARENESS_MAP.get(aw_raw, 1),
                "num_turns": len(conv.get("turns", [])),
            })

    df = pd.DataFrame(rows)
    return df

def load_all_splits(dataset_dir: str):
    """Returns (train_df, val_df, test_df)."""
    train = load_split(dataset_dir, "train")
    val = load_split(dataset_dir, "val")
    test = load_split(dataset_dir, "test")
    return train, val, test

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()
    tr, va, te = load_all_splits(args.dataset_dir)
    for name, df in [("train", tr), ("val", va), ("test", te)]:
        print(f"\n[{name}] n={len(df)}")
        print(df["scam_type"].value_counts())
        print("tactics:", df["tactic"].value_counts().to_dict())
        print("trust:", df["trust_label"].value_counts().to_dict(),
              "| tech:", df["tech_label"].value_counts().to_dict(),
              "| awareness:", df["awareness_label"].value_counts().to_dict())
