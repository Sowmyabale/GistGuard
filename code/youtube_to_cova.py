"""
youtube_to_cova.py — Convert YouTube scam-call transcripts into COVA-X-shaped
JSONs so the trained pipeline (Layer 1, Layer 3) can run on REAL conversations.

Input:  Labeled_ScamCallTranscripts_V1.csv  (columns: ID, Source, Content, Char_Len)
Output: <out_dir>/unknown/yt_<ID>.json  (one JSON per transcript)

Notes:
    - Transcripts have no speaker turns and no labels in the CSV, so each becomes
      a single-turn conversation with scam_type "unknown" and empty victim_config.
    - The "unknown" scam_type means: predictions on these are REAL-WORLD outputs,
      to be scored against manual labels (annotate a stratified sample, or use
      the full set once labels are added to the CSV as a 'scam_type' column —
      this converter picks that column up automatically if present).

Usage (any machine):
    python youtube_to_cova.py \
        --csv "/path/to/Labeled_ScamCallTranscripts_V1.csv" \
        --out_dir "/path/to/COVA-X Dataset/YouTube_Split"
"""

import os
import csv
import json
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    n_ok, n_skip, n_labeled = 0, 0, 0
    with open(args.csv, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        has_label = "scam_type" in cols
        for row in reader:
            content = (row.get("Content") or "").strip()
            if len(content) < 50:          # skip trivially short fragments
                n_skip += 1
                continue
            scam_type = (row.get("scam_type") or "unknown").strip() or "unknown"
            if has_label and scam_type != "unknown":
                n_labeled += 1
            conv = {
                "conversation_id": f"yt_{row.get('ID', n_ok)}",
                "scam_type": scam_type,
                "victim_config": {},
                "attacker_config": {},
                "turns": [{"role": "caller", "content": content, "turn_number": 1}],
                "source": row.get("Source", ""),
                "char_len": int(row.get("Char_Len") or len(content)),
            }
            # NOTE: files go under <out_dir>/test/<scam_type>/ so the existing
            # pipeline scripts (predict_youtube, generate_gists_lora with
            # --split test, annotate_tactics_llm with --splits test) work as-is.
            type_dir = os.path.join(args.out_dir, "test", scam_type)
            os.makedirs(type_dir, exist_ok=True)
            with open(os.path.join(type_dir, conv["conversation_id"] + ".json"),
                      "w", encoding="utf-8") as f2:
                json.dump(conv, f2, indent=2, ensure_ascii=False)
            n_ok += 1

    print(f"Converted: {n_ok} | skipped (too short): {n_skip} | "
          f"with gold labels: {n_labeled}")
    print(f"Output -> {args.out_dir}/test/<scam_type>/  (split folder included)")

if __name__ == "__main__":
    main()
