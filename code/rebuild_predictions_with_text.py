"""Rebuild test_predictions_with_text.jsonl locally.

The public repository does not redistribute COVA-X conversation text
(see README data statement). Run this script on a machine that has the
COVA-X dataset to recreate the joined file used by the external-baseline
scripts (run_smishx_baseline.py, run_ghafariasl_baseline.py).

Usage:
    python rebuild_predictions_with_text.py \
        --dataset_dir "/path/to/COVA-X_Dataset" \
        --predictions results_v1rerun/test_predictions.jsonl \
        --output results_v1rerun/test_predictions_with_text.jsonl
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True,
                    help="Folder with <scam_type>/*.json conversations")
    ap.add_argument("--predictions", required=True, help="test_predictions.jsonl")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    convs = {}
    for fpath in glob.glob(os.path.join(args.dataset_dir, "*", "*.json")):
        with open(fpath, encoding="utf-8") as f:
            conv = json.load(f)
        parts = [f"{t.get('role', '?')}: {t.get('content', '')}"
                 for t in conv.get("turns", [])]
        convs[conv.get("conversation_id", os.path.basename(fpath))] = "\n".join(parts)

    n = miss = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for line in open(args.predictions, encoding="utf-8"):
            row = json.loads(line)
            text = convs.get(row["conversation_id"])
            if text is None:
                miss += 1
                continue
            row["text"] = text
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows -> {args.output} (missing conversations: {miss})")


if __name__ == "__main__":
    main()
