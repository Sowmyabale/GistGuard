"""
predict_youtube.py — Run trained Layer 1 (v1.5 RoBERTa multi-task) over the
YouTube real-world transcripts and write predictions.

Real-world protocol (no victim_config, no gold labels needed for generation):
    - Scam type + tactic predicted per transcript.
    - Output JSONL feeds Layer 3 (gists) and, once labels exist, accuracy scoring.

"""

import os
import glob
import json
import argparse
from collections import Counter

import torch
from transformers import AutoTokenizer

from config import ID2LABEL, ID2TACTIC, MAX_LENGTH, MODEL_NAME
from train_layer1_multitask import MultiTaskScamClassifier
from dataset_utils import conversation_to_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--youtube_dir", required=True)
    ap.add_argument("--layer1_dir", default="models/layer1_multitask")
    ap.add_argument("--output", default="results_v1rerun/youtube_predictions.jsonl")
    ap.add_argument("--model_name", default=MODEL_NAME)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.youtube_dir, "*", "*.json")))
    print(f"{len(files)} YouTube transcripts")

    tokenizer = AutoTokenizer.from_pretrained(args.layer1_dir + "/tokenizer")
    model = MultiTaskScamClassifier(args.model_name, num_scam=8, num_tactic=6)
    state = torch.load(os.path.join(args.layer1_dir, "layer1_multitask.pt"),
                       map_location="cuda")
    model.load_state_dict(state)
    model = model.cuda().eval()

    rows = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            conv = json.load(f)
        rows.append({
            "conversation_id": conv.get("conversation_id", os.path.basename(fpath)),
            "scam_type": conv.get("scam_type", "unknown"),
            "source": conv.get("source", ""),
            "text": conversation_to_text(conv),
        })

    with open(args.output, "w", encoding="utf-8") as fout:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i: i + args.batch_size]
            enc = tokenizer([r["text"] for r in batch], truncation=True,
                            max_length=MAX_LENGTH, padding=True,
                            return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**enc)
            sp = torch.softmax(out["scam_logits"], -1).cpu().numpy()
            tp = torch.softmax(out["tactic_logits"], -1).cpu().numpy()
            for j, rec in enumerate(batch):
                rec.update({
                    "pred_scam": ID2LABEL[int(sp[j].argmax())],
                    "pred_tactic": ID2TACTIC[int(tp[j].argmax())],
                    "scam_probs": {ID2LABEL[k]: round(float(sp[j][k]), 4)
                                   for k in range(len(ID2LABEL))},
                })
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i // args.batch_size) % 10 == 0:
                print(f"  [{i + len(batch)}/{len(rows)}]")

    # ---- distribution summary ----
    scam_dist = Counter(r["pred_scam"] for r in rows)
    tac_dist = Counter(r["pred_tactic"] for r in rows)
    print("\n=== YOUTUBE PREDICTION SUMMARY ===")
    print(f"Total: {len(rows)}")
    print("Predicted scam types:", dict(scam_dist.most_common()))
    print("Predicted tactics:", dict(tac_dist.most_common()))
    print(f"Saved -> {args.output}")

if __name__ == "__main__":
    main()
