"""
eval_gist_quality.py — Layer 3 quality metrics for the paper (Doc 2 §6.1).

For each gist variant file, per scam type AND overall:
    qc_valid        % rows passing format QC (is_valid)
    ask_coverage    % warnings mentioning the requested item from the evidence card
    id_coverage     % warnings referencing the claimed identity
    unique          % unique warnings (diversity)
    distinct2       distinct bigram ratio (lexical diversity, 0-1)
    self_sim        mean pairwise bigram Jaccard between warnings of a scam type
    fk_grade        mean Flesch-Kincaid grade (readability; lower = easier)

Usage:
    python eval_gist_quality.py \
        --dataset_dir "/path/to/COVA-X_Split" \
        --files results_v1rerun/test_gists_v3.json results_v1rerun/test_gists_lora.json results_v1rerun/test_gists_lora_v4.json \
        --labels v3_prompted v3_lora v4_lora \
        --output results_v1rerun/gist_quality_summary.json
"""

import os
import re
import json
import glob
import argparse
from collections import defaultdict
from itertools import combinations

from generate_gists_v3 import is_valid
from extract_evidence import evidence_card_fields, ask_coverage, identity_coverage

# ----------------------------- readability ----------------------------------
def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(n, 1)

def fk_grade(text: str) -> float:
    sents = max(len(re.findall(r"[.!?]+", text)), 1)
    words = re.findall(r"[a-zA-Z']+", text)
    nw = max(len(words), 1)
    nsyl = sum(_syllables(w) for w in words)
    return 0.39 * (nw / sents) + 11.8 * (nsyl / nw) - 15.59

# ----------------------------- diversity ------------------------------------
def bigrams(text: str):
    ws = re.findall(r"[a-z0-9']+", text.lower())
    return set(zip(ws, ws[1:])) if len(ws) > 1 else set()

def distinct_n(texts, n=2):
    all_n, uniq = 0, set()
    for t in texts:
        ws = re.findall(r"[a-z0-9']+", t.lower())
        for i in range(len(ws) - n + 1):
            all_n += 1
            uniq.add(tuple(ws[i:i + n]))
    return len(uniq) / max(all_n, 1)

def mean_self_similarity(texts, sample=40):
    import random
    random.seed(42)
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return 0.0
    if len(texts) > sample:
        texts = random.sample(texts, sample)
    bgs = [bigrams(t) for t in texts]
    sims = []
    for a, b in combinations(bgs, 2):
        union = a | b
        sims.append(len(a & b) / max(len(union), 1))
    return sum(sims) / max(len(sims), 1)

# ----------------------------- main ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(f) for f in args.files]
    assert len(labels) == len(args.files)

    # load conversation texts for evidence computation
    convs = {}
    split_dir = os.path.join(args.dataset_dir, args.split)
    for fpath in glob.glob(os.path.join(split_dir, "*", "*.json")):
        with open(fpath, encoding="utf-8") as f:
            conv = json.load(f)
        parts = [f"{t.get('role','?')}: {t.get('content','')}" for t in conv.get("turns", [])]
        convs[conv.get("conversation_id", os.path.basename(fpath))] = (
            "\n".join(parts), conv.get("scam_type", "unknown"))

    summary = {}
    print(f"\n{'variant':16s} {'scam':26s} {'n':>5} {'qc%':>6} {'ask%':>6} {'id%':>6} "
          f"{'uniq%':>6} {'dist2':>6} {'selfsim':>7} {'fk':>5}")
    print("-" * 100)

    for label, fpath in zip(labels, args.files):
        data = json.load(open(fpath, encoding="utf-8"))
        by = defaultdict(list)
        for d in data:
            scam = d.get("true_scam") or d.get("scam_type") or "?"
            by[scam].append(d)

        rows = []
        for scam in sorted(by):
            ds = by[scam]
            warnings = [d.get("warning") or d.get("llama_gist") or "" for d in ds]
            actions = [d.get("action") or d.get("llama_action") or "" for d in ds]

            n = len(ds)
            qc = sum(is_valid(w, a) for w, a in zip(warnings, actions)) / n
            ask = idc = 0
            for d, w in zip(ds, warnings):
                text, st = convs.get(d.get("conversation_id"), ("", scam))
                fields = evidence_card_fields(text, st)
                ask += int(ask_coverage(w, fields["requested_items"]))
                idc += int(identity_coverage(w, fields["claimed_identity"]))
            row = {
                "scam": scam, "n": n,
                "qc_valid": round(qc * 100, 1),
                "ask_coverage": round(ask / n * 100, 1),
                "id_coverage": round(idc / n * 100, 1),
                "unique": round(len(set(warnings)) / n * 100, 1),
                "distinct2": round(distinct_n(warnings, 2), 3),
                "self_sim": round(mean_self_similarity(warnings), 3),
                "fk_grade": round(sum(fk_grade(w) for w in warnings) / n, 1),
            }
            rows.append(row)
            print(f"{label:16s} {scam:26s} {n:5d} {row['qc_valid']:6.1f} "
                  f"{row['ask_coverage']:6.1f} {row['id_coverage']:6.1f} "
                  f"{row['unique']:6.1f} {row['distinct2']:6.3f} "
                  f"{row['self_sim']:7.3f} {row['fk_grade']:5.1f}")

        # overall
        allw = [d.get("warning") or d.get("llama_gist") or "" for d in data]
        alla = [d.get("action") or d.get("llama_action") or "" for d in data]
        n = len(data)
        overall = {
            "n": n,
            "qc_valid": round(sum(is_valid(w, a) for w, a in zip(allw, alla)) / n * 100, 1),
            "unique": round(len(set(allw)) / n * 100, 1),
            "distinct2": round(distinct_n(allw, 2), 3),
            "self_sim": round(mean_self_similarity(allw), 3),
            "fk_grade": round(sum(fk_grade(w) for w in allw) / n, 1),
        }
        print(f"{label:16s} {'OVERALL':26s} {n:5d} {overall['qc_valid']:6.1f} "
              f"{'—':>6} {'—':>6} {overall['unique']:6.1f} {overall['distinct2']:6.3f} "
              f"{overall['self_sim']:7.3f} {overall['fk_grade']:5.1f}")
        print("-" * 100)
        summary[label] = {"per_scam": rows, "overall": overall}

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved -> {args.output}")

if __name__ == "__main__":
    main()
