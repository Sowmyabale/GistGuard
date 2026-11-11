"""Build item-level human-eval ratings CSV from the filled scoring booklet.

For each of the 50 items, the booklet shows three warning variants (A/B/C).
The three systems evaluated are Prompt-LM (untuned few-shot), SFT-v1, and
GistGuard (SFT-v2s) -- verified by exhaustive text-similarity matching of
all 150 booklet variant cells against every candidate output file (no
Template variants are present in the booklet). Variant->system assignment is
done PER ITEM by maximising total text similarity over the 3! = 6
permutations (booklet variant texts vs. system outputs for the same
conversation id).
"""
import argparse
import json
import re
import csv
import itertools
from difflib import SequenceMatcher
from pathlib import Path

from docx import Document

SYSTEM_FILES = {
    "Prompt-LM": "test_gists_v3.json",       # untuned, few-shot prompted
    "SFT-v1": "test_gists_lora.json",        # LoRA on silver_train
    "GistGuard": "test_gists_lora_v4b.json",  # LoRA on silver_v2_short (SFT-v2s)
}


def norm(t: str) -> str:
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def warning_part(cell: str) -> str:
    """Extract the WARNING text from a booklet variant cell."""
    m = re.search(r"WARNING:\s*(.*?)(?:ACTION:|$)", cell, re.S | re.I)
    if m:
        return m.group(1)
    return cell


def load_system_warnings(res_dir):
    out = {}
    for name, fn in SYSTEM_FILES.items():
        data = json.load(open(res_dir / fn, encoding="utf-8"))
        if isinstance(data, dict):
            data = list(data.values())
        conv = {}
        for r in data:
            cid = r.get("conv_id") or r.get("conversation_id") or r.get("id")
            w = r.get("warning") or r.get("gist") or ""
            conv[cid] = w
        out[name] = conv
    return out


def parse_booklet(booklet_path):
    doc = Document(str(booklet_path))
    tables = doc.tables

    # conversation ids and scam types live in paragraphs, paired with item order
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    cids = {}
    scam_by_item = {}
    for p in paras:
        m = re.match(r"Item (\d+) of 50 — (\w+)", p)
        if m:
            scam_by_item[int(m.group(1))] = m.group(2)
        m = re.search(r"Conversation ID:\s*(\S+)", p)
        if m:
            cids[len(cids) + 1] = m.group(1)

    items = []
    n = (len(tables) - 1) // 5
    for i in range(n):
        cid = cids.get(i + 1)
        variants = []
        for j in (1, 2, 3):
            cell = tables[2 + 5 * i + (j - 1)].rows[0].cells[0].text
            variants.append(cell)
        rating_rows = []
        rt = tables[5 + 5 * i]
        for row in rt.rows:
            cells = [c.text.strip() for c in row.cells]
            rating_rows.append(cells)
        items.append({
            "item": i + 1,
            "conv_id": cid,
            "scam": scam_by_item.get(i + 1),
            "variant_cells": variants,
            "rating_table": rating_rows,
        })
    return items


def parse_ratings(rating_rows):
    """Return list of 3 dicts (A, B, C) with rating fields."""
    per_variant = [dict() for _ in range(3)]

    def put(label, values):
        for k in range(3):
            v = values[k] if k < len(values) else ""
            per_variant[k][label] = v

    for row in rating_rows:
        if not row:
            continue
        label = row[0]
        values = row[1:4] if len(row) >= 4 else row[1:]
        if label.startswith("Accuracy"):
            put("accuracy", values)
        elif label.startswith("Clarity"):
            put("clarity", values)
        elif label.startswith("Actionability"):
            put("actionability", values)
        elif label.startswith("Empathy"):
            put("empathy", values)
        elif label.startswith("Groundedness"):
            put("groundedness", values)
        elif label.startswith("Would this warning make you stop"):
            put("stop_verify", values)
        elif label.startswith("Does it contain anything untrue"):
            put("untrue", values)
        elif label.startswith("Would you send this to your own"):
            put("send_grandparent", values)
        elif label.startswith("Overall preference"):
            put("overall_preference", values)
    return per_variant


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--booklet", required=True, help="filled scoring booklet .docx")
    ap.add_argument("--repo_dir", required=True, help="GistGuard repo root")
    args = ap.parse_args()
    repo = Path(args.repo_dir)
    res_dir = repo / "results" / "results_v1rerun"

    sys_warn = load_system_warnings(res_dir)
    items = parse_booklet(args.booklet)
    rows = []
    unmatched = []
    min_sim_seen = 1.0

    for it in items:
        cid = it["conv_id"]
        if cid is None:
            unmatched.append((it["item"], "no-conv-id"))
            continue
        booklet_warn = [warning_part(c) for c in it["variant_cells"]]
        sys_names = list(SYSTEM_FILES.keys())
        sys_texts = [sys_warn[s].get(cid, "") for s in sys_names]
        if any(not t for t in sys_texts):
            missing = [s for s, t in zip(sys_names, sys_texts) if not t]
            unmatched.append((it["item"], f"missing-sys:{missing}"))
            continue

        best_perm, best_score = None, -1.0
        for perm in itertools.permutations(range(3)):
            score = sum(sim(booklet_warn[p], sys_texts[j]) for j, p in enumerate(perm))
            if score > best_score:
                best_score, best_perm = score, perm
        permsims = [sim(booklet_warn[p], sys_texts[j]) for j, p in enumerate(best_perm)]
        min_sim_seen = min(min_sim_seen, *permsims)

        ratings = parse_ratings(it["rating_table"])
        for j, sname in enumerate(sys_names):
            r = ratings[best_perm[j]]
            rows.append({
                "item": it["item"],
                "conversation_id": cid,
                "scam": it.get("scam") or "",
                "system": sname,
                "accuracy": r.get("accuracy", ""),
                "clarity": r.get("clarity", ""),
                "actionability": r.get("actionability", ""),
                "empathy": r.get("empathy", ""),
                "groundedness": r.get("groundedness", ""),
                "stop_verify": r.get("stop_verify", ""),
                "untrue_or_invented": r.get("untrue", ""),
                "send_to_grandparent": r.get("send_grandparent", ""),
                "match_sim": round(permsims[j], 3),
            })

    out_csv = repo / "results" / "human_eval_ratings_item_level.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # aggregate
    agg = {}
    for r in rows:
        a = agg.setdefault(r["system"], {k: [] for k in
                          ("accuracy", "clarity", "actionability", "empathy", "groundedness")})
        for k in a:
            try:
                a[k].append(float(r[k]))
            except (ValueError, TypeError):
                pass
    print(f"items parsed: {len(items)}, rows written: {len(rows)}")
    print(f"unmatched: {unmatched}")
    print(f"min match similarity: {min_sim_seen:.3f}")
    for s, a in agg.items():
        means = {k: round(sum(v) / len(v), 2) for k, v in a.items()}
        overall = round(sum(means.values()) / len(means), 2)
        print(f"{s:10s} n={len(next(iter(a.values())))} {means} overall={overall}")

    ys = lambda col, s: sum(1 for r in rows if r["system"] == s and r[col].upper().startswith("Y"))
    ns = lambda col, s: sum(1 for r in rows if r["system"] == s and r[col].upper().startswith("N"))
    for s in SYSTEM_FILES:
        print(f"{s:10s} stop Y/N: {ys('stop_verify', s)}/{ns('stop_verify', s)}  "
              f"untrue Y/N: {ys('untrue_or_invented', s)}/{ns('untrue_or_invented', s)}  "
              f"send Y/N: {ys('send_to_grandparent', s)}/{ns('send_to_grandparent', s)}")


if __name__ == "__main__":
    main()
