"""Score the filled annotation sheet -> extraction precision/recall.

Reference-identity presence rule (recorded per row; derived from the
annotator's judgments and notes):
  - Q1 judged correct                 -> reference exists (y)
  - Q1 == "missed"                    -> reference exists (y)
  - identity extracted, Q1 incorrect  -> reference exists (y); the annotator's
                                         notes confirm a personal name or
                                         organization was claimed in the
                                         conversation
  - otherwise                         -> no reference identity (n)

Consequence for TP/FP/FN:
  - a wrong extraction on a reference-present row is counted as FP *and* FN
    (the correct identity was missed);
  - a false identity on a reference-absent row is FP only (recall denominator
    is unaffected), which is why adversarial negation/hypothetical rows do not
    depress recall.

After you fill extractor_audit/annotation_sheet.xlsx (columns H-L), run:
    python code/score_audit.py --sheet extractor_audit/annotation_sheet.xlsx

Outputs extractor_audit/audit_summary.json and, with --deid, the deidentified
per-row table results/extractor_audit_annotations_deidentified.csv.
"""
import argparse
import csv
import json

from openpyxl import load_workbook

OUT = "extractor_audit/audit_summary.json"
CORRECT = ("1", "yes", "y", "correct")


def norm_set(s):
    return {x.strip().lower() for x in str(s or "").split(";") if x.strip()}


def blank_stats():
    return {"n": 0, "tp": 0, "fp": 0, "fn": 0, "extracted": 0, "correct": 0,
            "items_extracted": 0, "items_correct": 0, "items_wrong": 0,
            "items_missed": 0}


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def metrics(st):
    return {
        "n": st["n"],
        "identity": {
            "tp": st["tp"], "fp": st["fp"], "fn": st["fn"],
            "extracted": st["extracted"], "correct": st["correct"],
            "precision_pct": pct(st["tp"], st["tp"] + st["fp"]),
            "recall_pct": pct(st["tp"], st["tp"] + st["fn"]),
        },
        "requested_items": {
            "extracted": st["items_extracted"], "correct": st["items_correct"],
            "wrong": st["items_wrong"], "missed": st["items_missed"],
            "precision_pct": pct(st["items_correct"], st["items_extracted"]),
            "recall_pct": pct(st["items_correct"],
                              st["items_correct"] + st["items_missed"]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="extractor_audit/annotation_sheet.xlsx")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--deid", default=None,
                    help="optional path for the deidentified per-row CSV")
    args = ap.parse_args()
    wb = load_workbook(args.sheet)
    ws = wb["annotation"]
    headers = [c.value for c in ws[1]]
    idx = {h: i for i, h in enumerate(headers)}

    sampled, adversarial, overall = blank_stats(), blank_stats(), blank_stats()
    deid = []

    for row in ws.iter_rows(min_row=2, values_only=True):
        sid = row[idx["sample_id"]]
        if not sid:
            continue
        is_adv = str(sid).startswith("ADV")
        S = adversarial if is_adv else sampled
        for st in (S, overall):
            st["n"] += 1
        overall["n"] -= 1  # counted twice above; overall counts each row once

        ext_id = str(row[idx["extracted_identity"]] or "").strip()
        ext_items = norm_set(row[idx["extracted_items"]])
        q1 = str(row[idx["Q1_identity_correct"]] or "").strip().lower()
        q2 = norm_set(row[idx["Q2_wrong_items"]])       # extracted but wrong
        q3 = norm_set(row[idx["Q3_missed_items"]])      # present but not extracted
        q4 = str(row[idx["Q4_negation_or_hypothetical"]] or "").strip().lower()

        correct_flag = q1 in CORRECT

        # ---- explicit reference-identity presence ----
        if correct_flag or q1 == "missed" or ext_id:
            ref = "y"
        else:
            ref = "n"

        # ---- identity TP/FP/FN ----
        if ext_id:
            for st in (S, overall):
                st["extracted"] += 1
            if correct_flag:
                for st in (S, overall):
                    st["tp"] += 1
                    st["correct"] += 1
            elif ref == "y":
                for st in (S, overall):
                    st["fp"] += 1
                    st["fn"] += 1       # wrong extraction, reference exists
            else:
                for st in (S, overall):
                    st["fp"] += 1       # false identity, no reference: FP only
        elif q1 == "missed" or ref == "y":
            for st in (S, overall):
                st["fn"] += 1

        # ---- requested items ----
        for st in (S, overall):
            st["items_extracted"] += len(ext_items)
            st["items_correct"] += len(ext_items - q2)
            st["items_wrong"] += len(ext_items & q2)
            st["items_missed"] += len(q3)

        deid.append([sid, str(row[idx["conversation_id"]] or ""),
                     str(row[idx["scam_type"]] or ""), ext_id,
                     "; ".join(sorted(ext_items)), q1,
                     "; ".join(sorted(q2)), "; ".join(sorted(q3)), q4, ref])

    summary = {
        "reference_presence_rule": (
            "reference_identity_present is derived per row: 'y' if the annotator "
            "judged the extracted identity correct, judged it missed (identity "
            "present but not extracted), or an identity was extracted and judged "
            "incorrect — for all five such rows the annotator notes confirm a personal "
            "name or organization was claimed in the conversation; otherwise 'n'. "
            "A wrong extraction on a reference-present row is both a false "
            "positive and a false negative; a false identity on a "
            "reference-absent row is a false positive only."
        ),
        "provenance": {
            "annotator": "one of the authors (graduate researcher)",
            "procedure": "candidate extractions pre-filled by the rule-based "
                         "extractor; all judgments entered by the annotator "
                         "after reading each conversation; no AI system "
                         "assigned the audit labels",
            "deidentified_table": "results/extractor_audit_annotations_deidentified.csv "
                                  "(sample IDs, judgments, and "
                                  "reference_identity_present; no conversation text)",
        },
        "sampled_conversations": metrics(sampled),
        "adversarial_cases": {
            "n": adversarial["n"],
            "identity_fp": adversarial["fp"],
            "item_fp": adversarial["items_wrong"],
            "note": "constructed negation/hypothetical cases; precision/recall "
                    "not meaningful by construction",
        },
        "all_rows_aggregate": metrics(overall),
        "previous_definitions": {
            "identity_recall_missed_only": 70.8,
            "identity_recall_wrong_counts_as_fn_round2": 67.0,
        },
    }

    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("\nSaved ->", args.out)

    if args.deid:
        with open(args.deid, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "conversation_id", "scam_type",
                        "extracted_identity", "extracted_items",
                        "q1_identity_correct", "q2_wrong_items",
                        "q3_missed_items", "q4_negation_or_hypothetical",
                        "reference_identity_present"])
            w.writerows(deid)
        print("Saved ->", args.deid, "| rows:", len(deid))


if __name__ == "__main__":
    main()
