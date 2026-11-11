"""Score the filled annotation sheet -> extraction precision/recall (#8).

After you fill extractor_audit/annotation_sheet.xlsx (columns H-L), run:
    python extractor_audit/score_audit.py

Outputs extractor_audit/audit_summary.json and prints the report.
"""
import argparse
import json
import sys

from openpyxl import load_workbook

OUT = "extractor_audit/audit_summary.json"

def norm_set(s):
    return {x.strip().lower() for x in str(s or "").split(";") if x.strip()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="extractor_audit/annotation_sheet.xlsx")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    wb = load_workbook(args.sheet)
    ws = wb["annotation"]
    headers = [c.value for c in ws[1]]
    idx = {h: i for i, h in enumerate(headers)}

    n = id_present = id_correct = id_missed = 0
    items_extracted = items_correct = items_wrong = items_missed = 0
    adversarial = {"n": 0, "identity_fp": 0, "item_fp": 0}
    negation_cases = 0

    for row in ws.iter_rows(min_row=2, values_only=True):
        sid = row[idx["sample_id"]]
        if not sid:
            continue
        n += 1
        is_adv = str(sid).startswith("ADV")
        ext_id = str(row[idx["extracted_identity"]] or "").strip()
        ext_items = norm_set(row[idx["extracted_items"]])
        q1 = str(row[idx["Q1_identity_correct"]] or "").strip().lower()
        q2 = norm_set(row[idx["Q2_wrong_items"]])       # extracted but wrong
        q3 = norm_set(row[idx["Q3_missed_items"]])       # present but not extracted
        q4 = str(row[idx["Q4_negation_or_hypothetical"]] or "").strip().lower()

        if q4 in ("y", "yes", "1"):
            negation_cases += 1

        if is_adv:
            adversarial["n"] += 1

        # ---- identity ----
        # A wrong extracted identity also means the correct identity was
        # missed, so recall counts two kinds of false negatives:
        #   (a) q1 == "missed"  (extractor found nothing, identity present)
        #   (b) an extracted identity judged not correct on a non-adversarial
        #       row (the correct identity was overlooked)
        # Adversarial rows are excluded from (b): many have no true identity
        # to miss, and their false positives are tallied separately above.
        if ext_id:
            id_present += 1
            if q1 in ("1", "yes", "y", "correct"):
                id_correct += 1
            elif is_adv:
                adversarial["identity_fp"] += 1
            else:
                id_missed += 1
        elif q1 == "missed":
            id_missed += 1

        # ---- requested items ----
        items_extracted += len(ext_items)
        items_correct += len(ext_items - q2)
        items_wrong += len(ext_items & q2)
        items_missed += len(q3)
        if is_adv and q2:
            adversarial["item_fp"] += len(q2)

    def pct(a, b):
        return round(100 * a / b, 1) if b else None

    summary = {
        "n_annotated": n,
        "n_sampled_conversations": n - adversarial["n"],
        "n_adversarial_cases": adversarial["n"],
        "provenance": {
            "annotator": "one of the authors (graduate researcher)",
            "procedure": "candidate extractions pre-filled by the rule-based extractor; "
                         "all judgments entered by the annotator after reading each conversation; "
                         "no AI system assigned the audit labels",
            "deidentified_table": "results/extractor_audit_annotations_deidentified.csv "
                                  "(sample IDs and judgments only, no conversation text)",
        },
        "recall_definition": "identity recall = correct / (correct + missed), where 'missed' "
                             "counts both identities the extractor found nothing for (q1=missed) "
                             "and identities it extracted incorrectly on non-adversarial rows "
                             "(a wrong extraction also omits the correct identity). Under the "
                             "previous definition (q1=missed only) identity recall was 70.8%.",
        "negation_or_hypothetical_flagged": negation_cases,
        "identity": {
            "extracted": id_present,
            "correct": id_correct,
            "precision_pct": pct(id_correct, id_present),
            "missed_by_extractor": id_missed,
            "recall_pct": pct(id_correct, id_correct + id_missed),
        },
        "requested_items": {
            "extracted": items_extracted,
            "correct": items_correct,
            "wrong": items_wrong,
            "precision_pct": pct(items_correct, items_extracted),
            "missed": items_missed,
            "recall_pct": pct(items_correct, items_correct + items_missed),
        },
        "adversarial": adversarial,
    }

    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("\nSaved ->", args.out)

if __name__ == "__main__":
    main()
