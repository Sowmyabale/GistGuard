"""Template baseline (reviewer recommendation): deterministic warning from the
evidence card only -- no LLM. Scored with the same metrics as all other systems.
"""
import json, sys, glob, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_gist_quality import ask_coverage, identity_coverage, is_valid, fk_grade
from extract_evidence import evidence_card_fields

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_dir", required=True,
                help="Folder with <scam_type>/*.json conversations")
ap.add_argument("--predictions", required=True, help="test_predictions.jsonl")
ap.add_argument("--output", required=True, help="Where to write template gists JSON")
args = ap.parse_args()

ASK_PLAIN = {
    "card number / full card": "card number",
    "CVV / security code": "security code",
    "SSN / Social Security number": "Social Security number",
    "one-time passcode / OTP": "one-time passcode",
    "gift card / prepaid card": "gift card",
    "wire transfer / payment": "payment",
    "bank account / routing number": "bank account number",
    "password / PIN": "password",
    "personal details (DOB, address)": "personal details",
    "remote access / screen share": "access to your computer",
    "Medicare / insurance ID": "Medicare ID",
    "cryptocurrency / wallet": "crypto payment",
    "checking account access": "bank account access",
}

convs = {}
for fp in glob.glob(os.path.join(args.dataset_dir, "*", "*.json")):
    conv = json.load(open(fp, encoding="utf-8"))
    parts = [f"{t.get('role','?')}: {t.get('content','')}" for t in conv.get("turns", [])]
    convs[conv.get("conversation_id")] = ("\n".join(parts), conv.get("scam_type", "unknown"))

preds = [json.loads(l) for l in open(args.predictions, encoding="utf-8")]
rows = []
for rec in preds:
    cid = rec["conversation_id"]
    text, st = convs.get(cid, ("", rec.get("true_scam", "bank")))
    fields = evidence_card_fields(text, st)
    identity = fields.get("claimed_identity") or ""
    items = fields.get("requested_items") or []
    item_txt = " and ".join(ASK_PLAIN.get(i, i.split(" /")[0]) for i in items[:2]) if items else "personal information"
    if identity:
        warning = f"This is not the real {identity}. They are asking for your {item_txt} to steal your money."
    else:
        warning = f"This caller is a scammer. They are asking for your {item_txt} to steal your money."
    action = "Hang up and call the number on the back of your card."
    rows.append({"conversation_id": cid, "true_scam": rec.get("true_scam"),
                 "pred_scam": rec.get("pred_scam"), "warning": warning,
                 "action": action, "valid": is_valid(warning, action),
                 "variant": "template"})

json.dump(rows, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

n = len(rows)
ask = idc = ask_den = id_den = 0
for r in rows:
    text, st = convs[r["conversation_id"]]
    fields = evidence_card_fields(text, st)
    ri, ci = fields["requested_items"], fields["claimed_identity"]
    if ri: ask_den += 1; ask += int(ask_coverage(r["warning"], ri))
    if ci: id_den += 1; idc += int(identity_coverage(r["warning"], ci))
print("n=%d qc=%.1f%% ask=%.1f%% (n=%d) id=%.1f%% (n=%d) fk=%.1f uniq=%.1f%% words=%.1f" % (
    n, 100*sum(r["valid"] for r in rows)/n, 100*ask/ask_den, ask_den, 100*idc/id_den, id_den,
    sum(fk_grade(r["warning"]) for r in rows)/n,
    100*len({r["warning"] for r in rows})/n,
    sum(len(r["warning"].split()) for r in rows)/n))
for r in rows[:3]:
    print("---"); print(r["warning"]); print("->", r["action"])
