"""Template baseline (reviewer recommendation): deterministic warning from the
evidence card only -- no LLM. Scored with the same metrics as all other systems.
"""
import json, sys, glob, os, re, argparse
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
    "account number / routing": "bank account number",
    "SSN": "Social Security number",
    "PIN / password / OTP": "password or one-time code",
    "Medicare / insurance ID": "Medicare ID",
    "personal information": "personal details",
    "gift cards": "gift cards",
    "wire / money transfer": "money transfer",
    "crypto": "crypto payment",
    "upfront fee / taxes": "upfront fee",
}

# Deterministic per-scam-type actions (verified-safe advice) plus the cue
# patterns used for the action-appropriateness metric.
ACTIONS = {
    "bank": ("Hang up and call the number on the back of your bank card.",
             [r"back of your (bank )?card", r"bank('s)? (official )?number", r"bank directly"]),
    "government_impersonation": ("Hang up. Real government agencies never demand payment "
                                 "by phone — verify on the official government website.",
                                 [r"official .*website", r"\.gov\b", r"government (agency )?directly"]),
    "medicare": ("Hang up and call Medicare at 1-800-MEDICARE to check.",
                 [r"1-800-medicare", r"medicare directly"]),
    "grandparent": ("Hang up and call your grandchild on their usual number to check "
                    "they are really safe.",
                    [r"call your grand(child|son|daughter)", r"usual number", r"really safe"]),
    "romance": ("Stop sending money and talk to someone you trust before you respond again.",
                [r"someone you trust", r"stop sending money"]),
    "investment": ("Hang up and check the offer with your bank or a licensed advisor "
                   "before paying anything.",
                   [r"licensed (financial )?advisor", r"your bank", r"check the offer"]),
    "lottery": ("Hang up — real prizes never ask for payment or fees to claim them.",
                [r"never ask for (payment|fees)", r"real prizes"]),
    "virtual_kidnapping": ("Hang up and call your family member directly on their own "
                           "number right now.",
                           [r"call your family member directly", r"own number"]),
}
_DEFAULT_ACTION = ("Hang up now, and verify any payment request with a family member "
                   "or your bank before acting.",
                   [r"hang up", r"family member|your bank"])

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
    action, cues = ACTIONS.get(rec.get("true_scam") or st, _DEFAULT_ACTION)
    appropriate = any(re.search(c, action.lower()) for c in cues)
    rows.append({"conversation_id": cid, "true_scam": rec.get("true_scam"),
                 "pred_scam": rec.get("pred_scam"), "warning": warning,
                 "action": action, "valid": is_valid(warning, action),
                 "action_appropriate": appropriate,
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
print("n=%d qc=%.1f%% ask=%.1f%% (n=%d) id=%.1f%% (n=%d) fk=%.1f uniq=%.1f%% words=%.1f action_ok=%.1f%%" % (
    n, 100*sum(r["valid"] for r in rows)/n, 100*ask/ask_den, ask_den, 100*idc/id_den, id_den,
    sum(fk_grade(r["warning"]) for r in rows)/n,
    100*len({r["warning"] for r in rows})/n,
    sum(len(r["warning"].split()) for r in rows)/n,
    100*sum(r["action_appropriate"] for r in rows)/n))
for r in rows[:3]:
    print("---"); print(r["warning"]); print("->", r["action"])
