"""Compute overall conditional ask/id coverage per variant, exactly matching
eval_gist_quality.py's per-scam methodology, plus per-variant numbers."""
import json, sys, glob, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_gist_quality import ask_coverage, identity_coverage
from extract_evidence import evidence_card_fields

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_dir", required=True,
                help="Folder with <scam_type>/*.json conversations")
ap.add_argument("--results_dir", required=True,
                help="Folder containing the variant gist files")
ap.add_argument("--output", default=None, help="Where to write the summary JSON")
args = ap.parse_args()

convs = {}
for fpath in glob.glob(os.path.join(args.dataset_dir, "*", "*.json")):
    with open(fpath, encoding="utf-8") as f:
        conv = json.load(f)
    parts = [f"{t.get('role','?')}: {t.get('content','')}" for t in conv.get("turns", [])]
    convs[conv.get("conversation_id", os.path.basename(fpath))] = ("\n".join(parts), conv.get("scam_type", "unknown"))

files = {
 "v3_prompted": "test_gists_v3.json",
 "SFT-v1": "test_gists_lora.json",
 "SFT-v2": "test_gists_lora_v4.json",
 "SFT-v2s": "test_gists_lora_v4b.json",
 "smishx": "test_gists_smishx_fixed.json",
 "ghafariasl": "test_gists_ghafariasl.json",
 "template": "test_gists_template.json",
}
import os as _os
files = {k: v for k, v in files.items()
         if _os.path.exists(_os.path.join(args.results_dir, v))}
out = {}
for name, f in files.items():
    data = json.load(open(os.path.join(args.results_dir, f), encoding="utf-8"))
    ask = idc = ask_den = id_den = 0
    for d in data:
        text, st = convs.get(d.get("conversation_id"), ("", d.get("true_scam", "?")))
        fields = evidence_card_fields(text, st)
        w = d.get("warning") or ""
        ri, ci = fields["requested_items"], fields["claimed_identity"]
        if ri:  # conditional denominator
            ask_den += 1
            ask += int(ask_coverage(w, ri))
        if ci:
            id_den += 1
            idc += int(identity_coverage(w, ci))
    out[name] = {"ask_pct": round(100*ask/max(ask_den,1),1), "ask_den": ask_den,
                 "id_pct": round(100*idc/max(id_den,1),1), "id_den": id_den, "n": len(data)}
    print("%-12s ask %5.1f%% (n=%4d)   id %5.1f%% (n=%4d)" % (name, out[name]["ask_pct"], ask_den, out[name]["id_pct"], id_den))
out_path = args.output or os.path.join(args.results_dir, "coverage_overall.json")
json.dump(out, open(out_path, "w"), indent=2)
print("saved ->", out_path)
