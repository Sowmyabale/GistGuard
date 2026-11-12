# -*- coding: utf-8 -*-
"""Paired bootstrap CIs for the GistGuard vs equal-evidence Prompt-LM comparison.

Replicates the paper's conditional coverage computation exactly (same functions as
code/eval_gist_quality.py): fields recomputed from conversation text, ask coverage on
conversations with an extracted requested item, identity coverage on conversations with an
extracted claimed identity. Paired bootstrap resamples conversation IDs with the same set for
both systems, 10,000 replicates, 95% percentile CI.
"""
import glob
import json
import os
import sys

import numpy as np

BASE = r"C:\Users\Sowmya\Desktop\Research in Aging\paper1_gxai_elderly"
sys.path.insert(0, os.path.join(BASE, "GistGuard", "code"))
from extract_evidence import evidence_card_fields, ask_coverage, identity_coverage

DS_DIR = os.path.join(r"C:\Users\Sowmya\Desktop\Research in Aging",
                      "dataset", "COVA-X Dataset", "COVA-X_Dataset")
OUT = os.path.join(BASE, "results", "results_v1rerun", "paired_bootstrap_ci.json")
RNG_SEED = 20261112
N_BOOT = 10000

# ---- load conversation texts by id ----
convs = {}
for fpath in glob.glob(os.path.join(DS_DIR, "*", "*.json")):
    with open(fpath, encoding="utf-8") as f:
        conv = json.load(f)
    cid = conv.get("conversation_id", os.path.basename(fpath))
    parts = [f"{t.get('role','?')}: {t.get('content','')}" for t in conv.get("turns", [])]
    convs[cid] = ("\n".join(parts), conv.get("scam_type", "unknown"))
print("conversations loaded:", len(convs))

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

gist = load(os.path.join(BASE, "results", "results_v1rerun", "test_gists_lora_v4b.json"))
eq = load(os.path.join(BASE, "results", "results_v1rerun", "test_gists_v3_equal.json"))
assert len(gist) == len(eq) == 2197
eq_by = {d["conversation_id"]: d for d in eq}

# ---- per-conversation paired indicators ----
ask_g, ask_e, id_g, id_e = [], [], [], []
n_missing = 0
for d in gist:
    cid = d["conversation_id"]
    e = eq_by.get(cid)
    text, st = convs.get(cid, (None, None))
    if e is None or text is None:
        n_missing += 1
        continue
    fields = evidence_card_fields(text, st)
    if fields["requested_items"]:
        ask_g.append(int(ask_coverage(d["warning"], fields["requested_items"])))
        ask_e.append(int(ask_coverage(e["warning"], fields["requested_items"])))
    if fields["claimed_identity"]:
        id_g.append(int(identity_coverage(d["warning"], fields["claimed_identity"])))
        id_e.append(int(identity_coverage(e["warning"], fields["claimed_identity"])))
assert n_missing == 0, n_missing
ask_g, ask_e = np.array(ask_g), np.array(ask_e)
id_g, id_e = np.array(id_g), np.array(id_e)

print(f"ask pairs n={len(ask_g)}  coverage GistGuard={ask_g.mean()*100:.1f} "
      f"Prompt-LM={ask_e.mean()*100:.1f}  diff={(ask_g.mean()-ask_e.mean())*100:.1f}")
print(f"id  pairs n={len(id_g)}  coverage GistGuard={id_g.mean()*100:.1f} "
      f"Prompt-LM={id_e.mean()*100:.1f}  diff={(id_g.mean()-id_e.mean())*100:.1f}")

# ---- paired bootstrap over conversation IDs ----
rng = np.random.default_rng(RNG_SEED)

def boot(a, b):
    n = len(a)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    da = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return np.percentile(da, [2.5, 50, 97.5]) * 100

ask_lo, ask_med, ask_hi = boot(ask_g, ask_e)
id_lo, id_med, id_hi = boot(id_g, id_e)
print(f"ask diff 95% CI: [{ask_lo:.1f}, {ask_hi:.1f}] (median {ask_med:.1f})")
print(f"id  diff 95% CI: [{id_lo:.1f}, {id_hi:.1f}] (median {id_med:.1f})")

summary = {
    "comparison": "GistGuard (SFT-v2s, supplied profile, greedy) vs Prompt-LM with "
                  "equal full-conversation evidence card (greedy), COVA-X test fold",
    "paired": True,
    "resampling": "paired bootstrap over conversation IDs; same resampled set for both "
                  "systems; 10,000 replicates; 95% percentile CI",
    "seed": RNG_SEED,
    "systems": {
        "gistguard": {"file": "results/results_v1rerun/test_gists_lora_v4b.json"},
        "prompt_lm_equal_evidence": {"file": "results/results_v1rerun/test_gists_v3_equal.json"},
    },
    "request_coverage": {
        "n": int(len(ask_g)),
        "gistguard_pct": round(float(ask_g.mean() * 100), 1),
        "prompt_lm_pct": round(float(ask_e.mean() * 100), 1),
        "difference_pp": round(float((ask_g.mean() - ask_e.mean()) * 100), 1),
        "ci95_pp": [round(float(ask_lo), 1), round(float(ask_hi), 1)],
        "ci_median_pp": round(float(ask_med), 1),
    },
    "identity_coverage": {
        "n": int(len(id_g)),
        "gistguard_pct": round(float(id_g.mean() * 100), 1),
        "prompt_lm_pct": round(float(id_e.mean() * 100), 1),
        "difference_pp": round(float((id_g.mean() - id_e.mean()) * 100), 1),
        "ci95_pp": [round(float(id_lo), 1), round(float(id_hi), 1)],
        "ci_median_pp": round(float(id_med), 1),
    },
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("saved ->", OUT)
