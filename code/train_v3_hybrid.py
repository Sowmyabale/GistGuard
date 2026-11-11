"""
train_v3_hybrid.py — V3 HYBRID baseline: semantic similarity + linguistic markers + light ML.

Motivation (Paper 1, ablation table):
    v2 was zero-shot only (LLM reads the conversation, no training signal on OUR data).
    v3 asks: how far can a *lightweight, interpretable* hybrid go?

Features per conversation:
    (a) Sentence-transformer embedding of the full conversation text  (384-dim, MiniLM)
    (b) Tactic-similarity vector: cosine(conv, tactic_description) for each of the
        6 tactics                                             (6-dim, zero-shot signal)
    (c) Linguistic-marker counts: handcrafted lexicons per tactic   (~60-dim)

Models: LogisticRegression (default), RandomForest, MLP  — all CPU-cheap after encoding.
Also reports a pure zero-shot similarity classifier (no training) for reference.

"""

import os
import re
import json
import argparse
import numpy as np
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from config import ID2LABEL, LABEL_MAP, TACTICS, SCAM_TYPE_TO_TACTIC, SEED
from dataset_utils import load_all_splits

# ---------------------------------------------------------------------------
# Linguistic marker lexicons (handcrafted, per tactic — Option 3)
# ---------------------------------------------------------------------------
TACTIC_LEXICONS = {
    "authority_impersonation": [
        "irs", "social security", "medicare", "bank", "federal", "agent", "officer",
        "official", "department", "government", "warrant", "suspended", "account has been",
        "verify your identity", "this is", "calling from", "on behalf of",
    ],
    "fear_threat": [
        "arrest", "warrant", "police", "lawsuit", "legal action", "immediately", "now",
        "urgent", "or else", "consequences", "danger", "emergency", "trouble",
        "you will be", "act now", "last chance", "final notice", "do not ignore",
    ],
    "isolation_secrecy": [
        "don't tell", "do not tell", "keep this between", "secret", "confidential",
        "no one else", "don't share", "privacy", "just between us", "not tell anyone",
        "stay on the line", "don't hang up", "don't contact",
    ],
    "over_familiarity": [
        "grandma", "grandpa", "sweetheart", "dear", "honey", "my love", "baby",
        "i miss you", "i love you", "trust me", "you know me", "it's me", "your favorite",
        "remember when", "how are you", "thinking of you",
    ],
    "reciprocity_gift": [
        "you won", "winner", "congratulations", "prize", "lottery", "jackpot", "free",
        "gift", "reward", "bonus", "claim", "no purchase", "selected", "lucky",
        "giveaway", "cash", "check", "wire transfer", "processing fee",
    ],
    "social_proof": [
        "everyone", "other investors", "other people", "thousands", "many people",
        "our clients", "successful", "doubled", "guaranteed", "returns", "profit",
        "opportunity", "limited spots", "testimonial", "reviews", "join them",
    ],
}

TACTIC_DESCRIPTIONS = {
    "authority_impersonation": "The caller pretends to be an authority figure such as a bank, government agency, or official, demanding compliance.",
    "fear_threat": "The message uses threats, urgency, or fear of consequences to pressure the victim into acting immediately.",
    "isolation_secrecy": "The scammer tells the victim to keep the conversation secret and not contact anyone else.",
    "over_familiarity": "The scammer pretends to be a family member or romantic partner to build false trust and intimacy.",
    "reciprocity_gift": "The scam offers a prize, lottery win, or free gift that requires the victim to pay a fee or share details.",
    "social_proof": "The scam claims others have profited or benefited, using testimonials and group pressure.",
}

WORD_RE = re.compile(r"[^a-z0-9' ]+")

def marker_features(text: str) -> np.ndarray:
    """Counts of lexicon hits per tactic -> 1 feature per lexicon entry."""
    # normalize: keep letters/digits/apostrophes, everything else -> single space
    low = " " + re.sub(r"[^a-z0-9' ]+", " ", text.lower()) + " "
    feats = []
    for tactic in TACTICS:
        for phrase in TACTIC_LEXICONS[tactic]:
            if " " in phrase:
                feats.append(low.count(phrase))
            else:
                feats.append(len(re.findall(rf"\b{re.escape(phrase)}\b", low)))
    return np.array(feats, dtype=np.float32)

# ---------------------------------------------------------------------------
# Embedding + similarity features
# ---------------------------------------------------------------------------
def encode_features(dfs, encoder_name: str, batch_size: int = 64):
    """Returns list of feature matrices (train/val/test) and the encoder."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding with {encoder_name} on {device} ...")
    enc = SentenceTransformer(encoder_name, device=device)

    tactic_emb = enc.encode([TACTIC_DESCRIPTIONS[t] for t in TACTICS],
                            normalize_embeddings=True, show_progress_bar=False)

    out = []
    for name, df in zip(["train", "val", "test"], dfs):
        embs = enc.encode(list(df["text"]), batch_size=batch_size,
                          normalize_embeddings=True, show_progress_bar=True,
                          convert_to_numpy=True)
        sims = embs @ tactic_emb.T                       # (n, 6) zero-shot tactic signal
        markers = np.stack([marker_features(t) for t in df["text"]])  # (n, n_lex)
        feats = np.concatenate([embs, sims, markers], axis=1)
        print(f"  [{name}] features {feats.shape}")
        out.append(feats.astype(np.float32))
    return out

def zero_shot_by_similarity(test_feats, n_emb: int, test_labels):
    """Pure zero-shot: max tactic similarity -> scam type (via dominant-tactic map)."""
    sims = test_feats[:, n_emb:n_emb + len(TACTICS)]
    pred_tactic = np.argmax(sims, axis=1)

    # dominant tactic -> possible scam types (invert SCAM_TYPE_TO_TACTIC)
    tactic_to_scams = {i: [] for i in range(len(TACTICS))}
    for scam, tac in SCAM_TYPE_TO_TACTIC.items():
        tactic_to_scams[TACTICS.index(tac)].append(scam)
    # deterministic tie-break by sorted scam name; unmapped tactics (e.g.
    # isolation_secrecy has no dominant scam) fall back to the full label set
    pred_scam = []
    for pt in pred_tactic:
        scams = sorted(tactic_to_scams.get(int(pt), [])) or sorted(LABEL_MAP.keys())
        pred_scam.append(min(LABEL_MAP[s] for s in scams))
    acc = accuracy_score(test_labels, pred_scam)
    f1 = f1_score(test_labels, pred_scam, average="macro")
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", default="results_v3hybrid")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    np.random.seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    train_df, val_df, test_df = load_all_splits(args.dataset_dir)
    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    feats = encode_features([train_df, val_df, test_df], args.encoder, args.batch_size)
    Xtr, Xva, Xte = feats
    ytr, yva, yte = train_df["label"].values, val_df["label"].values, test_df["label"].values
    n_emb = Xtr.shape[1] - len(TACTICS) - sum(len(v) for v in TACTIC_LEXICONS.values())

    summary = {}

    # ---- pure zero-shot reference (no training at all) ----
    zacc, zf1 = zero_shot_by_similarity(Xte, n_emb, yte)
    print(f"\n[zero-shot similarity] test acc={zacc:.4f}  macro-F1={zf1:.4f}")
    summary["zero_shot_similarity"] = {"test_accuracy": round(zacc, 4),
                                       "test_f1_macro": round(zf1, 4)}

    # ---- hybrid models ----
    models = {
        "hybrid_logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)),
        ]),
        "hybrid_rf": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=SEED, class_weight="balanced"),
        "hybrid_mlp": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", MLPClassifier(hidden_layer_sizes=(256,), max_iter=200,
                                  random_state=SEED, early_stopping=True)),
        ]),
    }

    names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    for mname, model in models.items():
        print(f"\n=== {mname} ===")
        model.fit(Xtr, ytr)
        va_pred = model.predict(Xva)
        te_pred = model.predict(Xte)
        va_acc = accuracy_score(yva, va_pred)
        te_acc = accuracy_score(yte, te_pred)
        te_f1 = f1_score(yte, te_pred, average="macro")
        print(f"val  acc={va_acc:.4f}")
        print(f"test acc={te_acc:.4f}  macro-F1={te_f1:.4f}")
        print(classification_report(yte, te_pred, labels=list(range(len(names))),
                                    target_names=names, digits=4))
        cm = confusion_matrix(yte, te_pred)
        np.save(os.path.join(args.output_dir, f"cm_{mname}.npy"), cm)
        summary[mname] = {"val_accuracy": round(va_acc, 4),
                          "test_accuracy": round(te_acc, 4),
                          "test_f1_macro": round(te_f1, 4)}

    with open(os.path.join(args.output_dir, "v3_hybrid_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {args.output_dir}/v3_hybrid_summary.json")

if __name__ == "__main__":
    main()
