"""
extract_evidence.py — Conversation Evidence Extractor (Layer 3 grounding, Doc 2 §3.1).

Rule-based + keyword/regex. No model, CPU-only, fully auditable. For every
conversation it extracts an EVIDENCE CARD:

    claimed_identity  : who the caller claims to be (per scam-type vocab)
    requested_items   : what the caller asks the victim for (card, CVV, SSN, ...)
    payment_channels  : how money should move (gift cards, wire, crypto, ...)
    pressure          : deadline / threat / secrecy phrases found

The card is injected into silver-v2 generation AND into the LoRA v4 input, so
groundedness does not depend on sampling luck.

Usage:
    # write cards for every split (CPU, minutes)
    python extract_evidence.py \
        --dataset_dir "/path/to/COVA-X_Split" \
        --splits train val test \
        --output results_v1rerun/evidence_cards.jsonl

    # or import:  from extract_evidence import evidence_card, load_cards
"""

import os
import re
import json
import glob
import argparse

# ---------------------------------------------------------------------------
# Vocabulary (extend freely; matching is case-insensitive substring/regex)
# ---------------------------------------------------------------------------

IDENTITY_PATTERNS = {
    "bank": [
        r"\b(chase|wells fargo|bank of america|citibank|citi bank|usbank|u\.s\. bank|"
        r"hsbc|capital one|pnc|td bank|fraud department|security department|"
        r"bank(ing)? (representative|officer|agent|fraud team))\b",
    ],
    "government_impersonation": [
        r"\b(irs|internal revenue service|social security administration|\bssa\b|"
        r"treasury( department)?|fbi|dea|department of (justice|homeland security)|"
        r"immigration|customs|marshal|police department|sheriff|courthouse|jury duty)\b",
    ],
    "medicare": [
        r"\b(medicare|medicaid|health ?care|health insurance|insurance (company|provider|agent)|"
        r"aetna|united ?health|humana|blue cross|anthem|cigna)\b",
    ],
    "grandparent": [
        r"\b(your )?(grandson|granddaughter|grandchild)\b|\bit'?s me\b|\bthis is your (grandson|granddaughter)\b",
    ],
    "romance": [
        r"\b(soldier|sergeant|general|army|navy|air force|deployed|overseas|"
        r"oil rig|engineer|doctor|surgeon|widow(er)?)\b",
    ],
    "investment": [
        r"\b(broker|financial advisor|adviser|portfolio manager|trading platform|"
        r"crypto( currency)? (exchange|trader|expert)|hedge fund|investment firm)\b",
    ],
    "lottery": [
        r"\b(lottery (office|commission|headquarters)|publisher'?s clearing house|\bpch\b|"
        r"prize (department|patrol|center)|sweepstakes (office|headquarters)|"
        r"claims? (department|agent))\b",
    ],
    "virtual_kidnapping": [
        r"\b(kidnapper|we have your (daughter|son|wife|husband|grandchild)|"
        r"your family (member )?is (in danger|kidnapped)|don't call the police)\b",
    ],
}

REQUESTED_ITEM_PATTERNS = [
    ("card number / full card", r"\b(full )?(credit |debit )?card (number|details)\b|\b16[- ]?digit\b"),
    ("CVV / security code", r"\bcvv\b|\bcvc\b|\bsecurity code\b|\b3[- ]?digit (code|number)\b|\bcode on the (back|front)\b"),
    ("account number / routing", r"\b(account|routing|checking) (number|details)\b|\bbank account\b"),
    ("SSN", r"\bsocial security (number)?\b|\bssn\b|\bSS ?number\b"),
    ("PIN / password / OTP", r"\b(bank )?(pin|password|passcode)\b|\bone[- ]?time (code|password|passcode)\b|\botp\b|\bverification code\b"),
    ("Medicare / insurance ID", r"\bmedicare (number|id|card)\b|\b(member|policy|group|insurance) (id|number)\b"),
    ("personal information", r"\b(date of birth|birth ?date|home address|mother'?s maiden name)\b"),
    ("gift cards", r"\bgift ?cards?\b|\bitunes\b|\bsteam (card|wallet)\b|\bgoogle play\b|\bamazon (card|gift)\b|\bvanilla (visa )?card\b"),
    ("wire / money transfer", r"\bwire transfer\b|\bwestern union\b|\bmoney ?gram\b|\bzelle\b|\bvenmo\b|\bcash ?app\b|\bpaypal\b|\bbank transfer\b"),
    ("crypto", r"\bbitcoin\b|\bcrypto(currency)?\b|\busdt\b|\bethereum\b|\bbtc\b"),
    ("upfront fee / taxes", r"\b(upfront|processing|administrative|handling) fee\b|\btaxes on (the )?(prize|winnings)\b|\bfee to (claim|release|unlock)\b"),
]

PAYMENT_CHANNEL_PATTERNS = [
    ("gift cards", r"\bgift ?cards?\b|\bitunes\b|\bsteam\b|\bgoogle play\b"),
    ("wire transfer", r"\bwire transfer\b|\bwestern union\b|\bmoney ?gram\b"),
    ("crypto", r"\bbitcoin\b|\bcrypto\b|\busdt\b|\bethereum\b"),
    ("cash deposit", r"\bcash (deposit|drop)\b|\batm\b deposit"),
]

PRESSURE_PATTERNS = [
    ("deadline", r"\b(within|in|before) (the next )?\d+ (minutes?|hours?|days?)\b|\bby (today|tonight|tomorrow)\b|\bright now\b|\bimmediately\b|\bact (fast|now|quickly)\b|\burgent(ly)?\b"),
    ("threat", r"\b(arrest(ed)?|warrant|lawsuit|sued|fines?|penalty|deport(ed)?|account (will be )?(closed|frozen|suspended)|charges? (will be )?filed|lose your (money|benefits))\b"),
    ("secrecy", r"\b(don'?t|do not) tell (anyone|your|the)|\bkeep (this|it) (a )?secret\b|\bstay on the (line|phone)\b|\bdon'?t hang up\b"),
]

_NONE = "none"

def _find_all(pattern_groups, text, per_scam_key=None):
    """pattern_groups: dict key->[patterns] OR list of (label, pattern)."""
    t = text.lower()
    found = []
    if isinstance(pattern_groups, dict):
        pats = pattern_groups.get(per_scam_key, [])
        for p in pats:
            m = re.search(p, t)
            if m:
                found.append(m.group(0))
    else:
        for label, p in pattern_groups:
            if re.search(p, t):
                found.append(label)
    return found

def evidence_card_fields(conv_text: str, scam_type: str):
    """-> dict of evidence fields from the raw conversation text."""
    claimed = _find_all(IDENTITY_PATTERNS, conv_text, scam_type)
    items = _find_all(REQUESTED_ITEM_PATTERNS, conv_text)
    channels = _find_all(PAYMENT_CHANNEL_PATTERNS, conv_text)
    t = conv_text.lower()
    pressure = [label for label, p in PRESSURE_PATTERNS if re.search(p, t)]
    return {
        "claimed_identity": claimed[0] if claimed else "",
        "requested_items": items,
        "payment_channels": channels,
        "pressure": pressure,
    }

def format_card(fields: dict) -> str:
    lines = []
    ci = fields.get("claimed_identity") or "not stated"
    lines.append(f"- Caller claims to be: {ci}")
    ri = fields.get("requested_items") or ["nothing specific found"]
    lines.append(f"- Caller asks for: {', '.join(ri)}")
    pc = fields.get("payment_channels") or ["none mentioned"]
    lines.append(f"- Payment/transfer method: {', '.join(pc)}")
    pr = fields.get("pressure") or ["no strong pressure phrase found"]
    lines.append(f"- Pressure: {', '.join(pr)}")
    return "\n".join(lines)

def evidence_card(conv_text: str, scam_type: str) -> str:
    return format_card(evidence_card_fields(conv_text, scam_type))

# ---- item-synonym lists for QC v2 ask-coverage (word-level, lowercase) ----
ASK_SYNONYMS = {
    "card number / full card": ["card number", "card details", "your card"],
    "CVV / security code": ["cvv", "security code", "code on the back", "code on your card"],
    "account number / routing": ["account number", "routing number", "bank account"],
    "SSN": ["social security", "ssn"],
    "PIN / password / OTP": ["pin", "password", "verification code", "one-time code", "passcode"],
    "Medicare / insurance ID": ["medicare number", "medicare id", "member id", "policy number", "insurance number"],
    "personal information": ["personal information", "date of birth", "home address"],
    "gift cards": ["gift card", "itunes", "google play", "steam"],
    "wire / money transfer": ["wire transfer", "western union", "moneygram", "zelle", "venmo", "bank transfer"],
    "crypto": ["bitcoin", "crypto"],
    "upfront fee / taxes": ["fee", "taxes", "upfront"],
}

def _word_pat(phrase: str) -> "re.Pattern":
    """Word-boundary, case-insensitive pattern for a literal phrase."""
    return re.compile(r"\b" + re.escape(phrase.lower()) + r"\b")

def ask_coverage(warning: str, requested_items) -> bool:
    """True if the warning mentions at least one requested item.

    "At least one" (any-coverage) is the headline definition; word-boundary
    matching prevents substring false positives (e.g. "pin" inside
    "shopping"). Returns True vacuously when no item was extracted.
    """
    if not requested_items:
        return True
    w = warning.lower()
    for item in requested_items:
        syns = ASK_SYNONYMS.get(item)
        if syns is None:
            syns = [item.split(" /")[0].strip().lower()]
        for syn in syns:
            if _word_pat(syn).search(w):
                return True
    return False

def ask_coverage_all(warning: str, requested_items) -> bool:
    """Strict variant: the warning must mention EVERY extracted item."""
    if not requested_items:
        return True
    return all(any(_word_pat(syn).search(warning.lower())
                   for syn in ASK_SYNONYMS.get(item, [item.split(" /")[0].strip().lower()]))
               for item in requested_items)

# Known identity mentions -> acceptable surface forms in a warning.
# Short agency names (IRS, SSA, FBI) are preserved explicitly.
_IDENTITY_ALIAS_GROUPS = [
    (r"\b(irs|internal revenue service)\b",
     ["irs", "internal revenue service", "tax agency", "tax department"]),
    (r"\bssa\b|\bsocial security administration\b",
     ["ssa", "social security", "social security administration"]),
    (r"\bfbi\b", ["fbi", "federal bureau of investigation"]),
    (r"\bmedicare\b", ["medicare"]),
    (r"\bmedicaid\b", ["medicaid"]),
    (r"\btreasury\b", ["treasury"]),
    (r"\bdea\b", ["dea"]),
    (r"\b(fraud department|security department|security team)\b",
     ["fraud department", "security department", "security team",
      "bank", "bank security", "bank's fraud team"]),
    (r"\b(police department|sheriff|courthouse|marshal|jury duty)\b",
     ["police", "police department", "sheriff", "law enforcement", "officer",
      "court", "courthouse", "marshal"]),
    (r"\b(immigration|customs)\b", ["immigration", "customs", "ice"]),
]
_BRAND_RE = re.compile(
    r"\b(chase|wells fargo|bank of america|citibank|citi bank|usbank|u\.s\. bank|"
    r"hsbc|capital one|pnc|td bank|aetna|united ?health|humana|blue cross|anthem|cigna|"
    r"publisher'?s clearing house|\bpch\b)\b")
_IDENTITY_GENERIC = [
    "bank", "government", "government agency", "federal agency",
    "medicare", "insurance", "insurance company", "health insurance",
    "grandchild", "grandson", "granddaughter", "grandkid", "family member", "relative",
    "romantic partner", "online partner", "person online", "online relationship",
    "investment advisor", "financial advisor", "adviser", "broker",
    "lottery", "lottery office", "prize office", "sweepstakes", "prize",
    "kidnapper", "trading platform", "investment firm",
]

def identity_coverage(warning: str, claimed_identity: str) -> bool:
    """True if the warning references the claimed identity (or none was found).

    Matching rules:
    - known agencies accept their listed aliases (short names like IRS count);
    - specific brand names (Chase, Wells Fargo, Aetna, ...) require the brand
      itself -- a generic "your bank" does NOT cover "Bank of America";
    - otherwise identity tokens (word-boundary, len>=2) or a scam-generic
      phrase (e.g. "fake romantic partner") count as coverage.
    """
    if not claimed_identity:
        return True
    w = warning.lower()
    ci = claimed_identity.lower().strip()
    for pat, aliases in _IDENTITY_ALIAS_GROUPS:
        if re.search(pat, ci):
            return any(_word_pat(a).search(w) for a in aliases)
    m = _BRAND_RE.search(ci)
    if m:
        return _word_pat(m.group(1)).search(w) is not None
    toks = [t for t in re.split(r"\W+", ci) if len(t) >= 2]
    if any(_word_pat(t).search(w) for t in toks):
        return True
    return any(_word_pat(g).search(w) for g in _IDENTITY_GENERIC)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    n = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for split in args.splits:
            split_dir = os.path.join(args.dataset_dir, split)
            for fpath in sorted(glob.glob(os.path.join(split_dir, "*", "*.json"))):
                with open(fpath, encoding="utf-8") as f:
                    conv = json.load(f)
                parts = [f"{t.get('role','?')}: {t.get('content','')}"
                         for t in conv.get("turns", [])]
                text = "\n".join(parts)
                st = conv.get("scam_type", "unknown")
                fields = evidence_card_fields(text, st)
                row = {"conversation_id": conv.get("conversation_id", os.path.basename(fpath)),
                       "split": split, "scam_type": st, **fields}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} evidence cards -> {args.output}")

    # quick stats
    with_items = with_ci = 0
    for l in open(args.output, encoding="utf-8"):
        r = json.loads(l)
        with_items += bool(r["requested_items"])
        with_ci += bool(r["claimed_identity"])
    print(f"requested_items found: {100*with_items/max(n,1):.1f}% | "
          f"claimed_identity found: {100*with_ci/max(n,1):.1f}%")

if __name__ == "__main__":
    main()
