"""
GXAI-Elderly v2 Framework Configuration
Shared constants, tactic database, and utilities.
"""

import os

# Paths (relative to this file)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "COVA-X Dataset", "COVA-X_Dataset")
YOUTUBE_DIR = os.path.join(PROJECT_ROOT, "dataset", "Youtube transcripts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "dataset", "COVA-X Dataset", "COVA-X_Split")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# EXPERIMENT CONSTANTS (Layer 1 / evaluation — index order matches the trained
# checkpoint's output heads; recovered from saved predictions and run logs)
# ============================================================================

SEED = 42
MODEL_NAME = "roberta-base"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
TACTIC_LOSS_WEIGHT = 0.5

# Layer 1 scam-type head (8 classes — order matches saved scam_probs keys)
SCAM_TYPES = [
    "bank",
    "government_impersonation",
    "grandparent",
    "investment",
    "lottery",
    "medicare",
    "romance",
    "virtual_kidnapping",
]
LABEL_MAP = {name: i for i, name in enumerate(SCAM_TYPES)}
ID2LABEL = {i: name for i, name in enumerate(SCAM_TYPES)}

# Layer 1 tactic head (6 classes — order matches saved tactic_probs keys)
TACTICS = [
    "authority_impersonation",
    "fear_threat",
    "isolation_secrecy",
    "over_familiarity",
    "reciprocity_gift",
    "social_proof",
]
TACTIC_MAP = {name: i for i, name in enumerate(TACTICS)}
ID2TACTIC = {i: name for i, name in enumerate(TACTICS)}

# Category-level tactic prior: dominant tactic for each scam category.
# NOTE: this is a category-derived label, NOT an independently annotated
# per-conversation tactic (see tactic_annotations.jsonl for LLM annotations).
SCAM_TYPE_TO_TACTIC = {
    "bank": "authority_impersonation",
    "government_impersonation": "authority_impersonation",
    "medicare": "authority_impersonation",
    "grandparent": "over_familiarity",
    "romance": "over_familiarity",
    "virtual_kidnapping": "fear_threat",
    "lottery": "reciprocity_gift",
    "investment": "social_proof",
}

# Layer 2 victim-profile label maps (raw victim_config value -> 0/1/2).
# 0 = low/skeptical/unaware, 1 = middle, 2 = high/trusting/aware.
TRUST_MAP = {
    "highly_skeptical": 0,
    "skeptical": 0,
    "cautious": 0,
    "somewhat_trusting": 1,
    "moderately_trusting": 1,
    "very_trusting": 2,
    "highly_trusting": 2,
}
TECH_MAP = {
    "very_low": 0,
    "low": 0,
    "basic": 1,
    "moderate": 1,
    "high": 2,
}
AWARENESS_MAP = {
    "unaware": 0,
    "vaguely_aware": 1,
    "heard_about_them": 1,
    "informed": 2,
    "highly_aware": 2,
    "highly_informed": 2,
}
TRUST_NAMES = ["skeptical", "somewhat trusting", "very trusting"]
TECH_NAMES = ["low comfort", "moderate comfort", "high comfort"]
AWARENESS_NAMES = ["unaware", "somewhat aware", "aware"]

# ============================================================================
# MANIPULATION TACTIC DATABASE for Semantic Similarity (Layer 1)
# Based on: Cialdini (2001), Hancock et al. (2007), Adkins et al. (2025)
# ============================================================================

MANIPULATION_TACTICS = {
    "urgency_pressure": [
        "act now",
        "immediate action required",
        "urgent",
        "today only",
        "expires soon",
        "deadline",
        "limited time",
        "dont wait",
        "right now",
        "hurry",
        "time sensitive",
        "last chance",
        "act fast",
        "within 24 hours",
        "before its too late",
    ],
    "authority_impersonation": [
        "this is the irs",
        "i am calling from your bank",
        "social security administration",
        "federal government",
        "official notice",
        "law enforcement",
        "we are authorized",
        "on behalf of",
        "department of",
        "agency",
        "government office",
        "your account has been flagged",
        "warrant for your arrest",
        "legal action",
    ],
    "fear_threat": [
        "your account will be suspended",
        "arrest warrant",
        "lawsuit",
        "penalty",
        "fines",
        "blocked",
        "suspended",
        "terminated",
        "unpaid taxes",
        "you owe",
        "criminal charges",
        "fraud detected",
        "unusual activity",
        "security breach",
    ],
    "social_proof": [
        "many people have already",
        "thousands of satisfied customers",
        "everyone is doing it",
        "your neighbor",
        "others have benefited",
        "join the winners",
        "dont miss out like others did",
        "people just like you",
    ],
    "reciprocity_gift": [
        "you have won",
        "free gift",
        "congratulations",
        "prize",
        "reward",
        "bonus",
        "exclusive offer",
        "special discount for you",
        "you are selected",
        "lucky winner",
        "no cost to you",
    ],
    "request_sensitive_info": [
        "confirm your social security number",
        "verify your bank account",
        "credit card number",
        "cvv",
        "password",
        "pin",
        "routing number",
        "wire transfer",
        "gift card",
        "bitcoin",
        "cryptocurrency",
        "send money",
        "western union",
        "moneygram",
    ],
    "isolation_secrecy": [
        "dont tell anyone",
        "this is confidential",
        "keep this between us",
        "no one else can know",
        "dont discuss with family",
        "secret",
        "private matter",
    ],
    "over_familiarity": [
        "grandma its me",
        "grandpa",
        "its your grandson",
        "i am in trouble",
        "i need your help",
        "you know i love you",
        "remember when",
        "i am hurt",
        "i was arrested",
        "bail money",
    ],
}

# Flatten all tactic examples for semantic similarity
ALL_TACTIC_EXAMPLES = []
for tactic, examples in MANIPULATION_TACTICS.items():
    for ex in examples:
        ALL_TACTIC_EXAMPLES.append((tactic, ex))

# ============================================================================
# LINGUISTIC DECEPTION MARKERS (Layer 1 - Option 3)
# Based on: Hancock et al. (2007), Adkins et al. (2025)
# ============================================================================

DECEPTION_MARKERS = {
    "first_person_pronouns": ["i", "me", "my", "mine", "myself", "we", "us", "our"],
    "other_pronouns": ["you", "your", "yours", "he", "she", "they", "them", "their"],
    "certainty_words": [
        "definitely", "absolutely", "certainly", "guaranteed", "surely",
        "without a doubt", "promise", "trust me", "i assure you", "100%",
    ],
    "urgency_verbs": [
        "act", "call", "click", "send", "pay", "transfer", "verify",
        "confirm", "update", "secure", "register", "claim",
    ],
    "negative_emotion": [
        "fear", "worry", "trouble", "problem", "danger", "threat",
        "risk", "lose", "suspended", "blocked", "arrest", "penalty",
    ],
    "cognitive_words": [
        "think", "know", "because", "reason", "understand", "consider",
        "evaluate", "analyze", "perhaps", "maybe", "possibly",
    ],
    "modal_verbs": [
        "may", "might", "could", "would", "should", "can", "will",
    ],
}

# ============================================================================
# ToM INFERENCE ATTRIBUTES (Layer 2)
# ============================================================================

SCAMMER_TOM_ATTRIBUTES = [
    "intent",
    "deception_strategy",
    "false_belief",
    "urgency_level",
]

VICTIM_TOM_ATTRIBUTES = [
    "susceptibility",
    "truth_bias",
    "emotional_state",
    "cognitive_load",
    "attention_level",
]

GIST_COMPLEXITY_LEVELS = {
    "simple": {
        "max_words": 30,
        "use_action_items": True,
        "avoid_technical": True,
        "format": "1-sentence warning + 1 action",
    },
    "moderate": {
        "max_words": 60,
        "use_action_items": True,
        "avoid_technical": False,
        "format": "2-3 sentence explanation + action items",
    },
    "detailed": {
        "max_words": 100,
        "use_action_items": True,
        "avoid_technical": False,
        "format": "brief explanation + why + action items",
    },
}

# ============================================================================
# SIGN-UP ATTRIBUTE MAPPING (Layer 2 input)
# ============================================================================

SIGNUP_TO_TOM_MAPPING = {
    "age_group": {
        "field": "age",
        "mapping": {
            "under_60": {"base_susceptibility": 0.3, "tom_decline": "none"},
            "60-69": {"base_susceptibility": 0.5, "tom_decline": "mild"},
            "70-79": {"base_susceptibility": 0.7, "tom_decline": "moderate"},
            "80+": {"base_susceptibility": 0.85, "tom_decline": "severe"},
        },
    },
    "living_situation": {
        "field": "living",
        "mapping": {
            "alone": {"isolation_risk": 0.8, "social_buffer": 0.2},
            "spouse": {"isolation_risk": 0.4, "social_buffer": 0.6},
            "family": {"isolation_risk": 0.2, "social_buffer": 0.8},
            "community": {"isolation_risk": 0.3, "social_buffer": 0.7},
        },
    },
    "tech_comfort": {
        "field": "tech_comfort",
        "mapping": "scale_1_5",
    },
    "prior_scam": {
        "field": "scam_awareness",
        "mapping": {
            "lost": {"awareness": 0.9, "truth_bias": 0.3},
            "stopped": {"awareness": 0.7, "truth_bias": 0.5},
            "unsure": {"awareness": 0.4, "truth_bias": 0.7},
            "no": {"awareness": 0.2, "truth_bias": 0.8},
        },
    },
    "trust_tendency": {
        "field": "trust_tendency",
        "mapping": "scale_1_5",
    },
    "urgency_response": {
        "field": "urgency_response",
        "mapping": "scale_1_5",
    },
    "financial_control": {
        "field": "financial_control",
        "mapping": {
            "self": {"autonomy": 1.0, "family_involved": False},
            "mostly_self": {"autonomy": 0.7, "family_involved": True},
            "family": {"autonomy": 0.3, "family_involved": True},
            "advisor": {"autonomy": 0.5, "family_involved": False},
        },
    },
}
