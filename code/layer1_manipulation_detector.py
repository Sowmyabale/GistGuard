"""
Layer 1: Zero-Shot Manipulation Detector
========================================
Hybrid approach combining:
  - Option 2: Semantic similarity with pre-trained encoder
  - Option 3: Linguistic deception markers (Hancock et al., 2007)

NO training on scam data. NO LLM prompting.
"""

import re
import numpy as np
from typing import Dict, List
from collections import Counter

_sentence_encoder = None
_encoder_available = None

def get_encoder():
    global _sentence_encoder, _encoder_available
    if _encoder_available is None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"  [Layer 1] Sentence encoder using device: {device}")
            _sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            _encoder_available = True
        except ImportError:
            _encoder_available = False
    if not _encoder_available:
        return None
    return _sentence_encoder

class ManipulationDetector:
    def __init__(self, tactic_db: Dict[str, List[str]], markers: Dict[str, List[str]]):
        self.tactic_db = tactic_db
        self.markers = markers
        self.tactic_names = list(tactic_db.keys())
        self._example_embeddings = None
        self._example_tactics = []

    def _build_embeddings(self):
        if self._example_embeddings is not None:
            return
        encoder = get_encoder()
        if encoder is None:
            return
        examples, tactics = [], []
        for tactic_name, phrases in self.tactic_db.items():
            for phrase in phrases:
                examples.append(phrase)
                tactics.append(tactic_name)
        self._example_embeddings = encoder.encode(examples, convert_to_numpy=True)
        self._example_tactics = tactics

    def semantic_scores(self, text: str) -> Dict[str, float]:
        encoder = get_encoder()
        if encoder is None:
            text_lower = text.lower()
            tactic_scores = {}
            for tactic, examples in self.tactic_db.items():
                max_overlap = 0
                for ex in examples:
                    ex_words = set(ex.lower().split())
                    text_words = set(text_lower.split())
                    overlap = len(ex_words & text_words) / max(len(ex_words), 1)
                    max_overlap = max(max_overlap, overlap)
                tactic_scores[tactic] = max_overlap
            max_score = max(tactic_scores.values()) if tactic_scores else 0.0
            if max_score >= 0.3:
                tactic_scores = {k: v / max_score for k, v in tactic_scores.items()}
            else:
                tactic_scores = {k: 0.0 for k in tactic_scores}
            return tactic_scores

        self._build_embeddings()
        text_vec = encoder.encode([text], convert_to_numpy=True)
        norms = np.linalg.norm(self._example_embeddings, axis=1) * np.linalg.norm(text_vec, axis=1)
        similarities = np.dot(self._example_embeddings, text_vec.T).flatten() / (norms + 1e-8)
        tactic_scores = {}
        for tactic, sim in zip(self._example_tactics, similarities):
            tactic_scores[tactic] = max(tactic_scores.get(tactic, 0.0), float(sim))
        max_score = max(tactic_scores.values()) if tactic_scores else 1.0
        if max_score > 0:
            tactic_scores = {k: v / max_score for k, v in tactic_scores.items()}
        return tactic_scores

    def linguistic_scores(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        tokens = re.findall(r"\b\w+\b", text_lower)
        token_count = len(tokens) if tokens else 1
        marker_counts = {}
        for marker_name, words in self.markers.items():
            marker_counts[marker_name] = sum(1 for w in words if w in text_lower)

        reciprocity_keywords = ["won", "winner", "prize", "reward", "free", "gift",
                                "congratulations", "selected", "lucky", "bonus"]
        has_reciprocity_kw = any(w in text_lower for w in reciprocity_keywords)

        tactic_linguistic = {}
        tactic_linguistic["urgency_pressure"] = (
            marker_counts.get("urgency_verbs", 0) * 2.0 +
            marker_counts.get("certainty_words", 0) * 1.0 +
            marker_counts.get("negative_emotion", 0) * 0.5
        ) / max(token_count * 0.1, 1.0)

        tactic_linguistic["authority_impersonation"] = (
            marker_counts.get("other_pronouns", 0) * 0.5 +
            marker_counts.get("certainty_words", 0) * 1.0 +
            (1.0 if any(w in text_lower for w in ["irs", "government", "official", "social security", "federal", "agency"]) else 0.0) * 3.0 +
            (1.0 if "bank" in text_lower else 0.0) * 1.0
        ) / max(token_count * 0.1, 1.0)

        tactic_linguistic["fear_threat"] = (
            marker_counts.get("negative_emotion", 0) * 2.0 +
            marker_counts.get("certainty_words", 0) * 0.5
        ) / max(token_count * 0.1, 1.0)

        tactic_linguistic["social_proof"] = (
            marker_counts.get("other_pronouns", 0) * 1.0 +
            marker_counts.get("certainty_words", 0) * 0.5
        ) / max(token_count * 0.1, 1.0)

        tactic_linguistic["reciprocity_gift"] = (
            marker_counts.get("certainty_words", 0) * 1.5 +
            (2.0 if has_reciprocity_kw else 0.0)
        ) / max(token_count * 0.1, 1.0)

        tactic_linguistic["request_sensitive_info"] = (
            marker_counts.get("urgency_verbs", 0) * 1.0 +
            (1.0 if any(w in text_lower for w in ["ssn", "social security", "card", "password", "wire", "transfer", "bitcoin"]) else 0.0) * 3.0
        ) / max(token_count * 0.1, 1.0)

        tactic_linguistic["isolation_secrecy"] = (
            1.0 if any(w in text_lower for w in ["dont tell", "secret", "confidential", "no one"]) else 0.0
        ) * 2.0

        tactic_linguistic["over_familiarity"] = (
            1.0 if any(w in text_lower for w in ["grandma", "grandpa", "grandson", "granddaughter", "its me", "i need your help"]) else 0.0
        ) * 3.0

        max_score = max(tactic_linguistic.values()) if tactic_linguistic else 0.0
        if max_score >= 0.5:
            tactic_linguistic = {k: min(v / max_score, 1.0) for k, v in tactic_linguistic.items()}
        return tactic_linguistic

    def detect(self, text: str, semantic_weight: float = 0.6) -> Dict:
        text_lower = text.lower()
        has_scam_keyword = any(
            kw in text_lower
            for kw in ["irs", "bank", "warrant", "arrest", "ssn", "password", "wire",
                       "won", "winner", "prize", "lottery", "grandma", "grandpa", "bail",
                       "suspended", "fraud", "debt", "owe", "urgent", "immediately",
                       "secret", "dont tell", "free gift", "claim your", "inheritance",
                       "verify your", "account locked", "tax refund", "overdue"]
        )

        has_significant_overlap = False
        for examples in self.tactic_db.values():
            for ex in examples:
                ex_words = set(ex.lower().split())
                text_words = set(text_lower.split())
                overlap = len(ex_words & text_words) / max(len(ex_words), 1)
                if overlap >= 0.7:
                    has_significant_overlap = True
                    break
            if has_significant_overlap:
                break

        sem_scores = self.semantic_scores(text)
        ling_scores = self.linguistic_scores(text)

        combined = {}
        for tactic in set(sem_scores.keys()) | set(ling_scores.keys()):
            s = sem_scores.get(tactic, 0.0)
            l = ling_scores.get(tactic, 0.0)
            combined[tactic] = semantic_weight * s + (1 - semantic_weight) * l

        sorted_tactics = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        primary_tactic = sorted_tactics[0][0] if sorted_tactics else "none"
        confidence = sorted_tactics[0][1] if sorted_tactics else 0.0

        threshold = 0.30 if has_scam_keyword else 0.70
        detected = confidence >= threshold

        if not has_scam_keyword and not has_significant_overlap:
            detected = False
            confidence = 0.0
            primary_tactic = "none"

        # Post-hoc corrections
        strong_reciprocity = any(w in text_lower for w in [
            "won", "winner", "prize", "lottery", "congratulations",
            "lucky winner", "selected as a winner", "selected as the winner"
        ])
        strong_authority = any(w in text_lower for w in [
            "irs", "government", "federal", "agency", "official",
            "social security", "department of", "law enforcement"
        ])
        if strong_reciprocity and not strong_authority and primary_tactic in ["authority_impersonation", "request_sensitive_info", "urgency_pressure"]:
            reciprocity_score = combined.get("reciprocity_gift", 0.0)
            if reciprocity_score >= 0.5:
                primary_tactic = "reciprocity_gift"
                confidence = reciprocity_score

        if strong_authority and primary_tactic == "urgency_pressure":
            authority_score = combined.get("authority_impersonation", 0.0)
            if authority_score >= 0.7:
                primary_tactic = "authority_impersonation"
                confidence = authority_score

        return {
            "detected": detected,
            "primary_tactic": primary_tactic,
            "confidence": round(confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in sorted_tactics},
            "details": {
                "semantic": {k: round(v, 4) for k, v in sem_scores.items()},
                "linguistic": {k: round(v, 4) for k, v in ling_scores.items()},
            },
        }

    def detect_conversation(self, turns: List[Dict]) -> Dict:
        all_scores = []
        scammer_texts = []
        for turn in turns:
            # COVA-X uses 'role' + 'content'; other sources may use 'speaker' + 'text'
            speaker = (turn.get("speaker") or turn.get("role") or "").lower()
            if "scammer" in speaker or "attacker" in speaker:
                text = turn.get("text") or turn.get("content") or ""
                scammer_texts.append(text)
                all_scores.append(self.detect(text))

        if not all_scores:
            return {"detected": False, "primary_tactic": "none", "confidence": 0.0, "all_scores": {}}

        max_conf = max(r["confidence"] for r in all_scores)
        tactic_counts = Counter(r["primary_tactic"] for r in all_scores if r["detected"])
        primary_tactic = tactic_counts.most_common(1)[0][0] if tactic_counts else "none"

        merged_scores = {}
        for r in all_scores:
            for tactic, score in r["all_scores"].items():
                merged_scores[tactic] = max(merged_scores.get(tactic, 0.0), score)

        any_detected = any(r["detected"] for r in all_scores)
        detected = any_detected and max_conf >= 0.30

        return {
            "detected": detected,
            "primary_tactic": primary_tactic,
            "confidence": round(max_conf, 4),
            "all_scores": {k: round(v, 4) for k, v in sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)},
            "num_scammer_turns": len(scammer_texts),
            "scammer_text_combined": " ".join(scammer_texts),
        }

