"""
Layer 2: Theory of Mind (ToM) Inference
========================================
Infers TWO mental states:
  A. SCAMMER'S mental state (from message + Layer 1 tactics)
  B. VICTIM'S mental state (from sign-up data + conversation responses)

Grounded in: Calso et al. (2020), Krendl et al. (2022), Stanley & Blanchard-Fields (2008)
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

def _turn_text(turn: Dict) -> str:
    """COVA-X uses 'content'; other sources may use 'text'."""
    return turn.get("text") or turn.get("content") or ""

@dataclass
class SignUpProfile:
    age_group: str = "60-69"
    living_situation: str = "spouse"
    tech_comfort: int = 3
    prior_scam: str = "no"
    trust_tendency: int = 3
    urgency_response: int = 3
    financial_control: str = "self"
    emergency_contact: bool = False

@dataclass
class ScammerToM:
    intent: str = ""
    deception_strategy: str = ""
    false_belief: str = ""
    urgency_level: str = "medium"
    emotional_manipulation: str = ""

@dataclass
class VictimToM:
    susceptibility: float = 0.5
    truth_bias: float = 0.5
    emotional_state: str = "neutral"
    cognitive_load: str = "medium"
    attention_level: str = "engaged"
    protection_level: str = "moderate"

class ToMInferenceEngine:
    def __init__(self):
        self.age_tom_decline = {
            "under_60": 0.0, "60-69": 0.15, "70-79": 0.35, "80+": 0.55,
        }
        self.isolation_risk = {
            "alone": 0.8, "spouse": 0.4, "family": 0.2, "community": 0.3,
        }
        self.scam_awareness = {
            "lost": 0.9, "stopped": 0.7, "unsure": 0.4, "no": 0.2,
        }
        self.tactic_to_emotion = {
            "urgency_pressure": "fear", "authority_impersonation": "fear",
            "fear_threat": "fear", "social_proof": "greed",
            "reciprocity_gift": "greed", "request_sensitive_info": "fear",
            "isolation_secrecy": "fear", "over_familiarity": "sympathy",
        }
        self.tactic_to_intent = {
            "urgency_pressure": "money", "authority_impersonation": "money",
            "fear_threat": "money", "social_proof": "money",
            "reciprocity_gift": "info", "request_sensitive_info": "info",
            "isolation_secrecy": "money", "over_familiarity": "money",
        }
        self.false_beliefs = {
            "urgency_pressure": "There is a real deadline and consequences for inaction.",
            "authority_impersonation": "The caller is a legitimate authority figure.",
            "fear_threat": "A real threat exists and compliance will prevent harm.",
            "social_proof": "Many others have safely participated and benefited.",
            "reciprocity_gift": "The reward is real and without strings attached.",
            "request_sensitive_info": "Providing information is necessary and safe.",
            "isolation_secrecy": "Secrecy protects the victim from embarrassment or harm.",
            "over_familiarity": "The caller is a loved one in genuine need.",
        }

    def infer_scammer(self, layer1_result: Dict) -> ScammerToM:
        primary_tactic = layer1_result.get("primary_tactic", "none")
        confidence = layer1_result.get("confidence", 0.0)
        all_scores = layer1_result.get("all_scores", {})

        intent = self.tactic_to_intent.get(primary_tactic, "unknown")
        top_tactics = [t for t, s in all_scores.items() if s > 0.5]
        if len(top_tactics) > 1:
            intent = "combination"

        emotion = self.tactic_to_emotion.get(primary_tactic, "unknown")

        if primary_tactic == "urgency_pressure" and confidence > 0.7:
            urgency = "high"
        elif primary_tactic in ["fear_threat", "authority_impersonation"] and confidence > 0.6:
            urgency = "high"
        elif confidence > 0.5:
            urgency = "medium"
        else:
            urgency = "low"

        false_belief = self.false_beliefs.get(primary_tactic, "The message is legitimate and safe to respond to.")

        return ScammerToM(
            intent=intent,
            deception_strategy=primary_tactic,
            false_belief=false_belief,
            urgency_level=urgency,
            emotional_manipulation=emotion,
        )

    def infer_victim(self, profile: SignUpProfile, layer1_result: Dict, victim_turns: Optional[List[Dict]] = None) -> VictimToM:
        age_decline = self.age_tom_decline.get(profile.age_group, 0.15)
        isolation = self.isolation_risk.get(profile.living_situation, 0.4)
        awareness = self.scam_awareness.get(profile.prior_scam, 0.2)
        trust = (profile.trust_tendency - 1) / 4.0
        urgency_response = (profile.urgency_response - 1) / 4.0

        base_susceptibility = (
            0.30 * age_decline +
            0.20 * isolation +
            0.20 * (1.0 - awareness) +
            0.15 * trust +
            0.15 * urgency_response
        )

        primary_tactic = layer1_result.get("primary_tactic", "none")
        tactic_adjustment = self._tactic_susceptibility_adjustment(primary_tactic, profile)
        susceptibility = min(1.0, max(0.0, base_susceptibility + tactic_adjustment))

        truth_bias = min(1.0, (trust * 0.6 + (1.0 - awareness) * 0.4))
        emotional_state = self._infer_emotional_state(primary_tactic, victim_turns)
        cognitive_load = self._infer_cognitive_load(victim_turns)
        attention = self._infer_attention(victim_turns)

        tech = (profile.tech_comfort - 1) / 4.0
        social_buffer = 1.0 - isolation
        protection = (tech * 0.3 + awareness * 0.4 + social_buffer * 0.3)

        if protection > 0.7:
            protection_level = "high"
        elif protection > 0.4:
            protection_level = "moderate"
        else:
            protection_level = "low"

        return VictimToM(
            susceptibility=round(susceptibility, 3),
            truth_bias=round(truth_bias, 3),
            emotional_state=emotional_state,
            cognitive_load=cognitive_load,
            attention_level=attention,
            protection_level=protection_level,
        )

    def _tactic_susceptibility_adjustment(self, tactic: str, profile: SignUpProfile) -> float:
        if tactic == "over_familiarity" and profile.living_situation == "alone":
            return 0.15
        if tactic == "urgency_pressure" and profile.urgency_response >= 4:
            return 0.15
        if tactic == "authority_impersonation" and profile.trust_tendency >= 4:
            return 0.10
        if tactic == "reciprocity_gift" and profile.prior_scam == "no":
            return 0.10
        return 0.0

    def _infer_emotional_state(self, tactic: str, victim_turns: Optional[List[Dict]]) -> str:
        if victim_turns:
            last_turn = _turn_text(victim_turns[-1]).lower()
            if any(w in last_turn for w in ["scared", "worried", "afraid", "what should i do"]):
                return "fear"
            if any(w in last_turn for w in ["ok", "sure", "i will", "how do i"]):
                return "trust"
            if any(w in last_turn for w in ["what", "confused", "dont understand", "repeat"]):
                return "confusion"
            if any(w in last_turn for w in ["excited", "great", "amazing", "wow"]):
                return "excitement"

        tactic_emotion = {
            "fear_threat": "fear", "urgency_pressure": "fear",
            "authority_impersonation": "fear", "reciprocity_gift": "excitement",
            "social_proof": "excitement", "over_familiarity": "trust",
        }
        return tactic_emotion.get(tactic, "neutral")

    def _infer_cognitive_load(self, victim_turns: Optional[List[Dict]]) -> str:
        if not victim_turns:
            return "medium"
        recent_turns = victim_turns[-3:]
        avg_len = sum(len(_turn_text(t)) for t in recent_turns) / max(len(recent_turns), 1)
        num_questions = sum(1 for t in recent_turns if "?" in _turn_text(t))
        if avg_len < 10 and num_questions >= 2:
            return "high"
        if avg_len < 20 or num_questions >= 1:
            return "medium"
        return "low"

    def _infer_attention(self, victim_turns: Optional[List[Dict]]) -> str:
        if not victim_turns:
            return "engaged"
        last = _turn_text(victim_turns[-1]).lower()
        if any(w in last for w in ["hold on", "wait", "busy", "later", "cant talk"]):
            return "distracted"
        return "engaged"

    def infer(self, layer1_result: Dict, profile: SignUpProfile, victim_turns: Optional[List[Dict]] = None) -> Dict:
        scammer = self.infer_scammer(layer1_result)
        victim = self.infer_victim(profile, layer1_result, victim_turns)
        return {
            "scammer_tom": asdict(scammer),
            "victim_tom": asdict(victim),
        }

