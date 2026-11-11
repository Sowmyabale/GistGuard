"""
GXAI-Elderly v2: End-to-End Pipeline
=====================================
Connects Layer 1 (Manipulation Detection) → Layer 2 (ToM Inference) → Layer 3 (Gist Generation)

Usage:
    python pipeline.py --input <conversation_json_or_text> --profile <profile_json>
"""

import json
import argparse
import os
import sys
from typing import Dict, Optional

# Add v2_framework to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v2_framework.config import MANIPULATION_TACTICS, DECEPTION_MARKERS
from v2_framework.layer1_manipulation_detector import ManipulationDetector
from v2_framework.layer2_tom_inference import ToMInferenceEngine, SignUpProfile
from v2_framework.layer3_gist_generator import GistGenerator

class GXAIPipeline:
    """
    Full 3-layer pipeline for elderly scam protection.
    """

    def __init__(self, layer3_model: str = "microsoft/DialoGPT-medium"):
        print("Initializing GXAI-Elderly v2 Pipeline...")

        # Layer 1: Zero-shot manipulation detector
        self.layer1 = ManipulationDetector(MANIPULATION_TACTICS, DECEPTION_MARKERS)

        # Layer 2: ToM inference engine
        self.layer2 = ToMInferenceEngine()

        # Layer 3: Gist generator
        self.layer3 = GistGenerator(model_name=layer3_model)

        print("All layers loaded.\n")

    def process_text(self, text: str, profile: SignUpProfile) -> Dict:
        """
        Process a single message/text through all 3 layers.
        """
        # --- Layer 1: Detect manipulation ---
        l1_result = self.layer1.detect(text)

        # --- Layer 2: Infer ToM states ---
        l2_result = self.layer2.infer(l1_result, profile)

        # --- Layer 3: Generate gist ---
        l3_result = self.layer3.generate(l1_result, l2_result, profile.tech_comfort)

        return {
            "input_text": text,
            "layer1_manipulation": l1_result,
            "layer2_tom": l2_result,
            "layer3_gist": l3_result,
        }

    def process_conversation(self, conversation: Dict, profile: SignUpProfile) -> Dict:
        """
        Process a full conversation (multi-turn) through all 3 layers.

        Args:
            conversation: Dict with 'conversation_id', 'turns' list
            profile: SignUpProfile from questionnaire
        """
        turns = conversation.get("turns", [])
        conv_id = conversation.get("conversation_id", "unknown")

        # --- Layer 1: Detect across all scammer turns ---
        l1_result = self.layer1.detect_conversation(turns)

        # --- Extract victim turns for Layer 2 ---
        victim_turns = [
            t for t in turns
            if "victim" in t.get("speaker", "").lower()
        ]

        # --- Layer 2: Infer ToM states ---
        l2_result = self.layer2.infer(l1_result, profile, victim_turns)

        # --- Layer 3: Generate gist ---
        l3_result = self.layer3.generate(l1_result, l2_result, profile.tech_comfort)

        return {
            "conversation_id": conv_id,
            "num_turns": len(turns),
            "layer1_manipulation": l1_result,
            "layer2_tom": l2_result,
            "layer3_gist": l3_result,
            "summary": {
                "scam_detected": l1_result["detected"],
                "primary_tactic": l1_result["primary_tactic"],
                "victim_susceptibility": l2_result["victim_tom"]["susceptibility"],
                "gist": l3_result["gist"],
                "action": l3_result["action_item"],
            },
        }

    def process_json_file(self, json_path: str, profile: SignUpProfile) -> Dict:
        """Load conversation from JSON file and process."""
        with open(json_path, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
        return self.process_conversation(conversation, profile)

def load_profile(profile_path: str) -> SignUpProfile:
    """Load sign-up profile from JSON file."""
    with open(profile_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return SignUpProfile(**data)

def demo():
    """Run demo with sample inputs."""
    pipeline = GXAIPipeline()

    # Default profile
    profile = SignUpProfile(
        age_group="70-79",
        living_situation="alone",
        tech_comfort=2,
        prior_scam="no",
        trust_tendency=4,
        urgency_response=4,
        financial_control="self",
    )

    test_cases = [
        {
            "name": "IRS Scam (Authority + Fear)",
            "text": "This is the IRS calling. You owe $4,500 in unpaid taxes. A warrant for your arrest will be issued within 24 hours if you do not pay immediately. Call us back at this number to settle your debt.",
        },
        {
            "name": "Grandparent Scam (Over-familiarity + Secrecy)",
            "text": "Grandma, it's me. I'm in trouble and I need your help. I was in an accident and I need $3,000 for bail. Please don't tell mom and dad, they'll be so mad. Can you wire me the money right now?",
        },
        {
            "name": "Lottery Scam (Reciprocity)",
            "text": "Congratulations! You have been selected as the winner of a $500,000 lottery prize. To claim your reward, simply call our claims department and provide your bank details for the transfer.",
        },
        {
            "name": "Normal Message (No scam)",
            "text": "Hey, are we still on for lunch tomorrow at noon? Let me know if the time works for you.",
        },
    ]

    print("=" * 70)
    print("GXAI-ELDERLY V2: END-TO-END PIPELINE DEMO")
    print("=" * 70)

    for case in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST: {case['name']}")
        print(f"{'='*70}")
        print(f"Input: {case['text'][:100]}...")

        result = pipeline.process_text(case["text"], profile)

        print(f"\n--- LAYER 1: MANIPULATION DETECTION ---")
        l1 = result["layer1_manipulation"]
        print(f"  Detected: {l1['detected']}")
        print(f"  Primary Tactic: {l1['primary_tactic']}")
        print(f"  Confidence: {l1['confidence']}")

        print(f"\n--- LAYER 2: ToM INFERENCE ---")
        l2 = result["layer2_tom"]
        print(f"  Scammer intent: {l2['scammer_tom']['intent']}")
        print(f"  Scammer strategy: {l2['scammer_tom']['deception_strategy']}")
        print(f"  Victim susceptibility: {l2['victim_tom']['susceptibility']}")
        print(f"  Victim truth bias: {l2['victim_tom']['truth_bias']}")
        print(f"  Victim emotional state: {l2['victim_tom']['emotional_state']}")

        print(f"\n--- LAYER 3: GIST EXPLANATION ---")
        l3 = result["layer3_gist"]
        print(f"  Gist: {l3['gist']}")
        print(f"  Action: {l3['action_item']}")
        print(f"  Complexity: {l3['complexity']} words")

def main():
    parser = argparse.ArgumentParser(description="GXAI-Elderly v2 Pipeline")
    parser.add_argument("--input", type=str, help="Path to conversation JSON file")
    parser.add_argument("--profile", type=str, help="Path to profile JSON file")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--text", type=str, help="Single text message to analyze")
    parser.add_argument("--output", type=str, default="v2_results/pipeline_output.json", help="Output JSON path")
    args = parser.parse_args()

    if args.demo:
        demo()
        return

    pipeline = GXAIPipeline()

    # Load profile
    if args.profile:
        profile = load_profile(args.profile)
    else:
        # Default profile
        profile = SignUpProfile(
            age_group="70-79",
            living_situation="alone",
            tech_comfort=2,
            prior_scam="no",
            trust_tendency=4,
            urgency_response=4,
            financial_control="self",
        )

    # Process input
    if args.text:
        result = pipeline.process_text(args.text, profile)
    elif args.input:
        result = pipeline.process_json_file(args.input, profile)
    else:
        print("Error: Provide --text, --input, or --demo")
        return

    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResult saved to: {args.output}")
    print(json.dumps(result.get("summary", result), indent=2))

if __name__ == "__main__":
    main()
