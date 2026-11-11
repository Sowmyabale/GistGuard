"""
GXAI-Elderly v2: End-to-End Pipeline
=====================================
Connects Layer 1 → Layer 2 → Layer 3
"""

import json
import argparse
import os
from typing import Dict, Optional

from config import MANIPULATION_TACTICS, DECEPTION_MARKERS
from layer1_manipulation_detector import ManipulationDetector
from layer2_tom_inference import ToMInferenceEngine, SignUpProfile
from layer3_gist_generator import GistGenerator

class GXAIPipeline:
    def __init__(self, layer3_model: str = "microsoft/DialoGPT-medium"):
        print("Initializing GXAI-Elderly v2 Pipeline...")
        self.layer1 = ManipulationDetector(MANIPULATION_TACTICS, DECEPTION_MARKERS)
        self.layer2 = ToMInferenceEngine()
        self.layer3 = GistGenerator(model_name=layer3_model)
        print("All layers loaded.\n")

    def process_text(self, text: str, profile: SignUpProfile) -> Dict:
        l1_result = self.layer1.detect(text)
        l2_result = self.layer2.infer(l1_result, profile)
        l3_result = self.layer3.generate(l1_result, l2_result, profile.tech_comfort)
        return {
            "input_text": text,
            "layer1_manipulation": l1_result,
            "layer2_tom": l2_result,
            "layer3_gist": l3_result,
        }

    def process_conversation(self, conversation: Dict, profile: SignUpProfile) -> Dict:
        turns = conversation.get("turns", [])
        conv_id = conversation.get("conversation_id", "unknown")

        l1_result = self.layer1.detect_conversation(turns)
        victim_turns = [t for t in turns
                        if "victim" in (t.get("speaker") or t.get("role") or "").lower()]
        l2_result = self.layer2.infer(l1_result, profile, victim_turns)
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
        with open(json_path, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
        return self.process_conversation(conversation, profile)

def load_profile(profile_path: str) -> SignUpProfile:
    with open(profile_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return SignUpProfile(**data)

def demo():
    pipeline = GXAIPipeline()
    profile = SignUpProfile(
        age_group="70-79", living_situation="alone", tech_comfort=2,
        prior_scam="no", trust_tendency=4, urgency_response=4, financial_control="self",
    )
    test_cases = [
        {"name": "IRS Scam", "text": "This is the IRS calling. You owe $4,500 in unpaid taxes. A warrant for your arrest will be issued within 24 hours if you do not pay immediately. Call us back at this number to settle your debt."},
        {"name": "Grandparent Scam", "text": "Grandma, it's me. I'm in trouble and I need your help. I was in an accident and I need $3,000 for bail. Please don't tell mom and dad, they'll be so mad. Can you wire me the money right now?"},
        {"name": "Lottery Scam", "text": "Congratulations! You have been selected as the winner of a $500,000 lottery prize. To claim your reward, simply call our claims department and provide your bank details for the transfer."},
        {"name": "Normal Message", "text": "Hey, are we still on for lunch tomorrow at noon? Let me know if the time works for you."},
    ]
    print("=" * 70)
    print("GXAI-ELDERLY V2: END-TO-END PIPELINE DEMO")
    print("=" * 70)
    for case in test_cases:
        print(f"\n{'='*70}\nTEST: {case['name']}\n{'='*70}")
        result = pipeline.process_text(case["text"], profile)
        print(f"Detected: {result['layer1_manipulation']['detected']} | Tactic: {result['layer1_manipulation']['primary_tactic']} | Conf: {result['layer1_manipulation']['confidence']}")
        print(f"Victim susceptibility: {result['layer2_tom']['victim_tom']['susceptibility']}")
        print(f"Gist: {result['layer3_gist']['gist']}")
        print(f"Action: {result['layer3_gist']['action_item']}")

def main():
    parser = argparse.ArgumentParser(description="GXAI-Elderly v2 Pipeline")
    parser.add_argument("--input", type=str, help="Path to conversation JSON file")
    parser.add_argument("--profile", type=str, help="Path to profile JSON file")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--text", type=str, help="Single text message to analyze")
    parser.add_argument("--output", type=str, default="results/pipeline_output.json", help="Output JSON path")
    args = parser.parse_args()

    if args.demo:
        demo()
        return

    pipeline = GXAIPipeline()
    if args.profile:
        profile = load_profile(args.profile)
    else:
        profile = SignUpProfile(age_group="70-79", living_situation="alone", tech_comfort=2,
                                prior_scam="no", trust_tendency=4, urgency_response=4, financial_control="self")

    if args.text:
        result = pipeline.process_text(args.text, profile)
    elif args.input:
        result = pipeline.process_json_file(args.input, profile)
    else:
        print("Error: Provide --text, --input, or --demo")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResult saved to: {args.output}")
    print(json.dumps(result.get("summary", result), indent=2))

if __name__ == "__main__":
    main()

