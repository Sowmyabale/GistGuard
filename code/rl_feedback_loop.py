"""
rl_feedback_loop.py
Simplified Reinforcement Learning Feedback Loop for GXAI-Elderly

Mechanism:
    1. Generate gist for a conversation
    2. Present to user (or evaluator) with thumbs UP/DOWN
    3. If DOWN: adjust cognitive profile weights and re-generate
    4. Track improvement across iterations

This is NOT full PPO/RLHF. It is a lightweight feedback-based adaptation loop
that demonstrates the concept of iterative profile refinement.

Usage:
    python rl_feedback_loop.py --conversation dataset/COVA-X/.../bank_001.json
    python rl_feedback_loop.py --batch_eval --num_samples 50
"""

import os
import sys
import json
import random
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from train_layer2_profiler import ConversationEncoder, CognitiveProfilerBiLSTM

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LAYER1_DIR = './layer1_bert_results/best_model'
LAYER2_DIR = './layer2_profiler_results'
LAYER3_ADAPTER_DIR = './layer3_llm_lora_results/lora_adapter'
LAYER3_BASE_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

# ============================================================================
# PROFILE ADJUSTMENT RULES (Simulated RL Policy)
# ============================================================================
class ProfileAdjustmentPolicy:
    """
    Simple rule-based policy for adjusting cognitive profiles based on feedback.
    
    Logic:
    - If gist was rated POOR on 'clarity' → assume tech_savviness is LOWER
    - If gist was rated POOR on 'trust' → assume trust_level is HIGHER (more gullible)
    - If gist was rated POOR on 'actionability' → add more explicit step-by-step
    """
    
    def __init__(self):
        self.adjustment_history = []
    
    def adjust_profile(self, current_profile, feedback):
        """
        Adjust profile based on user feedback.
        
        Args:
            current_profile: dict with trust_level, tech_savviness, scam_awareness
            feedback: dict with ratings for clarity, trust, actionability (1-5)
        
        Returns:
            adjusted_profile: dict with updated values
            explanation: str explaining what was adjusted
        """
        adjusted = current_profile.copy()
        explanations = []
        
        # Rule 1: Low clarity rating → lower tech_savviness assumption
        if feedback.get('clarity', 3) <= 2:
            if adjusted['tech_savviness'] > 0:
                adjusted['tech_savviness'] -= 1
                explanations.append("Clarity was low → assumed lower tech literacy")
        
        # Rule 2: Low trust rating → higher trust level (more susceptible)
        if feedback.get('trust', 3) <= 2:
            if adjusted['trust_level'] < 2:
                adjusted['trust_level'] += 1
                explanations.append("Trust was misplaced → assumed higher susceptibility")
        
        # Rule 3: Low actionability → lower scam_awareness
        if feedback.get('actionability', 3) <= 2:
            if adjusted['scam_awareness'] > 0:
                adjusted['scam_awareness'] -= 1
                explanations.append("Action unclear → assumed lower scam awareness")
        
        # Rule 4: High ratings across board → can slightly increase assumptions
        if all(feedback.get(k, 3) >= 4 for k in ['clarity', 'trust', 'actionability']):
            if adjusted['scam_awareness'] < 2:
                adjusted['scam_awareness'] += 1
                explanations.append("All ratings high → user may be more aware than assumed")
        
        self.adjustment_history.append({
            'from': current_profile,
            'to': adjusted,
            'feedback': feedback,
            'explanations': explanations
        })
        
        return adjusted, '; '.join(explanations) if explanations else 'No adjustment needed'

# ============================================================================
# SIMULATED FEEDBACK
# ============================================================================
def simulate_feedback(gist_text, true_scam_type, profile):
    """
    Simulate user feedback based on heuristics.
    In real deployment, this comes from actual users.
    """
    feedback = {}
    
    # Clarity: Check if gist is short and simple
    word_count = len(gist_text.split())
    if word_count < 50:
        feedback['clarity'] = random.choice([4, 5])  # Likely clear
    elif word_count > 120:
        feedback['clarity'] = random.choice([2, 3])  # Might be too long
    else:
        feedback['clarity'] = random.choice([3, 4])
    
    # Trust: Check if gist contains actionable advice
    has_action = any(word in gist_text.lower() for word in ['hang up', 'call', 'do not', 'verify', 'check'])
    feedback['trust'] = 4 if has_action else random.choice([2, 3])
    
    # Actionability: Check for explicit DO section
    has_do = '[do]' in gist_text.lower() or 'do:' in gist_text.lower()
    feedback['actionability'] = 4 if has_do else random.choice([2, 3])
    
    # Overall helpfulness
    feedback['helpful'] = int(round(sum(feedback.values()) / len(feedback)))
    
    return feedback

# ============================================================================
# FEEDBACK LOOP DEMONSTRATION
# ============================================================================
class FeedbackLoopDemo:
    def __init__(self):
        self.policy = ProfileAdjustmentPolicy()
        self.iteration_results = []
    
    def run_single_conversation(self, conv_json_path, max_iterations=3):
        """Run feedback loop on a single conversation."""
        with open(conv_json_path, 'r', encoding='utf-8') as f:
            conv = json.load(f)
        
        # Extract info
        victim = conv.get('victim_config', {})
        scam_type = conv.get('scam_type', 'unknown')
        age = victim.get('age', '70-74')
        gender = victim.get('gender', 'female')
        
        # Initial profile from ground truth (in real deployment, from Layer 2)
        trust_map_rev = {'highly_skeptical': 0, 'skeptical': 0, 'cautious': 1,
                         'somewhat_trusting': 1, 'moderately_trusting': 1,
                         'very_trusting': 2, 'highly_trusting': 2}
        tech_map_rev = {'low': 0, 'moderate': 1, 'high': 2}
        awareness_map_rev = {'unaware': 0, 'vaguely_aware': 1, 'aware': 2}
        
        profile = {
            'trust_level': trust_map_rev.get(victim.get('trust_level'), 1),
            'tech_savviness': tech_map_rev.get(victim.get('tech_savviness'), 0),
            'scam_awareness': awareness_map_rev.get(victim.get('scam_awareness'), 1),
        }
        
        print(f"Conversation: {conv.get('conversation_id')}")
        print(f"Scam type: {scam_type}")
        print(f"Initial profile: {profile}")
        print("-" * 60)
        
        # Simulate iterative refinement
        for iteration in range(max_iterations):
            # In real system: generate_gist(profile) here
            # For demo: we simulate the gist quality based on profile fit
            simulated_gist = self._simulate_gist(scam_type, profile, age, gender)
            feedback = simulate_feedback(simulated_gist, scam_type, profile)
            
            print(f"\nIteration {iteration + 1}:")
            print(f"  Gist preview: {simulated_gist[:100]}...")
            print(f"  Feedback: clarity={feedback['clarity']}, trust={feedback['trust']}, actionability={feedback['actionability']}")
            
            if feedback['helpful'] >= 4:
                print(f"  ✅ Gist rated helpful. Stopping.")
                break
            
            # Adjust profile
            new_profile, explanation = self.policy.adjust_profile(profile, feedback)
            print(f"  🔄 Adjusting profile: {explanation}")
            print(f"  Profile: {profile} → {new_profile}")
            profile = new_profile
        
        self.iteration_results.append({
            'conversation_id': conv.get('conversation_id'),
            'final_profile': profile,
            'iterations': iteration + 1,
            'improved': True  # Simplified metric
        })
        
        return profile
    
    def _simulate_gist(self, scam_type, profile, age, gender):
        """Simulate gist text for demo purposes."""
        # Higher scam awareness → more detailed gist
        detail_level = profile['scam_awareness']
        
        what = f"This is a {scam_type.replace('_', ' ')} scam."
        why = "The caller is trying to steal your money or information."
        
        if detail_level >= 2:
            reassure = "You recognized the warning signs. Trust your instincts."
            action = "Block the number and report it to the FTC at reportfraud.ftc.gov."
        elif detail_level == 1:
            reassure = "Many people get these calls. You are not alone."
            action = "Hang up now. Call a trusted family member to discuss."
        else:
            reassure = "It is okay to be confused. These callers sound very convincing."
            action = "Do not give any information. Hang up and call your bank using the number on your card."
        
        return f"[WHAT] {what}\n[WHY] {why}\n[REASSURE] {reassure}\n[DO] {action}"
    
    def get_statistics(self):
        """Get statistics about the feedback loop performance."""
        if not self.iteration_results:
            return {}
        
        total = len(self.iteration_results)
        avg_iterations = sum(r['iterations'] for r in self.iteration_results) / total
        
        return {
            'total_conversations': total,
            'average_iterations_to_convergence': avg_iterations,
            'adjustment_history': len(self.policy.adjustment_history)
        }

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GXAI-Elderly Feedback Loop Demo')
    parser.add_argument('--conversation', type=str, help='Path to single conversation JSON')
    parser.add_argument('--batch_eval', action='store_true', help='Run on batch of samples')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples for batch eval')
    args = parser.parse_args()
    
    demo = FeedbackLoopDemo()
    
    if args.conversation:
        demo.run_single_conversation(args.conversation)
    elif args.batch_eval:
        import glob
        files = []
        base_dir = "dataset/COVA-X Dataset/COVA-X_Dataset"
        for cat in os.listdir(base_dir):
            cat_dir = os.path.join(base_dir, cat)
            if os.path.isdir(cat_dir):
                files.extend(glob.glob(os.path.join(cat_dir, "*.json")))
        
        random.seed(42)
        random.shuffle(files)
        files = files[:args.num_samples]
        
        print(f"Running feedback loop on {len(files)} conversations...\n")
        for fpath in files:
            demo.run_single_conversation(fpath)
            print("\n" + "="*60 + "\n")
        
        stats = demo.get_statistics()
        print("\nFeedback Loop Statistics:")
        print(json.dumps(stats, indent=2))
    else:
        print("Usage: python rl_feedback_loop.py --conversation conv.json")
        print("       python rl_feedback_loop.py --batch_eval --num_samples 50")