"""
evaluate_end_to_end.py
End-to-End Evaluation: Chain Layer 1 -> Layer 2 -> Layer 3
Run after training all three layers.
Usage:
    python evaluate_end_to_end.py --conversation dataset/COVA-X Dataset/COVA-X_Dataset/bank/bank_001.json
    python evaluate_end_to_end.py --test_dir dataset/COVA-X Dataset/COVA-X_Dataset --limit 100
"""

import os
import sys
import json
import glob
import argparse, random
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Hugging Face
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# Import model classes from training scripts
sys.path.insert(0, os.path.dirname(__file__))
from train_layer2_profiler import ConversationEncoder, CognitiveProfilerBiLSTM, ConversationDataset

# ============================================================================
# CONFIGURATION
# ============================================================================
LAYER1_DIR = './layer1_bert_results/best_model'
LAYER2_DIR = './layer2_profiler_results'
LAYER3_ADAPTER_DIR = './layer3_llm_lora_results/lora_adapter'

# Default base model used for Layer 3 (must match training script)
LAYER3_BASE_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
# LAYER3_BASE_MODEL = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

LABEL_MAP = {
    'bank': 0, 'government_impersonation': 1, 'grandparent': 2,
    'investment': 3, 'lottery': 4, 'medicare': 5, 'romance': 6, 'virtual_kidnapping': 7
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# LOADERS
# ============================================================================
class EndToEndPipeline:
    def __init__(self):
        print("Loading Layer 1 (Scam Classifier)...")
        self.l1_tokenizer = AutoTokenizer.from_pretrained(LAYER1_DIR)
        self.l1_model = AutoModelForSequenceClassification.from_pretrained(LAYER1_DIR).to(device)
        self.l1_model.eval()
        
        print("Loading Layer 2 (Cognitive Profiler)...")
        self.l2_encoder = ConversationEncoder()
        self.l2_models = {}
        for target in ['trust_level', 'tech_savviness', 'scam_awareness']:
            model = CognitiveProfilerBiLSTM(
                embed_dim=384, hidden_dim=256, num_layers=2, num_classes=3, dropout=0.3
            ).to(device)
            path = os.path.join(LAYER2_DIR, f"{target}_best.pt")
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                self.l2_models[target] = model
            else:
                print(f"  Warning: {path} not found. Layer 2 {target} will not be available.")
        
        print("Loading Layer 3 (Gist Generator)...")
        self.l3_tokenizer = AutoTokenizer.from_pretrained(LAYER3_ADAPTER_DIR)
        # Load base model + LoRA adapter
        base = AutoModelForCausalLM.from_pretrained(
            LAYER3_BASE_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.l3_model = PeftModel.from_pretrained(base, LAYER3_ADAPTER_DIR)
        self.l3_model.eval()
        if not torch.cuda.is_available():
            self.l3_model = self.l3_model.to(device)
        print("All layers loaded.")
    
    # ===================== LAYER 1 =====================
    def predict_scam_type(self, conversation_text):
        inputs = self.l1_tokenizer(
            conversation_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length'
        ).to(device)
        with torch.no_grad():
            logits = self.l1_model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return ID2LABEL[pred_id], probs
    
    # ===================== LAYER 2 =====================
    def predict_cognitive_profile(self, victim_turns, attacker_turns):
        emb = self.l2_encoder.encode_conversation(victim_turns, attacker_turns)
        x = torch.FloatTensor(emb).unsqueeze(0).to(device)  # (1, seq, dim)
        profile = {}
        with torch.no_grad():
            for target, model in self.l2_models.items():
                logits = model(x)
                pred = torch.argmax(logits, dim=1).item()
                profile[target] = pred
        return profile
    
    # ===================== LAYER 3 =====================
    def generate_gist(self, scam_type, profile, victim_age='70-74', victim_gender='female'):
        trust_map = {0: 'low', 1: 'moderately_trusting', 2: 'high'}
        tech_map = {0: 'low', 1: 'moderate', 2: 'high'}
        awareness_map = {0: 'unaware', 1: 'vaguely_aware', 2: 'aware'}
        
        trust = trust_map.get(profile.get('trust_level', 1), 'moderately_trusting')
        tech = tech_map.get(profile.get('tech_savviness', 0), 'low')
        awareness = awareness_map.get(profile.get('scam_awareness', 1), 'vaguely_aware')
        
        prompt = (
            "You are a protective AI assistant helping an elderly person on a scam phone call. "
            "The scam type is: {scam}. "
            "The victim is a {age} year old {gender} with {trust} trust level, "
            "{tech} tech savviness, and {awareness} scam awareness. "
            "Generate a simple, reassuring explanation following Fuzzy Trace Theory. "
            "Use this exact format:\n"
            "[WHAT] <one-sentence bottom-line meaning>\n"
            "[WHY] <one-sentence reason>\n"
            "[REASSURE] <one-sentence emotional reassurance>\n"
            "[DO] <one-sentence action>\n\n"
            "Explanation:\n"
        ).format(
            scam=scam_type.replace('_', ' '),
            age=victim_age,
            gender=victim_gender,
            trust=trust,
            tech=tech,
            awareness=awareness
        )
        
        inputs = self.l3_tokenizer(prompt, return_tensors='pt').to(self.l3_model.device)
        with torch.no_grad():
            outputs = self.l3_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.6,
                do_sample=True,
                top_p=0.92,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.l3_tokenizer.eos_token_id,
            )
        response = self.l3_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt.strip() in response:
            response = response.replace(prompt.strip(), "").strip()
        return response
    
    # ===================== FULL PIPELINE =====================
    def process_conversation(self, conv_json_path):
        with open(conv_json_path, 'r', encoding='utf-8') as f:
            conv = json.load(f)
        
        turns = conv.get('turns', [])
        texts = [f"{t.get('role','?' )}: {t.get('content','')}" for t in turns]
        full_text = '\n'.join(texts)
        
        victim_turns = [t.get('content','') for t in turns if t.get('role')=='victim']
        attacker_turns = [t.get('content','') for t in turns if t.get('role')=='attacker']
        
        victim = conv.get('victim_config', {})
        age = victim.get('age', '70-74')
        gender = victim.get('gender', 'female')
        
        # Layer 1
        scam_type, probs = self.predict_scam_type(full_text)
        
        # Layer 2
        profile = self.predict_cognitive_profile(victim_turns, attacker_turns)
        
        # Layer 3
        gist = self.generate_gist(scam_type, profile, age, gender)
        
        return {
            'conversation_id': conv.get('conversation_id'),
            'true_scam_type': conv.get('scam_type'),
            'predicted_scam_type': scam_type,
            'scam_probs': {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))},
            'cognitive_profile': profile,
            'gist_explanation': gist,
        }

# ============================================================================
# BATCH EVALUATION
# ============================================================================
def evaluate_on_test_dir(test_dir, limit=None):
    pipeline = EndToEndPipeline()
    files = []
    for cat in os.listdir(test_dir):
        cat_dir = os.path.join(test_dir, cat)
        if os.path.isdir(cat_dir):
            files.extend(sorted(glob.glob(os.path.join(cat_dir, '*.json'))))
    if limit:
        random.seed(42); random.shuffle(files); files = files[:limit]
    
    results = []
    correct = 0
    for fpath in files:
        try:
            res = pipeline.process_conversation(fpath)
            results.append(res)
            if res['true_scam_type'] == res['predicted_scam_type']:
                correct += 1
        except Exception as e:
            print(f"Error on {fpath}: {e}")
    
    acc = correct / len(results) if results else 0
    print(f"\nProcessed {len(results)} conversations")
    print(f"Layer 1 Accuracy: {acc:.4f}")
    
    # Save results
    out_path = 'end_to_end_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GXAI-Elderly End-to-End Evaluation')
    parser.add_argument('--conversation', type=str, help='Path to a single conversation JSON')
    parser.add_argument('--test_dir', type=str, help='Directory with conversation subfolders')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of test files')
    args = parser.parse_args()
    
    if args.conversation:
        pipeline = EndToEndPipeline()
        result = pipeline.process_conversation(args.conversation)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.test_dir:
        evaluate_on_test_dir(args.test_dir, args.limit)
    else:
        print("Please provide --conversation or --test_dir")
