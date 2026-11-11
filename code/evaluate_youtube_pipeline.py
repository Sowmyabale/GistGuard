"""
evaluate_youtube_pipeline.py
Run the full GXAI-Elderly pipeline on all 243 YouTube transcripts.
Generates scam predictions, cognitive profiles, and gists for real-world validation.

Usage:
    python evaluate_youtube_pipeline.py --input_dir dataset/Youtube\ transcripts/converted_json
    python evaluate_youtube_pipeline.py --input_dir dataset/Youtube\ transcripts/converted_json --limit 50

Output: youtube_validation_results.json
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
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

LABEL_MAP = {
    'bank': 0, 'government_impersonation': 1, 'grandparent': 2,
    'investment': 3, 'lottery': 4, 'medicare': 5, 'romance': 6, 'virtual_kidnapping': 7
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

TRUST_MAP = {0: 'low', 1: 'moderate', 2: 'high'}
TECH_MAP = {0: 'low', 1: 'moderate', 2: 'high'}
AWARENESS_MAP = {0: 'unaware', 1: 'vaguely_aware', 2: 'aware'}

# ============================================================================
# LOAD ALL LAYERS
# ============================================================================
class YouTubePipeline:
    def __init__(self):
        print("Loading GXAI-Elderly Pipeline...")
        
        # Layer 1: BERT Scam Classifier
        print("  [Layer 1] Loading BERT classifier...")
        self.l1_tokenizer = AutoTokenizer.from_pretrained(LAYER1_DIR)
        self.l1_model = AutoModelForSequenceClassification.from_pretrained(LAYER1_DIR).to(DEVICE)
        self.l1_model.eval()
        
        # Layer 2: BiLSTM Cognitive Profiler
        print("  [Layer 2] Loading cognitive profiler...")
        self.l2_encoder = ConversationEncoder()
        self.l2_models = {}
        for target in ['trust_level', 'tech_savviness', 'scam_awareness']:
            model = CognitiveProfilerBiLSTM(
                embed_dim=384, hidden_dim=256, num_layers=2, num_classes=3, dropout=0.3
            ).to(DEVICE)
            path = os.path.join(LAYER2_DIR, f"{target}_best.pt")
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()
                self.l2_models[target] = model
        
        # Layer 3: LoRA LLM Gist Generator
        print("  [Layer 3] Loading gist generator...")
        self.l3_tokenizer = AutoTokenizer.from_pretrained(LAYER3_ADAPTER_DIR)
        base = AutoModelForCausalLM.from_pretrained(
            LAYER3_BASE_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.l3_model = PeftModel.from_pretrained(base, LAYER3_ADAPTER_DIR)
        self.l3_model.eval()
        if not torch.cuda.is_available():
            self.l3_model = self.l3_model.to(DEVICE)
        
        print("All layers loaded successfully!\n")
    
    def predict_scam_type(self, text):
        inputs = self.l1_tokenizer(
            text, return_tensors='pt', truncation=True, max_length=512, padding='max_length'
        ).to(DEVICE)
        with torch.no_grad():
            logits = self.l1_model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return ID2LABEL[pred_id], probs
    
    def predict_profile(self, turns_text):
        # For YouTube data, we don't have explicit victim/attacker roles
        # Treat all turns as a sequence
        victim_turns = [t for i, t in enumerate(turns_text) if i % 2 == 0]  # heuristic
        attacker_turns = [t for i, t in enumerate(turns_text) if i % 2 == 1]
        
        emb = self.l2_encoder.encode_conversation(victim_turns, attacker_turns)
        x = torch.FloatTensor(emb).unsqueeze(0).to(DEVICE)
        
        profile = {}
        with torch.no_grad():
            for target, model in self.l2_models.items():
                logits = model(x)
                pred = torch.argmax(logits, dim=1).item()
                profile[target] = pred
        return profile
    
    def generate_gist(self, scam_type, profile):
        trust = TRUST_MAP.get(profile.get('trust_level', 1), 'moderate')
        tech = TECH_MAP.get(profile.get('tech_savviness', 0), 'low')
        awareness = AWARENESS_MAP.get(profile.get('scam_awareness', 1), 'vaguely_aware')
        
        prompt = (
            "You are a protective AI assistant helping an elderly person on a scam phone call. "
            f"The scam type is: {scam_type.replace('_', ' ')}. "
            f"The victim has {trust} trust level, {tech} tech savviness, and {awareness} scam awareness. "
            "Generate a simple, reassuring explanation following Fuzzy Trace Theory. "
            "Use this exact format:\n"
            "[WHAT] <one-sentence bottom-line meaning>\n"
            "[WHY] <one-sentence reason>\n"
            "[REASSURE] <one-sentence emotional reassurance>\n"
            "[DO] <one-sentence action>\n\n"
            "Explanation:\n"
        )
        
        inputs = self.l3_tokenizer(prompt, return_tensors='pt').to(self.l3_model.device)
        with torch.no_grad():
            outputs = self.l3_model.generate(
                **inputs, max_new_tokens=150, temperature=0.6, do_sample=True,
                top_p=0.92, repetition_penalty=1.2, no_repeat_ngram_size=3,
                pad_token_id=self.l3_tokenizer.eos_token_id,
            )
        response = self.l3_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt.strip() in response:
            response = response.replace(prompt.strip(), "").strip()
        return response
    
    def process_youtube_conversation(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            conv = json.load(f)
        
        turns = conv.get('turns', [])
        texts = [t.get('content', '') for t in turns]
        full_text = '\n'.join([f"{t.get('role', '?')}: {t.get('content', '')}" for t in turns])
        
        # Layer 1: Predict scam type (no ground truth for YouTube)
        scam_type, probs = self.predict_scam_type(full_text)
        
        # Layer 2: Infer profile from victim responses
        profile = self.predict_profile(texts)
        
        # Layer 3: Generate gist
        gist = self.generate_gist(scam_type, profile)
        
        return {
            'conversation_id': conv.get('conversation_id', os.path.basename(json_path)),
            'num_turns': len(turns),
            'predicted_scam_type': scam_type,
            'scam_probs': {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))},
            'cognitive_profile': {
                'trust_level': TRUST_MAP.get(profile.get('trust_level')),
                'tech_savviness': TECH_MAP.get(profile.get('tech_savviness')),
                'scam_awareness': AWARENESS_MAP.get(profile.get('scam_awareness')),
            },
            'gist': gist,
        }

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GXAI-Elderly on YouTube transcripts')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with converted YouTube JSON files')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to process')
    parser.add_argument('--output', type=str, default='youtube_validation_results.json', help='Output filename')
    args = parser.parse_args()
    
    # Load pipeline
    pipeline = YouTubePipeline()
    
    # Find all JSON files
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    print(f"Found {len(files)} YouTube transcript files")
    
    if args.limit:
        files = files[:args.limit]
        print(f"Processing first {args.limit} files...")
    
    # Process all
    results = []
    scam_type_counts = {}
    
    for fpath in tqdm(files):
        try:
            result = pipeline.process_youtube_conversation(fpath)
            results.append(result)
            
            scam = result['predicted_scam_type']
            scam_type_counts[scam] = scam_type_counts.get(scam, 0) + 1
        except Exception as e:
            print(f"Error on {fpath}: {e}")
    
    # Save results
    output_data = {
        'generated_at': datetime.now().isoformat(),
        'model': 'GXAI-Elderly v1',
        'num_processed': len(results),
        'scam_type_distribution': scam_type_counts,
        'results': results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"YouTube Validation Complete!")
    print(f"Processed: {len(results)} conversations")
    print(f"Scam type distribution:")
    for scam, count in sorted(scam_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {scam}: {count}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")
    
    # Print sample
    if results:
        print("\nSample result:")
        print(json.dumps(results[0], indent=2))