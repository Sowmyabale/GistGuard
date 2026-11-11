"""
generate_all_gists_hpc.py
HPC-ready script to generate gist explanations for ALL COVA-X conversations
using a larger LLM (LLaMA-3-8B or Mistral-7B) with LoRA.

Run on HPC with A100/V100:
    sbatch generate_gists.slurm
    OR
    python generate_all_gists_hpc.py --model meta-llama/Meta-Llama-3.1-8B-Instruct

Outputs: results/all_gists_llama3.json
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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model options
MODEL_OPTIONS = {
    'llama3': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
    'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
}

# Paths
COVA_X_DIR = "dataset/COVA-X Dataset/COVA-X_Dataset"
LORA_ADAPTER_DIR = './layer3_llm_lora_results/lora_adapter'
OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generation config
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.15

# ============================================================================
# LOAD MODEL
# ============================================================================
def load_model(model_name_or_path):
    """Load base model + LoRA adapter for inference."""
    print(f"Loading model: {model_name_or_path}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A")
    
    # 4-bit quantization for memory efficiency on HPC
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load LoRA adapter if available
    if os.path.exists(LORA_ADAPTER_DIR) and os.path.exists(os.path.join(LORA_ADAPTER_DIR, 'adapter_config.json')):
        print(f"Loading LoRA adapter from {LORA_ADAPTER_DIR}...")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    else:
        print("Warning: LoRA adapter not found. Using base model only.")
        model = base_model
    
    model.eval()
    print("Model loaded successfully.\n")
    return model, tokenizer

# ============================================================================
# PROMPT BUILDERS
# ============================================================================
def build_prompt_llama3(scam_type, profile, victim_age='70-74', victim_gender='female'):
    """Build Llama-3 chat prompt for gist generation."""
    trust_map = {0: 'low', 1: 'moderate', 2: 'high'}
    tech_map = {0: 'low', 1: 'moderate', 2: 'high'}
    awareness_map = {0: 'unaware', 1: 'vaguely aware', 2: 'aware'}
    
    trust = trust_map.get(profile.get('trust_level', 1), 'moderate')
    tech = tech_map.get(profile.get('tech_savviness', 0), 'low')
    awareness = awareness_map.get(profile.get('scam_awareness', 1), 'vaguely aware')
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a protective AI assistant helping an elderly person who is on a scam phone call. "
                "Generate simple, reassuring explanations following Fuzzy Trace Theory: "
                "use bottom-line meaning, emotional congruence, no jargon. "
                "Format: [WHAT] [WHY] [REASSURE] [DO]"
            )
        },
        {
            "role": "user",
            "content": (
                f"Scam type: {scam_type.replace('_', ' ')}. "
                f"Victim profile: {victim_age} year old {victim_gender}, "
                f"{trust} trust, {tech} tech savviness, {awareness} scam awareness. "
                f"Generate a gist explanation."
            )
        }
    ]
    
    return messages

def build_prompt_mistral(scam_type, profile, victim_age='70-74', victim_gender='female'):
    """Build Mistral chat prompt for gist generation."""
    trust_map = {0: 'low', 1: 'moderate', 2: 'high'}
    tech_map = {0: 'low', 1: 'moderate', 2: 'high'}
    awareness_map = {0: 'unaware', 1: 'vaguely aware', 2: 'aware'}
    
    trust = trust_map.get(profile.get('trust_level', 1), 'moderate')
    tech = tech_map.get(profile.get('tech_savviness', 0), 'low')
    awareness = awareness_map.get(profile.get('scam_awareness', 1), 'vaguely aware')
    
    prompt = (
        f"[INST] You are a protective AI assistant helping an elderly person on a scam call. "
        f"Scam type: {scam_type.replace('_', ' ')}. "
        f"Victim: {victim_age} year old {victim_gender}, {trust} trust, {tech} tech, {awareness} awareness. "
        f"Generate a simple gist: [WHAT] [WHY] [REASSURE] [DO]. [/INST]"
    )
    return prompt

# ============================================================================
# GENERATE GIST
# ============================================================================
def generate_gist(model, tokenizer, scam_type, profile, victim_age, victim_gender, model_key='llama3'):
    """Generate a gist explanation for a single conversation."""
    if model_key == 'llama3':
        messages = build_prompt_llama3(scam_type, profile, victim_age, victim_gender)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif model_key == 'mistral':
        prompt = build_prompt_mistral(scam_type, profile, victim_age, victim_gender)
    else:
        # Generic prompt
        prompt = build_prompt_llama3(scam_type, profile, victim_age, victim_gender)
        prompt = str(prompt)
    
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if model_key == 'llama3':
        # Llama-3 format: assistant content after the prompt
        if "assistant" in response.lower() or "[/INST]" in response:
            parts = response.split("[/INST]")
            if len(parts) > 1:
                response = parts[-1].strip()
    else:
        # Remove prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
    
    return response

# ============================================================================
# PROCESS ALL COVA-X
# ============================================================================
def process_all_cova_x(model, tokenizer, model_key, limit=None):
    """Generate gists for all COVA-X conversations."""
    results = []
    files = []
    
    for category in os.listdir(COVA_X_DIR):
        cat_dir = os.path.join(COVA_X_DIR, category)
        if os.path.isdir(cat_dir):
            files.extend(sorted(glob.glob(os.path.join(cat_dir, "*.json"))))
    
    if limit:
        files = files[:limit]
    
    print(f"Processing {len(files)} conversations...")
    
    for fpath in tqdm(files):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                conv = json.load(f)
            
            victim = conv.get('victim_config', {})
            scam_type = conv.get('scam_type', 'unknown')
            age = victim.get('age', '70-74')
            gender = victim.get('gender', 'female')
            
            # Build profile from ground truth labels
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
            
            gist = generate_gist(model, tokenizer, scam_type, profile, age, gender, model_key)
            
            results.append({
                'conversation_id': conv.get('conversation_id'),
                'scam_type': scam_type,
                'victim_profile': profile,
                'victim_age': age,
                'victim_gender': gender,
                'gist': gist,
            })
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate gists for all COVA-X conversations using HPC GPU')
    parser.add_argument('--model', type=str, default='llama3',
                        choices=['llama3', 'mistral', 'tinyllama'],
                        help='Base model to use')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of conversations (for testing)')
    parser.add_argument('--output', type=str, default='all_gists.json',
                        help='Output filename')
    args = parser.parse_args()
    
    model_name = MODEL_OPTIONS[args.model]
    model, tokenizer = load_model(model_name)
    
    results = process_all_cova_x(model, tokenizer, args.model, args.limit)
    
    output_path = os.path.join(OUTPUT_DIR, args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': args.model,
            'model_path': model_name,
            'generated_at': datetime.now().isoformat(),
            'num_conversations': len(results),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(results)} gists.")
    print(f"Saved to: {output_path}")
    print("\nSample gist:")
    if results:
        print(results[0]['gist'][:500])