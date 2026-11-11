"""
train_layer3_llm_lora.py
Layer 3: Gist Translation Engine - Fine-tune Llama-3/Mistral with LoRA
Generates FTT-aligned explanations conditioned on scam type + cognitive profile.
Run on: Google Colab (free T4 GPU) or university GPU cluster with 16GB+ VRAM
"""

# ============================================================================
# STEP 0: Install dependencies (run once in Colab)
# ============================================================================
# !pip install transformers datasets peft accelerate bitsandbytes trl

import os
import json
import glob
import random
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# CONFIGURATION
# ============================================================================
# MODEL OPTIONS:
# Option A: Llama-3.1-8B (best open-source, needs ~16GB with LoRA)
# Option B: Mistral-7B-Instruct (strong instruction following)
# Option C: Google Colab fallback - use 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' for testing

MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'  # For Colab T4 (16GB) testing
# MODEL_NAME = 'meta-llama/Meta-Llama-3.1-8B-Instruct'  # Full model (needs HuggingFace token)
# MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.3'  # Alternative (needs HuggingFace token)

OUTPUT_DIR = './layer3_llm_lora_results'
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4  # Effective batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_R = 16  # Rank
LORA_ALPHA = 32  # Scaling
LORA_DROPOUT = 0.05

# ============================================================================
# STEP 1: Create Training Data from COVA-X
# ============================================================================
def load_cova_x_for_gist(base_dir="dataset/COVA-X Dataset/COVA-X_Dataset"):
    """Create instruction-following dataset for gist explanation generation."""
    examples = []
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Gist templates per scam type
    gist_what = {
        'bank': 'This caller is pretending to be from your bank to steal your money.',
        'government_impersonation': 'This person is pretending to be from the government to scare you.',
        'grandparent': 'This is not your grandchild. Someone is pretending to be family to get money.',
        'investment': 'This is a fake investment offer designed to take your savings.',
        'lottery': 'You have not won a lottery. This is a trick to get your bank details.',
        'medicare': 'This caller is pretending to be from Medicare to steal your health information.',
        'romance': 'This person is not who they say they are. They want your money, not love.',
        'virtual_kidnapping': 'This is a fake kidnapping call designed to panic you into paying money.',
    }
    
    gist_why = {
        'bank': 'Real banks never ask for your card number or PIN over the phone.',
        'government_impersonation': 'Real government agencies send letters first. They do not threaten arrest over the phone.',
        'grandparent': 'Scammers research families online and fake emergencies to create panic.',
        'investment': 'Real investments do not promise guaranteed high returns with no risk.',
        'lottery': 'Real lotteries do not ask winners to pay fees upfront.',
        'medicare': 'Medicare never calls to ask for your card number or personal details.',
        'romance': 'Romance scammers build trust over time, then ask for money with urgent stories.',
        'virtual_kidnapping': 'Scammers use fear and urgency to stop you from thinking clearly.',
    }
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        files = sorted(glob.glob(os.path.join(cat_dir, "*.json")))
        
        for fpath in files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                
                victim = conv.get('victim_config', {})
                scam_type = conv.get('scam_type')
                
                # Build instruction
                what = gist_what.get(scam_type, gist_what['bank'])
                why = gist_why.get(scam_type, gist_why['bank'])
                
                # Adapt based on victim profile
                reassurance = "You did the right thing by being careful. Many people get these calls."
                action = "Hang up now. Call the official number on your card or statement."
                
                if victim.get('trust_level') == 'high':
                    reassurance += " It is okay to trust people, but scammers abuse that trust."
                if victim.get('tech_savviness') == 'low':
                    action += " Ask a family member or friend to help you if needed."
                if victim.get('scam_awareness') == 'unaware':
                    reassurance += " You are not alone in this. These calls happen to many people."
                
                # Build the full explanation
                explanation = f"[WHAT] {what}\n\n[WHY] {why}\n\n[REASSURE] {reassurance}\n\n[DO] {action}"
                
                # Instruction format for LLM fine-tuning
                instruction = f"""### Instruction:
You are a protective AI assistant helping an elderly person who is on a scam phone call. 
The scam type is: {scam_type.replace('_', ' ')}.
The victim is a {victim.get('age', '70-74')} year old {victim.get('gender', 'person')} with {victim.get('trust_level', 'moderate')} trust level and {victim.get('tech_savviness', 'low')} tech savviness.
Generate a simple, reassuring explanation following Fuzzy Trace Theory principles (bottom-line meaning, emotional congruence, no jargon).

### Response:
{explanation}"""
                
                examples.append({
                    'text': instruction,
                    'scam_type': scam_type,
                    'victim_age': victim.get('age'),
                    'victim_trust': victim.get('trust_level'),
                })
            except Exception:
                pass
    
    return pd.DataFrame(examples)

# ============================================================================
# STEP 2: Load Model with 4-bit Quantization (for Colab T4)
# ============================================================================
def load_model_for_training():
    print(f"Loading model: {MODEL_NAME}")
    
    # 4-bit quantization config (fits on Colab T4 16GB)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],  # Attention layers
        lora_dropout=LORA_DROPOUT,
        bias='none',
        task_type='CAUSAL_LM',
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# ============================================================================
# STEP 3: Tokenize and Train
# ============================================================================
def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("Creating training dataset from COVA-X...")
    df = load_cova_x_for_gist()
    print(f"Created {len(df)} training examples")
    
    # Load model
    model, tokenizer = load_model_for_training()
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
        )
    
    dataset = Dataset.from_pandas(df[['text']])
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy='epoch',
        save_total_limit=2,
        report_to='none',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting LoRA fine-tuning...")
    trainer.train()
    
    # Save
    model.save_pretrained(f'{OUTPUT_DIR}/lora_adapter')
    tokenizer.save_pretrained(f'{OUTPUT_DIR}/lora_adapter')
    print(f"LoRA adapter saved to {OUTPUT_DIR}/lora_adapter")
    
    return model, tokenizer

# ============================================================================
# STEP 4: Generate Sample Explanations
# ============================================================================
def generate_explanation(model, tokenizer, scam_type, victim_age, victim_trust):
    """Generate a gist explanation using the fine-tuned model."""
    prompt = f"""### Instruction:
You are a protective AI assistant helping an elderly person who is on a scam phone call. 
The scam type is: {scam_type.replace('_', ' ')}.
The victim is a {victim_age} year old person with {victim_trust} trust level.
Generate a simple, reassuring explanation following Fuzzy Trace Theory principles.

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    if '### Response:' in response:
        response = response.split('### Response:')[1].strip()
    
    return response

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    # Train
    model, tokenizer = train()
    
    # Test generation
    print("\n" + "="*60)
    print("Sample Generated Explanations:")
    print("="*60)
    
    test_cases = [
        ('bank', '70-74', 'high'),
        ('romance', '60-64', 'high'),
        ('grandparent', '75-79', 'moderately_trusting'),
    ]
    
    for scam, age, trust in test_cases:
        print(f"\n--- Scam: {scam}, Age: {age}, Trust: {trust} ---")
        explanation = generate_explanation(model, tokenizer, scam, age, trust)
        print(explanation[:500])
    
    print("\nLayer 3 training complete!")
