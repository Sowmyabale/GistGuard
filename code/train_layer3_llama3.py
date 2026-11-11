"""
train_layer3_llama3.py
Train LLaMA-3.1-8B-Instruct with LoRA for high-quality gist generation.
Uses exemplar-based training data (not templates) for diverse, natural output.

Run on HPC A100:
    sbatch hpc_jobs/job_llama3_train.slurm
    OR
    python train_layer3_llama3.py

Requirements:
    - HuggingFace token with LLaMA-3 access
    - ~16GB VRAM (4-bit quantization)
    - ~50GB disk space for model
"""

import os
import json
import glob
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
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
MODEL_NAME = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
OUTPUT_DIR = './layer3_llama3_results'
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 2  # Lower for 8B model
GRADIENT_ACCUMULATION = 8  # Effective batch = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HIGH-QUALITY GIST EXEMPLARS (Few-shot examples)
# These are diverse, natural-language gists — NOT templates
# ============================================================================
EXEMPLARS = {
    'bank': [
        {
            'profile': {'trust': 'moderate', 'tech': 'low', 'awareness': 'vaguely_aware'},
            'gist': "[WHAT] This caller is pretending to be from your bank to steal your money.\n[WHY] Real banks never ask for your card number or PIN over the phone.\n[REASSURE] You did the right thing by being careful. Many people get these calls.\n[DO] Hang up now. Call the number on the back of your bank card if you are worried."
        },
        {
            'profile': {'trust': 'high', 'tech': 'high', 'awareness': 'aware'},
            'gist': "[WHAT] This is a vishing attack using a spoofed bank phone number.\n[WHY] The caller ID can be faked. Real banks verify identity through secure channels, not cold calls demanding instant action.\n[REASSURE] Your actual account is safe. This caller has no access to it.\n[DO] Do not verify any credentials. Block the number and report to your bank's fraud department immediately."
        },
        {
            'profile': {'trust': 'high', 'tech': 'low', 'awareness': 'unaware'},
            'gist': "[WHAT] The person on the phone is lying about being from your bank. They want to steal your money.\n[WHY] Your real bank already knows your information. They do not call and ask for passwords or card numbers.\n[REASSURE] You are not alone. Many kind people get tricked by these calls because the scammers sound very real.\n[DO] Please hang up right now. Do not call the number they gave you. Ask a family member to help you call your real bank."
        },
    ],
    'romance': [
        {
            'profile': {'trust': 'high', 'tech': 'low', 'awareness': 'unaware'},
            'gist': "[WHAT] This person is not who they say they are. They want your money, not a relationship.\n[WHY] Romance scammers spend weeks building trust, then create an emergency that needs money. The photos are usually stolen.\n[REASSURE] You are not foolish for caring about someone. Scammers are very good at what they do. Many smart people fall for this.\n[DO] Do not send money or gift cards. Talk to a family member about this person."
        },
        {
            'profile': {'trust': 'moderate', 'tech': 'moderate', 'awareness': 'vaguely_aware'},
            'gist': "[WHAT] This online relationship is a scam designed to get your money.\n[WHY] The 'crypto trading' opportunity is fake. They show fake profits to get you to invest more, then disappear.\n[REASSURE] It is natural to want connection. The scammer exploited that, and that is not your fault.\n[DO] Cut off all contact. Do not send any more money. Report to the FBI at ic3.gov."
        },
    ],
    'government_impersonation': [
        {
            'profile': {'trust': 'moderate', 'tech': 'low', 'awareness': 'unaware'},
            'gist': "[WHAT] This person is pretending to be from the government to scare you into paying money.\n[WHY] Real government agencies send letters by mail first. They never demand payment over the phone or threaten arrest.\n[REASSURE] You are not in trouble. This is a common scam that targets many people, especially seniors.\n[DO] Hang up immediately. Do not call the number they gave you. If worried, look up the official agency number yourself."
        },
        {
            'profile': {'trust': 'low', 'tech': 'high', 'awareness': 'aware'},
            'gist': "[WHAT] This is an IRS impersonation scam using fear tactics and caller ID spoofing.\n[WHY] The IRS never initiates contact via phone for owed taxes. They certainly never demand immediate wire transfer or gift cards to avoid arrest.\n[REASSURE] Your tax records are secure. This caller has no legal authority whatsoever.\n[DO] Hang up. Report the call to tigta.gov and the FTC. Block the number."
        },
    ],
    'grandparent': [
        {
            'profile': {'trust': 'high', 'tech': 'low', 'awareness': 'unaware'},
            'gist': "[WHAT] This is not your grandchild. Someone is pretending to be family to get money.\n[WHY] Scammers find family information online and fake emergencies to create panic. The crying voice can be faked with AI now.\n[REASSURE] Your real grandchild is safe. This caller is lying about who they are. You are a caring grandparent — that is why they targeted you.\n[DO] Hang up and call your grandchild directly on their real phone number to check. Do not send any money."
        },
    ],
    'investment': [
        {
            'profile': {'trust': 'moderate', 'tech': 'moderate', 'awareness': 'vaguely_aware'},
            'gist': "[WHAT] This is a fake investment opportunity designed to steal your savings.\n[WHY] Real investments carry risk and never guarantee returns. If it sounds too good to be true, it is. The 'guaranteed 50% return' is impossible.\n[REASSURE] You are smart to ask questions. Many people lose life savings to these scams. You are not being overly cautious.\n[DO] Do not send any money. Talk to a financial advisor you trust before investing in anything."
        },
    ],
    'lottery': [
        {
            'profile': {'trust': 'high', 'tech': 'low', 'awareness': 'unaware'},
            'gist': "[WHAT] You have not won a lottery. This is a trick to get your bank details.\n[WHY] Real lotteries do not notify winners by phone call, and they never ask for fees to claim a prize. You cannot win a lottery you never entered.\n[REASSURE] It is exciting to think about winning, but this is not real. You did nothing wrong by answering the call.\n[DO] Do not give any personal information. Hang up and do not call back. If you want to play the lottery, buy tickets from official stores only."
        },
    ],
    'medicare': [
        {
            'profile': {'trust': 'moderate', 'tech': 'low', 'awareness': 'vaguely_aware'},
            'gist': "[WHAT] This caller is pretending to be from Medicare to steal your health information.\n[WHY] Medicare never calls to ask for your card number or personal details over the phone. They already have your information on file.\n[REASSURE] Your Medicare benefits are safe. This caller has no connection to Medicare and cannot cancel your coverage.\n[DO] Hang up now. If you have questions about Medicare, call the number on your Medicare card — not the number the caller gave you."
        },
    ],
    'virtual_kidnapping': [
        {
            'profile': {'trust': 'high', 'tech': 'low', 'awareness': 'unaware'},
            'gist': "[WHAT] This is a fake kidnapping call designed to panic you into paying money.\n[WHY] Scammers use fear to stop you from thinking clearly. They do not actually have anyone. The crying voice is often recorded or fake.\n[REASSURE] Take a deep breath. Your family member is likely safe. These calls are designed to create panic so you act without thinking.\n[DO] Hang up and call your family member directly on their real number. Do not pay anything. If truly worried, call 911."
        },
    ],
}

# ============================================================================
# CREATE DIVERSE TRAINING DATA
# ============================================================================
def create_training_data(base_dir="dataset/COVA-X Dataset/COVA-X_Dataset"):
    """Create instruction-following dataset with exemplar-based examples."""
    examples = []
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        files = sorted(glob.glob(os.path.join(cat_dir, "*.json")))
        scam_exemplars = EXEMPLARS.get(category, EXEMPLARS['bank'])
        
        for fpath in files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                
                victim = conv.get('victim_config', {})
                scam_type = conv.get('scam_type')
                
                # Map victim profile to exemplar
                trust_map_rev = {'highly_skeptical': 'low', 'skeptical': 'low', 'cautious': 'moderate',
                                 'somewhat_trusting': 'moderate', 'moderately_trusting': 'moderate',
                                 'very_trusting': 'high', 'highly_trusting': 'high'}
                tech_map_rev = {'low': 'low', 'moderate': 'moderate', 'high': 'high'}
                
                v_trust = trust_map_rev.get(victim.get('trust_level', 'moderate'), 'moderate')
                v_tech = tech_map_rev.get(victim.get('tech_savviness', 'low'), 'low')
                v_age = victim.get('age', '70-74')
                v_gender = victim.get('gender', 'female')
                
                # Select best-matching exemplar
                best_exemplar = None
                for ex in scam_exemplars:
                    if ex['profile']['trust'] == v_trust and ex['profile']['tech'] == v_tech:
                        best_exemplar = ex
                        break
                if best_exemplar is None:
                    best_exemplar = random.choice(scam_exemplars)
                
                # Build few-shot prompt for training
                # Include 1 exemplar + the target conversation
                other_exemplar = random.choice([e for e in scam_exemplars if e != best_exemplar])
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a protective AI assistant helping an elderly person on a scam phone call. "
                            "Generate simple, reassuring explanations. Always use this format:\n"
                            "[WHAT] <one-sentence bottom-line meaning>\n"
                            "[WHY] <one-sentence reason>\n"
                            "[REASSURE] <one-sentence emotional reassurance>\n"
                            "[DO] <one-sentence action>"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Scam: {scam_type.replace('_', ' ')}, "
                            f"Victim: {v_age} {v_gender}, {v_trust} trust, {v_tech} tech"
                        )
                    },
                    {
                        "role": "assistant",
                        "content": best_exemplar['gist']
                    }
                ]
                
                # Store as text for tokenization
                examples.append({
                    'messages': messages,
                    'scam_type': scam_type,
                    'victim_age': v_age,
                    'victim_trust': v_trust,
                })
            except Exception:
                pass
    
    return pd.DataFrame(examples)

# ============================================================================
# LOAD MODEL
# ============================================================================
def load_model():
    print(f"Loading model: {MODEL_NAME}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=LORA_DROPOUT,
        bias='none',
        task_type='CAUSAL_LM',
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# ============================================================================
# TRAINING
# ============================================================================
def train():
    print("Creating training dataset from COVA-X...")
    df = create_training_data()
    print(f"Created {len(df)} training examples")
    
    # Load model
    model, tokenizer = load_model()
    
    # Apply chat template
    def format_messages(examples):
        texts = []
        for messages in examples['messages']:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {'text': texts}
    
    dataset = Dataset.from_pandas(df[['messages']])
    dataset = dataset.map(format_messages, batched=True)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
        )
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        bf16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy='epoch',
        save_total_limit=2,
        report_to='none',
        optim='paged_adamw_8bit',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    print("\nStarting LLaMA-3 LoRA fine-tuning...")
    trainer.train()
    
    # Save
    model.save_pretrained(f'{OUTPUT_DIR}/lora_adapter')
    tokenizer.save_pretrained(f'{OUTPUT_DIR}/lora_adapter')
    print(f"LoRA adapter saved to {OUTPUT_DIR}/lora_adapter")
    
    return model, tokenizer

# ============================================================================
# TEST GENERATION
# ============================================================================
def test_generation(model, tokenizer):
    print("\n" + "="*60)
    print("Sample Generated Gists:")
    print("="*60)
    
    test_cases = [
        ('bank', '70-74', 'female', 'moderate', 'low', 'vaguely_aware'),
        ('bank', '70-74', 'female', 'high', 'high', 'aware'),
        ('romance', '60-64', 'female', 'high', 'low', 'unaware'),
        ('grandparent', '75-79', 'male', 'high', 'low', 'unaware'),
    ]
    
    for scam, age, gender, trust, tech, awareness in test_cases:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a protective AI assistant helping an elderly person on a scam phone call. "
                    "Generate simple, reassuring explanations. Use format: [WHAT] [WHY] [REASSURE] [DO]"
                )
            },
            {
                "role": "user",
                "content": f"Scam: {scam}, Victim: {age} {gender}, {trust} trust, {tech} tech, {awareness} awareness"
            }
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        print(f"\n--- Scam: {scam}, Profile: {trust} trust, {tech} tech ---")
        print(response[:500])

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    # Check for HF token
    try:
        AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print("ERROR: Could not load LLaMA-3 model.")
        print("Make sure you have:")
        print("  1. huggingface-cli login (with a valid token)")
        print("  2. Access to meta-llama/Meta-Llama-3.1-8B-Instruct")
        print(f"\nOriginal error: {e}")
        exit(1)
    
    # Train
    model, tokenizer = train()
    
    # Test
    test_generation(model, tokenizer)
    
    print("\nLLaMA-3 training complete!")
    print(f"Model saved to: {OUTPUT_DIR}/lora_adapter/")