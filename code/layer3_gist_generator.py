"""
Layer 3: Gist-Based Explanation Generator
==========================================
Generates bottom-line-meaning explanations adapted to victim's ToM profile.
Approach: Few-shot prompting with curated examples.
Falls back to rule-based selection when transformers unavailable.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class GistConfig:
    max_words: int = 50
    use_action_items: bool = True
    avoid_technical: bool = True
    tone: str = "calm_warning"

class GistGenerator:
    # Model candidates in priority order (official repo may be gated; mirrors are not)
    LLAMA_CANDIDATES = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
    ]
    MISTRAL_CANDIDATES = ["mistralai/Mistral-7B-Instruct-v0.3"]

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.backend = "dialogpt"
        if model_name:
            m = model_name.lower()
            if "llama" in m:
                self.backend = "llama"
            elif "mistral" in m:
                self.backend = "mistral"
        self.tokenizer = None
        self.model = None
        self._init_model()
        self.few_shot_examples = self._build_example_bank()

    def _resolve_candidates(self):
        m = self.model_name.lower()
        if "llama" in m:
            return self.LLAMA_CANDIDATES
        if "mistral" in m:
            return self.MISTRAL_CANDIDATES
        return [self.model_name]

    def _init_model(self):
        if self.model_name is None:
            print("  [Layer 3] No model specified — rule-based gist generation.")
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"  [Layer 3] Gist model backend={self.backend}, device: {self.device}")

            last_err = None
            for candidate in self._resolve_candidates():
                try:
                    print(f"  [Layer 3] Loading {candidate} ...")
                    self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                    if self.backend in ("llama", "mistral"):
                        # 4-bit quantization: ~6 GB VRAM for an 8B model
                        from transformers import BitsAndBytesConfig
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            candidate,
                            quantization_config=bnb_config,
                            device_map="cuda" if self.device == "cuda" else None,
                        )
                        self.device = self.model.device
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(candidate)
                        self.model = self.model.to(self.device)
                    self.model.eval()
                    self.model_name = candidate
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    print(f"  [Layer 3] Loaded {candidate}")
                    return
                except Exception as e:
                    last_err = e
                    print(f"  [Layer 3] Failed to load {candidate}: {e}")
            raise last_err
        except Exception as e:
            print(f"Warning: Could not load gist model: {e}")
            print("Falling back to rule-based gist generation.")
            self.model = None

    def _build_example_bank(self) -> Dict[str, List[Dict]]:
        return {
            "urgency_pressure": [
                {"context": "Victim is confused, low tech comfort",
                 "gist": "This person is trying to rush you. Real companies give you time to decide. Take a day to think it over.",
                 "action": "Hang up. Call the company back using the number on their official website."},
                {"context": "Victim is highly trusting",
                 "gist": "The deadline they mention is fake. It is a trick to stop you from thinking. There is no real emergency.",
                 "action": "Do not send money. Wait and talk to a family member first."},
            ],
            "authority_impersonation": [
                {"context": "Victim is fearful, low awareness",
                 "gist": "Your bank or the government will never call and ask for your password or card number over the phone. This is a fake caller.",
                 "action": "Hang up. Call your bank using the number on your card or statement."},
                {"context": "Victim is confused",
                 "gist": "Real government offices send letters first. They do not threaten arrest over the phone. This caller is pretending.",
                 "action": "Do not give any information. Tell them you will call the office directly."},
            ],
            "fear_threat": [
                {"context": "Victim is scared, high susceptibility",
                 "gist": "The threats are empty. No one can freeze your account or arrest you over the phone without official papers. They are trying to scare you into paying.",
                 "action": "Hang up immediately. Call your bank or the police using the official number."},
                {"context": "Victim is uncertain",
                 "gist": "If there was a real problem with your account, you would get a letter in the mail. Phone threats are scams.",
                 "action": "Do not press any numbers. Hang up and check your account online or at a branch."},
            ],
            "social_proof": [
                {"context": "Victim is excited",
                 "gist": "Just because others joined does not mean it is safe. Scammers pay people to say good things. The reviews are fake.",
                 "action": "Do not invest. Search for the company name plus 'scam' online before deciding."},
            ],
            "reciprocity_gift": [
                {"context": "Victim is trusting",
                 "gist": "You did not enter any contest, so you cannot win. Free prizes are a trick to get your personal information or money.",
                 "action": "Do not pay any fee to collect a prize. Real prizes do not require payment."},
            ],
            "request_sensitive_info": [
                {"context": "Victim is confused",
                 "gist": "Your bank, Medicare, and the IRS already have your information. They never call to ask for it. This caller is a thief.",
                 "action": "Never give your Social Security number, card number, or password over the phone. Hang up."},
            ],
            "isolation_secrecy": [
                {"context": "Victim is trusting family caller",
                 "gist": "Real family in trouble will not ask you to keep secrets. They want you to tell other family members. Secrecy is a warning sign.",
                 "action": "Call your grandchild or their parent directly on a number you know to check if it is true."},
            ],
            "over_familiarity": [
                {"context": "Victim is emotional, grandparent scam",
                 "gist": "This is not your grandchild. Scammers can sound like young people. Your real grandchild would call their parents first, not ask you for secret money.",
                 "action": "Hang up. Call your grandchild on their regular phone number to check. Never wire money in secret."},
            ],
            "none": [
                {"context": "No scam detected",
                 "gist": "No warning signs detected in this message. It appears to be a normal conversation.",
                 "action": "Continue as usual. Stay alert for unexpected requests for money or personal information."},
            ],
        }

    def _get_gist_config(self, victim_tom: Dict, tech_comfort: int = 3) -> GistConfig:
        cognitive_load = victim_tom.get("cognitive_load", "medium")
        emotional_state = victim_tom.get("emotional_state", "neutral")

        if cognitive_load == "high" or emotional_state in ["fear", "confusion"]:
            max_words = 25
            tone = "reassuring"
        elif cognitive_load == "medium":
            max_words = 50
            tone = "calm_warning"
        else:
            max_words = 75
            tone = "calm_warning"

        avoid_technical = tech_comfort < 3
        return GistConfig(max_words=max_words, use_action_items=True, avoid_technical=avoid_technical, tone=tone)

    def generate(self, layer1_result: Dict, tom_result: Dict, tech_comfort: int = 3) -> Dict:
        primary_tactic = layer1_result.get("primary_tactic", "none")
        victim_tom = tom_result.get("victim_tom", {})
        scammer_tom = tom_result.get("scammer_tom", {})
        config = self._get_gist_config(victim_tom, tech_comfort)

        if self.model is None:
            return self._rule_based_gist(primary_tactic, victim_tom, scammer_tom, config)

        if self.backend in ("llama", "mistral"):
            try:
                return self._generate_llama(primary_tactic, victim_tom, config)
            except Exception as e:
                if not getattr(self, "_llama_err_logged", False):
                    self._llama_err_logged = True
                    import traceback
                    print("  [Layer 3] FIRST LLM ERROR (full traceback):")
                    traceback.print_exc()
                return self._rule_based_gist(primary_tactic, victim_tom, scammer_tom, config)

        prompt = self._build_prompt(primary_tactic, victim_tom, scammer_tom, config)
        try:
            gist_text = self._generate_with_model(prompt, config)
            if not gist_text or not gist_text.strip():
                raise ValueError("Model returned empty gist")
            return {
                "gist": gist_text,
                "action_item": self._extract_action(gist_text),
                "complexity": config.max_words,
                "method": "few_shot_llm",
            }
        except Exception as e:
            return self._rule_based_gist(primary_tactic, victim_tom, scammer_tom, config)

    def _generate_llama(self, tactic: str, victim_tom: Dict, config: GistConfig) -> Dict:
        """Chat-template generation with an instruction-tuned LLM (Llama-3.x / Mistral)."""
        import torch

        emotional_state = victim_tom.get("emotional_state", "neutral")

        # ZERO-SHOT: no curated examples — the LLM writes the gist itself.
        # (The example bank is only used by the rule-based fallback method.)
        user_msg = (
            f"An elderly person is receiving a scam attempt that uses the tactic: "
            f"{tactic.replace('_', ' ')}.\n"
            f"The person is currently feeling: {emotional_state}.\n\n"
            f"Give them the BOTTOM-LINE MEANING of what is happening, in plain, calm, "
            f"simple language — no technical terms, no jargon, no lecturing. "
            f"Focus on what the scammer is really trying to do.\n"
            f"Then give one concrete action item.\n\n"
            f"Reply in EXACTLY this format:\n"
            f"GIST: <one or two short sentences, under {config.max_words} words>\n"
            f"ACTION: <one short, concrete instruction>"
        )
        system_msg = ("You are a protective assistant that helps elderly people avoid scams. "
                      "You write very short, plain-language, actionable warnings.")

        # Build the Llama-3 chat prompt directly (no apply_chat_template dependency)
        prompt_text = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        gist, action = self._parse_llama_output(text)
        if not gist:
            raise ValueError("empty gist after parsing")
        return {
            "gist": gist,
            "action_item": action or self._extract_action(gist),
            "complexity": config.max_words,
            "method": f"{self.backend}_4bit",
        }

    @staticmethod
    def _parse_llama_output(text: str):
        gist, action = "", ""
        for line in text.splitlines():
            line = line.strip()
            low = line.lower()
            if low.startswith("gist:"):
                gist = line[5:].strip().strip('"')
            elif low.startswith("action:"):
                action = line[7:].strip().strip('"')
        if not gist:
            # fallback: first non-empty line
            for line in text.splitlines():
                if line.strip():
                    gist = line.strip()
                    break
        return gist, action

    def _rule_based_gist(self, tactic: str, victim_tom: Dict, scammer_tom: Dict, config: GistConfig) -> Dict:
        examples = self.few_shot_examples.get(tactic, self.few_shot_examples["none"])
        emotional_state = victim_tom.get("emotional_state", "neutral")
        best_example = examples[0]
        for ex in examples:
            if emotional_state in ex.get("context", "").lower():
                best_example = ex
                break

        gist = best_example["gist"]
        action = best_example["action"]

        if config.max_words <= 25:
            gist = self._shorten(gist, 20)
            action = self._shorten(action, 15)

        return {
            "gist": gist,
            "action_item": action,
            "complexity": config.max_words,
            "method": "few_shot_rule",
        }

    def _build_prompt(self, tactic: str, victim_tom: Dict, scammer_tom: Dict, config: GistConfig) -> str:
        examples = self.few_shot_examples.get(tactic, self.few_shot_examples["none"])[:2]
        example_text = ""
        for i, ex in enumerate(examples, 1):
            example_text += f"\nExample {i}:\n  Situation: {ex['context']}\n  Gist: {ex['gist']}\n  Action: {ex['action']}\n"

        emotional_state = victim_tom.get("emotional_state", "neutral")
        prompt = f"""You are a protective assistant helping an elderly person avoid scams.
Your job is to give the BOTTOM-LINE MEANING of a suspicious message in plain language.
Be brief, calm, and helpful. Do not explain what a gist is. Do not use technical terms.

The person is feeling: {emotional_state}.
The scam attempt uses: {tactic.replace('_', ' ')}.
{example_text}
Now write a short warning and action for this person. Keep it under {config.max_words} words.

Gist:"""
        return prompt

    def _generate_with_model(self, prompt: str, config: GistConfig) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_words + 20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        gist = generated[len(prompt):].strip()
        gist = gist.split("\n")[0].strip()
        return gist

    def _extract_action(self, text: str) -> str:
        action_markers = ["hang up", "call", "do not", "never", "check", "wait", "tell"]
        sentences = text.split(".")
        for sent in sentences:
            if any(m in sent.lower() for m in action_markers):
                return sent.strip()
        return "Be cautious and verify before taking any action."

    def _shorten(self, text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "."

