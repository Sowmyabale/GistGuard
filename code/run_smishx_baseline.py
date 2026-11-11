"""SOTA reproduction: SmishX (Wang et al., SOUPS 2025) detection prompt + user-friendly
explanation, applied to COVA-X multi-turn conversations (adapted — no URL/screenshot stage,
since call transcripts contain no URLs; disclosed in paper).

Outputs are saved in OUR gist format (warning/action) so eval_gist_quality.py scores them
with the SAME automatic metrics as our variants (qc%, ask%, id%, uniq%, distinct-2, FK).

Two backends:
  --backend openai   : faithful reproduction via GPT-4o (needs OPENAI_API_KEY env; ~2 calls/conv)
  --backend local    : same prompts, local Llama-3.1-8B-Instruct (unsloth) on GPU — free,
                       disclosed as 'SmishX prompts with Llama-3.1-8B'

Then score identically to ours:
  python eval_gist_quality.py --dataset_dir ".../COVA-X_Split" \
      --files results_v1rerun/test_gists_smishx.json --labels smishx \
      --output results_v1rerun/gist_quality_smishx.json
"""
import argparse, json, os, re, sys

# ---- SmishX original prompts (verbatim from github.com/yizhu-joy/SmishX main.py) ----
DETECTION_PROMPT = """I want you to act as a spam detector to determine whether a given SMS is phishing, spam, or legitimate. Your analysis should be thorough and evidence-based. Analyze the SMS by following these steps:
1. If the SMS is promoting any of the following categories: Online gambling, bets, spins, adult content, digital currency, lottery, it is either spam or phishing.
2. The SMS is legitimate if it is from known organizations, such as appointment reminders, OTP (One-Time Password) verification, delivery notifications, account updates, tracking information, or other expected messages.
3. The SMS is considered legitimate if it involves a conversation between friends, family members, or colleagues.
4. Promotions and advertisements are considered spam.
5. The SMS is phishing if it is fraudulent and attempts to deceive recipients into providing sensitive information. Phishing SMS may exhibit: fake prizes/rewards, urgent or alarming language, requests for personal information, grammatical errors.
8. Please give your rationales before making a decision. And your output should be in json format and should not have any other output:
- brand_impersonated: brand name associated with the SMS, if applicable.
- rationales: detailed rationales for the determination, up to 500 words.
- brief_reason: brief reason for the determination.
- category: True or False. If the SMS is legitimate, output False. Else, output True.
- advice: If the SMS is phishing, output potential risk and your advice for the recipients.

Below is the information of the SMS:"""

USER_FRIENDLY_PROMPT = """Based on the detailed analysis, I want you to create a simple and easy-to-understand response to tell the user whether the text message is a phishing attempt or a legitimate message. Use plain language and avoid technical terms. Explain your conclusion in 3 sentences, focusing on whether the message seems suspicious or safe. Provide a simple reason to support your conclusion, including clear evidence. The response should be reassuring and concise, easy for anyone to understand."""


def conv_text(r):
    t = r.get('text')
    if t:
        return t
    return f"[scam phone call transcript — predicted type: {r.get('pred_scam','unknown')}]"


def parse_json_loose(s):
    s = s.replace('```json', '').replace('```', '').strip()
    m = re.search(r'\{.*\}', s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class OpenAIBackend:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()

    def chat(self, prompt, json_mode=False):
        kw = dict(model='gpt-4o',
                  messages=[{'role': 'user', 'content': prompt}],
                  temperature=0.0)
        if json_mode:
            kw['response_format'] = {'type': 'json_object'}
        r = self.client.chat.completions.create(**kw)
        return r.choices[0].message.content


class LocalBackend:
    def __init__(self, model_name='unsloth/Meta-Llama-3.1-8B-Instruct', max_len=4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map='cuda')
        self.model.eval()
        self.max_len = max_len

    def chat(self, prompt, json_mode=False, max_new=400):
        import torch
        msgs = [{'role': 'user', 'content': prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        enc = self.tok(text, return_tensors='pt', truncation=True,
                       max_length=self.max_len).to('cuda')
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        return self.tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--backend', choices=['openai', 'local'], default='local')
    ap.add_argument('--model', default='unsloth/Meta-Llama-3.1-8B-Instruct')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding='utf-8')]
    if args.limit:
        rows = rows[:args.limit]

    done = {}
    if args.resume and os.path.exists(args.output):
        try:
            for g in json.load(open(args.output, encoding='utf-8')):
                done[g['conversation_id']] = g
        except Exception:
            done = {}
    todo = [r for r in rows if r['conversation_id'] not in done]
    print(f'total={len(rows)} done={len(done)} todo={len(todo)} backend={args.backend}')

    backend = OpenAIBackend() if args.backend == 'openai' else LocalBackend(args.model)
    results = list(done.values())

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        for r in batch:
            text = conv_text(r)
            det = None
            for attempt in range(2):
                out = backend.chat(DETECTION_PROMPT + f'\n- SMS to be analyzed: {text}',
                                   json_mode=True)
                det = parse_json_loose(out)
                if det and 'category' in det:
                    break
            det = det or {'category': True, 'brief_reason': 'parse failed', 'advice': ''}
            expl = backend.chat(USER_FRIENDLY_PROMPT +
                                f"\nThe SMS message: {text}\nThe analysis result: {det}")
            expl = (expl or '').strip().strip('"')
            results.append({
                'conversation_id': r['conversation_id'],
                'true_scam': r.get('true_scam'), 'pred_scam': r.get('pred_scam'),
                'variant': 'smishx',
                'warning': expl if expl else det.get('brief_reason', ''),
                'action': det.get('advice', ''),
                'smishx_category': det.get('category'),
                'valid': bool(expl),
            })
        json.dump(results, open(args.output, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print(f'  [{len(results)}/{len(rows)} done]', flush=True)

    print(f'\nSmishX reproduction done: {len(results)} | saved -> {args.output}')
    print('Next: score with eval_gist_quality.py --files', args.output, '--labels smishx')


if __name__ == '__main__':
    main()
