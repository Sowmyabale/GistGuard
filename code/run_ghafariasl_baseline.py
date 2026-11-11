"""SOTA reproduction (zero-shot adaptation): Ghafariasl et al. 2026
"Incremental Risk Assessment of Progressive Elder Financial Scams via
Instruction-Tuned Small Language Models" (Kansas State, arXiv:2609.00005).

Their formulation: for a multi-turn scam dialogue, the model produces
  (1) qualitative risk level   (2) continuous risk score
  (3) explanatory rationale    (4) safety recommendation

Their best results come from FINE-TUNED Phi-4 / LLaMA-3.2 on their own dataset
(investment/charity/tech-support, 2-8 turns). Their dataset and trained weights are
not public, so a full reproduction is not possible. What we reproduce here is a
ZERO-SHOT adaptation of their output formulation on COVA-X, using the same local
Llama-3.1-8B-Instruct used throughout our study. Paper disclosure:
"zero-shot adaptation of the Ghafariasl et al. (2026) formulation; fine-tuned-SLM
reproduction was not possible as their dataset/weights are unavailable."

For comparison with our gists: warning = explanatory rationale, action = safety recommendation.
Scored with the SAME eval_gist_quality.py metrics.

Then:
  python eval_gist_quality.py --dataset_dir ".../COVA-X_Split" \
      --files results_v1rerun/test_gists_ghafariasl.json --labels ghafariasl_zs \
      --output results_v1rerun/gist_quality_ghafariasl.json
"""
import argparse, json, os, re

GHAFARIASL_PROMPT = """You are an elder-fraud risk assessment system. Analyze the following multi-turn scam conversation and produce a structured risk assessment in JSON format with exactly these four fields (following the cumulative risk-assessment formulation of Ghafariasl et al. 2026):

- risk_level: one of "low", "medium", "high", "critical"
- risk_score: a continuous number from 0 to 100 representing the cumulative scam risk of the full conversation
- rationale: an explanatory rationale (2-3 sentences, plain language) explaining WHY this conversation is risky for an older adult — name the manipulation technique and what the scammer wants
- safety_recommendation: one concrete safety recommendation (1-2 sentences) telling the older adult exactly what to do right now

Output ONLY valid JSON, no other text.

Conversation:
"""


def conv_text(r):
    return r.get('text') or f"[phone call transcript, predicted scam: {r.get('pred_scam','unknown')}]"


def parse_json_loose(s):
    s = s.replace('```json', '').replace('```', '').strip()
    m = re.search(r'\{.*\}', s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


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

    def chat(self, prompt, max_new=350):
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
    print(f'total={len(rows)} done={len(done)} todo={len(todo)}')

    backend = LocalBackend(args.model)
    results = list(done.values())

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        for r in batch:
            out = None
            for _ in range(2):
                raw = backend.chat(GHAFARIASL_PROMPT + conv_text(r))
                out = parse_json_loose(raw)
                if out and 'rationale' in out:
                    break
            out = out or {'risk_level': 'unknown', 'risk_score': None,
                          'rationale': '', 'safety_recommendation': ''}
            results.append({
                'conversation_id': r['conversation_id'],
                'true_scam': r.get('true_scam'), 'pred_scam': r.get('pred_scam'),
                'variant': 'ghafariasl_zs',
                'warning': out.get('rationale', ''),
                'action': out.get('safety_recommendation', ''),
                'risk_level': out.get('risk_level'),
                'risk_score': out.get('risk_score'),
                'valid': bool(out.get('rationale')),
            })
        json.dump(results, open(args.output, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print(f'  [{len(results)}/{len(rows)} done]', flush=True)

    print(f'\nGhafariasl zero-shot adaptation done: {len(results)} | saved -> {args.output}')
    print('Next: python eval_gist_quality.py --files', args.output,
          '--labels ghafariasl_zs --output results_v1rerun/gist_quality_ghafariasl.json')


if __name__ == '__main__':
    main()
