# Run Manifest — GistGuard

Every number reported in the paper maps to the configuration, input split,
and output artifact listed here. Evaluation artifacts are recomputed with the
corrected word-boundary coverage scorer (`coverage_overall_v2.json`,
`gist_quality_summary_v2.json`); the original scorer outputs are kept with a
`_v1` suffix where preserved. Detection metrics are recomputed from saved
predictions, not retrained.

## Layer 1 — multi-task scam-type + tactic detection

| Item | Value |
|---|---|
| Script | `code/train_layer1_multitask.py` |
| Encoder | `roberta-base` (config `MODEL_NAME`), max length 512 |
| Heads | 8-class scam type + 6-class category-derived tactic |
| Splits | train 6,591 / val 2,197 / test 2,197 (evidence: `results/results_v1rerun/silver*.jsonl` row counts + logs) |
| Checkpoint | `models/layer1_multitask` (weights on request) |
| Predictions | `results/results_v1rerun/test_predictions.jsonl` |
| Metrics | `results/results_v1rerun/test_metrics.json` — scam acc 99.86%, macro F1 99.87%; Layer 2 trust/tech/aware acc 87.98/87.89/89.30% |
| Confusion matrix | `results/results_v1rerun/confusion_matrix_test.png`, `confidence_distribution.png`, `per_class_performance.png` |

## Tactic annotations (independent of scam-type derivation)

| Item | Value |
|---|---|
| Script | `code/annotate_tactics_llm.py` (Llama-3.1-8B, 7 categories incl. `none`) |
| Output | `results/results_v1rerun/tactic_annotations.jsonl` (10,985 conversations) |
| Distribution | authority_impersonation 3,892 / fear_threat 2,267 / over_familiarity 1,956 / isolation_secrecy 1,150 / reciprocity_gift 1,010 / none 382 / social_proof 88 |
| Agreement with category-derived labels | 52.8% (reported honestly in the response letter) |

## Layer 3 — gist variants (all on the same test predictions)

| Variant | Training data | Train script / log | Generation script | Output |
|---|---|---|---|---|
| Prompt-LM | none (untuned, few-shot prompted) | — | `code/generate_gists_v3.py` | `results/results_v1rerun/test_gists_v3.json` |
| SFT-v1 | `silver_train.silver.jsonl` (6,578) | `code/train_gist_lora.py` / `lora_train.log` | `code/generate_gists_lora.py` | `test_gists_lora.json` |
| SFT-v2 | `silver_v2.silver.jsonl` (4,821; tactic annotations + evidence cards) | `make_silver_v2.py` / `silver_v2.log` + `lora_v4_train.log` | `generate_gists_lora.py --adapter models/l3_gist_lora_v4` | `test_gists_lora_v4.json` |
| SFT-v2s (GistGuard) | `silver_v2_short.silver.jsonl` (2,833 = FK<=10 subset of SFT-v2 data; filter command in `silver_v2_short` note below) | `lora_v4b_train.log` | `generate_gists_lora.py --adapter models/l3_gist_lora_v4b` | `test_gists_lora_v4b.json` |
| Template | none (deterministic) | — | `code/template_baseline.py` | `test_gists_template.json` |
| SmishX (adapted) | none | — | `code/run_smishx_baseline.py` (verbatim SmishX prompts) | `test_gists_smishx_fixed.json` |
| Ghafariasl (adapted) | none | — | `code/run_ghafariasl_baseline.py` (their risk-assessment formulation) | `test_gists_ghafariasl.json` |

`silver_v2_short` filter (as run, 2026-10-18) — keeps examples whose training
warning has Flesch-Kincaid grade <= 10:
```
python -c "import json,sys; sys.path.insert(0,'.'); from eval_gist_quality import fk_grade; \
rows=[json.loads(l) for l in open('results_v1rerun/silver_v2.silver.jsonl')]; \
keep=[r for r in rows if fk_grade(r['warning'])<=10.0]; \
f=open('results_v1rerun/silver_v2_short.silver.jsonl','w'); \
[f.write(json.dumps(r,ensure_ascii=False)+'\n') for r in keep]; f.close(); \
print(len(rows),'->',len(keep))"
```

## Headline evaluation artifacts (corrected scorer)

| Metric | Artifact |
|---|---|
| Conditional ask/id coverage, all variants | `results/results_v1rerun/coverage_overall_v3.json` |
| Paired bootstrap CIs for the GistGuard vs equal-evidence Prompt-LM difference | `results/results_v1rerun/paired_bootstrap_ci.json` — paired bootstrap over conversation IDs (10,000 replicates, seed 20261112); ask diff +7.2 pp, 95% CI [4.4, 9.9] (n=1,556); identity diff +2.5 pp, 95% CI [1.0, 4.2] (n=1,457). Point estimates reproduce the headline conditional coverage exactly (73.2/93.5 vs 66.0/91.0). Script: `code/paired_bootstrap.py` |
| Per-scam quality table (qc, uniq, distinct-2, self-sim, FK) | `results/results_v1rerun/gist_quality_summary_v3.json` |
| Scorer regression tests | `code/tests/test_coverage_scorer.py` (24/24) |
| Evidence cards | `results/results_v1rerun/evidence_cards.jsonl` |
| Human evaluation (pilot, n=1) | `results/human_eval_ratings_item_level.csv` + `results/human_eval_pilot_summary.json` (see `code/human_eval_protocol.md`, pilot section) |
| Evidence-extractor audit (100-row annotation) | `results/extractor_audit_summary.json` — sampled 96 conversations: identity precision 92.4% / recall 66.3%, requested items precision 80.0% / recall 69.6%; adversarial 4 cases: identity TP 2 (two cases contain a claimed role — "bank fraud team", "engineer" — recovered correctly), identity FP 0, item FP 2; aggregate 100 rows: identity 92.6% / 67.0%, items 78.7% / 69.6%. Provenance: annotated by one of the authors (graduate researcher) on a structured sheet with candidate extractions pre-filled by the rule-based extractor; no AI system assigned the audit labels. Reference-identity presence (a stated personal name or organization) is recorded per row and the confusion counts derive from it (updated 2026-11-11, round 3): a wrong extraction on a reference-present row is both a false positive and a false negative; a false identity on a reference-absent row is a false positive only, so the adversarial rows depress no recall rate. Under the previous definitions, missed-only recall was 70.8% and round-2 recall (wrong counts as FN unconditionally) was 67.0% on the aggregate. Judgments released without conversation text: `results/extractor_audit_annotations_deidentified.csv` (96 sampled rows + 4 adversarial rows, with the `reference_identity_present` flag). Scoring script: `code/score_audit.py` |
| Profile-source comparison (supplied / predicted / absent) | `results/results_v1rerun/coverage_profiles_ablation.json` (73.2/93.5, 98.0/97.6, 97.7/97.7 conditional coverage; per-scam detail in `gist_quality_ablation_profiles.json`) |

## Label provenance (which checkpoint fed which experiment)

All headline warning experiments consume Layer-1 outputs from the
**original** multi-task checkpoint (`models/layer1_multitask`, saved
predictions `results/results_v1rerun/test_predictions.jsonl`). The
**retrained** seven-class tactic checkpoint
(`models/layer1_multitask_tacticv2`, `tactic_v2_summary.json`: scam acc
99.77 / macro F1 99.79; tactic top-1/2/3 62.5/79.3/89.2 vs LLM
annotations) is evaluated on detection metrics only; no warning output was
regenerated with its predictions. The manuscript reports the two
checkpoints' metrics separately (Section "Scam-Type and Tactic Detection").

## Readability-metric validation

The FK grade reported in our quality tables is computed by a self-contained
heuristic in `code/eval_gist_quality.py` (`fk_grade`), so results are
reproducible without optional dependencies. Validated against the
`textstat` reference implementation (`flesch_kincaid_grade`) on a random
sample of 200 GistGuard test warnings (seed 42): mean FK 7.34 (ours) vs
7.66 (textstat), MAE 0.43 grade levels, 199/200 (99.5%) within 1.5 grade
levels, max deviation 2.15. Conclusion: the two implementations agree
closely on this sample. All systems in the paper are scored with the same
implementation, so cross-system comparisons on this metric are internally
consistent; the validation does not by itself establish that absolute
textstat-based grades would rank systems identically.

## Detection baselines

| Baseline | Artifact |
|---|---|
| TF-IDF + logistic regression | `results/results_baselines/tfidf_baselines_summary.json` (99.73% acc — the paper's 97.9% row was an earlier run and is corrected to this artifact) |
| Fine-tuned BERT | `results/results_v1rerun/bert_baseline_summary.json` |
| v2 / v3-hybrid stacks | `results/results_v2/v2_eval_summary.json`, `results/results_v3hybrid/v3_hybrid_summary.json`, `results/results_v3hybrid_tactics/v3_tactics_summary.json` |

## Known artifact-vs-manuscript corrections applied in the revision

1. RoBERTa macro F1: manuscript 99.6% -> artifact 99.87% (`test_metrics.json`).
2. Train/val/test counts: manuscript 6,588/2,200/2,197 -> artifact 6,591/2,197/2,197.
3. LoRA LR 2e-4 with cosine schedule + warmup (script defaults/logs), not 1e-4 linear.
4. Tactic head was originally 6 category-derived outputs; per review we retrained a 7-class head on LLM tactic annotations (`models/layer1_multitask_tacticv2`, `results/results_v1rerun/tactic_v2_summary.json`): top-1 62.5% / top-2 79.3% / top-3 89.2% vs annotations (the old head's 53.1/62.2/74.0 was vs category-derived labels).
5. All ask/id coverage numbers recomputed with the corrected scorer (see table above).
6. Per-scam coverage fields in `gist_quality_all_sota.json` and
   `gist_quality_ablation_profiles.json` were regenerated 2026-11-11 with
   the corrected *conditional* scorer (denominator = conversations where the
   extractor found the field). Pre-2026-11-11 those fields came from an
   unconditional, vacuous-true scorer and displayed 100% throughout; they
   are superseded (see `_coverage_note` inside each file).
7. Table 3 numbers corrected to `tfidf_baselines_summary.json` /
   computed values: TF-IDF logistic-regression macro F1 99.75 (accuracy
   99.73); majority-class macro F1 3.7 (accuracy 17.6).
8. Detector checkpoints reported separately: original multi-task model
   (scam acc 99.86 / macro F1 99.87, `test_metrics.json`) vs retrained
   seven-class tactic model (scam acc 99.77 / macro F1 99.79,
   `tactic_v2_summary.json`); warning experiments all use the original
   model's saved predictions (see "Label provenance" above).
