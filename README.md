# GistGuard: Gist-First Explanations for Scam-Call Protection of Older Adults

Three-layer framework for protecting older adults from telephone scams:
**Layer 1** performs multi-task scam-type classification and manipulation-tactic
detection; **Layer 2** extracts conversation evidence (what the caller requested,
who they claimed to be) and maintains a conversation-state profile; **Layer 3**
generates a short, plain-language "gist" warning and one concrete action via
LoRA fine-tuning on quality-filtered silver data.

Companion paper: *Gist-First Explanations for Scam-Call Protection of Older
Adults* (S. Bale, H. K. Imandi, S. Kshirsagar, Wichita State University).

## Repository layout

```
code/     All experiment and pipeline scripts (Python 3.11, PyTorch + transformers)
results/  All experimental outputs: predictions, generated gists, quality
          evaluations, figures, and training/inference logs, organized by run
```

## Key results (COVA-X test fold, n = 2,197)

Coverage metrics use the corrected scorer (`coverage_overall_v3`,
`gist_quality_summary_v3`; category-specific identity aliases, plural-aware
word-boundary item matching, identity-fragment filtering; regression tests in
`code/tests/` — 24/24). ask/id coverage is conditional (conversations in which
the extractor found the field; denominators 1,556 and 1,457). fk =
Flesch-Kincaid grade level (lower is easier).

| Layer 3 variant | Format-valid | FK grade | ask cov. | id cov. | unique |
|---|---|---|---|---|---|
| Prompt-LM (untuned, few-shot prompted) | 99.8% | 5.5 | 32.8% | 60.9% | 73.6% |
| SFT-v2s (GistGuard) | 99.4% | 7.0 | 73.2% | 93.5% | 59.9% |
| Deterministic template | 100% | 4.8 | 100% | 100% | 8.8% |
| SmishX (adapted prompts) | 0.0% | 12.4 | 71.7% | 49.8% | 100% |
| Ghafariasl et al. (adapted formulation) | 0.0% | 15.2 | 65.4% | 55.3% | 98.1% |

The template covers every extracted field by construction (it repeats them);
its uniqueness (8.8%) shows the cost: near-identical, non-personalized
warnings. GistGuard's contribution is personalized, diverse, plain-language
warnings (human evaluation reported in the paper). Detection: 99.9% accuracy /
99.87% macro F1 on eight-class scam-type classification (recomputed from saved
predictions). Category-derived tactic top-3 accuracy is 100% against
scam-type-derived labels; against independently LLM-annotated per-conversation
tactics the same predictions achieve 53.1% / 62.2% / 74.0% (top-1/2/3) — a
retrained 7-class tactic head evaluated against these annotations is reported
in the revision (`tactic_v2_summary.json`, reproduced by
`train_layer1_tactic_v2.py`).

## Data

Experiments use the COVA-X dataset (Lochstampfor & Roy, 2026, arXiv:2606.06879),
accessed under its terms for non-commercial defensive research. **The dataset is
not redistributed in this repository**; request access at
https://scamlingua.org/ and cite the dataset in any derivative work.
Result files contain only conversation IDs, predictions, generated warnings,
and metrics — never conversation text. Scripts that need the raw text (e.g.
the external-baseline runs) take a `--dataset_dir` argument or rebuild the
joined file locally via `code/rebuild_predictions_with_text.py`.

A supplementary evaluation uses 243 public YouTube scam-call transcripts
(unlabeled; provenance documented in the paper).

## Not included

- Trained model weights (~3.2 GB) — available on request from the authors.
- The COVA-X dataset — access via https://scamlingua.org/ (redistribution not
  permitted by the dataset terms).

## Requirements

Python 3.11, PyTorch, transformers, peft, scikit-learn, pandas. See
`code/requirements_v1rerun.txt` and `code/requirements.txt`.

## Citation

```bibtex
@misc{gistguard2026,
  author = {Bale, Sowmya and Imandi, Hemanth Kumar and Kshirsagar, Shruti},
  title  = {GistGuard: Gist-First Explanations for Scam-Call Protection of Older Adults},
  year   = {2026},
  note   = {Wichita State University}
}
```
