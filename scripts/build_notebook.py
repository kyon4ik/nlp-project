"""Build the primary research notebook; outputs are populated by execute_notebook.py."""
from pathlib import Path
import argparse
import nbformat as nbf

cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

md("""
# Scientific Claim Verification Using Dense Retrieval, Evidence Extraction, and Natural Language Inference

**NLP semester project · two-person team · reproducible experimental report**

## Abstract

We study scientific claim verification on the official SciFact corpus using lexical retrieval, pretrained
sentence embeddings, task-specific contrastive fine-tuning, hybrid rank fusion, and neural reranking.
A trained sentence selector and binary stance classifier provide evidence and SUPPORT/CONTRADICT predictions;
the pipeline abstains when relevance filters reject all evidence. Oracle interventions diagnose retrieval
and evidence-selection losses. This notebook executes each stage and reports measured results, including
the unmodified official SciFact metrics. The supplied run uses the complete data with MiniLM retrieval and SciBERT on GPU;
larger GPU and repeated-seed configurations are available. Smoke experiments are explicitly excluded from findings.

Outputs below are produced by code, never inserted illustrative scores. The final discussion is generated
from the saved evaluation artifacts so it stays consistent with the selected execution configuration.
""")
md("""
## 1. Configuration and reproducibility

Run from the repository root. Locally, create the virtual environment described in README.md and select its
kernel. In Colab, upload/clone this complete directory, change to its root, and install requirements first
(`%pip install -r requirements.txt`), then restart the runtime. GPU detection is automatic.

`MODE='full'` is the intended research path. `smoke` checks the whole workflow using 40 training claims,
16 validation claims, 16 dev claims, and a reduced corpus. `extended` allows longer training; run
`scripts/run_seeded.py` for three seeds on the same split. A run directory is immutable with respect to its
configuration: choose a new directory when changing hyperparameters. Re-execution reuses checkpoints and caches.
""")
code("""
from pathlib import Path
import os, sys, json
assert Path('scifact_project').is_dir(), 'Change working directory to the project root first.'
from scifact_project.config import Config
MODE = 'full'
PROFILE = 'auto'                 # cpu uses MiniLM pair models; gpu uses SciBERT
SEED = 42
config = Config.for_mode(MODE, PROFILE, SEED).prepare()
from scifact_project.experiment import Study
study = Study(config)
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image
pd.set_option('display.max_columns', 16)
display(pd.DataFrame([config.dictionary()]).T.rename(columns={0: 'value'}))
display(Markdown(f'Actual device: **{config.device}**. Mode: **{MODE}**. Results: `{config.output_dir}`.'))
""")
md("""
## 2. Problem and research questions

For claim $q$ and a corpus of abstracts, produce evidence-bearing documents, sentence indices, and
document-level stances. A claim verdict is aggregated separately; absence of accepted evidence gives
NOT ENOUGH INFO. Scientific support is relative to the corpus and annotations, not a claim of universal truth.

The main question is how retrieval choices change scientific evidence retrieval and final verification.
We hypothesize that lexical and dense rankings complement each other, task supervision helps pretrained
embeddings, hard negatives improve early precision, and reranking improves small-K ranking. These are hypotheses,
not presumed findings. We also ask whether evidence selection or stance is the dominant downstream bottleneck.

Pipeline: claim → retrieval → cross-encoder reranking → sentence selection → binary stance → evidence-filtered
claim aggregation. Each stage has its own evaluation and saved predictions.
""")
md("""
## 3. Official SciFact data and annotation semantics

The archive is downloaded from the official AllenAI S3 release. SHA-256 checksums, split IDs and model revisions
are saved. Corpus IDs remain integers; evidence JSON object keys are their exact decimal strings. Abstract
sentences are never split again or renumbered. `evidence` defines relevance; `cited_doc_ids` does not.

Training/dev claims with empty evidence have no annotated evidence, whereas test claims have **missing** labels.
Test is never evaluated. Each claim-document annotation can contain several alternative rationale sets;
each set can require several sentences together. We preserve these sets for stance examples and official scoring.
""")
code("""
summary, documents, claims, rationales = study.eda()
display(summary)
display(documents[['words', 'sentences']].describe().round(2))
display(claims.groupby('split')['words'].describe().round(2))
display(rationales.groupby(['split', 'label'])['sentences'].agg(['count', 'mean', 'max']).round(2))
display(pd.Series(study.dataset.manifest).drop(['sha256', 'train_ids', 'validation_ids', 'dev_ids']))
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(documents['words'], bins=40)
axes[0].set(xlabel='Title + abstract words', ylabel='Documents')
for split in ['train', 'dev']:
    axes[1].hist(claims.loc[claims.split == split, 'words'], bins=25, alpha=.5, label=split)
axes[1].set(xlabel='Claim words', ylabel='Claims')
axes[1].legend(); plt.tight_layout(); plt.show()
""")
code("""
# Real support and contradiction examples from training annotations.
examples = []
for label in ['SUPPORT', 'CONTRADICT']:
    for c in study.dataset.train:
        matches = [(doc, r) for doc, rs in c.get('evidence', {}).items() for r in rs if r['label'] == label]
        if matches:
            doc, rationale = matches[0]
            examples.append({'claim_id': c['id'], 'claim': c['claim'], 'label': label, 'doc_id': int(doc),
                             'original_indices': rationale['sentences'], 'evidence': ' '.join(
                                 study.dataset.corpus[int(doc)]['abstract'][i] for i in rationale['sentences'])})
            break
display(pd.DataFrame(examples))
""")
md("""
## 4. Methodology and evaluation definitions

We form an internal validation split from original training claims using connected components that share
cited papers or identical normalized claim text. This keeps related claim variants together. The official dev
texts are used only to remove exact duplicated training copies; dev evidence labels never enter training,
negative mining, or tuning. The official train/dev split still shares corpus documents, as is normal for retrieval.

Recall@K is the macro mean fraction of a claim's gold documents retrieved. Hit@10 is the mean indicator that
at least one gold document appears in the first 10. MRR@10 uses the reciprocal rank of the first relevant document,
with zero for a miss. Retrieval excludes claims without evidence because their recall denominator is zero.
nDCG@10 has binary relevance. Evidence overlap is micro-averaged; best-rationale recall at a sentence budget
averages over annotated claim-document pairs, choosing the best covered alternative rationale.

Official SciFact abstract correctness requires the correct stance **and a complete gold rationale contained
in the first three predicted sentences**. Sentence correctness requires the whole rationale, not partial overlap.
The vendored evaluator is unmodified and tested against the official documentation's example. Claim accuracy/F1
are simplified project metrics using a documented aggregation rule; they are not official SciFact scores.
""")
code("""
import subprocess
checked = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True, check=True)
print(checked.stdout)
""")
md("""
## 5. Experiment A: BM25

BM25Okapi uses lowercase regex word tokens with default $k_1=1.5$, $b=0.75$. We compare abstract only with
title + abstract. No stemming or domain-specific stopword rules are added. All neural comparisons use the
same claim IDs, corpus, gold documents and metric implementation. The following table is internal validation only.
""")
code("display(study.bm25().round(4))")
md("""
## 6. Experiment B: pretrained dense retrieval

all-MiniLM-L6-v2 produces mean-pooled, L2-normalized vectors with cosine search in memory. Corpus embeddings
are cached with corpus text and truncation fingerprints. We keep the model's 256-token document limit in full
mode and use batched inference. Dense query scores are meaningful only within the same model/configuration.

SPECTER2 uses its document proximity adapter and its distinct ad-hoc query adapter, CLS pooling and 512 tokens,
following the official model cards. Documents concatenate title, tokenizer separator, and abstract. It is
scheduled after the core stages so an optional dependency/model failure cannot block the core study. Its full
corpus result, or an explicit failure record, appears in the final retrieval table.
""")
code("display(study.pretrained().round(4))")
md("""
## 7. Experiment C1: contrastive training with random negatives

Each positive is an annotated evidence abstract, regardless of SUPPORT/CONTRADICT. Negatives are sampled from
the remaining corpus. We exclude all known training-family evidence papers and cited papers. These exclusions
do not establish that every remaining paper is a true negative: SciFact annotations are incomplete.

The loss is multi-positive InfoNCE over explicit and in-batch document candidates. Other known positives are
positive targets; protected family documents are masked rather than used as false negatives. Both C1 and C2 start
from the same pretrained checkpoint and use identical seed, positive examples, batch size, schedule and epoch
budget. AdamW, linear warmup/decay, norm-1 gradient clipping, validation checkpointing and patience-based early
stopping are used. Checkpoints maximize validation Recall@10, breaking ties with MRR@10.
The executed GPU loops use dynamic AMP loss scaling and a batch-iteration learning-rate schedule:
overflowing batches can skip optimizer updates while the schedule advances. A repetition emitted a
scheduler-order warning; this numerical limitation is retained in the discussion.
""")
code("""
display(study.contrastive('random').round(4))
from scifact_project.data import read_jsonl
training_pairs = pd.DataFrame(read_jsonl(config.out / 'tables/retriever_random_pairs.jsonl'))
display(training_pairs[['claim_id', 'claim', 'positive', 'negative', 'negative_mode']].head())
""")
md("""
## 8. Experiment C2: BM25 hard negatives

We sample from the first 20 permissible candidates among BM25's top 100. Positive/family/citation exclusions
are identical to C1. Mining uses training claims only and does not inspect dev annotations. Saved pair manifests
make the exact positive and negative choices auditable. Hard negatives may still be unannotated relevant papers;
we discuss this source of noise rather than interpreting sampled negatives as verified scientific refutations.
""")
code("display(study.contrastive('hard').round(4))")
md("""
## 9. Experiment D: hybrid rank fusion

Reciprocal Rank Fusion combines rank positions rather than incompatible raw score scales:
$s(d)=w/(c+r_{BM25}(d))+(1-w)/(c+r_{dense}(d))$, omitting a term when a document is absent.
For each pretrained/random/hard dense retriever we tune c in {10, 60} and w in {0.25, 0.5, 0.75} on
internal validation only, with input depth 50. A trained dense variant is selected for the deployed hybrid
using the same validation criterion. Dev comparisons do not feed back into this choice.
""")
code("display(study.hybrid().round(4)); display(pd.Series(study.selection))")
md("""
## 10. Experiment E: pretrained cross-encoder reranking

ms-marco-MiniLM-L6-v2 scores claim–document pairs from the selected hybrid's top 20 candidates. This is a
general-domain pretrained relevance model, not a scientific stance classifier. Candidate-set equality is
asserted before/after reranking. We report its per-query scoring cost separately from initial retrieval and
index construction. Its top-20 recall cannot exceed the fixed candidate set's recall.
""")
code("display(study.reranking().round(4))")
md("""
## 11. Experiment F: supervised evidence extraction

The baseline ranks individual abstract sentences by pretrained embedding cosine similarity to the claim.
The supervised selector trains on claim–sentence pairs with binary labels from the union of alternative gold
rationales. It keeps all positives and samples up to three negatives per positive. Non-rationale sentences and
cited-but-unannotated abstracts provide negatives; the latter are weak labels, recorded by source.

The CPU profile fine-tunes the six-layer MiniLM encoder to complete the study within available resources.
The GPU profile uses SciBERT. Weighted cross entropy handles class imbalance; validation macro F1 selects epochs.
Sentence and pipeline thresholds are subsequently tuned using **retrieved validation documents**, with a maximum
of three sentences per document. We evaluate gold-document and retrieved-document settings and both overlap
and official complete-rationale metrics. Independent classification cannot fully model jointly necessary evidence.
""")
md("""
## 12. Experiment G: stance verification

Each complete annotated rationale creates a claim–evidence example labeled SUPPORT or CONTRADICT; alternatives
remain separate. No arbitrary unannotated abstract is assigned an NEI stance. The first simple model is TF-IDF
plus balanced logistic regression. The transformer is independently fine-tuned with the same weighted loss,
scheduling and checkpoint practices as the selector. NEI is handled only by pipeline relevance/confidence filters.
""")
code("display(study.supervised())")
md("""
## 13. End-to-end integration and validation-only acceptance tuning

The deployed pipeline verifies the reranked top five documents. It selects up to three sentences above a
validation-tuned selector threshold and predicts binary stance on those sentences in original abstract order.
Stance confidence and relevance-logit thresholds are tuned on internal validation official rationalized F1.
When accepted documents disagree, the claim verdict is the label of the document with largest selector score
times stance probability; ties choose the smallest document ID. This is our aggregation policy, not an official rule.
An empty accepted set yields NOT ENOUGH INFO. Scores are uncalibrated model outputs, not truth probabilities.
""")
code("""
end_to_end = study.final_evaluation()
display(pd.Series(study.selection))
print('All configurations are frozen. Final comparisons below use official dev; test remains unlabeled.')
""")
md("""
## 14. Oracle and controlled ablation settings

We compare predicted documents/evidence/stance, gold documents with predicted evidence/stance, and gold
documents with a complete gold rationale and predicted stance. Oracle evidence uses the shortest complete
alternative per document, deterministically. Oracle settings are diagnostics and never enter `predict`.
Both gold-document settings retain the same stance-confidence threshold; known gold relevance bypasses
the document-relevance filter in both. Thus the second oracle changes evidence, not acceptance policy.

Two retrieval ablations reuse the trained selector, stance model and sentence/stance thresholds with BM25
or the hybrid before reranking. The hybrid ordering ablation uses the exact same cross-encoder document scores
and acceptance thresholds, so only the ordering changes. BM25 omits that relevance-logit filter; its comparison
has this additional difference. Official scores retain document-level
predictions; a separate table shows simplified aggregate claim metrics.
""")
md("""
## 15. Supplementary SPECTER2 experiment

The adapters/Transformers versions are pinned together because the initially installed Transformers 5 release
was not the adapter-compatible environment used here. `scripts/probe_models.py` checks both adapter forward
passes. This cell evaluates the entire current-mode corpus when enabled and records any unavoidable failure.
SPECTER2 is supplementary and never changes the already selected pipeline after dev evaluation.
""")
code("""
study.specter2()
display(study.status.get('specter2'))
study.refresh_fusion_costs()
retrieval_table = study.save_retrieval_tables()
from scifact_project.report import generate_report
retrieval, evidence, stance, ablations = generate_report(study)
""")
md("## 16. Final retrieval comparisons (official dev)")
code("""
display(retrieval.loc[retrieval.split == 'dev'].drop(columns=['split', 'seed']).set_index('experiment').round(4))
display(Image(filename=str(config.out / 'figures/retrieval_recall.png')))
display(Image(filename=str(config.out / 'figures/retrieval_mrr.png')))
""")
md("## 17. Evidence, stance, and end-to-end comparisons")
code("""
display(evidence.round(4))
display(stance.round(4))
display(ablations.round(4))
display(pd.DataFrame(end_to_end['predicted_all']['official']).round(4))
for name in ['evidence_selection', 'confusion_matrices', 'end_to_end_ablations']:
    display(Image(filename=str(config.out / 'figures' / f'{name}.png')))
gold_stance = json.loads((config.out / 'metrics/stance_gold.json').read_text())
per_class = {label: values for label, values in gold_stance['per_class'].items() if isinstance(values, dict)}
display(pd.DataFrame.from_dict(per_class, orient='index').round(4))
""")
md("""
## 18. Data distributions, learning curves, and sampling uncertainty

Pair classifiers use class-weighted training cross entropy; plotted validation cross entropy is unweighted.
Their loss levels therefore measure different weightings. Retrieval curves show training InfoNCE loss and
held-out ranking quality, since validation Recall@10 is the checkpoint-selection criterion.
""")
code("""
for name in ['data_distributions', 'training_curves']:
    display(Image(filename=str(config.out / 'figures' / f'{name}.png')))
display(pd.read_csv(config.out / 'tables/training_summary.csv').round(2))
display(pd.DataFrame(json.loads((config.out / 'metrics/paired_bootstrap.json').read_text())))
seed_summary = config.out / 'tables/retriever_seed_summary.csv'
if seed_summary.exists():
    seeded = pd.read_csv(seed_summary)
    formatted = seeded[['experiment', 'seeds', 'runs']].copy()
    for metric in ['Recall@1', 'Recall@5', 'Recall@10', 'MRR@10']:
        formatted[metric] = [f'{mean:.4f} ± {std:.4f}' for mean, std in
                             zip(seeded[f'{metric}_mean'], seeded[f'{metric}_std'])]
    display(formatted)
""")
md("""
## 19. Measured discussion, five concrete errors, and conclusions

This report is generated from actual saved metrics. Differences are in percentage points, while tables use
fractions. Failure categories are determined from retrieval, selection, and stance outputs; they do not imply
unobserved linguistic explanations. The selected main checkpoint and downstream pipeline use one seed;
repeated-retriever results appear above when available. Bootstrap intervals resample citation families
to account for related claims and cannot replace repeated training seeds.
""")
code("display(Markdown((config.out / 'analysis.md').read_text()))")
md("""
## 20. Working annotation-free prediction

The reusable function loads the selected trained retriever, selector and stance checkpoints. It uses corpus
texts only, with no lookup of claim annotations or citation metadata. A new claim returns a verdict, rankings,
titles, original evidence indices and model scores. The example below is a real execution, not a gold oracle.
""")
code("""
from scifact_project.pipeline import ClaimVerifier
verifier = ClaimVerifier(config, study.dataset, study.selection)
import time
started = time.perf_counter()
prediction = verifier.predict('Antiretroviral therapy reduces rates of tuberculosis.')
print(f'Warm-model single-claim inference: {time.perf_counter() - started:.3f} seconds')
display(prediction)
""")
md("""
## Reproduction checklist and semester scope

All core comparisons, checkpoints, pair manifests, rankings, scoring outputs, figures (PNG/PDF), configuration,
package versions and source/model hashes live in the selected `outputs/` run directory. `scripts/execute_notebook.py`
re-executes and validates this notebook. Running it with populated caches displays results without retraining;
change the output directory for a clean run. `scripts/run_seeded.py` supports repeated seeds with a fixed split.

A feasible two-person semester plan is: weeks 1–3 data/metrics/baselines; 4–6 dense training and mining;
7–9 selector/stance training; 10–11 hybrid/reranker integration; 12–13 oracle experiments and repeated seeds;
14–15 manual error review, report and reproducibility audit. The implementation supplies these stages;
stronger GPU models, a scientific reranker, and repeated-seed extensions remain research follow-ups.

## References

- Wadden et al. (2020), [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/).
- [Official SciFact repository](https://github.com/allenai/scifact), [data schema](https://github.com/allenai/scifact/blob/master/doc/data.md),
  [evaluation rules](https://github.com/allenai/scifact/blob/master/doc/evaluation.md).
- Reimers & Gurevych (2019), [Sentence-BERT](https://aclanthology.org/D19-1410/);
  [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Singh et al. (2023), [SciRepEval](https://aclanthology.org/2023.emnlp-main.338/);
  [SPECTER2 document adapter](https://huggingface.co/allenai/specter2),
  [ad-hoc query adapter](https://huggingface.co/allenai/specter2_adhoc_query).
- Beltagy et al. (2019), [SciBERT](https://aclanthology.org/D19-1371/).
- [Cross-encoder ms-marco-MiniLM-L6-v2 model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).
- Cormack, Clarke & Büttcher (2009), [Reciprocal Rank Fusion](https://doi.org/10.1145/1571941.1572114).
- [SentenceTransformers contrastive loss documentation](https://sbert.net/docs/package_reference/sentence_transformer/losses.html),
  [AdapterHub installation](https://docs.adapterhub.ml/installation.html).
- [PyTorch mixed-precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
  [optimizer scheduling](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).
""")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['full', 'smoke', 'extended'], default='full')
parser.add_argument('--output', default='scientific_claim_verification.ipynb')
args = parser.parse_args()
for cell in cells:
    if cell.cell_type == 'code' and "MODE = 'full'" in cell.source:
        cell.source = cell.source.replace("MODE = 'full'", f"MODE = {args.mode!r}")
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python (SciFact)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13.11'}})
nbf.validate(notebook)
nbf.write(notebook, args.output)
print(f'Created notebook with {len(cells)} cells')
