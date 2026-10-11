"""Tables, publication-style figures and analysis generated from measured artifacts only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import write_json, claim_label, gold_docs, related_groups
from .metrics import retrieval_metrics


def load(config, name):
    return json.loads((config.out / name).read_text())


def result_tables(config):
    retrieval = pd.read_csv(config.out / "metrics/retrieval.csv")
    evidence = load(config, "metrics/evidence.json")
    evidence_table = pd.DataFrame([{k: r[k] for k in ["model", "setting", "threshold", "precision", "recall", "f1",
                                                    "best_rationale_recall@budget"]} | {
          "fixed_budget_best_rationale_recall": r["fixed_budget"]["best_rationale_recall@budget"],
          "official_selection_f1": r["official_selection"]["f1"]} for r in evidence])
    stance = []
    for name, filename in [("TF-IDF + logistic regression (gold rationale)", "stance_tfidf.json"),
                            ("Trained transformer (gold rationale)", "stance_gold.json"),
                            ("Trained transformer (predicted evidence, conditional)", "stance_predicted_evidence.json")]:
        metrics = load(config, f"metrics/{filename}")
        stance.append({"model_setting": name, "accuracy": metrics.get("accuracy"), "macro_f1": metrics.get("macro_f1"),
                       "examples": metrics.get("examples"), "coverage": metrics.get("gold_document_coverage", 1.)})
    stance_table = pd.DataFrame(stance)
    e2e = load(config, "metrics/end_to_end.json")
    e2e_table = pd.DataFrame([{"setting": name, "claim_accuracy": r["accuracy"], "claim_macro_f1": r["macro_f1"],
                                **{f"official_{metric}_f1": r["official"][metric]["f1"] for metric in
                                   ["abstract_label_only", "abstract_rationalized", "sentence_selection", "sentence_label"]}}
                               for name, r in e2e.items()])
    for name, table in [("retrieval_final", retrieval), ("evidence_final", evidence_table),
                         ("stance_final", stance_table), ("end_to_end_final", e2e_table)]:
        table.to_csv(config.out / "tables" / f"{name}.csv", index=False)
    training = []
    for task in ['dense_random', 'dense_hard', 'selector', 'stance']:
        history = load(config, f'metrics/{task}_history.json')
        curve = pd.DataFrame([{'epoch': h['epoch'], 'train_loss': h['train_loss'],
                              'validation_loss': h.get('validation_loss'),
                              'validation_recall@10': h['validation'].get('Recall@10'),
                              'validation_macro_f1': h['validation'].get('macro_f1'),
                              'elapsed_seconds': h['elapsed_seconds'],
                              'effective_batch_size': h.get('effective_batch_size', config.batch_size)} for h in history])
        curve.to_csv(config.out / 'metrics' / f'{task}_history.csv', index=False)
        best = max(history, key=lambda h: (h['validation']['Recall@10'], h['validation']['MRR@10'])
                   if task.startswith('dense') else (h['validation']['macro_f1'],))
        training.append({'model': task, 'checkpoint': str(config.out / 'checkpoints' / task),
                         'base_model': config.dense_model if task.startswith('dense') else config.pair_model,
                         'seed': config.seed, 'device': config.device, 'epochs_run': len(history),
                         'selected_epoch': best['epoch'], 'training_seconds': history[-1]['elapsed_seconds'],
                         'effective_batch_size': best.get('effective_batch_size', config.batch_size)})
    pd.DataFrame(training).to_csv(config.out / 'tables/training_summary.csv', index=False)
    write_json(config.out / 'metrics/training_summary.json', training)
    return retrieval, evidence_table, stance_table, e2e_table


def _save(config, name):
    plt.tight_layout()
    plt.savefig(config.out / "figures" / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.savefig(config.out / "figures" / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def figures(study, tables):
    config = study.config
    retrieval, evidence, stance, e2e = tables
    docs = pd.read_csv(config.out / "tables/document_lengths.csv")
    claims = pd.read_csv(config.out / "tables/claim_lengths.csv")
    rationales = pd.read_csv(config.out / "tables/rationale_lengths.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].hist(docs["words"], bins=45, color="#366e9e")
    axes[0, 0].set(xlabel="Document words (title + abstract)", ylabel="Documents")
    for split in ["train", "dev"]:
        axes[0, 1].hist(claims.loc[claims.split == split, "words"], bins=25, alpha=.55, label=split)
    axes[0, 1].set(xlabel="Claim words", ylabel="Claims"); axes[0, 1].legend()
    claims.loc[claims.split == "train", "evidence_documents"].value_counts().sort_index().plot.bar(ax=axes[1, 0], color="#366e9e")
    axes[1, 0].set(xlabel="Annotated evidence documents per training claim", ylabel="Claims")
    rationales.loc[rationales.split == "train", "sentences"].value_counts().sort_index().plot.bar(ax=axes[1, 1], color="#366e9e")
    axes[1, 1].set(xlabel="Sentences per training rationale", ylabel="Rationales")
    _save(config, "data_distributions")
    dev = retrieval.loc[retrieval.split == "dev"].set_index("experiment")
    plt.figure(figsize=(10, 5))
    for name, r in dev.iterrows():
        plt.plot([1, 3, 5, 10, 20], [r[f"Recall@{k}"] for k in [1, 3, 5, 10, 20]], marker="o", label=name)
    plt.xlabel("Retrieved documents K"); plt.ylabel("Macro Recall@K"); plt.ylim(0, 1)
    plt.xticks([1, 3, 5, 10, 20])
    plt.legend(fontsize=8, ncol=2); plt.grid(alpha=.2)
    _save(config, "retrieval_recall")
    plt.figure(figsize=(10, 4))
    dev["MRR@10"].sort_values().plot.barh(color="#366e9e")
    plt.xlabel("MRR@10 (official dev, evidence-bearing claims)"); plt.xlim(0, 1)
    _save(config, "retrieval_mrr")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, task in enumerate(["dense_random", "dense_hard", "selector", "stance"]):
        history = load(config, f"metrics/{task}_history.json")
        ax = axes.flat[i]
        ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], marker="o", label="Training loss")
        ax.set(xlabel="Epoch", ylabel="Loss", title=task)
        secondary = ax.twinx()
        if task.startswith("dense"):
            values = [h["validation"]["Recall@10"] for h in history]
            label = "Validation Recall@10"
        else:
            values = [h["validation"]["macro_f1"] for h in history]
            label = "Validation macro F1"
            ax.plot([h["epoch"] for h in history], [h["validation_loss"] for h in history], marker="s", label="Validation loss")
        secondary.plot([h["epoch"] for h in history], values, marker="^", color="#c35a3b", label=label)
        secondary.set_ylim(0, 1); secondary.set_ylabel(label)
        ax.legend(loc="lower left", fontsize=7)
    _save(config, "training_curves")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, filename, title in [(axes[0], "stance_gold.json", "Gold-rationale stance"),
                                (axes[1], "end_to_end.json", "Predicted pipeline claim verdict")]:
        r = load(config, f"metrics/{filename}")
        if filename == "end_to_end.json":
            r = r["predicted_all"]
        matrix = np.array(r["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(matrix)), r["label_order"], rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(matrix)), r["label_order"], fontsize=8)
        ax.set(xlabel="Predicted", ylabel="Gold", title=title)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    _save(config, "confusion_matrices")
    evidence.set_index(["model", "setting"])[["precision", "recall", "f1"]].plot.bar(figsize=(10, 4), ylim=(0, 1))
    plt.ylabel("Sentence overlap (different from official rationale scoring)")
    _save(config, "evidence_selection")
    ablation_labels = {'predicted_all': 'Predicted pipeline', 'gold_documents': 'Gold documents',
                       'gold_documents_gold_evidence': 'Gold documents + gold rationale',
                       'bm25_predicted': 'BM25 pipeline', 'hybrid_without_reranking': 'Hybrid before reranking'}
    plot_data = e2e.set_index("setting")[["official_abstract_rationalized_f1", "official_sentence_label_f1"]]
    plot_data = plot_data.rename(index=ablation_labels, columns={
        'official_abstract_rationalized_f1': 'Abstract + rationale + stance',
        'official_sentence_label_f1': 'Complete rationale + stance (sentences)'})
    plot_data.plot.barh(figsize=(10, 4), xlim=(0, 1))
    plt.xlabel("Official SciFact F1"); plt.legend(fontsize=8)
    _save(config, "end_to_end_ablations")


def paired_bootstrap(study, left, right, metric="MRR@10", repetitions=2000):
    claims = [c for c in study.dataset.dev if gold_docs(c)]
    differences = np.array([retrieval_metrics([c], study.rankings[right]["dev"])[metric] -
                            retrieval_metrics([c], study.rankings[left]["dev"])[metric] for c in claims])
    # Claim variants from one citation family are correlated; resample families rather than individual claims.
    groups = related_groups(study.dataset.dev)
    family_ids = sorted({groups[c['id']] for c in claims})
    sums = np.array([sum(d for c, d in zip(claims, differences) if groups[c['id']] == family)
                     for family in family_ids])
    counts = np.array([sum(groups[c['id']] == family for c in claims) for family in family_ids])
    rng = np.random.default_rng(study.config.seed)
    sampled = rng.integers(len(family_ids), size=(repetitions, len(family_ids)))
    estimates = sums[sampled].sum(1) / counts[sampled].sum(1)
    return {"left": left, "right": right, "metric": metric, "mean_difference": differences.mean(),
            "percentile_95_ci": np.quantile(estimates, [.025, .975]).tolist(), "resamples": repetitions,
            "clusters": len(family_ids),
            "interpretation": "Citation-family cluster sampling uncertainty; not between-training-seed variance"}


def analysis(study, tables):
    config = study.config
    retrieval, evidence, stance, e2e_table = tables
    dev = retrieval.loc[retrieval.split == "dev"].set_index("experiment")
    e2e = load(config, "metrics/end_to_end.json")
    best_mrr = dev["MRR@10"].idxmax()
    best_recall = dev["Recall@10"].idxmax()
    base = "bm25_title_abstract"

    def delta(a, b, metric):
        return 100 * (dev.loc[a, metric] - dev.loc[b, metric])

    bootstrap = [paired_bootstrap(study, base, "hybrid_reranked"),
                 paired_bootstrap(study, "dense_random", "dense_hard")]
    write_json(config.out / "metrics/paired_bootstrap.json", bootstrap)
    selected = study.selection["hybrid_name"]
    repeats_path = config.out / 'tables/retriever_seed_summary.csv'
    if repeats_path.exists():
        repeats = pd.read_csv(repeats_path)
        seed_note = ('The two task-specific retrievers were repeated with seeds ' + str(repeats.iloc[0]['seeds']) +
                     ' on the same grouped split and hyperparameters. The seed table reports measured mean ± sample standard deviation. '
                     'Downstream selector/stance training and the selected deployed pipeline use seed 42 only.')
        raw = pd.read_csv(config.out / 'metrics/retriever_seed_runs.csv')
        paired = raw.pivot(index='seed', columns='experiment', values='MRR@10')
        differences = paired['dense_hard'] - paired['dense_random']
        direction = ('positive for every seed' if (differences > 0).all() else
                     'negative for every seed' if (differences < 0).all() else 'mixed across seeds or tied')
        seed_note += (' The hard-minus-random MRR@10 differences are ' +
                      ', '.join(f'seed {s}: {100 * value:+.2f} points' for s, value in differences.items()) +
                      f'; their direction is {direction}.')
    else:
        seed_note = ('Only one seed was run in this configuration. Repeated runs are supported, '
                     'but no unmeasured mean ± standard deviation is claimed.')
    pred = e2e["predicted_all"]
    docs = e2e["gold_documents"]
    gold = e2e["gold_documents_gold_evidence"]
    metric = "abstract_rationalized"
    doc_gain = 100 * (docs["official"][metric]["f1"] - pred["official"][metric]["f1"])
    evidence_gain = 100 * (gold["official"][metric]["f1"] - docs["official"][metric]["f1"])
    hybrid_ms = 1000 * study.records[selected]['query_seconds']['dev'] / len(study.dataset.dev)
    rerank_ms = 1000 * study.records['hybrid_reranked']['query_seconds']['dev'] / len(study.dataset.dev)
    specter_note = (f"SPECTER2 changes MRR@10 versus BM25 by {delta('specter2', base, 'MRR@10'):+.2f} points "
                    f"and Recall@10 by {delta('specter2', base, 'Recall@10'):+.2f} points.\n"
                    if 'specter2' in dev.index else
                    'SPECTER2 did not produce an evaluated result; consult status.json for its recorded execution status.\n')
    evidence_by_setting = evidence.set_index(['model', 'setting'])
    evidence_delta = 100 * (evidence_by_setting.loc[('supervised', 'retrieved_documents'), 'official_selection_f1'] -
                           evidence_by_setting.loc[('similarity', 'retrieved_documents'), 'official_selection_f1'])
    ordering_delta = 100 * (pred['official'][metric]['f1'] -
                           e2e['hybrid_without_reranking']['official'][metric]['f1'])
    numerical_note = ''
    if config.profile == 'gpu':
        numerical_note = ('GPU training uses dynamic AMP loss scaling with a batch-iteration learning-rate schedule. '
                          'An overflowing batch can skip its optimizer update while the scheduler advances. '
                          'A follow-up implementation check should compare scheduling by successful optimizer updates; '
                          'bitwise GPU reproducibility is not claimed. '
                          'See [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) and '
                          '[optimizer scheduling](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).')
        if (config.out / 'metrics/numerical_warnings.json').exists():
            numerical_note += (' Observed scheduler-order warnings from retriever seeds 43 and 44 are retained in '
                               'metrics/numerical_warnings.json; the number of skipped updates was not measured.')
    text = f"""## Measured findings

This is a **{'SMOKE TEST: reduced data and corpus; not a scientific result' if config.mode == 'smoke' else 'full-data, compact CPU study' if config.profile == 'cpu' else 'full-data GPU study'}** with main-pipeline training seed {config.seed}; retrieval seed repetitions, if completed, are described below.
Retrieval is evaluated on {int(dev.loc[base, 'evaluable_claims'])} evidence-bearing official dev claims;
end-to-end metrics include all {len(study.dataset.dev)} dev claims. No test metrics are reported.

The highest dev MRR@10 is **{best_mrr}**, {dev.loc[best_mrr, 'MRR@10']:.4f},
{delta(best_mrr, base, 'MRR@10'):+.2f} percentage points relative to title+abstract BM25.
The highest Recall@10 is **{best_recall}**, {dev.loc[best_recall, 'Recall@10']:.4f}.
These are descriptive comparisons across prespecified experiments, not dev-driven model selection.
Deployment uses {selected} followed by reranking, chosen with internal validation labels only.

Adding titles to BM25 changes MRR@10 by {delta(base, 'bm25_abstract', 'MRR@10'):+.2f} points and
Recall@10 by {delta(base, 'bm25_abstract', 'Recall@10'):+.2f} points.

Pretrained MiniLM versus BM25 changes MRR@10 by {delta('dense_pretrained', base, 'MRR@10'):+.2f} points
and Recall@10 by {delta('dense_pretrained', base, 'Recall@10'):+.2f} points.
{specter_note}
Random-negative fine-tuning versus pretrained MiniLM changes MRR@10 by
{delta('dense_random', 'dense_pretrained', 'MRR@10'):+.2f} points and Recall@10 by
{delta('dense_random', 'dense_pretrained', 'Recall@10'):+.2f} points.
Hard negatives versus random negatives change MRR@10 by {delta('dense_hard', 'dense_random', 'MRR@10'):+.2f} points
and Recall@10 by {delta('dense_hard', 'dense_random', 'Recall@10'):+.2f} points.
The {selected} fusion changes MRR@10 versus its trained dense input by
{delta(selected, 'dense_' + study.selection['dense_mode'], 'MRR@10'):+.2f} points
and versus BM25 by {delta(selected, base, 'MRR@10'):+.2f} points.
Reranking the exact same {config.candidate_count} hybrid candidates changes Recall@1 by
{delta('hybrid_reranked', selected, 'Recall@1'):+.2f} points and MRR@10 by
{delta('hybrid_reranked', selected, 'MRR@10'):+.2f} points. Its candidate Recall@20 cannot improve when the pool is fixed.
Its measured cross-encoder scoring adds {rerank_ms:.2f} ms per claim to the hybrid's {hybrid_ms:.2f} ms,
averaged over batched queries; indexing and model loading are excluded. This throughput measure is not single-query latency.
The controlled end-to-end reranking-order comparison changes official rationalized abstract F1 by
{ordering_delta:+.2f} points, with the same cross-encoder acceptance scores and thresholds in both settings.

For retrieved documents, supervised selection changes official sentence-selection F1 versus the
embedding-similarity baseline by {evidence_delta:+.2f} points. This comparison uses each selector's
independently validation-tuned threshold and the same maximum sentence budget.

The trained gold-rationale stance model has accuracy {stance.iloc[1]['accuracy']:.4f} and macro F1
{stance.iloc[1]['macro_f1']:.4f}; the TF-IDF baseline has macro F1 {stance.iloc[0]['macro_f1']:.4f}.
Gold-rationale examples include every alternative rationale separately, so this table has a different
denominator from the claim-level pipeline and should not be compared as if the tasks were identical.

The predicted pipeline has claim macro F1 {pred['macro_f1']:.4f} and official rationalized abstract F1
{pred['official'][metric]['f1']:.4f}. Replacing retrieved documents with gold documents changes the latter by
{doc_gain:+.2f} points; additionally replacing selected evidence with a complete gold rationale changes it by
{evidence_gain:+.2f} points. The larger oracle difference is associated with
{'evidence selection' if evidence_gain > doc_gain else 'document retrieval / false-positive document filtering'}.
These interventions alter both coverage and false positives and are diagnostics, not an additive causal decomposition.
Even with gold documents and a complete gold rationale, official rationalized abstract F1 is
{gold['official'][metric]['f1']:.4f}; substantial verification error remains. The larger evidence-oracle gain
does not establish that evidence selection is the largest error source under every metric or denominator.
Gold evidence uses the shortest complete alternative per document; the deployed pipeline never sees annotations.
Both gold-document settings keep the same stance-confidence threshold and bypass document-relevance filtering;
the gold-evidence comparison therefore does not also change stance acceptance policy.

## Interpretation and limitations

Metric changes above explicitly retain their signs: an increase on one metric does not establish an increase on all metrics.
The retrieval plots and tables show where early-rank quality and recall disagree. Paired bootstrap intervals are saved
for MRR differences; citation-family cluster resampling accounts for correlated claim variants,
and its interpretation concerns evaluation sampling, not variation across training runs.
{seed_note}

{numerical_note}

Training and validation share neither claim IDs nor connected citation families. Two identical training texts matching
dev texts were removed without looking at dev labels. Original corpus papers may appear in both training and evaluation,
as required by the standard corpus retrieval setting. Broader near-duplicate variants in the original official split and
unknown overlap with pretrained model corpora remain possible; the internal family grouping reduces tuning leakage.

Negatives are **unannotated**, not verified irrelevant. The retriever excludes all training-family positives and cited documents,
and its in-batch loss masks other protected candidates. This reduces known false negatives without accessing dev labels.
The sentence selector also uses cited-but-unannotated abstracts as weak negatives, including NEI claims; this may introduce label noise.
Stance training uses only annotated SUPPORT/CONTRADICT rationales and never treats arbitrary abstracts as NEI ground truth.
Abstention is a validation-tuned relevance filter plus stance confidence threshold. Softmax probabilities and reranker logits
are uncalibrated; scores should not be read as reliable probabilities of scientific truth.

Sentence labels use the union of alternative gold rationale sentences. Independent sentence predictions approximate
joint rationales; overlap F1 can reward incomplete sets, whereas official SciFact scoring does not.
The official evaluator is unmodified, counts complete sets, and caps abstract scoring at three predicted sentences.
Its sentence-recall denominator counts all alternative rationale sentences. Selecting just one complete gold
alternative can therefore have sentence recall below one even when abstract rationale correctness is perfect.
Long abstracts are truncated: MiniLM uses {config.dense_length} tokens, pair models {config.pair_length}, reranker
{config.rerank_length}; SPECTER2, when available, uses 512 and distinct document/query adapters. These unequal context
windows are resource choices and a confound when comparing architectures.

The claim aggregation policy chooses the accepted document with highest selector score × stance probability.
This is a project-specific policy, not the official task: official metrics operate on claim–document predictions.
NEI denotes absence of sufficient annotated evidence in this corpus, not proof that a scientific claim is false.

Three realistic next studies are: (1) repeat downstream selector/stance training for seeds 42, 43, and 44 on the same grouped split;
(2) evaluate longer context and joint rationale selection; (3) improve relevance calibration
with carefully audited negatives and train a scientific reranker while retaining the same candidate pools.

## Concrete error cases
"""
    for row in load(config, "tables/error_examples.json"):
        text += f"\n**Claim {row['claim_id']}: {row['category']}**\n\n{row['claim']}\n\n"
        text += f"Gold claim aggregation: {row['gold_verdict']}; prediction: {row['predicted_verdict']}. "
        text += f"Gold documents: {row['gold_doc_ids']}; retrieved top {config.verification_documents}: {row['retrieved_top5']}.\n\n"
        for d in row.get('gold_document_diagnostics', []):
            text += (f"Gold document {d['doc_id']}: reranked position {d['reranked_position']}, "
                     f"accepted={d['accepted_by_pipeline']}, complete alternative={d['contains_complete_alternative']}, "
                     f"correct stance when accepted={d['correct_stance_if_accepted']}. ")
            if 'sentences_passing_selector' in d:
                text += f"Sentences passing the selector: {d['sentences_passing_selector']}. "
            text += '\n\n'
        if row["gold_rationales"]:
            r = row["gold_rationales"][0]
            text += f"Gold document {r['doc_id']}, original sentence indices {r['sentence_indices']}: {r['text']}\n\n"
        if row["predicted_documents"]:
            d = max(row["predicted_documents"], key=lambda entry: (entry['aggregation_score'], -entry['doc_id']))
            text += f"Document winning claim aggregation: {d['doc_id']}, label {d['label']}, stance score {d['stance_probability']:.3f}: "
            text += " ".join(f"[{s['index']}] {s['text']}" for s in d["evidence"]) + "\n"
        else:
            text += "No document survived the learned selector and validation-tuned acceptance filters.\n"
        text += "\nThe failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.\n"
    (config.out / "analysis.md").write_text(text, encoding="utf-8")
    return text


def generate_report(study):
    tables = result_tables(study.config)
    figures(study, tables)
    analysis(study, tables)
    return tables
