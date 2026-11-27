# Scientific claim verification

Given a short English scientific claim, the system searches a fixed collection of research abstracts and returns a verdict, the matching abstract, and sentence-level evidence. The intended users are research students and literature reviewers looking for evidence to inspect. The primary success criterion is finding the correct abstract, stance and complete rationale. The secondary criteria are ranking relevant abstracts early and performing well across all three claim labels.

## Data and evaluation protocol

SciFact was built by domain experts who rewrote citation sentences into verifiable claims and annotated whether abstracts support or contradict them, with one or more alternative sentence rationales. This English, largely biomedical dataset contains 1,409 claims and 5,183 abstracts. The project uses the original release and its supplied sentence boundaries and IDs. Claims and annotations are CC BY 4.0, the abstract collection is ODC-By 1.0, and the vendored evaluator is Apache 2.0. [SciFact paper](https://aclanthology.org/2020.emnlp-main.609/) · [License notice](vendor/scifact/LICENSE.md)

| Split | Role | Claims | Support | Contradict | Not enough info |
|---|---|---:|---:|---:|---:|
| Internal train | Fit weights and mine negatives | 656 | 265 | 144 | 247 |
| Internal validation | Choose models, checkpoints and thresholds | 151 | 67 | 29 | 55 |
| SciFact official dev | Score fixed configurations | 300 | 124 | 64 | 112 |
| SciFact official test | Released without labels | 300 | — | — | — |

Two of the original 809 training claims had exact normalized matches in dev and were removed before splitting. Training and validation use disjoint connected groups that share a citation or normalized claim text. Duplicate IDs, empty rationales and out-of-range evidence indices are checked. The data audit found zero blank claims, titles, or sentences and zero abstracts without sentences. Missing evidence annotations on labeled examples map to corpus-relative **NOT_ENOUGH_INFO** at claim level. Unannotated documents can still contain relevant evidence. [Split manifest](outputs/full_gpu_seed42/split_manifest.json) · [Dataset card](docs/RESEARCH_DETAILS.md#dataset-card)

Official dev labels were reserved for evaluation and later error analysis. The official test score is unavailable. A post hoc text audit flagged 48 dev claims with character 3–5-gram TF-IDF cosine of at least 0.90 against training or validation. No internal train–validation pairs were flagged. Some flagged pairs express opposite stances, and the screen cannot rule out semantic overlap. Re-scoring the same saved predictions on the remaining 252 dev claims gives rationalized abstract F1 of **0.3134** for the full pipeline and **0.3034** for the BM25 pipeline. On the 160 evidence-bearing claims in that subset, MRR@10 is **0.8431** and **0.7853**, respectively. [Text-overlap audit](outputs/full_gpu_seed42/metrics/defense_overlap_audit.json)

## Approach and training

SciFact’s VeriSci pipeline motivates a staged design with inspectable evidence. MultiVerS studies joint stance and rationale prediction with full-abstract context. Our measured comparisons use local configurations on the same data, with title + abstract BM25 as the lexical baseline, embedding similarity for sentence selection, and TF-IDF + logistic regression for stance. [VeriSci](https://aclanthology.org/2020.emnlp-main.609/) · [MultiVerS](https://aclanthology.org/2022.findings-naacl.6/)

BM25 lowercases text and extracts word tokens, retaining stop words and applying no stemming or lemmatization. The neural paths use uncased WordPiece, token and segment IDs, attention masks, dynamic batch padding and length limits of 256 tokens for dense retrieval and the SciBERT pair models, and 512 for reranking. MiniLM embeddings are mean-pooled and normalized for cosine ranking. Reciprocal rank fusion combines lexical and dense lists. A frozen MiniLM cross-encoder scores 20 abstracts and passes the top five to two separately fine-tuned SciBERT classifiers. The selector accepts up to three sentences and the stance head scores SUPPORT and CONTRADICT from the evidence in document order. The system emits NEI when no document clears the gates.

The two MiniLM retrievers were initialized from the same pretrained encoder and fine-tuned with masked multi-positive InfoNCE (temperature 0.05). SciBERT’s 12-layer encoder and task head were fine-tuned using inverse-frequency weighted cross-entropy. The selector retained all positives and sampled three negatives per positive. The shared recipe was AdamW at 2 × 10⁻⁵, weight decay 0.01, 10% linear warmup then linear decay, gradient norm clipping at 1.0, and patience of two epochs. The measured GTX 1660 Ti run used PyTorch mixed precision, batch size 16 and training/split seed 42. [Technical details](docs/RESEARCH_DETAILS.md) · [Run configuration](outputs/full_gpu_seed42/config.json) · [Environment](outputs/full_gpu_seed42/provenance.json)

| Fine-tuned model | Selection on internal validation | Best / run / max epochs | Elapsed |
|---|---|---:|---:|
| MiniLM with random negatives | Recall@10, then MRR@10 | 1 / 3 / 4 | 223 s |
| MiniLM with BM25 hard negatives | Recall@10, then MRR@10 | 1 / 3 / 4 | 176 s |
| SciBERT evidence selector | Macro F1 | 3 / 4 / 4 | 1,560 s |
| SciBERT stance | Macro F1 | 3 / 5 / 6 | 534 s |

The curves show training loss falling beyond the selected epochs. Stance validation loss reached 0.594 in epoch 2 and rose to 0.672 in epoch 3 while validation macro F1 peaked at 0.694. The weighted training and unweighted validation losses use different reductions. Dense retriever validation loss was not measured. The main run compared ten retrieval configurations and five pipeline settings, including two gold-input diagnostics. Both trainable retrievers were repeated with seeds 42–44 on the same split, and hard-negative MRR gains varied in direction. Downstream training and headline results use seed 42 only. The original training peak memory and skipped optimizer updates were not measured. [Learning curves](outputs/full_gpu_seed42/figures/training_curves.pdf) · [Training records](outputs/full_gpu_seed42/tables/training_summary.csv) · [Seed results](outputs/full_gpu_seed42/tables/retriever_seed_summary.csv)

## Results

On **188 evidence-bearing official dev claims**, title + abstract BM25 achieved Recall@10 **0.9158** and MRR@10 **0.7822**. The full reranked hybrid achieved **0.9488** and **0.8542**, respectively. Recall@10 averages the fraction of annotated evidence abstracts found in the first ten. MRR@10 averages the reciprocal rank of the first annotated abstract in the first ten, with zero when none appears. The family-paired bootstrap 95% interval for the MRR@10 gain is **+2.89 to +11.49 percentage points**. These metrics describe retrieval of annotated abstracts. [Retrieval table](outputs/full_gpu_seed42/tables/retrieval_final.csv) · [Bootstrap](outputs/full_gpu_seed42/metrics/paired_bootstrap.json)

All rows below use **300 official dev claims**. Rationalized abstract P/R/F1 is the official SciFact score requiring the correct abstract, correct stance and a complete annotated rationale among the first three evidence sentences. Precision is the share of predictions that qualify. Recall is the share of annotated answers recovered. F1 is their harmonic mean. Claim accuracy counts correct verdicts and claim macro F1 weights each label equally. The product of selector score and stance probability chooses a verdict when accepted documents disagree.

| Verification setting | Claim accuracy | Claim macro F1 | Abstract precision | Abstract recall | Abstract F1 |
|---|---:|---:|---:|---:|---:|
| BM25 pipeline | 0.6100 | 0.5785 | 0.2581 | 0.3828 | 0.3083 |
| Hybrid before reranking | 0.5967 | 0.5577 | 0.3052 | 0.3636 | 0.3319 |
| Hybrid + reranking | 0.5900 | 0.5531 | 0.2857 | 0.3732 | 0.3237 |

The BM25 pipeline skips the reranker relevance gate, so it has a filtering difference as well as a ranking difference. In the controlled ordering comparison the candidate pool, scores and thresholds stay fixed. Reranking raises MRR@10 by 4.56 points and reduces rationalized abstract F1 by 0.82 points. Supplying gold abstracts raises rationalized F1 to **0.4677**, and supplying complete gold evidence as well raises it to **0.6459**. These diagnostic changes also alter coverage and false positives. On 338 gold dev rationale examples, SciBERT stance macro F1 was **0.5874** versus **0.3875** for TF-IDF + logistic regression. On retrieved dev abstracts, sentence-selector F1 was **0.4142** versus **0.2162** for embedding similarity. [Detailed scores](outputs/full_gpu_seed42/metrics/end_to_end.json) · [Component tables](outputs/full_gpu_seed42/tables/) · [Evaluation code](scifact_project/metrics.py)

### Error analysis and limitations

| Gold claim label | Precision | Recall | F1 | Count |
|---|---:|---:|---:|---:|
| CONTRADICT | 0.3860 | 0.3438 | 0.3636 | 64 |
| SUPPORT | 0.6017 | 0.5726 | 0.5868 | 124 |
| NOT_ENOUGH_INFO | 0.6720 | 0.7500 | 0.7089 | 112 |

The verifier correctly supported claim 53 about ALDH1 and poorer breast cancer prognosis using the annotated abstract. On claim 51, which reverses the direction, it predicted SUPPORT again. The stance head mislabeled the ALDH1 sentence, and a DUSP4 abstract won the final aggregation. In claim 3 the gold abstract ranked first and the selector missed all annotated evidence, yielding an incorrect abstention. These failures motivate testing evidence sets and requiring closer alignment of entities and claim direction. [Confusion matrices](outputs/full_gpu_seed42/figures/confusion_matrices.pdf) · [Saved error analysis](outputs/full_gpu_seed42/tables/error_examples.json) · [Predictions](outputs/full_gpu_seed42/predictions/pipeline_predicted_all.json)

The corpus covers one benchmark and the system reads abstracts with bounded context. Weak negatives, lexical overlap, class imbalance and a single downstream seed limit generalization. The study supports evidence triage for human review. Verdict scores are uncalibrated and require scientific, clinical or editorial judgment.

## Reproducibility and live inference

The repository includes the executed notebook, pinned model revisions, original split and threshold manifests, code and saved metrics. The live command loads checkpoints and corpus embeddings, then accepts instructor-provided claims and shows BM25 tokens, the WordPiece pair preview, IDs, masks, evidence and scores. On the same 6 GB GTX 1660 Ti, 15 timed predictions after warm-up measured **987 ms median** and **0.97 claims/s** sequential throughput, with **1,331 MiB** peak PyTorch CUDA allocation including resident weights. Each fine-tuned SciBERT has **109,920,002** parameters. The dense MiniLM has **22,713,216** and the reranker **22,713,601**. [Benchmark record](outputs/full_gpu_seed42/metrics/defense_inference_benchmark.json) · [Reproduction instructions](README.md) · [Defense run sheet](docs/DEFENSE.md)
