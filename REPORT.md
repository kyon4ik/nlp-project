# Short results report

**Task:** verify scientific claims using retrieved abstracts and evidence sentences.
These are existing full GPU run results, not newly trained results. Source:
[`outputs/full_gpu_seed42/analysis.md`](outputs/full_gpu_seed42/analysis.md) and its
[`tables/`](outputs/full_gpu_seed42/tables/). Main training seed: 42; GTX 1660 Ti, 6 GB.

Retrieval uses 188 evidence-bearing dev claims. Final verification uses all 300 dev
claims. No test-set performance is reported. Settings were selected on internal validation.

| Measurement | Score |
|---|---:|
| Title + abstract BM25 MRR@10 | 0.7822 |
| Hybrid + reranking MRR@10 | 0.8542 |
| Pretrained hybrid Recall@10 (highest observed) | 0.9545 |
| Full pipeline claim accuracy | 0.5900 |
| Full pipeline claim macro F1 | 0.5531 |
| Full pipeline official rationalized abstract F1 | 0.3237 |
| Gold documents: rationalized abstract F1 | 0.4677 |
| Gold documents + gold evidence: rationalized abstract F1 | 0.6459 |

Reranking improves MRR over BM25 by **7.20 percentage points**, but better ranking
does not guarantee better final verification. In the controlled ordering comparison,
reranking reduced rationalized abstract F1 by 0.82 points.

Providing correct documents raises rationale F1 by 14.41 points; also providing correct
evidence raises it another 17.82 points. Evidence selection and stance remain major
challenges even when retrieval is strong. These oracle changes also affect coverage
and false positives, so their gains are not an additive causal error breakdown.

On 338 gold-rationale examples, trained stance macro F1 is **0.5874**, versus **0.3875**
for TF-IDF + logistic regression. This uses a different denominator from claim evaluation.

**Limitations:** downstream training was run with one seed. Retriever repetitions with
seeds 42–44 show mixed hard-negative gains. Negative examples may contain unannotated
relevant papers; long inputs are truncated. GPU scheduler-order warnings were recorded,
and skipped optimizer updates were not counted. These development-set results establish
a useful baseline, not reliable scientific fact-checking performance.
