## Measured findings

This is a **full-data GPU study** with main-pipeline training seed 42; retrieval seed repetitions, if completed, are described below.
Retrieval is evaluated on 188 evidence-bearing official dev claims;
end-to-end metrics include all 300 dev claims. No test metrics are reported.

The highest dev MRR@10 is **hybrid_reranked**, 0.8542,
+7.20 percentage points relative to title+abstract BM25.
The highest Recall@10 is **hybrid_pretrained**, 0.9545.
These are descriptive comparisons across prespecified experiments, not dev-driven model selection.
Deployment uses hybrid_hard followed by reranking, chosen with internal validation labels only.

Adding titles to BM25 changes MRR@10 by +1.44 points and
Recall@10 by +1.60 points.

Pretrained MiniLM versus BM25 changes MRR@10 by -5.08 points
and Recall@10 by -0.41 points.
SPECTER2 changes MRR@10 versus BM25 by -1.59 points and Recall@10 by +0.22 points.

Random-negative fine-tuning versus pretrained MiniLM changes MRR@10 by
+2.05 points and Recall@10 by
+1.49 points.
Hard negatives versus random negatives change MRR@10 by +0.57 points
and Recall@10 by +0.53 points.
The hybrid_hard fusion changes MRR@10 versus its trained dense input by
+5.10 points
and versus BM25 by +2.64 points.
Reranking the exact same 20 hybrid candidates changes Recall@1 by
+6.43 points and MRR@10 by
+4.56 points. Its candidate Recall@20 cannot improve when the pool is fixed.
Its measured cross-encoder scoring adds 174.23 ms per claim to the hybrid's 20.84 ms,
averaged over batched queries; indexing and model loading are excluded. This throughput measure is not single-query latency.
The controlled end-to-end reranking-order comparison changes official rationalized abstract F1 by
-0.82 points, with the same cross-encoder acceptance scores and thresholds in both settings.

For retrieved documents, supervised selection changes official sentence-selection F1 versus the
embedding-similarity baseline by +19.80 points. This comparison uses each selector's
independently validation-tuned threshold and the same maximum sentence budget.

The trained gold-rationale stance model has accuracy 0.6154 and macro F1
0.5874; the TF-IDF baseline has macro F1 0.3875.
Gold-rationale examples include every alternative rationale separately, so this table has a different
denominator from the claim-level pipeline and should not be compared as if the tasks were identical.

The predicted pipeline has claim macro F1 0.5531 and official rationalized abstract F1
0.3237. Replacing retrieved documents with gold documents changes the latter by
+14.41 points; additionally replacing selected evidence with a complete gold rationale changes it by
+17.82 points. The larger oracle difference is associated with
evidence selection.
These interventions alter both coverage and false positives and are diagnostics, not an additive causal decomposition.
Even with gold documents and a complete gold rationale, official rationalized abstract F1 is
0.6459; substantial verification error remains. The larger evidence-oracle gain
does not establish that evidence selection is the largest error source under every metric or denominator.
Gold evidence uses the shortest complete alternative per document; the deployed pipeline never sees annotations.
Both gold-document settings keep the same stance-confidence threshold and bypass document-relevance filtering;
the gold-evidence comparison therefore does not also change stance acceptance policy.

## Interpretation and limitations

Metric changes above explicitly retain their signs: an increase on one metric does not establish an increase on all metrics.
The retrieval plots and tables show where early-rank quality and recall disagree. Paired bootstrap intervals are saved
for MRR differences; citation-family cluster resampling accounts for correlated claim variants,
and its interpretation concerns evaluation sampling, not variation across training runs.
The two task-specific retrievers were repeated with seeds 42,43,44 on the same grouped split and hyperparameters. The seed table reports measured mean ± sample standard deviation. Downstream selector/stance training and the selected deployed pipeline use seed 42 only. The hard-minus-random MRR@10 differences are seed 42: +0.57 points, seed 43: -0.72 points, seed 44: +1.38 points; their direction is mixed across seeds or tied.

GPU training uses dynamic AMP loss scaling with a batch-iteration learning-rate schedule. An overflowing batch can skip its optimizer update while the scheduler advances. A follow-up implementation check should compare scheduling by successful optimizer updates; bitwise GPU reproducibility is not claimed. See [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) and [optimizer scheduling](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate). Observed scheduler-order warnings from retriever seeds 43 and 44 are retained in metrics/numerical_warnings.json; the number of skipped updates was not measured.

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
Long abstracts are truncated: MiniLM uses 256 tokens, pair models 256, reranker
512; SPECTER2, when available, uses 512 and distinct document/query adapters. These unequal context
windows are resource choices and a confound when comparing architectures.

The claim aggregation policy chooses the accepted document with highest selector score × stance probability.
This is a project-specific policy, not the official task: official metrics operate on claim–document predictions.
NEI denotes absence of sufficient annotated evidence in this corpus, not proof that a scientific claim is false.

Three realistic next studies are: (1) repeat downstream selector/stance training for seeds 42, 43, and 44 on the same grouped split;
(2) evaluate longer context and joint rationale selection; (3) improve relevance calibration
with carefully audited negatives and train a scientific reranker while retaining the same candidate pools.

## Concrete error cases

**Claim 3: Incorrect abstention after retrieval**

1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.

Gold claim aggregation: SUPPORT; prediction: NOT_ENOUGH_INFO. Gold documents: [14717500]; retrieved top 5: [14717500, 4414547, 2739854, 4632921, 19058822].

Gold document 14717500: reranked position 1, accepted=False, complete alternative=False, correct stance when accepted=None. Sentences passing the selector: []. 

Gold document 14717500, original sentence indices [2, 5]: We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associations" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele. We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.

No document survived the learned selector and validation-tuned acceptance filters.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 36: Evidence predicted for an unannotated/NEI claim**

A deficiency of vitamin B12 increases blood levels of homocysteine.

Gold claim aggregation: NOT_ENOUGH_INFO; prediction: CONTRADICT. Gold documents: []; retrieved top 5: [33409100, 18557974, 37424881, 42441846, 16252863].

Document winning claim aggregation: 18557974, label CONTRADICT, stance score 0.778: [9] In conclusion, a diet high in fresh berries, citrus fruit, and vegetables effectively increases serum and RBC folate and decreases plasma homocysteine.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 49: Correct claim verdict but incomplete/wrong rationale or document**

ADAR1 binds to Dicer to cleave pre-miRNA.

Gold claim aggregation: SUPPORT; prediction: SUPPORT. Gold documents: [5953485]; retrieved top 5: [5953485, 5702790, 23913146, 18467982, 13964633].

Gold document 5953485: reranked position 1, accepted=True, complete alternative=True, correct stance when accepted=True. Sentences passing the selector: [2]. 

Gold document 5953485, original sentence indices [1]: In this study, we investigated the interaction of the RNA editing mechanism with the RNA interference (RNAi) machinery and found that ADAR1 forms a complex with Dicer through direct protein-protein interaction.

Document winning claim aggregation: 5702790, label SUPPORT, stance score 0.978: [2] We find that purified Dicer-2 can efficiently cleave pre-miRNA, but that inorganic phosphate and the Dicer-2 partner protein R2D2 inhibit pre-miRNA cleavage.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 51: Stance or document aggregation error**

ALDH1 expression is associated with better breast cancer outcomes.

Gold claim aggregation: CONTRADICT; prediction: SUPPORT. Gold documents: [45638119]; retrieved top 5: [8963413, 45638119, 23830488, 21439640, 7821634].

Gold document 45638119: reranked position 2, accepted=True, complete alternative=True, correct stance when accepted=False. Sentences passing the selector: [4]. 

Gold document 45638119, original sentence indices [4]: In a series of 577 breast carcinomas, expression of ALDH1 detected by immunostaining correlated with poor prognosis.

Document winning claim aggregation: 7821634, label SUPPORT, stance score 0.962: [7] Reduced DUSP4 expression in primary tumors after NAC was associated with treatment-refractory high Ki-67 scores and shorter recurrence-free survival.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 54: Retrieval missed every evidence document at the verification budget**

AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in the lungs.

Gold claim aggregation: CONTRADICT; prediction: NOT_ENOUGH_INFO. Gold documents: [49556906]; retrieved top 5: [3973445, 23974474, 24721866, 1590744, 7005276].

Gold document 49556906: reranked position 9, accepted=False, complete alternative=False, correct stance when accepted=None. Sentences passing the selector: [5, 6, 7]. 

Gold document 49556906, original sentence indices [5]: Pharmacological activation of AMPK in myofibroblasts from lungs of humans with IPF display lower fibrotic activity, along with enhanced mitochondrial biogenesis and normalization of sensitivity to apoptosis.

No document survived the learned selector and validation-tuned acceptance filters.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.
