## Measured findings

This is a **SMOKE TEST: reduced data and corpus; not a scientific result** with one training seed (42).
Retrieval is evaluated on 12 evidence-bearing official dev claims;
end-to-end metrics include all 16 dev claims. No test metrics are reported.

The highest dev MRR@10 is **bm25_title_abstract**, 1.0000,
+0.00 percentage points relative to title+abstract BM25.
The highest Recall@10 is **hybrid_pretrained**, 1.0000.
These are descriptive comparisons across prespecified experiments, not dev-driven model selection.
Deployment uses hybrid_random followed by reranking, chosen with internal validation labels only.

Adding titles to BM25 changes MRR@10 by +4.17 points and
Recall@10 by +0.00 points.

Pretrained MiniLM versus BM25 changes MRR@10 by -13.89 points
and Recall@10 by -8.33 points.
Random-negative fine-tuning versus pretrained MiniLM changes MRR@10 by
+0.00 points and Recall@10 by
+0.00 points.
Hard negatives versus random negatives change MRR@10 by +0.00 points
and Recall@10 by +0.00 points.
The hybrid_random fusion changes MRR@10 versus its trained dense input by
+6.48 points
and versus BM25 by -7.41 points.
Reranking the exact same 5 hybrid candidates changes Recall@1 by
+0.00 points and MRR@10 by
-0.93 points. Its candidate Recall@20 cannot improve when the pool is fixed.

The trained gold-rationale stance model has accuracy 0.4444 and macro F1
0.3077; the TF-IDF baseline has macro F1 0.4156.
Gold-rationale examples include every alternative rationale separately, so this table has a different
denominator from the claim-level pipeline and should not be compared as if the tasks were identical.

The predicted pipeline has claim macro F1 0.4630 and official rationalized abstract F1
0.2857. Replacing retrieved documents with gold documents changes the latter by
+2.20 points; additionally replacing selected evidence with a complete gold rationale changes it by
+7.69 points. The larger oracle difference is associated with
evidence selection.
These interventions alter both coverage and false positives and are diagnostics, not an additive causal decomposition.
Gold evidence uses the shortest complete alternative per document; the deployed pipeline never sees annotations.

## Interpretation and limitations

Metric changes above explicitly retain their signs: an increase on one metric does not establish an increase on all metrics.
The retrieval plots and tables show where early-rank quality and recall disagree. Paired bootstrap intervals are saved
for MRR differences; their interpretation concerns evaluation-claim sampling, not variation across training runs.
Only one seed was run in this configuration. Repeated runs are supported, but no unmeasured mean ± standard deviation is claimed.

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
Long abstracts are truncated: MiniLM uses 128 tokens, pair models 128, reranker
512; SPECTER2, when available, uses 512 and distinct document/query adapters. These unequal context
windows are resource choices and a confound when comparing architectures.

The claim aggregation policy chooses the accepted document with highest selector score × stance probability.
This is a project-specific policy, not the official task: official metrics operate on claim–document predictions.
NEI denotes absence of sufficient annotated evidence in this corpus, not proof that a scientific claim is false.

Three realistic next studies are: (1) repeat all trainable models for seeds 42, 43, and 44 on the same grouped split;
(2) train SciBERT on a GPU and evaluate longer context and joint rationale selection; (3) improve relevance calibration
with carefully audited negatives and train a scientific reranker while retaining the same candidate pools.

## Concrete error cases

**Claim 3: Stance or document aggregation error**

1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.

Gold claim aggregation: SUPPORT; prediction: CONTRADICT. Gold documents: [14717500]; retrieved top 5: [14717500, 4414547, 380526, 461550, 263364].

Gold document 14717500, original sentence indices [2, 5]: We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associations" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele. We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.

Accepted document 14717500, label CONTRADICT, stance score 0.678: [6] We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible for both hearing loss and sickle cell anemia create genome-wide significant synthetic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants. [7] In conclusion, uncommon or rare genetic variants can easily create synthetic associations that are credited to common variants, and this possibility requires careful consideration in the interpretation and follow up of GWAS signals. [5] We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 48: Retrieval missed every evidence document**

A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection.

Gold claim aggregation: CONTRADICT; prediction: NOT_ENOUGH_INFO. Gold documents: [13734012]; retrieved top 5: [380526, 253672, 92308, 601033, 14717500].

Gold document 13734012, original sentence indices [4]: RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million).

No document survived the learned selector and validation-tuned acceptance filters.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 5: Stance or document aggregation error**

1/2000 in UK have abnormal PrP positivity.

Gold claim aggregation: SUPPORT; prediction: CONTRADICT. Gold documents: [13734012]; retrieved top 5: [13734012, 456304, 4414547, 103007, 14717500].

Gold document 13734012, original sentence indices [4]: RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million).

Accepted document 13734012, label CONTRADICT, stance score 0.682: [7] CONCLUSIONS This study corroborates previous studies and suggests a high prevalence of infection with abnormal PrP, indicating vCJD carrier status in the population compared with the 177 vCJD cases to date. [4] RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million). [6] Genetic testing of the positive specimens for the genotype at PRNP codon 129 revealed a high proportion that were valine homozygous compared with the frequency in the normal population, and in stark contrast with confirmed clinical cases of vCJD, all of which were methionine homozygous at PRNP codon 129.   


The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 49: Stance or document aggregation error**

ADAR1 binds to Dicer to cleave pre-miRNA.

Gold claim aggregation: SUPPORT; prediction: CONTRADICT. Gold documents: [5953485]; retrieved top 5: [5953485, 654735, 153744, 164985, 461550].

Gold document 5953485, original sentence indices [1]: In this study, we investigated the interaction of the RNA editing mechanism with the RNA interference (RNAi) machinery and found that ADAR1 forms a complex with Dicer through direct protein-protein interaction.

Accepted document 5953485, label CONTRADICT, stance score 0.703: [2] Most importantly, ADAR1 increases the maximum rate (Vmax) of pre-microRNA (miRNA) cleavage by Dicer and facilitates loading of miRNA onto RNA-induced silencing complexes, identifying a new role of ADAR1 in miRNA processing and RNAi mechanisms. [1] In this study, we investigated the interaction of the RNA editing mechanism with the RNA interference (RNAi) machinery and found that ADAR1 forms a complex with Dicer through direct protein-protein interaction. [3] ADAR1 differentiates its functions in RNA editing and RNAi by the formation of either ADAR1/ADAR1 homodimer or Dicer/ADAR1 heterodimer complexes, respectively.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 50: Stance or document aggregation error**

AIRE is expressed in some skin tumors.

Gold claim aggregation: SUPPORT; prediction: CONTRADICT. Gold documents: [12580014]; retrieved top 5: [12580014, 301838, 496873, 293661, 275294].

Gold document 12580014, original sentence indices [1]: We report that autoimmune regulator (Aire), a transcriptional regulator, is inducibly expressed in human and mouse tumor keratinocytes in a K17-dependent manner and is required for timely onset of Gli2-induced skin tumorigenesis in mice.

Accepted document 12580014, label CONTRADICT, stance score 0.713: [2] The induction of Aire mRNA in keratinocytes depends on a functional interaction between K17 and the heterogeneous nuclear ribonucleoprotein hnRNP K. Further, K17 colocalizes with Aire protein in the nucleus of tumor-prone keratinocytes, and each factor is bound to a specific promoter region featuring an NF-κB consensus sequence in a relevant subset of K17- and Aire-dependent proinflammatory genes. [1] We report that autoimmune regulator (Aire), a transcriptional regulator, is inducibly expressed in human and mouse tumor keratinocytes in a K17-dependent manner and is required for timely onset of Gli2-induced skin tumorigenesis in mice. [3] These findings provide radically new insight into keratin intermediate filament and Aire function, along with a molecular basis for the K17-dependent amplification of inflammatory and immune responses in diseased epithelia.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.
