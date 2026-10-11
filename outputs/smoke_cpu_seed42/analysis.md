## Measured findings

This is a **SMOKE TEST: reduced data and corpus; not a scientific result** with one training seed (42).
Retrieval is evaluated on 12 evidence-bearing official dev claims;
end-to-end metrics include all 16 dev claims. No test metrics are reported.

The highest dev MRR@10 is **bm25_title_abstract**, 1.0000,
+0.00 percentage points relative to title+abstract BM25.
The highest Recall@10 is **hybrid_pretrained**, 1.0000.
These are descriptive comparisons across prespecified experiments, not dev-driven model selection.
Deployment uses hybrid_random followed by reranking, chosen with internal validation labels only.

Pretrained MiniLM versus BM25 changes MRR@10 by -13.89 points
and Recall@10 by -8.33 points.
Random-negative fine-tuning versus pretrained MiniLM changes MRR@10 by
+6.94 points and Recall@10 by
+8.33 points.
Hard negatives versus random negatives change MRR@10 by -0.35 points
and Recall@10 by +0.00 points.
The hybrid_random fusion changes MRR@10 versus its trained dense input by
+1.39 points
and versus BM25 by -5.56 points.
Reranking the exact same 5 hybrid candidates changes Recall@1 by
+8.33 points and MRR@10 by
+5.56 points. Its candidate Recall@20 cannot improve when the pool is fixed.

The trained gold-rationale stance model has accuracy 0.5000 and macro F1
0.4582; the TF-IDF baseline has macro F1 0.4156.
Gold-rationale examples include every alternative rationale separately, so this table has a different
denominator from the claim-level pipeline and should not be compared as if the tasks were identical.

The predicted pipeline has claim macro F1 0.5805 and official rationalized abstract F1
0.2941. Replacing retrieved documents with gold documents changes the latter by
+9.05 points; additionally replacing selected evidence with a complete gold rationale changes it by
+15.38 points. The larger oracle difference is associated with
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
256; SPECTER2, when available, uses 512 and distinct document/query adapters. These unequal context
windows are resource choices and a confound when comparing architectures.

The claim aggregation policy chooses the accepted document with highest selector score × stance probability.
This is a project-specific policy, not the official task: official metrics operate on claim–document predictions.
NEI denotes absence of sufficient annotated evidence in this corpus, not proof that a scientific claim is false.

Three realistic next studies are: (1) repeat all trainable models for seeds 42, 43, and 44 on the same grouped split;
(2) train SciBERT on a GPU and evaluate longer context and joint rationale selection; (3) improve relevance calibration
with carefully audited negatives and train a scientific reranker while retaining the same candidate pools.

## Concrete error cases

**Claim 3: Correct claim verdict but incomplete/wrong rationale or document**

1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.

Gold claim aggregation: SUPPORT; prediction: SUPPORT. Gold documents: [14717500]; retrieved top 5: [14717500, 4414547, 380526, 461550, 263364].

Gold document 14717500, original sentence indices [2, 5]: We propose as an alternative explanation that variants much less common than the associated one may create "synthetic associations" by occurring, stochastically, more often in association with one of the alleles at the common site versus the other allele. We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies.

Accepted document 14717500, label SUPPORT, stance score 0.508: [5] We show that they are not only possible, but inevitable, and that under simple but reasonable genetic models, they are likely to account for or contribute to many of the recently identified signals reported in genome-wide association studies. [4] Here, we use simple computer simulations to show the conditions under which such synthetic associations will arise and how they may be recognized. [6] We also illustrate the behavior of synthetic associations in real datasets by showing that rare causal mutations responsible for both hearing loss and sickle cell anemia create genome-wide significant synthetic associations, in the latter case extending over a 2.5-Mb interval encompassing scores of "blocks" of associated variants.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 36: Evidence predicted for an unannotated/NEI claim**

A deficiency of vitamin B12 increases blood levels of homocysteine.

Gold claim aggregation: NOT_ENOUGH_INFO; prediction: SUPPORT. Gold documents: []; retrieved top 5: [33409100, 11705328, 18174210, 520579, 275294].

Accepted document 33409100, label SUPPORT, stance score 0.503: [9] It was lowered 6.3 micromol/L (25.8%; P < .001) in the vitamin group and 0.4 micromol/L (1.7%; P = .14) in the placebo group at 3 months, but there was no significant effect on mortality (448 vitamin group deaths vs 436 placebo group deaths) (hazard ratio [HR], 1.04; 95% CI, 0.91-1.18). [8] RESULTS Mean baseline homocysteine level was 24.0 micromol/L in the vitamin group and 24.2 micromol/L in the placebo group. [4] Median follow-up was 3.2 years for 2056 participants aged 21 years or older with advanced chronic kidney disease (estimated creatinine clearance < or =30 mL/min) (n = 1305) or end-stage renal disease (n = 751) and high homocysteine levels (> or = 15 micromol/L).   


The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 42: Stance or document aggregation error**

A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+)- thalassemia trait subjects.

Gold claim aggregation: CONTRADICT; prediction: SUPPORT. Gold documents: [18174210]; retrieved top 5: [18174210, 33409100, 56893404, 11705328, 663464].

Gold document 18174210, original sentence indices [1, 9]: Individuals homozygous for alpha(+)-thalassaemia have microcytosis and an increased erythrocyte count. We estimated that the haematological profile in children homozygous for alpha(+)-thalassaemia reduces the risk of SMA during acute malaria compared to children of normal genotype (relative risk 0.52; 95% confidence interval [CI] 0.24-1.12, p = 0.09).   


Accepted document 18174210, label SUPPORT, stance score 0.512: [7] This model predicted that children homozygous for alpha(+)-thalassaemia lose less Hb than children of normal genotype for a reduction in erythrocyte count of >1.1 x 10(12)/l as a result of the reduced mean cell Hb in homozygous alpha(+)-thalassaemia. [5] We observed a reduction in median erythrocyte count of approximately 1.5 x 10(12)/l in all children with acute falciparum malaria relative to values in community children (p < 0.001). [4] METHODS AND FINDINGS Data from children living on the north coast of Papua New Guinea who had participated in a case-control study of the protection afforded by alpha(+)-thalassaemia against severe malaria were reanalysed to assess the genotype-specific reduction in erythrocyte count and Hb levels associated with acute malarial disease.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 48: Incorrect abstention after retrieval**

A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection.

Gold claim aggregation: CONTRADICT; prediction: NOT_ENOUGH_INFO. Gold documents: [13734012]; retrieved top 5: [13734012, 253672, 601033, 380526, 14717500].

Gold document 13734012, original sentence indices [4]: RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million).

No document survived the learned selector and validation-tuned acceptance filters.

The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.

**Claim 5: Correct claim verdict but incomplete/wrong rationale or document**

1/2000 in UK have abnormal PrP positivity.

Gold claim aggregation: SUPPORT; prediction: SUPPORT. Gold documents: [13734012]; retrieved top 5: [13734012, 323335, 4414547, 14717500, 70490].

Gold document 13734012, original sentence indices [4]: RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million).

Accepted document 13734012, label SUPPORT, stance score 0.502: [5] The prevalence in those born in 1941-60 (733 per million, 269 to 1596 per million) did not differ significantly from those born between 1961 and 1985 (412 per million, 198 to 758 per million) and was similar in both sexes and across the three broad geographical areas sampled. [7] CONCLUSIONS This study corroborates previous studies and suggests a high prevalence of infection with abnormal PrP, indicating vCJD carrier status in the population compared with the 177 vCJD cases to date. [6] Genetic testing of the positive specimens for the genotype at PRNP codon 129 revealed a high proportion that were valine homozygous compared with the frequency in the normal population, and in stark contrast with confirmed clinical cases of vCJD, all of which were methionine homozygous at PRNP codon 129.   


The failure category is assigned from observed stage outputs; it does not establish a linguistic cause without further manual review.
