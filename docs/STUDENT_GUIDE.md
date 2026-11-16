# How each step works

1. **Load the data (`data.py`).** SciFact provides claims, paper abstracts and human-marked evidence. An abstract is already split into sentences; evidence points to their indices.
2. **Separate learning from evaluation (`data.py`).** The cleaned training data is split into 656 training and 151 validation claims. Related citation families stay together to reduce leakage. Two exact train/dev text duplicates are removed. Models learn on training data; validation chooses thresholds and settings. The 300 official dev claims measure results afterward. Test labels are unavailable.
3. **Find papers (`retrieval.py`).** BM25 scores matching words, giving more weight to informative words. MiniLM turns claims and abstracts into vectors: similar directions indicate similar meaning. Contrastive training pulls a claim toward its evidence paper and away from negative examples. Random negatives are sampled papers; hard negatives are confusing high-ranked papers. Unannotated negatives can still be relevant.
4. **Combine and reorder (`experiment.py`, `retrieval.py`).** Reciprocal rank fusion combines keyword and vector rankings using rank positions. A cross-encoder then reads the claim and each candidate together to reorder the top 20. Reordering cannot recover a paper absent from those 20 candidates. The verification stage considers five documents.
5. **Choose sentences (`training.py`, `pipeline.py`).** A trained selector scores claim–sentence pairs. A validation-chosen threshold and a maximum budget of three sentences determine the evidence. The simpler comparison uses embedding similarity. A correct paper with incomplete evidence can still fail evaluation.
6. **Predict stance (`training.py`, `pipeline.py`).** SciBERT reads the claim and selected evidence to predict support or contradiction. Relevance and confidence filters allow abstention. The claim-level policy chooses the accepted document with the highest selector score multiplied by stance probability. These scores are not calibrated probabilities of truth.
7. **Measure performance (`metrics.py`).** Recall@10 averages the fraction of known relevant papers retrieved in the first ten. MRR@10 averages the reciprocal rank of the first relevant paper (rank 2 gives 0.5; none in ten gives 0). Accuracy is the fraction of correct labels. Precision measures how many predictions are correct; recall measures how many gold answers are found; F1 is their harmonic mean. Macro F1 averages class F1 values equally. Official rationalized abstract F1 requires the correct paper, stance and a complete evidence rationale.
8. **Inspect mistakes (`report.py`).** Compare normal predictions with gold-document and gold-evidence settings. “Gold” means human-provided answers supplied for diagnosis, unavailable at deployment. These comparisons help locate weaknesses but do not isolate independent causal effects.

Read `config.py` for experiment settings and `scripts/run_experiments.py` for stage order.
Start with BM25 versus pretrained MiniLM; then study fusion, reranking, sentence selection,
and stance. SPECTER2 and repeated-seed scripts are optional advanced comparisons.

For practice, explain why a system can find the right paper yet receive a wrong final verdict.
Then compare the three verification settings in the report: which missing information helps most?
