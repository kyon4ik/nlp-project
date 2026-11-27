# Ten-minute defense run sheet

Present continuously in English. The timings appear in the PowerPoint speaker notes and slide headings. Agree on speaker handoffs in rehearsal and have every team member explain the complete pipeline.

| Time | Slides | What to show and explain |
|---|---:|---|
| 0:00–1:00 | 1 | Frame the literature-verification task, intended users, three labels and evidence-based success criteria. |
| 1:00–2:30 | 2–3 | Give dataset counts and provenance, explain split hygiene and trace raw text into BM25 words and neural token IDs. |
| 2:30–4:30 | 4–5 | Follow the abstract and sentence pipeline. Explain mean pooling, attention across the two segments within each pair encoder, classification heads, and the abstention policy. |
| 4:30–6:30 | 6–7 | Cover optimizer, loss, training and selection, then point to the best validation epochs, time and settings. |
| 6:30–8:30 | 8–9 | Define the metrics, compare the fixed baseline and full pipeline, then show a correct example and the reversal error. |
| 8:30–10:00 | 10–11 | Run a new or instructor-provided claim, explain the trace and result, then give the main finding, limitations and next experiment. |

## Before the defense

- Open the [PowerPoint](../presentation/scientific_claim_verification.pptx) in presenter view and keep the [PDF](../presentation/scientific_claim_verification.pdf) available.
- Open the offline prompt and wait for it to say “Verifier ready.” Keep the project report and `outputs/full_gpu_seed42` open so any team member can navigate to the configurations and artifacts.
- Share the project URL in the group and confirm the instructor can open the submission revision. [GitHub repository](https://github.com/kyon4ik/nlp-project)

## Live demonstration

```bash
HF_HUB_OFFLINE=1 python scripts/demo.py --device cuda
```

Enter an instructor-provided English claim. If none is provided, enter **“Moonlight doubles the growth rate of broccoli seedlings.”** This text does not appear in the released claim splits. The rehearsed run abstained. Point out what the lexical tokenizer keeps, the subword split of “moonlight,” and where the claim and context separate. Read one attention mask entry and explain how padding is handled in a batch. Point to either the winning evidence and two stance scores or the candidate scores behind abstention.

A second rehearsed claim, **“Smoking increases the likelihood of developing lung cancer,”** received CONTRADICT with stance softmax 0.942 even though its abstract about smoking reduction reports results consistent with a higher risk among persistent smokers. If the demonstration returns an unexpected result, explain whether the evidence addresses the claim and proceed to the conclusion.

## Individual practice

| Question | Points every team member should be able to explain |
|---|---|
| How were the data and labels produced? | Experts wrote claims from citation sentences, annotated abstract stances and identified alternative minimal rationales. NEI is corpus-relative at claim level. |
| How was leakage reduced? | Exact train/dev text matches were removed before training. Citation-family groups separate train from internal validation. Official dev labels were excluded from tuning. A text-only audit flagged 48 official dev claims for high lexical similarity and gives a 252-claim sensitivity check. |
| What goes into the Transformer? | WordPieces map to IDs, plus type IDs and attention masks. Pair truncation preserves the claim and shortens the context. Padding lets examples share a batch. |
| Why use two kinds of retrievers? | BM25 handles explicit matching, dense MiniLM finds semantic matches, and rank fusion combines them. The cross-encoder reads each of the top 20 pairs jointly and can only reorder those candidates. |
| What was trained, with what loss? | The shared MiniLM bi-encoder used masked multi-positive InfoNCE. The two full SciBERT models used class-weighted cross-entropy. AdamW updated their parameters. The pretrained reranker remained frozen. |
| What selected the models? | Retriever validation Recall@10, then MRR@10. Pair models used validation macro F1. The sentence threshold came from validation overlap F1, and the gates from validation rationalized abstract F1 and then claim macro F1. |
| What do the learning curves show? | The stance training loss continued falling after its best validation macro F1. The rise in validation loss indicates widening fit differences, consistent with overfitting. Dense validation loss was not measured. |
| Why are there two F1 scores? | Claim macro F1 weights all three labels equally. Rationalized abstract F1 counts abstract-level answers with the correct stance and a complete annotated rationale. |
| Which errors matter most? | Only 22 of 64 contradictions were identified. Claim 51 mislabels direction and selects a paper about DUSP4 for an ALDH1 claim. The oracle rows test how supplying annotated evidence changes observed scores. |
| Where can you show the implementation? | `data.py` has split logic, `training.py` has losses and optimization, `pipeline.py` has the decision rule, and `metrics.py` calls the official scorer. The run directory has `config.json`, `selection.json`, and the recorded checkpoint paths. |
