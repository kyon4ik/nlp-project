# Scientific claim verification

Given an English scientific claim, the verifier searches SciFact abstracts, selects evidence sentences and predicts **SUPPORT**, **CONTRADICT**, or **NOT_ENOUGH_INFO**. The last label means the system abstained within the corpus. This project is intended for students and literature reviewers inspecting potential evidence.

- [Project report](REPORT.md) and [executed research notebook](scientific_claim_verification.ipynb)
- [Technical documentation](docs/RESEARCH_DETAILS.md) and [student guide](docs/STUDENT_GUIDE.md)
- [Defense slides (PowerPoint)](presentation/scientific_claim_verification.pptx), [PDF](presentation/scientific_claim_verification.pdf) and [10-minute run sheet](docs/DEFENSE.md)
- [Project repository](https://github.com/kyon4ik/nlp-project)

## Read the saved results

Run from the repository root with Python only.

```bash
python scripts/show_results.py
```

The headline run uses the 300-claim SciFact official dev set for final evaluation after selection on a separate internal validation set. Retrieval uses 188 evidence-bearing dev claims. Official test labels are unavailable.

## Set up, train and evaluate

Use a Python environment with the CPU or CUDA PyTorch build for your system. [Executed versions](outputs/full_gpu_seed42/requirements-executed.txt)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_experiments.py --mode smoke --profile cpu
python scripts/run_experiments.py --mode full --profile gpu --output-dir outputs/reproduction_gpu_seed42
python scripts/audit_run.py --run outputs/full_gpu_seed42
```

The runner downloads public SciFact JSONL and pinned pretrained models on first use. The smoke run uses a reduced corpus for integration checks. A full run fits trainable models, selects on internal validation, scores official dev and writes tables, plots and exported predictions. Use a fresh output directory when reproducing the full study or changing settings. The [technical documentation](docs/RESEARCH_DETAILS.md) gives stage commands, dataset details and the CPU profile.

## Run the live demo

On the prepared defense machine, run the interactive prompt and enter an English claim when it is ready.

```bash
HF_HUB_OFFLINE=1 python scripts/demo.py --device cuda
```

For a single prediction, pass a quoted claim. The same GPU-trained checkpoints run on CPU with `--device cpu`.

```bash
HF_HUB_OFFLINE=1 python scripts/demo.py --device cpu 'Moonlight doubles the growth rate of broccoli seedlings.'
```

A fresh checkout needs the SciFact corpus, trained weights, corpus embedding cache and pinned reranker weights before the demo. Use the full run above, then pass `--run outputs/reproduction_gpu_seed42`, or restore the ignored files from the matching run on the prepared machine. See [checkpoint and demo setup](docs/RESEARCH_DETAILS.md#checkpoint-loading-and-inference). The prompt uses the actual retrieval and inference pipeline.

## Repository map

| Location | Purpose |
|---|---|
| `scifact_project/` | Data loading, model training, retrieval, evaluation and inference |
| `scripts/run_experiments.py` | Full and staged experiments |
| `scripts/demo.py` | Interactive inference and benchmark |
| `scripts/audit_run.py` | Evaluation and text-overlap audits |
| `outputs/full_gpu_seed42/` | Main configuration, saved metrics, plots and model decisions |
| `scientific_claim_verification.ipynb` | Executed experiment report |
| `vendor/scifact/` | Unmodified official evaluation code and licensing |
