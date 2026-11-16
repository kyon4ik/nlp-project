# Scientific claim verification — student guide

Given a scientific claim, this project finds papers, selects evidence sentences,
and predicts **SUPPORT**, **CONTRADICT**, or **NOT_ENOUGH_INFO**.
The last label means insufficient evidence in this corpus, not that the claim is false.

Start with [the step-by-step explanation](docs/STUDENT_GUIDE.md) and
[the short results report](REPORT.md). The original implementation and experiments
remain available; [research details](docs/RESEARCH_DETAILS.md) describe advanced options.

## Easiest way to start

From the project root, display the saved results (Python only; no GPU or downloads):

```bash
python scripts/show_results.py
```

To run experiments, create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_experiments.py --mode smoke --profile cpu
```

The smoke run downloads needed data/models and checks the workflow on a small subset.
Its scores are not research results. For the full GPU experiment use:

```bash
python scripts/run_experiments.py --mode full --profile gpu
```

Full runs can take substantial time. Use a new `--output-dir` when changing settings.
If CUDA is hidden by a sandbox, verify GPU access outside it before switching to CPU.

## Where things live

| Location | Purpose |
|---|---|
| `scifact_project/` | Data loading, models, evaluation, reporting |
| `scripts/run_experiments.py` | Main experiment command |
| `scripts/show_results.py` | Read saved metrics without loading models |
| `outputs/full_gpu_seed42/` | Saved main experiment and metrics |
| `tests/` | Checks for scoring and pipeline behavior |
| `scientific_claim_verification.ipynb` | Detailed executable research notebook |
| `vendor/scifact/` | Official evaluation code |
