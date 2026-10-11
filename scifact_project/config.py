from dataclasses import asdict, dataclass
from pathlib import Path
import os
import random
import numpy as np
import torch


@dataclass
class Config:
    mode: str = "full"
    profile: str = "auto"
    seed: int = 42
    split_seed: int = 42
    data_dir: str = "data/raw/data"
    output_dir: str = "outputs/full_seed42"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    pair_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    dense_epochs: int = 2
    selector_epochs: int = 3
    stance_epochs: int = 5
    batch_size: int = 16
    inference_batch: int = 32
    dense_length: int = 256
    pair_length: int = 192
    rerank_length: int = 256
    learning_rate: float = 2e-5
    temperature: float = 0.05
    patience: int = 2
    candidate_count: int = 20
    verification_documents: int = 5
    sentence_budget: int = 3
    validation_fraction: float = 0.2
    specter2: bool = True
    threads: int = 4

    @classmethod
    def for_mode(cls, mode="full", profile="auto", seed=42, **overrides):
        if mode not in {"smoke", "full", "extended"}:
            raise ValueError(mode)
        if profile == "auto":
            profile = "gpu" if torch.cuda.is_available() else "cpu"
        if profile not in {"cpu", "gpu"}:
            raise ValueError(profile)
        values = dict(mode=mode, profile=profile, seed=seed,
                      output_dir=f"outputs/{mode}_{profile}_seed{seed}")
        if profile == "gpu":
            values.update(pair_model="allenai/scibert_scivocab_uncased",
                          dense_epochs=4, selector_epochs=4, stance_epochs=6,
                          inference_batch=64, pair_length=256, rerank_length=512)
        if mode == "smoke":
            values.update(dense_epochs=1, selector_epochs=1, stance_epochs=1,
                          batch_size=4, inference_batch=8, candidate_count=5,
                          dense_length=128, pair_length=128, specter2=False)
        if mode == "extended":
            values.update(dense_epochs=5, selector_epochs=5, stance_epochs=8)
        values.update(overrides)
        return cls(**values)

    @property
    def device(self):
        return "cuda" if self.profile == "gpu" and torch.cuda.is_available() else "cpu"

    @property
    def out(self):
        return Path(self.output_dir)

    def prepare(self):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
        for name in ["cache", "checkpoints", "metrics", "predictions", "figures", "tables"]:
            (self.out / name).mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(self.threads)
        set_seed(self.seed)
        return self

    def dictionary(self):
        return asdict(self)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
