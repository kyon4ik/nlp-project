"""Explicit PyTorch training loops to make negatives, masking and selection auditable."""
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from .config import set_seed
from .data import document_text, gold_docs, read_jsonl, write_json, write_jsonl
from .metrics import retrieval_metrics, classification_metrics
from .retrieval import load_dense, adaptive_encode, dense_rankings
from .model_revisions import revision


def _train_retriever(dataset, bm25, config, negative_mode, allow_resume=True):
    checkpoint = config.out / "checkpoints" / f"dense_{negative_mode}"
    history_path = config.out / "metrics" / f"dense_{negative_mode}_history.json"
    if allow_resume and (checkpoint / "modules.json").exists() and history_path.exists() and (checkpoint / "training_complete.json").exists():
        return load_dense(config, checkpoint), __import__("json").loads(history_path.read_text())
    set_seed(config.seed)
    model = load_dense(config)
    rng = np.random.default_rng(config.seed)
    group_positive = defaultdict(set)
    for c in dataset.train:
        group_positive[dataset.groups[c["id"]]].update(gold_docs(c))
    pairs = []
    for c in dataset.train:
        positives = gold_docs(c)
        # Protect family evidence and cited documents to reduce known false negatives.
        excluded = positives | group_positive[dataset.groups[c["id"]]] | set(c.get("cited_doc_ids", []))
        candidates = [int(d) for d in dataset.doc_ids if int(d) not in excluded]
        if negative_mode == "hard":
            ranking, _ = bm25.rank(c["claim"], min(len(dataset.doc_ids), 100))
            hard = [d for d in ranking if d not in excluded]
            candidates = hard[:20] or candidates
        for positive in sorted(positives):
            negative = int(rng.choice(candidates))
            pairs.append({"claim_id": c["id"], "claim": c["claim"], "positive": positive,
                          "negative": negative, "positives": sorted(positives),
                          "protected": sorted(excluded), "negative_mode": negative_mode})
    if not pairs:
        raise ValueError("No retriever positives")
    write_jsonl(config.out / "tables" / f"retriever_{negative_mode}_pairs.jsonl", pairs)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(pairs, batch_size=config.batch_size, shuffle=True, collate_fn=lambda x: x,
                        generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    total_steps = len(loader) * config.dense_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total_steps * .1)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=config.device == "cuda")
    best, stale, history = (-1., -1.), 0, []
    start = time.perf_counter()
    for epoch in range(config.dense_epochs):
        model.train()
        total_loss = examples = 0
        for step, batch in enumerate(loader):
            documents = list(dict.fromkeys([r["positive"] for r in batch] + [r["negative"] for r in batch]))
            query_inputs = {k: v.to(config.device) for k, v in model.tokenize([r["claim"] for r in batch]).items()}
            doc_inputs = {k: v.to(config.device) for k, v in model.tokenize(
                [document_text(dataset.corpus[d]) for d in documents]).items()}
            positive_mask = torch.tensor([[d in r["positives"] for d in documents] for r in batch], device=config.device)
            allowed_mask = torch.tensor([[d not in r["protected"] or d in r["positives"]
                                          for d in documents] for r in batch], device=config.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=config.device, dtype=torch.float16, enabled=config.device == "cuda"):
                q = torch.nn.functional.normalize(model(query_inputs)["sentence_embedding"], dim=1)
                d = torch.nn.functional.normalize(model(doc_inputs)["sentence_embedding"], dim=1)
                logits = (q @ d.T).float() / config.temperature
                logits = logits.masked_fill(~allowed_mask, -1e4)
                # Multi-positive InfoNCE prevents another known positive from becoming a negative.
                numerator = torch.logsumexp(logits.masked_fill(~positive_mask, -1e4), dim=1)
                loss = (torch.logsumexp(logits, dim=1) - numerator).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(batch)
            examples += len(batch)
            if step % 10 == 0:
                print(f"dense {negative_mode} epoch {epoch + 1} step {step + 1}/{len(loader)} loss={loss.item():.4f}", flush=True)
        model.eval()
        embeddings = adaptive_encode(model, [document_text(d) for d in dataset.documents], config.inference_batch)
        rankings, _, _ = dense_rankings(model, embeddings, dataset, dataset.validation, config)
        metrics = retrieval_metrics(dataset.validation, rankings)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / examples,
                        "validation": metrics, "effective_batch_size": config.batch_size,
                        "elapsed_seconds": time.perf_counter() - start})
        score = (metrics["Recall@10"], metrics["MRR@10"])
        print(f"dense {negative_mode} validation: {metrics}", flush=True)
        if score > best:
            best, stale = score, 0
            model.save(str(checkpoint))
        else:
            stale += 1
        write_json(history_path, history)
        if stale >= config.patience:
            break
    del model, optimizer, scheduler, loader
    gc.collect()
    write_json(checkpoint / "training_complete.json", {"epochs_run": len(history), "seed": config.seed,
                                                       "effective_batch_size": config.batch_size})
    return load_dense(config, checkpoint), history


def downsample_evidence(rows, seed, ratio=3):
    """Keep all positive sentences; negative sampling is training-only."""
    rng = np.random.default_rng(seed)
    positives = [r for r in rows if r["label"] == 1]
    negatives = [r for r in rows if r["label"] == 0]
    size = min(len(negatives), ratio * len(positives))
    indices = rng.choice(len(negatives), size=size, replace=False)
    return positives + [negatives[i] for i in indices]


class PairClassifier:
    def __init__(self, config, path=None):
        self.config = config
        source = str(path or config.pair_model)
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision(source))
        self.model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2, revision=revision(source)).to(config.device)

    def encode(self, rows):
        return self.tokenizer([r['claim'] for r in rows], [r['text'] for r in rows],
                              padding=True, truncation='only_second', max_length=self.config.pair_length,
                              return_tensors='pt')

    @torch.inference_mode()
    def probabilities(self, rows):
        if not rows:
            return np.empty((0, 2))
        self.model.eval()
        outputs = []
        position, batch = 0, self.config.inference_batch
        while position < len(rows):
            try:
                chunk = rows[position:position + batch]
                encoded = self.encode(chunk).to(self.config.device)
                outputs.append(self.model(**encoded).logits.float().softmax(-1).cpu().numpy())
                position += batch
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
        return np.concatenate(outputs)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def _train_pair_classifier(train_rows, validation_rows, config, task, allow_resume=True):
    checkpoint = config.out / "checkpoints" / task
    history_path = config.out / "metrics" / f"{task}_history.json"
    if allow_resume and (checkpoint / "config.json").exists() and history_path.exists() and (checkpoint / "training_complete.json").exists():
        return PairClassifier(config, checkpoint), __import__("json").loads(history_path.read_text())
    set_seed(config.seed)
    classifier = PairClassifier(config)
    tokenizer, model = classifier.tokenizer, classifier.model
    label_names = ['NOT_EVIDENCE', 'EVIDENCE'] if task == 'selector' else ['CONTRADICT', 'SUPPORT']
    model.config.id2label = dict(enumerate(label_names))
    model.config.label2id = {label: i for i, label in enumerate(label_names)}
    # Cached per-example tokenization, dynamic padding within batches.
    encoded = tokenizer([r["claim"] for r in train_rows], [r["text"] for r in train_rows],
                        truncation="only_second", max_length=config.pair_length)
    examples = [{**{k: v[i] for k, v in encoded.items()}, "labels": r["label"]} for i, r in enumerate(train_rows)]

    def collate(batch):
        labels = torch.tensor([r["labels"] for r in batch], dtype=torch.long)
        padded = tokenizer.pad([{k: v for k, v in r.items() if k != "labels"} for r in batch],
                               padding=True, return_tensors="pt")
        return padded, labels

    loader = DataLoader(examples, batch_size=config.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(config.seed))
    counts = np.bincount([r["label"] for r in train_rows], minlength=2)
    weights = torch.tensor(len(train_rows) / (2 * np.maximum(counts, 1)), dtype=torch.float, device=config.device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    epochs = config.selector_epochs if task == "selector" else config.stance_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(len(loader) * epochs * .1)), len(loader) * epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.device == "cuda")
    start = time.perf_counter()
    history, best, stale = [], -1., 0
    write_json(config.out / "tables" / f"{task}_training_summary.json",
               {"train_examples": len(train_rows), "validation_examples": len(validation_rows),
                "train_class_counts": counts.tolist(), "class_weights": weights.tolist(),
                "base_model": config.pair_model, "max_length": config.pair_length})
    for epoch in range(epochs):
        model.train()
        total_loss = seen = 0
        for step, (inputs, labels) in enumerate(loader):
            inputs = {k: v.to(config.device) for k, v in inputs.items()}
            labels = labels.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=config.device, dtype=torch.float16, enabled=config.device == "cuda"):
                logits = model(**inputs).logits
                loss = loss_function(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(labels)
            seen += len(labels)
            if step % 25 == 0:
                print(f"{task} epoch {epoch + 1} step {step + 1}/{len(loader)} loss={loss.item():.4f}", flush=True)
        probabilities = classifier.probabilities(validation_rows)
        labels = [r["label"] for r in validation_rows]
        val_loss = torch.nn.functional.nll_loss(torch.from_numpy(np.log(np.clip(probabilities, 1e-8, 1))),
                                               torch.tensor(labels)).item()
        metrics = classification_metrics(labels, probabilities.argmax(1), ["Not Evidence", "Evidence"]
                                          if task == "selector" else ["CONTRADICT", "SUPPORT"])
        score = metrics["macro_f1"]
        history.append({"epoch": epoch + 1, "train_loss": total_loss / seen,
                        "validation_loss": val_loss, "validation": metrics,
                        "effective_batch_size": config.batch_size,
                        "elapsed_seconds": time.perf_counter() - start})
        print(f"{task} validation macro F1={score:.4f}", flush=True)
        if score > best:
            best, stale = score, 0
            classifier.save(checkpoint)
        else:
            stale += 1
        write_json(history_path, history)
        if stale >= config.patience:
            break
    del classifier, model, optimizer, scheduler, loader
    gc.collect()
    write_json(checkpoint / "training_complete.json", {"epochs_run": len(history), "seed": config.seed,
                                                       "effective_batch_size": config.batch_size})
    return PairClassifier(config, checkpoint), history


def _retry_training(function, arguments, config):
    effective, resume = config, True
    while True:
        try:
            return function(*arguments, config=effective, allow_resume=resume)
        except torch.cuda.OutOfMemoryError:
            if effective.batch_size == 1:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            effective = replace(effective, batch_size=max(1, effective.batch_size // 2),
                                inference_batch=max(1, effective.inference_batch // 2))
            resume = False
            print(f"CUDA OOM: restarting this model from the original seed with batch={effective.batch_size}", flush=True)


def train_retriever(dataset, bm25, config, negative_mode):
    from functools import partial
    return _retry_training(partial(_train_retriever, negative_mode=negative_mode), (dataset, bm25), config)


def train_pair_classifier(train_rows, validation_rows, config, task):
    from functools import partial
    return _retry_training(partial(_train_pair_classifier, task=task), (train_rows, validation_rows), config)
