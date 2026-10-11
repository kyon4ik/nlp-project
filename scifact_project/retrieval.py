from pathlib import Path
import gc
import hashlib
import json
import time
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from .data import document_text, tokens, write_json, sha256
from .model_revisions import revision


def sorted_ids(scores, doc_ids, count=50):
    # Stable ties preserve the official corpus order.
    return [int(doc_ids[i]) for i in np.argsort(-np.asarray(scores), kind="stable")[:count]]


class BM25Retriever:
    def __init__(self, dataset, title=True):
        self.ids = dataset.doc_ids
        self.title = title
        self.model = BM25Okapi([tokens(document_text(d, title)) for d in dataset.documents])

    def rank(self, text, count=50):
        scores = self.model.get_scores(tokens(text))
        ids = sorted_ids(scores, self.ids, count)
        mapping = {int(d): float(s) for d, s in zip(self.ids, scores)}
        return ids, [mapping[d] for d in ids]

    def rankings(self, claims, count=50):
        rankings, scores = {}, {}
        start = time.perf_counter()
        for c in claims:
            ids, values = self.rank(c["claim"], count)
            rankings[str(c["id"])], scores[str(c["id"])] = ids, values
        return rankings, scores, time.perf_counter() - start


def adaptive_encode(model, texts, batch_size, **kwargs):
    """Retry inference with smaller batches on CUDA OOM, never silently shorten inputs."""
    while True:
        try:
            return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False, **kwargs)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()


def corpus_embeddings(model, dataset, config, tag):
    texts = [document_text(d) for d in dataset.documents]
    # Fingerprint every text, model tag and truncation length to reject stale caches.
    checkpoint_file = config.out / 'checkpoints' / f'dense_{tag}' / 'model.safetensors'
    weight_id = sha256(checkpoint_file) if checkpoint_file.exists() else revision(config.dense_model)
    fingerprint = hashlib.sha256(json.dumps([tag, config.dense_model, weight_id, config.dense_length, texts]).encode()).hexdigest()[:16]
    path = config.out / "cache" / f"corpus_{tag}_{fingerprint}.npy"
    if path.exists():
        return np.load(path)
    embeddings = adaptive_encode(model, texts, config.inference_batch)
    np.save(path, embeddings)
    write_json(path.with_suffix(".json"), {"fingerprint": fingerprint, "doc_ids": dataset.doc_ids.tolist(),
                                         "shape": list(embeddings.shape), "normalized": True})
    return embeddings


def dense_rankings(model, embeddings, dataset, claims, config, count=50):
    start = time.perf_counter()
    queries = adaptive_encode(model, [c["claim"] for c in claims], config.inference_batch)
    similarities = queries @ embeddings.T
    rankings, scores = {}, {}
    for c, row in zip(claims, similarities):
        indices = np.argsort(-row, kind="stable")[:count]
        rankings[str(c["id"])] = [int(dataset.doc_ids[i]) for i in indices]
        scores[str(c["id"])] = [float(row[i]) for i in indices]
    return rankings, scores, time.perf_counter() - start


def load_dense(config, path=None):
    source = str(path or config.dense_model)
    model = SentenceTransformer(source, device=config.device, revision=revision(source))
    model.max_seq_length = config.dense_length
    return model


def rrf_rankings(left, right, constant=60, weight=0.5, count=50):
    result, scores = {}, {}
    for claim_id in left:
        fused = {}
        for ranking, w in [(left[claim_id], weight), (right[claim_id], 1 - weight)]:
            for rank, doc in enumerate(ranking, 1):
                fused[int(doc)] = fused.get(int(doc), 0.0) + w / (constant + rank)
        docs = sorted(fused, key=lambda d: (-fused[d], d))[:count]
        result[claim_id] = docs
        scores[claim_id] = [fused[d] for d in docs]
    return result, scores


def rerank(claims, rankings, dataset, config):
    model = CrossEncoder(config.reranker_model, device=config.device, max_length=config.rerank_length,
                         revision=revision(config.reranker_model))
    pairs, metadata = [], []
    for c in claims:
        for doc in rankings[str(c["id"])][:config.candidate_count]:
            pairs.append([c["claim"], document_text(dataset.corpus[doc])])
            metadata.append((str(c["id"]), doc))
    start = time.perf_counter()
    batch_size = config.inference_batch
    while True:
        try:
            values = model.predict(pairs, batch_size=batch_size, show_progress_bar=False,
                                   activation_fn=torch.nn.Identity(), convert_to_numpy=True)
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
    grouped = {str(c["id"]): [] for c in claims}
    for (cid, doc), score in zip(metadata, values):
        grouped[cid].append((doc, float(score)))
    output = {cid: [d for d, score in sorted(pairs, key=lambda p: (-p[1], p[0]))]
              for cid, pairs in grouped.items()}
    scores = {cid: [score for d, score in sorted(pairs, key=lambda p: (-p[1], p[0]))]
              for cid, pairs in grouped.items()}
    duration = time.perf_counter() - start
    del model
    gc.collect()
    return output, scores, duration


def specter_rankings(dataset, splits, config):
    """Official CLS encoding; proximity adapter for docs, adhoc adapter for claims."""
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base", revision=revision("allenai/specter2_base"))
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base", revision=revision("allenai/specter2_base"))
    model.load_adapter("allenai/specter2", source="hf", load_as="document", version=revision("allenai/specter2"))
    model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="query", version=revision("allenai/specter2_adhoc_query"))
    # Adapter modules are created on CPU. Move only after loading both adapters.
    model = model.to(config.device).eval()
    assert all(p.device.type == config.device for p in model.parameters())

    @torch.inference_mode()
    def encode(texts, adapter):
        model.set_active_adapters(adapter)
        arrays = []
        batch = min(config.inference_batch, 8 if config.device == "cpu" else 32)
        position = 0
        while position < len(texts):
            try:
                inputs = tokenizer(texts[position:position + batch], padding=True, truncation=True,
                                   max_length=512, return_tensors="pt", return_token_type_ids=False).to(config.device)
                vectors = model(**inputs).last_hidden_state[:, 0, :]
                arrays.append(torch.nn.functional.normalize(vectors.float(), dim=1).cpu().numpy())
                position += batch
                if position % 400 == 0:
                    print(f"SPECTER2 {adapter}: {position}/{len(texts)}", flush=True)
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
        return np.concatenate(arrays)

    texts = [d["title"] + tokenizer.sep_token + " ".join(d["abstract"]) for d in dataset.documents]
    fingerprint = hashlib.sha256(json.dumps([revision('allenai/specter2_base'), revision('allenai/specter2'), texts]).encode()).hexdigest()[:16]
    path = config.out / "cache" / f"specter2_documents_{fingerprint}.npy"
    start = time.perf_counter()
    if path.exists():
        docs = np.load(path)
    else:
        docs = encode(texts, "document")
        np.save(path, docs)
        write_json(path.with_suffix('.json'), {'doc_ids': dataset.doc_ids.tolist(), 'shape': list(docs.shape),
                   'maximum_tokens': 512, 'pooling': 'CLS', 'normalization': 'L2',
                   'base_revision': revision('allenai/specter2_base'), 'adapter_revision': revision('allenai/specter2')})
    indexing_seconds = time.perf_counter() - start
    results = {}
    for name, claims in splits.items():
        start = time.perf_counter()
        queries = encode([c["claim"] for c in claims], "query")
        similarities = queries @ docs.T
        rankings, scores = {}, {}
        for c, row in zip(claims, similarities):
            order = np.argsort(-row, kind="stable")[:50]
            rankings[str(c["id"])] = [int(dataset.doc_ids[i]) for i in order]
            scores[str(c["id"])] = [float(row[i]) for i in order]
        results[name] = (rankings, scores, time.perf_counter() - start)
    del model
    gc.collect()
    return results, indexing_seconds
