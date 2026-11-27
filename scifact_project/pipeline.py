"""Annotation-free prediction path; oracle inputs are explicit diagnostic arguments."""
from collections import defaultdict
from pathlib import Path
import gc
import hashlib
import json
import numpy as np
from .data import document_text, gold_docs, gold_sentences, tokens, LABELS, write_json
from .metrics import sentence_metrics, pipeline_metrics, classification_metrics
from .retrieval import adaptive_encode, load_dense, BM25Retriever, rrf_rankings
from .training import PairClassifier
from .model_revisions import revision


def sentence_score_maps(dataset, claims, rankings, selector, embedding_model, config, split):
    """Score retrieved + oracle documents once; the prediction path later chooses its own docs."""
    fingerprint = hashlib.sha256(json.dumps([rankings, [c["id"] for c in claims], config.dictionary()],
                                           sort_keys=True).encode()).hexdigest()[:16]
    cache = config.out / "cache" / f"sentence_scores_{split}_{fingerprint}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rows = []
    for c in claims:
        docs = set(rankings[str(c["id"])][:config.verification_documents]) | gold_docs(c)
        for doc in sorted(docs):
            for i, text in enumerate(dataset.corpus[doc]["abstract"]):
                rows.append({"claim_id": c["id"], "claim": c["claim"], "doc_id": doc,
                             "sentence_index": i, "text": text})
    unique_texts = list(dict.fromkeys(r["text"] for r in rows))
    vectors = adaptive_encode(embedding_model, unique_texts, config.inference_batch)
    lookup = {text: v for text, v in zip(unique_texts, vectors)}
    queries = adaptive_encode(embedding_model, [c["claim"] for c in claims], config.inference_batch)
    query_lookup = {c["id"]: v for c, v in zip(claims, queries)}
    probabilities = selector.probabilities(rows)[:, 1]
    output = {"similarity": defaultdict(dict), "supervised": defaultdict(dict)}
    for r, probability in zip(rows, probabilities):
        cid, doc, index = str(r["claim_id"]), str(r["doc_id"]), r["sentence_index"]
        for name, score in [("similarity", float(query_lookup[r["claim_id"]] @ lookup[r["text"]])),
                            ("supervised", float(probability))]:
            if doc not in output[name][cid]:
                output[name][cid][doc] = [0.] * len(dataset.corpus[int(doc)]["abstract"])
            output[name][cid][doc][index] = score
    write_json(cache, output)
    return output


def select_sentences(scores, threshold, budget=3):
    # Order by evidence score. Original indices retained; official scorer uses first 3 in this order.
    order = np.argsort(-np.asarray(scores), kind="stable")[:budget]
    return [int(i) for i in order if scores[i] >= threshold]


def winning_document(documents):
    return max(documents, key=lambda d: (d['aggregation_score'], -d['doc_id']), default=None)


def selected_map(claims, rankings, score_maps, threshold, config, oracle=False, force_budget=False):
    result = {}
    for c in claims:
        cid = str(c["id"])
        docs = sorted(gold_docs(c)) if oracle else rankings[cid][:config.verification_documents]
        result[cid] = {str(doc): select_sentences(score_maps[cid][str(doc)],
                                                 -float("inf") if force_budget else threshold,
                                                 config.sentence_budget) for doc in docs}
    return result


def tune_sentence_threshold(claims, rankings, score_maps, config, kind):
    grid = [.15, .25, .35, .45, .55, .65, .75, .85] if kind == "supervised" else [.15, .25, .35, .45, .55, .65]
    records = []
    for threshold in grid:
        selected = selected_map(claims, rankings, score_maps, threshold, config)
        records.append({"threshold": threshold, **sentence_metrics(claims, selected)})
    best = max(records, key=lambda r: (r["f1"], r["precision"]))
    return best["threshold"], records


def stance_score_maps(dataset, claims, score_maps, stance, config, split):
    """Predict stance for top-b evidence (each b=1..budget), avoiding repeated tuning inference."""
    fingerprint = hashlib.sha256(json.dumps(score_maps, sort_keys=True).encode()).hexdigest()[:16]
    path = config.out / "cache" / f"stance_scores_{split}_{fingerprint}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows, keys = [], []
    for c in claims:
        cid = str(c["id"])
        for doc, scores in score_maps[cid].items():
            order = np.argsort(-np.asarray(scores), kind="stable")[:config.sentence_budget]
            for size in range(1, len(order) + 1):
                indices = sorted(int(i) for i in order[:size])
                rows.append({"claim": c["claim"], "text": " ".join(dataset.corpus[int(doc)]["abstract"][i] for i in indices)})
                keys.append((cid, doc, str(size)))
    probabilities = stance.probabilities(rows)
    result = {str(c["id"]): {} for c in claims}
    for (cid, doc, size), probability in zip(keys, probabilities):
        result[cid].setdefault(doc, {})[size] = probability.tolist()
    write_json(path, result)
    return result


def compose_predictions(dataset, claims, rankings, sentence_scores, stance_scores, config,
                        threshold, stance_threshold=.5, document_threshold=-float("inf"),
                        retrieval_scores=None, oracle_docs=False, gold_evidence=False,
                        gold_stance_probabilities=None):
    output = {}
    for c in claims:
        cid = str(c["id"])
        docs = sorted(gold_docs(c)) if oracle_docs else rankings[cid][:config.verification_documents]
        evidence, details = {}, []
        score_lookup = dict(zip(rankings.get(cid, []), (retrieval_scores or {}).get(cid, [])))
        for doc in docs:
            key = str(doc)
            if gold_evidence:
                # One complete gold alternative: deterministic shortest rationale, then original order.
                rationale = min(c["evidence"][key], key=lambda r: (len(r["sentences"]), r["sentences"]))
                indices = sorted(rationale["sentences"])
                probabilities = np.array(gold_stance_probabilities[cid][key])
                selection_confidence = 1.
            else:
                scores = sentence_scores[cid][key]
                indices = select_sentences(scores, threshold, config.sentence_budget)
                if not indices:
                    continue
                probabilities = np.array(stance_scores[cid][key][str(len(indices))])
                selection_confidence = max(scores[i] for i in indices)
            stance_confidence = float(probabilities.max())
            retrieval_score = score_lookup.get(doc)
            # Keep stance acceptance identical across predicted- and gold-evidence oracles.
            # Gold documents alone justify bypassing the document-relevance filter.
            if (stance_confidence < stance_threshold or
                    (not oracle_docs and retrieval_score is not None and retrieval_score < document_threshold)):
                continue
            label = LABELS[int(probabilities.argmax())]
            evidence[key] = {"label": label, "sentences": indices}
            details.append({"doc_id": doc, "title": dataset.corpus[doc]["title"], "label": label,
                            "selection_score": float(selection_confidence), "stance_probability": stance_confidence,
                            "stance_probabilities": dict(zip(LABELS, probabilities.tolist())),
                            "retrieval_score": retrieval_score,
                            "evidence": [{"index": i, "text": dataset.corpus[doc]["abstract"][i]} for i in indices],
                            "aggregation_score": float(selection_confidence * stance_confidence)})
        chosen = winning_document(details)
        output[cid] = {"id": c["id"], "verdict": chosen["label"] if chosen else "NOT_ENOUGH_INFO",
                       "evidence": evidence, "documents": details,
                       "ranked_documents": [{"doc_id": int(d), "title": dataset.corpus[int(d)]["title"]}
                                            for d in rankings.get(cid, [])[:config.candidate_count]],
                       "abstained": chosen is None,
                       "aggregation": "Highest selection score × stance probability; ties use smallest document ID"}
    return output


def tune_pipeline(dataset, claims, rankings, sentence_scores, stance_scores, retrieval_scores, config, threshold):
    results = []
    for stance_threshold in [.5, .65, .8]:
        for document_threshold in [-10., -2., 0., 2.]:
            p = compose_predictions(dataset, claims, rankings, sentence_scores, stance_scores, config, threshold,
                                    stance_threshold, document_threshold, retrieval_scores)
            metrics = pipeline_metrics(claims, p)
            results.append({"stance_threshold": stance_threshold, "document_threshold": document_threshold,
                            "validation_official_abstract_f1": metrics["official"]["abstract_rationalized"]["f1"],
                            "validation_claim_macro_f1": metrics["macro_f1"]})
    return max(results, key=lambda r: (r["validation_official_abstract_f1"], r["validation_claim_macro_f1"])), results


def gold_stance_map(dataset, claims, stance):
    rows, keys = [], []
    for c in claims:
        for doc, rs in c.get("evidence", {}).items():
            if not rs:
                continue
            rationale = min(rs, key=lambda r: (len(r["sentences"]), r["sentences"]))
            rows.append({"claim": c["claim"], "text": " ".join(dataset.corpus[int(doc)]["abstract"][i]
                                                                for i in sorted(rationale["sentences"]))})
            keys.append((str(c["id"]), doc))
    probabilities = stance.probabilities(rows)
    result = {str(c["id"]): {} for c in claims}
    for (cid, doc), probability in zip(keys, probabilities):
        result[cid][doc] = probability.tolist()
    return result


class ClaimVerifier:
    """Reusable public prediction function. Loads trained weights and no claim annotations."""
    def __init__(self, config, dataset, selection):
        self.config, self.dataset, self.selection = config, dataset, selection
        self.bm25 = BM25Retriever(dataset)
        self.dense = load_dense(config, config.out / "checkpoints" / f"dense_{selection['dense_mode']}")
        self.embeddings = np.load(selection["dense_embeddings_path"])
        self.selector = PairClassifier(config, config.out / "checkpoints/selector")
        self.stance = PairClassifier(config, config.out / "checkpoints/stance")
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(config.reranker_model, device=config.device, max_length=config.rerank_length,
                                     revision=revision(config.reranker_model))

    def predict(self, claim, *, trace=False):
        config, dataset = self.config, self.dataset
        bm_ids, _ = self.bm25.rank(claim, 50)
        query = adaptive_encode(self.dense, [claim], config.inference_batch)[0]
        cosine = self.embeddings @ query
        ids = [int(dataset.doc_ids[i]) for i in np.argsort(-cosine, kind="stable")[:50]]
        fused, _ = rrf_rankings({"-1": bm_ids}, {"-1": ids}, self.selection["rrf_constant"], self.selection["rrf_weight"])
        candidates = fused["-1"][:config.candidate_count]
        pairs = [[claim, document_text(dataset.corpus[d])] for d in candidates]
        import torch
        batch_size = config.inference_batch
        while True:
            try:
                scores = self.reranker.predict(pairs, batch_size=batch_size,
                                               activation_fn=torch.nn.Identity(), show_progress_bar=False)
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
        # Match the batch evaluation's document-ID tie breaking exactly.
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))
        rankings = {"-1": [candidates[i] for i in order]}
        retrieval_scores = {"-1": [float(scores[i]) for i in order]}
        rows = [{"claim": claim, "text": sentence, "doc_id": doc, "index": i}
                for doc in rankings["-1"][:config.verification_documents]
                for i, sentence in enumerate(dataset.corpus[doc]["abstract"])]
        evidence_probabilities = self.selector.probabilities(rows)[:, 1]
        sentence_scores = {"-1": {str(d): [] for d in rankings["-1"][:config.verification_documents]}}
        for row, value in zip(rows, evidence_probabilities):
            sentence_scores["-1"][str(row["doc_id"])].append(float(value))
        # Lightweight context objects contain claim text only, never gold annotations.
        cs = [{"id": -1, "claim": claim}]
        stance_scores = {"-1": {}}
        stance_rows, keys = [], []
        for doc, values in sentence_scores["-1"].items():
            indices = select_sentences(values, self.selection["sentence_threshold"], config.sentence_budget)
            if indices:
                stance_rows.append({"claim": claim, "text": " ".join(dataset.corpus[int(doc)]["abstract"][i]
                                                                       for i in sorted(indices))})
                keys.append((doc, str(len(indices))))
        for (doc, size), probability in zip(keys, self.stance.probabilities(stance_rows)):
            stance_scores["-1"][doc] = {size: probability.tolist()}
        result = compose_predictions(dataset, cs, rankings, sentence_scores, stance_scores, config,
                                     self.selection["sentence_threshold"], self.selection["stance_threshold"],
                                     self.selection["document_threshold"], retrieval_scores)["-1"]
        if trace:
            classifier, head, example = self.selector, 'evidence selector', None
            winner = winning_document(result['documents'])
            if winner:
                evidence = sorted(winner['evidence'], key=lambda s: s['index'])
                classifier, head = self.stance, 'stance'
                example = {'claim': claim, 'text': ' '.join(s['text'] for s in evidence)}
            elif rows:
                example = rows[int(np.argmax(evidence_probabilities))]
            encoding = {}
            if example:
                encoding = {key: batch[0].tolist() for key, batch in classifier.encode([example]).items()}
            result['trace'] = {
                'bm25_tokens': tokens(claim), 'pair_head': head, 'pair_text': example['text'] if example else None,
                'wordpieces': classifier.tokenizer.convert_ids_to_tokens(encoding['input_ids']) if encoding else [],
                'encoding': encoding,
                'candidates': [
                    {'doc_id': doc, 'document_logit': score,
                     'best_evidence_score': max(sentence_scores['-1'][str(doc)], default=None),
                     'stance_probabilities': next(iter(stance_scores['-1'].get(str(doc), {}).values()), None)}
                    for doc, score in zip(rankings['-1'][:config.verification_documents], retrieval_scores['-1'])
                ],
            }
        return result
