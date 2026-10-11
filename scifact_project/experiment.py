"""Notebook-facing orchestration; saved artifacts make each stage safely resumable."""
from dataclasses import replace
from pathlib import Path
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from .config import Config
from .data import (Dataset, eda_tables, evidence_examples, stance_examples, write_json, write_jsonl,
                   document_text, gold_docs, gold_sentences, sha256, LABELS, claim_label)
from .metrics import retrieval_metrics, classification_metrics, sentence_metrics, pipeline_metrics
from .retrieval import (BM25Retriever, load_dense, corpus_embeddings, dense_rankings,
                        rrf_rankings, rerank, specter_rankings)
from .training import train_retriever, train_pair_classifier, downsample_evidence, PairClassifier
from .pipeline import (sentence_score_maps, stance_score_maps, selected_map, tune_sentence_threshold,
                       compose_predictions, tune_pipeline, gold_stance_map)


class Study:
    def __init__(self, config):
        self.config = config.prepare()
        config_path = config.out / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config.dictionary():
            raise ValueError("Existing run uses a different configuration. Choose a new output_dir.")
        write_json(config_path, config.dictionary())
        self.dataset = Dataset(config)
        self.splits = {"validation": self.dataset.validation, "dev": self.dataset.dev}
        self.rankings, self.scores, self.records = {}, {}, {}
        for path in (config.out / "predictions").glob("retrieval_*.json"):
            value = json.loads(path.read_text())
            self.rankings[value["experiment"]] = value["rankings"]
            self.scores[value["experiment"]] = value["scores"]
            self.records[value["experiment"]] = value["record"]
        self.selection = self._load("selection.json", {})
        self.status = self._load("status.json", {})
        self.start = time.perf_counter()
        cpu_name = platform.processor()
        if not cpu_name and Path('/proc/cpuinfo').exists():
            cpu_name = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                             if line.startswith('model name')), '')
        provenance = {
            "python": platform.python_version(), "platform": platform.platform(),
            "device": config.device, "cuda_available": torch.cuda.is_available(),
            "cpu": cpu_name, "threads": config.threads,
            "logical_cpus": os.cpu_count(), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if config.device == "cuda" else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if config.device == "cuda" else None,
            "training_precision": "mixed float16/float32" if config.device == "cuda" else "float32",
            "packages": {name: importlib.metadata.version(name) for name in
                         ["torch", "transformers", "sentence-transformers", "adapters", "numpy", "pandas",
                          "scikit-learn", "matplotlib", "rank-bm25", "nbformat", "nbclient"]},
            "upstream_evaluator_sha256": {p.name: sha256(p) for p in
                         Path("vendor/scifact/verisci/evaluate/lib").glob("*.py")},
            "model_revisions": {p.parent.parent.name: p.name for p in
                         Path(".cache/huggingface/hub").glob("models--*/snapshots/*")},
        }
        write_json(config.out / "provenance.json", provenance)

    def _load(self, filename, default=None):
        path = self.config.out / filename
        return json.loads(path.read_text()) if path.exists() else default

    def _done(self, stage, details=None):
        self.status[stage] = {"status": "completed", "details": details}
        write_json(self.config.out / "status.json", self.status)
        print("Completed", stage, flush=True)

    def _store_retrieval(self, name, results, record):
        self.rankings[name] = {split: value[0] for split, value in results.items()}
        self.scores[name] = {split: value[1] for split, value in results.items()}
        self.records[name] = {"experiment": name, "seed": self.config.seed, "mode": self.config.mode,
                              "device": self.config.device, "config": self.config.dictionary(),
                              "query_seconds": {split: value[2] for split, value in results.items()}, **record}
        write_json(self.config.out / "predictions" / f"retrieval_{name}.json",
                   {"experiment": name, "rankings": self.rankings[name], "scores": self.scores[name],
                    "record": self.records[name]})
        # Tuning uses this table only. Dev comparisons are generated after selection is frozen.
        return retrieval_metrics(self.dataset.validation, self.rankings[name]["validation"])

    def eda(self):
        tables = eda_tables(self.dataset)
        for name, table in zip(["dataset_summary", "document_lengths", "claim_lengths", "rationale_lengths"], tables):
            table.to_csv(self.config.out / "tables" / f"{name}.csv", index=False)
        self._done("data", {"corpus": len(self.dataset.documents), "train": len(self.dataset.train),
                            "validation": len(self.dataset.validation), "dev": len(self.dataset.dev)})
        return tables

    def bm25(self):
        for name, title in [("bm25_abstract", False), ("bm25_title_abstract", True)]:
            if name in self.rankings:
                continue
            start = time.perf_counter()
            model = BM25Retriever(self.dataset, title)
            indexing = time.perf_counter() - start
            results = {split: model.rankings(claims) for split, claims in self.splits.items()}
            self._store_retrieval(name, results, {"model": "rank-bm25/BM25Okapi", "title": title,
                                                 "k1": 1.5, "b": .75, "indexing_seconds": indexing})
        self._done("bm25")
        return self.validation_retrieval_table()

    def pretrained(self):
        if "dense_pretrained" not in self.rankings:
            start = time.perf_counter()
            model = load_dense(self.config)
            embeddings = corpus_embeddings(model, self.dataset, self.config, "pretrained")
            indexing = time.perf_counter() - start
            results = {split: dense_rankings(model, embeddings, self.dataset, claims, self.config)
                       for split, claims in self.splits.items()}
            self._store_retrieval("dense_pretrained", results,
                                 {"model": self.config.dense_model, "indexing_seconds": indexing,
                                  "maximum_tokens": self.config.dense_length, "normalization": "L2", "similarity": "cosine"})
            del model
            gc.collect()
        self._done("pretrained_dense")
        return self.validation_retrieval_table()

    def contrastive(self, negative_mode):
        name = f"dense_{negative_mode}"
        if name not in self.rankings:
            bm25 = BM25Retriever(self.dataset)
            model, history = train_retriever(self.dataset, bm25, self.config, negative_mode)
            start = time.perf_counter()
            embeddings = corpus_embeddings(model, self.dataset, self.config, negative_mode)
            indexing = time.perf_counter() - start
            results = {split: dense_rankings(model, embeddings, self.dataset, claims, self.config)
                       for split, claims in self.splits.items()}
            self._store_retrieval(name, results,
                                 {"model": str(self.config.out / "checkpoints" / name),
                                  "base_model": self.config.dense_model, "negative_mode": negative_mode,
                                  "objective": "masked multi-positive InfoNCE with explicit and in-batch negatives",
                                  "training_seconds": history[-1]["elapsed_seconds"], "epochs_run": len(history),
                                  "selected_epoch": max(history, key=lambda h: (h["validation"]["Recall@10"],
                                                                                 h["validation"]["MRR@10"]))["epoch"],
                                  "indexing_seconds": indexing})
            del model
            gc.collect()
        self._done(name)
        return self.validation_retrieval_table()

    def hybrid(self):
        tuning = []
        for dense in ["pretrained", "random", "hard"]:
            name = f"hybrid_{dense}"
            if name in self.rankings:
                continue
            trials = []
            for constant in [10, 60]:
                for weight in [.25, .5, .75]:
                    ranking, _ = rrf_rankings(self.rankings["bm25_title_abstract"]["validation"],
                                              self.rankings[f"dense_{dense}"]["validation"], constant, weight)
                    trial = {"dense": dense, "rrf_constant": constant, "rrf_weight": weight,
                             **retrieval_metrics(self.dataset.validation, ranking)}
                    trials.append(trial)
                    tuning.append(trial)
            best = max(trials, key=lambda r: (r["Recall@10"], r["MRR@10"]))
            results = {}
            for split in self.splits:
                start = time.perf_counter()
                ranking, scores = rrf_rankings(self.rankings["bm25_title_abstract"][split],
                                               self.rankings[f"dense_{dense}"][split],
                                               best["rrf_constant"], best["rrf_weight"])
                fusion_seconds = time.perf_counter() - start
                total_query_seconds = (self.records['bm25_title_abstract']['query_seconds'][split] +
                                       self.records[f'dense_{dense}']['query_seconds'][split] + fusion_seconds)
                results[split] = (ranking, scores, total_query_seconds)
            self._store_retrieval(name, results, {"model": "RRF", **best, "input_depth": 50,
                                "cost_scope": "Sum of measured lexical query, dense query and fusion times"})
        if tuning:
            write_json(self.config.out / "metrics/hybrid_tuning.json", tuning)
        # Choose the trained dense variant within hybrid by validation only.
        dense_mode = max(["random", "hard"], key=lambda name: (
            retrieval_metrics(self.dataset.validation, self.rankings[f"hybrid_{name}"]["validation"])["Recall@10"],
            retrieval_metrics(self.dataset.validation, self.rankings[f"hybrid_{name}"]["validation"])["MRR@10"]))
        chosen = self.records[f"hybrid_{dense_mode}"]
        paths = sorted((self.config.out / "cache").glob(f"corpus_{dense_mode}_*.npy"))
        self.selection.update({"dense_mode": dense_mode, "hybrid_name": f"hybrid_{dense_mode}",
                               "rrf_constant": chosen["rrf_constant"], "rrf_weight": chosen["rrf_weight"],
                               "dense_embeddings_path": str(paths[0])})
        write_json(self.config.out / "selection.json", self.selection)
        self._done("hybrid", self.selection)
        return self.validation_retrieval_table()

    def reranking(self):
        if "hybrid_reranked" not in self.rankings:
            base = self.selection["hybrid_name"]
            results = {split: rerank(claims, self.rankings[base][split], self.dataset, self.config)
                       for split, claims in self.splits.items()}
            for split in self.splits:
                for cid, ranking in results[split][0].items():
                    assert set(ranking) == set(self.rankings[base][split][cid][:self.config.candidate_count])
            self._store_retrieval("hybrid_reranked", results,
                                 {"model": self.config.reranker_model, "candidate_source": base,
                                  "candidates": self.config.candidate_count, "maximum_tokens": self.config.rerank_length,
                                  "cost_scope": "Cross-encoder scoring and sorting only; initial retrieval excluded"})
        self._done("reranking")
        return self.validation_retrieval_table()

    def supervised(self):
        if "supervised" in self.status:
            saved = self._load("metrics/stance_validation.json")
            if saved is None:
                classifier = PairClassifier(self.config, self.config.out / "checkpoints/stance")
                rows = stance_examples(self.dataset.validation, self.dataset.corpus)
                saved = classification_metrics([r["label"] for r in rows], classifier.probabilities(rows).argmax(1))
                write_json(self.config.out / "metrics/stance_validation.json", saved)
                del classifier
                gc.collect()
            return saved
        config, dataset = self.config, self.dataset
        training = downsample_evidence(evidence_examples(dataset.train, dataset.corpus), config.seed)
        validation = evidence_examples(dataset.validation, dataset.corpus)
        selector, selector_history = train_pair_classifier(training, validation, config, "selector")
        del selector
        gc.collect()
        train = stance_examples(dataset.train, dataset.corpus)
        val = stance_examples(dataset.validation, dataset.corpus)
        stance, stance_history = train_pair_classifier(train, val, config, "stance")
        probabilities = stance.probabilities(val)
        metrics = classification_metrics([r["label"] for r in val], probabilities.argmax(1))
        metrics.update({"setting": "Every complete gold rationale separately", "split": "internal_validation",
                        "checkpoint": str(config.out / "checkpoints/stance")})
        write_json(config.out / "metrics/stance_validation.json", metrics)
        baseline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True),
                                 LogisticRegression(C=1., max_iter=1000, solver="liblinear", class_weight="balanced", random_state=config.seed))
        baseline.fit([r["claim"] + " [SEP] " + r["text"] for r in train], [r["label"] for r in train])
        joblib.dump(baseline, config.out / "checkpoints/stance_tfidf.joblib")
        del stance
        gc.collect()
        self._done("supervised", {"selector_epochs_run": len(selector_history), "stance_epochs_run": len(stance_history)})
        return metrics

    def final_evaluation(self, force=False):
        if not force and "final_evaluation" in self.status:
            return self._load("metrics/end_to_end.json")
        config, dataset = self.config, self.dataset
        selector = PairClassifier(config, config.out / "checkpoints/selector")
        stance = PairClassifier(config, config.out / "checkpoints/stance")
        embedding_model = load_dense(config)
        sentence_maps, stance_maps = {}, {}
        for split, claims in self.splits.items():
            # Pool supports three controlled retrieval comparisons. Oracle docs are appended solely for diagnostics.
            pooled = {}
            for c in claims:
                cid = str(c["id"])
                pooled[cid] = list(dict.fromkeys(d for name in ["hybrid_reranked", self.selection["hybrid_name"],
                                                               "bm25_title_abstract"]
                                                for d in self.rankings[name][split][cid][:config.verification_documents]))
            score_config = replace(config, verification_documents=3 * config.verification_documents)
            sentence_maps[split] = sentence_score_maps(dataset, claims, pooled, selector, embedding_model, score_config, split)
            stance_maps[split] = stance_score_maps(dataset, claims, sentence_maps[split]["supervised"], stance, config, split)
        thresholds, evidence_results = {}, []
        for kind in ["similarity", "supervised"]:
            threshold, tuning = tune_sentence_threshold(dataset.validation,
                        self.rankings["hybrid_reranked"]["validation"], sentence_maps["validation"][kind], config, kind)
            thresholds[kind] = threshold
            write_json(config.out / "metrics" / f"{kind}_threshold_tuning.json", tuning)
            for oracle in [False, True]:
                selected = selected_map(dataset.dev, self.rankings["hybrid_reranked"]["dev"],
                                        sentence_maps["dev"][kind], threshold, config, oracle)
                forced = selected_map(dataset.dev, self.rankings["hybrid_reranked"]["dev"],
                                      sentence_maps["dev"][kind], threshold, config, oracle, True)
                record = {"model": kind, "setting": "gold_documents" if oracle else "retrieved_documents",
                          "threshold": threshold, "sentence_budget_per_document": config.sentence_budget,
                          **sentence_metrics(dataset.dev, selected),
                          "fixed_budget": sentence_metrics(dataset.dev, forced)}
                # Official rationale selection independent of stance via official selection column.
                pseudo = {str(c["id"]): {"evidence": {d: {"label": "SUPPORT", "sentences": ix}
                            for d, ix in selected[str(c["id"])].items() if ix}} for c in dataset.dev}
                from .metrics import official_metrics
                record["official_selection"] = official_metrics(dataset.dev, pseudo)["sentence_selection"]
                evidence_results.append(record)
                write_json(config.out / "predictions" / f"evidence_{kind}_{record['setting']}.json", selected)
        write_json(config.out / "metrics/evidence.json", evidence_results)
        best, tuning = tune_pipeline(dataset, dataset.validation, self.rankings["hybrid_reranked"]["validation"],
                    sentence_maps["validation"]["supervised"], stance_maps["validation"],
                    self.scores["hybrid_reranked"]["validation"], config, thresholds["supervised"])
        self.selection.update({"sentence_threshold": thresholds["supervised"], **best,
                               "tuning_split": "internal_validation_only"})
        write_json(config.out / "selection.json", self.selection)
        write_json(config.out / "metrics/pipeline_threshold_tuning.json", tuning)
        # All trainable checkpoints and all acceptance/fusion choices are now fixed.
        dev_stance = stance_examples(dataset.dev, dataset.corpus)
        probabilities = stance.probabilities(dev_stance)
        stance_metrics = classification_metrics([r["label"] for r in dev_stance], probabilities.argmax(1))
        stance_metrics.update({"setting": "Every complete gold rationale separately", "split": "official_dev",
                              "checkpoint": str(config.out / "checkpoints/stance")})
        write_json(config.out / "metrics/stance_gold.json", stance_metrics)
        write_jsonl(config.out / "predictions/stance_gold.jsonl", [{**r, "probabilities": p.tolist(),
                    "prediction": int(p.argmax())} for r, p in zip(dev_stance, probabilities)])
        baseline = joblib.load(config.out / "checkpoints/stance_tfidf.joblib")
        labels = baseline.predict([r["claim"] + " [SEP] " + r["text"] for r in dev_stance])
        write_json(config.out / "metrics/stance_tfidf.json",
                   classification_metrics([r["label"] for r in dev_stance], labels))
        result = {}
        settings = [("predicted_all", "hybrid_reranked", False, False),
                    ("gold_documents", "hybrid_reranked", True, False),
                    ("gold_documents_gold_evidence", "hybrid_reranked", True, True),
                    ("bm25_predicted", "bm25_title_abstract", False, False),
                    ("hybrid_without_reranking", self.selection["hybrid_name"], False, False)]
        gold_probabilities = gold_stance_map(dataset, dataset.dev, stance)
        all_predictions = {}
        for name, retrieval, oracle_docs, gold_evidence in settings:
            # Same cross-encoder document acceptance scores in the ordering-only hybrid ablation.
            scoring = self.scores[retrieval]["dev"]
            if name == "hybrid_without_reranking":
                scoring = {}
                for cid, ranks in self.rankings[retrieval]["dev"].items():
                    score_lookup = dict(zip(self.rankings["hybrid_reranked"]["dev"][cid],
                                            self.scores["hybrid_reranked"]["dev"][cid]))
                    scoring[cid] = [score_lookup.get(doc) for doc in ranks]
            predictions = compose_predictions(dataset, dataset.dev, self.rankings[retrieval]["dev"],
                        sentence_maps["dev"]["supervised"], stance_maps["dev"], config, thresholds["supervised"],
                        best["stance_threshold"], best["document_threshold"] if retrieval == "hybrid_reranked" or
                            name == "hybrid_without_reranking" else -float("inf"),
                        scoring, oracle_docs, gold_evidence, gold_probabilities)
            result[name] = pipeline_metrics(dataset.dev, predictions)
            result[name]["oracle_documents"] = oracle_docs
            result[name]["oracle_evidence"] = gold_evidence
            all_predictions[name] = predictions
            write_json(config.out / "predictions" / f"pipeline_{name}.json", predictions)
            write_jsonl(config.out / "predictions" / f"official_{name}.jsonl",
                        [{"id": p["id"], "evidence": p["evidence"]} for p in predictions.values()])
        # Report conditional document stance performance with predicted evidence and its coverage.
        labels, predictions = [], []
        retrieved_gold_docs = 0
        for c in dataset.dev:
            for doc in gold_docs(c):
                retrieved_gold_docs += 1
                p = all_predictions["gold_documents"][str(c["id"])]["evidence"].get(str(doc))
                if p:
                    labels.append(LABELS.index(c["evidence"][str(doc)][0]["label"]))
                    predictions.append(LABELS.index(p["label"]))
        conditional = classification_metrics(labels, predictions) if labels else {}
        conditional.update({"gold_document_coverage": len(labels) / retrieved_gold_docs,
                            "total_gold_documents": retrieved_gold_docs,
                            "setting": "Gold documents, predicted evidence; stance only on accepted documents"})
        write_json(config.out / "metrics/stance_predicted_evidence.json", conditional)
        write_json(config.out / "metrics/end_to_end.json", result)
        self._error_analysis(all_predictions["predicted_all"], sentence_maps['dev']['supervised'])
        del selector, stance, embedding_model
        gc.collect()
        self._done("final_evaluation", {"selection": self.selection})
        return result

    def specter2(self, retry=False):
        if "specter2" in self.rankings:
            return
        if self.status.get("specter2", {}).get("status") == "unavailable":
            if not retry:
                return
            write_json(self.config.out / 'metrics/specter2_retry_log.json', {
                'previous_failure': self.status['specter2'],
                'fix': 'Load both adapter modules before moving the complete model to CUDA; GPU forward probes passed.'})
        if not self.config.specter2:
            self.status["specter2"] = {"status": "not_requested", "reason": "Disabled by execution configuration"}
            write_json(self.config.out / "status.json", self.status)
            return
        try:
            results, indexing = specter_rankings(self.dataset, self.splits, self.config)
            self._store_retrieval("specter2", results, {"model": "allenai/specter2_base",
                            "document_adapter": "allenai/specter2", "query_adapter": "allenai/specter2_adhoc_query",
                            "maximum_tokens": 512, "pooling": "CLS", "normalization": "L2",
                            "indexing_seconds": indexing})
            self._done("specter2", {'adapter_device_fix': 'Both adapters loaded before complete-model device transfer'})
        except Exception as error:
            import traceback
            self.status["specter2"] = {"status": "unavailable", "error": repr(error), "traceback": traceback.format_exc(),
                                      "attempted_fix": "Pinned adapters 1.2.0 with transformers 4.51.3 and loaded adapters before CUDA transfer; see scripts/probe_models.py"}
            write_json(self.config.out / "status.json", self.status)
            print("SPECTER2 unavailable:", repr(error), flush=True)

    def validation_retrieval_table(self):
        return pd.DataFrame([{"experiment": name, **retrieval_metrics(self.dataset.validation, ranks["validation"])}
                             for name, ranks in self.rankings.items()]).set_index("experiment")

    def save_retrieval_tables(self):
        rows = []
        order = ['bm25_abstract', 'bm25_title_abstract', 'dense_pretrained', 'specter2',
                 'dense_random', 'dense_hard', 'hybrid_pretrained', 'hybrid_random',
                 'hybrid_hard', 'hybrid_reranked']
        for name in sorted(self.rankings, key=lambda n: order.index(n) if n in order else len(order)):
            ranks = self.rankings[name]
            for split, claims in self.splits.items():
                row = {"experiment": name, "split": split, "seed": self.config.seed,
                       **retrieval_metrics(claims, ranks[split]),
                       "query_seconds": self.records[name]["query_seconds"][split],
                       "cost_scope": self.records[name].get('cost_scope', 'Batched query inference and search; indexing/model load excluded'),
                       "milliseconds_per_query": self.records[name]["query_seconds"][split] * 1000 / len(claims)}
                rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(self.config.out / "metrics/retrieval.csv", index=False)
        write_json(self.config.out / "metrics/retrieval.json", rows)
        return table

    def refresh_fusion_costs(self):
        """Measure fusion on the stored exact inputs, retaining all selected rankings/configurations."""
        for name in ['hybrid_pretrained', 'hybrid_random', 'hybrid_hard']:
            if name not in self.records:
                continue
            record = self.records[name]
            dense = 'dense_' + name.removeprefix('hybrid_')
            for split in self.splits:
                start = time.perf_counter()
                rankings, _ = rrf_rankings(self.rankings['bm25_title_abstract'][split], self.rankings[dense][split],
                                           record['rrf_constant'], record['rrf_weight'])
                duration = time.perf_counter() - start
                assert rankings == self.rankings[name][split]
                record['query_seconds'][split] = (self.records['bm25_title_abstract']['query_seconds'][split] +
                                                  self.records[dense]['query_seconds'][split] + duration)
                record.setdefault('fusion_seconds', {})[split] = duration
            record['cost_scope'] = 'Measured lexical query + dense query + fusion; fusion timed on identical stored rankings'
            write_json(self.config.out / 'predictions' / f'retrieval_{name}.json',
                       {'experiment': name, 'rankings': self.rankings[name], 'scores': self.scores[name], 'record': record})

    def _error_analysis(self, predictions, sentence_scores=None):
        rows = []
        ranks = self.rankings["hybrid_reranked"]["dev"]
        for c in self.dataset.dev:
            cid, p = str(c["id"]), predictions[str(c["id"])]
            gold, pred = claim_label(c), p["verdict"]
            official_f1 = pipeline_metrics([c], {cid: p})["official"]["abstract_rationalized"]["f1"]
            if gold == pred and (not gold_docs(c) or official_f1 == 1.):
                continue
            retrieved = set(ranks[cid][:self.config.verification_documents])
            if not gold_docs(c):
                category = "Evidence predicted for an unannotated/NEI claim"
            elif not gold_docs(c) & retrieved:
                category = "Retrieval missed every evidence document at the verification budget"
            elif pred == "NOT_ENOUGH_INFO":
                category = "Incorrect abstention after retrieval"
            elif pred != gold:
                category = "Stance or document aggregation error"
            else:
                category = "Correct claim verdict but incomplete/wrong rationale or document"
            diagnostics = []
            for doc in sorted(gold_docs(c)):
                key = str(doc)
                predicted = p['evidence'].get(key)
                indices = predicted['sentences'] if predicted else []
                diagnostic = {'doc_id': doc,
                    'reranked_position': ranks[cid].index(doc) + 1 if doc in ranks[cid] else None,
                    'accepted_by_pipeline': predicted is not None,
                    'contains_complete_alternative': any(set(r['sentences']) <= set(indices)
                                                        for r in c['evidence'][key]),
                    'correct_stance_if_accepted': (predicted['label'] == c['evidence'][key][0]['label'])
                                                   if predicted else None}
                if sentence_scores is not None:
                    from .pipeline import select_sentences
                    values = sentence_scores[cid][key]
                    diagnostic['sentences_passing_selector'] = select_sentences(
                        values, self.selection['sentence_threshold'], self.config.sentence_budget)
                    diagnostic['gold_sentence_selector_scores'] = {str(i): values[i] for i in gold_sentences(c, doc)}
                diagnostics.append(diagnostic)
            rows.append({"claim_id": c["id"], "claim": c["claim"], "category": category,
                         "gold_verdict": gold, "predicted_verdict": pred,
                         "gold_doc_ids": sorted(gold_docs(c)), "retrieved_top5": ranks[cid][:self.config.verification_documents],
                         "predicted_documents": p["documents"],
                         "gold_document_diagnostics": diagnostics,
                         "gold_rationales": [{"doc_id": int(doc), "label": r["label"],
                              "sentence_indices": r["sentences"],
                              "text": " ".join(self.dataset.corpus[int(doc)]["abstract"][i] for i in r["sentences"])}
                              for doc, rs in c.get("evidence", {}).items() for r in rs],
                         "official_abstract_f1_for_claim": official_f1})
        write_json(self.config.out / "tables/error_analysis_all.json", rows)
        # Diversify algorithmic failure categories before adding additional cases.
        selected, categories = [], set()
        for row in rows:
            if row["category"] not in categories:
                selected.append(row)
                categories.add(row["category"])
        for row in rows:
            if len(selected) >= 5:
                break
            if row not in selected:
                selected.append(row)
        write_json(self.config.out / "tables/error_examples.json", selected[:5])

    def run(self):
        self.eda()
        self.bm25()
        self.pretrained()
        self.contrastive("random")
        self.contrastive("hard")
        self.hybrid()
        self.reranking()
        self.supervised()
        self.final_evaluation()
        # Independent supplementary model, scheduled last so it cannot prevent core completion.
        self.specter2()
        self.save_retrieval_tables()
        from .report import generate_report
        generate_report(self)
        self._done("study", {"execution_seconds_this_invocation": time.perf_counter() - self.start,
                             "smoke_only": self.config.mode == "smoke"})
        return self
