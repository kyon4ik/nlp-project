"""Shared metrics. All values are fractions (not percentages)."""
from collections import Counter
from types import SimpleNamespace
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .data import claim_label, gold_docs, gold_sentences, LABELS


def require_annotations(claims):
    if any('evidence' not in c for c in claims):
        raise ValueError('Cannot evaluate unlabeled claims: missing evidence annotations (e.g. official test).')


def retrieval_metrics(claims, rankings, ks=(1, 3, 5, 10, 20)):
    require_annotations(claims)
    per_claim = []
    for claim in claims:
        relevant = gold_docs(claim)
        if not relevant:
            continue
        ranked = [int(d) for d in rankings.get(str(claim["id"]), rankings.get(claim["id"], []))]
        if len(ranked) != len(set(ranked)):
            raise ValueError("Duplicate retrieval document IDs")
        row = {f"Recall@{k}": len(relevant & set(ranked[:k])) / len(relevant) for k in ks}
        row["Hit@10"] = float(bool(relevant & set(ranked[:10])))
        row["MRR@10"] = next((1.0 / rank for rank, d in enumerate(ranked[:10], 1) if d in relevant), 0.0)
        dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in relevant)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(10, len(relevant))))
        row["nDCG@10"] = dcg / idcg
        per_claim.append(row)
    return {"evaluable_claims": len(per_claim), **{key: float(np.mean([r[key] for r in per_claim]))
             if per_claim else 0.0 for key in [*(f"Recall@{k}" for k in ks), "Hit@10", "MRR@10", "nDCG@10"]}}


def prf(correct, predicted, gold):
    p = correct / predicted if predicted else 0.0
    r = correct / gold if gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def sentence_metrics(claims, selected):
    """Micro sentence overlap plus macro best-rationale recall; includes false positives on NEI."""
    require_annotations(claims)
    correct = predicted = gold = 0
    rationale_recalls = []
    for c in claims:
        prediction = selected.get(str(c["id"]), {})
        for doc, indices in prediction.items():
            predicted += len(set(indices))
            correct += len(set(indices) & gold_sentences(c, int(doc)))
        for doc, rs in c.get("evidence", {}).items():
            if not rs:
                continue
            gold += len(gold_sentences(c, int(doc)))
            indices = set(prediction.get(str(doc), []))
            rationale_recalls.append(max(len(indices & set(r["sentences"])) / len(r["sentences"]) for r in rs))
    return {**prf(correct, predicted, gold), "predicted_sentences": predicted, "gold_sentences": gold,
            "best_rationale_recall@budget": float(np.mean(rationale_recalls)) if rationale_recalls else 0.0}


def classification_metrics(labels, predictions, names=LABELS):
    order = list(range(len(names)))
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=order, average="macro", zero_division=0)),
            "per_class": classification_report(labels, predictions, labels=order, target_names=names,
                                                 output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=order).tolist(),
            "label_order": names, "examples": len(labels)}


def official_metrics(claims, predictions):
    """Execute unmodified upstream SciFact scoring code with in-memory schema adapters."""
    require_annotations(claims)
    vendor = Path(__file__).resolve().parents[1] / "vendor/scifact"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from verisci.evaluate.lib.data import EvidenceAbstract, ClaimPredictions, PredictedAbstract, make_label
    from verisci.evaluate.lib.metrics import compute_metrics

    class Gold:
        def __init__(self):
            self.claims = {c["id"]: SimpleNamespace(evidence={int(d): EvidenceAbstract(
                int(d), make_label(rs[0]["label"]), [r["sentences"] for r in rs])
                for d, rs in c.get("evidence", {}).items() if rs}) for c in claims}

        def get_claim(self, claim_id):
            return self.claims[claim_id]

    class Predictions(list):
        pass

    gold = Gold()
    rows = Predictions()
    rows.gold = gold
    for c in claims:
        evidence = predictions.get(str(c["id"]), {}).get("evidence", {})
        rows.append(ClaimPredictions(c["id"], {int(d): PredictedAbstract(int(d), make_label(p["label"]),
                                      p["sentences"]) for d, p in evidence.items()}))
    return compute_metrics(rows).to_dict()


def pipeline_metrics(claims, predictions):
    names = ["CONTRADICT", "SUPPORT", "NOT_ENOUGH_INFO"]
    labels = [names.index(claim_label(c)) for c in claims]
    outputs = [names.index(predictions[str(c["id"])]["verdict"]) for c in claims]
    result = classification_metrics(labels, outputs, names)
    result["official"] = official_metrics(claims, predictions)
    return result
