from types import SimpleNamespace
import numpy as np
from scifact_project.pipeline import select_sentences, compose_predictions
from scifact_project.retrieval import rrf_rankings


def test_rrf_uses_ranks_and_unions_candidates():
    ranking, scores = rrf_rankings({"1": [11, 12]}, {"1": [12, 13]}, constant=10, weight=.5)
    assert ranking["1"][0] == 12
    assert set(ranking["1"]) == {11, 12, 13}
    assert scores["1"][0] == .5 / 12 + .5 / 11


def test_indices_are_original_and_budget_is_respected():
    assert select_sentences([.3, .9, .8, .95, .1], .5, budget=2) == [3, 1]
    assert select_sentences([.1, .2], .8, budget=3) == []


def test_prediction_does_not_require_evidence_annotations():
    dataset = SimpleNamespace(corpus={11: {"title": "paper", "abstract": ["unrelated", "supports"]}})
    config = SimpleNamespace(verification_documents=5, sentence_budget=3, candidate_count=20)
    claims = [{"id": 1, "claim": "a new claim"}]
    ranks = {"1": [11]}
    selection = {"1": {"11": [.1, .9]}}
    stance = {"1": {"11": {"1": [.1, .9]}}}
    p = compose_predictions(dataset, claims, ranks, selection, stance, config, .5)
    assert p["1"]["verdict"] == "SUPPORT"
    assert p["1"]["evidence"]["11"]["sentences"] == [1]
    assert p["1"]["documents"][0]["evidence"][0]["index"] == 1
    p = compose_predictions(dataset, claims, ranks, selection, stance, config, .99)
    assert p["1"]["verdict"] == "NOT_ENOUGH_INFO"
    assert p["1"]["evidence"] == {}


def test_conflicting_documents_use_explicit_aggregation_policy():
    corpus = {11: {"title": "a", "abstract": ["support"]}, 12: {"title": "b", "abstract": ["contradict"]}}
    dataset = SimpleNamespace(corpus=corpus)
    config = SimpleNamespace(verification_documents=5, sentence_budget=3, candidate_count=20)
    p = compose_predictions(dataset, [{"id": 1, "claim": "x"}], {"1": [11, 12]},
                            {"1": {"11": [.8], "12": [.95]}},
                            {"1": {"11": {"1": [.1, .9]}, "12": {"1": [.8, .2]}}}, config, .5)
    assert p["1"]["verdict"] == "CONTRADICT"
    assert p["1"]["evidence"]["11"]["label"] == "SUPPORT"
    assert p["1"]["evidence"]["12"]["label"] == "CONTRADICT"


def test_gold_evidence_oracle_retains_stance_acceptance_policy():
    dataset = SimpleNamespace(corpus={11: {"title": "paper", "abstract": ["evidence"]}})
    config = SimpleNamespace(verification_documents=5, sentence_budget=3, candidate_count=20)
    claims = [{"id": 1, "claim": "x", "evidence": {"11": [{"label": "SUPPORT", "sentences": [0]}]}}]
    arguments = (dataset, claims, {"1": [11]}, {"1": {"11": [.95]}},
                 {"1": {"11": {"1": [.15, .85]}}}, config, .5)
    predicted = compose_predictions(*arguments, stance_threshold=.9, oracle_docs=True)
    oracle = compose_predictions(*arguments, stance_threshold=.9, oracle_docs=True, gold_evidence=True,
                                 gold_stance_probabilities={"1": {"11": [.15, .85]}})
    assert predicted["1"]["verdict"] == oracle["1"]["verdict"] == "NOT_ENOUGH_INFO"
