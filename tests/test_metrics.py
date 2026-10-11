import pytest
from scifact_project.metrics import retrieval_metrics, official_metrics, sentence_metrics
from scifact_project.data import gold_docs, related_groups, stance_examples


def claim(evidence, cid=1):
    return {"id": cid, "claim": "a claim", "evidence": evidence}


def test_recall_hit_mrr_and_nei_exclusion():
    cs = [claim({"11": [{"label": "SUPPORT", "sentences": [0]}],
                 "15": [{"label": "SUPPORT", "sentences": [1]}]}),
          claim({"31": [{"label": "CONTRADICT", "sentences": [2]}]}, 2), claim({}, 3)]
    result = retrieval_metrics(cs, {"1": [16, 11, 17], "2": [99], "3": [1]})
    assert result["evaluable_claims"] == 2
    assert result["Recall@3"] == pytest.approx(0.25)
    assert result["Hit@10"] == pytest.approx(0.5)
    assert result["MRR@10"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        retrieval_metrics(cs, {"1": [11, 11]})


def test_macro_recall_and_rank_eleven_is_a_miss_at_ten():
    cs = [claim({"11": [{"label": "SUPPORT", "sentences": [0]}]}),
          claim({str(d): [{"label": "SUPPORT", "sentences": [0]}] for d in [21, 22, 23]}, 2)]
    rankings = {"1": list(range(100, 110)) + [11], "2": [21]}
    result = retrieval_metrics(cs, rankings)
    assert result["Recall@1"] == pytest.approx(1 / 6)
    assert result["Recall@20"] == pytest.approx(2 / 3)
    assert result["MRR@10"] == pytest.approx(.5)


def test_official_documented_example_and_full_rationale_requirement():
    c = claim({"11": [{"label": "SUPPORT", "sentences": [0, 1]},
                      {"label": "SUPPORT", "sentences": [11]}],
               "15": [{"label": "SUPPORT", "sentences": [4]}]})
    p = {"1": {"evidence": {"11": {"label": "SUPPORT", "sentences": [1, 11, 13]},
                             "16": {"label": "CONTRADICT", "sentences": [18, 20]}}}}
    result = official_metrics([c], p)
    assert result["abstract_rationalized"]["f1"] == pytest.approx(0.5)
    assert result["sentence_label"]["precision"] == pytest.approx(1 / 5)
    assert result["sentence_label"]["recall"] == pytest.approx(1 / 4)
    p["1"]["evidence"] = {"11": {"label": "SUPPORT", "sentences": [1]}}
    assert official_metrics([c], p)["sentence_selection"]["recall"] == 0
    assert sentence_metrics([c], {"1": {"11": [1]}})["recall"] > 0


def test_three_sentence_cap_and_wrong_label():
    c = claim({"11": [{"label": "SUPPORT", "sentences": [3]}]})
    p = {"1": {"evidence": {"11": {"label": "SUPPORT", "sentences": [0, 1, 2, 3]}}}}
    with pytest.warns(UserWarning):
        result = official_metrics([c], p)
    assert result["abstract_rationalized"]["recall"] == 0
    assert result["sentence_label"]["recall"] == 1
    p["1"]["evidence"]["11"] = {"label": "CONTRADICT", "sentences": [3]}
    result = official_metrics([c], p)
    assert result["sentence_selection"]["recall"] == 1
    assert result["sentence_label"]["recall"] == 0


def test_empty_predictions_and_citations_not_gold():
    c = claim({})
    c["cited_doc_ids"] = [11]
    assert gold_docs(c) == set()
    assert official_metrics([c], {"1": {"evidence": {}}})["abstract_rationalized"]["f1"] == 0
    with pytest.raises(ValueError, match='unlabeled'):
        retrieval_metrics([{'id': 7, 'claim': 'unlabeled test'}], {'7': [11]})
    with pytest.raises(ValueError, match='unlabeled'):
        official_metrics([{'id': 7, 'claim': 'unlabeled test'}], {})


def test_related_claims_grouped_transitively():
    cs = [{"id": 1, "claim": "a", "cited_doc_ids": [11]},
          {"id": 2, "claim": "b", "cited_doc_ids": [11, 15]},
          {"id": 3, "claim": "c", "cited_doc_ids": [15]},
          {"id": 4, "claim": "d", "cited_doc_ids": [17]}]
    g = related_groups(cs)
    assert g[1] == g[2] == g[3]
    assert g[1] != g[4]


def test_stance_preserves_alternative_multisentence_rationales():
    cs = [claim({"11": [{"label": "CONTRADICT", "sentences": [0, 2]},
                         {"label": "CONTRADICT", "sentences": [1]}]})]
    rows = stance_examples(cs, {11: {"abstract": ["first", "second", "third"]}})
    assert len(rows) == 2
    assert rows[0]["text"] == "first third"
    assert rows[0]["sentence_indices"] == [0, 2]
    assert rows[0]["label"] == 0
