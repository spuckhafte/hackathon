"""Tests for Track B (citation recovery) and Track C (efficiency).

Network-free.  The metric and masking functions are pure, so they are tested
directly; the parts that need the real index are tested through small stubs.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from nyaya import config
from nyaya.eval import citation_recovery as CR
from nyaya.eval import efficiency as EF


# ==========================================================================
# Metrics
# ==========================================================================
def test_recall_at_k():
    gold = {5, 7, 9}
    ranked = [1, 5, 2, 7, 3, 4, 6, 8, 10, 11, 9]
    assert CR.recall_at_k(ranked, gold, 10) == pytest.approx(2 / 3)
    assert CR.recall_at_k(ranked, gold, 50) == 1.0
    assert CR.recall_at_k(ranked, gold, 2) == pytest.approx(1 / 3)


def test_recall_with_empty_gold_or_results():
    assert CR.recall_at_k([1, 2], set(), 10) == 0.0
    assert CR.recall_at_k([], {1}, 10) == 0.0


def test_recall_counts_each_gold_document_once():
    """Duplicate results must not inflate recall."""
    assert CR.recall_at_k([5, 5, 5], {5, 7}, 10) == pytest.approx(0.5)


def test_reciprocal_rank():
    assert CR.reciprocal_rank([9, 1, 2], {9}) == 1.0
    assert CR.reciprocal_rank([1, 9, 2], {9}) == 0.5
    assert CR.reciprocal_rank([1, 2, 3], {9}) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert CR.reciprocal_rank([1, 7, 9], {7, 9}) == 0.5


# ==========================================================================
# Bootstrap CI
# ==========================================================================
def test_bootstrap_ci_brackets_the_mean():
    vals = [0.0, 0.25, 0.5, 0.5, 0.75, 1.0] * 10
    lo, hi = CR.bootstrap_ci(vals)
    assert lo < float(np.mean(vals)) < hi


def test_bootstrap_ci_is_reproducible():
    vals = [0.0, 1.0, 0.5, 0.25] * 15
    assert CR.bootstrap_ci(vals, seed=1) == CR.bootstrap_ci(vals, seed=1)
    assert CR.bootstrap_ci(vals, seed=1) != CR.bootstrap_ci(vals, seed=2)


def test_bootstrap_ci_narrows_as_the_sample_grows():
    rng = np.random.default_rng(0)
    small = rng.random(20).tolist()
    big = rng.random(2000).tolist()
    lo_s, hi_s = CR.bootstrap_ci(small)
    lo_b, hi_b = CR.bootstrap_ci(big)
    assert (hi_s - lo_s) > (hi_b - lo_b)


def test_bootstrap_ci_degenerate_cases():
    assert CR.bootstrap_ci([]) == (0.0, 0.0)
    assert CR.bootstrap_ci([0.4]) == (0.4, 0.4)
    assert CR.bootstrap_ci([0.5] * 20) == (0.5, 0.5)


# ==========================================================================
# Masking — the part the whole track depends on
# ==========================================================================
def test_citations_are_masked():
    text = "The appellant relies on [1951] S.C.R. 51 for the proposition."
    out, counts = CR.mask_text(text)
    assert "S.C.R." not in out
    assert "1951" not in out
    assert counts["citations"] == 1


def test_case_names_are_masked():
    text = "It follows Maneka Gandhi v. Union of India in every respect."
    out, counts = CR.mask_text(text)
    assert "Maneka" not in out
    assert counts["names"] >= 1


def test_sentences_that_discuss_a_precedent_are_dropped_wholesale():
    """Masking the citation is not enough — the surrounding words leak too."""
    text = ("The land was acquired in 1994. The court relied upon the decision "
            "in Smith v. Jones for the measure of compensation. "
            "The award followed.")
    out, counts = CR.mask_text(text)
    assert "relied" not in out
    assert "Smith" not in out
    assert counts["cue_sentences"] >= 1
    assert "The land was acquired" in out       # the facts survive


@pytest.mark.parametrize("cue", [
    "relied upon", "reliance was placed", "cited with approval",
    "referred to the decision", "in the case of", "this Court held in",
    "laid down in", "reported in", "see also", "overruled", "distinguished",
])
def test_every_citation_cue_drops_its_sentence(cue):
    text = f"A fact sentence here. The bench {cue} something. Another fact."
    out, counts = CR.mask_text(text)
    assert counts["cue_sentences"] >= 1
    assert "A fact sentence here" in out


def test_masking_keeps_the_facts():
    text = ("The appellant was dismissed from service on 4 March 1994 after a "
            "departmental enquiry into alleged misappropriation of funds.")
    out, _counts = CR.mask_text(text)
    assert "dismissed from service" in out
    assert "misappropriation" in out


def test_masking_empty_text():
    out, counts = CR.mask_text("")
    assert out == ""
    assert counts["citations"] == 0


def test_build_query_uses_only_the_first_fraction_of_paragraphs():
    names = ["alpha", "bravo", "charlie", "delta", "echo",
             "foxtrot", "golf", "hotel", "india", "juliett"]
    paras = [{"id": i, "zone": "body", "text": f"the factory {n} was closed"}
             for i, n in enumerate(names)]
    q, _counts = CR.build_query(paras, fraction=0.2)
    assert "alpha" in q or "bravo" in q
    assert "juliett" not in q
    assert "hotel" not in q


def test_build_query_caps_the_term_count():
    paras = [{"id": 1, "zone": "body",
              "text": " ".join(f"distinctword{i}" for i in range(500))}]
    q, _counts = CR.build_query(paras, fraction=1.0, max_terms=20)
    assert len(q.split()) <= 20


def test_build_query_ignores_non_body_zones():
    paras = [{"id": 1, "zone": "bench", "text": "JUSTICE SOMEBODY"},
             {"id": 2, "zone": "body", "text": "the factory was closed"}]
    q, _counts = CR.build_query(paras, fraction=1.0)
    assert "somebodi" not in q and "justic" not in q


def test_build_query_on_a_judgment_with_no_body():
    q, _counts = CR.build_query([{"id": 1, "zone": "title", "text": "X v Y"}])
    assert q == ""


def test_the_mask_token_never_reaches_the_query():
    paras = [{"id": 1, "zone": "body",
              "text": "A fact. Reported in [1951] S.C.R. 51. Another fact."}]
    q, _counts = CR.build_query(paras, fraction=1.0)
    assert "masked" not in q


# ==========================================================================
# Eligibility filter: no answer from the future, and never itself
# ==========================================================================
def _query(doc_id=100, date="2010-05-05", gold=(1, 2, 3)):
    return CR.RecoveryQuery(doc_id=doc_id, node_id=doc_id * 2, year=2010,
                            date=date, title="T", query="q", gold=list(gold))


def test_the_judgment_itself_is_never_a_result():
    q = _query()
    keep = CR._eligible_filter({100: "2010-05-05", 7: "1990-01-01"}, q)
    assert keep(100) is False
    assert keep(7) is True


def test_judgments_decided_later_are_excluded():
    q = _query(date="2010-05-05")
    dates = {5: "2010-05-04", 6: "2010-05-05", 7: "2010-05-06"}
    keep = CR._eligible_filter(dates, q)
    assert keep(5) is True          # strictly earlier
    assert keep(6) is False         # same day: cannot have been cited
    assert keep(7) is False         # later


def test_a_document_with_no_date_is_kept_rather_than_silently_dropped():
    q = _query(date="2010-05-05")
    keep = CR._eligible_filter({9: ""}, q)
    assert keep(9) is True


# ==========================================================================
# Aggregation
# ==========================================================================
def _rows():
    return [
        {"doc_id": 1, "mode": "B0", "recall@10": 0.0, "recall@50": 0.5,
         "mrr": 0.0, "gold": 2, "retrieved": 10, "ms": 5.0},
        {"doc_id": 2, "mode": "B0", "recall@10": 1.0, "recall@50": 1.0,
         "mrr": 1.0, "gold": 2, "retrieved": 10, "ms": 7.0},
        {"doc_id": 1, "mode": "S4", "recall@10": 0.5, "recall@50": 1.0,
         "mrr": 0.5, "gold": 2, "retrieved": 10, "ms": 9.0},
        {"doc_id": 2, "mode": "S4", "recall@10": 1.0, "recall@50": 1.0,
         "mrr": 1.0, "gold": 2, "retrieved": 10, "ms": 11.0},
    ]


def test_aggregate_means_and_cis():
    agg = CR.aggregate(_rows())
    assert agg["B0"]["recall@10"] == pytest.approx(0.5)
    assert agg["S4"]["recall@10"] == pytest.approx(0.75)
    assert agg["B0"]["queries"] == 2
    assert agg["B0"]["recall@10_ci_lo"] <= agg["B0"]["recall@10"] <= \
        agg["B0"]["recall@10_ci_hi"]


def test_aggregate_counts_errors_without_crashing():
    rows = _rows() + [{"doc_id": 3, "mode": "S4", "error": "boom",
                       "recall@10": 0.0, "recall@50": 0.0, "mrr": 0.0,
                       "gold": 3, "retrieved": 0, "ms": 0.0}]
    agg = CR.aggregate(rows)
    assert agg["S4"]["errors"] == 1
    assert agg["S4"]["queries"] == 3


def test_every_metric_gets_a_confidence_interval():
    agg = CR.aggregate(_rows())
    for metric in CR.METRICS:
        assert f"{metric}_ci_lo" in agg["B0"]
        assert f"{metric}_ci_hi" in agg["B0"]


# ==========================================================================
# Track C percentiles
# ==========================================================================
def test_percentile_and_summary():
    vals = list(range(1, 101))
    assert EF.percentile(vals, 50) == pytest.approx(50.5)
    s = EF.summarise(vals)
    assert s["n"] == 100
    assert s["min"] == 1 and s["max"] == 100
    assert s["p95"] > s["p50"]


def test_summary_of_empty_and_single():
    assert EF.summarise([])["n"] == 0
    s = EF.summarise([4.0])
    assert s["p50"] == s["p95"] == 4.0


def test_p95_is_not_the_mean_on_a_long_tail():
    """The reason Track C quotes percentiles: one slow query moves the mean."""
    vals = [10.0] * 19 + [1000.0]
    s = EF.summarise(vals)
    assert s["p50"] == 10.0
    assert s["mean"] > 50
    assert s["p95"] > s["p50"]


def test_store_sizes_reports_every_known_store():
    s = EF.store_sizes()
    assert set(s["stores"]) == set(EF.STORE_FILES)
    assert s["total_bytes"] >= 0
    assert isinstance(s["total_human"], str)


def test_tier_fidelity_degrades_when_the_eval_has_not_been_run(monkeypatch,
                                                              tmp_path):
    monkeypatch.setattr(EF.config, "INDEX", tmp_path)
    out = EF.tier_fidelity()
    assert out["available"] is False
    assert "tiers" in out["note"]


def test_tier_fidelity_reads_the_s5_evaluation(monkeypatch, tmp_path):
    """Folded in, never recomputed - a second implementation would disagree."""
    (tmp_path / "tiers_eval.json").write_text(json.dumps({
        "k": 10, "r": 400, "queries": 2,
        "mean_scored_exhaustive": 1000.0, "mean_scored_tier1": 100.0,
        "mean_reduction": 0.9, "mean_speedup": 3.5,
        "identical_topk": 1, "identical_fraction": 0.5,
        "mean_recall_at_k": 0.9, "top1_same": 2, "fell_through": 0,
        "rows": [{"query": "a", "overlap": 10, "reduction": 0.9},
                 {"query": "b", "overlap": 3, "reduction": 0.95}],
    }), encoding="utf-8")
    monkeypatch.setattr(EF.config, "INDEX", tmp_path)
    out = EF.tier_fidelity()
    assert out["available"] is True
    assert out["mean_speedup"] == 3.5
    assert out["worst_query"]["query"] == "b"      # the honest failure mode
    assert out["worst_query"]["overlap"] == 3


# ==========================================================================
# Both tracks must go through search(), not a private pipeline
# ==========================================================================
def test_track_b_calls_the_shared_search_entry_point():
    import inspect

    src = inspect.getsource(CR.run_system)
    assert "from nyaya.search import search" in src
    assert "search(" in src


def test_track_c_calls_the_shared_search_entry_point():
    import inspect

    src = inspect.getsource(EF.measure_mode)
    assert "from nyaya.search import" in src
    assert "search(" in src


def test_the_popularity_baseline_never_reads_the_query():
    """G0 exists to quantify the authority tautology, so it must be query-blind."""
    import inspect

    src = inspect.getsource(CR.run_popularity_baseline)
    assert "q.query" not in src
    assert "search(" not in src


def test_systems_list_matches_the_contract_modes():
    from nyaya.contracts import MODES as CONTRACT_MODES

    assert tuple(CR.SYSTEMS) == tuple(CONTRACT_MODES)
    assert tuple(EF.MODES) == tuple(CONTRACT_MODES)


# ==========================================================================
# One implementation per metric: delegate to M1's eval/metrics.py
# ==========================================================================
def test_gold_set_converts_to_m1s_graded_qrels_shape():
    """His functions take Mapping[doc_id, grade]; passing a raw set raises."""
    assert CR._as_qrels({5, 7}) == {5: 1, 7: 1}
    assert CR._as_qrels(set()) == {}


def test_passing_a_set_straight_to_m1s_metrics_would_break():
    """Pins why the adapter exists: this is the cross-module breakage it avoids."""
    m = CR._import_metrics()
    if m is None:
        pytest.skip("eval/metrics.py not present")
    with pytest.raises(AttributeError):
        m.recall_at_k([1, 2, 3], {2, 5}, 2)
    with pytest.raises(AttributeError):
        m.reciprocal_rank([1, 2, 3], {2, 5})


def test_my_metrics_agree_with_m1s():
    """Two implementations that disagree is the trap; this is the check."""
    m = CR._import_metrics()
    if m is None:
        pytest.skip("eval/metrics.py not present")
    rng = np.random.default_rng(358)
    for _ in range(200):
        pool = list(range(30))
        ranked = rng.permutation(pool).tolist()
        gold = set(rng.choice(pool, size=int(rng.integers(1, 6)),
                              replace=False).tolist())
        qrels = CR._as_qrels(gold)
        for k in (1, 5, 10, 50):
            mine = len(set(ranked[:k]) & gold) / len(gold)
            theirs = m.recall_at_k(ranked, qrels, k)
            assert mine == pytest.approx(theirs)
            assert CR.recall_at_k(ranked, gold, k) == pytest.approx(mine)
        mine_rr = next((1.0 / i for i, d in enumerate(ranked, 1) if d in gold), 0.0)
        assert CR.reciprocal_rank(ranked, gold) == pytest.approx(mine_rr)
        assert m.reciprocal_rank(ranked, qrels) == pytest.approx(mine_rr)


def test_delegation_handles_his_nan_for_empty_relevance():
    """His recall_at_k returns nan when nothing is judged relevant; ours gives 0."""
    assert CR.recall_at_k([1, 2], set(), 10) == 0.0


def test_duplicate_results_cannot_push_recall_above_one():
    """M1's recall_at_k counts positions, not distinct docs: [5,5,5] against
    gold {5,7} gives 1.5.  We de-duplicate before delegating."""
    assert CR.recall_at_k([5, 5, 5], {5, 7}, 10) == pytest.approx(0.5)
    assert CR.recall_at_k([5, 5, 7, 7], {5, 7}, 10) == pytest.approx(1.0)


def test_dedup_preserves_order():
    assert CR._dedup_ranked([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert CR._dedup_ranked([]) == []
