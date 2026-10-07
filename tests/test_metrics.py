"""Tests for nyaya/eval/metrics.py  (Member 1, wave 4).

This module is the project's single implementation of the reported metrics, so
the tests pin the definitions against worked examples a reader can verify by
hand rather than against the implementation's own output.
"""
from __future__ import annotations

import math

import pytest

from nyaya.eval import metrics as M


# --------------------------------------------------------------------------
# Hand-worked examples
# --------------------------------------------------------------------------

def test_precision_at_k_denominator_is_k_not_the_list_length():
    """A system returning two documents when ten were asked for has not earned
    the precision of a two-document list."""
    assert M.precision_at_k([1, 2], {1: 1, 2: 1}, 10) == pytest.approx(0.2)
    assert M.precision_at_k([1, 2], {1: 1, 2: 1}, 2) == pytest.approx(1.0)


def test_precision_binarises_at_grade_one():
    rels = {1: 0, 2: 1, 3: 2}
    assert M.precision_at_k([1, 2, 3], rels, 3) == pytest.approx(2 / 3)


def test_recall_is_over_all_judged_relevant_documents():
    rels = {1: 1, 2: 1, 3: 2, 4: 0}
    assert M.recall_at_k([1, 2], rels, 20) == pytest.approx(2 / 3)
    assert M.recall_at_k([1, 2, 3], rels, 20) == pytest.approx(1.0)


def test_average_precision_divides_by_the_total_relevant_not_the_found():
    """The classic MAP bug: dividing by the number found gives 1.0 for a system
    that retrieved one of three relevant documents."""
    rels = {1: 1, 2: 1, 3: 1}
    assert M.average_precision([1], rels) == pytest.approx(1 / 3)
    assert M.average_precision([1, 2, 3], rels) == pytest.approx(1.0)


def test_average_precision_worked_example():
    # ranks 1 and 3 relevant, 2 relevant in total: (1/1 + 2/3) / 2
    rels = {10: 1, 30: 1}
    got = M.average_precision([10, 20, 30, 40], rels)
    assert got == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_dcg_matches_the_textbook_formula():
    assert M.dcg([3.0]) == pytest.approx(3.0)
    assert M.dcg([3.0, 1.0]) == pytest.approx(3.0 + 1.0 / math.log2(3))


def test_ndcg_uses_graded_gain_two_to_the_rel_minus_one():
    rels = {1: 2, 2: 1}
    assert M.ndcg_at_k([1, 2], rels, 10) == pytest.approx(1.0)
    swapped = M.ndcg_at_k([2, 1], rels, 10)
    ideal = 3.0 + 1.0 / math.log2(3)
    assert swapped == pytest.approx((1.0 + 3.0 / math.log2(3)) / ideal)
    assert swapped < 1.0


def test_ndcg_ideal_is_the_judged_grades_not_the_retrieved_ones():
    """A system that misses the only grade-2 authority must not score 1.0."""
    rels = {1: 2, 2: 1}
    assert M.ndcg_at_k([2], rels, 10) < 1.0


def test_ndcg_rewards_a_grade_two_three_times_a_grade_one():
    assert (2 ** 2 - 1) / (2 ** 1 - 1) == 3


def test_reciprocal_rank():
    assert M.reciprocal_rank([9, 8, 1], {1: 1}) == pytest.approx(1 / 3)
    assert M.reciprocal_rank([9, 8], {1: 1}) == 0.0


# --------------------------------------------------------------------------
# Duplicates: the Track A pooling bug
# --------------------------------------------------------------------------

def test_recall_can_never_exceed_one_even_with_duplicates():
    """Pooling the top 10 of six systems produces duplicates, and counting
    positions instead of documents made recall_at_k return 1.5."""
    assert M.recall_at_k([5, 5, 5], {5: 1, 7: 1}, 10) == pytest.approx(0.5)
    for k in (1, 2, 5, 20):
        assert M.recall_at_k([5, 5, 7, 7, 5], {5: 1, 7: 1}, k) <= 1.0


def test_every_metric_counts_documents_not_positions():
    rels = {5: 1, 7: 1}
    dup = [5, 5, 5, 7, 7]
    uniq = [5, 7]
    for k in (2, 5, 10):
        assert M.precision_at_k(dup, rels, k) == M.precision_at_k(uniq, rels, k)
        assert M.recall_at_k(dup, rels, k) == M.recall_at_k(uniq, rels, k)
        assert M.ndcg_at_k(dup, rels, k) == pytest.approx(
            M.ndcg_at_k(uniq, rels, k))
    assert M.average_precision(dup, rels) == pytest.approx(
        M.average_precision(uniq, rels))


def test_dedupe_preserves_order():
    assert M.dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert M.dedupe([]) == []


def test_score_query_reports_the_deduplicated_length():
    s = M.score_query("q1", "S4", [5, 5, 7], {5: 1, 7: 1})
    assert s.n_returned == 2


# --------------------------------------------------------------------------
# Strictness at the boundary
# --------------------------------------------------------------------------

def test_a_bare_set_of_relevant_ids_is_rejected_with_a_useful_message():
    """Member 2's gold is a binary set; it used to fail deep inside with
    AttributeError, which is a bad way to find out."""
    with pytest.raises(TypeError) as exc:
        M.recall_at_k([5], {5, 7}, 10)
    assert "graded mapping" in str(exc.value)


@pytest.mark.parametrize("bad", [[1, 2], (1, 2), "12", 7, None])
def test_a_non_mapping_qrels_is_rejected(bad):
    with pytest.raises(TypeError):
        M.precision_at_k([1], bad, 5)


# --------------------------------------------------------------------------
# Empty and degenerate inputs
# --------------------------------------------------------------------------

def test_no_judged_relevant_documents_gives_nan_not_zero():
    """A query with nothing relevant judged is undefined, not perfect and not a
    failure, and averaging must skip it rather than count it as zero."""
    for fn in (M.recall_at_k, M.ndcg_at_k):
        assert math.isnan(fn([1, 2], {1: 0, 2: 0}, 10))
    assert math.isnan(M.average_precision([1, 2], {1: 0}))


def test_an_empty_ranking_scores_zero_where_that_is_defined():
    rels = {1: 1}
    assert M.precision_at_k([], rels, 5) == 0.0
    assert M.recall_at_k([], rels, 5) == 0.0
    assert M.ndcg_at_k([], rels, 5) == 0.0
    assert M.average_precision([], rels) == 0.0


def test_precision_at_zero_is_zero_not_a_division_error():
    assert M.precision_at_k([1], {1: 1}, 0) == 0.0


# --------------------------------------------------------------------------
# Aggregation and significance
# --------------------------------------------------------------------------

def test_aggregate_is_a_macro_average_and_skips_nan():
    a = M.score_query("q1", "S4", [1], {1: 1})
    b = M.score_query("q2", "S4", [9], {1: 0})      # nothing relevant -> NaN
    agg = M.aggregate([a, b])
    assert agg["n_queries"] == 2
    assert agg["P@5"] == pytest.approx((0.2 + 0.0) / 2)
    assert agg[M.MAP_NAME] == pytest.approx(1.0)     # q2's NaN is skipped


def test_sign_test_on_a_clear_win():
    a = [0.9] * 9 + [0.1]
    b = [0.1] * 9 + [0.9]
    out = M.sign_test(a, b)
    assert out["wins"] == 9 and out["losses"] == 1
    assert out["p_value"] < 0.05 and out["significant_at_05"]


def test_sign_test_on_a_coin_flip_is_not_significant():
    out = M.sign_test([1, 0, 1, 0], [0, 1, 0, 1])
    assert out["wins"] == 2 and out["losses"] == 2
    assert out["p_value"] == pytest.approx(1.0)
    assert not out["significant_at_05"]


def test_sign_test_discards_ties():
    out = M.sign_test([1, 1, 1], [1, 1, 1])
    assert out["ties"] == 3 and out["n"] == 0
    assert math.isnan(out["p_value"])


def test_bootstrap_ci_brackets_the_mean_difference():
    out = M.bootstrap_ci([0.5] * 20, [0.3] * 20, iterations=500)
    assert out["mean_diff"] == pytest.approx(0.2)
    assert out["lo"] == pytest.approx(0.2) and out["hi"] == pytest.approx(0.2)
    assert out["excludes_zero"]


def test_bootstrap_ci_straddles_zero_when_there_is_no_effect():
    a = [0.5, 0.1, 0.5, 0.1, 0.5, 0.1]
    b = [0.1, 0.5, 0.1, 0.5, 0.1, 0.5]
    out = M.bootstrap_ci(a, b, iterations=2000)
    assert out["mean_diff"] == pytest.approx(0.0, abs=1e-9)
    assert not out["excludes_zero"]


def test_bootstrap_is_reproducible_from_its_seed():
    a, b = [0.4, 0.6, 0.5], [0.1, 0.2, 0.3]
    assert M.bootstrap_ci(a, b, iterations=500, seed=7) == \
        M.bootstrap_ci(a, b, iterations=500, seed=7)


def test_per_query_comparison_counts_wins_ties_and_losses():
    a = [M.score_query(f"q{i}", "S4", [1] if i < 2 else [9], {1: 1})
         for i in range(3)]
    b = [M.score_query(f"q{i}", "B0", [9], {1: 1}) for i in range(3)]
    out = M.per_query_comparison(a, b, metric="nDCG@10")
    assert out["counts"] == {"improved": 2, "tied": 1, "worsened": 0}
    assert out["a_system"] == "S4" and out["b_system"] == "B0"
    assert "sign_test" in out and "bootstrap" in out


def test_metric_names_are_the_ones_the_report_quotes():
    # MAP carries its cutoff: AP is averaged over the full run, whose depth is
    # eval/run.RETRIEVE_K = 20.  tune.py scores at k=10 and says MAP@10.
    assert M.METRIC_NAMES == ("P@5", "P@10", "R@20", "MAP@20", "nDCG@10")
    assert M.MAP_NAME == "MAP@20" and M.AP_CUTOFF == 20
    assert M.P_CUTOFFS == (5, 10)
    assert M.R_CUTOFF == 20 and M.NDCG_CUTOFF == 10


# --------------------------------------------------------------------------
# The judged query set on disk
# --------------------------------------------------------------------------

def test_the_real_qrels_load_and_look_like_the_agreed_set():
    from nyaya import config

    qpath = config.EVAL_DIR / "queries" / "qrels.jsonl"
    if not qpath.exists():
        pytest.skip("qrels not present")
    qrels = M.load_qrels(qpath)
    summary = M.qrels_summary(qrels)
    assert summary["queries"] >= 20
    assert summary["judgments"] >= 300
    assert set(summary["by_grade"]) <= {0, 1, 2}
    assert summary["min_relevant_per_query"] >= 1
