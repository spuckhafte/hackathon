"""Tests for PageRank and g(d) (S4 part 2).  Network-free, tiny hand graphs.

The point of most of these is dangling-node mass: our real graph is ~97% sinks,
so a PageRank that leaks mass would look plausible and rank garbage.
"""
from __future__ import annotations

import numpy as np
import pytest

from nyaya import config
from nyaya.authority.pagerank import (
    BENCH_PRIOR_FALLBACK,
    CitationGraph,
    bench_norm,
    build_authority,
    edge_weight,
    log_minmax,
    pagerank,
    pagerank_networkx,
    treatment_status,
)
from nyaya.contracts import CitationEdge


def _edge(src, dst, treatment=""):
    return CitationEdge(src=src, dst=dst, treatment=treatment)


def _graph(pairs, n_nodes, weighting="uniform", doc_to_node=None):
    doc_to_node = doc_to_node or {i: i for i in range(n_nodes)}
    edges = [_edge(s, d, t) if len(x) == 3 else _edge(s, d)
             for x in pairs for (s, d, t) in [(x + ("",))[:3]]]
    return CitationGraph.from_edges(edges, n_nodes, doc_to_node, weighting)


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------
def test_src_doc_ids_are_mapped_into_node_id_space():
    """src is a doc_id, dst is already a node_id; the graph must be one space."""
    g = CitationGraph.from_edges([_edge(0, 500)], 1000, {0: 900})
    assert list(g.weights) == [(900, 500)]


def test_unresolved_edges_are_not_in_the_graph():
    g = CitationGraph.from_edges([_edge(0, -1), _edge(0, 5)], 10, {0: 1})
    assert g.n_edges == 1


def test_self_citations_are_dropped():
    g = CitationGraph.from_edges([_edge(0, 7)], 10, {0: 7})
    assert g.n_edges == 0
    assert g.dropped_self == 1


def test_repeat_citations_of_the_same_precedent_fold_into_one_edge():
    """A precedent cited in ten paragraphs is one citing-cited relationship."""
    g = CitationGraph.from_edges([_edge(0, 5)] * 10, 10, {0: 1})
    assert g.n_edges == 1
    assert g.weights[(1, 5)] == 10.0
    assert g.parallel_collapsed == 9


def test_out_of_range_targets_are_dropped_not_crashed():
    g = CitationGraph.from_edges([_edge(0, 99)], 10, {0: 1})
    assert g.n_edges == 0 and g.dropped_oob == 1


def test_degrees():
    g = _graph([(0, 2), (1, 2), (2, 3)], 4)
    assert g.in_degree().tolist() == [0, 0, 2, 1]
    assert g.out_degree().tolist() == [1, 1, 1, 0]


# --------------------------------------------------------------------------
# Edge weighting — the S3 / S4 ablation switch
# --------------------------------------------------------------------------
def test_uniform_weighting_is_plain_pagerank():
    for t in ("", "followed", "overruled", "distinguished"):
        assert edge_weight(_edge(0, 1, t), "uniform") == 1.0


def test_treatment_weighting_reads_config():
    assert edge_weight(_edge(0, 1, "followed"), "treatment") == \
        config.TREATMENT_WEIGHTS["followed"]
    assert edge_weight(_edge(0, 1, "overruled"), "treatment") == 0.0
    assert edge_weight(_edge(0, 1, "distinguished"), "treatment") == \
        config.TREATMENT_WEIGHTS["distinguished"]


def test_unlabelled_edge_falls_back_to_mentioned():
    """Every edge is unlabelled until M1's wave-3 classifier runs."""
    assert edge_weight(_edge(0, 1, ""), "treatment") == \
        config.TREATMENT_WEIGHTS["mentioned"]


def test_unknown_weighting_is_rejected():
    with pytest.raises(ValueError):
        edge_weight(_edge(0, 1), "vibes")


def test_overruled_edges_transfer_no_authority():
    """A judgment that only ever got overruled must gain nothing from citations.

    This is also the zero-out-weight case: node 0's only edge has weight 0.0, so
    node 0 is a sink with edges.  Dividing by its out-weight would NaN the whole
    vector, which is exactly the bug this pins.
    """
    g = _graph([(0, 3, "overruled")], 4, weighting="treatment")
    pr, info = pagerank(g)
    assert not np.isnan(pr).any()
    assert pr.sum() == pytest.approx(1.0, abs=1e-12)
    assert info["dangling_nodes"] == 4               # node 0 sends nothing
    # node 3's score is pure teleport + redistributed dangling mass
    assert pr[3] == pytest.approx(pr[1], rel=1e-9)   # node 1 is cited by nobody


# --------------------------------------------------------------------------
# Power iteration: the invariants
# --------------------------------------------------------------------------
def test_pagerank_sums_to_one_on_a_graph_that_is_almost_all_sinks():
    """The real shape: 1 citing node, 999 sinks."""
    g = _graph([(0, 1), (0, 2)], 1000)
    pr, info = pagerank(g)
    assert pr.sum() == pytest.approx(1.0, abs=1e-12)
    # only node 0 has out-links; the 999 others are sinks, cited or not
    assert info["dangling_nodes"] == 999


def test_pagerank_on_a_graph_with_no_edges_is_uniform():
    g = CitationGraph(50)
    pr, info = pagerank(g)
    assert pr.sum() == pytest.approx(1.0)
    assert np.allclose(pr, 1 / 50)
    assert info["dangling_nodes"] == 50


def test_dangling_mass_is_redistributed_not_dropped():
    """Drop the sink mass and the vector decays; this test pins the fix."""
    g = _graph([(0, 1)], 3)                 # 1 and 2 are sinks
    pr, _ = pagerank(g)
    assert pr.sum() == pytest.approx(1.0, abs=1e-12)
    # the cited node must beat the isolated one
    assert pr[1] > pr[2]


def test_a_cited_node_outranks_an_uncited_one():
    g = _graph([(0, 3), (1, 3), (2, 3)], 5)
    pr, _ = pagerank(g)
    assert pr[3] == max(pr)
    assert pr[4] < pr[3]


def test_more_citations_means_more_pagerank():
    g = _graph([(0, 4), (1, 4), (2, 4), (3, 5)], 6)
    pr, _ = pagerank(g)
    assert pr[4] > pr[5]


def test_authority_flows_transitively():
    """0 -> 1 -> 2: node 2 inherits authority node 1 accumulated."""
    g = _graph([(0, 1), (3, 1), (4, 1), (1, 2)], 6)
    pr, _ = pagerank(g)
    assert pr[2] > pr[5]        # 5 is isolated


def test_weight_splits_proportionally():
    """One citing node, two targets, one cited twice as hard."""
    g = CitationGraph(4)
    g.weights[(0, 1)] = 2.0
    g.weights[(0, 2)] = 1.0
    pr, _ = pagerank(g)
    assert pr[1] > pr[2] > pr[3]


def test_iteration_converges_and_reports_it():
    g = _graph([(0, 1), (1, 2), (2, 0)], 3)
    pr, info = pagerank(g)
    assert info["converged"] is True
    assert info["iterations"] <= config.PAGERANK_ITERATIONS
    assert np.allclose(pr, 1 / 3, atol=1e-6)    # a symmetric cycle is uniform


def test_lambda_zero_is_the_uniform_distribution():
    g = _graph([(0, 1), (0, 2)], 4)
    pr, _ = pagerank(g, lam=0.0)
    assert np.allclose(pr, 1 / 4)


# --------------------------------------------------------------------------
# networkx cross-check (validation only — never ships a ranking)
# --------------------------------------------------------------------------
def test_matches_networkx_on_a_graph_with_sinks():
    g = _graph([(0, 2), (1, 2), (2, 3), (4, 2), (4, 3)], 8)
    ours, _ = pagerank(g, iterations=500, tol=1e-14)
    theirs = pagerank_networkx(g)
    assert np.abs(ours - theirs).max() < 1e-9


def test_matches_networkx_on_a_weighted_graph():
    g = CitationGraph(6)
    g.weights.update({(0, 1): 3.0, (0, 2): 1.0, (1, 2): 2.0, (3, 1): 5.0})
    ours, _ = pagerank(g, iterations=500, tol=1e-14)
    theirs = pagerank_networkx(g)
    assert np.abs(ours - theirs).max() < 1e-9


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def test_log_minmax_spreads_a_skewed_distribution():
    """Plain min-max would leave every small value indistinguishable from 0."""
    v = np.array([1e-6, 2e-6, 4e-6, 1e-3])
    plain = (v - v.min()) / (v.max() - v.min())
    logged = log_minmax(v)
    assert plain[1] < 0.01                 # crushed against zero
    assert logged[1] > 0.05                # usable
    assert logged.min() == 0.0 and logged.max() == pytest.approx(1.0)


def test_log_minmax_preserves_the_ordering():
    v = np.array([3e-5, 1e-6, 7e-4, 2e-5])
    assert np.argsort(log_minmax(v)).tolist() == np.argsort(v).tolist()


def test_log_minmax_handles_zeros_and_constants():
    assert log_minmax(np.zeros(4)).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert log_minmax(np.array([5e-5] * 3)).tolist() == [0.0, 0.0, 0.0]
    assert log_minmax(np.array([])).size == 0


def test_bench_norm_uses_the_largest_bench_ever():
    assert bench_norm(np.array([13.0]))[0] == pytest.approx(1.0)
    assert bench_norm(np.array([0.0]))[0] == 0.0
    assert bench_norm(np.array([2.0]))[0] == pytest.approx(2 / config.MAX_BENCH)
    assert bench_norm(np.array([20.0]))[0] == pytest.approx(1.0)   # clipped


# --------------------------------------------------------------------------
# g(d)
# --------------------------------------------------------------------------
def test_g_is_the_weighted_sum_from_config():
    g = _graph([(0, 1), (2, 1)], 4)
    pr, _ = pagerank(g)
    bench = np.array([2, 5, 2, 13], dtype=float)
    auth = build_authority(g, pr, bench)
    a = auth[1]
    expected = a.penalty * (config.G_PAGERANK_WEIGHT * a.pagerank_norm
                            + config.G_BENCH_WEIGHT * a.bench_norm)
    assert a.g == pytest.approx(expected)
    assert 0.0 <= a.g <= 1.0


def test_authority_records_degrees_and_defaults_to_ok():
    g = _graph([(0, 1), (2, 1), (1, 3)], 4)
    pr, _ = pagerank(g)
    auth = build_authority(g, pr, np.full(4, 2.0))
    assert auth[1].in_degree == 2
    assert auth[1].out_degree == 1
    assert auth[1].status == "ok"
    assert auth[1].penalty == config.TREATMENT_PENALTY["ok"]


def test_every_node_gets_an_authority_record():
    g = _graph([(0, 1)], 25)
    pr, _ = pagerank(g)
    auth = build_authority(g, pr, np.full(25, 2.0))
    assert len(auth) == 25
    assert sorted(auth) == list(range(25))


def test_a_bigger_bench_raises_g_all_else_equal():
    g = CitationGraph(3)
    pr, _ = pagerank(g)                       # no edges: pagerank is uniform
    auth = build_authority(g, pr, np.array([2.0, 7.0, 13.0]))
    assert auth[0].g < auth[1].g < auth[2].g


# --------------------------------------------------------------------------
# Treatment-derived status (the path that goes live in wave 3)
# --------------------------------------------------------------------------
def test_unlabelled_edges_leave_everything_ok():
    g = _graph([(0, 1), (2, 1), (3, 1)], 5)
    st = treatment_status(g)
    assert st[1][0] == "ok" and st[1][1] == 1.0


def test_one_overruling_edge_marks_a_judgment_overruled():
    g = _graph([(0, 1, "followed"), (2, 1, "overruled")], 4, weighting="treatment")
    st = treatment_status(g)
    assert st[1][0] == "overruled"
    assert st[1][1] == config.TREATMENT_PENALTY["overruled"]
    assert st[1][2] == [2]                    # overruled_by carries the citer


def test_frequently_distinguished_crosses_the_configured_share():
    pairs = [(0, 1, "distinguished"), (2, 1, "distinguished"), (3, 1, "followed")]
    st = treatment_status(_graph(pairs, 5, weighting="treatment"))
    assert 2 / 3 >= config.DISTINGUISHED_SHARE
    assert st[1][0] == "distinguished"


def test_one_distinguishing_edge_is_not_enough():
    """config.DISTINGUISHED_MIN_EDGES guards against a single data point."""
    st = treatment_status(_graph([(0, 1, "distinguished")], 4, weighting="treatment"))
    assert st[1][0] == "ok"


def test_overruled_status_drives_the_penalty_into_g():
    g = _graph([(0, 1, "overruled"), (2, 3, "followed")], 5, weighting="treatment")
    pr, _ = pagerank(g)
    auth = build_authority(g, pr, np.full(5, 5.0), use_treatment=True)
    assert auth[1].status == "overruled"
    assert auth[1].penalty < auth[3].penalty
    assert auth[1].g < auth[3].g


def test_s3_is_plain_pagerank_even_when_the_edges_carry_labels():
    """S3 must know nothing about treatment, or the ablation under-reports.

    The edges here are labelled, but `use_treatment=False` is the S3 build: every
    judgment stays `ok` with penalty 1.0.  If the penalty leaked into S3, the two
    stores would differ only in edge weights and the headline "overruled
    judgments lose authority" effect would already be in the baseline.
    """
    g = _graph([(0, 1, "overruled"), (2, 3, "followed")], 5, weighting="uniform")
    pr, _ = pagerank(g)
    auth = build_authority(g, pr, np.full(5, 5.0), use_treatment=False)
    assert auth[1].status == "ok"
    assert auth[1].penalty == 1.0


# --------------------------------------------------------------------------
# The documented choice for universe-only nodes
# --------------------------------------------------------------------------
def test_bench_prior_is_a_plausible_supreme_court_bench():
    """It must not be 1: the Court never sits alone, which is the whole reason
    bench_size_meta is unusable."""
    assert BENCH_PRIOR_FALLBACK >= 2


# --------------------------------------------------------------------------
# S3 and S4 are separate stores, and the g(d) delta is the novelty claim
# --------------------------------------------------------------------------
def test_the_two_weightings_write_to_different_files():
    """Building S4 must not destroy S3: the ablation is a flag, not a rebuild."""
    from nyaya.authority.pagerank import (
        AUTHORITY_JSON,
        AUTHORITY_TREATMENT_JSON,
        authority_path,
        stats_path,
    )

    assert authority_path("uniform") == AUTHORITY_JSON
    assert authority_path("treatment") == AUTHORITY_TREATMENT_JSON
    assert authority_path("uniform") != authority_path("treatment")
    assert stats_path("uniform") != stats_path("treatment")


def _write_store(tmp_path, name, graph, weighting, bench):
    from nyaya.authority.pagerank import build_authority, pagerank, write_authority

    pr, _info = pagerank(graph)
    auth = build_authority(graph, pr, bench,
                           use_treatment=(weighting == "treatment"))
    stats = {"weighting": weighting}
    return write_authority(auth, stats, out=tmp_path / name), auth


def test_an_overruled_judgment_loses_g_between_s3_and_s4(tmp_path, monkeypatch):
    """The claim: a much-cited judgment that has been overruled must drop.

    Node 1 is cited four times; one of those citations overrules it.  Under
    uniform weighting (S3) it keeps a high g(d).  Under treatment weighting (S4)
    the overruling edge carries weight 0.0 and the penalty multiplies g(d) down.
    """
    import pandas as pd

    import nyaya.authority.pagerank as P

    pairs = [(0, 1, "followed"), (2, 1, "followed"), (3, 1, "followed"),
             (4, 1, "overruled")]
    bench = np.full(6, 5.0)
    g_uniform = _graph(pairs, 6, weighting="uniform")
    g_treat = _graph(pairs, 6, weighting="treatment")

    p3, _a3 = _write_store(tmp_path, "s3.json", g_uniform, "uniform", bench)
    p4, _a4 = _write_store(tmp_path, "s4.json", g_treat, "treatment", bench)

    universe = pd.DataFrame({
        "node_id": list(range(6)),
        "title": [f"CASE {i}" for i in range(6)],
        "year": [1960 + i for i in range(6)],
    })
    cmp = P.compare_weightings(uniform=p3, treatment=p4, universe=universe)

    assert cmp["status_counts"].get("overruled") == 1
    assert cmp["changed"] >= 1
    worst = cmp["biggest_drops"][0]
    assert worst["node_id"] == 1
    assert worst["status"] == "overruled"
    assert worst["delta"] < 0                       # g(d) fell
    assert worst["g_s4"] < worst["g_s3"]
    assert worst["penalty"] == config.TREATMENT_PENALTY["overruled"]


def test_an_unlabelled_graph_shows_no_movement(tmp_path):
    """Honest default: with every edge unlabelled, S4 == S3 and compare says so."""
    import pandas as pd

    import nyaya.authority.pagerank as P

    pairs = [(0, 1), (2, 1)]
    bench = np.full(4, 3.0)
    p3, _ = _write_store(tmp_path, "s3.json", _graph(pairs, 4, "uniform"),
                         "uniform", bench)
    p4, _ = _write_store(tmp_path, "s4.json", _graph(pairs, 4, "treatment"),
                         "treatment", bench)
    universe = pd.DataFrame({"node_id": list(range(4)),
                             "title": ["a", "b", "c", "d"],
                             "year": [1950] * 4})
    cmp = P.compare_weightings(uniform=p3, treatment=p4, universe=universe)
    assert cmp["changed"] == 0
    assert cmp["biggest_drops"] == []


def test_compare_requires_the_treatment_store_to_exist(tmp_path):
    import pandas as pd

    import nyaya.authority.pagerank as P

    bench = np.full(3, 2.0)
    p3, _ = _write_store(tmp_path, "s3.json", _graph([(0, 1)], 3, "uniform"),
                         "uniform", bench)
    with pytest.raises(FileNotFoundError):
        P.compare_weightings(uniform=p3, treatment=tmp_path / "nope.json",
                             universe=pd.DataFrame({"node_id": [0, 1, 2],
                                                    "title": ["a", "b", "c"],
                                                    "year": [1950] * 3}))


def test_a_frequently_distinguished_judgment_also_drops(tmp_path):
    import pandas as pd

    import nyaya.authority.pagerank as P

    pairs = [(0, 1, "distinguished"), (2, 1, "distinguished"),
             (3, 1, "followed")]
    bench = np.full(5, 4.0)
    p3, _ = _write_store(tmp_path, "s3.json", _graph(pairs, 5, "uniform"),
                         "uniform", bench)
    p4, _ = _write_store(tmp_path, "s4.json", _graph(pairs, 5, "treatment"),
                         "treatment", bench)
    universe = pd.DataFrame({"node_id": list(range(5)),
                             "title": [f"c{i}" for i in range(5)],
                             "year": [1970] * 5})
    cmp = P.compare_weightings(uniform=p3, treatment=p4, universe=universe)
    worst = cmp["biggest_drops"][0]
    assert worst["status"] == "distinguished"
    assert worst["delta"] < 0
