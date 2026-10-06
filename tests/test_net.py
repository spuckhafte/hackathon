"""Tests for net score, heap top-K, index elimination and the id mapping.

The DocNodeMap tests are deliberately heavy: doc_id and node_id are both dense
integer ranges from 0, so confusing them produces plausible wrong numbers rather
than an exception, and nothing downstream would notice.
"""
from __future__ import annotations

import numpy as np
import pytest

from nyaya import config
from nyaya.contracts import Authority
from nyaya.rank.net import (
    DocNodeMap,
    EliminationReport,
    eliminate_documents,
    eliminate_low_idf_terms,
    g_of_docs,
    heap_comparisons,
    measure_elimination,
    net_score,
    normalise_text_scores,
    rank,
    sort_comparisons,
    status_of_docs,
    top_k,
)


# ==========================================================================
# DocNodeMap — the dangerous join
# ==========================================================================
# 4 indexed documents sampled out of an 11-node universe.
_DOCS = [0, 1, 2, 3]
_NODES = [0, 4, 7, 10]


@pytest.fixture
def mapping():
    return DocNodeMap(_DOCS, _NODES, n_nodes=11)


def test_round_trip_both_ways(mapping):
    for doc, node in zip(_DOCS, _NODES):
        assert mapping.node_of(doc) == node
        assert mapping.doc_of(node) == doc


def test_a_universe_only_node_has_no_doc(mapping):
    """Nodes 1,2,3,5,6,8,9 are judgments whose text we never indexed."""
    for node in (1, 2, 3, 5, 6, 8, 9):
        assert mapping.doc_of(node) == -1


def test_missing_mapping_is_minus_one_never_zero(mapping):
    """0 would silently mean 'node 0', which is a real judgment."""
    assert mapping.doc_of(5) == -1
    assert mapping.node_of(99) == -1
    assert mapping.node_of(-1) == -1
    assert mapping.doc_of(-3) == -1


def test_the_two_id_spaces_are_different_sizes(mapping):
    assert mapping.n_docs == 4
    assert mapping.n_nodes == 11
    assert len(mapping) == 4


def test_vectorised_lookup_matches_scalar(mapping):
    docs = np.array([0, 1, 2, 3], dtype=np.int32)
    assert mapping.nodes_of(docs).tolist() == _NODES
    nodes = np.array([0, 4, 7, 10], dtype=np.int32)
    assert mapping.docs_of(nodes).tolist() == _DOCS


def test_vectorised_lookup_handles_out_of_range(mapping):
    assert mapping.nodes_of(np.array([0, 99, -1])).tolist() == [0, -1, -1]
    assert mapping.docs_of(np.array([5, 1000])).tolist() == [-1, -1]


def test_identity_mapping_is_not_assumed(mapping):
    """doc 1 maps to node 4, not node 1 - the whole point of the class."""
    assert mapping.node_of(1) == 4
    assert mapping.node_of(1) != 1


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        DocNodeMap([0, 0, 1], [0, 1, 2])
    with pytest.raises(ValueError):
        DocNodeMap([0, 1, 2], [3, 3, 4])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        DocNodeMap([0, 1], [0])


def test_empty_mapping():
    m = DocNodeMap([], [])
    assert len(m) == 0
    assert m.node_of(0) == -1


# ==========================================================================
# g(d) lookup through the mapping
# ==========================================================================
def _authority():
    return {
        0: Authority(doc_id=0, g=0.90, status="ok"),
        4: Authority(doc_id=4, g=0.50, status="overruled"),
        7: Authority(doc_id=7, g=0.10, status="distinguished"),
        # node 10 deliberately has NO record
    }


def test_g_is_read_through_the_node_mapping(mapping):
    g = g_of_docs(np.array([0, 1, 2]), _authority(), mapping)
    assert g.tolist() == [0.90, 0.50, 0.10]


def test_g_would_be_wrong_if_doc_ids_were_used_directly(mapping):
    """Guard against the off-by-identity bug: doc 1 must get node 4's g."""
    g = g_of_docs(np.array([1]), _authority(), mapping)
    assert g[0] == 0.50            # node 4's g
    assert g[0] != 0.90            # NOT node 1's / doc 1's


def test_a_node_without_an_authority_record_gets_the_default(mapping):
    g = g_of_docs(np.array([3]), _authority(), mapping)
    assert g.tolist() == [0.0]
    g2 = g_of_docs(np.array([3]), _authority(), mapping, default=0.25)
    assert g2.tolist() == [0.25]


def test_g_with_no_authority_store_is_all_zero(mapping):
    assert g_of_docs(np.array([0, 1]), None, mapping).tolist() == [0.0, 0.0]


def test_g_accepts_dict_records_from_a_json_round_trip(mapping):
    auth = {0: {"g": 0.4, "status": "ok"}}
    assert g_of_docs(np.array([0]), auth, mapping).tolist() == [0.4]


def test_status_lookup(mapping):
    assert status_of_docs(np.array([0, 1, 2, 3]), _authority(), mapping) == \
        ["ok", "overruled", "distinguished", "ok"]


# ==========================================================================
# Normalisation and net score
# ==========================================================================
def test_text_scores_are_scaled_by_the_candidate_maximum():
    out = normalise_text_scores(np.array([2.0, 4.0, 8.0]))
    assert out.tolist() == [0.25, 0.5, 1.0]


def test_normalisation_handles_all_zero_and_empty():
    assert normalise_text_scores(np.zeros(3)).tolist() == [0, 0, 0]
    assert normalise_text_scores(np.array([])).size == 0


def test_normalisation_makes_alpha_mean_the_same_thing_on_long_and_short_queries():
    """A raw tf-idf sum grows with the number of query terms; normalised, it
    does not, so one ALPHA is defensible across queries."""
    short = normalise_text_scores(np.array([1.0, 2.0]))
    long = normalise_text_scores(np.array([10.0, 20.0]))
    assert short.tolist() == long.tolist()


def test_net_score_is_the_configured_combination():
    text = np.array([1.0, 0.0])
    g = np.array([0.0, 1.0])
    out = net_score(text, g)
    assert out[0] == pytest.approx(config.ALPHA)
    assert out[1] == pytest.approx(1 - config.ALPHA)


def test_net_score_adds_the_proximity_term():
    text = np.zeros(2)
    g = np.zeros(2)
    out = net_score(text, g, prox=np.array([1.0, 0.0]))
    assert out[0] == pytest.approx(config.BETA)
    assert out[1] == 0.0


def test_net_score_rejects_a_misaligned_g_array():
    """A misaligned join here is exactly the silent bug this guards."""
    with pytest.raises(ValueError):
        net_score(np.zeros(3), np.zeros(2))


def test_net_score_rejects_a_misaligned_proximity_array():
    with pytest.raises(ValueError):
        net_score(np.zeros(3), np.zeros(3), prox=np.zeros(2))


def test_authority_breaks_a_text_tie():
    text = np.array([1.0, 1.0])
    g = np.array([0.1, 0.9])
    out = net_score(text, g)
    assert out[1] > out[0]


# ==========================================================================
# Top-K by heap
# ==========================================================================
def test_top_k_returns_the_k_best_in_descending_order():
    ids = [10, 11, 12, 13, 14]
    scores = [0.1, 0.9, 0.5, 0.7, 0.3]
    assert top_k(ids, scores, 3) == [(11, 0.9), (13, 0.7), (12, 0.5)]


def test_top_k_matches_a_full_sort():
    rng = np.random.default_rng(358)
    ids = np.arange(500)
    scores = rng.random(500)
    heaped = [d for d, _s in top_k(ids, scores, 10)]
    sorted_top = np.argsort(-scores, kind="stable")[:10].tolist()
    assert heaped == sorted_top


def test_top_k_when_k_exceeds_the_candidate_count():
    assert top_k([1, 2], [0.5, 0.9], 10) == [(2, 0.9), (1, 0.5)]


def test_top_k_with_no_candidates_or_zero_k():
    assert top_k([], [], 5) == []
    assert top_k([1, 2], [0.1, 0.2], 0) == []


def test_ties_break_on_the_lower_doc_id():
    """Deterministic output matters for the evaluation harness."""
    out = top_k([5, 2, 9], [0.5, 0.5, 0.5], 3)
    assert [d for d, _s in out] == [2, 5, 9]


def test_top_k_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        top_k([1, 2, 3], [0.1, 0.2], 2)


def test_heap_beats_a_sort_in_comparisons():
    n, k = 13942, 10
    assert heap_comparisons(n, k) < sort_comparisons(n)
    assert sort_comparisons(n) / heap_comparisons(n, k) > 3


def test_comparison_counters_degenerate_safely():
    assert heap_comparisons(0, 10) == 0
    assert sort_comparisons(1) == 0


# ==========================================================================
# Index elimination
# ==========================================================================
class StubIndex:
    """`court` is in every document (idf ~0), `privacy` is rare."""

    def __init__(self):
        self._p = {
            "court": list(range(100)),
            "the": list(range(100)),
            "privacy": [3, 7, 11],
            "detention": [3, 7, 40, 41],
            "article": [7, 11, 40, 50, 60],
            "safeguard": [7, 70],
        }
        self.doc_count = 100
        self.doc_norm = np.ones(100)
        self.indexed = np.arange(100)

    def df(self, term):
        return len(self._p.get(term, ()))

    def idf(self, term):
        import math
        d, n = self.df(term), self.doc_count
        return math.log(1.0 + (n - d + 0.5) / (d + 0.5))

    def postings(self, term):
        docs = self._p.get(term)
        if docs is None:
            return None
        return StubPostings(docs)


class StubPostings:
    def __init__(self, docs):
        self._d = np.asarray(sorted(docs), dtype=np.int32)
        self.df = int(self._d.size)

    @property
    def doc_ids(self):
        return self._d

    def tf_array(self):
        return np.full(self.df, 2, dtype=np.int32)

    def index_of(self, doc_id):
        j = int(np.searchsorted(self._d, doc_id))
        return j if j < self.df and int(self._d[j]) == doc_id else -1


class StubCtx:
    def __init__(self, index, authority=None, docnode=None):
        self.index = index
        self.authority = authority
        self.docnode = docnode
        self.params = None


def test_a_term_in_every_document_is_dropped():
    idx = StubIndex()
    kept, dropped = eliminate_low_idf_terms(["court", "privacy"], idx)
    assert kept == ["privacy"]
    assert [t for t, _v in dropped] == ["court"]
    assert dropped[0][1] < config.IDF_ELIMINATION_THRESHOLD


def test_elimination_never_empties_the_query():
    """All-boilerplate queries must still return something."""
    idx = StubIndex()
    kept, dropped = eliminate_low_idf_terms(["court", "the"], idx)
    assert len(kept) == 1
    assert len(dropped) == 1


def test_no_index_means_no_term_elimination():
    kept, dropped = eliminate_low_idf_terms(["a", "b"], None)
    assert kept == ["a", "b"] and dropped == []


def test_short_queries_are_not_document_eliminated():
    """With 2 terms, 'at least half' is 1, which eliminates nothing."""
    idx = StubIndex()
    cands = np.arange(100, dtype=np.int32)
    out, required = eliminate_documents(cands, ["privacy", "detention"], idx)
    assert required == 1
    assert out.size == 100


def test_long_queries_keep_only_documents_matching_half_the_terms():
    idx = StubIndex()
    cands = np.arange(100, dtype=np.int32)
    terms = ["privacy", "detention", "article", "safeguard"]
    out, required = eliminate_documents(cands, terms, idx)
    assert required == 2
    # doc 7 has all four; doc 3 has privacy+detention; doc 11 has privacy+article;
    # doc 40 has detention+article.  doc 41/50/60/70 have one each.
    assert out.tolist() == [3, 7, 11, 40]


def test_document_elimination_respects_the_candidate_set():
    idx = StubIndex()
    cands = np.array([7, 11], dtype=np.int32)
    terms = ["privacy", "detention", "article", "safeguard"]
    out, _required = eliminate_documents(cands, terms, idx)
    assert out.tolist() == [7, 11]


def test_document_elimination_on_an_empty_candidate_set():
    idx = StubIndex()
    out, required = eliminate_documents(np.empty(0, np.int32),
                                        ["a", "b", "c", "d"], idx)
    assert out.size == 0 and required == 1


def test_unknown_terms_do_not_crash_elimination():
    idx = StubIndex()
    cands = np.arange(100, dtype=np.int32)
    out, _r = eliminate_documents(cands, ["privacy", "zzz", "yyy", "xxx"], idx)
    assert out.size >= 0


# ==========================================================================
# rank() end to end
# ==========================================================================
def _ctx_with_authority():
    mapping = DocNodeMap(list(range(100)), list(range(0, 200, 2)), n_nodes=200)
    auth = {mapping.node_of(d): Authority(doc_id=mapping.node_of(d), g=0.5)
            for d in range(100)}
    # make doc 11 the most authoritative judgment in the corpus
    auth[mapping.node_of(11)] = Authority(doc_id=mapping.node_of(11), g=1.0)
    return StubCtx(StubIndex(), authority=auth, docnode=mapping), mapping


def test_rank_returns_doc_ids_and_scores():
    ctx, _m = _ctx_with_authority()
    cands = np.arange(100, dtype=np.int32)
    out, report = rank(cands, ["privacy"], ctx, k=3)
    assert len(out) == 3
    ids = [d for d, _s in out]
    assert set(ids) <= {3, 7, 11}
    assert report.candidates == 100
    assert all(0.0 <= s <= 1.5 for _d, s in out)


def test_rank_uses_authority_to_order_equally_relevant_documents():
    """Docs 3, 7 and 11 all contain 'privacy' once; doc 11 has g=1.0."""
    ctx, _m = _ctx_with_authority()
    out, _r = rank(np.arange(100, dtype=np.int32), ["privacy"], ctx, k=3)
    assert out[0][0] == 11


def test_rank_reports_elimination_counts():
    ctx, _m = _ctx_with_authority()
    cands = np.arange(100, dtype=np.int32)
    terms = ["privacy", "detention", "article", "safeguard"]
    _out, report = rank(cands, terms, ctx, k=5)
    assert report.elimination.scored_without_elimination == 100
    assert report.elimination.scored_with_elimination == 4
    assert report.elimination.documents_saved == 96
    assert report.elimination.fraction_saved == pytest.approx(0.96)


def test_rank_without_elimination_scores_everything():
    ctx, _m = _ctx_with_authority()
    cands = np.arange(100, dtype=np.int32)
    terms = ["privacy", "detention", "article", "safeguard"]
    _out, report = rank(cands, terms, ctx, k=5, eliminate=False)
    assert report.scored == 100


def test_rank_accepts_an_injected_scorer():
    """M3's BM25F plugs in here; nothing in net.py assumes the fallback."""
    ctx, _m = _ctx_with_authority()

    def always_doc_42(doc_ids, terms, c):
        out = np.zeros(doc_ids.size)
        out[doc_ids == 42] = 1.0
        return out

    out, report = rank(np.arange(100, dtype=np.int32), ["privacy"], ctx, k=1,
                       score_fn=always_doc_42)
    assert out[0][0] == 42
    assert report.scorer == "always_doc_42"


def test_rank_on_an_empty_candidate_set():
    ctx, _m = _ctx_with_authority()
    out, report = rank(np.empty(0, np.int32), ["privacy"], ctx, k=10)
    assert out == []
    assert report.scored == 0


def test_rank_passes_an_explicit_mapping_through():
    ctx, mapping = _ctx_with_authority()
    out, _r = rank(np.array([11], dtype=np.int32), ["privacy"], ctx, k=1,
                   mapping=mapping)
    assert out[0][0] == 11


def test_measure_elimination_reports_both_counts_and_the_accuracy_cost():
    ctx, _m = _ctx_with_authority()
    cands = np.arange(100, dtype=np.int32)
    terms = ["privacy", "detention", "article", "safeguard"]
    m = measure_elimination(cands, terms, ctx, k=4)
    assert m["scored_without_elimination"] == 100
    assert m["scored_with_elimination"] == 4
    assert m["documents_saved"] == 96
    # on this stub the surviving documents are exactly the good ones, so the
    # saving is free; the real corpus is reported honestly either way
    assert m["top_k_identical"] is True
    assert m["heap_comparisons"] < m["sort_comparisons"] or m["scored_with_elimination"] < 10


def test_elimination_report_json():
    r = EliminationReport(scored_without_elimination=100,
                          scored_with_elimination=25)
    blob = r.as_json()
    assert blob["documents_saved"] == 75
    assert blob["fraction_saved"] == pytest.approx(0.75)


def test_elimination_report_handles_zero():
    r = EliminationReport()
    assert r.fraction_saved == 0.0
