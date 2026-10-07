"""Tests for S5 — champion lists and the two-tier index.  Network-free."""
from __future__ import annotations

import numpy as np
import pytest

from nyaya import config
from nyaya.contracts import Authority
from nyaya.index.tiers import (
    MIN_DF_FOR_CHAMPIONS,
    TieredIndex,
    TierStats,
    build,
    champion_order,
    tiered_candidates,
)
from nyaya.rank.net import DocNodeMap


# --------------------------------------------------------------------------
# A stub S3 with controllable tf, plus an S4 with controllable g(d)
# --------------------------------------------------------------------------
class StubPostings:
    def __init__(self, docs, tfs=None):
        self._d = np.asarray(docs, dtype=np.int32)
        self._tf = np.asarray(tfs if tfs is not None else [1] * len(docs),
                              dtype=np.int32)

    @property
    def doc_ids(self):
        return self._d

    @property
    def df(self):
        return int(self._d.size)

    def tf_array(self):
        return self._tf

    def index_of(self, doc_id):
        j = int(np.searchsorted(self._d, doc_id))
        return j if j < self.df and int(self._d[j]) == doc_id else -1


class StubIndex:
    def __init__(self, postings):
        self._p = postings
        self._terms = sorted(postings)
        self.doc_count = 50
        self.n_doc_slots = 50
        self.indexed = np.arange(50)

    def df(self, term):
        p = self._p.get(term)
        return p.df if p else 0

    def idf(self, term):
        import math
        d, n = self.df(term), self.doc_count
        return math.log(1.0 + (n - d + 0.5) / (d + 0.5))

    def postings(self, term):
        return self._p.get(term)


def _index():
    return StubIndex({
        # 20 documents, tf descending with doc id so the text order is known
        "common": StubPostings(list(range(20)), [20 - i for i in range(20)]),
        "mid": StubPostings([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 9]),
        "rare": StubPostings([30, 31, 32]),              # df 3 < MIN_DF
        "eight": StubPostings(list(range(40, 48))),      # df 8 == MIN_DF
    })


def _mapping():
    # doc d -> node 2d, so the two id spaces are never confusable
    return DocNodeMap(list(range(50)), list(range(0, 100, 2)), n_nodes=100)


def _authority(mapping, high_g_docs=()):
    auth = {}
    for d in range(50):
        node = mapping.node_of(d)
        auth[node] = Authority(doc_id=node,
                               g=1.0 if d in high_g_docs else 0.0)
    return auth


# --------------------------------------------------------------------------
# champion_order: the ordering criterion is the point
# --------------------------------------------------------------------------
def test_champions_are_ordered_by_text_weight_when_g_is_flat():
    idx = _index()
    g = np.zeros(50)
    order = champion_order(idx.postings("common"), g, idx, "common")
    assert order[0] == 0          # highest tf
    assert order[-1] == 19        # lowest tf


def test_authority_promotes_a_document_into_the_champion_list():
    """The headline design decision: w(t,d) + g(d), not w(t,d) alone.

    Doc 19 has the *lowest* tf for `common`, so a tf-only champion list would
    rank it last.  Give it g=1.0 and it must move up — those are exactly the
    documents the final net score promotes, so excluding them would optimise the
    champion list for a ranking we do not ship.
    """
    idx = _index()
    g_flat = np.zeros(50)
    g_auth = np.zeros(50)
    g_auth[19] = 1.0

    flat_order = champion_order(idx.postings("common"), g_flat, idx, "common").tolist()
    auth_order = champion_order(idx.postings("common"), g_auth, idx, "common").tolist()
    assert flat_order.index(19) > auth_order.index(19)


def test_champion_order_is_a_permutation_of_the_postings():
    idx = _index()
    order = champion_order(idx.postings("mid"), np.zeros(50), idx, "mid")
    assert sorted(order.tolist()) == sorted(idx.postings("mid").doc_ids.tolist())


def test_champion_order_on_empty_postings():
    assert champion_order(StubPostings([]), np.zeros(5)).size == 0


def test_champion_order_without_an_authority_vector():
    idx = _index()
    out = champion_order(idx.postings("mid"), np.array([]), idx, "mid")
    assert out.size == 10


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
@pytest.fixture
def tiers():
    mapping = _mapping()
    idx = _index()
    return build(index=idx, authority=_authority(mapping), mapping=mapping,
                 r=5, verbose=False)


def test_build_truncates_each_list_to_r(tiers):
    assert tiers.champions("common").size == 5
    assert tiers.champions("mid").size == 5


def test_terms_below_the_df_floor_get_no_champion_list(tiers):
    """A df-3 term's postings are already shorter than r; a champion list would
    be a verbatim copy."""
    assert tiers.champions("rare") is None
    assert not tiers.has("rare")


def test_a_term_exactly_at_the_df_floor_is_included(tiers):
    assert MIN_DF_FOR_CHAMPIONS == 8
    assert tiers.champions("eight") is not None


def test_build_records_its_own_compression(tiers):
    s = tiers.stats
    assert s.terms_with_champions == 3           # common, mid, eight
    assert s.terms_too_rare == 1                 # rare
    assert s.postings_entries == 20 + 10 + 3 + 8
    assert 0 < s.compression < 1


def test_build_without_authority_still_works():
    idx = _index()
    t = build(index=idx, authority={}, mapping=_mapping(), r=4, verbose=False)
    assert t.champions("common").size == 4


def test_tier_stats_compression_handles_zero():
    assert TierStats().compression == 0.0


# --------------------------------------------------------------------------
# tier1 union
# --------------------------------------------------------------------------
def test_tier1_is_the_union_of_the_champion_lists(tiers):
    ids, missing = tiers.tier1(["common", "mid"])
    assert missing == []
    assert ids.tolist() == sorted(set(ids.tolist()))
    assert ids.size <= 10


def test_tier1_reports_terms_with_no_champion_list(tiers):
    ids, missing = tiers.tier1(["common", "rare"])
    assert missing == ["rare"]


def test_tier1_of_unknown_terms_is_empty(tiers):
    ids, missing = tiers.tier1(["zzz"])
    assert ids.size == 0
    assert missing == ["zzz"]


# --------------------------------------------------------------------------
# tiered_candidates: the fall-through rule
# --------------------------------------------------------------------------
def test_tier1_answers_when_it_holds_at_least_k(tiers):
    idx = _index()
    cands, trace = tiered_candidates(["common", "mid"], tiers, idx, k=3)
    assert trace.tier == 1
    assert trace.fell_through is False
    assert cands.size >= 3
    assert "tier 1 held" in trace.reason


def test_fall_through_to_tier_2_when_tier_1_is_too_small(tiers):
    """r=5, so `common` alone offers 5 documents; k=20 forces tier 2."""
    idx = _index()
    cands, trace = tiered_candidates(["common"], tiers, idx, k=20)
    assert trace.fell_through is True
    assert trace.tier == 2
    assert cands.size == 20                      # the full postings
    assert "fell through" in trace.reason


def test_a_term_without_champions_contributes_its_full_postings(tiers):
    """Otherwise tier 1 would silently ignore the most discriminative word."""
    idx = _index()
    cands, trace = tiered_candidates(["common", "rare"], tiers, idx, k=3)
    assert trace.terms_without_champions == ["rare"]
    assert set([30, 31, 32]) <= set(cands.tolist())


def test_a_restriction_is_applied_before_the_fall_through_test(tiers):
    """`privacy bench:>=5` can have a healthy tier 1 that the filter collapses;
    the fall-through must notice."""
    idx = _index()
    restrict = np.array([0, 1], dtype=np.int32)
    cands, trace = tiered_candidates(["common", "mid"], tiers, idx, k=10,
                                     restrict=restrict)
    assert trace.fell_through is True
    assert set(cands.tolist()) <= {0, 1}


def test_restriction_is_honoured_in_tier_1(tiers):
    idx = _index()
    restrict = np.arange(0, 6, dtype=np.int32)
    cands, trace = tiered_candidates(["common", "mid"], tiers, idx, k=2,
                                     restrict=restrict)
    assert trace.tier == 1
    assert set(cands.tolist()) <= set(restrict.tolist())


def test_tier2_is_a_superset_of_tier1(tiers):
    idx = _index()
    t1, _ = tiered_candidates(["common"], tiers, idx, k=1)
    t2, trace = tiered_candidates(["common"], tiers, idx, k=99)
    assert trace.fell_through is True
    assert set(t1.tolist()) <= set(t2.tolist())


def test_trace_is_json_serialisable(tiers):
    import json

    idx = _index()
    _c, trace = tiered_candidates(["common"], tiers, idx, k=3)
    assert "tier" in json.dumps(trace.as_json())


def test_no_terms_at_all(tiers):
    idx = _index()
    cands, trace = tiered_candidates([], tiers, idx, k=10)
    assert cands.size == 0
    assert trace.fell_through is True


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def test_round_trip_through_disk(tmp_path, tiers):
    tiers.save(tmp_path)
    assert (tmp_path / "champions.json").exists()
    assert (tmp_path / "tiers.npz").exists()

    back = TieredIndex.load(tmp_path)
    assert back.r == tiers.r
    for term in ("common", "mid", "eight"):
        assert back.champions(term).tolist() == tiers.champions(term).tolist()
    assert back.champions("rare") is None
    assert back.stats.terms_with_champions == tiers.stats.terms_with_champions


def test_load_without_a_build_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        TieredIndex.load(tmp_path)


def test_saving_an_empty_index(tmp_path):
    t = TieredIndex({}, r=10)
    t.save(tmp_path)
    back = TieredIndex.load(tmp_path)
    assert back.champions("anything") is None


# --------------------------------------------------------------------------
# The ordering uses the shipped ALPHA
# --------------------------------------------------------------------------
def test_champion_ordering_uses_the_same_alpha_as_the_net_score():
    """If these ever diverge, tier 1 is optimised for a different ranking."""
    import inspect

    src = inspect.getsource(champion_order)
    assert "config.ALPHA" in src


# --------------------------------------------------------------------------
# free_text_candidates — the drop-in for search.py (M3's integration point)
# --------------------------------------------------------------------------
def test_free_text_candidates_uses_s5_when_present(tiers):
    from nyaya.index.tiers import free_text_candidates

    class Ctx:
        index = _index()
        tiers = None

    c = Ctx()
    c.tiers = tiers
    cands, trace = free_text_candidates(["common", "mid"], c, k=3)
    assert trace is not None
    assert trace.tier == 1
    assert cands.size >= 3


def test_free_text_candidates_falls_back_identically_without_s5():
    """ctx.tiers=None must give byte-identical behaviour to the current code."""
    from nyaya.index.tiers import free_text_candidates

    class Ctx:
        index = _index()
        tiers = None

    cands, trace = free_text_candidates(["common", "mid"], Ctx(), k=3)
    assert trace is None
    expected = np.unique(np.concatenate([
        Ctx.index.postings("common").doc_ids,
        Ctx.index.postings("mid").doc_ids])).astype(np.int32)
    assert cands.tolist() == expected.tolist()


def test_free_text_candidates_with_no_known_terms():
    from nyaya.index.tiers import free_text_candidates

    class Ctx:
        index = _index()
        tiers = None

    cands, trace = free_text_candidates(["zzz"], Ctx(), k=3)
    assert cands.size == 0 and trace is None


def test_free_text_candidates_honours_a_restriction(tiers):
    from nyaya.index.tiers import free_text_candidates

    class Ctx:
        index = _index()
        tiers = None

    c = Ctx()
    c.tiers = tiers
    restrict = np.array([0, 1, 2], dtype=np.int32)
    cands, _trace = free_text_candidates(["common", "mid"], c, k=2,
                                         restrict=restrict)
    assert set(cands.tolist()) <= {0, 1, 2}
