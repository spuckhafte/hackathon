"""Tests for the scorers and the "why this result" panel.

The one that earns its place is
`test_bm25f_saturates_after_combining_zones_not_before`: it is the whole
difference between BM25F and "BM25 per zone, summed", it is easy to get wrong,
and a wrong implementation still produces plausible-looking numbers.

Network-free: the scorer tests run against a hand-built stub index, so they are
exact rather than approximate.  Author: Sumedha (M3)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nyaya import config
from nyaya.rank import bm25f, explain, tfidf, zonescore

NZ = len(config.ZONES)
ZI = config.ZONE_ID


# ==========================================================================
# A stub index with exactly the shape PositionalIndex exposes
# ==========================================================================


class StubPostings:
    def __init__(self, doc_ids, ztf):
        self.doc_ids = np.asarray(doc_ids, dtype=np.int32)
        self._ztf = np.asarray(ztf, dtype=np.int32)
        self.df = len(doc_ids)

    def zone_tf_matrix(self):
        return self._ztf

    def tf_array(self):
        return self._ztf.sum(axis=1)

    def positions_of(self, doc_id):
        return np.empty(0, dtype=np.int32)

    def zone_tf_of(self, doc_id):
        i = int(np.nonzero(self.doc_ids == doc_id)[0][0])
        return {z: int(self._ztf[i, j]) for z, j in ZI.items() if self._ztf[i, j]}

    def skip_stride(self):
        return 0


class StubIndex:
    """Three documents, six zones, a handful of terms."""

    def __init__(self, postings, zone_len, doc_norm=None, n=None):
        self._p = postings
        self.zone_len = np.asarray(zone_len, dtype=np.int32)
        self.doc_len = self.zone_len.sum(axis=1)
        self.doc_count = int(n or self.zone_len.shape[0])
        self.doc_norm = (np.asarray(doc_norm, dtype=np.float32)
                         if doc_norm is not None
                         else np.ones(self.zone_len.shape[0], dtype=np.float32))
        self.indexed = np.arange(self.zone_len.shape[0], dtype=np.int32)

    def postings(self, term):
        return self._p.get(term)

    def df(self, term):
        p = self._p.get(term)
        return p.df if p else 0

    def idf(self, term):
        d = self.df(term)
        if d <= 0:
            return 0.0
        n = self.doc_count
        return math.log(1.0 + (n - d + 0.5) / (d + 0.5))


class Ctx:
    def __init__(self, index):
        self.index = index
        self.authority = None


def _zrow(**kw):
    row = [0] * NZ
    for z, v in kw.items():
        row[ZI[z]] = v
    return row


@pytest.fixture
def stub():
    """doc 0: term in the title only.  doc 1: term in the body only.
    doc 2: term once in the title AND once in the body."""
    postings = {
        "alpha": StubPostings([0, 1, 2], [_zrow(title=1), _zrow(body=1),
                                          _zrow(title=1, body=1)]),
        "beta": StubPostings([1], [_zrow(body=3)]),
    }
    zone_len = [_zrow(title=10, body=1000),
                _zrow(title=10, body=1000),
                _zrow(title=10, body=1000)]
    return StubIndex(postings, zone_len, n=100)


# ==========================================================================
# lnc.ltc
# ==========================================================================


def test_query_weights_are_unit_length(stub):
    w = tfidf.ltc_query_weights(["alpha", "beta"], stub)
    assert math.isclose(sum(v * v for v in w.values()), 1.0, rel_tol=1e-9)


def test_idf_is_applied_once_on_the_query_side(stub):
    """That is what the 'n' in lnc and the 't' in ltc mean."""
    w = tfidf.ltc_query_weights(["alpha", "beta"], stub)
    # beta is rarer than alpha, so it must carry more query weight
    assert w["beta"] > w["alpha"]
    # the document side must NOT multiply by idf again: doc 1 has beta tf=3 and
    # alpha tf=1, and its score is the query weights times log tfs alone.
    s = tfidf.lnc_ltc(np.array([1], dtype=np.int32), ["alpha", "beta"], Ctx(stub))
    expected = w["alpha"] * 1.0 + w["beta"] * (1.0 + math.log(3))
    assert float(s[0]) == pytest.approx(expected, rel=1e-9)


def test_a_term_in_every_document_carries_almost_no_query_weight(stub):
    """idf uses ln(1 + (N-df+0.5)/(df+0.5)), which is small but positive at
    df = N; the stop list is what removes such a term, and this checks the
    scorer would not have let it matter anyway."""
    everywhere = StubPostings(list(range(100)), [_zrow(body=1)] * 100)
    idx = StubIndex({**stub._p, "the": everywhere}, stub.zone_len, n=100)
    assert idx.idf("the") == pytest.approx(math.log(1 + 0.5 / 100.5), abs=1e-9)
    w = tfidf.ltc_query_weights(["the", "beta"], idx)
    assert w["the"] < 0.01 * w["beta"]


def test_scores_align_with_the_candidate_array_whatever_its_order(stub):
    ctx = Ctx(stub)
    asc = tfidf.lnc_ltc(np.array([0, 1, 2], dtype=np.int32), ["alpha"], ctx)
    desc = tfidf.lnc_ltc(np.array([2, 1, 0], dtype=np.int32), ["alpha"], ctx)
    assert list(asc) == pytest.approx(list(desc)[::-1])


def test_a_document_not_in_the_postings_scores_zero(stub):
    s = tfidf.lnc_ltc(np.array([0, 1, 2], dtype=np.int32), ["beta"], Ctx(stub))
    assert float(s[0]) == 0.0 and float(s[2]) == 0.0 and float(s[1]) > 0.0


def test_the_join_does_not_credit_a_neighbouring_doc_id(stub):
    """searchsorted alone would credit doc 5 with doc 7's postings; the equality
    check is what stops it."""
    p = StubPostings([7], [_zrow(body=5)])
    idx = StubIndex({"gamma": p}, [_zrow(body=100)] * 10, n=10)
    s = tfidf.lnc_ltc(np.array([5, 6], dtype=np.int32), ["gamma"], Ctx(idx))
    assert float(s.sum()) == 0.0


# ==========================================================================
# weighted zone
# ==========================================================================


def test_per_zone_contributions_sum_to_the_combined_score(stub):
    ctx = Ctx(stub)
    ids = np.array([0, 1, 2], dtype=np.int32)
    combined, per_zone = zonescore.zone_scores(ids, ["alpha"], ctx)
    total = sum(v for v in per_zone.values())
    assert list(combined) == pytest.approx(list(total))


def test_the_bench_zone_is_never_scored():
    """config gives it weight 0: it is a filter-only zone, reachable through
    judge: but never ranked on."""
    assert config.ZONE_WEIGHTS["bench"] == 0.0
    assert "bench" not in zonescore._scored_zones(config.ZONE_WEIGHTS)


def test_uniform_and_weighted_differ_and_that_is_the_s1_s2_ablation(stub):
    ctx = Ctx(stub)
    ids = np.array([0, 1], dtype=np.int32)      # doc 0 title hit, doc 1 body hit
    flat = zonescore.uniform(ids, ["alpha"], ctx)
    tuned = zonescore.score(ids, ["alpha"], ctx)
    assert not np.allclose(flat, tuned)
    # The weighted set now *de-emphasises* a title hit relative to a body hit,
    # which is the measured correction: a cause title in this corpus is a party
    # name, so a title match is weak evidence about subject matter.  The test
    # asserts the ratio follows config rather than hard-coding a direction.
    ratio = (config.ZONE_WEIGHTS["title"] / config.ZONE_WEIGHTS["body"])
    assert (tuned[0] / tuned[1] > flat[0] / flat[1]) == (ratio > 1.0)
    assert config.ZONE_WEIGHTS["title"] < config.ZONE_WEIGHTS["body"],         "the measured title weight should sit below the body weight"


def test_pivoted_normalisation_damps_the_short_zone_bonus():
    """The measured bug: plain cosine let a 10-token title outscore everything.

    The zone lengths have to *vary* for a pivot to do anything, so this builds
    its own index: doc 0 has an unusually short body, doc 1 an average one.
    """
    postings = {"alpha": StubPostings([0, 1], [_zrow(body=1), _zrow(body=1)])}
    idx = StubIndex(postings, [_zrow(body=25), _zrow(body=2500)], n=100)
    ctx = Ctx(idx)
    ids = np.array([0, 1], dtype=np.int32)
    old = zonescore.PIVOT_SLOPE
    try:
        zonescore.PIVOT_SLOPE = 1.0
        plain = zonescore.score(ids, ["alpha"], ctx)
        zonescore.PIVOT_SLOPE = 0.3
        pivoted = zonescore.score(ids, ["alpha"], ctx)
    finally:
        zonescore.PIVOT_SLOPE = old
    assert plain[0] / plain[1] > pivoted[0] / pivoted[1]


# ==========================================================================
# BM25F — the property that defines it
# ==========================================================================


def test_bm25f_saturates_after_combining_zones_not_before(stub):
    """One occurrence in the title plus one in the body must score **less** than
    the sum of two separate first occurrences.

    That inequality is the whole difference between BM25F and "BM25 per zone,
    summed".  Saturating per zone would give doc 2 (title + body) essentially
    the score of doc 0 plus the score of doc 1; saturating once, after the
    weighted tfs are added, gives it strictly less.
    """
    ctx = Ctx(stub)
    s = bm25f.bm25f_scores(np.array([0, 1, 2], dtype=np.int32), ["alpha"], ctx)
    title_only, body_only, both = float(s[0]), float(s[1]), float(s[2])
    assert both > title_only and both > body_only      # more evidence scores more
    assert both < title_only + body_only               # but with diminishing returns


def test_the_saturation_is_the_one_in_the_formula(stub):
    """Recompute BM25F by hand for one document and match it exactly."""
    ctx = Ctx(stub)
    idx = stub
    avg = bm25f.avg_zone_lengths(idx)
    w, b, k1 = config.ZONE_WEIGHTS, config.BM25F_B, config.BM25F_K1
    # doc 2 has alpha once in the title and once in the body
    tf_tilde = 0.0
    for zone, tf in (("title", 1), ("body", 1)):
        j = ZI[zone]
        ln = float(idx.zone_len[2, j])
        tf_tilde += w[zone] * tf / (1 - b[zone] + b[zone] * ln / avg[j])
    expected = idx.idf("alpha") * tf_tilde / (k1 + tf_tilde)
    got = bm25f.bm25f_scores(np.array([2], dtype=np.int32), ["alpha"], ctx)
    assert float(got[0]) == pytest.approx(expected, rel=1e-12)


def test_avg_zone_length_ignores_documents_without_the_zone():
    """Averaging over the zeros would make every document that has the zone look
    over-long."""
    idx = StubIndex({}, [_zrow(title=10, body=100),
                         _zrow(body=100),
                         _zrow(title=30, body=100)], n=3)
    avg = bm25f.avg_zone_lengths(idx)
    assert avg[ZI["title"]] == pytest.approx(20.0)      # not 40/3


def test_bm25f_ignores_a_zone_whose_weight_is_zero(stub):
    ctx = Ctx(stub)
    p = StubPostings([0], [_zrow(bench=5)])
    idx = StubIndex({"judgename": p}, stub.zone_len, n=100)
    s = bm25f.bm25f_scores(np.array([0], dtype=np.int32), ["judgename"], Ctx(idx))
    assert float(s[0]) == 0.0


def test_a_repeated_query_term_is_one_term(stub):
    ctx = Ctx(stub)
    once = bm25f.bm25f_scores(np.array([2], dtype=np.int32), ["alpha"], ctx)
    twice = bm25f.bm25f_scores(np.array([2], dtype=np.int32),
                               ["alpha", "alpha"], ctx)
    assert float(once[0]) == pytest.approx(float(twice[0]))


# ==========================================================================
# explain
# ==========================================================================


def _explanation(**kw):
    from nyaya.contracts import ScoreExplanation

    ex = ScoreExplanation(**kw)
    return ex


def test_verify_reproduces_the_net_score():
    ex = _explanation(text_norm=0.5, g=0.2, proximity=1.0, alpha=0.8, beta=0.05)
    ex.net = 0.8 * 0.5 + 0.2 * 0.2 + 0.05 * 1.0
    v = explain.verify(ex)
    assert v["ok"]
    assert v["sum"] == pytest.approx(ex.net)
    assert v["alpha_x_text"] == pytest.approx(0.4)
    assert v["one_minus_alpha_x_g"] == pytest.approx(0.04)
    assert v["beta_x_proximity"] == pytest.approx(0.05)


def test_verify_catches_a_net_score_that_does_not_add_up():
    ex = _explanation(text_norm=0.5, g=0.2, alpha=0.8, beta=0.0)
    ex.net = 99.0
    assert not explain.verify(ex)["ok"]


def test_build_fills_the_contract_and_the_arithmetic_checks_out(stub):
    ctx = Ctx(stub)
    ids = np.array([0, 1, 2], dtype=np.int32)
    inp = explain.ExplainInputs(doc_id=2, doc_ids=ids, terms=["alpha"], ctx=ctx,
                                scorer="bm25f", text_raw=1.5, text_norm=0.75,
                                proximity=1.0, prox_kind="p")
    ex = explain.build(inp)
    assert ex.scorer == "bm25f"
    assert ex.top_terms and ex.top_terms[0][0] == "alpha"
    assert explain.verify(ex)["ok"]
    assert any("saturates AFTER" in n for n in ex.notes)
    assert any("same paragraph" in n for n in ex.notes)


def test_the_zone_scorer_explanation_sums_to_the_text_score(stub):
    ctx = Ctx(stub)
    ids = np.array([0, 1, 2], dtype=np.int32)
    combined, _ = zonescore.zone_scores(ids, ["alpha"], ctx)
    ex = explain.build(explain.ExplainInputs(
        doc_id=2, doc_ids=ids, terms=["alpha"], ctx=ctx, scorer="weighted-zone",
        text_raw=float(combined[2]), text_norm=1.0))
    assert sum(ex.per_zone.values()) == pytest.approx(float(combined[2]), rel=1e-9)
    assert any("sum to the text score" in n for n in ex.notes)


def test_a_crosswalk_expansion_is_explained_not_just_applied(stub):
    ctx = Ctx(stub)
    ex = explain.build(explain.ExplainInputs(
        doc_id=0, doc_ids=np.array([0], dtype=np.int32), terms=["alpha"],
        ctx=ctx, scorer="lnc.ltc",
        expansions={"ipc§302": ["bns§103"]}))
    note = " ".join(ex.notes)
    assert "ipc§302 OR bns§103" in note
    assert "mob-lynching" in note          # the verified caveat travels with it


def test_a_mode_without_authority_says_so(stub):
    ex = explain.build(explain.ExplainInputs(
        doc_id=0, doc_ids=np.array([0], dtype=np.int32), terms=["alpha"],
        ctx=Ctx(stub), scorer="lnc.ltc", use_authority=False))
    assert ex.g == 0.0
    assert any("does not use authority" in n for n in ex.notes)


def test_render_prints_the_arithmetic(stub):
    ex = _explanation(text_norm=0.5, g=0.2, alpha=0.8, beta=0.0, scorer="bm25f")
    ex.net = 0.8 * 0.5 + 0.2 * 0.2
    out = explain.render(ex, doc_id=7)
    assert "why this result" in out and "doc 7" in out
    assert "alpha x text" in out and "net" in out
    assert "MISMATCH" not in out
