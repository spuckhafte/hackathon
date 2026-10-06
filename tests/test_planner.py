"""Tests for the Boolean planner.  Network-free, built on a stub index.

The stub is the point: the planner's contract with S3 is tiny (``df``,
``postings``, ``prefix_terms``, ``doc_count``, ``indexed``), so a hand-written
index with known document frequencies lets the *ordering* decisions be asserted
exactly rather than inferred from timings.
"""
from __future__ import annotations

import numpy as np
import pytest

from nyaya.contracts import (
    AndNode,
    FilterNode,
    NotNode,
    OrNode,
    ParsedQuery,
    PhraseNode,
    ProxNode,
    TermNode,
)
from nyaya.query.planner import (
    ORDER_POLICY,
    PlanStep,
    SearchContext,
    _order_operands,
    compare_orderings,
    difference,
    estimate_size,
    execute,
    execute_explained,
    explain_plan,
    flatten_and,
    intersect,
    plan_query,
    union,
)


# --------------------------------------------------------------------------
# A stub S3 with known dfs
# --------------------------------------------------------------------------
class StubPostings:
    def __init__(self, doc_ids):
        self._d = np.asarray(sorted(doc_ids), dtype=np.int32)
        self.df = int(self._d.size)
        self.probes = 0

    @property
    def doc_ids(self):
        return self._d

    def index_of(self, doc_id):
        self.probes += 1
        j = int(np.searchsorted(self._d, doc_id))
        return j if j < self.df and int(self._d[j]) == doc_id else -1

    def zone_tf_of(self, doc_id):
        # every term sits in the body; "privacy" is also in the title of doc 1
        j = self.index_of(doc_id)
        if j < 0:
            return {}
        return {"body": 2, "title": 1} if doc_id == 1 else {"body": 2}

    def skip_stride(self):
        return 2


class StubIndex:
    """`rare` df 3, `mid` df 6, `common` df 20."""

    def __init__(self):
        self._p = {
            "rare": StubPostings([2, 5, 9]),
            "mid": StubPostings([1, 2, 3, 5, 8, 9]),
            "common": StubPostings(range(20)),
            "bailable": StubPostings([4, 5]),
            "bailment": StubPostings([6]),
        }
        self.doc_count = 20
        self.indexed = np.arange(20, dtype=np.int32)

    def df(self, term):
        p = self._p.get(term)
        return p.df if p else 0

    def postings(self, term):
        return self._p.get(term)

    def prefix_terms(self, prefix):
        return sorted(t for t in self._p if t.startswith(prefix))


class StubParams:
    """S2 stub: `year:>=2000` is 4 documents, `act:ipc` is 2."""

    def __init__(self):
        self.n_docs = 20
        self._sel = {
            ("year", ">=", "2000"): np.array([1, 2, 5, 9], dtype=np.int32),
            ("act", "=", "ipc"): np.array([2, 5], dtype=np.int32),
            ("bench", ">=", "5"): np.arange(0, 12, dtype=np.int32),
        }

    def select(self, field, op, value):
        key = (str(field), str(op), str(value))
        if key in self._sel:
            return self._sel[key]
        raise KeyError(field)


@pytest.fixture
def ctx():
    return SearchContext(index=StubIndex(), params=StubParams())


# --------------------------------------------------------------------------
# Set algebra
# --------------------------------------------------------------------------
def test_set_primitives():
    a = np.array([1, 3, 5, 7], dtype=np.int32)
    b = np.array([3, 4, 5], dtype=np.int32)
    assert intersect(a, b).tolist() == [3, 5]
    assert union([a, b]).tolist() == [1, 3, 4, 5, 7]
    assert difference(a, b).tolist() == [1, 7]
    assert intersect(a, np.empty(0, np.int32)).tolist() == []
    assert difference(np.empty(0, np.int32), a).tolist() == []
    assert union([]).tolist() == []


def test_results_are_always_sorted_int32(ctx):
    for node in (TermNode(term="mid"),
                 AndNode([TermNode(term="mid"), TermNode(term="rare")]),
                 OrNode([TermNode(term="rare"), TermNode(term="bailment")]),
                 NotNode(TermNode(term="mid"), TermNode(term="rare"))):
        out = execute(node, ctx)
        assert out.dtype == np.int32
        assert out.tolist() == sorted(out.tolist())


# --------------------------------------------------------------------------
# Correctness of each node type
# --------------------------------------------------------------------------
def test_term(ctx):
    assert execute(TermNode(term="rare"), ctx).tolist() == [2, 5, 9]


def test_unknown_term_is_empty_not_an_error(ctx):
    assert execute(TermNode(term="zzz"), ctx).tolist() == []


def test_and(ctx):
    node = AndNode([TermNode(term="mid"), TermNode(term="rare")])
    assert execute(node, ctx).tolist() == [2, 5, 9]


def test_or(ctx):
    node = OrNode([TermNode(term="rare"), TermNode(term="bailment")])
    assert execute(node, ctx).tolist() == [2, 5, 6, 9]


def test_not_is_a_set_difference(ctx):
    node = NotNode(TermNode(term="mid"), TermNode(term="rare"))
    assert execute(node, ctx).tolist() == [1, 3, 8]


def test_filter(ctx):
    assert execute(FilterNode("act", "=", "ipc"), ctx).tolist() == [2, 5]


def test_unknown_filter_degrades_instead_of_crashing(ctx):
    """A search box must not 500 because someone invented a field name."""
    ids, plan, _stats = execute_explained(FilterNode("vibes", "=", "good"), ctx)
    assert ids.size == ctx.doc_count
    assert "unusable filter" in plan.note


def test_prefix_term_is_an_or_over_the_dictionary(ctx):
    ids = execute(TermNode(term="bail", prefix=True), ctx)
    assert ids.tolist() == [4, 5, 6]


def test_prefix_with_no_matches(ctx):
    ids, plan, _ = execute_explained(TermNode(term="zzz", prefix=True), ctx)
    assert ids.tolist() == []
    assert "no dictionary term" in plan.note


def test_zone_restriction(ctx):
    """Only doc 1 has the term in its title, per the stub's zone_tf_of."""
    ids = execute(TermNode(term="mid", zone="title"), ctx)
    assert ids.tolist() == [1]


def test_unknown_zone_is_ignored_with_a_note(ctx):
    ids, plan, _ = execute_explained(TermNode(term="rare", zone="footnote"), ctx)
    assert ids.tolist() == [2, 5, 9]
    assert "unknown zone" in plan.note


# --------------------------------------------------------------------------
# THE optimisation: ascending document frequency
# --------------------------------------------------------------------------
def test_and_operands_are_ordered_rarest_first(ctx):
    node = AndNode([TermNode(term="common"), TermNode(term="mid"),
                    TermNode(term="rare")])
    ordered = _order_operands(node.operands, ctx)
    assert [o.term for o in ordered] == ["rare", "mid", "common"]


def test_the_plan_records_the_order_it_chose(ctx):
    node = AndNode([TermNode(term="common"), TermNode(term="rare")])
    _ids, plan, _stats = execute_explained(node, ctx)
    assert plan.op == "AND"
    assert "TERM(rare)[3]" in plan.note
    assert plan.note.index("rare") < plan.note.index("common")


def test_intermediate_result_shrinks_down_the_chain(ctx):
    node = AndNode([TermNode(term="common"), TermNode(term="mid"),
                    TermNode(term="rare")])
    _ids, plan, _stats = execute_explained(node, ctx)
    runnings = [c.running for c in plan.children]
    assert runnings == sorted(runnings, reverse=True)


def test_ordering_does_not_change_the_answer(ctx):
    node = AndNode([TermNode(term="common"), TermNode(term="mid"),
                    TermNode(term="rare")])
    cmp = compare_orderings(node, ctx)
    assert cmp["same_answer"] is True
    assert cmp["results"] == 3


def test_rarest_first_compares_fewer_entries_than_as_written(ctx):
    node = AndNode([TermNode(term="common"), TermNode(term="mid"),
                    TermNode(term="rare")])
    cmp = compare_orderings(node, ctx)
    assert cmp["policies"]["size"]["compared"] < \
        cmp["policies"]["as-written"]["compared"]
    assert cmp["speedup_vs_as_written"] > 1.0


def test_an_empty_intermediate_result_skips_the_rest_of_the_chain(ctx):
    node = AndNode([TermNode(term="common"), TermNode(term="bailment"),
                    TermNode(term="rare")])
    ids, plan, _stats = execute_explained(node, ctx)
    assert ids.tolist() == []
    # rare(3) and bailment(1) are evaluated; common(20) must never be opened
    labels = [c.label for c in plan.children]
    assert "TERM(common)" not in labels


def test_skip_pointers_are_used_for_short_against_long(ctx):
    """A 2-document candidate set against a 20-document list is probed, not merged.

    SKIP_RATIO is 8, so 20 >= 8*2 crosses the threshold.
    """
    node = AndNode([TermNode(term="bailable"), TermNode(term="common")])
    ids, plan, stats = execute_explained(node, ctx)
    assert ids.tolist() == [4, 5]
    assert stats.skip_probes == 2
    common = next(c for c in plan.children if c.label == "TERM(common)")
    assert "skip-probed" in common.note


def test_a_near_equal_pair_is_merged_rather_than_probed(ctx):
    """Probing only pays when the lists are lopsided; 3 vs 6 is a plain merge."""
    node = AndNode([TermNode(term="rare"), TermNode(term="mid")])
    _ids, _plan, stats = execute_explained(node, ctx)
    assert stats.skip_probes == 0
    assert stats.merges == 1


def test_flatten_and_collapses_nesting():
    inner = AndNode([TermNode(term="a"), TermNode(term="b")])
    outer = AndNode([inner, TermNode(term="c")])
    assert [n.term for n in flatten_and(outer)] == ["a", "b", "c"]


def test_nested_and_is_optimised_as_one_chain(ctx):
    """`(common AND mid) AND rare` must still start with rare."""
    nested = AndNode([AndNode([TermNode(term="common"), TermNode(term="mid")]),
                      TermNode(term="rare")])
    _ids, plan, _stats = execute_explained(nested, ctx)
    assert plan.note.index("rare") < plan.note.index("common")


# --------------------------------------------------------------------------
# Ordering policy — measured, not assumed
# --------------------------------------------------------------------------
def test_default_policy_is_pure_size_ordering():
    assert ORDER_POLICY == "size"


def test_filters_first_is_wrong_when_a_term_is_rarer_than_the_filters(ctx):
    """bench:>=5 is 12 docs, rare is 3: forcing the filter first is worse."""
    node = AndNode([TermNode(term="rare"), FilterNode("bench", ">=", "5")])
    by_size = _order_operands(node.operands, ctx, "size")
    by_filter = _order_operands(node.operands, ctx, "filters-first")
    assert isinstance(by_size[0], TermNode)
    assert isinstance(by_filter[0], FilterNode)
    cmp = compare_orderings(node, ctx)
    assert cmp["same_answer"] is True
    assert cmp["policies"]["size"]["compared"] <= \
        cmp["policies"]["filters-first"]["compared"]


def test_a_selective_filter_still_sorts_first_under_the_size_rule(ctx):
    """The size rule subsumes the heuristic: act:ipc is 2 docs, mid is 6."""
    node = AndNode([TermNode(term="mid"), FilterNode("act", "=", "ipc")])
    ordered = _order_operands(node.operands, ctx)
    assert isinstance(ordered[0], FilterNode)


def test_a_filter_wins_an_exact_tie_because_it_needs_no_postings_io(ctx):
    node = AndNode([TermNode(term="bailable"), FilterNode("act", "=", "ipc")])
    # both are 2 documents
    assert estimate_size(node.operands[0], ctx) == 2
    assert estimate_size(node.operands[1], ctx) == 2
    assert isinstance(_order_operands(node.operands, ctx)[0], FilterNode)


# --------------------------------------------------------------------------
# Size estimation
# --------------------------------------------------------------------------
def test_estimate_term_is_the_exact_df(ctx):
    assert estimate_size(TermNode(term="mid"), ctx) == 6


def test_estimate_and_is_the_minimum_child(ctx):
    node = AndNode([TermNode(term="common"), TermNode(term="rare")])
    assert estimate_size(node, ctx) == 3


def test_estimate_or_is_the_sum_of_children(ctx):
    node = OrNode([TermNode(term="rare"), TermNode(term="mid")])
    assert estimate_size(node, ctx) == 9


def test_estimate_not_is_bounded_by_the_left_side(ctx):
    node = NotNode(TermNode(term="mid"), TermNode(term="common"))
    assert estimate_size(node, ctx) == 6


def test_estimate_phrase_is_bounded_by_its_rarest_term(ctx):
    assert estimate_size(PhraseNode(terms=["common", "rare"]), ctx) == 3


def test_estimate_prox_is_bounded_by_its_narrower_side(ctx):
    node = ProxNode(kind="n", k=5, left=TermNode(term="common"),
                    right=TermNode(term="rare"))
    assert estimate_size(node, ctx) == 3


# --------------------------------------------------------------------------
# NOT and OR evaluation order
# --------------------------------------------------------------------------
def test_not_evaluates_the_right_side_only_against_the_left(ctx):
    node = NotNode(TermNode(term="rare"), TermNode(term="common"))
    _ids, plan, _stats = execute_explained(node, ctx)
    right = plan.children[1]
    assert "only against the left result" in right.note
    assert right.running <= 3


def test_or_does_not_push_candidates_into_its_branches(ctx):
    """A document can satisfy a union through a branch the candidates exclude,
    so pushing the candidate set down would drop real matches."""
    node = AndNode([TermNode(term="rare"),
                    OrNode([TermNode(term="bailment"), TermNode(term="mid")])])
    assert execute(node, ctx).tolist() == [2, 5, 9]


# --------------------------------------------------------------------------
# Whole-query planning
# --------------------------------------------------------------------------
def test_plan_query_folds_filters_in_beside_the_tree(ctx):
    q = ParsedQuery(raw="x", root=TermNode(term="mid"),
                    filters=[FilterNode("act", "=", "ipc")],
                    is_free_text=False)
    ids, plan, _stats = plan_query(q, ctx)
    assert ids.tolist() == [2, 5]
    assert plan.op == "AND"


def test_free_text_is_a_union_not_a_conjunction(ctx):
    """A bag of words must not be silently ANDed — that destroys recall."""
    q = ParsedQuery(raw="x", root=None, free_terms=["rare", "bailment"],
                    is_free_text=True)
    ids, plan, _stats = plan_query(q, ctx)
    assert ids.tolist() == [2, 5, 6, 9]
    assert plan.op == "OR"


def test_filters_only_query_needs_no_text(ctx):
    q = ParsedQuery(raw="act:ipc", root=None, free_terms=[],
                    filters=[FilterNode("act", "=", "ipc")], is_free_text=False)
    ids, _plan, _stats = plan_query(q, ctx)
    assert ids.tolist() == [2, 5]


def test_an_empty_query_matches_every_indexed_document(ctx):
    q = ParsedQuery(raw="", root=None, free_terms=[], is_free_text=True)
    ids, plan, _stats = plan_query(q, ctx)
    assert ids.size == 20
    assert plan is None


# --------------------------------------------------------------------------
# explain_plan is a rendering of data, not a log
# --------------------------------------------------------------------------
def test_explain_plan_shows_dfs_and_running_sizes(ctx):
    node = AndNode([TermNode(term="common"), TermNode(term="rare")])
    _ids, plan, stats = execute_explained(node, ctx)
    out = explain_plan(plan, stats, title="t")
    assert "TERM(rare)" in out and "TERM(common)" in out
    assert "est df" in out and "running" in out
    assert "entries compared" in out
    # every line is the same width - it is a box
    widths = {len(line) for line in out.splitlines()}
    assert len(widths) == 1


def test_explain_plan_handles_no_plan():
    assert "no plan" in explain_plan(None)


def test_plan_is_json_serialisable(ctx):
    import json

    node = AndNode([TermNode(term="rare"), FilterNode("act", "=", "ipc")])
    _ids, plan, stats = execute_explained(node, ctx)
    blob = json.dumps({"plan": plan.as_json(), "stats": stats.as_json()})
    assert "FILTER" in blob and "merge_entries" in blob


def test_plan_walk_visits_every_step(ctx):
    node = AndNode([TermNode(term="rare"), TermNode(term="mid")])
    _ids, plan, _stats = execute_explained(node, ctx)
    assert len(list(plan.walk())) == 3        # the AND plus two children


# --------------------------------------------------------------------------
# Degradation when M1's proximity module is absent
# --------------------------------------------------------------------------
def test_phrase_degrades_to_and_and_says_so(ctx, monkeypatch):
    import nyaya.query.planner as P

    monkeypatch.setattr(P, "_proximity_module", lambda: None)
    _ids, plan, _stats = execute_explained(
        PhraseNode(terms=["rare", "mid"]), ctx)
    assert "degraded to AND" in plan.note


def test_prox_degrades_to_and_and_says_so(ctx, monkeypatch):
    import nyaya.query.planner as P

    monkeypatch.setattr(P, "_proximity_module", lambda: None)
    node = ProxNode(kind="n", k=5, left=TermNode(term="rare"),
                    right=TermNode(term="mid"))
    ids, plan, _stats = execute_explained(node, ctx)
    assert ids.tolist() == [2, 5, 9]
    assert "window not enforced" in plan.note


def test_prox_calls_member_1s_module_when_it_exists(ctx, monkeypatch):
    import nyaya.query.planner as P

    seen = {}

    class FakeProx:
        @staticmethod
        def proximity_docs(left_ids, right_ids, lt, rt, kind, k, c):
            seen.update(kind=kind, k=k, left_terms=lt, right_terms=rt)
            return np.array([5], dtype=np.int32)

    monkeypatch.setattr(P, "_proximity_module", lambda: FakeProx)
    node = ProxNode(kind="s", k=0, left=TermNode(term="common"),
                    right=TermNode(term="rare"))
    ids, _plan, _stats = execute_explained(node, ctx)
    assert ids.tolist() == [5]
    assert seen["kind"] == "s"
    # the cheaper side is evaluated first, so `rare` leads
    assert seen["left_terms"] == ["rare"]


def test_prox_swaps_to_put_the_cheaper_side_first(ctx, monkeypatch):
    import nyaya.query.planner as P

    monkeypatch.setattr(P, "_proximity_module", lambda: None)
    node = ProxNode(kind="n", k=3, left=TermNode(term="common"),
                    right=TermNode(term="rare"))
    _ids, plan, _stats = execute_explained(node, ctx)
    assert "cheaper side first" in plan.note


# --------------------------------------------------------------------------
# No index loaded at all
# --------------------------------------------------------------------------
def test_filters_still_work_without_a_positional_index():
    ctx = SearchContext(index=None, params=StubParams())
    assert execute(FilterNode("act", "=", "ipc"), ctx).tolist() == [2, 5]


def test_term_without_an_index_is_empty():
    ctx = SearchContext(index=None, params=StubParams())
    assert execute(TermNode(term="rare"), ctx).tolist() == []
