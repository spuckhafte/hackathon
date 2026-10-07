"""The Boolean engine: execute a parsed query against S3 and S2.

Member 2 (Suvrat).  Everything here is one idea — **process operands in order of
increasing document frequency** (Manning, IIR 1.3) — applied to every node type.

Why it matters.  An intersection of sorted posting lists costs roughly the size
of the *smaller* list per step, and the intermediate result can only shrink.  So
for ``murder AND intention AND court``, starting with ``court`` (df ~13,000) means
every later merge walks a 13,000-long list; starting with ``murder`` (df ~1,900)
means the second merge walks 1,900 and the third walks whatever survived, which
is smaller again.  The answer is identical; the work is not.  Concretely, on this
corpus a three-term AND chain ordered rarest-first touches roughly an order of
magnitude fewer postings than the same chain ordered as the user typed it, and
:func:`explain_plan` prints exactly that so the claim is checkable rather than
asserted.

The rules, in the order the optimiser applies them:

1. **Filters first.**  A parametric selection (store S2) is usually far more
   selective than any term and costs nothing to produce — ``bench:>=5`` is 1,176
   of 13,942 documents, ``act:ndps`` fewer still.  They are also *cheap*: a
   numeric field is two binary searches.  So filters are evaluated first and
   their intersection becomes the candidate set everything else narrows.
2. **AND: ascending estimated size.**  For a term that is a document frequency
   straight from the dictionary; for a subtree it is :func:`estimate_size`.
3. **OR: ascending too**, which does not change the cost of a union but keeps
   the plan readable, and lets a cheap child short-circuit nothing — a union's
   size is the *sum* of its children, so it is estimated that way.
4. **NOT last.**  ``a NOT b`` is a set difference, so ``b`` is only worth
   computing against whatever ``a`` actually produced.
5. **Skip pointers** are used by the short-against-long intersection: when one
   list is much smaller than the other we probe the long list with
   ``index_of()`` (M1's sqrt(P)-spaced skips) instead of merging it end to end.

Proximity and phrase matching are Member 1's (``nyaya/query/proximity.py``); this
module calls them and never reimplements them.

The plan is **data, not a log line**: :func:`plan` returns a :class:`PlanStep`
tree and :func:`explain_plan` renders it with the df of every operand and the
intermediate result size after each step.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.contracts import (
    AndNode,
    FilterNode,
    Node,
    NotNode,
    OrNode,
    ParsedQuery,
    PhraseNode,
    ProxNode,
    TermNode,
)
from nyaya.utils import ensure_utf8_stdout

EMPTY = np.empty(0, dtype=np.int32)

#: An intersection where one side is at least this many times longer than the
#: other is done by probing the long list through its skip pointers rather than
#: merging both end to end.
SKIP_RATIO = 8


# ==========================================================================
# 1. The context the planner executes against
# ==========================================================================


class SearchContext:
    """The loaded stores.  ``nyaya/search.py`` (M3) builds the real one.

    Kept duck-typed on purpose: the planner needs ``.index`` (S3) and ``.params``
    (S2) and nothing else, so a test can pass a stub with two attributes.
    """

    def __init__(self, index=None, params=None, paras=None, authority=None,
                 tiers=None, docs=None, mode: str = "S4"):
        self.index = index
        self.params = params
        self.paras = paras
        self.authority = authority
        self.tiers = tiers
        self.docs = docs or []
        self.mode = mode

    @property
    def doc_count(self) -> int:
        if self.index is not None:
            return int(self.index.doc_count)
        if self.params is not None:
            return int(self.params.n_docs)
        return 0

    @property
    def all_docs(self) -> np.ndarray:
        """Every indexed doc id, for ``NOT`` and for a filter-only query."""
        if self.index is not None and hasattr(self.index, "indexed"):
            return np.asarray(self.index.indexed, dtype=np.int32)
        n = self.doc_count
        return np.arange(n, dtype=np.int32)


# ==========================================================================
# 2. Sorted-array set algebra (the same primitives S2 and S3 both hand back)
# ==========================================================================


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Plain sorted-array intersection, smaller list driving."""
    if a.size == 0 or b.size == 0:
        return EMPTY
    if a.size > b.size:
        a, b = b, a
    return np.intersect1d(a, b, assume_unique=True).astype(np.int32)


def intersect_with_skips(small: np.ndarray, postings, probes: Dict[str, int]
                         ) -> np.ndarray:
    """Intersect a short candidate list against a long postings list by probing.

    This is where M1's skip pointers earn their keep.  Merging a 20-document
    candidate set against a 13,000-document postings list end to end reads 13,000
    entries; probing the long list 20 times through its sqrt(P)-spaced skip array
    reads about 20 * (sqrt(13000)/2) and, more importantly, touches only a few
    memory-mapped pages.  Falls back to a merge when the postings object has no
    ``index_of``.
    """
    probe = getattr(postings, "index_of", None)
    if probe is None:
        return intersect(small, np.asarray(postings.doc_ids, dtype=np.int32))
    keep = np.empty(small.size, dtype=np.int32)
    n = 0
    for doc_id in small.tolist():
        if probe(int(doc_id)) >= 0:
            keep[n] = doc_id
            n += 1
    probes["skip_probes"] = probes.get("skip_probes", 0) + int(small.size)
    return keep[:n]


def union(arrays: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(a, dtype=np.int32) for a in arrays if a is not None and a.size]
    if not arrays:
        return EMPTY
    if len(arrays) == 1:
        return arrays[0]
    return np.unique(np.concatenate(arrays)).astype(np.int32)


def difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return EMPTY
    if b.size == 0:
        return a
    return np.setdiff1d(a, b, assume_unique=True).astype(np.int32)


# ==========================================================================
# 3. Size estimation — what the optimiser sorts on
# ==========================================================================


def estimate_size(node: Node, ctx: SearchContext) -> int:
    """Estimated number of documents ``node`` will produce.

    For a term this is the exact df, read from the dictionary for free — no
    postings are touched.  For a composite it is the standard cheap bound:
    an AND is at most its smallest child, an OR at most the sum of its children,
    a NOT at most its left child.  A proximity node is bounded by its narrower
    side, because a proximity match implies both terms are present.
    """
    if isinstance(node, TermNode):
        return _term_df(node, ctx)
    if isinstance(node, PhraseNode):
        # Every term must co-occur, so the rarest term bounds it; a phrase is
        # strictly rarer still, which is why phrases make such good first steps.
        dfs = [ctx.index.df(t) for t in node.terms] if ctx.index else []
        return min(dfs) if dfs else 0
    if isinstance(node, ProxNode):
        return min(estimate_size(node.left, ctx), estimate_size(node.right, ctx))
    if isinstance(node, AndNode):
        sizes = [estimate_size(c, ctx) for c in node.operands]
        return min(sizes) if sizes else 0
    if isinstance(node, OrNode):
        return sum(estimate_size(c, ctx) for c in node.operands)
    if isinstance(node, NotNode):
        return estimate_size(node.left, ctx)
    if isinstance(node, FilterNode):
        return _filter_size(node, ctx)
    return ctx.doc_count


def _term_df(node: TermNode, ctx: SearchContext) -> int:
    if ctx.index is None:
        return 0
    if node.prefix:
        # A root query is an OR over everything the prefix scan finds.
        return sum(ctx.index.df(t) for t in ctx.index.prefix_terms(node.term))
    return ctx.index.df(node.term)


def _filter_size(node: FilterNode, ctx: SearchContext) -> int:
    """Exact, because a parametric selection is cheap enough to just run."""
    if ctx.params is None:
        return ctx.doc_count
    try:
        return int(ctx.params.select(node.field, node.op, node.value).size)
    except (KeyError, ValueError):
        return ctx.doc_count


# ==========================================================================
# 4. The plan — returned as data
# ==========================================================================


@dataclass
class PlanStep:
    """One node of the chosen plan, with the numbers that justify the order."""

    op: str                              # AND | OR | NOT | TERM | PHRASE | PROX | FILTER
    label: str                           # human-readable operand
    estimated: int = 0                   # df / estimated size, BEFORE running
    produced: int = -1                   # actual size after running
    running: int = -1                    # intermediate result after this step
    children: List["PlanStep"] = field(default_factory=list)
    note: str = ""

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def as_json(self) -> Dict[str, Any]:
        return {
            "op": self.op, "label": self.label, "estimated": self.estimated,
            "produced": self.produced, "running": self.running,
            "note": self.note,
            "children": [c.as_json() for c in self.children],
        }


@dataclass
class PlanStats:
    """Cost counters, so "cheaper" is a measurement and not a claim."""

    postings_entries_read: int = 0       # sum of df of every list we touched
    #: The real cost of the Boolean work: entries actually compared.  A linear
    #: merge of two sorted lists compares about len(a)+len(b); a skip-probed
    #: intersection compares about one probe per candidate.  This is the counter
    #: that moves when the operand order changes -- `postings_entries_read` does
    #: not, because both orders open the same lists.
    merge_entries: int = 0
    skip_probes: int = 0
    merges: int = 0
    filters_applied: int = 0
    terms_dropped_by_idf: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def as_json(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ==========================================================================
# 5. Execution
# ==========================================================================


def execute(node: Node, ctx: SearchContext) -> np.ndarray:
    """THE frozen signature (CONTRACTS Section 4): sorted doc ids.

    A thin wrapper over :func:`execute_explained`, which also returns the plan.
    """
    ids, _plan, _stats = execute_explained(node, ctx)
    return ids


def execute_explained(node: Optional[Node], ctx: SearchContext
                      ) -> Tuple[np.ndarray, Optional[PlanStep], PlanStats]:
    """Run ``node`` and return ``(doc_ids, plan, stats)``."""
    stats = PlanStats()
    t0 = time.perf_counter()
    if node is None:
        ids, plan = ctx.all_docs, None
    else:
        ids, plan = _run(node, ctx, stats, candidates=None)
    stats.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return ids, plan, stats


def _run(node: Node, ctx: SearchContext, stats: PlanStats,
         candidates: Optional[np.ndarray]) -> Tuple[np.ndarray, PlanStep]:
    """Dispatch.  ``candidates``, when given, is a set the result must lie in."""
    if isinstance(node, FilterNode):
        return _run_filter(node, ctx, stats, candidates)
    if isinstance(node, TermNode):
        return _run_term(node, ctx, stats, candidates)
    if isinstance(node, PhraseNode):
        return _run_phrase(node, ctx, stats, candidates)
    if isinstance(node, ProxNode):
        return _run_prox(node, ctx, stats, candidates)
    if isinstance(node, AndNode):
        return _run_and(node, ctx, stats, candidates)
    if isinstance(node, OrNode):
        return _run_or(node, ctx, stats, candidates)
    if isinstance(node, NotNode):
        return _run_not(node, ctx, stats, candidates)
    raise TypeError(f"planner cannot execute {type(node).__name__}")


# -- leaves ----------------------------------------------------------------


def _run_filter(node: FilterNode, ctx: SearchContext, stats: PlanStats,
                candidates: Optional[np.ndarray]) -> Tuple[np.ndarray, PlanStep]:
    step = PlanStep("FILTER", node.describe())
    if ctx.params is None:
        step.note = "no parametric index loaded; filter ignored"
        out = candidates if candidates is not None else ctx.all_docs
    else:
        try:
            out = ctx.params.select(node.field, node.op, node.value)
            stats.filters_applied += 1
        except (KeyError, ValueError) as exc:
            step.note = f"unusable filter ({exc.__class__.__name__}); ignored"
            out = candidates if candidates is not None else ctx.all_docs
    step.estimated = step.produced = int(out.size)
    if candidates is not None:
        before = int(out.size)
        out = intersect(out, candidates)
        stats.merges += 1
        stats.merge_entries += before + int(candidates.size)
    step.running = int(out.size)
    return out, step


def _run_term(node: TermNode, ctx: SearchContext, stats: PlanStats,
              candidates: Optional[np.ndarray]) -> Tuple[np.ndarray, PlanStep]:
    if node.prefix:
        return _run_prefix(node, ctx, stats, candidates)
    step = PlanStep("TERM", node.describe(), estimated=_term_df(node, ctx))
    posts = ctx.index.postings(node.term) if ctx.index else None
    if posts is None:
        step.note = "not in the dictionary"
        step.produced = step.running = 0
        return EMPTY, step
    stats.postings_entries_read += int(posts.df)
    doc_ids = np.asarray(posts.doc_ids, dtype=np.int32)
    step.produced = int(doc_ids.size)
    if candidates is not None:
        # Short-against-long: probe the long list through its skips instead of
        # merging it end to end.
        if doc_ids.size >= SKIP_RATIO * max(candidates.size, 1):
            probes: Dict[str, int] = {}
            out = intersect_with_skips(candidates, posts, probes)
            stats.skip_probes += probes.get("skip_probes", 0)
            step.note = f"skip-probed {candidates.size} into df {posts.df}"
        else:
            out = intersect(candidates, doc_ids)
            stats.merges += 1
            stats.merge_entries += int(candidates.size) + int(doc_ids.size)
    else:
        out = doc_ids
    if node.zone:
        out, zone_note = _restrict_to_zone(out, node, ctx)
        step.note = (step.note + "; " if step.note else "") + zone_note
    step.running = int(out.size)
    return out, step


def _run_prefix(node: TermNode, ctx: SearchContext, stats: PlanStats,
                candidates: Optional[np.ndarray]) -> Tuple[np.ndarray, PlanStep]:
    """``bail!`` — a dictionary prefix scan, executed as an OR over the matches."""
    terms = ctx.index.prefix_terms(node.term) if ctx.index else []
    step = PlanStep("PREFIX", node.describe(),
                    estimated=sum(ctx.index.df(t) for t in terms) if terms else 0)
    if not terms:
        step.note = "no dictionary term has this prefix"
        step.produced = step.running = 0
        return EMPTY, step
    step.note = f"{len(terms)} dictionary terms"
    parts = []
    for t in sorted(terms, key=ctx.index.df):        # rarest first, as always
        p = ctx.index.postings(t)
        if p is None:
            continue
        stats.postings_entries_read += int(p.df)
        parts.append(np.asarray(p.doc_ids, dtype=np.int32))
        step.children.append(PlanStep("TERM", f"TERM({t})", estimated=int(p.df),
                                      produced=int(p.df)))
    out = union(parts)
    step.produced = int(out.size)
    if candidates is not None:
        out = intersect(out, candidates)
        stats.merges += 1
    step.running = int(out.size)
    return out, step


def _restrict_to_zone(doc_ids: np.ndarray, node: TermNode,
                      ctx: SearchContext) -> Tuple[np.ndarray, str]:
    """``title:(privacy)`` — keep only documents with the term in that zone."""
    zone = node.zone
    if zone not in config.ZONE_ID or ctx.index is None:
        return doc_ids, f"unknown zone {zone!r}; ignored"
    posts = ctx.index.postings(node.term)
    if posts is None:
        return EMPTY, f"zone {zone}"
    keep = [d for d in doc_ids.tolist()
            if posts.zone_tf_of(int(d)).get(zone, 0) > 0]
    return np.asarray(keep, dtype=np.int32), f"zone {zone}"


# -- Member 1's operators --------------------------------------------------


def _proximity_module():
    """Import M1's proximity module lazily; it may not have landed yet."""
    try:
        from nyaya.query import proximity
        return proximity
    except Exception:
        return None


def _run_phrase(node: PhraseNode, ctx: SearchContext, stats: PlanStats,
                candidates: Optional[np.ndarray]) -> Tuple[np.ndarray, PlanStep]:
    step = PlanStep("PHRASE", node.describe(), estimated=estimate_size(node, ctx))
    prox = _proximity_module()
    for t in node.terms:
        p = ctx.index.postings(t) if ctx.index else None
        if p is not None:
            stats.postings_entries_read += int(p.df)
        step.children.append(PlanStep("TERM", f"TERM({t})",
                                      estimated=ctx.index.df(t) if ctx.index else 0))
    if prox is not None and hasattr(prox, "phrase_docs"):
        out = np.asarray(prox.phrase_docs(node.terms, ctx, zone=node.zone),
                         dtype=np.int32)
    else:
        # Degrade to a conjunction, and SAY SO: a phrase is strictly stronger
        # than an AND, so silently substituting one would overstate recall
        # precision in the evaluation.
        step.note = "proximity module unavailable - degraded to AND over the terms"
        out, _ = _run_and(AndNode([TermNode(term=t, zone=node.zone)
                                   for t in node.terms]), ctx, stats, candidates)
        step.produced = step.running = int(out.size)
        return out, step
    step.produced = int(out.size)
    if candidates is not None:
        out = intersect(out, candidates)
        stats.merges += 1
    step.running = int(out.size)
    return out, step


def _run_prox(node: ProxNode, ctx: SearchContext, stats: PlanStats,
              candidates: Optional[np.ndarray]) -> Tuple[np.ndarray, PlanStep]:
    step = PlanStep("PROX", node.describe(), estimated=estimate_size(node, ctx))
    # Both sides have to be evaluated, so evaluate the cheaper one first and let
    # it constrain the other.
    left, right = node.left, node.right
    if estimate_size(right, ctx) < estimate_size(left, ctx):
        left, right = right, left
        step.note = "operands swapped: cheaper side first"
    left_ids, lstep = _run(left, ctx, stats, candidates)
    right_ids, rstep = _run(right, ctx, stats, left_ids if left_ids.size else None)
    step.children = [lstep, rstep]

    prox = _proximity_module()
    if prox is None or not hasattr(prox, "proximity_docs"):
        step.note = ((step.note + "; ") if step.note else "") + \
            "proximity module unavailable - degraded to AND (window not enforced)"
        out = intersect(left_ids, right_ids)
        stats.merges += 1
    else:
        out = np.asarray(
            prox.proximity_docs(left_ids, right_ids,
                                _leaf_terms(left), _leaf_terms(right),
                                node.kind, node.k, ctx),
            dtype=np.int32)
    step.produced = step.running = int(out.size)
    return out, step


def _leaf_terms(node: Node) -> List[str]:
    if isinstance(node, TermNode):
        return [node.term]
    if isinstance(node, PhraseNode):
        return list(node.terms)
    out: List[str] = []
    for c in node.children():
        out.extend(_leaf_terms(c))
    return out


# -- composites: where the optimisation lives ------------------------------


def _run_and(node: AndNode, ctx: SearchContext, stats: PlanStats,
             candidates: Optional[np.ndarray],
             policy: str = None) -> Tuple[np.ndarray, PlanStep]:
    """Intersect operands rarest-first, filters before terms.

    The ordering is the whole point.  ``sort`` key puts every FilterNode ahead of
    every text operand (filters are cheap *and* selective), then orders by
    estimated size ascending.  Each step narrows ``running``, and the next
    operand is intersected against that smaller set — so the chain gets cheaper
    as it goes, instead of repeatedly walking the longest list.
    """
    operands = flatten_and(node)
    step = PlanStep("AND", f"AND over {len(operands)} operands")
    ordered = _order_operands(operands, ctx, policy)
    step.note = "order: " + " -> ".join(_short(o, ctx) for o in ordered)

    running = candidates
    for operand in ordered:
        ids, child = _run(operand, ctx, stats, running)
        step.children.append(child)
        running = ids
        if running.size == 0:
            child.note = (child.note + "; " if child.note else "") + \
                "empty - remaining operands skipped"
            break
    out = running if running is not None else ctx.all_docs
    step.estimated = min((estimate_size(o, ctx) for o in operands), default=0)
    step.produced = step.running = int(out.size)
    return out, step


#: How an AND node orders its operands.
#:
#: ``"size"`` (the default) is the textbook rule: ascending estimated document
#: frequency, whatever kind of operand it is.
#:
#: ``"filters-first"`` puts every parametric filter ahead of every text operand.
#: It is the obvious heuristic — a filter costs no postings I/O at all, and
#: ``act:ndps`` or ``bench:>=5`` is usually far more selective than any word — but
#: "usually" is doing a lot of work there, and when it is wrong it is wrong by a
#: lot.  On ``privacy AND act:const AND bench:>=5`` the filters are 1,176 and
#: 6,788 documents while *privacy* is 126, so forcing the filters first starts the
#: chain with the two largest operands and does ~7x more comparison work for an
#: identical answer.
#:
#: So the default is the pure rule.  Filters still land first in the common case,
#: because they usually *are* the most selective — the size rule subsumes the
#: heuristic instead of hard-coding it.  :func:`compare_orderings` measures both.
ORDER_POLICY = "size"


def _order_operands(operands: Sequence[Node], ctx: SearchContext,
                    policy: str = None) -> List[Node]:
    """Order AND operands cheapest-first under ``policy``."""
    policy = policy or ORDER_POLICY
    if policy == "filters-first":
        return sorted(operands,
                      key=lambda o: (0 if isinstance(o, FilterNode) else 1,
                                     estimate_size(o, ctx)))
    if policy == "as-written":
        return list(operands)
    # "size": ascending estimated size; a filter wins an exact tie because it
    # needs no postings I/O to produce.
    return sorted(operands,
                  key=lambda o: (estimate_size(o, ctx),
                                 0 if isinstance(o, FilterNode) else 1))


def _run_or(node: OrNode, ctx: SearchContext, stats: PlanStats,
            candidates: Optional[np.ndarray]) -> Tuple[np.ndarray, PlanStep]:
    """Union every operand.  Ordered rarest-first too, for a readable plan.

    A union has to touch all of its children, so ordering cannot make it cheaper
    — but the estimate is the *sum* of the children, not the min, and printing the
    children in the same ascending order as everywhere else keeps the plan easy
    to read against an AND plan.
    """
    step = PlanStep("OR", f"OR over {len(node.operands)} operands")
    ordered = _order_operands(node.operands, ctx)
    step.note = "order: " + " -> ".join(_short(o, ctx) for o in ordered)
    parts: List[np.ndarray] = []
    for operand in ordered:
        # Note: candidates are NOT pushed into an OR branch — a document may
        # satisfy the union through a branch that the candidate set excludes.
        ids, child = _run(operand, ctx, stats, None)
        step.children.append(child)
        parts.append(ids)
    out = union(parts)
    step.estimated = sum(estimate_size(o, ctx) for o in node.operands)
    step.produced = int(out.size)
    if candidates is not None:
        out = intersect(out, candidates)
        stats.merges += 1
    step.running = int(out.size)
    return out, step


def _run_not(node: NotNode, ctx: SearchContext, stats: PlanStats,
             candidates: Optional[np.ndarray]) -> Tuple[np.ndarray, PlanStep]:
    """``a NOT b`` — set difference, never a bare negation (contracts).

    ``b`` is computed *after* ``a`` and constrained to it: there is no point
    discovering every document containing ``b`` when only the ones that also
    matched ``a`` can possibly be removed.
    """
    step = PlanStep("NOT", "NOT (left minus right)")
    left_ids, lstep = _run(node.left, ctx, stats, candidates)
    right_ids, rstep = _run(node.right, ctx, stats,
                            left_ids if left_ids.size else None)
    rstep.note = (rstep.note + "; " if rstep.note else "") + \
        "evaluated only against the left result"
    step.children = [lstep, rstep]
    out = difference(left_ids, right_ids)
    step.estimated = estimate_size(node.left, ctx)
    step.produced = step.running = int(out.size)
    return out, step


def _short(node: Node, ctx: SearchContext) -> str:
    return f"{_label(node)}[{estimate_size(node, ctx)}]"


def _label(node: Node) -> str:
    if isinstance(node, (TermNode, PhraseNode, ProxNode, FilterNode)):
        return node.describe()
    return type(node).__name__.replace("Node", "").upper()


# ==========================================================================
# 6. Whole-query planning, including the filters that ride alongside the tree
# ==========================================================================


def plan_query(q: ParsedQuery, ctx: SearchContext
               ) -> Tuple[np.ndarray, Optional[PlanStep], PlanStats]:
    """Execute a whole :class:`~nyaya.contracts.ParsedQuery`.

    ``ParsedQuery`` keeps ``filters`` beside the tree rather than inside it, so a
    free-text query can still be filtered.  We fold them into one AND node with
    the tree, which is what puts them through the same rarest-first optimiser —
    and, because filters sort first, a query like ``privacy act:const
    bench:>=5`` narrows to ~1,000 documents before the postings for *privacy*
    are opened at all.
    """
    operands: List[Node] = list(q.filters)
    if q.root is not None:
        operands.append(q.root)
    elif q.free_terms:
        # Free text: the candidate set is the OR of the terms.  Scoring, not the
        # planner, decides what is actually good — a bag of words is not a
        # conjunction, and treating it as one would silently destroy recall.
        operands.append(OrNode([TermNode(term=t) for t in q.free_terms]))

    if not operands:
        stats = PlanStats()
        return ctx.all_docs, None, stats
    root: Node = operands[0] if len(operands) == 1 else AndNode(operands)
    return execute_explained(root, ctx)


# ==========================================================================
# 7. explain_plan — the `postings-plan` video shot
# ==========================================================================

_BOX = {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│"}


def explain_plan(plan: Optional[PlanStep], stats: Optional[PlanStats] = None,
                 title: str = "query plan", width: int = 76) -> str:
    """Render the chosen plan with the numbers that justify it.

    Deliberately a *string*, built from the :class:`PlanStep` tree, so the same
    data can go to a terminal, the Streamlit page or the report.
    """
    lines: List[str] = []
    bar = _BOX["h"] * (width - 2)
    lines.append(_BOX["tl"] + bar + _BOX["tr"])
    lines.append(_BOX["v"] + f" {title}".ljust(width - 2) + _BOX["v"])
    lines.append(_BOX["v"] + " " + "─" * (width - 4) + " " + _BOX["v"])
    if plan is None:
        lines.append(_BOX["v"] + "  (no plan: every indexed document matches)"
                     .ljust(width - 2) + _BOX["v"])
    else:
        head = f"  {'step':<34}{'est df':>9}{'produced':>10}{'running':>9}"
        lines.append(_BOX["v"] + head.ljust(width - 2) + _BOX["v"])
        for row in _render(plan, 0):
            lines.append(_BOX["v"] + row[:width - 2].ljust(width - 2) + _BOX["v"])
    if stats is not None:
        lines.append(_BOX["v"] + " " + "─" * (width - 4) + " " + _BOX["v"])
        for row in _render_stats(stats):
            lines.append(_BOX["v"] + row[:width - 2].ljust(width - 2) + _BOX["v"])
    lines.append(_BOX["bl"] + bar + _BOX["br"])
    return "\n".join(lines)


#: A prefix scan can match 50 dictionary terms; printing all of them buries the
#: plan.  The ones that matter are the first few (rarest first), so the rest are
#: summarised.
MAX_RENDERED_CHILDREN = 6


def _render(step: PlanStep, depth: int) -> List[str]:
    pad = "  " + ("  " * depth)
    name = f"{pad}{step.label}"
    est = f"{step.estimated:,}" if step.estimated >= 0 else "-"
    prod = f"{step.produced:,}" if step.produced >= 0 else "-"
    run = f"{step.running:,}" if step.running >= 0 else "-"
    out = [f"{name:<36}{est:>9}{prod:>10}{run:>9}"[:200]]
    if step.note:
        out.append(f"{pad}  . {step.note}")
    shown = step.children[:MAX_RENDERED_CHILDREN]
    for c in shown:
        out.extend(_render(c, depth + 1))
    hidden = len(step.children) - len(shown)
    if hidden > 0:
        total = sum(c.produced for c in step.children[MAX_RENDERED_CHILDREN:]
                    if c.produced > 0)
        out.append(f"{pad}  ... and {hidden} more ({total:,} further postings)")
    return out


def _render_stats(stats: PlanStats) -> List[str]:
    rows = [
        f"  postings entries read : {stats.postings_entries_read:,}",
        f"  entries compared      : {stats.merge_entries:,}  (the real Boolean cost)",
        f"  skip-pointer probes   : {stats.skip_probes:,}",
        f"  sorted-array merges   : {stats.merges:,}",
        f"  parametric filters    : {stats.filters_applied:,}",
        f"  elapsed               : {stats.elapsed_ms:.2f} ms",
    ]
    if stats.terms_dropped_by_idf:
        rows.append(f"  dropped (low idf)     : "
                    f"{', '.join(stats.terms_dropped_by_idf)}")
    return rows


# ==========================================================================
# 8. The ordering experiment the report quotes
# ==========================================================================


def flatten_and(node: Node) -> List[Node]:
    """Collapse nested AND nodes into one operand list.

    ``a AND b AND c`` parses as ``AND(AND(a, b), c)`` or ``AND(a, b, c)``
    depending on how the parser associates, and the optimiser can only order
    operands it can see.  Flattening first is what makes a three-term chain one
    three-way decision instead of two nested two-way ones.
    """
    if isinstance(node, AndNode):
        out: List[Node] = []
        for c in node.operands:
            out.extend(flatten_and(c))
        return out
    return [node]


def _cost_of_order(operands: Sequence[Node], ctx: SearchContext,
                   policy: str) -> Tuple[np.ndarray, PlanStats]:
    """Run a flattened AND chain under one ordering policy and report the cost."""
    stats = PlanStats()
    t0 = time.perf_counter()
    running: Optional[np.ndarray] = None
    for operand in _order_operands(operands, ctx, policy):
        ids, _child = _run(operand, ctx, stats, running)
        running = ids
        if running.size == 0:
            break
    stats.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return (running if running is not None else ctx.all_docs), stats


def compare_orderings(node: Node, ctx: SearchContext) -> Dict[str, Any]:
    """Run an AND chain under every ordering policy and compare the cost.

    Same answer, different work — that is the claim, and this returns the numbers
    that back it instead of asserting them.  All three run over the same
    *flattened* operand list, so the only difference is the order.
    """
    operands = flatten_and(node)
    out: Dict[str, Any] = {
        "operands": [_label(o) for o in operands],
        "policies": {},
    }
    answers: List[np.ndarray] = []
    for policy in ("size", "filters-first", "as-written"):
        ids, stats = _cost_of_order(operands, ctx, policy)
        answers.append(np.sort(ids))
        out["policies"][policy] = {
            "order": [_short(o, ctx) for o in _order_operands(operands, ctx, policy)],
            "compared": stats.merge_entries + stats.skip_probes,
            "postings_read": stats.postings_entries_read,
            "elapsed_ms": round(stats.elapsed_ms, 3),
            "results": int(ids.size),
        }
    out["same_answer"] = all(np.array_equal(answers[0], a) for a in answers[1:])
    out["results"] = int(answers[0].size)
    best = out["policies"][ORDER_POLICY]["compared"]
    worst = out["policies"]["as-written"]["compared"]
    out["speedup_vs_as_written"] = worst / max(best, 1)
    out["speedup_vs_filters_first"] = (
        out["policies"]["filters-first"]["compared"] / max(best, 1))
    return out


#: The AND chains the df-ordering experiment is reported over.  Eight real
#: multi-term conjunctions across different areas of law, with chain lengths
#: from three to four, so the aggregate is not one lucky query.  Frozen here
#: rather than typed on the command line so the reported figure is reproducible:
#: ``python -m nyaya.query.planner --compare-suite`` regenerates it.
AND_CHAIN_SUITE = (
    "court AND intention AND murder",
    "state AND land AND acquisition AND compensation",
    "evidence AND dying AND declaration",
    "constitution AND article AND reservation AND promotion",
    "appeal AND sentence AND death AND confirmation",
    "contract AND performance AND specific AND relief",
    "bail AND anticipatory AND discretion",
    "tax AND sale AND interstate AND trade",
)

#: The ordering policies ``compare_orderings`` reports, in table order.
ORDER_POLICIES = ("size", "filters-first", "as-written")


def _query_operands(q) -> List[Node]:
    """The flat AND operand list a parsed query contributes to the experiment."""
    operands = list(q.filters)
    if q.root is not None:
        operands.append(q.root)
    elif q.free_terms:
        operands.extend(TermNode(term=t) for t in q.free_terms)
    return operands


def compare_suite(queries: Sequence[str] = AND_CHAIN_SUITE,
                  ctx: SearchContext = None) -> Dict[str, Any]:
    """Run :func:`compare_orderings` over a whole query set and total it up.

    One query proves the ordering rule can help; a suite is what the report can
    quote.  The aggregate ratio is computed on the *totals* (entries compared
    under ``as-written`` over entries compared under the default), not as a mean
    of per-query ratios, so one 327x outlier cannot carry the number.
    """
    from nyaya.query.parser import parse

    if ctx is None:
        ctx = _load_context()
    per_query: List[Dict[str, Any]] = []
    totals = {p: 0 for p in ORDER_POLICIES}
    for raw in queries:
        cmp = compare_orderings(AndNode(_query_operands(parse(raw))), ctx)
        for p in ORDER_POLICIES:
            totals[p] += cmp["policies"][p]["compared"]
        per_query.append({
            "query": raw,
            "results": cmp["results"],
            "same_answer": cmp["same_answer"],
            "compared": {p: cmp["policies"][p]["compared"] for p in ORDER_POLICIES},
            "ratio_vs_as_written": cmp["speedup_vs_as_written"],
        })
    ratios = [r["ratio_vs_as_written"] for r in per_query]
    cheapest = max(per_query, key=lambda r: r["ratio_vs_as_written"])
    least = min(per_query, key=lambda r: r["ratio_vs_as_written"])
    return {
        "policy": ORDER_POLICY,
        "n_queries": len(per_query),
        "totals": totals,
        "aggregate_ratio_vs_as_written": totals["as-written"] / max(totals[ORDER_POLICY], 1),
        "aggregate_ratio_vs_filters_first": (
            totals["filters-first"] / max(totals[ORDER_POLICY], 1)),
        "same_answer_everywhere": all(r["same_answer"] for r in per_query),
        "best_ratio": max(ratios), "best_query": cheapest["query"],
        "worst_ratio": min(ratios), "worst_query": least["query"],
        "per_query": per_query,
    }


def print_suite(s: Dict[str, Any]) -> None:
    print(f"df-ordering experiment over {s['n_queries']} AND chains "
          f"(default policy: {s['policy']})")
    print(f"  {'query':<56}{'size':>9}{'filters':>9}{'written':>9}{'ratio':>9}")
    for r in s["per_query"]:
        c = r["compared"]
        print(f"  {r['query']:<56}{c['size']:>9,}{c['filters-first']:>9,}"
              f"{c['as-written']:>9,}{r['ratio_vs_as_written']:>8.2f}x")
    t = s["totals"]
    print(f"  {'TOTAL':<56}{t['size']:>9,}{t['filters-first']:>9,}"
          f"{t['as-written']:>9,}"
          f"{s['aggregate_ratio_vs_as_written']:>8.2f}x")
    print(f"  identical answers under all three policies: "
          f"{s['same_answer_everywhere']}")
    print(f"  aggregate: {t['as-written']:,} -> {t['size']:,} entries compared, "
          f"{s['aggregate_ratio_vs_as_written']:.2f}x less work "
          f"({s['aggregate_ratio_vs_filters_first']:.2f}x vs filters-first)")
    print(f"  best  {s['best_ratio']:>8.2f}x  {s['best_query']}")
    print(f"  worst {s['worst_ratio']:>8.2f}x  {s['worst_query']}")


# ==========================================================================
# 9. CLI
# ==========================================================================


def _load_context() -> SearchContext:
    from nyaya.index.parametric import ParametricIndex
    from nyaya.index.positional import PositionalIndex

    index = None
    try:
        index = PositionalIndex.load()
    except SystemExit as exc:
        print(f"note: no positional index yet ({exc}); "
              "filters still work", file=sys.stderr)
    params = None
    try:
        params = ParametricIndex.load()
    except FileNotFoundError:
        print("note: no parametric index yet", file=sys.stderr)
    return SearchContext(index=index, params=params)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(
        prog="python -m nyaya.query.planner",
        description="Execute a Boolean query and show the chosen plan.")
    ap.add_argument("query", nargs="*")
    ap.add_argument("--compare", action="store_true",
                    help="also run the AND chain in the order written")
    ap.add_argument("--compare-suite", action="store_true",
                    help="run --compare over AND_CHAIN_SUITE and total it up")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args(argv)

    from nyaya.query.parser import parse

    if args.compare_suite:
        print_suite(compare_suite(ctx=_load_context()))
        return 0
    if not args.query:
        ap.error("give a query, or --compare-suite")

    q = parse(" ".join(args.query))
    ctx = _load_context()
    ids, plan, stats = plan_query(q, ctx)
    print(explain_plan(plan, stats, title=f"plan for: {q.raw}"))
    print(f"\n{ids.size:,} documents matched; first {args.show}: "
          f"{ids[:args.show].tolist()}")
    for w in getattr(q, "warnings", []):
        print(f"  warning: {w}")

    if args.compare:
        cmp = compare_orderings(AndNode(_query_operands(q)), ctx)
        print()
        print(f"ordering experiment  (same answer: {cmp['same_answer']}, "
              f"{cmp['results']:,} results)")
        print(f"  {'policy':<16}{'compared':>12}{'postings':>11}{'ms':>8}")
        for name, v in cmp["policies"].items():
            mark = " <- default" if name == ORDER_POLICY else ""
            print(f"  {name:<16}{v['compared']:>12,}{v['postings_read']:>11,}"
                  f"{v['elapsed_ms']:>8.2f}{mark}")
        for name, v in cmp["policies"].items():
            print(f"  {name:<16}{' -> '.join(v['order'])}")
        print(f"  default is {cmp['speedup_vs_as_written']:.2f}x cheaper than "
              f"as-written, {cmp['speedup_vs_filters_first']:.2f}x than "
              f"filters-first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
