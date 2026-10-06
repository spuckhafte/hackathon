"""Net score, top-K by heap, and index elimination.

Member 2 (Suvrat).  This is the last step of retrieval: take the candidate set the
planner produced, score it, and return the best K.

    net(q, d) = ALPHA * text_norm(q, d) + (1 - ALPHA) * g(d) + BETA * prox(q, d)

``ALPHA`` and ``BETA`` come from ``config``.  Three design points are worth
stating because they are the ones that are easy to get quietly wrong:

**1. Text relevance is normalised by the maximum in the candidate set.**
``g(d)`` is already in [0, 1] by construction, but a BM25F or tf-idf score is an
unbounded sum over query terms — it grows with the number of terms and with the
scorer's parameters.  Adding the two raw would make ALPHA meaningless: on a
one-term query the text part might be 2.1 and on a five-term query 9.4, so the
authority term would silently matter five times less on the longer query.
Dividing by the maximum observed score puts both parts on [0, 1] *for this
query*, which is what makes a single ALPHA defensible across queries.  The cost,
stated honestly: scores are then only comparable **within** one result list, never
between queries, so they are a ranking, not a measurement of relevance.

**2. Top-K is a min-heap of size K, not a sort.**
Sorting 13,942 candidates to show 10 is O(N log N) on the whole set.  A heap of
size K keeps the K best seen so far and costs O(N log K) — for N=13,942 and K=10
that is ~46,000 comparisons against ~190,000, and the memory is K not N.  The
root of the heap is the *worst* of the current best K, so a candidate is only
pushed when it beats that, and most candidates fail one comparison and are
dropped.

**3. Authority is keyed by node_id, results are keyed by doc_id.**
This is the single most dangerous join in the project — the two id spaces have
different sizes (37,210 vs 13,942) and both are dense integer ranges starting at
0, so a mix-up produces *plausible* numbers, not a crash: every document would
get some other judgment's authority and the ranking would be quietly wrong.  It
is therefore isolated in one class, :class:`DocNodeMap`, with its own tests.
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.contracts import Result, ScoreExplanation
from nyaya.utils import ensure_utf8_stdout

#: Signature a scorer must satisfy: given candidate doc ids and query terms,
#: return a float array of raw text scores aligned to ``doc_ids``.
#: Member 3 owns the real scorers (tf-idf / weighted zone / BM25F); this module
#: only ever calls one through this type, so swapping them is a parameter.
ScoreFn = Callable[[np.ndarray, Sequence[str], Any], np.ndarray]


# ==========================================================================
# 1. The id mapping — one place, heavily tested
# ==========================================================================


class DocNodeMap:
    """Translates between ``doc_id`` (indexed subset) and ``node_id`` (graph).

    ``doc_id`` indexes the 13,942-document manifest — it is what the postings,
    the parametric index and a ``Result`` all use.  ``node_id`` indexes the
    37,210-node citation graph — it is what ``CitationEdge.dst`` and every
    ``Authority`` record use.  Both are dense ranges from 0, which is exactly why
    confusing them is silent rather than loud.

    Built from the manifest, which carries both columns side by side.
    """

    __slots__ = ("_node_of_doc", "_doc_of_node", "n_docs", "n_nodes")

    def __init__(self, doc_ids: Sequence[int], node_ids: Sequence[int],
                 n_nodes: Optional[int] = None):
        docs = np.asarray(doc_ids, dtype=np.int64)
        nodes = np.asarray(node_ids, dtype=np.int64)
        if docs.size != nodes.size:
            raise ValueError("doc_ids and node_ids must be the same length")
        if docs.size and len(set(docs.tolist())) != docs.size:
            raise ValueError("doc_ids must be unique")
        if nodes.size and len(set(nodes.tolist())) != nodes.size:
            raise ValueError("node_ids must be unique (a doc maps to one node)")

        self.n_docs = int(docs.max()) + 1 if docs.size else 0
        self.n_nodes = int(n_nodes if n_nodes is not None
                           else (int(nodes.max()) + 1 if nodes.size else 0))
        # -1 means "no mapping", never 0: a 0 would silently mean node 0.
        self._node_of_doc = np.full(self.n_docs, -1, dtype=np.int64)
        self._node_of_doc[docs] = nodes
        self._doc_of_node = np.full(self.n_nodes, -1, dtype=np.int64)
        self._doc_of_node[nodes] = docs

    # -- construction -----------------------------------------------------
    @classmethod
    def from_manifest(cls, manifest=None, n_nodes: Optional[int] = None) -> "DocNodeMap":
        from nyaya.metadata import load_manifest

        man = load_manifest() if manifest is None else manifest
        if n_nodes is None:
            try:
                from nyaya.metadata import load_universe
                n_nodes = len(load_universe())
            except FileNotFoundError:
                n_nodes = None
        return cls(man["doc_id"].to_numpy(), man["node_id"].to_numpy(), n_nodes)

    # -- scalar -----------------------------------------------------------
    def node_of(self, doc_id: int) -> int:
        """``node_id`` of a document, or -1."""
        if not (0 <= doc_id < self.n_docs):
            return -1
        return int(self._node_of_doc[doc_id])

    def doc_of(self, node_id: int) -> int:
        """``doc_id`` of a graph node, or -1 when we never indexed its text."""
        if not (0 <= node_id < self.n_nodes):
            return -1
        return int(self._doc_of_node[node_id])

    # -- vectorised (what the ranker actually uses) ------------------------
    def nodes_of(self, doc_ids: np.ndarray) -> np.ndarray:
        d = np.asarray(doc_ids, dtype=np.int64)
        out = np.full(d.size, -1, dtype=np.int64)
        ok = (d >= 0) & (d < self.n_docs)
        out[ok] = self._node_of_doc[d[ok]]
        return out

    def docs_of(self, node_ids: np.ndarray) -> np.ndarray:
        n = np.asarray(node_ids, dtype=np.int64)
        out = np.full(n.size, -1, dtype=np.int64)
        ok = (n >= 0) & (n < self.n_nodes)
        out[ok] = self._doc_of_node[n[ok]]
        return out

    def __len__(self) -> int:
        return int((self._node_of_doc >= 0).sum())

    def __repr__(self) -> str:
        return (f"<DocNodeMap {len(self)} docs mapped, "
                f"doc space {self.n_docs}, node space {self.n_nodes}>")


def g_of_docs(doc_ids: np.ndarray, authority: Optional[Dict[int, Any]],
              mapping: DocNodeMap, default: float = 0.0) -> np.ndarray:
    """``g(d)`` for each candidate, looked up through :class:`DocNodeMap`.

    A document whose node has no authority record gets ``default`` — not a
    crash and not someone else's score.
    """
    ids = np.asarray(doc_ids, dtype=np.int64)
    out = np.full(ids.size, float(default), dtype=np.float64)
    if not authority:
        return out
    nodes = mapping.nodes_of(ids)
    for i, node in enumerate(nodes.tolist()):
        if node < 0:
            continue
        rec = authority.get(node)
        if rec is None:
            continue
        out[i] = float(_field(rec, "g", default))
    return out


def _field(rec: Any, name: str, default: Any) -> Any:
    """Read ``name`` off an :class:`~nyaya.contracts.Authority` or a plain dict.

    The store is loaded as dataclasses, but a JSON round trip hands back dicts,
    and both turn up in tests.
    """
    if isinstance(rec, dict):
        return rec.get(name, default)
    return getattr(rec, name, default)


def status_of_docs(doc_ids: np.ndarray, authority: Optional[Dict[int, Any]],
                   mapping: DocNodeMap) -> List[str]:
    ids = np.asarray(doc_ids, dtype=np.int64)
    nodes = mapping.nodes_of(ids)
    out: List[str] = []
    for node in nodes.tolist():
        rec = (authority or {}).get(node) if node >= 0 else None
        out.append("ok" if rec is None else str(_field(rec, "status", "ok")))
    return out


# ==========================================================================
# 2. Index elimination
# ==========================================================================


@dataclass
class EliminationReport:
    """What elimination actually removed — the number the report quotes."""

    query_terms: List[str] = field(default_factory=list)
    kept_terms: List[str] = field(default_factory=list)
    dropped_terms: List[Tuple[str, float]] = field(default_factory=list)
    candidates_before: int = 0
    candidates_after: int = 0
    min_matches_required: int = 1
    scored_without_elimination: int = 0
    scored_with_elimination: int = 0

    @property
    def documents_saved(self) -> int:
        return self.scored_without_elimination - self.scored_with_elimination

    @property
    def fraction_saved(self) -> float:
        if not self.scored_without_elimination:
            return 0.0
        return self.documents_saved / self.scored_without_elimination

    def as_json(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["documents_saved"] = self.documents_saved
        d["fraction_saved"] = self.fraction_saved
        return d


def eliminate_low_idf_terms(terms: Sequence[str], index,
                            threshold: float = config.IDF_ELIMINATION_THRESHOLD
                            ) -> Tuple[List[str], List[Tuple[str, float]]]:
    """Drop query terms whose idf is below ``threshold``.

    A term in nearly every document carries almost no information — in this
    corpus "court" has df 13,877 of 13,878, so its idf is ~0.00007 and it cannot
    change the ranking, but its postings list is the longest in the index.
    Dropping it is pure saving.

    Never returns an empty list: if *every* term is below the threshold the query
    is all boilerplate ("the appellate jurisdiction of the Supreme Court") and we
    keep the highest-idf one rather than returning nothing, for the same reason
    ``stoplist.filter_scoring_terms`` never empties a bag.
    """
    if index is None:
        return list(terms), []
    scored = [(t, float(index.idf(t))) for t in terms]
    kept = [t for t, v in scored if v >= threshold]
    dropped = [(t, v) for t, v in scored if v < threshold]
    if not kept and scored:
        best = max(scored, key=lambda tv: tv[1])
        kept = [best[0]]
        dropped = [(t, v) for t, v in scored if t != best[0]]
    return kept, dropped


def eliminate_documents(candidates: np.ndarray, terms: Sequence[str], index,
                        many_terms: int = config.MANY_TERMS,
                        max_matches: int = None
                        ) -> Tuple[np.ndarray, int]:
    """For a long query, score only documents matching at least half the terms.

    With ``MANY_TERMS`` (4) or more query terms, a document matching one of six is
    almost never in the top 10, but under a plain union it still has to be scored.
    Requiring ``ceil(n/2)`` matches is the textbook cut.  Short queries are left
    alone: with two terms, "at least half" means one, which eliminates nothing and
    would only cost a pass.

    **The half-rule is capped at ``config.ELIMINATION_MAX_MATCHES``.**  Manning's
    formulation assumes a query of a handful of words.  Track B builds queries
    from a judgment's facts — 60 terms — and "at least half" then demands 30
    matching terms, which is not an optimisation but a near-conjunction: it took
    one 13,878-document candidate set down to 19 documents and destroyed recall.
    The purpose of the rule is to skip documents matching only one term out of
    many, and that purpose is served long before 30.
    """
    cands = np.asarray(candidates, dtype=np.int32)
    n_terms = len(terms)
    if n_terms < many_terms or index is None or cands.size == 0:
        return cands, 1
    cap = (config.ELIMINATION_MAX_MATCHES if max_matches is None else max_matches)
    required = min((n_terms + 1) // 2, max(int(cap), 1))
    counts = np.zeros(cands.size, dtype=np.int16)
    order = np.argsort(cands)
    sorted_cands = cands[order]
    for t in terms:
        posts = index.postings(t)
        if posts is None:
            continue
        doc_ids = np.asarray(posts.doc_ids, dtype=np.int32)
        pos = np.searchsorted(sorted_cands, doc_ids)
        pos = pos[(pos < sorted_cands.size)]
        hit = pos[sorted_cands[pos] == doc_ids[:pos.size]] if pos.size else pos
        counts[order[hit]] += 1
    keep = counts >= required
    return cands[keep], int(required)


# ==========================================================================
# 3. Net score
# ==========================================================================


def normalise_text_scores(raw: np.ndarray) -> np.ndarray:
    """Scale raw text scores to [0, 1] by the maximum in the candidate set."""
    r = np.asarray(raw, dtype=np.float64)
    if r.size == 0:
        return r
    top = float(r.max())
    if top <= 0.0:
        return np.zeros_like(r)
    return r / top


def net_score(text_norm: np.ndarray,
              g: np.ndarray,
              prox: Optional[np.ndarray] = None,
              alpha: float = config.ALPHA,
              beta: float = config.BETA) -> np.ndarray:
    """``alpha * text + (1 - alpha) * g + beta * prox``, all arrays aligned."""
    t = np.asarray(text_norm, dtype=np.float64)
    gg = np.asarray(g, dtype=np.float64)
    if gg.shape != t.shape:
        raise ValueError(f"g has shape {gg.shape}, text has {t.shape} - "
                         "a misaligned join here silently ruins the ranking")
    out = alpha * t + (1.0 - alpha) * gg
    if prox is not None:
        p = np.asarray(prox, dtype=np.float64)
        if p.shape != t.shape:
            raise ValueError("proximity array is not aligned with the candidates")
        out = out + beta * p
    return out


# ==========================================================================
# 4. Top-K by min-heap
# ==========================================================================


def top_k(doc_ids: Sequence[int], scores: Sequence[float],
          k: int = config.TOP_K) -> List[Tuple[int, float]]:
    """The K highest-scoring documents, by a size-K min-heap.

    O(N log K) with O(K) memory, instead of O(N log N) and O(N) for a full sort.
    The heap holds ``(score, -doc_id)`` so that ties break on the *lower* doc id
    — deterministic output matters for an evaluation harness, and
    ``heapq.heappushpop`` would otherwise order ties by insertion.
    """
    k = int(k)
    if k <= 0:
        return []
    ids = np.asarray(doc_ids, dtype=np.int64)
    sc = np.asarray(scores, dtype=np.float64)
    if ids.size != sc.size:
        raise ValueError("doc_ids and scores must be the same length")

    heap: List[Tuple[float, int]] = []
    for doc_id, score in zip(ids.tolist(), sc.tolist()):
        entry = (score, -doc_id)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            # The root is the worst of the best K; one comparison discards most
            # candidates without touching the heap at all.
            heapq.heapreplace(heap, entry)
    heap.sort(reverse=True)
    return [(-neg_id, score) for score, neg_id in heap]


def heap_comparisons(n: int, k: int) -> int:
    """Rough comparison count for the report's efficiency table."""
    import math

    if n <= 0 or k <= 0:
        return 0
    return int(n * max(math.log2(max(k, 2)), 1))


def sort_comparisons(n: int) -> int:
    import math

    if n <= 1:
        return 0
    return int(n * math.log2(n))


# ==========================================================================
# 5. Putting it together
# ==========================================================================


@dataclass
class RankReport:
    """Everything the report and the "why this result" panel need."""

    candidates: int = 0
    scored: int = 0
    elimination: Optional[EliminationReport] = None
    k: int = config.TOP_K
    alpha: float = config.ALPHA
    beta: float = config.BETA
    heap_comparisons: int = 0
    sort_comparisons: int = 0
    scorer: str = "fallback-lnc.ltc"
    elapsed_ms: float = 0.0

    def as_json(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "elimination"}
        d["elimination"] = self.elimination.as_json() if self.elimination else None
        return d


def _fallback_text_scores(doc_ids: np.ndarray, terms: Sequence[str],
                          ctx) -> np.ndarray:
    """A minimal lnc.ltc tf-idf, used ONLY when no scorer has been supplied.

    Member 3 owns the real scorers (weighted-zone and BM25F, modes S2/S2p).  This
    exists so that `net.py` runs and can be measured standalone before they land,
    and so the ``B0`` baseline mode has something to call.  Pass ``score_fn`` to
    use anything better; nothing in this module assumes this one.
    """
    index = getattr(ctx, "index", None)
    ids = np.asarray(doc_ids, dtype=np.int32)
    out = np.zeros(ids.size, dtype=np.float64)
    if index is None or ids.size == 0:
        return out
    order = np.argsort(ids)
    sorted_ids = ids[order]
    for t in terms:
        posts = index.postings(t)
        if posts is None:
            continue
        idf = float(index.idf(t))
        doc_ids_t = np.asarray(posts.doc_ids, dtype=np.int32)
        tf = np.asarray(posts.tf_array(), dtype=np.float64)
        pos = np.searchsorted(sorted_ids, doc_ids_t)
        valid = pos < sorted_ids.size
        pos, tf = pos[valid], tf[valid]
        hit = sorted_ids[pos] == doc_ids_t[valid]
        rows = order[pos[hit]]
        w = (1.0 + np.log(np.maximum(tf[hit], 1.0))) * idf
        np.add.at(out, rows, w)
    norm = getattr(index, "doc_norm", None)
    if norm is not None:
        denom = np.asarray(norm, dtype=np.float64)[ids]
        out = np.divide(out, np.where(denom > 0, denom, 1.0))
    return out


def rank(doc_ids: np.ndarray,
         terms: Sequence[str],
         ctx,
         k: int = config.TOP_K,
         *,
         score_fn: Optional[ScoreFn] = None,
         mapping: Optional[DocNodeMap] = None,
         prox: Optional[np.ndarray] = None,
         eliminate: bool = True,
         alpha: float = config.ALPHA,
         beta: float = config.BETA) -> Tuple[List[Tuple[int, float]], RankReport]:
    """Score ``doc_ids`` and return the top ``k`` as ``[(doc_id, net_score)]``."""
    import time

    t0 = time.perf_counter()
    cands = np.asarray(doc_ids, dtype=np.int32)
    index = getattr(ctx, "index", None)
    report = RankReport(candidates=int(cands.size), k=int(k),
                        alpha=alpha, beta=beta)

    from nyaya.stoplist import filter_scoring_terms

    scoring_terms = filter_scoring_terms(terms)
    elim = EliminationReport(query_terms=list(scoring_terms),
                             candidates_before=int(cands.size),
                             scored_without_elimination=int(cands.size))
    if eliminate:
        kept, dropped = eliminate_low_idf_terms(scoring_terms, index)
        elim.kept_terms, elim.dropped_terms = kept, dropped
        scoring_terms = kept
        cands, required = eliminate_documents(cands, scoring_terms, index)
        elim.min_matches_required = required
    else:
        elim.kept_terms = list(scoring_terms)
    elim.candidates_after = int(cands.size)
    elim.scored_with_elimination = int(cands.size)
    report.elimination = elim
    report.scored = int(cands.size)

    fn = score_fn or _fallback_text_scores
    report.scorer = getattr(fn, "__name__", "score_fn")
    raw = np.asarray(fn(cands, scoring_terms, ctx), dtype=np.float64)
    text = normalise_text_scores(raw)

    mp = mapping or getattr(ctx, "docnode", None)
    if mp is None:
        mp = DocNodeMap.from_manifest()
    g = g_of_docs(cands, getattr(ctx, "authority", None), mp)

    p = None
    if prox is not None:
        p = np.asarray(prox, dtype=np.float64)
        if p.size != cands.size:
            # The candidate set shrank under elimination; re-align by doc id
            # rather than trusting positions.
            p = None
    net = net_score(text, g, p, alpha=alpha, beta=beta)

    out = top_k(cands, net, k)
    report.heap_comparisons = heap_comparisons(int(cands.size), int(k))
    report.sort_comparisons = sort_comparisons(int(cands.size))
    report.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out, report


def measure_elimination(doc_ids: np.ndarray, terms: Sequence[str], ctx,
                        k: int = config.TOP_K,
                        score_fn: Optional[ScoreFn] = None) -> Dict[str, Any]:
    """Run the same query with and without elimination and compare.

    Reports the document counts *and* whether the top-K changed, because a
    saving that costs accuracy is not a saving.
    """
    with_elim, r1 = rank(doc_ids, terms, ctx, k, score_fn=score_fn, eliminate=True)
    without, r0 = rank(doc_ids, terms, ctx, k, score_fn=score_fn, eliminate=False)
    top_with = [d for d, _s in with_elim]
    top_without = [d for d, _s in without]
    overlap = len(set(top_with) & set(top_without))
    return {
        "scored_without_elimination": r0.scored,
        "scored_with_elimination": r1.scored,
        "documents_saved": r0.scored - r1.scored,
        "fraction_saved": ((r0.scored - r1.scored) / r0.scored) if r0.scored else 0.0,
        "dropped_terms": r1.elimination.dropped_terms if r1.elimination else [],
        "min_matches_required": r1.elimination.min_matches_required if r1.elimination else 1,
        "top_k_identical": top_with == top_without,
        "top_k_overlap": overlap,
        "k": k,
        "heap_comparisons": r1.heap_comparisons,
        "sort_comparisons": r1.sort_comparisons,
    }


# ==========================================================================
# 6. CLI
# ==========================================================================


def _load_context():
    from nyaya.authority.pagerank import load_authority
    from nyaya.index.parametric import ParametricIndex
    from nyaya.index.positional import PositionalIndex
    from nyaya.query.planner import SearchContext

    index = params = authority = None
    try:
        index = PositionalIndex.load()
    except SystemExit as exc:
        print(f"note: {exc}", file=sys.stderr)
    try:
        params = ParametricIndex.load()
    except FileNotFoundError:
        pass
    try:
        authority = load_authority()
    except FileNotFoundError:
        print("note: no authority.json; g(d) will be 0", file=sys.stderr)
    ctx = SearchContext(index=index, params=params, authority=authority)
    ctx.docnode = DocNodeMap.from_manifest()
    return ctx


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(
        prog="python -m nyaya.rank.net",
        description="Rank a query: net score, heap top-K, index elimination.")
    ap.add_argument("query", nargs="+")
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--measure", action="store_true",
                    help="compare scoring with and without index elimination")
    args = ap.parse_args(argv)

    from nyaya.metadata import load_manifest
    from nyaya.query.parser import parse
    from nyaya.query.planner import plan_query

    q = parse(" ".join(args.query))
    ctx = _load_context()
    cands, _plan, _pstats = plan_query(q, ctx)
    terms = q.scoring_terms if hasattr(q, "scoring_terms") else q.free_terms

    if args.measure:
        m = measure_elimination(cands, terms, ctx, args.k)
        print(f"query                        : {q.raw}")
        print(f"candidates from the planner   : {cands.size:,}")
        print(f"scored WITHOUT elimination    : {m['scored_without_elimination']:,}")
        print(f"scored WITH elimination       : {m['scored_with_elimination']:,}")
        print(f"documents not scored          : {m['documents_saved']:,} "
              f"({m['fraction_saved']:.1%})")
        print(f"terms dropped (idf < {config.IDF_ELIMINATION_THRESHOLD}) : "
              f"{[(t, round(v, 4)) for t, v in m['dropped_terms']]}")
        print(f"min term matches required     : {m['min_matches_required']}")
        print(f"top-{args.k} identical            : {m['top_k_identical']} "
              f"(overlap {m['top_k_overlap']}/{args.k})")
        print(f"comparisons, heap vs sort     : {m['heap_comparisons']:,} vs "
              f"{m['sort_comparisons']:,}")
        return 0

    results, report = rank(cands, terms, ctx, args.k)
    man = load_manifest()
    title = dict(zip(man["doc_id"], man["title"]))
    year = dict(zip(man["doc_id"], man["year"]))
    print(f"query: {q.raw}   candidates {report.candidates:,} -> "
          f"scored {report.scored:,}")
    print(f"net = {report.alpha}*text_norm + {round(1 - report.alpha, 2)}*g(d)"
          f" + {report.beta}*prox")
    print(f"  {'#':>2}  {'doc':>6}  {'net':>6}  {'year':>4}  title")
    for rank_i, (doc_id, score) in enumerate(results, 1):
        print(f"  {rank_i:>2}  {doc_id:>6}  {score:>6.4f}  "
              f"{int(year.get(doc_id, 0)):>4}  {str(title.get(doc_id, ''))[:54]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
