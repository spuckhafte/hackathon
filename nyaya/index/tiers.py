"""Store S5 — champion lists and the two-tier index.

Member 2 (Suvrat).  A *safe* optimisation would preserve the exhaustive ranking
exactly.  This one does not, and that is the point: it is a **non-safe** speed-up
that trades a small, measured amount of fidelity for a large reduction in the
number of documents scored (Manning, IIR 7.1.2-7.1.3).

**Champion lists.**  For every term in the dictionary we precompute the ``r``
(= ``config.CHAMPION_LIST_SIZE``, 400) documents that could plausibly be the best
answers *for that term alone*, and store only those.  At query time tier 1 is the
union of the query terms' champion lists.  If that union already holds at least
``k`` documents we score it and stop; otherwise we fall through to tier 2, the
full postings, and lose nothing.

**What the champions are ordered by matters.**  The naive choice is term
frequency, or tf-idf.  We order by

    w(t, d) + g(d)

— the term's own weight *plus the document's static authority* — because that is
the same quantity the final ranking uses (``net.py``: ``ALPHA * text + (1 - ALPHA)
* g``).  A champion list ordered by text weight alone systematically throws away
the documents that would have won on authority, which is exactly the population
the full ranking promotes; the list would be optimised for a ranking we do not
ship.  This is the lecture's point about champion lists and static quality
scores, and it is why ``g(d)`` has to exist before S5 can be built.

The two numbers that justify the store, both measured by
``python -m nyaya.index.tiers evaluate``:

* **speed-up** — documents scored in tier 1 versus exhaustively;
* **fidelity** — how often tier 1's top 10 equals the exhaustive top 10.

CLI::

    python -m nyaya.index.tiers build [--r 400]
    python -m nyaya.index.tiers stats
    python -m nyaya.index.tiers evaluate [--k 10] [--queries N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from nyaya import config
from nyaya.utils import ensure_utf8_stdout, timed

CHAMPIONS_JSON = config.INDEX / "champions.json"
TIERS_NPZ = config.INDEX / "tiers.npz"

#: Terms this rare are already shorter than the champion list, so a champion
#: list for them would be a verbatim copy of the postings and pure waste.
MIN_DF_FOR_CHAMPIONS = 8


# ==========================================================================
# 1. Build
# ==========================================================================


def champion_order(postings, g: np.ndarray,
                   index=None, term: str = "") -> np.ndarray:
    """Document ids of one term, ordered by ``w(t, d) + g(d)`` descending.

    ``w(t, d)`` is the log-frequency weight ``1 + ln tf`` scaled by the term's
    idf, i.e. the same ``ltc``-style weight the scorers use, and ``g(d)`` is the
    static authority already in [0, 1].  The two are added, not multiplied: the
    final net score is a weighted *sum*, so the champion ordering has to be one
    too or the list is sorted by a different objective than the ranking.
    """
    doc_ids = np.asarray(postings.doc_ids, dtype=np.int32)
    if doc_ids.size == 0:
        return doc_ids
    tf = np.asarray(postings.tf_array(), dtype=np.float64)
    idf = float(index.idf(term)) if (index is not None and term) else 1.0
    w = (1.0 + np.log(np.maximum(tf, 1.0))) * idf
    # Normalise w into [0, 1] for this term so that adding g(d) is meaningful:
    # an unnormalised w would swamp g for a common term and be swamped by it for
    # a rare one, and the mix would vary per term for no principled reason.
    top = float(w.max())
    if top > 0:
        w = w / top
    gg = g[doc_ids] if g.size else np.zeros(doc_ids.size)
    score = config.ALPHA * w + (1.0 - config.ALPHA) * gg
    order = np.argsort(-score, kind="stable")
    return doc_ids[order]


@dataclass
class TierStats:
    r: int = config.CHAMPION_LIST_SIZE
    vocab: int = 0
    terms_with_champions: int = 0
    terms_too_rare: int = 0
    champion_entries: int = 0
    postings_entries: int = 0

    @property
    def compression(self) -> float:
        if not self.postings_entries:
            return 0.0
        return self.champion_entries / self.postings_entries

    def as_json(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["compression"] = self.compression
        return d


class TieredIndex:
    """Store S5.  ``champions(term)`` -> the term's tier-1 doc ids."""

    def __init__(self, champions: Dict[str, np.ndarray], r: int,
                 stats: Optional[TierStats] = None):
        self._champions = champions
        self.r = int(r)
        self.stats = stats or TierStats(r=r)

    # -- read side --------------------------------------------------------
    def champions(self, term: str) -> Optional[np.ndarray]:
        """Tier-1 doc ids for ``term``, or ``None`` if it has no champion list."""
        return self._champions.get(term)

    def has(self, term: str) -> bool:
        return term in self._champions

    def tier1(self, terms: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
        """Union of the champion lists of ``terms``.

        Returns ``(doc_ids, missing)`` where ``missing`` are the terms with no
        champion list — rare terms whose full postings are already short.  Those
        must be unioned in from tier 2 or the tier-1 candidate set would silently
        ignore the most discriminative words in the query.
        """
        parts: List[np.ndarray] = []
        missing: List[str] = []
        for t in terms:
            c = self._champions.get(t)
            if c is None:
                missing.append(t)
            else:
                parts.append(c)
        if not parts:
            return np.empty(0, dtype=np.int32), missing
        return np.unique(np.concatenate(parts)).astype(np.int32), missing

    # -- persistence ------------------------------------------------------
    def save(self, path: Path = None) -> Path:
        base = Path(path or config.INDEX)
        base.mkdir(parents=True, exist_ok=True)
        terms = sorted(self._champions)
        lengths = np.array([self._champions[t].size for t in terms],
                           dtype=np.int64)
        flat = (np.concatenate([self._champions[t] for t in terms])
                if terms else np.empty(0, dtype=np.int32))
        np.savez_compressed(base / TIERS_NPZ.name,
                            flat=flat.astype(np.int32),
                            offsets=np.concatenate([[0], np.cumsum(lengths)]))
        (base / CHAMPIONS_JSON.name).write_text(
            json.dumps({"r": self.r, "terms": terms,
                        "stats": self.stats.as_json()}, ensure_ascii=False),
            encoding="utf-8")
        return base / CHAMPIONS_JSON.name

    @classmethod
    def load(cls, path: Path = None) -> "TieredIndex":
        base = Path(path or config.INDEX)
        meta = base / CHAMPIONS_JSON.name
        if not meta.exists():
            raise FileNotFoundError(
                f"{meta} missing — run `python -m nyaya.index.tiers build`")
        blob = json.loads(meta.read_text(encoding="utf-8"))
        with np.load(base / TIERS_NPZ.name) as z:
            flat, offsets = z["flat"], z["offsets"]
        champions = {}
        for i, t in enumerate(blob["terms"]):
            champions[t] = flat[int(offsets[i]):int(offsets[i + 1])]
        stats = TierStats(**{k: v for k, v in blob.get("stats", {}).items()
                             if k in TierStats.__dataclass_fields__})
        return cls(champions, int(blob["r"]), stats)

    def __repr__(self) -> str:
        return (f"<TieredIndex r={self.r} terms={len(self._champions):,} "
                f"entries={self.stats.champion_entries:,}>")


def build(index=None, authority=None, mapping=None,
          r: int = config.CHAMPION_LIST_SIZE,
          min_df: int = MIN_DF_FOR_CHAMPIONS,
          verbose: bool = True) -> TieredIndex:
    """Build S5 from S3 (postings) and S4 (g(d))."""
    from nyaya.rank.net import DocNodeMap, g_of_docs

    if index is None:
        from nyaya.index.positional import PositionalIndex
        index = PositionalIndex.load()
    if authority is None:
        from nyaya.authority.pagerank import load_authority
        try:
            authority = load_authority()
        except FileNotFoundError:
            authority = {}
            if verbose:
                print("S5 WARNING: no authority.json — champion lists will be "
                      "ordered by text weight alone, which is NOT what the final "
                      "ranking uses.  Build S4 first.", file=sys.stderr)
    if mapping is None:
        mapping = DocNodeMap.from_manifest()

    # g(d) as a dense array over doc_id, looked up once rather than per term.
    n_slots = int(getattr(index, "n_doc_slots", index.doc_count))
    g = g_of_docs(np.arange(n_slots, dtype=np.int32), authority, mapping)

    terms = list(getattr(index, "_terms", []))
    stats = TierStats(r=r, vocab=len(terms))
    champions: Dict[str, np.ndarray] = {}
    for i, term in enumerate(terms):
        if verbose and i and i % 200_000 == 0:
            print(f"  ... {i:,}/{len(terms):,} terms", file=sys.stderr)
        df = index.df(term)
        stats.postings_entries += df
        if df < min_df:
            stats.terms_too_rare += 1
            continue
        posts = index.postings(term)
        if posts is None:
            continue
        ordered = champion_order(posts, g, index, term)
        champions[term] = ordered[:r].copy()
        stats.champion_entries += int(champions[term].size)
    stats.terms_with_champions = len(champions)

    if verbose:
        print(f"S5: {stats.terms_with_champions:,} champion lists "
              f"({stats.terms_too_rare:,} terms below df {min_df} skipped), "
              f"{stats.champion_entries:,} entries "
              f"= {stats.compression:.1%} of {stats.postings_entries:,} postings",
              file=sys.stderr)
    return TieredIndex(champions, r, stats)


# ==========================================================================
# 2. Query time: tier 1, then fall through
# ==========================================================================


@dataclass
class TierTrace:
    """Which tier answered, and why — for the explanation panel and the report."""

    tier: int = 1
    terms: List[str] = field(default_factory=list)
    terms_without_champions: List[str] = field(default_factory=list)
    tier1_candidates: int = 0
    tier2_candidates: int = 0
    scored: int = 0
    fell_through: bool = False
    reason: str = ""

    def as_json(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def tiered_candidates(terms: Sequence[str], tiers: TieredIndex, index,
                      k: int = config.TOP_K,
                      restrict: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, TierTrace]:
    """Tier-1 candidates, falling through to full postings when too few.

    ``restrict`` is the planner's candidate set (filters, Boolean structure).  It
    is applied *before* the fall-through test, because a query like ``privacy
    bench:>=5`` can have a healthy tier-1 union that collapses to two documents
    once the filter is applied — and in that case we do need tier 2.
    """
    trace = TierTrace(terms=list(terms))
    tier1, missing = tiers.tier1(terms)
    trace.terms_without_champions = missing

    # A term with no champion list contributes its full (short) postings.
    extra: List[np.ndarray] = []
    for t in missing:
        posts = index.postings(t) if index is not None else None
        if posts is not None:
            extra.append(np.asarray(posts.doc_ids, dtype=np.int32))
    if extra:
        tier1 = np.unique(np.concatenate([tier1] + extra)).astype(np.int32)

    if restrict is not None:
        tier1 = np.intersect1d(tier1, np.asarray(restrict, dtype=np.int32),
                               assume_unique=False).astype(np.int32)
    trace.tier1_candidates = int(tier1.size)

    if tier1.size >= k:
        trace.tier = 1
        trace.scored = int(tier1.size)
        trace.reason = f"tier 1 held {tier1.size} >= k={k} documents"
        return tier1, trace

    # --- fall through to tier 2 -----------------------------------------
    trace.fell_through = True
    trace.tier = 2
    trace.reason = (f"tier 1 held only {tier1.size} < k={k}; "
                    "fell through to the full postings")
    parts: List[np.ndarray] = []
    for t in terms:
        posts = index.postings(t) if index is not None else None
        if posts is not None:
            parts.append(np.asarray(posts.doc_ids, dtype=np.int32))
    tier2 = (np.unique(np.concatenate(parts)).astype(np.int32) if parts
             else np.empty(0, dtype=np.int32))
    if restrict is not None:
        tier2 = np.intersect1d(tier2, np.asarray(restrict, dtype=np.int32),
                               assume_unique=False).astype(np.int32)
    trace.tier2_candidates = int(tier2.size)
    trace.scored = int(tier2.size)
    return tier2, trace


# ==========================================================================
# 3. Evaluation: the speed-up and the fidelity cost
# ==========================================================================

#: Queries used when none are supplied.  Deliberately a mix: common doctrinal
#: phrases, rare statutes, and one all-boilerplate query, so the fidelity number
#: is not measured only on the easy cases.
DEFAULT_QUERIES = [
    "right to privacy",
    "preventive detention",
    "anticipatory bail",
    "basic structure of the constitution",
    "doctrine of res judicata",
    "unlawful activities prevention",
    "specific performance of contract",
    "dying declaration evidence",
    "compassionate appointment",
    "arbitration award set aside",
    "capital punishment rarest of rare",
    "quashing of first information report",
    "promissory estoppel",
    "sales tax inter state trade",
    "reservation in promotion",
    "custodial violence compensation",
    "environmental clearance precautionary principle",
    "cheque dishonour negotiable instruments",
    "adverse possession of land",
    "appellate jurisdiction of the supreme court",
]


def evaluate(queries: Optional[Sequence[str]] = None,
             k: int = config.TOP_K,
             verbose: bool = True) -> Dict[str, Any]:
    """Measure tier-1 speed-up and top-k fidelity against exhaustive scoring."""
    from nyaya.authority.pagerank import load_authority
    from nyaya.index.positional import PositionalIndex
    from nyaya.query.normalise import scoring_terms as norm_scoring_terms
    from nyaya.query.parser import parse
    from nyaya.query.planner import SearchContext
    from nyaya.rank.net import DocNodeMap, rank

    index = PositionalIndex.load()
    tiers = TieredIndex.load()
    try:
        authority = load_authority()
    except FileNotFoundError:
        authority = {}
    mapping = DocNodeMap.from_manifest()
    ctx = SearchContext(index=index, authority=authority)
    ctx.docnode = mapping

    rows: List[Dict[str, Any]] = []
    for raw in (queries or DEFAULT_QUERIES):
        q = parse(raw)
        terms = list(getattr(q, "scoring_terms", None) or q.free_terms)
        if not terms:
            continue

        # --- exhaustive: the union of the full postings -------------------
        parts = []
        for t in terms:
            p = index.postings(t)
            if p is not None:
                parts.append(np.asarray(p.doc_ids, dtype=np.int32))
        full = (np.unique(np.concatenate(parts)).astype(np.int32) if parts
                else np.empty(0, dtype=np.int32))
        t0 = time.perf_counter()
        exhaustive, _r0 = rank(full, terms, ctx, k, mapping=mapping,
                               eliminate=False)
        ms_full = (time.perf_counter() - t0) * 1000.0

        # --- tier 1 -------------------------------------------------------
        t0 = time.perf_counter()
        cands, trace = tiered_candidates(terms, tiers, index, k)
        tiered, _r1 = rank(cands, terms, ctx, k, mapping=mapping,
                           eliminate=False)
        ms_tier = (time.perf_counter() - t0) * 1000.0

        top_full = [d for d, _s in exhaustive]
        top_tier = [d for d, _s in tiered]
        overlap = len(set(top_full) & set(top_tier))
        rows.append({
            "query": raw,
            "terms": terms,
            "scored_exhaustive": int(full.size),
            "scored_tier1": int(trace.scored),
            "fell_through": trace.fell_through,
            "reduction": 1.0 - (trace.scored / full.size) if full.size else 0.0,
            "ms_exhaustive": round(ms_full, 2),
            "ms_tier1": round(ms_tier, 2),
            "speedup": (ms_full / ms_tier) if ms_tier > 0 else 0.0,
            "identical_topk": top_full == top_tier,
            "overlap": overlap,
            "recall_at_k": overlap / k if k else 0.0,
            "top1_same": bool(top_full[:1] == top_tier[:1]),
        })
        if verbose:
            print(f"  {raw[:38]:<38} {full.size:>7,} -> {trace.scored:>6,} "
                  f"overlap {overlap}/{k}", file=sys.stderr)

    n = max(len(rows), 1)
    summary = {
        "k": k,
        "r": tiers.r,
        "queries": len(rows),
        "mean_scored_exhaustive": sum(r["scored_exhaustive"] for r in rows) / n,
        "mean_scored_tier1": sum(r["scored_tier1"] for r in rows) / n,
        "mean_reduction": sum(r["reduction"] for r in rows) / n,
        "mean_speedup": sum(r["speedup"] for r in rows) / n,
        "identical_topk": sum(1 for r in rows if r["identical_topk"]),
        "identical_fraction": sum(1 for r in rows if r["identical_topk"]) / n,
        "mean_recall_at_k": sum(r["recall_at_k"] for r in rows) / n,
        "top1_same": sum(1 for r in rows if r["top1_same"]),
        "fell_through": sum(1 for r in rows if r["fell_through"]),
        "rows": rows,
    }
    return summary


def print_evaluation(s: Dict[str, Any]) -> None:
    p = print
    p("=" * 78)
    p(f"S5 two-tier index — speed-up vs fidelity   (r={s['r']}, k={s['k']}, "
      f"{s['queries']} queries)")
    p("=" * 78)
    p(f"  {'query':<34}{'full':>8}{'tier1':>8}{'cut':>7}{'ms f':>7}"
      f"{'ms t1':>7}{'top-k':>7}")
    for r in s["rows"]:
        mark = "=" if r["identical_topk"] else f"{r['overlap']}/{s['k']}"
        p(f"  {r['query'][:33]:<34}{r['scored_exhaustive']:>8,}"
          f"{r['scored_tier1']:>8,}{r['reduction']:>6.0%}"
          f"{r['ms_exhaustive']:>7.1f}{r['ms_tier1']:>7.1f}{mark:>7}")
    p("")
    p(f"mean documents scored     : {s['mean_scored_exhaustive']:,.0f} "
      f"exhaustive -> {s['mean_scored_tier1']:,.0f} in tier 1 "
      f"({s['mean_reduction']:.1%} fewer)")
    p(f"mean wall-clock speed-up  : {s['mean_speedup']:.2f}x")
    p(f"top-{s['k']} identical          : {s['identical_topk']}/{s['queries']} "
      f"({s['identical_fraction']:.0%})")
    p(f"mean recall@{s['k']} vs exhaustive: {s['mean_recall_at_k']:.3f}")
    p(f"rank-1 unchanged          : {s['top1_same']}/{s['queries']}")
    p(f"fell through to tier 2    : {s['fell_through']}/{s['queries']}")
    p("")
    p("This is a NON-SAFE optimisation: champion lists are not guaranteed to")
    p("contain the exhaustive top k, so the fidelity number above is the price")
    p("of the speed-up and belongs in the report next to it.")
    p("=" * 78)


# ==========================================================================
# 4. CLI
# ==========================================================================


def print_stats(t: TieredIndex) -> None:
    s = t.stats
    p = print
    p("=" * 68)
    p("S5 champion lists")
    p("=" * 68)
    p(f"r (champions per term)   : {s.r}")
    p(f"dictionary terms         : {s.vocab:,}")
    p(f"  with a champion list   : {s.terms_with_champions:,}")
    p(f"  below df {MIN_DF_FOR_CHAMPIONS} (skipped)     : {s.terms_too_rare:,}")
    p(f"champion entries         : {s.champion_entries:,}")
    p(f"full postings entries    : {s.postings_entries:,}")
    p(f"tier 1 is               : {s.compression:.1%} of the postings")
    p(f"ordered by              : ALPHA*w(t,d)_norm + (1-ALPHA)*g(d), "
      f"ALPHA={config.ALPHA}")
    p("=" * 68)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.index.tiers",
                                 description="Champion lists and the two-tier index.")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--r", type=int, default=config.CHAMPION_LIST_SIZE)
    b.add_argument("--min-df", type=int, default=MIN_DF_FOR_CHAMPIONS)
    sub.add_parser("stats")
    e = sub.add_parser("evaluate")
    e.add_argument("--k", type=int, default=config.TOP_K)
    e.add_argument("--queries", type=int, default=0,
                   help="use only the first N default queries")
    args = ap.parse_args(argv)

    if args.command == "build":
        with timed("S5 build"):
            t = build(r=args.r, min_df=args.min_df)
            out = t.save()
        print(f"wrote S5 -> {out}")
        print_stats(t)
    elif args.command == "stats":
        print_stats(TieredIndex.load())
    else:
        qs = DEFAULT_QUERIES[:args.queries] if args.queries else None
        with timed("S5 evaluate"):
            s = evaluate(qs, k=args.k)
        (config.INDEX / "tiers_eval.json").write_text(
            json.dumps(s, indent=2), encoding="utf-8")
        print_evaluation(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
