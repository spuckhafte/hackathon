"""Retrieval metrics  (Member 1, wave 4).

**This is the only implementation in the project.**  Member 2's efficiency and
citation-recovery evaluation imports from here rather than writing a second one:
two implementations of nDCG that disagree by a few points is a textbook way to
lose evaluation marks, and the disagreement is always found by the marker rather
than by us.

Relevance is graded 0 / 1 / 2 (``eval/queries/qrels.jsonl``), which changes two
of the definitions and both changes are deliberate:

* **Precision and recall binarise** at ``rel >= REL_BINARY`` (1).  "Precision"
  over graded judgments has no agreed meaning, and inventing one would make our
  numbers incomparable with every published baseline.
* **nDCG uses the grades**, with gain ``2**rel - 1``, so a grade-2 authority is
  worth three times a grade-1 one.  That is the whole reason for grading, and the
  ideal ranking it is normalised against is the grades sorted descending.

Pooling, and the honest reading of recall
-----------------------------------------
The qrels were pooled from the top 10 of B0 and S4, so a document no system
retrieved was never judged.  R@20 is therefore **pooled recall** - recall with
respect to the judged pool, not the collection - and it is reported under that
name.  Unpooled recall is not computable for this corpus and claiming it would be
false.  A document retrieved but not in the pool counts as non-relevant, which is
the standard TREC assumption and is also the direction that cannot flatter us.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: A graded judgment counts as relevant for the binary metrics at or above this.
REL_BINARY = 1

#: The cutoffs the report quotes.  Named here so the harness, the plots and the
#: report cannot drift apart.
P_CUTOFFS = (5, 10)
R_CUTOFF = 20
NDCG_CUTOFF = 10

METRIC_NAMES = ("P@5", "P@10", "R@20", "MAP", "nDCG@10")


# ==========================================================================
# 1. Per-query metrics
# ==========================================================================

def check_qrels(rels: Any) -> Mapping[int, int]:
    """Validate the judgment argument, loudly.

    These functions take a **graded** ``{doc_id: grade}`` mapping.  A bare ``set``
    of relevant ids is the other natural shape and it used to fail deep inside
    with ``AttributeError: 'set' object has no attribute 'values'``, which is a
    bad way to find out.  This module is the project's single implementation of
    these metrics, so the boundary is worth being strict about.
    """
    if isinstance(rels, Mapping):
        return rels
    if isinstance(rels, (set, frozenset)):
        raise TypeError(
            "qrels must be a graded mapping {doc_id: grade}, not a set of "
            "relevant ids; wrap a binary set as {d: 1 for d in s}")
    raise TypeError(f"qrels must be a Mapping, got {type(rels).__name__}")


def dedupe(ranked: Sequence[int]) -> List[int]:
    """Order-preserving de-duplication of a ranking.

    Every metric here counts *documents*, not positions.  ``search()`` does not
    emit a document twice, but Track A pools the top 10 of six systems into one
    list, and a duplicate there made ``recall_at_k`` return 1.5 - a recall above
    1.0 in the report is the kind of thing a marker spots on the first page.
    De-duplicating once, here, fixes every metric at the same time.
    """
    seen: set = set()
    out: List[int] = []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def precision_at_k(ranked: Sequence[int], rels: Mapping[int, int], k: int) -> float:
    """Fraction of the top ``k`` that are relevant (binarised at REL_BINARY).

    The denominator is ``k``, not ``min(k, len(ranked))``: a system that returns
    four documents when ten were asked for has not earned the precision of a
    four-document list.
    """
    rels = check_qrels(rels)
    if k <= 0:
        return 0.0
    hits = sum(1 for d in dedupe(ranked)[:k] if rels.get(d, 0) >= REL_BINARY)
    return hits / float(k)


def recall_at_k(ranked: Sequence[int], rels: Mapping[int, int], k: int) -> float:
    """Pooled recall: relevant documents found in the top ``k``, over all judged
    relevant documents for this query.  See the module docstring on pooling."""
    rels = check_qrels(rels)
    total = sum(1 for v in rels.values() if v >= REL_BINARY)
    if total == 0:
        return float("nan")
    hits = sum(1 for d in dedupe(ranked)[:k] if rels.get(d, 0) >= REL_BINARY)
    return hits / float(total)


def average_precision(ranked: Sequence[int], rels: Mapping[int, int],
                      k: Optional[int] = None) -> float:
    """AP: the mean of the precisions at each relevant document's rank.

    Divided by the **total number of judged relevant documents**, not by the
    number found, so a system that retrieves two of six relevant documents
    perfectly does not score 1.0.  That divisor is what makes MAP a
    recall-sensitive measure, and getting it wrong is the classic MAP bug.
    """
    rels = check_qrels(rels)
    total = sum(1 for v in rels.values() if v >= REL_BINARY)
    if total == 0:
        return float("nan")
    uniq = dedupe(ranked)
    cut = uniq if k is None else uniq[:k]
    hits = 0
    acc = 0.0
    for i, d in enumerate(cut, start=1):
        if rels.get(d, 0) >= REL_BINARY:
            hits += 1
            acc += hits / float(i)
    return acc / float(total)


def dcg(gains: Sequence[float]) -> float:
    """sum over i of gain_i / log2(i + 1), ranks 1-based."""
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1) if g)


def ndcg_at_k(ranked: Sequence[int], rels: Mapping[int, int], k: int = NDCG_CUTOFF
              ) -> float:
    """Graded nDCG with gain ``2**rel - 1``.

    The ideal ranking is every judged grade sorted descending, truncated to ``k``
    - not the grades of the documents this system happened to retrieve, which
    would let a system that misses the best authority still score 1.0.
    """
    rels = check_qrels(rels)
    gains = [2.0 ** rels.get(d, 0) - 1.0 for d in dedupe(ranked)[:k]]
    ideal = sorted((2.0 ** v - 1.0 for v in rels.values()), reverse=True)[:k]
    denom = dcg(ideal)
    if denom == 0.0:
        return float("nan")
    return dcg(gains) / denom


def reciprocal_rank(ranked: Sequence[int], rels: Mapping[int, int]) -> float:
    rels = check_qrels(rels)
    for i, d in enumerate(dedupe(ranked), start=1):
        if rels.get(d, 0) >= REL_BINARY:
            return 1.0 / i
    return 0.0


# ==========================================================================
# 2. One query, all metrics
# ==========================================================================

@dataclass
class QueryScore:
    """Every metric for one query under one system."""

    qid: str
    system: str
    n_returned: int = 0
    n_relevant_judged: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        row = {"qid": self.qid, "system": self.system,
               "returned": self.n_returned,
               "relevant_judged": self.n_relevant_judged}
        row.update({m: self.metrics.get(m, float("nan")) for m in METRIC_NAMES})
        row["RR"] = self.metrics.get("RR", float("nan"))
        return row


def score_query(qid: str, system: str, ranked: Sequence[int],
                rels: Mapping[int, int]) -> QueryScore:
    """All five reported metrics plus reciprocal rank, for one ranking."""
    rels = check_qrels(rels)
    ranked = dedupe(ranked)
    out = QueryScore(qid=qid, system=system, n_returned=len(ranked),
                     n_relevant_judged=sum(1 for v in rels.values()
                                           if v >= REL_BINARY))
    for k in P_CUTOFFS:
        out.metrics[f"P@{k}"] = precision_at_k(ranked, rels, k)
    out.metrics[f"R@{R_CUTOFF}"] = recall_at_k(ranked, rels, R_CUTOFF)
    out.metrics["MAP"] = average_precision(ranked, rels)
    out.metrics[f"nDCG@{NDCG_CUTOFF}"] = ndcg_at_k(ranked, rels, NDCG_CUTOFF)
    out.metrics["RR"] = reciprocal_rank(ranked, rels)
    return out


# ==========================================================================
# 3. Aggregation
# ==========================================================================

def _mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v == v]           # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate(scores: Sequence[QueryScore]) -> Dict[str, float]:
    """Macro average over queries: every query counts the same.

    Macro, not micro, because the queries are the sample we are generalising
    over.  A micro average would let one query with twenty judged relevant
    documents outvote four queries with three.
    """
    out: Dict[str, float] = {}
    for m in METRIC_NAMES + ("RR",):
        out[m] = _mean(s.metrics.get(m, float("nan")) for s in scores)
    out["n_queries"] = float(len(scores))
    return out


# ==========================================================================
# 4. Significance, because 20 queries is not many
# ==========================================================================

def sign_test(a: Sequence[float], b: Sequence[float], eps: float = 1e-9
              ) -> Dict[str, Any]:
    """Two-sided paired sign test on per-query differences ``a - b``.

    Exact, from the binomial distribution, with ties discarded - the standard
    non-parametric test for "did system A beat system B" when the sample is 20
    queries and the metric is not normally distributed.  Reported because a
    two-point difference in mean nDCG over 20 queries is not evidence of
    anything, and saying so is part of doing this honestly.
    """
    wins = sum(1 for x, y in zip(a, b) if x - y > eps)
    losses = sum(1 for x, y in zip(a, b) if y - x > eps)
    ties = len(a) - wins - losses
    n = wins + losses
    if n == 0:
        return {"wins": wins, "losses": losses, "ties": ties, "n": 0,
                "p_value": float("nan"), "significant_at_05": False}
    # P(X >= max(wins, losses)) * 2 under X ~ Binomial(n, 0.5)
    hi = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(hi, n + 1)) / float(2 ** n)
    p = min(1.0, 2.0 * tail)
    return {"wins": wins, "losses": losses, "ties": ties, "n": n,
            "p_value": p, "significant_at_05": p < 0.05}


def bootstrap_ci(a: Sequence[float], b: Sequence[float], iterations: int = 10_000,
                 alpha: float = 0.05, seed: int = 358) -> Dict[str, Any]:
    """Percentile bootstrap CI for the mean paired difference ``a - b``.

    Resamples *queries*, which is the unit of variation, not documents.  If the
    interval straddles zero the improvement is not established at this sample
    size, and the report must say so rather than quoting the point estimate
    alone.
    """
    import random

    diffs = [x - y for x, y in zip(a, b) if (x == x and y == y)]
    n = len(diffs)
    if n == 0:
        return {"mean_diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "excludes_zero": False, "n": 0}
    rnd = random.Random(seed)
    means: List[float] = []
    for _ in range(iterations):
        sample = [diffs[rnd.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iterations)]
    hi = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    mean_diff = sum(diffs) / n
    return {"mean_diff": mean_diff, "lo": lo, "hi": hi,
            "excludes_zero": (lo > 0.0 or hi < 0.0), "n": n,
            "iterations": iterations}


def per_query_comparison(a_scores: Sequence[QueryScore],
                         b_scores: Sequence[QueryScore],
                         metric: str = "nDCG@10") -> Dict[str, Any]:
    """Per-query win/tie/loss of system A over system B, plus both tests.

    With 20 test queries the per-query breakdown *is* the result; the mean is a
    summary of it, and a mean that moves because one query moved a lot is not an
    improvement.
    """
    by_qid = {s.qid: s for s in b_scores}
    rows: List[Dict[str, Any]] = []
    a_vals: List[float] = []
    b_vals: List[float] = []
    for s in a_scores:
        other = by_qid.get(s.qid)
        if other is None:
            continue
        av = s.metrics.get(metric, float("nan"))
        bv = other.metrics.get(metric, float("nan"))
        a_vals.append(av)
        b_vals.append(bv)
        delta = av - bv
        rows.append({"qid": s.qid, "a": av, "b": bv, "delta": delta,
                     "outcome": ("improved" if delta > 1e-9
                                 else "worsened" if delta < -1e-9 else "tied")})
    rows.sort(key=lambda r: -r["delta"])
    counts = {o: sum(1 for r in rows if r["outcome"] == o)
              for o in ("improved", "tied", "worsened")}
    return {
        "metric": metric,
        "a_system": a_scores[0].system if a_scores else "",
        "b_system": b_scores[0].system if b_scores else "",
        "counts": counts,
        "mean_a": _mean(a_vals),
        "mean_b": _mean(b_vals),
        "sign_test": sign_test(a_vals, b_vals),
        "bootstrap": bootstrap_ci(a_vals, b_vals),
        "per_query": rows,
    }


# ==========================================================================
# 5. Loading the judged query set
# ==========================================================================

def load_queries(path) -> List[Dict[str, Any]]:
    from nyaya.utils import read_jsonl

    return list(read_jsonl(path))


def load_qrels(path) -> Dict[str, Dict[int, int]]:
    """``{qid: {doc_id: rel}}`` from ``eval/queries/qrels.jsonl``."""
    from nyaya.utils import read_jsonl

    out: Dict[str, Dict[int, int]] = {}
    for row in read_jsonl(path):
        out.setdefault(str(row["qid"]), {})[int(row["doc_id"])] = int(row["rel"])
    return out


def qrels_summary(qrels: Mapping[str, Mapping[int, int]]) -> Dict[str, Any]:
    grades: Dict[int, int] = {}
    for per_q in qrels.values():
        for v in per_q.values():
            grades[v] = grades.get(v, 0) + 1
    rel_per_q = [sum(1 for v in d.values() if v >= REL_BINARY)
                 for d in qrels.values()]
    return {
        "queries": len(qrels),
        "judgments": sum(len(d) for d in qrels.values()),
        "by_grade": dict(sorted(grades.items())),
        "mean_relevant_per_query": round(_mean(rel_per_q), 2) if rel_per_q else 0,
        "min_relevant_per_query": min(rel_per_q) if rel_per_q else 0,
        "max_relevant_per_query": max(rel_per_q) if rel_per_q else 0,
    }
