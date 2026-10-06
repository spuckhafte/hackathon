"""lnc.ltc tf-idf — the B0 baseline, term-at-a-time with an accumulator.

The textbook form (Manning, *IIR* 6.4 and 7.1), written out rather than
vectorised into unreadability, because the point of B0 is that every other mode
is measured against something whose arithmetic a reader can check.

The SMART notation
------------------
``lnc.ltc`` names the two weighting schemes, document side then query side::

    document   l n c    log tf  ·  no idf  ·  cosine normalisation
    query      l t c    log tf  ·  idf     ·  cosine normalisation

So the document vector is ``1 + ln(tf)`` with **no idf**, normalised by its own
length; the query vector is ``(1 + ln(tf_q)) · idf`` normalised by its length.
idf appears exactly once, on the query side - that is the whole trick of
``lnc.ltc``, and it is why the document norm can be precomputed at index time
(``index.doc_norm``) and reused by every query.

Term-at-a-time with an accumulator
----------------------------------
We walk one postings list at a time and add that term's contribution into an
accumulator keyed by document, rather than walking all lists in parallel
(document-at-a-time).  Term-at-a-time is the right shape here because the
postings are on disk in term order: one sequential read per query term, and the
accumulator is the only random access.  The accumulator is a dense array over
the *candidate* set, not over all 13,878 documents, so a filtered query pays for
what it selected.

Author: Sumedha (M3)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["score", "ltc_query_weights", "lnc_ltc", "explain_terms"]


def ltc_query_weights(terms: Sequence[str], index) -> Dict[str, float]:
    """The ``ltc`` query vector: log tf, times idf, cosine normalised.

    Cosine-normalising the *query* changes no ranking on its own - it divides
    every document's score by the same constant - but it puts scores from
    different queries on one scale, which is what makes a mean over a query set
    meaningful.
    """
    tf = Counter(terms)
    raw: Dict[str, float] = {}
    for term, n in tf.items():
        idf = float(index.idf(term)) if index is not None else 0.0
        if idf <= 0.0:
            continue                      # a term in every document says nothing
        raw[term] = (1.0 + math.log(n)) * idf
    norm = math.sqrt(sum(w * w for w in raw.values()))
    if norm == 0.0:
        return {}
    return {t: w / norm for t, w in raw.items()}


def lnc_ltc(doc_ids: np.ndarray, terms: Sequence[str], ctx,
            *, index=None, collect: Optional[Dict[str, np.ndarray]] = None
            ) -> np.ndarray:
    """Cosine similarity of the ``lnc`` document vectors with the ``ltc`` query.

    Returns one score per entry of ``doc_ids``, aligned with it.  ``collect``, if
    given, is filled with the per-term contribution arrays so the "why this
    result" panel can name the terms that did the work without scoring twice.
    """
    index = index if index is not None else getattr(ctx, "index", None)
    ids = np.asarray(doc_ids, dtype=np.int32)
    acc = np.zeros(ids.size, dtype=np.float64)
    if index is None or ids.size == 0:
        return acc

    # One sorted copy of the candidate set drives every postings merge.
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]

    weights = ltc_query_weights(terms, index)
    for term, qw in weights.items():
        posts = index.postings(term)
        if posts is None:
            continue
        d = np.asarray(posts.doc_ids, dtype=np.int32)
        tf = np.asarray(posts.tf_array(), dtype=np.float64)
        rows, tf_hit = _align(sorted_ids, order, d, tf)
        if rows.size == 0:
            continue
        # lnc: log tf, no idf.  The 'c' is applied once, below, from doc_norm.
        contrib = (1.0 + np.log(np.maximum(tf_hit, 1.0))) * qw
        np.add.at(acc, rows, contrib)
        if collect is not None:
            col = np.zeros(ids.size, dtype=np.float64)
            np.add.at(col, rows, contrib)
            collect[term] = col

    norm = getattr(index, "doc_norm", None)
    if norm is not None:
        denom = np.asarray(norm, dtype=np.float64)[ids]
        denom = np.where(denom > 0.0, denom, 1.0)
        acc = acc / denom
        if collect is not None:
            for t in list(collect):
                collect[t] = collect[t] / denom
    return acc


def _align(sorted_ids: np.ndarray, order: np.ndarray,
           posting_ids: np.ndarray, values: np.ndarray
           ) -> Tuple[np.ndarray, np.ndarray]:
    """Rows of the accumulator that this postings list touches.

    `searchsorted` against the sorted candidate ids, then an equality check:
    the equality check is what stops a near-miss from being credited to its
    neighbour, which is the classic silent bug in this join.
    """
    pos = np.searchsorted(sorted_ids, posting_ids)
    inb = pos < sorted_ids.size
    pos, vals, pids = pos[inb], values[inb], posting_ids[inb]
    hit = sorted_ids[pos] == pids
    return order[pos[hit]], vals[hit]


def score(doc_ids: np.ndarray, terms: Sequence[str], ctx) -> np.ndarray:
    """`ScoreFn` entry point for `nyaya.rank.net.rank`."""
    return lnc_ltc(doc_ids, terms, ctx)


score.__name__ = "lnc.ltc"


def explain_terms(doc_id: int, doc_ids: np.ndarray, terms: Sequence[str], ctx,
                  top: int = 3) -> List[Tuple[str, float]]:
    """The ``top`` query terms contributing most to one document's score."""
    collect: Dict[str, np.ndarray] = {}
    lnc_ltc(doc_ids, terms, ctx, collect=collect)
    ids = np.asarray(doc_ids, dtype=np.int32)
    where = np.nonzero(ids == int(doc_id))[0]
    if where.size == 0:
        return []
    i = int(where[0])
    rows = [(t, float(v[i])) for t, v in collect.items() if v[i] != 0.0]
    rows.sort(key=lambda kv: -kv[1])
    return rows[:top]
