"""BM25F (mode S2p) — combine the zone tfs **first**, saturate **once**.

The distinction that is the whole point of BM25F
------------------------------------------------
It is tempting to run BM25 on each zone and add the scores up with weights.
**That is not BM25F and it is wrong**, because BM25's ``tf / (k1 + tf)``
saturation is non-linear: running it per zone saturates each zone separately, so
a term appearing once in the title and once in the body earns nearly two full
"first occurrence" bonuses, while the same two occurrences inside the body earn
one bonus and a small increment.  The structure of the document would change the
score for reasons that have nothing to do with relevance.

BM25F (Robertson, Zaragoza & Taylor 2004) instead builds **one pseudo term
frequency** per (term, document) by length-normalising each zone's tf and
weighting it, and only then applies the saturation::

    tf~(t,d)  =  SUM over zones   w_z * tf_z(t,d)
                                  ------------------------------------
                                  1 - b_z + b_z * (len_z(d) / avglen_z)

    score(t,d) = idf(t) * tf~(t,d) / (k1 + tf~(t,d))

So: weight and length-normalise per zone, **sum**, then saturate once.  Each
zone keeps its own ``b_z`` (`config.BM25F_B`) because a 9-token title and a
3,644-token body need different length normalisation - a title that is twice the
average length is not "padded" the way a body twice the average length is, so
the title's ``b`` is low (0.3) and the body's is high (0.75).

Note what is *not* here: no document-side cosine normalisation.  BM25's length
handling is the ``b`` term inside the saturation; adding a cosine on top would
normalise for length twice.

Author: Sumedha (M3)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.rank.tfidf import _align

__all__ = ["score", "bm25f_scores", "zone_pseudo_tf", "avg_zone_lengths"]

ZONE_IDS: Dict[str, int] = dict(config.ZONE_ID)

#: Cached on the index object itself, not in a dict keyed by id(): CPython
#: reuses an id once the object is collected, so an id-keyed cache silently
#: hands one index another index's averages.  A unit test caught exactly that.
_AVG_ATTR = "_nyaya_avg_zone_len"


def avg_zone_lengths(index) -> np.ndarray:
    """Mean tokens per zone over the documents that *have* that zone.

    Averaging over documents that lack the zone would drag ``avglen`` toward
    zero and make every document that does have it look over-long.  Cached per
    index object: it is a scan of a 13,878 x 6 array and the value never moves.
    """
    hit = getattr(index, _AVG_ATTR, None)
    if hit is not None:
        return hit
    zl = np.asarray(index.zone_len, dtype=np.float64)
    live = np.asarray(index.doc_len, dtype=np.float64) > 0
    out = np.ones(zl.shape[1], dtype=np.float64)
    for j in range(zl.shape[1]):
        col = zl[live, j]
        present = col > 0
        if present.any():
            out[j] = float(col[present].mean())
    try:
        setattr(index, _AVG_ATTR, out)
    except Exception:                                       # pragma: no cover
        pass
    return out


def zone_pseudo_tf(ztf_rows: np.ndarray, doc_rows: np.ndarray, index,
                   weights: Dict[str, float], b: Dict[str, float],
                   avg: np.ndarray) -> np.ndarray:
    """``tf~`` for one term: weighted, length-normalised, summed over zones.

    ``ztf_rows`` is (n, 6) zone term frequencies, ``doc_rows`` the matching doc
    ids.  Saturation is deliberately **not** applied here - see the module
    docstring.
    """
    zl = np.asarray(index.zone_len, dtype=np.float64)[doc_rows]
    out = np.zeros(ztf_rows.shape[0], dtype=np.float64)
    for zone, j in ZONE_IDS.items():
        w = float(weights.get(zone, 0.0))
        if w <= 0.0:
            continue                      # bench is filter-only: weight 0
        tf = ztf_rows[:, j].astype(np.float64)
        if not tf.any():
            continue
        bz = float(b.get(zone, 0.75))
        lens = zl[:, j]
        denom = 1.0 - bz + bz * (lens / (avg[j] if avg[j] > 0 else 1.0))
        denom = np.where(denom > 0.0, denom, 1.0)
        out += w * tf / denom
    return out


def bm25f_scores(doc_ids: np.ndarray, terms: Sequence[str], ctx,
                 *, index=None, k1: float = None,
                 weights: Optional[Dict[str, float]] = None,
                 b: Optional[Dict[str, float]] = None,
                 collect: Optional[Dict[str, np.ndarray]] = None,
                 per_zone: Optional[Dict[str, np.ndarray]] = None
                 ) -> np.ndarray:
    """BM25F over the candidate set, aligned with ``doc_ids``.

    ``collect`` receives per-term score arrays and ``per_zone`` the pre-saturation
    weighted-tf mass per zone, both for the "why this result" panel.  Note what
    ``per_zone`` can and cannot claim: because saturation happens *after* the
    zones are summed, a zone's share of the final score is not well defined, so
    the panel reports each zone's share of ``tf~`` and says so.
    """
    index = index if index is not None else getattr(ctx, "index", None)
    ids = np.asarray(doc_ids, dtype=np.int32)
    out = np.zeros(ids.size, dtype=np.float64)
    if index is None or ids.size == 0:
        return out
    k1 = config.BM25F_K1 if k1 is None else float(k1)
    weights = weights or config.ZONE_WEIGHTS
    b = b or config.BM25F_B
    avg = avg_zone_lengths(index)

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    if per_zone is not None:
        for z in config.ZONES:
            per_zone.setdefault(z, np.zeros(ids.size, dtype=np.float64))

    seen = set()
    for term in terms:
        if term in seen:
            continue                      # a repeated query term is one term
        seen.add(term)
        posts = index.postings(term)
        if posts is None:
            continue
        idf = float(index.idf(term))
        if idf <= 0.0:
            continue
        d = np.asarray(posts.doc_ids, dtype=np.int32)
        ztf = np.asarray(posts.zone_tf_matrix(), dtype=np.float64)
        rows_idx = np.arange(d.size)
        rows, keep = _align(sorted_ids, order, d, rows_idx.astype(np.float64))
        if rows.size == 0:
            continue
        sel = keep.astype(np.int64)
        tf_tilde = zone_pseudo_tf(ztf[sel], d[sel], index, weights, b, avg)
        # ---- the single saturation, after the zones have been combined ----
        contrib = idf * tf_tilde / (k1 + tf_tilde)
        np.add.at(out, rows, contrib)
        if collect is not None:
            col = np.zeros(ids.size, dtype=np.float64)
            np.add.at(col, rows, contrib)
            collect[term] = col
        if per_zone is not None:
            zl = np.asarray(index.zone_len, dtype=np.float64)[d[sel]]
            for zone, j in ZONE_IDS.items():
                w = float(weights.get(zone, 0.0))
                if w <= 0.0:
                    continue
                bz = float(b.get(zone, 0.75))
                denom = 1.0 - bz + bz * (zl[:, j] / (avg[j] if avg[j] > 0 else 1.0))
                mass = w * ztf[sel][:, j] / np.where(denom > 0, denom, 1.0)
                np.add.at(per_zone[zone], rows, mass)
    return out


def score(doc_ids: np.ndarray, terms: Sequence[str], ctx) -> np.ndarray:
    """`ScoreFn` entry point for `nyaya.rank.net.rank` (modes S2p, S3, S4)."""
    return bm25f_scores(doc_ids, terms, ctx)


score.__name__ = "bm25f"


def explain(doc_id: int, doc_ids: np.ndarray, terms: Sequence[str], ctx,
            top: int = 3) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    """``(top terms, per-zone tf~ mass)`` for one document."""
    collect: Dict[str, np.ndarray] = {}
    zones: Dict[str, np.ndarray] = {}
    bm25f_scores(doc_ids, terms, ctx, collect=collect, per_zone=zones)
    ids = np.asarray(doc_ids, dtype=np.int32)
    where = np.nonzero(ids == int(doc_id))[0]
    if where.size == 0:
        return [], {}
    i = int(where[0])
    rows = sorted(((t, float(v[i])) for t, v in collect.items() if v[i] != 0.0),
                  key=lambda kv: -kv[1])
    zmass = {z: float(v[i]) for z, v in zones.items() if v[i] != 0.0}
    return rows[:top], zmass
