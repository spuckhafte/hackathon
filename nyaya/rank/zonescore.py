"""Weighted-zone scoring (mode S2) — one cosine per zone, combined by weight.

The idea (Manning, *IIR* 6.1)
----------------------------
A judgment is not a flat bag of words.  A query term appearing in the **title**
of a 9-token title is strong evidence about what the case is about; the same
term appearing once in a 3,644-token body is weak evidence.  Flat tf-idf cannot
express that, because it has already thrown the structure away.

So we score each zone separately and combine::

    score(d, q) = SUM over zones  w_z * cosine_z(d, q)   /   SUM w_z

with ``w_z`` from `config.ZONE_WEIGHTS`.  Dividing by the weight sum keeps the
result on the same 0-1 scale as the flat cosine, so S2 and B0 scores are
comparable and `net_score`'s alpha means the same thing in both.

How this differs from BM25F (which is the next module along)
------------------------------------------------------------
Here each zone is scored **independently and then mixed**.  BM25F mixes the
*term frequencies* first and saturates once.  Both are defensible; this one is
the linear-combination-of-scores form, and it is kept precisely so the report
can show the difference between the two on the same query set rather than
asserting it.

`config.ZONE_WEIGHTS["bench"] = 0.0` - the coram line is a **filter-only** zone.
It is searchable through ``judge:``, and scoring it would rank a case highly for
containing a judge's surname in a list.

Author: Sumedha (M3)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.rank.tfidf import _align, ltc_query_weights

__all__ = ["score", "zone_scores", "per_zone_contributions", "ZONE_IDS"]

ZONE_IDS: Dict[str, int] = dict(config.ZONE_ID)

#: Pivoted length normalisation slope (Singhal, Buckley & Mitra 1996).
#: 1.0 is plain cosine, 0.0 is no length normalisation.  Ablatable.
PIVOT_SLOPE = 0.3

#: Cached on the index object (see bm25f._AVG_ATTR for why not id()).
_PIVOT_ATTR = "_nyaya_zone_pivot"


def _pivot(index, zone_j: int) -> float:
    """Mean sqrt(zone length) over the documents that have the zone.

    Averaged over documents that *have* the zone, not all documents: including
    the zeros would drag the pivot down and re-create the short-zone bias the
    pivot exists to remove.
    """
    cache = getattr(index, _PIVOT_ATTR, None)
    if cache is None:
        cache = {}
        try:
            setattr(index, _PIVOT_ATTR, cache)
        except Exception:                                   # pragma: no cover
            pass
    if zone_j in cache:
        return cache[zone_j]
    zl = np.asarray(index.zone_len, dtype=np.float64)
    live = np.asarray(index.doc_len, dtype=np.float64) > 0
    col = zl[live, zone_j]
    present = col[col > 0]
    val = float(np.sqrt(present).mean()) if present.size else 1.0
    cache[zone_j] = val
    return val

#: Zones with a non-zero weight — the only ones worth reading postings for.
def _scored_zones(weights: Dict[str, float]) -> List[str]:
    return [z for z in config.ZONES if weights.get(z, 0.0) > 0.0]


def zone_scores(doc_ids: np.ndarray, terms: Sequence[str], ctx,
                *, weights: Optional[Dict[str, float]] = None,
                index=None,
                collect: Optional[Dict[str, np.ndarray]] = None
                ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return ``(combined, per_zone)``, both aligned with ``doc_ids``.

    ``per_zone[z]`` is the **weighted** contribution of zone ``z``, so the
    explanation panel can print the parts and have them sum to the total.
    """
    weights = weights or config.ZONE_WEIGHTS
    index = index if index is not None else getattr(ctx, "index", None)
    ids = np.asarray(doc_ids, dtype=np.int32)
    zones = _scored_zones(weights)
    per_zone = {z: np.zeros(ids.size, dtype=np.float64) for z in zones}
    if index is None or ids.size == 0 or not zones:
        return np.zeros(ids.size, dtype=np.float64), per_zone

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    qw = ltc_query_weights(terms, index)

    # Numerators per zone, then one cosine normalisation per zone at the end.
    for term, w in qw.items():
        posts = index.postings(term)
        if posts is None:
            continue
        d = np.asarray(posts.doc_ids, dtype=np.int32)
        ztf = np.asarray(posts.zone_tf_matrix(), dtype=np.float64)
        for z in zones:
            col = ztf[:, ZONE_IDS[z]]
            nz = col > 0
            if not nz.any():
                continue
            rows, tf_hit = _align(sorted_ids, order, d[nz], col[nz])
            if rows.size == 0:
                continue
            contrib = (1.0 + np.log(tf_hit)) * w
            np.add.at(per_zone[z], rows, contrib)
            if collect is not None:
                key = f"{z}:{term}"
                acc = collect.setdefault(key, np.zeros(ids.size, dtype=np.float64))
                np.add.at(acc, rows, contrib)

    # ---- length normalisation, pivoted ----------------------------------
    # There is no precomputed per-zone vector norm, so the zone's token count
    # stands in for it and the normaliser is sqrt(len_z).  Plain cosine then
    # over-rewards very short zones, and here that is not a theoretical worry:
    # measured, the query "right to privacy of personal data" ranked
    # "MAK DATA P. LTD. v. Commissioner of Income Tax" first, because "data"
    # sat in a 9-token title and 1/sqrt(9) times the title weight of 3 beat
    # every real privacy judgment.
    #
    # The textbook remedy is pivoted length normalisation (Singhal, Buckley &
    # Mitra, SIGIR 1996): replace the normaliser by a line through the pivot,
    #
    #     denom = (1 - s) * pivot_z  +  s * sqrt(len_z)
    #
    # where pivot_z is the mean of sqrt(len) over documents that have the zone.
    # At s = 1 this is plain cosine; at s = 0 no length normalisation at all.
    # PIVOT_SLOPE = 0.3 is Singhal's recommended region and it is a module
    # constant so the choice can be ablated like any other.
    zl = getattr(index, "zone_len", None)
    total_w = sum(weights[z] for z in zones) or 1.0
    combined = np.zeros(ids.size, dtype=np.float64)
    for z in zones:
        if zl is not None:
            lens = np.asarray(zl, dtype=np.float64)[ids, ZONE_IDS[z]]
            raw = np.sqrt(np.where(lens > 0.0, lens, 1.0))
            denom = (1.0 - PIVOT_SLOPE) * _pivot(index, ZONE_IDS[z])                 + PIVOT_SLOPE * raw
            denom = np.where(denom > 0.0, denom, 1.0)
            per_zone[z] = per_zone[z] / denom
            if collect is not None:
                for key in [k for k in collect if k.startswith(f"{z}:")]:
                    collect[key] = collect[key] / denom
        per_zone[z] *= weights[z]
        if collect is not None:
            for key in [k for k in collect if k.startswith(f"{z}:")]:
                collect[key] = collect[key] * weights[z] / total_w
        combined += per_zone[z]
    combined /= total_w
    for z in zones:
        per_zone[z] = per_zone[z] / total_w
    return combined, per_zone


def score(doc_ids: np.ndarray, terms: Sequence[str], ctx) -> np.ndarray:
    """`ScoreFn` entry point for `nyaya.rank.net.rank` (mode S2)."""
    combined, _ = zone_scores(doc_ids, terms, ctx)
    return combined


score.__name__ = "weighted-zone"


def uniform(doc_ids: np.ndarray, terms: Sequence[str], ctx) -> np.ndarray:
    """Mode S1: the same machinery with every zone weighted 1.

    S1 is "B0 + statute canonicalisation + the legal stop list" and **no** zone
    weighting, so it is this scorer with flat weights over the full index rather
    than a third index on disk.  `bench` stays at 0 because it is filter-only in
    every mode.
    """
    flat = {z: (0.0 if z == "bench" else 1.0) for z in config.ZONES}
    combined, _ = zone_scores(doc_ids, terms, ctx, weights=flat)
    return combined


uniform.__name__ = "uniform-zone"


def per_zone_contributions(doc_id: int, doc_ids: np.ndarray,
                           terms: Sequence[str], ctx,
                           weights: Optional[Dict[str, float]] = None
                           ) -> Dict[str, float]:
    """The weighted per-zone numbers for one document, for the F5 panel."""
    _, per_zone = zone_scores(doc_ids, terms, ctx, weights=weights)
    ids = np.asarray(doc_ids, dtype=np.int32)
    where = np.nonzero(ids == int(doc_id))[0]
    if where.size == 0:
        return {}
    i = int(where[0])
    return {z: float(v[i]) for z, v in per_zone.items() if v[i] != 0.0}
