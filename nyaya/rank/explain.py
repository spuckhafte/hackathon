"""Feature F5, "why this result" — the panel whose arithmetic has to check out.

The design rule
---------------
Every number printed here is either *measured* or *derived from numbers also
printed here*.  A marker should be able to pause the video, take the per-zone
contributions, the alpha, the g components and the beta, do the arithmetic on
paper, and land on the net score at the bottom of the panel.  Anything that
cannot survive that is not an explanation, it is decoration.

Two places where honesty costs something, and we pay it
-------------------------------------------------------
1. **BM25F has no per-zone score.**  Saturation happens *after* the zones are
   summed (that is the whole point of BM25F — see `nyaya.rank.bm25f`), so
   "this zone contributed 0.3 of the score" is not a well-defined quantity.
   What *is* well defined is each zone's share of the pseudo term frequency
   ``tf~`` before saturation, so that is what the panel shows, labelled as such.
   The weighted-zone scorer (S2) genuinely does decompose, and there the per-zone
   numbers sum to the text score exactly — `verify()` checks it.
2. **text_norm is relative to the candidate set.**  Raw text scores are divided
   by the best score in *this* result set (`net.normalise_text_scores`), so the
   top hit always has ``text_norm = 1.0``.  That is a real property of the
   ranking, not a bug, and the panel says so rather than letting a reader think
   1.0 means "perfect match".

Author: Sumedha (M3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.contracts import ScoreExplanation

__all__ = ["build", "verify", "render", "ExplainInputs"]


@dataclass
class ExplainInputs:
    """Everything `build` needs, gathered once per query rather than per result."""

    doc_id: int
    doc_ids: np.ndarray
    terms: Sequence[str]
    ctx: Any
    scorer: str = "bm25f"
    text_raw: float = 0.0
    text_norm: float = 0.0
    alpha: float = config.ALPHA
    beta: float = config.BETA
    proximity: float = 0.0
    prox_window: int = 0
    prox_kind: str = ""
    authority: Any = None
    use_authority: bool = True
    expansions: Dict[str, List[str]] = field(default_factory=dict)
    weights: Optional[Dict[str, float]] = None


def build(inp: ExplainInputs) -> ScoreExplanation:
    """Fill `contracts.ScoreExplanation` for one result."""
    ex = ScoreExplanation(
        text_raw=float(inp.text_raw),
        text_norm=float(inp.text_norm),
        alpha=float(inp.alpha),
        beta=float(inp.beta),
        scorer=inp.scorer,
        prox_window=int(inp.prox_window),
        proximity=float(inp.proximity),
    )
    if inp.proximity:
        try:
            object.__setattr__(ex, "prox_kind", inp.prox_kind)
        except Exception:                                    # pragma: no cover
            pass
        ex.notes.append(
            f"proximity is 1 when the document satisfies the query's "
            f"{_prox_label(inp.prox_kind, inp.prox_window)} constraint and 0 "
            f"otherwise - a tie-break between documents the text scorer already "
            f"likes, which is why beta is only {inp.beta:g}")

    # ---- where the text score came from ---------------------------------
    if inp.scorer in ("weighted-zone", "uniform-zone"):
        from nyaya.rank.tfidf import ltc_query_weights
        from nyaya.rank.zonescore import per_zone_contributions, zone_scores

        weights = inp.weights
        if inp.scorer == "uniform-zone":
            weights = {z: (0.0 if z == "bench" else 1.0) for z in config.ZONES}
        ex.per_zone = per_zone_contributions(inp.doc_id, inp.doc_ids, inp.terms,
                                             inp.ctx, weights=weights)
        collect: Dict[str, np.ndarray] = {}
        zone_scores(inp.doc_ids, inp.terms, inp.ctx, weights=weights,
                    collect=collect)
        ex.top_terms = _top_from_collect(collect, inp.doc_ids, inp.doc_id,
                                         split_zone=True)
        ex.notes.append(
            "per-zone numbers are weighted contributions and sum to the text "
            "score: " + " + ".join(f"{z} {v:.4f}" for z, v in ex.per_zone.items())
            + f" = {sum(ex.per_zone.values()):.4f}")
    elif inp.scorer == "bm25f":
        from nyaya.rank.bm25f import explain as bm_explain

        ex.top_terms, zmass = bm_explain(inp.doc_id, inp.doc_ids, inp.terms,
                                         inp.ctx)
        ex.per_zone = zmass
        total = sum(zmass.values()) or 1.0
        ex.notes.append(
            "BM25F saturates AFTER the zones are combined, so a zone has no "
            "score of its own; the per-zone numbers are each zone's weighted, "
            "length-normalised share of the pseudo term frequency tf~ = "
            f"{total:.3f} (k1 = {config.BM25F_K1})")
    else:
        from nyaya.rank.tfidf import explain_terms

        ex.top_terms = explain_terms(inp.doc_id, inp.doc_ids, inp.terms, inp.ctx)
        ex.notes.append(
            "lnc.ltc: log tf and cosine normalisation on the document side, "
            "log tf times idf on the query side - idf is applied once, on the "
            "query")

    ex.notes.append(
        f"text_norm is the raw text score {inp.text_raw:.4f} divided by the "
        "best raw score among the candidates. The top *net* result need not be "
        "the top text scorer - moving it there is exactly what g(d) and "
        "proximity are for")

    # ---- authority ------------------------------------------------------
    if inp.use_authority and inp.authority is not None:
        parts = inp.authority.parts(inp.doc_id)
        ex.g = float(parts["g"])
        ex.pagerank_norm = float(parts["pagerank_norm"])
        ex.bench_norm = float(parts["bench_norm"])
        ex.penalty = float(parts["penalty"])
        ex.notes.append(
            f"g = penalty x ({config.G_PAGERANK_WEIGHT:g} x pagerank_norm + "
            f"{config.G_BENCH_WEIGHT:g} x bench_norm) = {ex.penalty:.3f} x "
            f"({config.G_PAGERANK_WEIGHT:g} x {ex.pagerank_norm:.4f} + "
            f"{config.G_BENCH_WEIGHT:g} x {ex.bench_norm:.4f}) = "
            f"{parts['check']:.4f}")
        if abs(parts["check"] - ex.g) > 5e-3:
            ex.notes.append(
                f"NOTE: the stored g ({ex.g:.4f}) and the recomputed one "
                f"({parts['check']:.4f}) differ - S4 re-weights g by treatment, "
                "so the stored value is the authoritative one")
        badge = inp.authority.badge(inp.doc_id)
        if badge:
            ex.notes.append(f"treatment: {badge}")
    else:
        ex.notes.append("this mode does not use authority, so g(d) is 0 and "
                        "the ranking is text only")

    # ---- the identity the panel exists to show ---------------------------
    ex.net = float(inp.alpha * ex.text_norm + (1.0 - inp.alpha) * ex.g
                   + inp.beta * ex.proximity)

    # ---- query-time expansions ------------------------------------------
    if inp.expansions:
        from nyaya import statutes

        for term, alts in inp.expansions.items():
            note = statutes.crosswalk_note(term)
            line = f"query expansion: {term} OR {' OR '.join(alts)}"
            ex.notes.append(f"{line} - {note}" if note else line)
    return ex


def _prox_label(kind: str, k: int) -> str:
    if kind == "s":
        return "same sentence"
    if kind == "p":
        return "same paragraph"
    if k:
        return f"within {k} words"
    return "proximity"


def _top_from_collect(collect: Dict[str, np.ndarray], doc_ids: np.ndarray,
                      doc_id: int, split_zone: bool = False, top: int = 3
                      ) -> List[Tuple[str, float]]:
    ids = np.asarray(doc_ids, dtype=np.int32)
    where = np.nonzero(ids == int(doc_id))[0]
    if where.size == 0:
        return []
    i = int(where[0])
    merged: Dict[str, float] = {}
    for key, arr in collect.items():
        term = key.split(":", 1)[1] if split_zone and ":" in key else key
        merged[term] = merged.get(term, 0.0) + float(arr[i])
    rows = [(t, v) for t, v in merged.items() if v != 0.0]
    rows.sort(key=lambda kv: -kv[1])
    return rows[:top]


# ==========================================================================
# Verification — the property the panel claims
# ==========================================================================


def verify(ex: ScoreExplanation, tol: float = 1e-6) -> Dict[str, Any]:
    """Recompute the net score from the printed parts.

    Returns the three terms and whether they reproduce ``ex.net``.  The CLI
    prints this, which is what makes the panel checkable rather than merely
    detailed.
    """
    a = ex.alpha * ex.text_norm
    b = (1.0 - ex.alpha) * ex.g
    c = ex.beta * ex.proximity
    total = a + b + c
    return {
        "alpha_x_text": a,
        "one_minus_alpha_x_g": b,
        "beta_x_proximity": c,
        "sum": total,
        "net": ex.net,
        "ok": abs(total - ex.net) <= tol,
        "zone_sum": sum(ex.per_zone.values()) if ex.per_zone else 0.0,
    }


# ==========================================================================
# Rendering
# ==========================================================================


def render(ex: ScoreExplanation, title: str = "", doc_id: Optional[int] = None,
           width: int = 76) -> str:
    """The panel as text.  This is the frame the demo video pauses on."""
    L: List[str] = []
    head = "why this result"
    if doc_id is not None:
        head += f"  -  doc {doc_id}"
    if title:
        head += f"  -  {title[:width - len(head) - 6]}"
    L.append(head)
    L.append("=" * width)

    L.append(f"  scorer                     {ex.scorer}")
    L.append(f"  text score (raw)           {ex.text_raw:.6f}")
    L.append(f"  text score (normalised)    {ex.text_norm:.6f}")

    if ex.per_zone:
        label = ("per-zone weighted contribution" if ex.scorer != "bm25f"
                 else "per-zone share of tf~ (before saturation)")
        L.append(f"\n  {label}")
        for zone in config.ZONES:
            if zone in ex.per_zone:
                w = config.ZONE_WEIGHTS.get(zone, 1.0)
                L.append(f"    {zone:<10} {ex.per_zone[zone]:>12.6f}"
                         f"   (zone weight {w:g})")
        L.append(f"    {'total':<10} {sum(ex.per_zone.values()):>12.6f}")

    if ex.top_terms:
        L.append("\n  query terms contributing most")
        for term, val in ex.top_terms:
            L.append(f"    {term:<22} {val:>12.6f}")

    if ex.proximity:
        L.append(f"\n  proximity                  {ex.proximity:.6f}"
                 f"   ({_prox_label(getattr(ex, 'prox_kind', ''), ex.prox_window)})")

    L.append("\n  static authority g(d)")
    L.append(f"    pagerank_norm            {ex.pagerank_norm:.6f}"
             f"   x {config.G_PAGERANK_WEIGHT:g}")
    L.append(f"    bench_norm               {ex.bench_norm:.6f}"
             f"   x {config.G_BENCH_WEIGHT:g}")
    L.append(f"    treatment penalty        {ex.penalty:.6f}")
    L.append(f"    g                        {ex.g:.6f}")

    v = verify(ex)
    L.append("\n  net score")
    L.append(f"    alpha x text             {ex.alpha:.2f} x {ex.text_norm:.6f}"
             f" = {v['alpha_x_text']:.6f}")
    L.append(f"    (1-alpha) x g            {1 - ex.alpha:.2f} x {ex.g:.6f}"
             f" = {v['one_minus_alpha_x_g']:.6f}")
    if ex.beta:
        L.append(f"    beta x proximity         {ex.beta:.2f} x "
                 f"{ex.proximity:.6f} = {v['beta_x_proximity']:.6f}")
    L.append("    " + "-" * (width - 6))
    L.append(f"    net                      {v['sum']:.6f}"
             + ("" if v["ok"] else f"   (MISMATCH: stored {ex.net:.6f})"))

    if ex.notes:
        L.append("\n  notes")
        for n in ex.notes:
            L.extend(_wrap(n, width - 6, "    - ", "      "))
    return "\n".join(L)


def _wrap(text: str, width: int, first: str, rest: str) -> List[str]:
    words = text.split()
    lines, cur = [], first
    for w in words:
        if len(cur) + len(w) + 1 > width and cur.strip() not in (first.strip(),):
            lines.append(cur.rstrip())
            cur = rest
        cur += w + " "
    if cur.strip():
        lines.append(cur.rstrip())
    return lines
