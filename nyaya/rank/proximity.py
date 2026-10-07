"""Proximity scoring: the smallest window containing the query  (M1, wave 3).

The feature ``config.BETA`` weights in the net score.  Definition, following
Manning et al. 7.2.2:

    omega  = the width, in tokens, of the smallest window of the document that
             contains **every distinct query term present in that document**
    prox   = min(1, |q| / omega)

so a document where the query words sit side by side scores 1.0 and one where
they are scattered over two pages scores near zero.  ``|q|`` is the number of
distinct query terms actually found, not the number typed: otherwise a document
matching three of five terms would be punished twice, once by the text scorer
that already saw three matches and again here.

The legal refinement, and why it matters
----------------------------------------
Two query words in the same *paragraph* mean something in a judgment: the court
was discussing them together.  Two words in different paragraphs usually mean the
document merely contains both.  So the window is measured **inside a single
paragraph** when the terms co-occur in one, and when they do not the plain
document-wide window is kept but multiplied by
:data:`CROSS_PARA_DISCOUNT`, because co-occurrence across a paragraph break is
weaker evidence of the same discussion.

Note what this is *not*: taking "the better of the global and the in-paragraph
window" would be a no-op, because a paragraph-restricted window can never be
narrower than the global one.  The refinement has to change the score in the
*other* direction - discounting scattered matches - to do any work at all.
``per_paragraph=False`` gives the plain Manning definition, which is the
ablation.

The algorithm is the standard sliding window over the **merged** position lists:
walk the merged list once, keeping a count per term in the current window, and
shrink from the left whenever every term is present.  Linear in the total number
of postings for the query in that document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.query.proximity import _index

#: Windows wider than this are treated as "not proximate at all".  Without a cap
#: a 480,000-token volume-sized document (the corpus has 18 of them) can always
#: produce *some* window containing every term, and omega would be meaningless.
MAX_WINDOW = 2000

#: Applied when the query terms never co-occur inside one paragraph.  Half,
#: because such a document is genuinely weaker evidence than one where the court
#: discussed the terms in the same breath, but it is not irrelevant - it still
#: contains every term.  One number, in one place, so the ablation is a flag.
CROSS_PARA_DISCOUNT = 0.5


@dataclass
class ProximityScore:
    """The proximity feature, with the evidence for the explanation panel."""

    score: float = 0.0
    omega: int = 0                  # width of the best window, in tokens
    n_terms: int = 0                # distinct query terms found in the document
    start: int = -1                 # first position of the best window
    end: int = -1
    within_paragraph: bool = False
    para_id: int = -1
    terms_found: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "proximity": round(self.score, 4), "window": self.omega,
            "terms_found": self.n_terms, "span": [self.start, self.end],
            "within_paragraph": self.within_paragraph, "para_id": self.para_id,
        }


def _merged_postings(idx, terms: Sequence[str], doc_id: int
                     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """``(positions, term_ids, terms_found)`` for one document, position-sorted.

    One array of positions and a parallel array saying which query term each
    position belongs to - the shape the sliding window wants.
    """
    pos_chunks: List[np.ndarray] = []
    id_chunks: List[np.ndarray] = []
    found: List[str] = []
    for t in dict.fromkeys(terms):          # distinct, order preserved
        if not t:
            continue
        pl = idx.postings(t)
        if pl is None:
            continue
        p = pl.positions_of(doc_id)
        if p.size == 0:
            continue
        pos_chunks.append(np.asarray(p, dtype=np.int64))
        id_chunks.append(np.full(p.size, len(found), dtype=np.int64))
        found.append(t)
    if not pos_chunks:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), [])
    pos = np.concatenate(pos_chunks)
    tid = np.concatenate(id_chunks)
    order = np.argsort(pos, kind="stable")
    return pos[order], tid[order], found


def smallest_window(positions: np.ndarray, term_ids: np.ndarray,
                    n_terms: int) -> Tuple[int, int, int]:
    """``(omega, start, end)`` of the narrowest span covering all ``n_terms``.

    Textbook two-pointer sweep: extend the right edge, and whenever the window
    holds every term, record it and advance the left edge.  O(number of
    positions), one pass, no allocation inside the loop.
    """
    if n_terms == 0 or positions.size == 0:
        return 0, -1, -1
    counts = [0] * n_terms
    have = 0
    best = (1 << 30, -1, -1)
    left = 0
    for right in range(positions.size):
        t = int(term_ids[right])
        counts[t] += 1
        if counts[t] == 1:
            have += 1
        while have == n_terms:
            width = int(positions[right]) - int(positions[left]) + 1
            if width < best[0]:
                best = (width, int(positions[left]), int(positions[right]))
            lt = int(term_ids[left])
            counts[lt] -= 1
            if counts[lt] == 0:
                have -= 1
            left += 1
    if best[1] < 0:
        return 0, -1, -1
    return best


def proximity_score(terms: Sequence[str], doc_id: int, ctx: Any,
                    per_paragraph: bool = True) -> ProximityScore:
    """``min(1, |q| / omega)`` for one document, and the window that produced it."""
    idx = _index(ctx)
    pos, tid, found = _merged_postings(idx, terms, int(doc_id))
    n = len(found)
    if n == 0:
        return ProximityScore()
    if n == 1:
        # A single term is trivially "proximate to itself"; scoring it 1.0 would
        # let any one-word match top the proximity feature, so it scores 0 and
        # the text relevance decides, which is the honest reading of the feature.
        return ProximityScore(score=0.0, omega=1, n_terms=1,
                              start=int(pos[0]), end=int(pos[0]),
                              terms_found=found)

    omega, start, end = smallest_window(pos, tid, n)
    if not per_paragraph:
        return ProximityScore(score=_score_from(omega, n), omega=omega,
                              n_terms=n, start=start, end=end,
                              terms_found=found)

    p_omega, p_start, p_end, para = _best_within_paragraph(
        idx, int(doc_id), pos, tid, n)
    if p_omega:
        return ProximityScore(score=_score_from(p_omega, n), omega=p_omega,
                              n_terms=n, start=p_start, end=p_end,
                              within_paragraph=True, para_id=para,
                              terms_found=found)
    return ProximityScore(score=_score_from(omega, n) * CROSS_PARA_DISCOUNT,
                          omega=omega, n_terms=n, start=start, end=end,
                          within_paragraph=False, terms_found=found)


def _score_from(omega: int, n_terms: int) -> float:
    if omega <= 0 or omega > MAX_WINDOW:
        return 0.0
    return min(1.0, n_terms / float(omega))


def _best_within_paragraph(idx, doc_id: int, pos: np.ndarray, tid: np.ndarray,
                           n_terms: int) -> Tuple[int, int, int, int]:
    """The narrowest all-terms window that lies inside one paragraph."""
    paras = idx.para_ids_at(doc_id, pos)
    best = (0, -1, -1, -1)
    for para in np.unique(paras):
        mask = paras == para
        sub_pos = pos[mask]
        sub_tid = tid[mask]
        if np.unique(sub_tid).size < n_terms:
            continue
        omega, start, end = smallest_window(sub_pos, sub_tid, n_terms)
        if omega and (best[0] == 0 or omega < best[0]):
            best = (omega, start, end, int(para))
    return best


def proximity_scores(terms: Sequence[str], doc_ids: Sequence[int], ctx: Any,
                     per_paragraph: bool = True) -> np.ndarray:
    """The feature for a whole candidate list, in ``doc_ids`` order."""
    out = np.zeros(len(doc_ids), dtype=np.float32)
    for i, d in enumerate(doc_ids):
        out[i] = proximity_score(terms, int(d), ctx,
                                 per_paragraph=per_paragraph).score
    return out


def explain(terms: Sequence[str], doc_id: int, ctx: Any,
            per_paragraph: bool = True) -> str:
    """One printable line, for the demo and the "why this result" panel."""
    s = proximity_score(terms, doc_id, ctx, per_paragraph=per_paragraph)
    if s.n_terms == 0:
        return f"doc {doc_id}: no query term present"
    if s.within_paragraph:
        where = f"within paragraph {s.para_id}"
        formula = f"min(1, {s.n_terms}/{s.omega})"
    else:
        where = f"across paragraphs (x{CROSS_PARA_DISCOUNT} discount)"
        formula = f"min(1, {s.n_terms}/{s.omega}) * {CROSS_PARA_DISCOUNT}"
    return (f"doc {doc_id}: {s.n_terms} of {len(set(terms))} terms, smallest "
            f"window {s.omega} tokens at [{s.start}, {s.end}] {where} -> "
            f"prox = {formula} = {s.score:.4f}  (beta = {config.BETA})")
