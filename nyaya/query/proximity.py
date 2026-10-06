"""Positional operators: phrase, /n, /s, /p  (Member 1, wave 3).

Everything here is the textbook positional intersection (Manning et al. 2.4.2):
intersect the two terms' *document* lists first, then walk the two sorted
*position* lists of each surviving document with two pointers.  Both halves are
linear in their input - no nested loop over positions, no re-reading of text -
which is the whole reason the index stores positions at all.

The four operators differ only in the predicate applied to a candidate pair of
positions ``(i, j)``:

    phrase    j == i + 1                 consecutive terms, in order
    /n k      abs(j - i) <= k            within k tokens, either order
    /s        sent_id(i) == sent_id(j)   same sentence
    /p        para_id(i) == para_id(j)   same paragraph

**Zones.**  ``nyaya.index.positional`` concatenates the six zones into one
position space with no gap between them, so position ``len(title) - 1`` and
position ``len(title)`` are adjacent integers while belonging to different
zones.  Every operator here therefore rejects a match whose two positions fall
in different zones, using ``index.zone_at``.  Without that, the phrase
``"bihar union"`` would match the seam between a title ending in "Bihar" and a
parties zone beginning with "Union" - a match that exists nowhere in the
judgment.  ``/s`` gets this for free (sentence ids already restart per zone) but
the check is applied uniformly so the rule is one line, not four.

**What ``/p`` means.**  On 72% of the corpus a paragraph is a ~110-word window
invented by ``nyaya.zones`` because PyMuPDF returns no blank lines for these
PDFs; on the other 28% it is the paragraph the court itself numbered.  So ``/p``
is "within the same ~110-word window" more often than "within the same numbered
paragraph".  Stated here because it changes how a user should read the operator.

Correctness evidence
--------------------
``tests/test_proximity.py`` re-derives every answer with a regular expression
over the raw judgment text, independently of the index, and asserts the two
agree.  ``scripts/prox_check.py`` runs the same comparison over the real corpus
and prints the agreement rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config

DT = np.int32

#: Operator kinds, as ``contracts.ProxNode.kind`` spells them.
KIND_WINDOW = "n"
KIND_SENTENCE = "s"
KIND_PARAGRAPH = "p"


# ==========================================================================
# 0. Getting at the index
# ==========================================================================

def _index(ctx: Any):
    """Accept a ``SearchContext``, a bare ``PositionalIndex``, or anything with
    ``.index``.  Member 3 owns ``SearchContext``; this module must not care."""
    idx = getattr(ctx, "index", None)
    return ctx if idx is None else idx


def _zone_filter(zone: Optional[str]) -> Optional[int]:
    if zone is None:
        return None
    if zone not in config.ZONE_ID:
        raise ValueError(f"unknown zone {zone!r}; expected one of {config.ZONES}")
    return config.ZONE_ID[zone]


# ==========================================================================
# 1. Document-list intersection, skip-aware
# ==========================================================================

def intersect_doc_ids(lists: Sequence[np.ndarray]) -> np.ndarray:
    """Intersect sorted int32 doc-id arrays, smallest first.

    Ordering by length is the standard optimisation: the result can never be
    larger than the shortest list, so starting there keeps every later
    intersection as small as possible.
    """
    if not lists:
        return np.empty(0, dtype=DT)
    order = sorted(range(len(lists)), key=lambda i: len(lists[i]))
    out = lists[order[0]]
    for i in order[1:]:
        if out.size == 0:
            break
        out = np.intersect1d(out, lists[i], assume_unique=True)
    return out.astype(DT, copy=False)


def _postings_for(idx, terms: Sequence[str]) -> Optional[List[Any]]:
    """All terms' postings lists, or None if any term is absent.

    A missing term makes every positional operator empty by definition - a
    phrase cannot occur if one of its words occurs nowhere - so this returns
    None rather than an empty list to keep that case explicit at the call site.
    """
    out = []
    for t in terms:
        pl = idx.postings(t)
        if pl is None:
            return None
        out.append(pl)
    return out


# ==========================================================================
# 2. The two-pointer walks
# ==========================================================================

def phrase_hits(positions: Sequence[np.ndarray]) -> np.ndarray:
    """Start positions where ``positions[0], positions[1], ...`` are consecutive.

    The classic positional merge, generalised to k terms: hold the candidate set
    of start offsets and advance it one term at a time.  Implemented with
    ``np.intersect1d`` on shifted arrays, which is the same O(sum of list
    lengths) merge the two-pointer loop performs, done in numpy.
    """
    if not positions:
        return np.empty(0, dtype=DT)
    starts = np.asarray(positions[0], dtype=np.int64)
    for k in range(1, len(positions)):
        if starts.size == 0:
            return np.empty(0, dtype=DT)
        nxt = np.asarray(positions[k], dtype=np.int64) - k
        starts = np.intersect1d(starts, nxt, assume_unique=True)
    return starts.astype(DT, copy=False)


def window_pairs(left: np.ndarray, right: np.ndarray, k: int,
                 ordered: bool = False) -> List[Tuple[int, int]]:
    """Pairs ``(i, j)`` from two sorted position lists with ``|j - i| <= k``.

    A genuine two-pointer walk, because this is the one operator whose output is
    a *set of pairs* rather than a set of offsets: for each left position, scan
    forward over the right positions that are still inside the window, then
    retreat the right pointer to the first candidate for the next left position.
    Linear in ``len(left) + len(right) + len(result)``.
    """
    pairs: List[Tuple[int, int]] = []
    if left.size == 0 or right.size == 0 or k < 0:
        return pairs
    j0 = 0
    nr = int(right.size)
    for i in map(int, left):
        while j0 < nr and int(right[j0]) < i - k:
            j0 += 1
        j = j0
        while j < nr:
            rj = int(right[j])
            if rj > i + k:
                break
            if rj != i and (not ordered or rj > i):
                pairs.append((i, rj))
            j += 1
    return pairs


def _same_bucket_pairs(idx, doc_id: int, left: np.ndarray, right: np.ndarray,
                       kind: str) -> List[Tuple[int, int]]:
    """Pairs sharing a sentence (``/s``) or a paragraph (``/p``).

    One vectorised ``searchsorted`` per side turns the position arrays into
    bucket ids, after which this is a merge over two sorted id arrays - the same
    shape as the window walk, not a cross product.
    """
    if left.size == 0 or right.size == 0:
        return []
    if kind == KIND_SENTENCE:
        lb = idx.sent_ids_at(doc_id, left)
        rb = idx.sent_ids_at(doc_id, right)
    else:
        lb = idx.para_ids_at(doc_id, left)
        rb = idx.para_ids_at(doc_id, right)
    pairs: List[Tuple[int, int]] = []
    ri = 0
    nr = int(right.size)
    for a in range(int(left.size)):
        b = int(lb[a])
        while ri < nr and int(rb[ri]) < b:
            ri += 1
        rj = ri
        while rj < nr and int(rb[rj]) == b:
            if int(right[rj]) != int(left[a]):
                pairs.append((int(left[a]), int(right[rj])))
            rj += 1
    return pairs


# ==========================================================================
# 3. The two signatures frozen in docs/CONTRACTS.md Section 4
# ==========================================================================

def phrase_docs(terms: Sequence[str], ctx: Any,
                zone: Optional[str] = None) -> np.ndarray:
    """Documents containing ``terms`` as a consecutive phrase, ascending ids.

    ``zone`` restricts the match to one zone; ``None`` means any single zone -
    never *across* two, which is the cross-zone rule in the module docstring.
    """
    idx = _index(ctx)
    terms = [t for t in terms if t]
    if not terms:
        return np.empty(0, dtype=DT)
    if len(terms) == 1:
        return term_docs(terms[0], ctx, zone=zone)
    pls = _postings_for(idx, terms)
    if pls is None:
        return np.empty(0, dtype=DT)
    want = _zone_filter(zone)
    cands = intersect_doc_ids([pl.doc_ids for pl in pls])
    out: List[int] = []
    for d in map(int, cands):
        starts = phrase_hits([pl.positions_of(d) for pl in pls])
        if starts.size == 0:
            continue
        if _any_start_in_one_zone(idx, d, starts, len(terms), want):
            out.append(d)
    return np.asarray(out, dtype=DT)


def _any_start_in_one_zone(idx, doc_id: int, starts: np.ndarray, n_terms: int,
                           want: Optional[int]) -> bool:
    """Is at least one phrase occurrence wholly inside a single zone?"""
    z_first = idx.zone_at(doc_id, starts)
    z_last = idx.zone_at(doc_id, starts + (n_terms - 1))
    same = z_first == z_last
    if want is not None:
        same &= z_first == want
    return bool(same.any())


def proximity_docs(left_ids: Optional[np.ndarray], right_ids: Optional[np.ndarray],
                   left_terms: Sequence[str], right_terms: Sequence[str],
                   kind: str, k: int, ctx: Any,
                   zone: Optional[str] = None) -> np.ndarray:
    """Documents where a left term and a right term satisfy the ``kind`` predicate.

    ``left_ids`` / ``right_ids`` are the candidate document sets the planner
    already computed for the two sides (``None`` means "not narrowed"); they are
    only used to shrink the candidate list, never to decide a match, so the
    planner can pass whatever it has.

    ``left_terms`` / ``right_terms`` may each hold several terms - that is how an
    expanded root (``bail!``) or an OR group participates in a proximity: a
    document matches if *any* left term and *any* right term satisfy the
    predicate, so the position lists of each side are merged first.
    """
    idx = _index(ctx)
    left_terms = [t for t in left_terms if t]
    right_terms = [t for t in right_terms if t]
    if not left_terms or not right_terms:
        return np.empty(0, dtype=DT)
    if kind not in (KIND_WINDOW, KIND_SENTENCE, KIND_PARAGRAPH):
        raise ValueError(f"unknown proximity kind {kind!r}")

    lp = [pl for pl in (idx.postings(t) for t in left_terms) if pl is not None]
    rp = [pl for pl in (idx.postings(t) for t in right_terms) if pl is not None]
    if not lp or not rp:
        return np.empty(0, dtype=DT)

    cands = intersect_doc_ids([
        _union_doc_ids([pl.doc_ids for pl in lp]),
        _union_doc_ids([pl.doc_ids for pl in rp]),
    ])
    for narrowing in (left_ids, right_ids):
        if narrowing is not None and len(narrowing):
            cands = np.intersect1d(cands, np.asarray(narrowing, dtype=DT),
                                   assume_unique=True)
    want = _zone_filter(zone)

    out: List[int] = []
    for d in map(int, cands):
        left = _merge_positions([pl.positions_of(d) for pl in lp])
        right = _merge_positions([pl.positions_of(d) for pl in rp])
        if left.size == 0 or right.size == 0:
            continue
        if kind == KIND_WINDOW:
            pairs = window_pairs(left, right, k)
        else:
            pairs = _same_bucket_pairs(idx, d, left, right, kind)
        if not pairs:
            continue
        if _any_pair_in_one_zone(idx, d, pairs, want):
            out.append(d)
    return np.asarray(out, dtype=DT)


def _any_pair_in_one_zone(idx, doc_id: int, pairs: Sequence[Tuple[int, int]],
                          want: Optional[int]) -> bool:
    a = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    b = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    za = idx.zone_at(doc_id, a)
    zb = idx.zone_at(doc_id, b)
    ok = za == zb
    if want is not None:
        ok &= za == want
    return bool(ok.any())


def _union_doc_ids(lists: Sequence[np.ndarray]) -> np.ndarray:
    if len(lists) == 1:
        return np.asarray(lists[0], dtype=DT)
    return np.unique(np.concatenate([np.asarray(x, dtype=DT) for x in lists]))


def _merge_positions(arrays: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(a, dtype=np.int64) for a in arrays if len(a)]
    if not arrays:
        return np.empty(0, dtype=np.int64)
    if len(arrays) == 1:
        return arrays[0]
    return np.unique(np.concatenate(arrays))


# ==========================================================================
# 4. Helpers the planner and the scorers also want
# ==========================================================================

def term_docs(term: str, ctx: Any, zone: Optional[str] = None) -> np.ndarray:
    """One term's documents, optionally restricted to a zone.

    The zone restriction is answered from the stored per-zone term frequencies,
    not from positions - that is what the zoned index is for.
    """
    idx = _index(ctx)
    pl = idx.postings(term)
    if pl is None:
        return np.empty(0, dtype=DT)
    if zone is None:
        return np.asarray(pl.doc_ids, dtype=DT)
    zid = _zone_filter(zone)
    col = pl.zone_tf_matrix()[:, zid]
    return np.asarray(pl.doc_ids, dtype=DT)[col > 0]


def phrase_occurrences(terms: Sequence[str], doc_id: int, ctx: Any,
                       zone: Optional[str] = None) -> np.ndarray:
    """Start positions of every in-zone occurrence of the phrase in one document.

    Used by the snippet generator, which needs the *places*, not just the fact
    of a match.
    """
    idx = _index(ctx)
    pls = _postings_for(idx, [t for t in terms if t])
    if pls is None:
        return np.empty(0, dtype=DT)
    starts = phrase_hits([pl.positions_of(doc_id) for pl in pls])
    if starts.size == 0:
        return starts
    want = _zone_filter(zone)
    z_first = idx.zone_at(doc_id, starts)
    z_last = idx.zone_at(doc_id, starts + (len(pls) - 1))
    keep = z_first == z_last
    if want is not None:
        keep &= z_first == want
    return starts[keep]


@dataclass
class ProxExplain:
    """What a proximity match looked like, for the "why this result" panel."""

    doc_id: int
    kind: str
    k: int
    n_pairs: int
    best_gap: int
    sample: List[Tuple[int, int]]


def explain_proximity(left_terms: Sequence[str], right_terms: Sequence[str],
                      doc_id: int, kind: str, k: int, ctx: Any,
                      sample: int = 5) -> Optional[ProxExplain]:
    """The pairs that made one document match - printable, for the demo."""
    idx = _index(ctx)
    lp = [pl for pl in (idx.postings(t) for t in left_terms) if pl is not None]
    rp = [pl for pl in (idx.postings(t) for t in right_terms) if pl is not None]
    if not lp or not rp:
        return None
    left = _merge_positions([pl.positions_of(doc_id) for pl in lp])
    right = _merge_positions([pl.positions_of(doc_id) for pl in rp])
    if left.size == 0 or right.size == 0:
        return None
    pairs = (window_pairs(left, right, k) if kind == KIND_WINDOW
             else _same_bucket_pairs(idx, doc_id, left, right, kind))
    if not pairs:
        return None
    gaps = [abs(b - a) for a, b in pairs]
    return ProxExplain(doc_id=doc_id, kind=kind, k=k, n_pairs=len(pairs),
                       best_gap=min(gaps), sample=pairs[:sample])
