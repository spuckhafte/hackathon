"""Snippets: the evidence shown under each result  (Member 1, wave 3).

A snippet answers "why is this document here?" with the judgment's own words, so
three things have to be true of it and none of them is cosmetic.

**It must be a real paragraph.**  Snippets are chosen from the paragraphs of the
paragraph store, ranked by query-term density, not cut out of a character offset.
A judgment's paragraph is the unit a lawyer cites, and a snippet that straddles
two of them cannot be cited at all.

**Its gutter must not overclaim.**  Only 28% of this corpus printed its own
paragraph numbers; for the rest ``nyaya.zones`` invented ~110-word windows.
Member 3's ``ParagraphStore.ref()`` knows which is which, so the gutter reads
``¶14`` only when the court really numbered it and ``w14`` when we made the
window up.  Printing ``¶14`` for an invented window is the kind of thing a
legal-domain reader notices immediately, and it would be a lie.

**A statute must appear as the judgment wrote it.**  The index holds ``ipc§302``,
but the judgment says "Section 302 of the Indian Penal Code" and that is what
must be on screen.  ``statutes.iter_statute_spans()`` gives the character offsets
of the original wording, so a query for ``ipc§302`` highlights the real phrase
rather than a token that appears nowhere in the text.

Ranking a paragraph
-------------------
``density = distinct query terms present / sqrt(paragraph length in tokens)``

Distinct terms rather than total hits, so a paragraph that repeats one word does
not beat one that contains all of them; divided by the square root of the length
rather than the length, so a long paragraph that genuinely discusses everything
is not beaten by a three-word fragment containing one term.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.query.proximity import _index

#: Characters of context kept around the matched region of a long paragraph.
SNIPPET_CHARS = 320
#: Marker pair.  Plain text so it survives a Windows console in the demo; the
#: Streamlit page swaps them for <mark> tags.
HL_OPEN = "**"
HL_CLOSE = "**"
ELLIPSIS = " ... "

#: How many snippets a result card gets by default.
MAX_SNIPPETS = 2

#: Zones a snippet may be drawn from.  Title, parties and bench are metadata
#: shown elsewhere on the result card, and `statutes` is synthetic - it holds
#: canonical tokens that appear nowhere in the judgment's own text.
SNIPPET_ZONES = ("body", "order")
_SNIPPET_ZONE_IDS = np.asarray([config.ZONE_ID[z] for z in SNIPPET_ZONES])


@dataclass
class Snippet:
    """One piece of evidence, ready to render."""

    doc_id: int
    para_id: int
    zone: str
    text: str                      # with HL_OPEN/HL_CLOSE around matches
    gutter: str = ""               # "¶14" or "w14" - never a lie
    is_printed_para: bool = False
    density: float = 0.0
    terms: List[str] = field(default_factory=list)
    statutes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "para_id": self.para_id,
                "zone": self.zone, "gutter": self.gutter,
                "is_printed_para": self.is_printed_para,
                "density": round(self.density, 4), "terms": self.terms,
                "statutes": self.statutes, "text": self.text}

    def plain(self) -> str:
        return self.text.replace(HL_OPEN, "").replace(HL_CLOSE, "")


# ==========================================================================
# 1. Which paragraphs
# ==========================================================================

def _para_positions(idx, terms: Sequence[str], doc_id: int
                    ) -> Dict[int, Dict[str, List[int]]]:
    """``{para_id: {term: [positions]}}`` for one document.

    Taken from the index, not by re-reading text: the positions are already
    there, and ``para_ids_at`` turns them into paragraph ids with one vectorised
    searchsorted per term.
    """
    out: Dict[int, Dict[str, List[int]]] = {}
    for t in dict.fromkeys(terms):
        if not t:
            continue
        pl = idx.postings(t)
        if pl is None:
            continue
        pos = pl.positions_of(doc_id)
        if pos.size == 0:
            continue
        # Only body and order positions may nominate a snippet.  A statute token
        # lives in the synthetic `statutes` zone, whose positions sit near the
        # start of the document and are attributed to its first paragraph - so
        # without this filter a query for ipc302 proposes paragraph 1 of every
        # matching judgment, which holds counsel's names and nothing else.
        zones = idx.zone_at(doc_id, pos)
        keep = np.isin(zones, _SNIPPET_ZONE_IDS)
        if not keep.any():
            continue
        pos = pos[keep]
        paras = idx.para_ids_at(doc_id, pos)
        for p, q in zip(paras.tolist(), pos.tolist()):
            out.setdefault(int(p), {}).setdefault(t, []).append(int(q))
    return out


def rank_paragraphs(idx, terms: Sequence[str], doc_id: int,
                    k: int = MAX_SNIPPETS) -> List[Tuple[int, float, List[str]]]:
    """``[(para_id, density, terms_present), ...]``, densest first."""
    hits = _para_positions(idx, terms, doc_id)
    if not hits:
        return []
    n_paras = max(idx.n_paragraphs(doc_id), 1)
    # paragraph lengths, from the boundary array: one subtraction per paragraph
    starts = idx._para_flat[int(idx._para_off[doc_id]):int(idx._para_off[doc_id + 1])]
    doc_len = int(idx.doc_len[doc_id])
    lengths = np.diff(np.append(np.asarray(starts, dtype=np.int64), doc_len)) \
        if starts.size else np.asarray([doc_len])

    scored: List[Tuple[int, float, List[str]]] = []
    for para, per_term in hits.items():
        length = int(lengths[para]) if 0 <= para < lengths.size else doc_len
        density = len(per_term) / max(float(np.sqrt(max(length, 1))), 1e-9)
        scored.append((para, density, sorted(per_term)))
    scored.sort(key=lambda r: (-r[1], r[0]))
    return scored[:k]


# ==========================================================================
# 2. Highlighting
# ==========================================================================

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'­-]*")
_STEM_CACHE: Dict[str, str] = {}


def _stem_of(word: str) -> str:
    cached = _STEM_CACHE.get(word)
    if cached is None:
        from nyaya.analyze import analyze_terms

        got = analyze_terms(word)
        cached = got[0] if got else ""
        _STEM_CACHE[word] = cached
    return cached


def term_spans(text: str, terms: Sequence[str]) -> List[Tuple[int, int]]:
    """Character spans of the words whose **stem** is a query term.

    Not a prefix match on the stem, which is the obvious implementation and is
    wrong: Porter maps "privacy" to "privaci", and "privacy" does not start with
    "privaci", so a prefix rule highlights nothing at all on the single most
    important query in our evaluation set.  Stemming each word of the paragraph
    with the same ``analyze()`` the index used is exact by construction - the
    highlight then marks precisely the words that made the document match - and a
    110-word paragraph costs nothing to stem.

    Statute tokens are excluded here and handled by :func:`statute_spans_for`,
    because "ipc§302" occurs nowhere in the judgment's text.
    """
    wanted = {t for t in terms if t and "§" not in t}
    if not wanted:
        return []
    out: List[Tuple[int, int]] = []
    for m in _WORD_RE.finditer(text):
        if _stem_of(m.group(0)) in wanted:
            out.append((m.start(), m.end()))
    return out


def statute_spans_for(text: str, tokens: Sequence[str]
                      ) -> List[Tuple[int, int, str]]:
    """``(start, end, token)`` where ``text`` mentions one of ``tokens``.

    Uses Member 3's ``iter_statute_spans`` so the span covers the judgment's own
    wording - "Section 302 of the Indian Penal Code" - while the query matched the
    canonical ``ipc§302``.
    """
    wanted = {t for t in tokens if t and "§" in t}
    if not wanted:
        return []
    from nyaya import statutes as stat_mod

    out: List[Tuple[int, int, str]] = []
    for span in stat_mod.iter_statute_spans(text):
        hit = next((t for t in span.tokens if t in wanted), None)
        if hit:
            out.append((span.start, span.end, hit))
    return out


def _merge_spans(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ordered = sorted(spans)
    out: List[Tuple[int, int]] = []
    for a, b in ordered:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def highlight(text: str, terms: Sequence[str]) -> Tuple[str, List[Tuple[int, int]],
                                                        List[str]]:
    """``(marked_text, spans_in_the_original, statute_tokens_shown)``."""
    spans: List[Tuple[int, int]] = list(term_spans(text, terms))
    shown: List[str] = []
    for a, b, tok in statute_spans_for(text, terms):
        spans.append((a, b))
        if tok not in shown:
            shown.append(tok)
    merged = _merge_spans(spans)
    out: List[str] = []
    cursor = 0
    for a, b in merged:
        out.append(text[cursor:a])
        out.append(HL_OPEN + text[a:b] + HL_CLOSE)
        cursor = b
    out.append(text[cursor:])
    return "".join(out), merged, shown


def _window_around(text: str, spans: Sequence[Tuple[int, int]],
                   width: int = SNIPPET_CHARS) -> Tuple[int, int]:
    """Character window covering as many matches as possible, cut at word edges."""
    if not spans:
        return 0, min(len(text), width)
    first, last = spans[0][0], spans[-1][1]
    if last - first <= width:
        pad = (width - (last - first)) // 2
        start, end = max(0, first - pad), min(len(text), last + pad)
    else:
        start, end = first, min(len(text), first + width)
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return start, end


# ==========================================================================
# 3. The public entry point
# ==========================================================================

def snippets_for(doc_id: int, terms: Sequence[str], ctx: Any,
                 k: int = MAX_SNIPPETS, paras: Any = None,
                 width: int = SNIPPET_CHARS) -> List[Snippet]:
    """Up to ``k`` snippets for one document, densest paragraph first.

    ``paras`` is Member 3's ``ParagraphStore``; it is taken from ``ctx`` when not
    given, and when there is no store at all the snippet still renders - with an
    honest ``w<id>`` gutter - so a demo never dies for want of store S1.
    """
    idx = _index(ctx)
    store = paras if paras is not None else getattr(ctx, "paras", None)
    ranked = rank_paragraphs(idx, terms, int(doc_id), k=k)
    out: List[Snippet] = []
    for para_id, density, present in ranked:
        text, zone, gutter, printed = _paragraph_text(store, int(doc_id), para_id)
        if not text:
            continue
        marked, spans, statutes = highlight(text, terms)
        if len(text) > width and spans:
            a, b = _window_around(text, spans, width)
            clipped, _s, statutes = highlight(text[a:b], terms)
            prefix = ELLIPSIS.lstrip() if a > 0 else ""
            suffix = ELLIPSIS.rstrip() if b < len(text) else ""
            marked = f"{prefix}{clipped}{suffix}"
        out.append(Snippet(doc_id=int(doc_id), para_id=para_id, zone=zone,
                           text=marked, gutter=gutter, is_printed_para=printed,
                           density=density, terms=present, statutes=statutes))
    return out


def _paragraph_text(store: Any, doc_id: int, para_id: int
                    ) -> Tuple[str, str, str, bool]:
    """``(text, zone, gutter, is_printed)`` from the paragraph store if present.

    The gutter comes from ``ParagraphStore.ref().short()``, which is the whole
    point: it prints ``¶14`` only for a paragraph the court numbered and ``w14``
    for one this project invented.
    """
    if store is None:
        return "", "", f"w{para_id}", False
    try:
        para = store.get(doc_id, para_id)
    except Exception:
        para = None
    if para is None:
        return "", "", f"w{para_id}", False
    gutter = f"w{para_id}"
    printed = False
    try:
        ref = store.ref_for(doc_id, para)
        gutter = ref.short()
        printed = bool(ref.is_printed)
    except Exception:
        pass
    return para.text, getattr(para, "zone", ""), gutter, printed


def render(snips: Sequence[Snippet]) -> str:
    """Plain-text rendering for the CLI demo."""
    lines: List[str] = []
    for s in snips:
        tag = s.gutter or f"w{s.para_id}"
        extra = f"  [{', '.join(s.statutes)}]" if s.statutes else ""
        lines.append(f"  {tag:>8} ({s.zone}, density {s.density:.3f}){extra}")
        lines.append(f"           {s.text}")
    return "\n".join(lines)
