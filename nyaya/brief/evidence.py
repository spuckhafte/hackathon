"""Selecting the paragraphs a Precedent Brief is allowed to use (guide 9.4).

The pack is the contract between retrieval and composition: whatever writes the
brief — a language model or the extractive composer — may use **only** these
paragraphs, and must cite one of them for every sentence.  So the selection has
to be good, and each paragraph has to arrive carrying everything a reader needs
to check it: a citable id, an honest paragraph reference, the case, and the
case's treatment badge.

Why the badge travels with the paragraph
----------------------------------------
A brief that quotes a paragraph from an overruled judgment without saying so is
worse than no brief.  The badge is attached to the *evidence*, not just to the
result list, so it cannot be lost on the way into the summary.

Scoring
-------
BM25 over the paragraphs **inside the top-k retrieved judgments** — a second,
finer-grained retrieval pass over a candidate set of a few hundred paragraphs
rather than the whole corpus:

    score(p) = Σ_t  idf(t) · tf_p(t)·(k1+1)
                    ------------------------------------------
                    tf_p(t) + k1·(1 − b + b·len_p/avg_len_p)
               × zone_weight(p.zone)

idf comes from the **document-level** index, which is the right collection
statistic and avoids a second pass over 437,654 paragraphs.  The zone weight is
why a query about outcomes surfaces order-zone paragraphs: `config.ZONE_WEIGHTS`
gives `order` 1.5 against `body` 1.0.

Author: Sumedha (M3)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyaya import config

__all__ = ["EvidenceParagraph", "EvidencePack", "select_paragraphs", "build_pack"]

#: BM25 parameters for the paragraph pass.  `b` is lower than the body's 0.75:
#: paragraphs are already short and of similar length, so aggressive length
#: normalisation mostly amplifies noise.
PARA_K1 = 1.2
PARA_B = 0.5


@dataclass
class EvidenceParagraph:
    """One paragraph the brief may cite."""

    pid: str                  # "P1" - what a sentence cites
    doc_id: int
    para_id: int
    zone: str
    text: str
    score: float
    cite: str                 # "¶ 14" | "¶ 14 (cont.)" | "window 14 of 32"
    cite_kind: str            # 'printed' | 'printed-split' | 'window'
    title: str = ""
    year: int = 0
    neutral_cite: str = ""
    status: str = "ok"
    badge: str = ""

    def label(self) -> str:
        """How the brief's source list prints this paragraph."""
        head = f"[{self.pid}] {self.title[:60]} ({self.year}) {self.cite}"
        return head + (f"  !! {self.badge[:80]}" if self.badge else "")

    def as_json(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class EvidencePack:
    """Everything the composer is allowed to use, and nothing else."""

    query: str
    paragraphs: List[EvidenceParagraph] = field(default_factory=list)
    doc_ids: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def by_pid(self, pid: str) -> Optional[EvidenceParagraph]:
        for p in self.paragraphs:
            if p.pid == pid.strip().strip("[]").upper():
                return p
        return None

    @property
    def pids(self) -> List[str]:
        return [p.pid for p in self.paragraphs]

    def render_sources(self) -> str:
        return "\n".join(p.label() for p in self.paragraphs)

    def as_json(self) -> Dict[str, Any]:
        return {"query": self.query, "doc_ids": self.doc_ids,
                "notes": self.notes,
                "paragraphs": [p.as_json() for p in self.paragraphs]}


# ==========================================================================
# BM25 over paragraphs
# ==========================================================================


def _paragraph_scores(paras: Sequence[Tuple[int, Any]], terms: Sequence[str],
                      ctx, weights: Optional[Dict[str, float]] = None
                      ) -> List[float]:
    from nyaya.analyze import analyze_terms

    weights = weights or config.ZONE_WEIGHTS
    index = getattr(ctx, "index", None)
    idf = {t: (float(index.idf(t)) if index is not None else 1.0) for t in set(terms)}

    tokenised = [analyze_terms(p.text) for _, p in paras]
    lengths = [max(1, len(tk)) for tk in tokenised]
    avg_len = sum(lengths) / max(1, len(lengths))

    out: List[float] = []
    for (doc_id, p), toks, ln in zip(paras, tokenised, lengths):
        counts: Dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        s = 0.0
        for t in set(terms):
            tf = counts.get(t, 0)
            if not tf or idf[t] <= 0.0:
                continue
            denom = tf + PARA_K1 * (1.0 - PARA_B + PARA_B * ln / avg_len)
            s += idf[t] * tf * (PARA_K1 + 1.0) / denom
        out.append(s * float(weights.get(p.zone, 1.0)))
    return out


def select_paragraphs(doc_ids: Sequence[int], terms: Sequence[str], ctx,
                      max_paras: int = config.BRIEF_MAX_PARAS,
                      max_per_doc: int = config.BRIEF_MAX_PARAS_PER_DOC,
                      min_words: int = 25,
                      ) -> List[Tuple[int, Any, float]]:
    """The best ``max_paras`` paragraphs, at most ``max_per_doc`` per judgment.

    The per-document cap is what stops one long, repetitive judgment from
    filling the whole pack: a brief that cites five paragraphs of one case is a
    quotation, not a synthesis.
    """
    store = getattr(ctx, "paras", None)
    if store is None:
        return []
    candidates: List[Tuple[int, Any]] = []
    from nyaya.brief.compose import BOILERPLATE

    for d in doc_ids:
        for p in store.paras_of(int(d)):
            if len(p.text.split()) < min_words:
                continue
            if BOILERPLATE.search(p.text[:200]):
                continue      # the reporter's own index, not the Court's words
            candidates.append((int(d), p))
    if not candidates:
        return []

    scores = _paragraph_scores(candidates, terms, ctx)
    ranked = sorted(zip(candidates, scores), key=lambda kv: -kv[1])

    taken: Dict[int, int] = {}
    out: List[Tuple[int, Any, float]] = []
    for (doc_id, para), score in ranked:
        if score <= 0.0:
            break
        if taken.get(doc_id, 0) >= max_per_doc:
            continue
        taken[doc_id] = taken.get(doc_id, 0) + 1
        out.append((doc_id, para, float(score)))
        if len(out) >= max_paras:
            break
    return out


def build_pack(query: str, results: Sequence[Any], terms: Sequence[str], ctx,
               max_paras: int = config.BRIEF_MAX_PARAS,
               max_per_doc: int = config.BRIEF_MAX_PARAS_PER_DOC
               ) -> EvidencePack:
    """Turn ranked results into the pack the composer may cite from."""
    store = getattr(ctx, "paras", None)
    pack = EvidencePack(query=query)
    if store is None:
        pack.notes.append("the paragraph store (S1) is not built, so there is "
                          "no evidence to brief from")
        return pack

    by_doc = {int(r.doc_id): r for r in results}
    pack.doc_ids = list(by_doc)
    chosen = select_paragraphs(list(by_doc), terms, ctx, max_paras, max_per_doc)

    for i, (doc_id, para, score) in enumerate(chosen, 1):
        ref = store.ref_for(doc_id, para)
        r = by_doc.get(doc_id)
        pack.paragraphs.append(EvidenceParagraph(
            pid=f"P{i}", doc_id=doc_id, para_id=para.id, zone=para.zone,
            text=para.text.strip(), score=score,
            cite=ref.cite(), cite_kind=ref.kind,
            title=getattr(r, "title", ""), year=int(getattr(r, "year", 0) or 0),
            neutral_cite=getattr(r, "neutral_cite", "") or "",
            status=getattr(r, "status", "ok"),
            badge=getattr(r, "badge", "") or "",
        ))

    windows = sum(1 for p in pack.paragraphs if p.cite_kind == "window")
    if windows:
        pack.notes.append(
            f"{windows} of {len(pack.paragraphs)} cited passages come from "
            "judgments that print no paragraph numbers; they are cited as "
            "'window N of M' and must never be rendered as a paragraph number")
    flagged = [p for p in pack.paragraphs if p.badge]
    if flagged:
        pack.notes.append(
            f"{len(flagged)} cited judgment(s) carry a treatment warning: "
            + "; ".join(f"{p.pid} {p.badge[:70]}" for p in flagged))
    return pack
