"""The Precedent Brief (feature F7, guide Section 9.4).

    query -> search() -> top 5 judgments
          -> BM25 over their paragraphs, zone-weighted  (evidence.py)
          -> <= 8 paragraphs, <= 2 per judgment, each with an honest citation
          -> compose: extractive by default, abstractive only with a key (compose.py)
          -> tf-idf cosine grounding check on every sentence (verify.py)

Three design commitments, in the order they matter:

1. **A cited paragraph is cited honestly.**  The citation comes from
   `ParagraphStore.ref_for`, so a judgment that prints no paragraph numbers is
   cited as ``window 14 of 32`` and *never* as ``¶ 14``.  Only 28% of this
   corpus prints its own numbering; a generated legal brief that invents a
   paragraph number is the worst failure this system could have.
2. **The extractive path is the real one.**  There is no model API key in this
   environment and the `anthropic` SDK is not installed, so the abstractive
   composer has never run.  The extractive composer quotes real sentences, so
   it cannot hallucinate and cannot break the demo; `Brief.composer` says which
   one produced the text and `Brief.notes` says why.
3. **Every sentence is checked, whoever wrote it.**  The verifier is not a
   formality reserved for a generator we do not trust - it runs on the
   extractive output too, where it passes tautologically, and that fact is
   reported rather than presented as a result.

Demo: ``python -m nyaya.cli --brief "dying declaration as the sole basis of
conviction"``

Author: Sumedha (M3)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from nyaya import config
from nyaya.brief.evidence import EvidencePack, EvidenceParagraph, build_pack
from nyaya.brief.verify import VerificationReport

__all__ = ["Brief", "build", "render"]


@dataclass
class Brief:
    query: str
    text: str = ""
    composer: str = "extractive"
    pack: Optional[EvidencePack] = None
    verification: Optional[VerificationReport] = None
    results: List[Any] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def grounded(self) -> bool:
        return bool(self.verification and self.verification.passed)

    def as_json(self) -> Dict[str, Any]:
        return {
            "query": self.query, "text": self.text, "composer": self.composer,
            "grounded": self.grounded, "notes": self.notes,
            "elapsed_ms": self.elapsed_ms,
            "pack": self.pack.as_json() if self.pack else None,
            "verification": (self.verification.as_json()
                             if self.verification else None),
        }


def build(query: str, k: int = config.BRIEF_TOP_DOCS, mode: str = "S4",
          ctx=None, use_llm: bool = True,
          max_paras: int = config.BRIEF_MAX_PARAS,
          max_per_doc: int = config.BRIEF_MAX_PARAS_PER_DOC) -> Brief:
    """Retrieve, select evidence, compose and verify.  Never raises."""
    from nyaya.brief.compose import compose_extractive, compose_llm, llm_available
    from nyaya.brief.verify import verify
    from nyaya.query.parser import parse
    from nyaya.search import load_context, search

    t0 = time.perf_counter()
    b = Brief(query=query)
    ctx = ctx or load_context(mode)
    q = parse(query)
    terms = list(q.free_terms) or list(q.scoring_terms)

    b.results = search(query, k=k, mode=mode, explain=False, ctx=ctx)
    if not b.results:
        b.notes.append("nothing was retrieved, so there is no brief to write")
        b.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return b

    b.pack = build_pack(query, b.results, terms, ctx, max_paras, max_per_doc)
    b.notes.extend(b.pack.notes)
    if not b.pack.paragraphs:
        b.notes.append("no paragraph in the retrieved judgments matched the "
                       "query well enough to cite")
        b.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return b

    text, note = ("", "")
    if use_llm:
        text, note = compose_llm(b.pack)
    if text.strip():
        b.composer = "abstractive"
        b.notes.append(f"written by {note}")
    else:
        b.composer = "extractive"
        ok, why = llm_available()
        b.notes.append(note if (use_llm and note and ok) else why)
        text = compose_extractive(b.pack, terms, ctx)
    b.text = text.strip()

    b.verification = verify(b.text, b.pack, ctx, composer=b.composer)
    b.notes.extend(b.verification.notes)
    b.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return b


def render(b: Brief, width: int = 78) -> str:
    """The whole feature as text: brief, sources, grounding check."""
    from nyaya.brief.verify import render as render_check

    bar = "=" * width
    L = [bar, f"PRECEDENT BRIEF  -  {b.query}", bar, ""]
    if not b.text:
        L.append("  (no brief: " + "; ".join(b.notes) + ")")
        return "\n".join(L)
    L.append(_wrap(b.text, width - 2, "  "))
    L.append("")
    L.append("sources")
    L.append("-" * width)
    for p in (b.pack.paragraphs if b.pack else []):
        L.append(f"  {p.label()}")
    L.append("")
    if b.verification is not None:
        L.append(render_check(b.verification, width))
    if b.notes:
        L.append("\nnotes")
        for n in dict.fromkeys(b.notes):
            L.append(_wrap(f"- {n}", width - 4, "  ", "    "))
    L.append(f"\n  {b.elapsed_ms:.0f} ms  ·  composer: {b.composer}  ·  "
             f"grounded: {b.grounded}")
    return "\n".join(L)


def _wrap(text: str, width: int, indent: str = "", cont: str = None) -> str:
    cont = cont if cont is not None else indent
    words, lines, cur = text.split(), [], indent
    for w in words:
        if len(cur) + len(w) + 1 > width and cur.strip():
            lines.append(cur.rstrip())
            cur = cont
        cur += w + " "
    if cur.strip():
        lines.append(cur.rstrip())
    return "\n".join(lines)
