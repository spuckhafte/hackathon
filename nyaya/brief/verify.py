"""The grounding check: does each sentence actually say what its paragraph says?

Why this exists
---------------
A summary that cites a paragraph is not grounded because it cites one.  The
check is cheap and it is the difference between "the model produced a citation"
and "the sentence is supported": build a tf-idf vector for the sentence and one
for the paragraph it cites, take the cosine, and flag anything below
`config.BRIEF_COSINE_THRESHOLD`.  A sentence with **no** citation is flagged
whatever it says, because an uncited sentence in a legal brief is an assertion.

Each flag carries the **closest** paragraph in the pack, so the failure is
actionable: either the sentence cited the wrong one of the eight (the closest is
shown, and usually it is right), or nothing in the pack supports it at all,
which is the serious case.

idf comes from the document-level index, so "privacy" is rare and "court" is
not — the same statistic the ranking used, rather than a second notion of
importance invented for this panel.

On the extractive composer the cosine is ~1.0 by construction, because the
sentence *is* the paragraph's own sentence.  That is a tautological pass and the
report says so: the verifier earns its keep against generated text, and
`tests/test_brief.py` exercises it with a deliberately unsupported sentence.

Author: Sumedha (M3)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyaya import config
from nyaya.brief.evidence import EvidencePack, EvidenceParagraph

__all__ = ["vector", "cosine", "VerifiedSentence", "VerificationReport", "verify"]


def vector(terms: Sequence[str], ctx) -> Dict[str, float]:
    """A unit-length tf-idf vector over already-analysed terms."""
    index = getattr(ctx, "index", None)
    tf: Dict[str, int] = {}
    for t in terms:
        tf[t] = tf.get(t, 0) + 1
    raw: Dict[str, float] = {}
    for t, n in tf.items():
        idf = float(index.idf(t)) if index is not None else 1.0
        if idf <= 0.0:
            continue
        raw[t] = (1.0 + math.log(n)) * idf
    norm = math.sqrt(sum(v * v for v in raw.values()))
    if norm == 0.0:
        return {}
    return {t: v / norm for t, v in raw.items()}


def _sentence_vectors(text: str, ctx) -> List[Dict[str, float]]:
    """One tf-idf vector per sentence of a paragraph."""
    from nyaya.analyze import analyze_terms, sentences

    out = []
    for s in sentences(text):
        v = vector(analyze_terms(s), ctx)
        if v:
            out.append(v)
    return out


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Both vectors are unit length, so the dot product is the cosine."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b.get(t, 0.0) for t, v in a.items()))


@dataclass
class VerifiedSentence:
    index: int
    text: str
    pid: str
    cosine: float = 0.0            # the support score: max(whole, best sentence)
    cosine_para: float = 0.0       # against the whole paragraph
    cosine_sent: float = 0.0       # against its best single sentence
    ok: bool = False
    reason: str = ""
    closest_pid: str = ""
    closest_cosine: float = 0.0
    cite: str = ""              # the honest paragraph reference, from ParaRef
    title: str = ""
    badge: str = ""

    def as_json(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class VerificationReport:
    sentences: List[VerifiedSentence] = field(default_factory=list)
    threshold: float = config.BRIEF_COSINE_THRESHOLD
    composer: str = "extractive"
    notes: List[str] = field(default_factory=list)

    @property
    def flagged(self) -> List[VerifiedSentence]:
        return [s for s in self.sentences if not s.ok]

    @property
    def passed(self) -> bool:
        return bool(self.sentences) and not self.flagged

    @property
    def mean_cosine(self) -> float:
        if not self.sentences:
            return 0.0
        return sum(s.cosine for s in self.sentences) / len(self.sentences)

    def as_json(self) -> Dict[str, Any]:
        return {"threshold": self.threshold, "composer": self.composer,
                "passed": self.passed, "mean_cosine": self.mean_cosine,
                "n_flagged": len(self.flagged), "notes": self.notes,
                "sentences": [s.as_json() for s in self.sentences]}


def verify(brief_text: str, pack: EvidencePack, ctx,
           threshold: float = config.BRIEF_COSINE_THRESHOLD,
           composer: str = "extractive") -> VerificationReport:
    """Check every sentence of ``brief_text`` against the paragraph it cites."""
    from nyaya.analyze import analyze_terms
    from nyaya.brief.compose import split_cited_sentences

    rep = VerificationReport(threshold=threshold, composer=composer)
    para_vecs = {p.pid: vector(analyze_terms(p.text), ctx) for p in pack.paragraphs}
    sent_vecs = {p.pid: _sentence_vectors(p.text, ctx) for p in pack.paragraphs}

    def support(sv: Dict[str, float], pid: str) -> Tuple[float, float, float]:
        """How well paragraph ``pid`` supports a sentence vector.

        The maximum of (a) the cosine against the whole paragraph and (b) the
        best cosine against any single sentence of it.  (a) alone is not enough
        and this was measured, not assumed: a one-sentence claim quoted verbatim
        out of a 300-word paragraph scored 0.111 against its own source, because
        the paragraph vector is dominated by everything else in it.  A sentence
        is supported if *some part* of the paragraph supports it; an unsupported
        sentence still matches none of them.
        """
        whole = cosine(sv, para_vecs.get(pid, {}))
        best = max((cosine(sv, v) for v in sent_vecs.get(pid, [])), default=0.0)
        return max(whole, best), whole, best

    for s in split_cited_sentences(brief_text):
        sv = vector(analyze_terms(s.body), ctx)
        best_pid, best_cos = "", 0.0
        for pid in para_vecs:
            c, _, _ = support(sv, pid)
            if c > best_cos:
                best_pid, best_cos = pid, c

        v = VerifiedSentence(index=s.index, text=s.text, pid=s.pid,
                             closest_pid=best_pid, closest_cosine=best_cos)
        cited = pack.by_pid(s.pid) if s.pid else None
        if cited is not None:
            v.cite, v.title, v.badge = cited.cite, cited.title, cited.badge

        if not s.pid:
            v.reason = ("no citation: an uncited sentence in a legal brief is "
                        "an assertion")
        elif cited is None:
            v.reason = (f"cites [{s.pid}], which is not in the evidence pack "
                        f"(the pack has {', '.join(pack.pids)})")
        else:
            v.cosine, v.cosine_para, v.cosine_sent = support(sv, s.pid)
            if v.cosine >= threshold:
                v.ok = True
            elif best_pid and best_pid != s.pid and best_cos >= threshold:
                v.reason = (f"cosine {v.cosine:.3f} against [{s.pid}] is below "
                            f"{threshold:.2f}; [{best_pid}] supports it better "
                            f"({best_cos:.3f}) - probably the wrong citation")
            else:
                v.reason = (f"cosine {v.cosine:.3f} is below {threshold:.2f} and "
                            f"nothing in the pack supports it better "
                            f"({best_cos:.3f}) - unsupported")
        rep.sentences.append(v)

    if composer == "extractive":
        rep.notes.append(
            "the extractive composer quotes real sentences, so a high cosine "
            "here is true by construction; the check exists for generated text")
    if rep.flagged:
        rep.notes.append(f"{len(rep.flagged)} of {len(rep.sentences)} sentences "
                         "did not verify and are shown with the closest "
                         "supporting paragraph")
    return rep


def render(rep: VerificationReport, width: int = 78) -> str:
    """The verifier panel, for the CLI and the UI."""
    L = ["grounding check", "=" * width,
         f"  composer   {rep.composer}",
         f"  threshold  tf-idf cosine >= {rep.threshold:.2f}",
         f"  sentences  {len(rep.sentences)}   "
         f"verified {len(rep.sentences) - len(rep.flagged)}   "
         f"flagged {len(rep.flagged)}",
         f"  mean cos   {rep.mean_cosine:.3f}", ""]
    for s in rep.sentences:
        mark = "ok  " if s.ok else "FLAG"
        L.append(f"  [{mark}] cos={s.cosine:.3f} "
                 f"(para {s.cosine_para:.3f} / best sentence {s.cosine_sent:.3f})"
                 f"  [{s.pid or '--'}] {s.cite}  {s.title[:34]}")
        L.append(f"          {s.text[:150]}")
        if not s.ok:
            L.append(f"          -> {s.reason}")
            if s.closest_pid:
                L.append(f"          -> closest in the pack: [{s.closest_pid}] "
                         f"at {s.closest_cosine:.3f}")
        if s.badge:
            L.append(f"          !! {s.badge[:110]}")
    for n in rep.notes:
        L.append(f"\n  note: {n}")
    return "\n".join(L)
