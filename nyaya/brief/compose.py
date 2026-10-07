"""Writing the brief from the evidence pack.

Two composers, and the order matters
------------------------------------
1. **Extractive** (`compose_extractive`) — the default and the only one that
   runs here.  It selects real sentences out of the pack, so every sentence is
   verbatim from a judgment and its citation is correct **by construction**.
   It cannot hallucinate, it needs no network, and it cannot break the demo.
2. **Abstractive** (`compose_llm`) — behind a provider interface that is used
   only when an API key is actually present.  **No key is available in this
   environment and the `anthropic` SDK is not installed**, so this path has
   never run; it is written, it is wired, and `brief()` reports plainly that it
   fell back rather than pretending.

The instructions the abstractive path would send are kept here rather than in a
prompt file because they are part of the system's behaviour: use only the given
paragraphs, four to six sentences, end every sentence with exactly one
paragraph id.  Whatever it returns is then put through the same verifier as
anything else — the verifier is not a formality for a trusted generator, it is
the thing that makes an untrusted one usable.

Author: Sumedha (M3)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyaya import config
from nyaya.brief.evidence import EvidencePack, EvidenceParagraph

__all__ = ["compose_extractive", "compose_llm", "llm_available", "SYSTEM_PROMPT",
           "Sentence", "split_cited_sentences"]

MIN_SENTENCES = 4
MAX_SENTENCES = 6
#: A sentence shorter than this is a fragment, not a proposition.
MIN_SENTENCE_WORDS = 8
MAX_SENTENCE_WORDS = 70

CITE_RE = re.compile(r"\[(P\d+)\]\s*$")
#: Any citation marker, anywhere - the splitter cuts on these.
CITE_ANY = re.compile(r"\[(P\d+)\]")

#: S.C.R. page furniture that is a list, not a proposition.  A brief must not
#: quote the reporter's own index back at the reader.
BOILERPLATE = re.compile(
    r"list of (keywords|acts)|case law reference|headnote|"
    r"^\s*(civil|criminal) appeal no|^\s*from the judgment and order",
    re.I)

SYSTEM_PROMPT = """\
You are summarising Indian Supreme Court authority for a lawyer.

You may use ONLY the numbered paragraphs supplied below. You may not use any
other knowledge, and you may not state anything that is not in them.

Write {lo} to {hi} sentences. End EVERY sentence with exactly one paragraph id
in square brackets, like [P3]. A sentence with no id, or with more than one, is
rejected. Do not add a preamble, a heading or a conclusion.

If a paragraph comes from a judgment marked with a treatment warning, do not
present it as settled law without saying so.
"""


def llm_available() -> Tuple[bool, str]:
    """Is an abstractive composer actually usable here?  Returns ``(ok, why)``."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(var):
            try:
                __import__("anthropic" if var.startswith("ANTHROPIC") else "openai")
                return True, f"{var} is set"
            except ImportError:
                return False, f"{var} is set but its SDK is not installed"
    return False, ("no model API key in the environment, so the brief is "
                   "extractive: every sentence is verbatim from a judgment")


# ==========================================================================
# Sentences
# ==========================================================================


@dataclass
class Sentence:
    text: str
    pid: str = ""
    index: int = 0

    @property
    def body(self) -> str:
        """The sentence without its trailing citation."""
        return CITE_RE.sub("", self.text).strip()


def split_cited_sentences(brief_text: str) -> List[Sentence]:
    """Split a brief into cited sentences.

    The cut is made **at each ``[Pn]`` marker**, not at sentence boundaries.
    Splitting on sentences first is wrong and was a real bug: a full stop ends
    the sentence *before* its citation, so "... conviction. [P1] The next ..."
    read as a sentence ending in "conviction." and a next sentence *starting*
    with "[P1]", and every sentence came back uncited.

    Within a segment, `analyze.sentences` still does the work — the same
    splitter the index used, so "Puttaswamy v. Union of India" is one sentence —
    and any sentence before the last one in a segment is reported as **uncited**,
    which is how a generator writing two sentences per citation gets caught.
    """
    from nyaya.analyze import sentences

    out: List[Sentence] = []
    pos = 0
    for m in CITE_ANY.finditer(brief_text):
        seg = brief_text[pos:m.start()].strip()
        pos = m.end()
        pid = m.group(1)
        sents = [x for x in (t.strip() for t in sentences(seg)) if x] if seg else []
        for extra in sents[:-1]:
            out.append(Sentence(text=extra, pid="", index=len(out)))
        last = sents[-1] if sents else ""
        out.append(Sentence(text=f"{last} [{pid}]".strip(), pid=pid,
                            index=len(out)))
    tail = brief_text[pos:].strip()
    for extra in (t.strip() for t in sentences(tail)) if tail else []:
        if extra:
            out.append(Sentence(text=extra, pid="", index=len(out)))
    return out


# ==========================================================================
# The extractive composer
# ==========================================================================


def compose_extractive(pack: EvidencePack, terms: Sequence[str], ctx,
                       lo: int = MIN_SENTENCES, hi: int = MAX_SENTENCES
                       ) -> str:
    """One best sentence from each of the top paragraphs, in pack order.

    Verbatim, so the citation is right by construction and the cosine verifier
    will score it ~1.0.  That is a tautological pass and the report says so —
    the verifier earns its keep against *generated* text, and the extractive
    path is here so the feature degrades to something true rather than to
    nothing.
    """
    from nyaya.brief.verify import cosine, vector

    qv = vector(terms, ctx)
    picks: List[Tuple[float, str, str]] = []
    for p in pack.paragraphs:
        best = _best_sentence(p, qv, ctx)
        if best is None:
            continue
        score, text = best
        picks.append((score, text, p.pid))
    if not picks:
        return ""
    picks.sort(key=lambda kv: -kv[0])
    kept = picks[:max(lo, min(hi, len(picks)))]
    # restore pack order: the pack is already ranked, so this reads as a brief
    order = {p.pid: i for i, p in enumerate(pack.paragraphs)}
    kept.sort(key=lambda kv: order.get(kv[2], 99))
    return " ".join(f"{_tidy(text)} [{pid}]" for _, text, pid in kept)


def _best_sentence(p: EvidenceParagraph, qv: Dict[str, float], ctx
                   ) -> Optional[Tuple[float, str]]:
    from nyaya.analyze import analyze_terms, sentences
    from nyaya.brief.verify import cosine, vector

    best: Optional[Tuple[float, str]] = None
    for s in sentences(p.text):
        words = s.split()
        if not (MIN_SENTENCE_WORDS <= len(words) <= MAX_SENTENCE_WORDS):
            continue
        if BOILERPLATE.search(s):
            continue
        if not s[:1].isupper():
            continue      # a fragment the splitter cut mid-sentence, not a claim
        sim = cosine(qv, vector(analyze_terms(s), ctx))
        if best is None or sim > best[0]:
            best = (sim, s)
    if best is None:
        # no sentence in range: fall back to the head of the paragraph, trimmed
        words = p.text.split()[:MAX_SENTENCE_WORDS]
        if len(words) < MIN_SENTENCE_WORDS:
            return None
        return (0.0, " ".join(words))
    return best


def _tidy(text: str) -> str:
    text = " ".join(text.split())
    text = re.sub(r"\s+([.,;:])", r"", text)   # OCR leaves "fully . entitled"
    text = re.sub(r"\s{2,}", " ", text)
    text = text.rstrip(" .;,")
    return text + "."


# ==========================================================================
# The abstractive composer (never exercised here)
# ==========================================================================


def compose_llm(pack: EvidencePack, lo: int = MIN_SENTENCES,
                hi: int = MAX_SENTENCES, model: str = "claude-sonnet-4-5"
                ) -> Tuple[str, str]:
    """Ask a model to write the brief from the pack.  ``(text, note)``.

    Returns ``("", reason)`` and never raises when no key or SDK is present, so
    `brief()` can fall back without a try/except around the feature.
    """
    ok, why = llm_available()
    if not ok:
        return "", why
    prompt = render_prompt(pack, lo, hi)
    try:                                                     # pragma: no cover
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=model, max_tokens=700,
                system=SYSTEM_PROMPT.format(lo=lo, hi=hi),
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content
                           if getattr(b, "type", "") == "text"), f"model {model}"
        import openai

        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=os.environ.get("NYAYA_LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system",
                       "content": SYSTEM_PROMPT.format(lo=lo, hi=hi)},
                      {"role": "user", "content": prompt}])
        return resp.choices[0].message.content or "", "openai"
    except Exception as exc:                                 # pragma: no cover
        return "", f"the model call failed ({exc.__class__.__name__}), so the "\
                   "brief fell back to the extractive composer"


def render_prompt(pack: EvidencePack, lo: int = MIN_SENTENCES,
                  hi: int = MAX_SENTENCES) -> str:
    """The user message: the question, then the only paragraphs that may be used."""
    lines = [f"QUESTION: {pack.query}", "", "PARAGRAPHS:"]
    for p in pack.paragraphs:
        warn = f"  [TREATMENT WARNING: {p.badge}]" if p.badge else ""
        lines.append(f"\n[{p.pid}] {p.title} ({p.year}), {p.cite}{warn}\n"
                     f"{p.text}")
    return "\n".join(lines)
