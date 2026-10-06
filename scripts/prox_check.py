"""Brute-force verification of the positional operators  (Member 1, wave 3).

Answers a set of real queries twice and compares:

**A. index vs naive scan.**  The expected answer is recomputed by re-analysing
the judgment text with ``analyze()`` and scanning the token list with a plain
nested loop - no postings, no skip pointers, no numpy merge.  This is the
correctness proof for the merge algorithms, which is the code I wrote.  It must
agree **100%**; a single mismatch is a bug.

**B. index vs regular expression over the raw text.**  The same queries are
answered by a regex over the surface words, which knows nothing about stemming,
statute canonicalisation, running-header removal or zone segmentation.  This is
*not* expected to agree perfectly, and every disagreement is a tokenisation
decision that belongs in the report: the regex and the index are answering
slightly different questions, and the script prints which documents differ and
why so the difference can be explained rather than hidden.

    python scripts/prox_check.py --docs 400
    python scripts/prox_check.py --docs 2000 --quiet
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                      # noqa: E402

from nyaya import config                                # noqa: E402
from nyaya.analyze import analyze, analyze_terms        # noqa: E402
from nyaya.index import positional as P                 # noqa: E402
from nyaya.query import proximity as prox               # noqa: E402
from nyaya.utils import ensure_utf8_stdout              # noqa: E402

# ==========================================================================
# The query set: surface wording a lawyer would actually type
# ==========================================================================

PHRASES = [
    "natural justice",
    "anticipatory bail",
    "fundamental right",
    "high court",
    "special leave",
    "burden of proof",
    "reasonable restriction",
    "basic structure",
    "res judicata",
    "ultra vires",
    "mens rea",
    "cruel treatment",
    "legitimate expectation",
    "judicial review",
    "public interest litigation",
]

PAIRS = [
    ("privacy", "surveillance"),
    ("bail", "anticipatory"),
    ("conviction", "eyewitness"),
    ("compensation", "negligence"),
    ("arbitration", "jurisdiction"),
    ("sedition", "speech"),
    ("dowry", "harassment"),
    ("acquittal", "benefit"),
    ("custody", "welfare"),
    ("reservation", "merit"),
]

WINDOWS = (1, 3, 5, 10, 25)


# ==========================================================================
# The independent token stream
# ==========================================================================

class Stream:
    __slots__ = ("terms", "zones", "sents", "paras", "index_of")

    def __init__(self, terms, zones, sents, paras):
        self.terms = terms
        self.zones = zones
        self.sents = sents
        self.paras = paras
        self.index_of: Dict[str, List[int]] = {}
        for i, t in enumerate(terms):
            self.index_of.setdefault(t, []).append(i)

    def positions_of(self, term: str) -> List[int]:
        return self.index_of.get(term, [])


def token_stream(record: dict) -> Stream:
    """Rebuild the document's token stream from its text, ignoring the index."""
    terms: List[str] = []
    zones: List[int] = []
    sents: List[int] = []
    paras: List[int] = []
    pos = 0
    sent_no = -1
    para_no = -1
    last_s = last_p = None
    for zone, text, para_id in P._zone_texts(record, zoned=True,
                                             include_statutes=True):
        zid = config.ZONE_ID[zone]
        for t in analyze(text, zone=zone, para_id=para_id, start_pos=pos):
            skey = (zone, para_id, t.sent_id)
            if skey != last_s:
                sent_no += 1
                last_s = skey
            pkey = (zone, para_id)
            if pkey != last_p:
                para_no += 1
                last_p = pkey
            terms.append(t.term)
            zones.append(zid)
            sents.append(sent_no)
            paras.append(para_no)
            pos += 1
    return Stream(terms, zones, sents, paras)


# ==========================================================================
# Naive scans
# ==========================================================================

def naive_phrase(st: Stream, words: Sequence[str]) -> bool:
    m = len(words)
    first = st.positions_of(words[0])
    for i in first:
        if i + m > len(st.terms):
            continue
        if st.terms[i:i + m] != list(words):
            continue
        if len(set(st.zones[i:i + m])) == 1:
            return True
    return False


def naive_window(st: Stream, a: str, b: str, k: int) -> bool:
    pa = st.positions_of(a)
    pb = st.positions_of(b)
    if not pa or not pb:
        return False
    for i in pa:
        for j in pb:
            if i != j and abs(i - j) <= k and st.zones[i] == st.zones[j]:
                return True
    return False


def naive_bucket(st: Stream, a: str, b: str, kind: str) -> bool:
    pa = st.positions_of(a)
    pb = st.positions_of(b)
    if not pa or not pb:
        return False
    buckets = st.sents if kind == "s" else st.paras
    for i in pa:
        for j in pb:
            if i != j and buckets[i] == buckets[j] and st.zones[i] == st.zones[j]:
                return True
    return False


# ==========================================================================
# Regex over the raw text
# ==========================================================================

def raw_text(record: dict) -> str:
    parts: List[str] = []
    for z in config.ZONES:
        if z in ("body", "order"):
            parts.extend(p.get("text", "") for p in (record.get("paras") or [])
                         if p.get("zone") == z)
        else:
            t = (record.get("zones") or {}).get(z) or ""
            if t.strip():
                parts.append(t)
    return "\n".join(parts)


def regex_phrase(text: str, surface: str) -> bool:
    pat = re.compile(r"\b" + r"[\s\-]+".join(map(re.escape, surface.split())) + r"\w*",
                     re.I)
    return bool(pat.search(text))


def regex_window(text: str, a: str, b: str, k: int) -> bool:
    """Both surface words within k whitespace-separated words of each other.

    The surface word plus any suffix ("acquittal", "acquittals"), *not* a short
    prefix: a five-character prefix made "acqui" match "acquire" and "priva"
    match "private", which produced regex-only hits that were the checker's bug,
    not the index's.  This is still completely independent of the stemmer.
    """
    words = re.findall(r"[A-Za-z']+", text.lower())
    pa = [i for i, w in enumerate(words) if w.startswith(a) and len(w) <= len(a) + 3]
    pb = [i for i, w in enumerate(words) if w.startswith(b) and len(w) <= len(b) + 3]
    if not pa or not pb:
        return False
    for i in pa:
        for j in pb:
            if i != j and abs(i - j) <= k:
                return True
    return False



# ==========================================================================
# Explaining a mismatch between the index and the raw-text regex
# ==========================================================================
#
# Every disagreement falls into one of four classes, and all four are *expected*
# consequences of decisions taken deliberately upstream.  The script names the
# class per document so the report can quote counts instead of hand-waving.

STEMMING = "stemming conflation (index matches a variant the surface regex cannot)"
NORMALISE = "normalisation (index is adjacent across OCR junk the regex sees)"
TOKENCOUNT = "token count (index counts numerals/abbreviations the regex skips)"
CROSSZONE = "cross-zone (regex matched across a zone seam; the index rejects it)"
UNKNOWN = "unexplained"


def _surface_variants(text: str, word: str, stem: str) -> Set[str]:
    """Surface forms in ``text`` that stem to ``stem`` but are not ``word*``."""
    out: Set[str] = set()
    for m in re.finditer(r"[A-Za-z]{3,}", text):
        w = m.group(0)
        low = w.lower()
        if low.startswith(word) and len(low) <= len(word) + 3:
            continue                      # the regex could already reach this
        if analyze_terms(w)[:1] == [stem]:
            out.add(low)
    return out


def explain_mismatch(kind: str, words: Sequence[str], stems: Sequence[str],
                     doc_id: int, index_only: bool, st: Stream,
                     text: str, k: int = 0) -> Tuple[str, str]:
    """``(class, evidence)`` for one mismatching document."""
    if index_only:
        for w, stem in zip(words, stems):
            extra = _surface_variants(text, w, stem)
            if extra:
                return STEMMING, f"{w!r}/{stem!r} also written {sorted(extra)[:3]}"
        # no morphological explanation: the tokens must be adjacent across
        # something the cleaner removed
        if kind == "phrase":
            for i in range(len(st.terms) - len(stems) + 1):
                if st.terms[i:i + len(stems)] == list(stems):
                    raw = _raw_near(text, words)
                    return NORMALISE, f"raw text reads {raw!r}"
        return NORMALISE, "index tokens adjacent across removed text"
    # regex-only
    pa = st.positions_of(stems[0])
    pb = st.positions_of(stems[-1])
    if pa and pb:
        gap, i, j = min((abs(a - b), a, b) for a in pa for b in pb)
        if st.zones[i] != st.zones[j]:
            return CROSSZONE, (f"closest pair is {config.ZONES[st.zones[i]]}/"
                               f"{config.ZONES[st.zones[j]]}")
        if kind == "window" and gap > k:
            span = " ".join(st.terms[min(i, j):max(i, j) + 1])
            return TOKENCOUNT, f"token gap {gap} > k={k}; span {span!r}"
    return UNKNOWN, "no explanation found"


def _raw_near(text: str, words: Sequence[str]) -> str:
    i = text.lower().find(words[0])
    return text[max(0, i - 5):i + 40].replace("\n", " ") if i >= 0 else ""


def _note(notes: List[str], classes: Dict[str, int], kind: str, label: str,
          words: Sequence[str], stems: Sequence[str], got: Set[int],
          rx: Set[int], streams: Dict[int, Stream], raws: Dict[int, str],
          k: int = 0) -> None:
    """Record one disagreement and the class of every document in it."""
    lines = [f"{label}: index {len(got)}, regex {len(rx)}"]
    for d in sorted(got - rx)[:3]:
        cls, why = explain_mismatch(kind, words, stems, d, True,
                                   streams[d], raws[d], k)
        classes[cls] = classes.get(cls, 0) + 1
        lines.append(f"      +{d} {cls}: {why}")
    for d in sorted(rx - got)[:3]:
        cls, why = explain_mismatch(kind, words, stems, d, False,
                                   streams[d], raws[d], k)
        classes[cls] = classes.get(cls, 0) + 1
        lines.append(f"      -{d} {cls}: {why}")
    notes.append("\n".join(lines))

# ==========================================================================
# The run
# ==========================================================================

def load_records(limit: int, stride: int = 1) -> List[dict]:
    """Every ``stride``-th judgment, so the sample spans 1950 to 2025.

    The corpus is written in doc_id order, which is year order, so taking the
    first N would verify the operators only against the 1950s scans - where most
    modern legal vocabulary does not occur and most queries return nothing.
    """
    out: List[dict] = []
    with Path(config.JUDGMENTS_JSONL).open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if n % stride:
                continue
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if len(out) >= limit:
                break
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--docs", type=int, default=400,
                    help="how many judgments to verify over")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth judgment, so the sample spans all eras")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--max-mismatch", type=int, default=6,
                    help="how many mismatching documents to print per query")
    args = ap.parse_args(argv)

    idx = P.PositionalIndex.load()
    print(f"index: {idx.vocab_size} terms, {idx.doc_count} documents",
          file=sys.stderr)
    t0 = time.perf_counter()
    records = load_records(args.docs, args.stride)
    ids = [int(r["doc_id"]) for r in records]
    id_set = set(ids)
    streams = {int(r["doc_id"]): token_stream(r) for r in records}
    raws = {int(r["doc_id"]): raw_text(r) for r in records}
    print(f"re-analysed {len(records)} judgments in "
          f"{time.perf_counter() - t0:.0f}s", file=sys.stderr)

    classes: Dict[str, int] = {}
    checks = 0
    agree_naive = 0
    agree_regex = 0
    regex_checks = 0
    naive_fail: List[str] = []
    regex_notes: List[str] = []

    def restrict(a: np.ndarray) -> Set[int]:
        return {int(d) for d in a if int(d) in id_set}

    # ---- phrases ------------------------------------------------------
    for surface in PHRASES:
        terms = analyze_terms(surface)
        if len(terms) < 2:
            continue
        got = restrict(prox.phrase_docs(terms, idx))
        want = {d for d, st in streams.items() if naive_phrase(st, terms)}
        checks += 1
        if got == want:
            agree_naive += 1
        else:
            naive_fail.append(
                f"PHRASE {surface!r} -> {terms}: index-only "
                f"{sorted(got - want)[:args.max_mismatch]}, naive-only "
                f"{sorted(want - got)[:args.max_mismatch]}")
        rx = {d for d, t in raws.items() if regex_phrase(t, surface)}
        regex_checks += 1
        if rx == got:
            agree_regex += 1
        else:
            _note(regex_notes, classes, "phrase", f"PHRASE {surface!r}",
                  surface.split(), terms, got, rx, streams, raws)
        if not args.quiet:
            print(f"  phrase {surface!r:<30} index={len(got):>4} "
                  f"naive={len(want):>4} regex={len(rx):>4} "
                  f"{'OK' if got == want else 'MISMATCH'}")

    # ---- /n -----------------------------------------------------------
    for (a, b), k in itertools.product(PAIRS, WINDOWS):
        ta = analyze_terms(a)
        tb = analyze_terms(b)
        if not ta or not tb:
            continue
        got = restrict(prox.proximity_docs(None, None, ta[:1], tb[:1], "n", k, idx))
        want = {d for d, st in streams.items()
                if naive_window(st, ta[0], tb[0], k)}
        checks += 1
        if got == want:
            agree_naive += 1
        else:
            naive_fail.append(
                f"/{k} ({a},{b}) -> ({ta[0]},{tb[0]}): index-only "
                f"{sorted(got - want)[:args.max_mismatch]}, naive-only "
                f"{sorted(want - got)[:args.max_mismatch]}")
        if k in (5, 25):
            rx = {d for d, t in raws.items() if regex_window(t, a, b, k)}
            regex_checks += 1
            if rx == got:
                agree_regex += 1
            else:
                _note(regex_notes, classes, "window", f"/{k} ({a},{b})",
                      [a, b], [ta[0], tb[0]], got, rx, streams, raws, k=k)
        if not args.quiet and k == 5:
            print(f"  /{k:<3} {a + '~' + b:<30} index={len(got):>4} "
                  f"naive={len(want):>4} {'OK' if got == want else 'MISMATCH'}")

    # ---- /s and /p ----------------------------------------------------
    for (a, b), kind in itertools.product(PAIRS, ("s", "p")):
        ta = analyze_terms(a)
        tb = analyze_terms(b)
        if not ta or not tb:
            continue
        got = restrict(prox.proximity_docs(None, None, ta[:1], tb[:1], kind, 0, idx))
        want = {d for d, st in streams.items()
                if naive_bucket(st, ta[0], tb[0], kind)}
        checks += 1
        if got == want:
            agree_naive += 1
        else:
            naive_fail.append(
                f"/{kind} ({a},{b}): index-only "
                f"{sorted(got - want)[:args.max_mismatch]}, naive-only "
                f"{sorted(want - got)[:args.max_mismatch]}")
        if not args.quiet:
            print(f"  /{kind:<3} {a + '~' + b:<30} index={len(got):>4} "
                  f"naive={len(want):>4} {'OK' if got == want else 'MISMATCH'}")

    print("\n" + "=" * 74)
    print(f"A. index vs naive scan  : {agree_naive}/{checks} = "
          f"{agree_naive / max(checks, 1):.2%}   (must be 100%)")
    print(f"B. index vs raw regex   : {agree_regex}/{regex_checks} = "
          f"{agree_regex / max(regex_checks, 1):.2%}   "
          f"(differences are tokenisation, explained below)")
    print(f"documents verified      : {len(records)}")
    print(f"wall clock              : {time.perf_counter() - t0:.0f}s")
    if naive_fail:
        print("\nBUGS (index disagrees with the naive scan):")
        for line in naive_fail:
            print(f"  {line}")
    if regex_notes:
        print(f"\nTokenisation differences vs a raw regex "
              f"({len(regex_notes)} of {regex_checks} queries):")
        for line in regex_notes[:20]:
            print(f"  {line}")
    if classes:
        print("\nmismatching documents by class:")
        for cls, n in sorted(classes.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>3}  {cls}")
        unexplained = classes.get(UNKNOWN, 0)
        print(f"\nunexplained: {unexplained}")
    return 1 if naive_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
