"""`analyze()` — the single shared text pipeline for documents **and** queries.

Why one function
----------------
Everything in a vector-space or BM25 system is a comparison of two term sets.
If the document side and the query side are produced by different code, the
comparison is silently wrong: the index holds ``ipc§302`` while the query holds
``302``, the index holds ``convict`` while the query holds ``conviction``, and
recall collapses with no error message anywhere.  So the index builder (M1) and
the query normaliser (M3) both call **this** function, and it is frozen.

IR framing: term normalisation as equivalence classing (guide Section 4.3)
--------------------------------------------------------------------------
Each pass below collapses a set of surface forms onto one index term:

    pass 0  character stream   "appel-\\nlant", curly quotes, NBSP  -> clean text
    pass 1  statute mentions   "S. 302 IPC" / "Sec. 302" / ...      -> ipc§302
    pass 2  case + punctuation "Appellants'" / "APPELLANT"          -> appellant
    pass 3  stop words         kept in positions, dropped from tf-idf
    pass 4  morphology         "convicted"/"conviction"/"convicts"  -> convict

Pass 1 runs **first, on raw text**: tokenisation destroys exactly the
punctuation the statute patterns need ("I.P.C.", "u/s", "304-B"), so by the time
you have tokens it is too late.  Pass 4 runs **last** and is skipped for the
tokens where morphology is meaningless or harmful (statute tokens, numbers,
abbreviations, case names).

What callers may rely on
------------------------
* `pos` is a running offset **within the zone**, starting at `start_pos`
  (default 0), and it counts *every* token including stop words — this is what
  makes phrase and ``/k`` proximity exact.
* `sent_id` and `para_id` are 0-based and monotone non-decreasing.
* Stop words are present in the returned list.  Scorers must call
  `is_scoring_stopword(tok.term)` to exclude them from tf-idf / BM25F.
* The same `text` always yields the same tokens; there is no hidden state
  except a pure memoisation cache for the stemmer.

Demo:  ``python -m nyaya.analyze --demo "...convicted under Sec. 302 of the
Indian Penal Code, 1860 and the Appellants'..."``

Author: Sumedha (M3)
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

from nyaya.contracts import Token
from nyaya import statutes
from nyaya.statutes import canonicalise_statutes, is_statute_token
from nyaya.stoplist import (
    DOMAIN_DF,
    SCORING_STOPWORDS,
    STANDARD_STOPWORDS,
    is_scoring_stopword,
)

__all__ = [
    "analyze",
    "analyze_terms",
    "canonicalise_statutes",
    "clean_text",
    "sentences",
    "stem_term",
    "is_scoring_stopword",
    "SCORING_STOPWORDS",
    "pipeline_trace",
]

# ==========================================================================
# Pass 0 — the character stream
# ==========================================================================
_SMART = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    " ": " ", " ": " ", " ": " ", "﻿": "",
    "­": "",                 # soft hyphen
    "…": "...",
}
_SMART_RE = re.compile("|".join(map(re.escape, _SMART)))
#: "appel-\n lant" -> "appellant".  PDF extraction leaves these everywhere and
#: they would otherwise create two junk terms per line break.
_DEHYPHEN = re.compile(r"([A-Za-z])-[ \t]*\r?\n[ \t]*([a-z])")
_WS = re.compile(r"[ \t\r\f\v]+")
_MANY_NL = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise the character stream without changing token boundaries.

    Blank lines are preserved on purpose: they are the paragraph boundaries that
    reset the statute act-inheritance heuristic (pass 1).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _SMART_RE.sub(lambda m: _SMART[m.group(0)], text)
    text = _DEHYPHEN.sub(r"\1\2", text)
    text = _WS.sub(" ", text)
    text = _MANY_NL.sub("\n\n", text)
    return text.strip()


# ==========================================================================
# Pass 2 — tokenisation, case folding, sentence splitting
# ==========================================================================
# One scan, five token shapes.  The statute shape is first so that `ipc§302`
# survives intact; `§` cannot appear in any other shape, which is exactly why it
# was chosen as the separator.
TOKEN_RE = re.compile(
    r"""
      (?P<stat>[A-Za-z]{2,10}§[0-9]+[A-Za-z0-9]*)
    | (?P<slash>(?:r/w|u/s|w\.e\.f\.?|s/o|w/o|d/o|and/or))
    | (?P<init>(?:[A-Za-z]\.){2,})
    | (?P<abbr>[A-Za-z][A-Za-z']*\.)
    | (?P<word>[A-Za-z][A-Za-z']*)
    | (?P<num>\d+(?:[A-Za-z]{1,2})?)
    | (?P<punct>[.?!]+)
    """,
    re.X | re.I,
)

#: Sentence splitting must not break on legal abbreviations.  Everything here
#: ends in a full stop that is *not* a sentence end.  Judgments are full of
#: them: "Puttaswamy v. Union of India", "Kapur, J.", "Nos. 1 to 4 of 2012",
#: "M/s Tata Ltd.", "AIR 1973 SC 1461".
NO_BREAK_ABBREV = frozenset("""
v. vs. versus j. jj. cj. cji. s. ss. sec. secs. sect. art. arts. cl. cls.
no. nos. ltd. pvt. co. corp. inc. mr. mrs. ms. dr. smt. sri. shri. kum.
hon'ble honble messrs. m/s. p. pp. para. paras. pr. ch. chap. vol. vols. ed.
edn. rs. govt. dept. dt. ors. anr. etc. viz. i.e. e.g. cf. ibid. id. seq.
air scc scr sc hc all. bom. cal. mad. ker. guj. raj. pat. del. ori. mp. mys.
cri. crl. lj. l.j. supp. rev. appl. appln. o. r. u/s r/w w.e.f. st. mt.
jan. feb. mar. apr. jun. jul. aug. sept. sep. oct. nov. dec.
""".split())

#: A single or multiple initial: "K.", "K.S.", "A.I.R." — never a sentence end.
_INITIALS_RE = re.compile(r"^(?:[A-Za-z]\.){1,4}$")

_Tok = Tuple[str, str, int]          # (kind, raw, start offset)


def _scan(text: str) -> List[_Tok]:
    return [(m.lastgroup, m.group(0), m.start()) for m in TOKEN_RE.finditer(text)]


def _fold(raw: str, kind: str) -> str:
    """Case folding + punctuation stripping.  ``"Appellants'"`` -> ``appellant``."""
    t = raw.lower()
    if kind in ("init", "abbr", "slash"):
        t = t.replace(".", "")
    if t.endswith("'s"):
        t = t[:-2]
    t = t.strip("'")
    return t


def _is_sentence_end(kind: str, raw: str) -> bool:
    """Does this token close a sentence?  (Called only for '.'-ending tokens.)"""
    low = raw.lower()
    if low in NO_BREAK_ABBREV:
        return False
    if _INITIALS_RE.match(raw):
        return False
    return True


def sentences(text: str) -> List[str]:
    """Split ``text`` into sentences using the legal abbreviation exceptions.

    Exposed because M1's treatment classifier and snippet generator need the
    *same* sentence boundaries that `analyze()` used for `sent_id`.
    """
    toks = _scan(text)
    out: List[str] = []
    start = 0
    for i, (kind, raw, off) in enumerate(toks):
        end_here = False
        if kind == "punct":
            end_here = True
        elif kind in ("abbr", "init") and raw.endswith("."):
            end_here = _is_sentence_end(kind, raw)
        if not end_here:
            continue
        if not _next_looks_like_new_sentence(toks, i):
            continue
        stop = off + len(raw)
        chunk = text[start:stop].strip()
        if chunk:
            out.append(chunk)
        start = stop
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _next_looks_like_new_sentence(toks: List[_Tok], i: int) -> bool:
    """A full stop only breaks if what follows could start a sentence.

    Guards against OCR noise and against abbreviations we have not listed:
    "...u/s. 302 ipc and the appellant" keeps going because "and" is lower-case.
    """
    for kind, raw, _ in toks[i + 1:]:
        if kind == "punct":
            continue
        first = raw[0]
        return first.isupper() or first.isdigit() or kind == "stat"
    return True          # end of text


# ==========================================================================
# Pass 4 — stemming, with the exceptions that matter
# ==========================================================================
_STEMMER = None
_STEM_CACHE: Dict[str, str] = {}

#: Never stem these: legal abbreviations and Latin where Porter does damage
#: ("ors" -> "or", "ipso" -> "ipso", "res" -> "re", "dicta" -> "dicta").
NO_STEM_TERMS = frozenset("""
air scc scr sc hc ors anr ltd pvt govt ibid viz cf etc
ratio dicta dictum obiter ratione decidendi stare decisis
mens rea actus reus habeas corpus mandamus certiorari quo warranto
prima facie suo motu sine die ultra vires intra vires res judicata
inter alia ab initio bona fide mala fide ex parte de novo locus standi
amicus curiae sub judice in limine nolle prosequi caveat
""".split())

#: Ordinary English words that happen to appear inside party names.  The
#: party-context exception below must NOT fire on them, or the `parties` zone
#: would hold "committee" while a query holds the stem "committe".
PARTY_COMMON_WORDS = frozenset("""
state states union india indian government govt republic president governor
committee commission commissioner board authority corporation corp company
companies limited ltd private pvt bank banks trust society association
council municipal municipality panchayat corporation district collector
officer officers director directors general secretary chairman chairperson
department ministry department employees employee workers workmen union
others another anr ors national central federation institute university
college school hospital department service services department enterprises
industries industry mills factory works department authority development
housing transport electricity power water supply revenue police
tax taxes income customs excise insurance life
sons brothers traders agencies agency estate estates
public private sector undertaking undertakings
and of the for in by versus vs v
""".split())

# The hand list above is only the fallback.  The real gate is the *measured*
# vocabulary in nyaya._lexicon: any token that is frequent in body text is an
# ordinary word, whatever it is standing next to.
from nyaya import _lexicon                                  # noqa: E402

_ORDINARY_WORDS = PARTY_COMMON_WORDS | _lexicon.COMMON_WORDS

#: Case-name tokens a user is likely to type *without* "v." — the deterministic
#: half of the proper-noun exception (see docstring of `_mark_no_stem`).
PROTECTED_NAMES = frozenset("""
puttaswamy kesavananda bharati maneka golaknath minerva shankari
shreya singhal vishaka bachan kehar indira nandini satpathy
keshavananda sajjan gopalan makhan mohd mohammed shah bano
navtej johar sabarimala ayodhya ismail faruqui joseph
subramanian swamy vineet narain prakash jolly varghese
olga tellis francis coralie sunil batra hussainara khatoon
rudul daniel latifi githa hariharan selvi sheela barse
dk basu nilabati behera arnesh lalita kumari
kartar antulay bhagwati krishna iyer
""".split())


def _stemmer():
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import PorterStemmer

        _STEMMER = PorterStemmer()
    return _STEMMER


def stem_term(term: str) -> str:
    """Memoised Porter stem.  The cache matters: the builder calls this tens of
    millions of times and PorterStemmer is pure Python."""
    cached = _STEM_CACHE.get(term)
    if cached is not None:
        return cached
    out = _stemmer().stem(term)
    _STEM_CACHE[term] = out
    return out


def _never_stem(term: str) -> bool:
    """Term-shape exceptions — a pure function of the term, so documents and
    queries always agree."""
    if not term or "§" in term:
        return True
    if len(term) <= 2:
        return True
    if not term.isalpha():                  # "1860", "65b", "302a"
        return True
    if term in NO_STEM_TERMS:
        return True
    if term in PROTECTED_NAMES or term in _lexicon.PARTY_NAME_WORDS:
        return True
    return False


_PARTY_MARKERS = frozenset({"v", "vs", "versus"})
_PARTY_WINDOW = 6


def _mark_no_stem(terms: List[str], raws: List[str], kinds: List[str]) -> List[bool]:
    """Decide, per token, whether pass 4 is skipped.

    Two rules, and the split between them is deliberate:

    1. **Term-shape rules** (`_never_stem`) are pure functions of the term, so
       a document and a query always reach the same decision.  This covers
       statute tokens, numbers, short tokens, legal abbreviations and the
       curated PROTECTED_NAMES list of famous case names.
    2. **Party context.**  Around a "v." / "vs." separator, the capitalised run
       on each side is a case name, and stemming it is pure damage
       ("Puttaswamy" -> "puttaswami").  This rule *is* context-dependent, so it
       is gated hard: it only fires on capitalised tokens that are not ordinary
       English party vocabulary (PARTY_COMMON_WORDS), which is what stops it
       from leaving "Committee" unstemmed in a title while a query stems it.
       Known limitation: a rare name typed without "v." still gets stemmed.  The
       mitigation is PROTECTED_NAMES, refreshed in wave 2 from the `parties`
       zone of the built corpus.
    """
    n = len(terms)
    flags = [_never_stem(t) for t in terms]
    for i, t in enumerate(terms):
        if t not in _PARTY_MARKERS:
            continue
        if t == "v" and not raws[i].endswith("."):
            continue                        # a stray "v", not a case separator
        left = range(max(0, i - _PARTY_WINDOW), i)
        right = range(i + 1, min(n, i + 1 + _PARTY_WINDOW))
        for side in (left, right):
            for j in side:
                raw, term = raws[j], terms[j]
                if not raw[:1].isupper():
                    if side is right:
                        break               # the capitalised run has ended
                    continue                # scanning left: keep looking
                if term in _ORDINARY_WORDS or term in STANDARD_STOPWORDS:
                    continue
                if term.isalpha() and len(term) > 3:
                    flags[j] = True
    return flags


# ==========================================================================
# The pipeline
# ==========================================================================


def analyze(
    text: str,
    *,
    zone: str = "body",
    para_id: int = 0,
    stem: bool = True,
    canonicalise: bool = True,
    start_pos: int = 0,
    inherit_act: Optional[bool] = None,
    protect_names: bool = True,
) -> List[Token]:
    """Turn text into the token stream that is indexed and queried.

    Parameters
    ----------
    zone, para_id
        Copied onto every Token.  `analyze()` is called **per paragraph** (store
        S1 is a paragraph store), so `para_id` is constant across the result.
    stem
        Pass 4 on/off.  This is the stemming ablation: a flag, not a code change.
    canonicalise
        Pass 1 on/off.  With it off you get the B0 baseline tokenizer, which is
        exactly what the "statute canonicalisation is worth N points of MAP"
        experiment needs.
    start_pos
        First `pos` value.  M1 concatenates paragraphs into a zone, so it passes
        the running offset here instead of re-numbering afterwards.
    inherit_act
        Override the act-inheritance heuristic (default
        `statutes.INHERIT_ACT_DEFAULT`).
    protect_names
        Apply the party-context stemming exception.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # ---- pass 1: statute canonicalisation, on raw text -------------------
    if canonicalise:
        kw = {} if inherit_act is None else {"inherit_act": inherit_act}
        cleaned = canonicalise_statutes(cleaned, **kw)

    # ---- pass 2: tokenise, fold, number sentences ------------------------
    scanned = _scan(cleaned)
    kinds: List[str] = []
    raws: List[str] = []
    terms: List[str] = []
    sent_ids: List[int] = []

    sent_id = 0
    for i, (kind, raw, _off) in enumerate(scanned):
        if kind == "punct":
            if _next_looks_like_new_sentence(scanned, i):
                sent_id += 1
            continue
        term = _fold(raw, kind)
        if not term:
            continue
        kinds.append(kind)
        raws.append(raw)
        terms.append(term)
        sent_ids.append(sent_id)
        if kind in ("abbr", "init") and raw.endswith("."):
            if _is_sentence_end(kind, raw) and _next_looks_like_new_sentence(scanned, i):
                sent_id += 1

    # ---- pass 4: stemming, with exceptions -------------------------------
    if stem:
        no_stem = (
            _mark_no_stem(terms, raws, kinds)
            if protect_names
            else [_never_stem(t) for t in terms]
        )
        terms = [t if skip else stem_term(t) for t, skip in zip(terms, no_stem)]

    # ---- pass 3 is a *marking* pass: stop words stay in the stream -------
    # (they are excluded by the scorers via is_scoring_stopword, not here)
    return [
        Token(term=t, pos=start_pos + i, sent_id=s, para_id=para_id, zone=zone, raw=r)
        for i, (t, s, r) in enumerate(zip(terms, sent_ids, raws))
    ]


def analyze_terms(text: str, **kw) -> List[str]:
    """Terms only — the form the query normaliser and the df script want."""
    return [t.term for t in analyze(text, **kw)]


def analyze_scoring_terms(text: str, **kw) -> List[str]:
    """Terms with the scoring stop words removed — the tf-idf / BM25F bag.

    Positions are *not* meaningful here; use `analyze()` for anything positional.
    """
    return [t for t in analyze_terms(text, **kw) if not is_scoring_stopword(t)]


# ==========================================================================
# The demo (guide figure: raw -> cleaned -> tokens -> statutes -> stems)
# ==========================================================================
DEMO_SENTENCE = (
    "The Appellant was convicted under Sec. 302 of the Indian Penal Code, 1860 "
    "and sentenced to life imprisonment. The Trial Court also convicted him "
    "u/s 34 and S. 120B, and the Appellants' plea under Art. 21 of the "
    "Constitution was rejected by the High Court following K.S. Puttaswamy v. "
    "Union of India, (2017) 10 SCC 1."
)


def pipeline_trace(text: str, *, stem: bool = True) -> Dict[str, object]:
    """Every intermediate stage, for the demo and for the report figure."""
    cleaned = clean_text(text)
    canon = canonicalise_statutes(cleaned)
    plain = [_fold(r, k) for k, r, _ in _scan(cleaned) if k != "punct"]
    toks_nostem = analyze(text, stem=False)
    toks = analyze(text, stem=stem)
    return {
        "raw": text,
        "cleaned": cleaned,
        "canonicalised": canon,
        "tokens_no_canon": plain,
        "tokens_folded": [t.term for t in toks_nostem],
        "tokens_stemmed": [t.term for t in toks],
        "tokens": toks,
        "statutes": statutes.extract_statutes(cleaned),
        "sentences": sentences(canon),
    }


def _print_demo(text: str, stem: bool = True) -> None:
    tr = pipeline_trace(text, stem=stem)
    bar = "=" * 78

    def head(n: str) -> None:
        print(f"\n{bar}\n{n}\n{bar}")

    head("PASS 0  raw input")
    print(tr["raw"])

    head("PASS 0  cleaned character stream (NFKC, quotes, de-hyphenation)")
    print(tr["cleaned"])

    head("PASS 1  statute canonicalisation  (runs on RAW text, before tokenising)")
    print(tr["canonicalised"])
    print("\n  statute tokens found :", ", ".join(tr["statutes"]) or "(none)")
    for t in tr["statutes"]:
        alts = statutes.crosswalk(t)
        print(f"    {t:<12} = {statutes.pretty(t)}")
        if alts:
            print(f"    {'':<12}   query-time crosswalk -> {', '.join(alts)}")
    print("\n  without pass 1, the same text tokenises to:")
    print("   ", " ".join(tr["tokens_no_canon"][:40]), "...")

    head("PASS 2  tokenise + case fold + sentence split")
    for sid, s in enumerate(tr["sentences"]):
        print(f"  sent {sid}: {s}")
    print()
    print("  tokens:", " ".join(tr["tokens_folded"]))

    head("PASS 3  stop words: KEPT in positions, EXCLUDED from tf-idf")
    toks = tr["tokens"]
    kept = [t.term for t in toks]
    scoring = [t.term for t in toks if not is_scoring_stopword(t.term)]
    print(f"  positional index : {len(kept)} tokens  (every token, so phrases stay exact)")
    print(f"  tf-idf vector    : {len(scoring)} tokens")
    print("  dropped from tf-idf:",
          " ".join(sorted({t.term for t in toks if is_scoring_stopword(t.term)})))

    head("PASS 4  Porter stemming, with exceptions")
    before = tr["tokens_folded"]
    after = tr["tokens_stemmed"]
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    print("  stemmed  :", ", ".join(f"{a} -> {b}" for a, b in changed) or "(nothing)")
    unchanged_interesting = [
        a for a, b in zip(before, after)
        if a == b and (("§" in a) or not a.isalpha() or a in PROTECTED_NAMES
                       or a in NO_STEM_TERMS)
    ]
    print("  NOT stemmed:", ", ".join(dict.fromkeys(unchanged_interesting)) or "(none)")

    head("HOW IT IS INDEXED  (term, pos, sent, zone)   * = not in tf-idf")
    for t in toks:
        mark = "*" if is_scoring_stopword(t.term) else " "
        print(f"  {mark} pos={t.pos:<4} sent={t.sent_id}  {t.term:<16} <- {t.raw!r}")


def _main(argv: Optional[List[str]] = None) -> int:
    from nyaya.utils import ensure_utf8_stdout

    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.analyze")
    ap.add_argument("--demo", nargs="?", const=DEMO_SENTENCE, default=None,
                    help="print the pipeline pass by pass for one sentence")
    ap.add_argument("--no-stem", action="store_true")
    args = ap.parse_args(argv)

    if args.demo is None:
        ap.print_help()
        return 0
    _print_demo(args.demo, stem=not args.no_stem)
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(_main())
