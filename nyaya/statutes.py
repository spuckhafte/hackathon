"""Statute recognition and canonicalisation — the project's equivalence classing.

IR framing (guide Section 4.3)
------------------------------
Term normalisation is the construction of **equivalence classes**: many surface
forms, one index term.  Nowhere in this corpus is that more valuable than in
statute citations, where a single provision appears as

    "Section 302 of the Indian Penal Code, 1860"
    "S. 302 IPC"          "u/s 302 I.P.C."       "sec. 302"
    "IPC 302"             "Section 302"  (act named earlier in the paragraph)

Six surface forms, one concept.  Without canonicalisation a query for ``302 IPC``
matches the minority of documents that happen to use the same spelling, and the
idf of ``302`` is meaningless because it is diluted across every numeric string
in the corpus.  We map all of them onto one token::

    ipc§302

The section sign ``§`` is deliberate: it cannot occur inside a normal word, so
the tokeniser can keep the token intact with a single rule, and it round-trips
through cp1252 so the live demo survives a Windows console.

This module owns the *patterns*; `nyaya.analyze` owns the *pipeline* that calls
them.  Pass 1 of `analyze()` runs on raw text, before tokenisation, because
tokenisation destroys exactly the punctuation these patterns depend on
("I.P.C.", "u/s", "304-B").

Author: Sumedha (M3)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# ==========================================================================
# 1. Act abbreviation table:  surface form  ->  short code
# ==========================================================================
# Each entry is (code, regex source).  The regexes are matched case-insensitively
# against raw judgment text, so they allow the dotted abbreviations Indian
# judgments actually use ("Cr.P.C.", "I.P.C.", "N.D.P.S.").  Order matters only
# for resolution (longest / most specific first) — see ACT_PATTERNS below.

_ACT_SOURCES: List[Tuple[str, str]] = [
    # --- the big four -----------------------------------------------------
    ("IPC", r"(?:indian\s+penal\s+code|penal\s+code)(?:\s*,?\s*1860)?|i\.\s?p\.\s?c\.?|ipc\b"),
    ("CRPC", r"(?:code\s+of\s+criminal\s+procedure|criminal\s+procedure\s+code)"
             r"(?:\s*,?\s*(?:1973|1898))?|cr\.?\s?p\.?\s?c\.?|crpc\b"),
    ("CONST", r"constitution\s+of\s+india(?:\s*,?\s*1950)?|the\s+constitution\b|constitution\b"),
    ("CPC", r"(?:code\s+of\s+civil\s+procedure|civil\s+procedure\s+code)(?:\s*,?\s*1908)?"
            r"|c\.\s?p\.\s?c\.?|cpc\b"),
    # NB: deliberately no "I.E.A." abbreviation — it collides with "i.e. a ...".
    ("EVID", r"(?:indian\s+evidence\s+act|evidence\s+act)(?:\s*,?\s*1872)?"),
    # --- the 2023 codes (in force 1 July 2024) ----------------------------
    ("BNSS", r"bharatiya\s+nagarik\s+suraksha\s+sanhita(?:\s*,?\s*2023)?|bnss\b"),
    ("BNS", r"bharatiya\s+nyaya\s+sanhita(?:\s*,?\s*2023)?|bns\b"),
    ("BSA", r"bharatiya\s+sakshya\s+adhiniyam(?:\s*,?\s*2023)?|bsa\b"),
    # --- heavily litigated special acts -----------------------------------
    ("NDPS", r"narcotic\s+drugs\s+and\s+psychotropic\s+substances\s+act(?:\s*,?\s*1985)?"
             r"|n\.\s?d\.\s?p\.\s?s\.?\s*act|ndps\s*act|ndps\b"),
    ("PCA", r"prevention\s+of\s+corruption\s+act(?:\s*,?\s*(?:1988|1947))?|p\.\s?c\.\s?act"),
    ("UAPA", r"unlawful\s+activities\s*\(?\s*prevention\s*\)?\s*act(?:\s*,?\s*1967)?|uapa\b"),
    ("TADA", r"terrorist\s+and\s+disruptive\s+activities\s*\(?\s*prevention\s*\)?\s*act"
             r"(?:\s*,?\s*(?:1987|1985))?|tada\b"),
    ("POTA", r"prevention\s+of\s+terrorism\s+act(?:\s*,?\s*2002)?|pota\b"),
    ("NI", r"negotiable\s+instruments\s+act(?:\s*,?\s*1881)?|n\.\s?i\.?\s*act"),
    ("COMP", r"(?:indian\s+)?companies\s+act(?:\s*,?\s*(?:1956|2013))?"),
    ("ITA", r"(?:indian\s+)?income[\s-]*tax\s+act(?:\s*,?\s*(?:1961|1922))?|i\.\s?t\.?\s*act"),
    ("MVA", r"motor\s+vehicles\s+act(?:\s*,?\s*(?:1988|1939))?|m\.\s?v\.?\s*act"),
    ("ARB", r"arbitration\s+and\s+conciliation\s+act(?:\s*,?\s*1996)?|arbitration\s+act"),
    ("SARFAESI", r"securitisation\s+and\s+reconstruction\s+of\s+financial\s+assets"
                 r"[^.;:]{0,40}?act(?:\s*,?\s*2002)?|sarfaesi\s*act|sarfaesi\b"),
    ("HMA", r"hindu\s+marriage\s+act(?:\s*,?\s*1955)?|h\.\s?m\.?\s*act"),
    ("LAA", r"land\s+acquisition\s+act(?:\s*,?\s*(?:1894|2013))?"),
    ("CPACT", r"consumer\s+protection\s+act(?:\s*,?\s*(?:1986|2019))?"),
    ("RPA", r"representation\s+of\s+the\s+people\s+act(?:\s*,?\s*(?:1951|1950))?|r\.\s?p\.?\s*act"),
]

#: Human-readable long name per code; used by the UI and the `act:` facet.
ACT_NAMES: Dict[str, str] = {
    "IPC": "Indian Penal Code, 1860",
    "CRPC": "Code of Criminal Procedure, 1973",
    "CONST": "Constitution of India",
    "CPC": "Code of Civil Procedure, 1908",
    "EVID": "Indian Evidence Act, 1872",
    "BNS": "Bharatiya Nyaya Sanhita, 2023",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "BSA": "Bharatiya Sakshya Adhiniyam, 2023",
    "NDPS": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "PCA": "Prevention of Corruption Act, 1988",
    "UAPA": "Unlawful Activities (Prevention) Act, 1967",
    "TADA": "Terrorist and Disruptive Activities (Prevention) Act, 1987",
    "POTA": "Prevention of Terrorism Act, 2002",
    "NI": "Negotiable Instruments Act, 1881",
    "COMP": "Companies Act",
    "ITA": "Income Tax Act, 1961",
    "MVA": "Motor Vehicles Act, 1988",
    "ARB": "Arbitration and Conciliation Act, 1996",
    "SARFAESI": "SARFAESI Act, 2002",
    "HMA": "Hindu Marriage Act, 1955",
    "LAA": "Land Acquisition Act",
    "CPACT": "Consumer Protection Act",
    "RPA": "Representation of the People Act, 1951",
}

#: Flat surface-form -> code table, for documentation, the report and tests.
#: (The regexes above are what the canonicaliser actually runs; this is the
#: readable summary of them.)
ACT_ABBREVIATIONS: Dict[str, str] = {
    "indian penal code": "IPC", "penal code": "IPC", "ipc": "IPC", "i.p.c.": "IPC",
    "indian penal code, 1860": "IPC", "penal code, 1860": "IPC",
    "code of criminal procedure": "CRPC", "criminal procedure code": "CRPC",
    "crpc": "CRPC", "cr.p.c.": "CRPC", "cr. p. c.": "CRPC",
    "constitution of india": "CONST", "the constitution": "CONST", "constitution": "CONST",
    "code of civil procedure": "CPC", "civil procedure code": "CPC",
    "cpc": "CPC", "c.p.c.": "CPC",
    "indian evidence act": "EVID", "evidence act": "EVID",
    "bharatiya nyaya sanhita": "BNS", "bns": "BNS",
    "bharatiya nagarik suraksha sanhita": "BNSS", "bnss": "BNSS",
    "bharatiya sakshya adhiniyam": "BSA", "bsa": "BSA",
    "narcotic drugs and psychotropic substances act": "NDPS", "ndps act": "NDPS",
    "ndps": "NDPS",
    "prevention of corruption act": "PCA",
    "unlawful activities (prevention) act": "UAPA", "uapa": "UAPA",
    "terrorist and disruptive activities (prevention) act": "TADA", "tada": "TADA",
    "prevention of terrorism act": "POTA", "pota": "POTA",
    "negotiable instruments act": "NI",
    "companies act": "COMP",
    "income tax act": "ITA", "income-tax act": "ITA",
    "motor vehicles act": "MVA",
    "arbitration and conciliation act": "ARB", "arbitration act": "ARB",
    "sarfaesi act": "SARFAESI", "sarfaesi": "SARFAESI",
    "hindu marriage act": "HMA",
    "land acquisition act": "LAA",
    "consumer protection act": "CPACT",
    "representation of the people act": "RPA",
}

#: Compiled (code, fullmatch regex) pairs, specific-before-general.  Used to turn
#: a matched act *surface* back into its short code.
ACT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (code, re.compile(r"\s*(?:%s)\s*" % src, re.I))
    for code, src in sorted(_ACT_SOURCES, key=lambda kv: -len(kv[1]))
]

#: One alternation that matches any act surface form.
_ACT = "(?:" + "|".join("(?:%s)" % src for _, src in _ACT_SOURCES) + ")"


def act_code(surface: str) -> Optional[str]:
    """Resolve a matched act surface form (e.g. ``"I.P.C."``) to its code."""
    s = surface.strip().strip(",")
    for code, pat in ACT_PATTERNS:
        if pat.fullmatch(s):
            return code
    return None


# ==========================================================================
# 2. Section / article patterns
# ==========================================================================
# A section number: 302, 302A, 304-B, 34(1), 65B.  The letter suffix must be
# attached to the digits or hyphenated — a *spaced* suffix is accepted only for
# a single capital letter that is not followed by a dot, so that "s. 302 I.P.C."
# does not parse the "I" of I.P.C. as the suffix of section 302.
_SEC_NUM = (
    r"\d{1,4}"
    # the \b after the hyphenated suffix matters: without it, the S.C.R. headnote
    # "Penal Code, 1860-s. 302-Murder" parsed as section "302-Mu".
    r"(?:\s*-\s*[A-Za-z]{1,2}\b|[A-Za-z]{1,2}\b"
    # A *spaced* single capital is accepted ("s. 304 B") but not when the next
    # word is "of": the OCR'd "Article 32 B of the Constitution" is Article 32.
    r"|\s+(?!of\b|and\b|or\b|to\b|read\b|r/w)[A-Z](?![.\w])(?!\s+of\b))?"
    r"(?:\s*\(\s*[0-9A-Za-z]{1,3}\s*\))*"
)

# The words that introduce a section.  "§" is deliberately NOT included: leaving
# it out is what makes canonicalise_statutes() idempotent, since a token that is
# already "ipc§302" must not be re-parsed.
_SECTION_WORD = r"(?:u/s\.?|under\s+sections?|sections?|sects?[.,]?|secs?[.,]?|ss[.,]?|s\.)"
_ARTICLE_WORD = r"(?:articles?|arts?[.,]|art(?=\s*\d))"

# "302, 307 and 34", "302/34", "391 to 393", "420 r/w 34"
_SEP = r"(?:\s*(?:,|;|&|/|and|or|to|r/w|read\s+with)\s*)"
_SEC_LIST = rf"{_SEC_NUM}(?:{_SEP}(?:{_SECTION_WORD}\s*)?{_SEC_NUM})*"

# "Section 4 of the Dowry Prohibition Act" names an act we do not know.  Letting
# the inheritance heuristic fire there would invent "ipc§4", so a bare section
# followed by "of (the) ... Act/Code/Rules" is left alone.  The guard is narrow:
# the act phrase must start with "of" and contain no sentence-ending punctuation,
# so an ordinary "...under Section 34.  The Act provides..." is unaffected.
# PDF text wraps mid-phrase, so the gap must tolerate a single newline ("of the
# Finance \nAct, 1920" was a measured false positive) but not a blank line, which
# would mean the "Act" belongs to the next paragraph.
_GAP = r"(?:[^.;:\n]|\n(?!\s*\n)){0,60}?"
_UNKNOWN_ACT_AHEAD = (
    rf"(?!\s*,?\s*of\s+(?:the\s+)?{_GAP}"
    r"\b(?:act|code|sanhita|adhiniyam|rules?|regulations?|ordinance|constitution)\b)"
)

# "Middleton v. Texas Power and Light Co. (249 U.S. 152)" — the "S. 152" there is
# a US Reports page, not a section.  A bare "s."/"ss." immediately preceded by
# another dotted initial is part of a reporter abbreviation, never a section.
_NOT_IN_ABBREV = r"(?<![A-Za-z]\.)"

#: What may sit between a section list and the name of its act.  Judgments write
#: "s. 302 of the Indian Penal Code", "s. 302, IPC", and in the S.C.R. headnote
#: style "Penal Code, 1860-s. 302-Murder".  The optional "indian" matters: the
#: measured recall misses were dominated by "of the Indian Income-tax Act".
_CONNECTOR = r"\s*(?:[,\-–:;]\s*)?(?:of\s+)?(?:the\s+)?(?:indian\s+)?"

#: "of the Constitution", "of our Constitution", "of the Indian Constitution".
_CONSTITUTION_AHEAD = (
    r"\s*(?:[,\-–]\s*)?(?:of\s+)?(?:the|our|its|this)?\s*"
    r"(?:indian\s+)?constitution(?:\s+of\s+india)?"
)

#: Like _UNKNOWN_ACT_AHEAD but WITHOUT "constitution" in the act-word list.
#: The Article shape needs its own: the shared guard lists "constitution" as an
#: act word, which made it block the very phrase it should accept
#: ("article 14 of our Constitution" was a measured recall miss).
_OTHER_ACT_AHEAD = (
    rf"(?!\s*,?\s*of\s+(?:the\s+)?{_GAP}"
    r"\b(?:act|code|sanhita|adhiniyam|rules?|regulations?|ordinance)\b)"
)

#: "the Constitution 42nd Amendment" must not read as Article 42nd.
_NOT_ORDINAL = r"(?!\d+(?:st|nd|rd|th)\b)"

#: Order VII Rule 11 CPC and friends — a CPC/CrPC citation shape that has no
#: section number at all.
_ORDER_RULE = (
    r"\border\s*(?:no\.?\s*)?(?P<ord_num>[ivxl]+|\d{1,2})\s*,?\s*"
    r"(?:r\.|rules?)\s*(?P<rule_num>\d{1,3}[a-z]?)(?![0-9A-Za-z])"
)

MASTER = re.compile(
    rf"""
      (?P<sec_act>\b{_SECTION_WORD}\s*(?P<sa_list>{_SEC_LIST})
                  {_CONNECTOR}(?P<sa_act>{_ACT}))
    | (?P<art>\b{_ARTICLE_WORD}\s*{_NOT_ORDINAL}
              (?P<ar_list>(?>{_SEC_LIST}))(?![0-9A-Za-z'’])
              (?:{_CONSTITUTION_AHEAD}|{_OTHER_ACT_AHEAD}))
    | (?P<ordrule>{_ORDER_RULE}
                  (?:{_CONNECTOR}(?P<or_act>{_ACT})|{_UNKNOWN_ACT_AHEAD}))
    | (?P<act_sec>\b(?P<as_act>{_ACT})\s*(?:[,\-–:;]\s*)?
                  {_SECTION_WORD}\s*(?P<as_list>{_SEC_LIST}))
    | (?P<act_num>\b(?P<an_act>{_ACT})\s*,?\s*{_NOT_ORDINAL}(?P<an_list>\d{{1,3}}[A-Za-z]{{0,2}})\b)
    | (?P<act_only>\b(?P<ao_act>{_ACT}))
    | (?P<sec_only>{_NOT_IN_ABBREV}\b{_SECTION_WORD}\s*
                   (?P<so_list>(?>{_SEC_LIST}))(?![0-9A-Za-z])
                   {_UNKNOWN_ACT_AHEAD})
    """,
    re.I | re.X,
)

_SEC_NUM_RE = re.compile(_SEC_NUM, re.I)
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}

#: The canonical token shape, for everyone downstream.
STATUTE_TOKEN_RE = re.compile(r"[a-z]{2,10}§[0-9]+[a-z]*(?:o[0-9]+r[0-9]+[a-z]?)?")
CANON_TOKEN_RE = re.compile(r"\b[a-z]{2,10}§[0-9a-z]+\b")


def _roman_value(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s.lower()):
        v = _ROMAN.get(ch, 0)
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def _int_to_roman(n: int) -> str:
    out, table = "", ((100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
                      (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))
    for v, sym in table:
        while n >= v:
            out += sym
            n -= v
    return out


def _roman_to_int(s: str) -> int:
    """Roman numeral -> int, tolerating the one OCR error that matters.

    Scanned S.C.R. volumes routinely print "Order XXIII" as "Order XXlll" —
    lower-case L for capital I.  Read literally that is 130, which silently
    produced ``cpc§o130r3`` in the measurement.  So: accept the reading only if
    it round-trips to the same numeral; otherwise retry with l->i, and give up
    (return 0, i.e. no token) rather than emit a wrong Order number.  CPC Orders
    stop at LI, so anything above 60 is an error by definition.
    """
    for cand in (s.lower(), s.lower().replace("l", "i")):
        n = _roman_value(cand)
        if 0 < n <= 60 and _int_to_roman(n) == cand:
            return n
    return 0


def normalise_section(raw: str) -> str:
    """``"304 - B"`` -> ``"304b"``, ``"34(1)"`` -> ``"34"``, ``"302A"`` -> ``"302a"``.

    Sub-clauses are dropped on purpose: a judgment that cites 34(1) and one that
    cites 34 are talking about the same provision for retrieval purposes, and
    keeping the clause would split the postings list.
    """
    s = raw.strip().lower()
    s = re.sub(r"\(.*?\)", "", s)          # drop sub-clauses
    s = re.sub(r"[\s\-.]", "", s)          # "304 - b" -> "304b"
    return s


def canonical_token(act: str, section: str) -> str:
    """``("IPC", "302A")`` -> ``"ipc§302a"``."""
    return f"{act.lower()}§{normalise_section(section)}"


def _expand_list(span: str, act: str) -> List[str]:
    """Turn ``"302, 307 and 34"`` into ``[ipc§302, ipc§307, ipc§34]``."""
    out = []
    for m in _SEC_NUM_RE.finditer(span):
        sec = normalise_section(m.group(0))
        if sec:
            out.append(f"{act.lower()}§{sec}")
    return out


# ==========================================================================
# 3. The canonicaliser
# ==========================================================================
#: Default for the act-inheritance heuristic.  `analyze` exposes it as a flag so
#: the ablation ("how much does inheritance buy?") is a measurement, not a guess.
INHERIT_ACT_DEFAULT = True

#: How far a named act may reach forward to a bare section number, in characters
#: of the same paragraph.  Measured: without a limit, a bare "section 2" five
#: hundred characters after an unrelated act reference was the single largest
#: source of wrong canonical tokens.  ~700 characters is about three sentences,
#: which is the realistic scope of "the act we are talking about".
INHERIT_MAX_CHARS = 700

#: A *section* belongs to an Act; a *provision of the Constitution* is an
#: Article.  Judgments obey this absolutely, so a bare "section 302" must never
#: inherit CONST however recently the Constitution was named.  This single rule
#: removed most of the measured inheritance false positives, and it is the kind of
#: domain fact a generic normaliser cannot know.
NEVER_INHERIT = frozenset({"CONST"})

#: Seed act for the *query* side only.  A user who types "sec. 302" or "s. 420"
#: with no act almost always means the Penal Code, so the query normaliser passes
#: ``default_act=QUERY_FALLBACK_ACT``.  Document text gets ``default_act=None``:
#: guessing an act for an indexed judgment would fabricate citations, whereas
#: guessing for a query only widens the candidate set, which recall can absorb.
QUERY_FALLBACK_ACT = "IPC"

_PARA_SPLIT = re.compile(r"(\n[ \t]*\n+)")


def canonicalise_statutes(
    text: str,
    *,
    inherit_act: bool = INHERIT_ACT_DEFAULT,
    default_act: Optional[str] = None,
) -> str:
    """Replace every statute mention span with one canonical token.

    Runs on **raw text**, before tokenisation, because the patterns depend on
    punctuation the tokeniser throws away ("I.P.C.", "u/s", "304-B").

    The act-inheritance heuristic
    -----------------------------
    Judgments name the act once and then write bare section numbers for pages::

        "...convicted under Section 302 of the Indian Penal Code, 1860.
         The Trial Court also found him guilty under Section 34 and
         Section 120B."

    A bare "Section 34" therefore inherits **the most recently named act in the
    same paragraph**.  The state is reset at every blank-line paragraph break,
    because carrying an act across paragraph boundaries is where the heuristic
    starts inventing citations.  Set ``inherit_act=False`` to switch it off and
    measure it; with it off, a bare section with no act is left as ordinary
    text ("section 34") rather than guessed at.

    ``default_act`` seeds the context (used by the ``statutes`` zone builder,
    where a headnote line like "Penal Code, 1860: s. 302" has already told us
    the act).
    """
    if not text:
        return text

    out_parts: List[str] = []
    for chunk in _PARA_SPLIT.split(text):
        if _PARA_SPLIT.fullmatch(chunk):      # the separator itself
            out_parts.append(chunk)
            continue
        out_parts.append(_canonicalise_paragraph(chunk, inherit_act, default_act))
    return "".join(out_parts)


@dataclass
class StatuteSpan:
    """One recognised statute mention: where it was, and what it became.

    Exposed so that (a) the precision/recall measurement can hand-check real
    matches against their surrounding text, and (b) M1's snippet generator can
    highlight the original wording while the index holds the canonical token.
    """

    start: int
    end: int
    surface: str
    tokens: List[str]
    shape: str                 # which pattern fired
    act: Optional[str]
    inherited: bool = False    # the act came from the paragraph, not this span


def iter_statute_spans(
    text: str,
    *,
    inherit_act: bool = INHERIT_ACT_DEFAULT,
    default_act: Optional[str] = None,
) -> List[StatuteSpan]:
    """Every statute mention in ``text``, with offsets into ``text``."""
    spans: List[StatuteSpan] = []
    offset = 0
    for chunk in _PARA_SPLIT.split(text):
        if not _PARA_SPLIT.fullmatch(chunk):
            for sp in _paragraph_spans(chunk, inherit_act, default_act):
                sp.start += offset
                sp.end += offset
                spans.append(sp)
        offset += len(chunk)
    return spans


#: Words that, standing immediately before "Article(s)", mean the articles belong
#: to some other instrument.  Measured false positives: "Portuguese Civil
#: Code-Articles 474, 505, 510" and "Civil Service Regulations ... Article 465A".
_OTHER_INSTRUMENT_BEHIND = re.compile(
    r"\b(?:code|act|regulations?|rules?|convention|treaty|charter|statute|"
    r"ordinance|agreement|articles\s+of\s+association|memorandum)\s*"
    r"[-–—,:;()\s]{0,4}$",
    re.I,
)


#: An instrument named *immediately* before a bare section number, with only
#: punctuation, a year or whitespace in between — the S.C.R. headnote shape
#: "... Rules, 1957 - Section 43".
_INSTRUMENT_ADJACENT = re.compile(
    r"\b(?:act|code|rules?|regulations?|ordinance|sanhita|adhiniyam|scheme|"
    r"bye-?laws?|order)\b[\s,.:;0-9()\-–]*$",
    re.I,
)
_ANY_ACT = re.compile(_ACT, re.I)


def _unknown_instrument_behind(text: str, start: int) -> bool:
    """Is a bare section number sitting right after an instrument we do not know?

    S.C.R. headnotes read "Kerala State Electricity Board Rules, 1957 - Section
    43/Rule 68".  Inheriting whatever act was named earlier in the paragraph
    files that under the wrong statute, which is what the measurement showed.
    So if an instrument is named immediately before the section and it is not one
    we recognise, the inheritance heuristic stands down.  When it *is*
    recognised ("Indian Penal Code, 1860: Sections 53-A, 55 and 302")
    inheritance proceeds, and when no instrument is adjacent at all
    ("Code of Criminal Procedure (Act V of 1898) ss. 439, 209") this test says
    nothing and the other guards decide.
    """
    window = text[max(0, start - 30):start]
    if not _INSTRUMENT_ADJACENT.search(window):
        return False
    return not _ANY_ACT.search(window)


def _preceded_by_other_instrument(text: str, start: int) -> bool:
    window = text[max(0, start - 60):start]
    if re.search(r"\bconstitution\b\s*[-,:;()\s]{0,4}$", window, re.I):
        return False
    return bool(_OTHER_INSTRUMENT_BEHIND.search(window))


def _paragraph_spans(
    text: str, inherit_act: bool, default_act: Optional[str]
) -> List[StatuteSpan]:
    current_act: Optional[str] = default_act
    act_at = -10 ** 9 if default_act is None else 0
    spans: List[StatuteSpan] = []

    for m in MASTER.finditer(text):
        toks: List[str] = []
        replace = True

        if m.group("sec_act") is not None:
            act = act_code(m.group("sa_act"))
            if act:
                current_act, act_at = act, m.end()
                toks = _expand_list(m.group("sa_list"), act)
            else:
                replace = False
        elif m.group("art") is not None:
            if _preceded_by_other_instrument(text, m.start()):
                continue            # "Portuguese Civil Code - Articles 474, 505"
            act, current_act, act_at = "CONST", "CONST", m.end()
            toks = _expand_list(m.group("ar_list"), "CONST")
        elif m.group("ordrule") is not None:
            act = act_code(m.group("or_act")) if m.group("or_act") else current_act
            if act in (None, "CONST"):
                act = act or "CPC"
            o = m.group("ord_num")
            o_i = _roman_to_int(o) if not o.isdigit() else int(o)
            rule = m.group("rule_num").lower()
            if o_i:
                current_act, act_at = act, m.end()
                toks = [f"{act.lower()}§o{o_i}r{rule}"]
            else:
                replace = False
        elif m.group("act_sec") is not None:
            act = act_code(m.group("as_act"))
            if act:
                current_act, act_at = act, m.end()
                toks = _expand_list(m.group("as_list"), act)
            else:
                replace = False
        elif m.group("act_num") is not None:
            act = act_code(m.group("an_act"))
            if act:
                current_act, act_at = act, m.end()
                toks = _expand_list(m.group("an_list"), act)
            else:
                replace = False
        elif m.group("act_only") is not None:
            act = act_code(m.group("ao_act"))
            if act:
                current_act, act_at = act, m.end()   # context only; words stay
            replace = False
        else:                               # sec_only
            act = current_act if inherit_act else default_act
            if act in NEVER_INHERIT and act is not default_act:
                act = None              # a section is never a constitutional one
            elif act is current_act and m.start() - act_at > INHERIT_MAX_CHARS:
                act = None              # the act was named too long ago to trust
            elif act is current_act and _unknown_instrument_behind(text, m.start()):
                act = None              # this section belongs to some other Act
            if act:
                toks = _expand_list(m.group("so_list"), act)
            else:
                replace = False             # no act known: leave it as plain text

        if not replace or not toks:
            continue
        shape = next(k for k in ("sec_act", "art", "ordrule", "act_sec",
                                 "act_num", "sec_only")
                     if m.group(k) is not None)
        spans.append(StatuteSpan(
            start=m.start(), end=m.end(), surface=m.group(0), tokens=toks,
            shape=shape, act=act, inherited=(shape == "sec_only"),
        ))
    return spans


def _canonicalise_paragraph(
    text: str, inherit_act: bool, default_act: Optional[str]
) -> str:
    out: List[str] = []
    last = 0
    for sp in _paragraph_spans(text, inherit_act, default_act):
        out.append(text[last:sp.start])
        out.append(" " + " ".join(sp.tokens) + " ")
        last = sp.end
    out.append(text[last:])
    return "".join(out)


# ==========================================================================
# 4. Extraction helpers (the `statutes` zone and the `act:` parametric field)
# ==========================================================================


def extract_statutes(text: str, *, inherit_act: bool = INHERIT_ACT_DEFAULT) -> List[str]:
    """Canonical statute tokens present in ``text``, de-duplicated, in order.

    This is what builds the synthetic ``statutes`` zone (config.ZONES) and the
    ``act:`` parametric field.
    """
    seen, out = set(), []
    for m in CANON_TOKEN_RE.finditer(canonicalise_statutes(text, inherit_act=inherit_act)):
        t = m.group(0)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extract_acts(text: str, **kw) -> List[str]:
    """Short act codes present, de-duplicated: ``["IPC", "CONST"]``."""
    seen, out = set(), []
    for tok in extract_statutes(text, **kw):
        a = ACT_OF(tok)
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def ACT_OF(token: str) -> str:
    """``"ipc§302"`` -> ``"IPC"``.  Empty string if it is not a statute token."""
    if "§" not in token:
        return ""
    return token.split("§", 1)[0].upper()


def SECTION_OF(token: str) -> str:
    """``"ipc§302a"`` -> ``"302a"``."""
    if "§" not in token:
        return ""
    return token.split("§", 1)[1]


def is_statute_token(term: str) -> bool:
    return "§" in term


def pretty(token: str) -> str:
    """``"ipc§302"`` -> ``"Section 302, Indian Penal Code, 1860"`` for the UI."""
    act, sec = ACT_OF(token), SECTION_OF(token)
    if not act:
        return token
    name = ACT_NAMES.get(act, act)
    if sec.startswith("o") and "r" in sec:
        o, r = sec[1:].split("r", 1)
        return f"Order {o} Rule {r}, {name}"
    label = "Article" if act == "CONST" else "Section"
    return f"{label} {sec.upper()}, {name}"


# ==========================================================================
# 5. IPC <-> BNS crosswalk  (QUERY TIME ONLY — never at index time)
# ==========================================================================
# Why query time only: the index must record what a judgment actually says.  A
# 1985 judgment never mentioned the Bharatiya Nyaya Sanhita, so writing bns§103
# into its postings would be a lie that corrupts df/idf for every BNS term.
# Instead the query normaliser expands  ipc§302 -> (ipc§302 OR bns§103)  as a
# synonym class at query time — classic query expansion, and it is reversible,
# measurable and switchable.
#
# The 2023 codes replaced the IPC, CrPC and Evidence Act with effect from
# 1 July 2024, so a user searching "302 IPC" and a user searching "103 BNS"
# want the same body of law, split across the commencement date.
#
# SOURCES — every mapping below was checked against a fetched source, not from
# memory.  See CROSSWALK_SOURCES.  Mappings that are not one-to-one carry a
# list of targets and a note in CROSSWALK_NOTES.

CROSSWALK_SOURCES: List[str] = []

#: old canonical token -> list of new canonical tokens (and the reverse, built
#: below).  A list, not a string, because the mapping is frequently 1 -> many.
CROSSWALK: Dict[str, List[str]] = {}

#: Provisions with no counterpart in the new codes, e.g. repealed outright.
CROSSWALK_OMITTED: Dict[str, str] = {}

#: Free-text caveats shown in the "why this result" panel when an expansion
#: fired on a non-one-to-one mapping.
CROSSWALK_NOTES: Dict[str, str] = {}

_REVERSE_CROSSWALK: Dict[str, List[str]] = {}


def _build_reverse() -> None:
    _REVERSE_CROSSWALK.clear()
    for old, news in CROSSWALK.items():
        for new in news:
            _REVERSE_CROSSWALK.setdefault(new, [])
            if old not in _REVERSE_CROSSWALK[new]:
                _REVERSE_CROSSWALK[new].append(old)


def crosswalk(token: str) -> List[str]:
    """Equivalent statute tokens in the *other* code, for query expansion.

    ``crosswalk("ipc§302") -> ["bns§103"]`` and ``crosswalk("bns§103") ->
    ["ipc§302"]``.  Returns ``[]`` when there is no verified mapping — we never
    guess, because a wrong expansion silently retrieves the wrong offence.
    """
    token = token.strip().lower()
    if token in CROSSWALK:
        return list(CROSSWALK[token])
    if token in _REVERSE_CROSSWALK:
        return list(_REVERSE_CROSSWALK[token])
    return []


def crosswalk_note(token: str) -> str:
    """Human-readable caveat for a non-one-to-one or omitted mapping."""
    token = token.strip().lower()
    if token in CROSSWALK_OMITTED:
        return CROSSWALK_OMITTED[token]
    return CROSSWALK_NOTES.get(token, "")


def expand_statutes(tokens: Iterable[str]) -> Dict[str, List[str]]:
    """``["ipc§302"]`` -> ``{"ipc§302": ["bns§103"]}`` — feeds ParsedQuery.expansions."""
    out: Dict[str, List[str]] = {}
    for t in tokens:
        alts = crosswalk(t)
        if alts:
            out[t] = alts
    return out


from nyaya._crosswalk_table import (  # noqa: E402  (data table, kept separate)
    CROSSWALK as _CW,
    CROSSWALK_NOTES as _CWN,
    CROSSWALK_OMITTED as _CWO,
    SOURCES as _CWS,
)

CROSSWALK.update(_CW)
CROSSWALK_NOTES.update(_CWN)
CROSSWALK_OMITTED.update(_CWO)
CROSSWALK_SOURCES.extend(_CWS)
_build_reverse()
