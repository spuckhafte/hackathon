"""Zone segmentation and paragraph numbering  (Member 1, wave 1; guide S4.4).

Indian judgments carry no section markup, so the six zones of
``config.ZONES`` have to be recovered from typography.  Everything here was
written against measurements on the real corpus, not from a style guide; the
numbers quoted in the comments come from ``scripts/zone_audit.py``.

What the printed Supreme Court Reports actually give us, on every era of the
collection:

    [2024] 10 S.C.R. 150 : 2024 INSC 749        <- scr_cite : neutral_cite
    Atul Kumar                                  <- petitioner
    v.                                          <- the party separator
    The Chairman (Joint Seat Allocation ...)     <- respondent
    (Writ Petition (Civil) No 609 of 2024)       <- case_no  -> case_type
    30 September 2024                            <- date
    [Dr Dhananjaya Y Chandrachud, CJI, J B Pardiwala and
     Manoj Misra, JJ.]                           <- the bench, and its size
    ... headnote, then the judgment ...
    Result of the Case: Writ Petition disposed of.

That five-to-eight line preamble is the *header region*.  It is the only place
the bench is stated, and it is the authoritative source of bench size: the
metadata ``judge`` column names a single judge on 37k of 42k rows, so trusting
it would collapse every Constitution Bench to a single-judge court and wreck
the bench component of g(d).

Zones produced here:

``title``    the metadata title, else the header lines above the ``v.``
``parties``  petitioner / respondent, split on ``v.`` / ``versus`` / ``vs``
``bench``    the ``[... , JJ.]`` line or a ``CORAM`` line; sets ``bench_size``
``statutes`` left empty - a synthetic keyword zone that Member 3's
             ``analyze.py`` fills in wave 2 through :func:`attach_statute_zone`
``order``    the operative paragraphs at the end, found by disposal cues
``body``     everything else

The header region is *removed* from ``body`` on purpose, so that the case title
and the judges' names are counted once, in their own zone, instead of also
inflating ``body`` term frequencies that the BM25F weights then double-count.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from nyaya import config
from nyaya.contracts import JudgmentRecord, Para

# ==========================================================================
# 1. Header-region patterns
# ==========================================================================

#: How many lines at the top of the document we treat as the preamble.  Measured
#: max over 30 audited documents is 11; 24 leaves slack for OCR noise lines.
HEADER_LINES = 24

#: The judge line is searched over a much wider window than the rest of the
#: preamble.  Reason, measured: the pre-1990 PDFs are page-range scans of a
#: bound SCR volume, so a file often opens with the *tail* of the previous
#: judgment and the real header starts 20-80 lines in.  Widening is safe because
#: find_bench() validates that a candidate ends in a judicial title, so a
#: bracketed citation like "[1950] 1 S.C.R. 15" can never be mistaken for one.
BENCH_SEARCH_LINES = 140

#: "[2017] 11 S.C.R. 1029".  The volume number is often OCR'd ("IO" for "10"),
#: so the volume is matched loosely and the year/page strictly.
SCR_RE = re.compile(
    r"\[(?P<year>1[89]\d\d|20\d\d)\]\s*(?P<vol>[0-9IOlSB]{1,3})?\s*"
    r"S\s*\.?\s*C\s*\.?\s*R\s*\.?\s*(?P<page>\d{1,4})",
    re.I,
)
#: "2024 INSC 735" - the Court's own neutral citation, post-2023 in the reports.
NEUTRAL_RE = re.compile(r"\b(?P<year>1[89]\d\d|20\d\d)\s*INSC\s*(?P<num>\d{1,5})\b", re.I)

#: A line that is only a party separator.
PARTY_SEP_RE = re.compile(r"^\s*(?:v\s*[./]?|vs\s*\.?|versus)\s*$", re.I)
#: ... and the same thing inside a one-line title.
PARTY_SPLIT_RE = re.compile(r"\s+(?:v\s*[./]|vs\s*\.?|versus)\s+", re.I)

#: "(Civil Appeal No. 238 of 2012)", "(Writ Petition (Civil) No 609 of 2024)",
#: "Criminal Appeal No(s). 2161-2162 of 2024".  OCR frequently eats the space in
#: "of2012", so the year is matched without requiring one.
CASE_NO_RE = re.compile(
    r"(?P<body>(?:Civil|Criminal|Writ|Special\s+Leave|Review|Transfer(?:red)?|"
    r"Contempt|Election|Arbitration|Original|Suo\s*Motu)"
    r"[A-Za-z()\s./,&'-]{0,60}?"
    r"No\s*\(?s?\)?\s*\.?\s*[0-9]{1,6}(?:\s*[-/]\s*[0-9]{1,6})?"
    r"(?:\s*of\s*(?:19|20)\d\d)?)",
    re.I,
)

#: Case type, looked up in the case number / header.  Order matters: the more
#: specific phrase has to be tested first ("special leave petition" contains
#: "petition", "criminal appeal" contains "appeal").
CASE_TYPE_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("special leave petition", re.compile(r"special\s+leave\s+(?:petition|to\s+appeal)|\bSLP\b", re.I)),
    ("review petition", re.compile(r"review\s+petition", re.I)),
    ("transfer petition", re.compile(r"transfer(?:red)?\s+(?:petition|case)", re.I)),
    ("criminal appeal", re.compile(r"crimina?l\s+appeal", re.I)),
    ("civil appeal", re.compile(r"civil\s+appeal", re.I)),
    ("writ petition", re.compile(r"writ\s+petition", re.I)),
)

#: Dates as the reports print them, in both era styles.
_MONTHS = ("january february march april may june july august september "
           "october november december").split()
_MONTH_ID = {m: i + 1 for i, m in enumerate(_MONTHS)}
DATE_RE = re.compile(
    r"\b(?:(?P<d1>\d{1,2})\s*(?:st|nd|rd|th)?\s+(?P<m1>[A-Za-z]{3,12})\s*,?\s*(?P<y1>(?:19|20)\d\d)"
    r"|(?P<m2>[A-Za-z]{3,12})\s+(?P<d2>\d{1,2})\s*,?\s*(?P<y2>(?:19|20)\d\d))\b"
)

# --- bench ----------------------------------------------------------------
#: Judicial titles, which are not names and must not be counted as judges.
_TITLE_TOKENS = frozenset("""
j jj cj cji c.j c.j. c.j.i cjs js jj. j. cj. cji. actingcj
""".split())
#: The plural "JJ." is OCR'd as "IJ.", "lJ." or "JI." all over the scanned
#: volumes (measured: 4 of 33 audited documents), so every variant is spelled
#: out once here and reused by the terminator scan below.
_JJ_ALT = r"(?:JJ|IJ|lJ|JI)"
_TITLE_TAIL_RE = re.compile(
    r"[,;\s]*\b(?:" + _JJ_ALT + r"|J|CJ|CJI|C\s*\.?\s*J\s*\.?\s*I?)\s*\.?\s*\*?\s*$")
_CORAM_RE = re.compile(r"^\s*(?:CORAM|BEFORE|BENCH)\s*[:\-]\s*(?P<body>.+)$", re.I | re.M)
#: Where a judge-line ends.  OCR turns the closing "]" into ")" or even "J".
_BENCH_CLOSERS = ("]", ")")
#: ...and the volumes are equally inconsistent about the *opening* delimiter:
#: "[S. R. DAS C.J., ...]" in some years, "(S. R. DAS C.J., ...)" in others.
_BENCH_OPENERS = ("[", "(")
#: Longest plausible judge line.  The widest real bench (Kesavananda, 13 judges)
#: fits in ~220 characters; anything longer has swallowed a neighbouring field.
MAX_BENCH_CHARS = 250
#: "X v. Y" written out, used only to reject a bench candidate that has
#: swallowed the party line.  Case-SENSITIVE and lower-case on purpose: the
#: reports print the separator as "v.", while "SUJATA V. MANOHAR" is a judge's
#: own initial, and matching case-insensitively would throw her bench away.
_BENCH_REJECT_PARTY = re.compile(r"\s+(?:v\.|vs\.?|versus)\s+")
_BENCH_NAME_SPLIT = re.compile(r",|;|\band\b|\bAND\b|&", re.I)
_HAS_NAME = re.compile(r"[A-Za-z]{2,}")

# --- paragraph numbering --------------------------------------------------
#: "12." or "12)" at the start of a line, optionally followed by a tab.  The
#: reports print the number flush left; a number inside a sentence never is.
PARA_NUM_RE = re.compile(r"^[ \t]{0,6}(?P<n>\d{1,3})\s*[.)]\s*(?P<rest>.*)$")
#: The first numbered paragraph often shares its line with the author credit:
#: "The Judgment of the Court was delivered by A. K. SIKRI, J.  1.  All the ..."
FIRST_PARA_INLINE_RE = re.compile(r"\b(?:J|JJ|CJ|CJI)\s*\.?\s*[:\-]?\s+1\s*[.)]\s+")

#: Fallback paragraph size, in words.  PyMuPDF returns no blank lines at all for
#: these PDFs (measured: 0 blank lines in 18/18 audited documents across six
#: eras), so "split on blank lines" is not available and we window over complete
#: sentences instead - the unit stays readable and `/p` proximity stays honest.
FALLBACK_PARA_WORDS = 110
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

# --- order zone -----------------------------------------------------------
#: Disposal cues, i.e. the sentence that starts the operative part.  All of
#: these were lifted from documents in this corpus, not invented.
ORDER_CUES = (
    "in the result",
    "for the foregoing reasons",
    "for all the foregoing reasons",
    "for all these reasons",
    "for the reasons aforesaid",
    "for the reasons stated above",
    "in view of the foregoing",
    "in view of the above discussion",
    "in the aforesaid view of the matter",
    "accordingly, the appeal",
    "accordingly, the appeals",
    "accordingly, the petition",
    "accordingly, we",
    "result of the case",
    "the appeal is allowed",
    "the appeal is dismissed",
    "the appeals are allowed",
    "the appeals are dismissed",
    "the appeal therefore fails",
    "the appeal accordingly",
    "this appeal is dismissed",
    "the writ petition is dismissed",
    "the writ petitions are dismissed",
    "the petition is dismissed",
    "the special leave petition is dismissed",
    "stand disposed of",
    "stands disposed of",
    "are disposed of",
    "is disposed of",
    "we allow the appeal",
    "we dismiss the appeal",
    "we dismiss these appeals",
    "we therefore allow",
    "we therefore dismiss",
    "the conviction and sentence",
    "appeal dismissed",
    "appeal allowed",
    "appeals dismissed",
    "petitions dismissed",
    "no order as to costs",
)
#: Only the tail of the judgment is eligible to be the order: a cue that fires
#: in the headnote (they do, all the time) must not swallow the whole document.
ORDER_TAIL_SHARE = 0.25
ORDER_MAX_PARAS = 3

_NORM_UNION = re.compile(r"\bunion\s+of\s+india\b", re.I)
_NORM_STATE = re.compile(r"\bstate\s+of\s+([A-Z][A-Za-z.\s]{2,30})", re.I)
_WORD_RE = re.compile(r"[A-Za-z]+|[^\W\d_]+", re.UNICODE)


# ==========================================================================
# 2. Header parsing
# ==========================================================================

@dataclass
class Header:
    """What the preamble of a judgment yields."""

    scr_cite: str = ""
    neutral_cite: str = ""
    case_no: str = ""
    case_type: str = ""
    date: str = ""
    judges: List[str] = field(default_factory=list)
    bench_text: str = ""
    bench_source: str = ""        # 'bracket' | 'coram' | 'none'
    petitioner: str = ""
    respondent: str = ""
    title_lines: List[str] = field(default_factory=list)
    end_line: int = 0             # first body line index


#: Honorifics the reports print in front of a name.  "Dr" is deliberately kept,
#: because Member 2's metadata keeps it too and stripping it on one side only
#: would turn one judge into two in the parametric index.
_HONORIFIC_RE = re.compile(
    r"^(?:SHRI|SHRIMATI|SMT|HON'?BLE|HONOURABLE|JUSTICE|MR|MRS)\b\.?\s*", re.I)


def _clean_name(frag: str) -> str:
    frag = frag.replace("*", " ").strip(" .,:;-_'\"\t")
    frag = _HONORIFIC_RE.sub("", frag).strip()
    frag = re.sub(r"\s{2,}", " ", frag)
    frag = _TITLE_TAIL_RE.sub("", frag).strip(" .,:;-_")
    return frag


def parse_judges(bench_text: str) -> List[str]:
    """Split a judge line into names, dropping the judicial titles.

    ``[Dr Dhananjaya Y Chandrachud, CJI, J B Pardiwala and Manoj Misra, JJ.]``
    -> three names.  The trailing ``*`` marks the authoring judge and is
    stripped (it is not part of the name, and leaving it in would make the same
    judge two different people in the parametric index).
    """
    text = bench_text.strip().strip("[]()")
    text = _TITLE_TAIL_RE.sub("", text)
    names: List[str] = []
    for frag in _BENCH_NAME_SPLIT.split(text):
        name = _clean_name(frag)
        if not name or not _HAS_NAME.search(name):
            continue
        flat = name.replace(" ", "").replace(".", "").lower()
        if flat in _TITLE_TOKENS or len(flat) < 3:
            continue
        if flat in ("shri", "smt", "dr", "justice", "honble", "hon'ble"):
            continue
        names.append(name)
    # de-duplicate, preserving order
    seen = set()
    out = []
    for n in names:
        k = n.lower()
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


#: Plural "JJ." - the terminator of a multi-judge line.  Tried before the
#: singular form so that "A. N. RAY C.J., ... AND P. N. BHAGWATI, JJ." is not
#: truncated to one judge at the "C.J." in the first name.
_END_JJ_RE = re.compile(r"\b" + _JJ_ALT + r"\s*\.")
#: Singular "J." for a single-judge line, excluding the "J" of "C.J."/"CJ".
_END_J_RE = re.compile(r"(?<![A-Za-z])(?<!C\.)J\s*\.")


def _bench_end_candidates(window: str) -> List[int]:
    """Cut positions to try, in order of how much we trust them.

    A real closing bracket first, then the OCR fallbacks.  Needed because the
    scanned volumes routinely print ``JJ.J`` where the typesetter meant
    ``JJ.]`` - so insisting on a bracket loses the bench on roughly a quarter of
    pre-1990 documents, while accepting *any* "J." truncates multi-judge lines
    at the Chief Justice's own "C.J.".
    """
    cuts: List[int] = []
    for closer in _BENCH_CLOSERS:
        i = window.find(closer)
        if i >= 0:
            cuts.append(i)
    m = _END_JJ_RE.search(window)
    if m:
        cuts.append(m.end())
    m = _END_J_RE.search(window)
    if m:
        cuts.append(m.end())
    out: List[int] = []
    for c in cuts:                     # keep order, drop duplicates
        if c not in out:
            out.append(c)
    return out


def find_bench(header_text: str) -> Tuple[str, str]:
    """Return ``(bench_text, source)`` for the first judge line in the preamble.

    Scanned by hand rather than by one monster regex, because the closing
    bracket is unreliable: the reports print ``[... , JJ.]`` but the OCR of the
    older volumes yields ``JJ.)`` and even ``JJ.J``.  We therefore find a ``[``,
    read forward to the first ``]`` / ``)``, and *validate* that the captured
    text ends in a judicial title - that validation is what keeps a stray
    bracketed citation like ``[1950] 1 S.C.R. 15`` out of the bench zone.
    """
    coram = _CORAM_RE.search(header_text)
    if coram:
        body = coram.group("body").strip()
        if _HAS_NAME.search(body):
            return body, "coram"

    opens = sorted(i for i, ch in enumerate(header_text) if ch in _BENCH_OPENERS)
    for start in opens:
        window = header_text[start + 1: start + 1 + 400]
        for cut in _bench_end_candidates(window):
            body = " ".join(window[:cut].split())
            if not body or not _TITLE_TAIL_RE.search(body):
                continue
            # A judge line is short and contains none of the other header
            # fields.  Without this the citation's own "[" wins: its "]" is
            # rejected, the scan falls through to the next "JJ." and swallows
            # citation + title + case number into the bench zone.
            if len(body) > MAX_BENCH_CHARS:
                continue
            # The judge list starts at the innermost opener: OCR sometimes
            # mangles the case number so badly ("Civi!Appea!No.16850of2017)")
            # that it is not recognised and ends up glued to the front.
            inner = max(body.rfind("["), body.rfind("("))
            if inner > 0:
                body = body[inner + 1:].strip()
                if not body or not _TITLE_TAIL_RE.search(body):
                    continue
            if (SCR_RE.search(body) or NEUTRAL_RE.search(body)
                    or _BENCH_REJECT_PARTY.search(body) or CASE_NO_RE.search(body)):
                continue
            names = parse_judges(body)
            # Two guards against a false positive, now that a bare "(" can open
            # a candidate: either the line names two judges, or it is long
            # enough to be a real single-judge bench line.  Without this, the
            # running footer "(Sen, J.)" of the 1982 volumes wins.
            if names and (len(names) >= 2 or len(body.split()) >= 4):
                return body, "bracket"
    return "", "none"


def normalise_party(name: str) -> str:
    """Light canonicalisation of a party name.

    Only two rules, because both are high-frequency and both matter for the
    citation matcher: "Union of India" and "State of <X>" appear in thousands of
    titles with varying case, punctuation and trailing noise.
    """
    s = " ".join(name.split())
    s = s.strip(" .,:;-_'\"")
    s = re.sub(r"\s*(?:etc\.?|&\s*(?:anr|ors)\.?|and\s+(?:another|others|anr\.?|ors\.?))\s*$",
               "", s, flags=re.I)
    s = _NORM_UNION.sub("Union of India", s)

    def _state(m: re.Match) -> str:
        tail = " ".join(w.capitalize() for w in m.group(1).split())
        return f"State of {tail}"

    s = _NORM_STATE.sub(_state, s)
    return s.strip(" .,:;-_")


def split_parties(title: str) -> Tuple[str, str]:
    parts = PARTY_SPLIT_RE.split(title, maxsplit=1)
    if len(parts) == 2:
        return normalise_party(parts[0]), normalise_party(parts[1])
    return normalise_party(title), ""


def _iso_date(m: re.Match) -> str:
    if m.group("m1"):
        mon, day, year = m.group("m1"), m.group("d1"), m.group("y1")
    else:
        mon, day, year = m.group("m2"), m.group("d2"), m.group("y2")
    key = mon.strip(".").lower()
    num = _MONTH_ID.get(key)
    if num is None:
        for name, i in _MONTH_ID.items():
            if name.startswith(key[:3]) and len(key) >= 3:
                num = i
                break
    if num is None:
        return ""
    return f"{int(year):04d}-{num:02d}-{int(day):02d}"


def parse_header(text: str) -> Header:
    """Parse the preamble of a judgment into a :class:`Header`."""
    lines = text.splitlines()
    head_lines = lines[:HEADER_LINES]
    header_text = "\n".join(head_lines)
    bench_region = "\n".join(lines[:BENCH_SEARCH_LINES])
    h = Header()

    m = SCR_RE.search(header_text)
    if m:
        vol = (m.group("vol") or "").upper().replace("I", "1").replace("O", "0") \
                                          .replace("L", "1").replace("S", "5").replace("B", "8")
        h.scr_cite = (f"[{m.group('year')}] {vol} S.C.R. {m.group('page')}"
                      if vol else f"[{m.group('year')}] S.C.R. {m.group('page')}")
    m = NEUTRAL_RE.search(header_text)
    if m:
        h.neutral_cite = f"{m.group('year')} INSC {int(m.group('num'))}"

    m = CASE_NO_RE.search(header_text)
    if m:
        h.case_no = " ".join(m.group("body").split())
    # Prefer the case number: the headnote names several proceedings ("...in a
    # writ petition under art. 32...") and matching those mislabels appeals.
    for name, pat in CASE_TYPE_PATTERNS:
        if h.case_no and pat.search(h.case_no):
            h.case_type = name
            break
    else:
        for name, pat in CASE_TYPE_PATTERNS:
            if pat.search(header_text):
                h.case_type = name
                break

    h.bench_text, h.bench_source = find_bench(bench_region)
    h.judges = parse_judges(h.bench_text) if h.bench_text else []

    # The date line sits between the case number and the bench; search the
    # preamble but skip anything that is part of the citation we already took.
    for ln in head_lines:
        if SCR_RE.search(ln) or NEUTRAL_RE.search(ln):
            continue
        dm = DATE_RE.search(ln)
        if dm:
            iso = _iso_date(dm)
            if iso:
                h.date = iso
                break

    # --- parties, from the stand-alone "v." line -------------------------
    sep_idx = next((i for i, ln in enumerate(head_lines) if PARTY_SEP_RE.match(ln)), None)
    if sep_idx is not None:
        above, below = [], []
        for ln in head_lines[:sep_idx]:
            s = ln.strip()
            if not s or SCR_RE.search(s) or NEUTRAL_RE.search(s):
                continue
            above.append(s)
        for ln in head_lines[sep_idx + 1:]:
            s = ln.strip()
            if not s:
                continue
            if (s.startswith("[") or CASE_NO_RE.search(s) or DATE_RE.search(s)
                    or s.startswith("(")):
                break
            below.append(s)
        h.petitioner = normalise_party(" ".join(above[-4:]))
        h.respondent = normalise_party(" ".join(below[:4]))
        h.title_lines = above[-4:] + ["v."] + below[:4]

    # --- where the body starts ------------------------------------------
    # After the bench line if we found one, else after the date / case number,
    # else right away.  Index is a *line* index into the full document.
    end = 0
    if h.bench_text:
        key = h.bench_text.split()[0] if h.bench_text.split() else ""
        for i, ln in enumerate(lines[:BENCH_SEARCH_LINES]):
            if key and key in ln:
                end = i + 1
                # the judge line can wrap; absorb continuation lines that still
                # have no closing bracket
                if not any(c in ln for c in _BENCH_CLOSERS) and end < len(lines):
                    end += 1
                break
    if end == 0 and sep_idx is not None:
        end = sep_idx + 1 + len([ln for ln in head_lines[sep_idx + 1:sep_idx + 5]
                                 if ln.strip()])
    h.end_line = min(end, len(lines))
    return h


# ==========================================================================
# 3. Paragraph numbering
# ==========================================================================

@dataclass
class Paragraphs:
    paras: List[Tuple[str, str]]      # (label, text); label "" when invented
    strategy: str                     # 'printed' | 'sentence-window'
    n_printed: int = 0


def _split_printed(lines: Sequence[str]) -> Optional[List[Tuple[str, str]]]:
    """Split on the judgment's own paragraph numbers, if it has any.

    The acceptance rule is what makes this safe: a candidate ``n.`` only starts
    a new paragraph when ``n`` is the number we are expecting (previous + 1) or
    is ``1``.  Without that, every "(2) of section 14" and every "34." inside a
    quoted passage would open a spurious paragraph; with it, the sequence has to
    actually be a sequence.  Returns ``None`` when fewer than 3 numbers line up,
    which is the signal to fall back.
    """
    starts: List[Tuple[int, str]] = []
    expected = 1
    for i, ln in enumerate(lines):
        m = PARA_NUM_RE.match(ln)
        if not m:
            continue
        n = int(m.group("n"))
        if n == expected or (n == 1 and not starts):
            starts.append((i, str(n)))
            expected = n + 1
    if len(starts) < 3:
        return None
    out: List[Tuple[str, str]] = []
    # anything before the first number is the unnumbered opening
    if starts[0][0] > 0:
        lead = "\n".join(lines[: starts[0][0]]).strip()
        if lead:
            out.append(("", lead))
    for k, (idx, label) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        chunk = "\n".join(lines[idx:end]).strip()
        m = PARA_NUM_RE.match(lines[idx])
        if m:
            rest = "\n".join([m.group("rest")] + list(lines[idx + 1:end])).strip()
            chunk = rest or chunk
        if chunk:
            out.append((label, chunk))
    return out


def _split_sentence_window(text: str,
                           target_words: int = FALLBACK_PARA_WORDS
                           ) -> List[Tuple[str, str]]:
    """Fallback for judgments with no printed numbering (common pre-2000).

    Blank-line segmentation is unavailable - PyMuPDF returns a single run of
    lines for these PDFs - so we accumulate whole sentences until the chunk
    reaches ``target_words``.  Deterministic, never splits mid-sentence, and
    keeps the paragraph a usable unit of evidence for snippets and ``/p``.
    """
    flat = " ".join(ln.strip() for ln in text.splitlines() if ln.strip())
    sents = [s for s in _SENT_SPLIT_RE.split(flat) if s.strip()]
    out: List[Tuple[str, str]] = []
    buf: List[str] = []
    count = 0
    for s in sents:
        buf.append(s.strip())
        count += len(_WORD_RE.findall(s))
        if count >= target_words:
            out.append(("", " ".join(buf)))
            buf, count = [], 0
    if buf:
        out.append(("", " ".join(buf)))
    return out


def number_paragraphs(body_text: str) -> Paragraphs:
    lines = body_text.splitlines()
    printed = _split_printed(lines)
    if printed:
        return Paragraphs(printed, "printed",
                          n_printed=sum(1 for lbl, _ in printed if lbl))
    return Paragraphs(_split_sentence_window(body_text), "sentence-window")


# ==========================================================================
# 4. Order zone
# ==========================================================================

def find_order_span(paras: Sequence[Tuple[str, str]]) -> Tuple[int, str]:
    """Index of the first order paragraph, and how it was found.

    Only the final ``ORDER_TAIL_SHARE`` of the judgment is eligible, because
    every headnote repeats the disposal ("Dismissing the appeals, the Court
    HELD: ...") and an unanchored cue search would therefore classify most of
    the document as the order.
    """
    n = len(paras)
    if n == 0:
        return -1, "none"
    first_eligible = max(0, n - max(ORDER_MAX_PARAS, int(round(n * ORDER_TAIL_SHARE))))
    for i in range(first_eligible, n):
        low = paras[i][1].lower()
        if any(cue in low for cue in ORDER_CUES):
            return max(i, n - ORDER_MAX_PARAS), "cue"
    return max(0, n - min(2, n)), "tail"


def cue_sentence_index(text: str) -> int:
    """Index of the first sentence of ``text`` that carries a disposal cue.

    Why this exists: the fallback paragraphs are ~110-word windows, so the
    window that contains "In the result, the appeal is dismissed" usually
    *starts* two or three sentences earlier, in the middle of the reasoning.
    Cutting the zone at the cue sentence (plus ORDER_CUE_LEAD sentences of
    lead-in) instead of at the window boundary raised hand-checked
    order-boundary precision from 15/33 to 25/33 on the audit sample - the
    single biggest accuracy win in this file.
    """
    sents = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    for i, s in enumerate(sents):
        low = s.lower()
        if any(cue in low for cue in ORDER_CUES):
            return i
    return 0


#: Sentences of lead-in kept in front of the cue sentence.  One, not zero:
#: several of the cues are the *reporter's* tailpiece ("Appeal allowed."), and
#: the court's own operative sentence ("I would therefore allow the appeal and
#: dismiss the counter-claim.") is the sentence immediately before it.  Cutting
#: at the cue exactly loses that sentence on 6 of the 33 audited documents.
ORDER_CUE_LEAD = 1


def split_at_cue(text: str, lead: int = ORDER_CUE_LEAD) -> Tuple[str, str]:
    """``text`` -> ``(reasoning_before_cue, operative_from_cue)``."""
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    k = max(0, cue_sentence_index(text) - lead)
    if k <= 0:
        return "", text
    return " ".join(sents[:k]), " ".join(sents[k:])


# ==========================================================================
# 5. Assembly
# ==========================================================================

@dataclass
class Segmentation:
    """Everything the segmenter learned about one judgment."""

    zones: Dict[str, str]
    paras: List[Para]
    header: Header
    para_strategy: str
    n_printed_labels: int
    order_strategy: str
    n_order_paras: int

    def info(self) -> Dict[str, object]:
        return {
            "para_strategy": self.para_strategy,
            "n_printed_labels": self.n_printed_labels,
            "order_strategy": self.order_strategy,
            "n_order_paras": self.n_order_paras,
            "bench_source": self.header.bench_source,
            "bench_size": len(self.header.judges),
        }


def segment(text: str, meta_title: str = "") -> Segmentation:
    """Split one cleaned judgment into the six zones plus numbered paragraphs."""
    header = parse_header(text)
    lines = text.splitlines()
    body_text = "\n".join(lines[header.end_line:]).strip()
    if not body_text:                      # pathological header parse; use it all
        body_text = text
        header.end_line = 0

    pg = number_paragraphs(body_text)
    order_from, order_how = find_order_span(pg.paras)

    # Trim the order zone back to the cue *sentence*.  Only for the fallback
    # strategy: where the judgment printed its own paragraph numbers, the
    # paragraph is the court's own unit of thought and we do not cut it up.
    chunks = list(pg.paras)
    if (order_how == "cue" and pg.strategy == "sentence-window"
            and 0 <= order_from < len(chunks)):
        head, tail = split_at_cue(chunks[order_from][1])
        if head and tail:
            label = chunks[order_from][0]
            chunks[order_from:order_from + 1] = [(label, head), ("", tail)]
            order_from += 1
    pg = Paragraphs(chunks, pg.strategy, pg.n_printed)

    paras: List[Para] = []
    for i, (label, ptext) in enumerate(pg.paras):
        zone = "order" if order_from >= 0 and i >= order_from else "body"
        paras.append(Para(id=i + 1, zone=zone, text=ptext, label=label))

    title = " ".join(meta_title.split()) if meta_title.strip() else ""
    if not title:
        if header.title_lines:
            title = " ".join(header.title_lines)
        else:
            title = " ".join(lines[:3]).strip()
    # The metadata title is the clean source for the parties: the header lines
    # of a scanned volume pick up page numbers and marginal letters ("A B .
    # JUSTICE K S PUTTASWAMY"), and the title already contains " versus ".
    if meta_title.strip():
        petitioner, respondent = split_parties(title)
        if not respondent:
            petitioner = petitioner or header.petitioner
            respondent = header.respondent
    else:
        petitioner, respondent = header.petitioner, header.respondent
        if not petitioner:
            petitioner, respondent = split_parties(title)

    zones = {
        "title": title,
        "parties": " ".join(x for x in (petitioner, respondent) if x),
        "bench": header.bench_text,
        "statutes": "",                     # Member 3 fills this in wave 2
        "order": "\n\n".join(p.text for p in paras if p.zone == "order"),
        "body": "\n\n".join(p.text for p in paras if p.zone == "body"),
    }
    assert set(zones) == set(config.ZONES), "zone set drifted from config.ZONES"
    header.petitioner, header.respondent = petitioner, respondent
    return Segmentation(
        zones=zones, paras=paras, header=header,
        para_strategy=pg.strategy, n_printed_labels=pg.n_printed,
        order_strategy=order_how,
        n_order_paras=sum(1 for p in paras if p.zone == "order"),
    )


def attach_statute_zone(record: JudgmentRecord, tokens: Iterable[str]) -> JudgmentRecord:
    """Hook for wave 2: fill the synthetic ``statutes`` zone.

    Member 3's ``nyaya/analyze.py`` canonicalises statute references into tokens
    like ``ipc§302``; those are not contiguous text anywhere in the judgment, so
    the zone is synthesised from them here instead of being cut out of the body.
    Deliberately not importing her module - this stays a pure function of the
    tokens she passes in.
    """
    toks = [t for t in (tokens or []) if t]
    record.statute_tokens = list(dict.fromkeys(toks))
    record.zones["statutes"] = " ".join(record.statute_tokens)
    record.acts = list(dict.fromkeys(t.split("§")[0].upper() for t in record.statute_tokens))
    return record


def build_record(doc_id: int, source_path: str, year: int, text: str,
                 meta_title: str = "", n_pages: int = 0,
                 meta: Optional[Dict[str, object]] = None
                 ) -> Tuple[JudgmentRecord, Dict[str, object]]:
    """Cleaned text + manifest row -> a :class:`JudgmentRecord` and an info dict.

    Merge policy, agreed with Member 2: his metadata-derived value wins whenever
    it is non-empty, because it comes from the court's own structured record;
    ours is the fallback and fills the (many) rows where his is blank.  The one
    exception is ``bench_size``/``judges``, where *ours* wins when we actually
    found a judge line, because ``bench_size_meta`` is single-judge on 37k of
    42k rows.
    """
    meta = meta or {}
    seg = segment(text, meta_title=meta_title)
    h = seg.header

    def pick(meta_key: str, mine: str) -> str:
        val = meta.get(meta_key)
        if val is None:
            return mine
        sval = str(val).strip()
        return sval if sval and sval.lower() != "nan" else mine

    judges_meta = list(meta.get("judges") or [])
    if h.judges:
        judges, bench_size, bench_src = h.judges, len(h.judges), h.bench_source
    else:
        judges = [str(j) for j in judges_meta]
        bench_size = int(meta.get("bench_size_meta") or len(judges) or 0)
        bench_src = "metadata"

    rec = JudgmentRecord(
        doc_id=int(doc_id),
        source_path=source_path,
        title=seg.zones["title"],
        petitioner=h.petitioner,
        respondent=h.respondent,
        date=pick("decision_date", h.date),
        year=int(year),
        judges=judges,
        bench_size=bench_size,
        case_no=pick("case_no", h.case_no),
        neutral_cite=pick("neutral_cite", h.neutral_cite),
        scr_cite=pick("scr_cite", h.scr_cite),
        disposal=str(meta.get("disposal") or ""),
        case_type=pick("case_type", h.case_type),
        zones=seg.zones,
        paras=seg.paras,
        n_tokens=len(_WORD_RE.findall(text)),
        n_pages=int(n_pages),
    )
    info = seg.info()
    info["bench_source"] = bench_src
    return rec, info
