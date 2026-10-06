"""Stop lists — and the two different jobs a stop list has to do.

IR framing (guide Section 4.3)
------------------------------
A stop word is a term whose idf is so low that it adds noise to a vector-space
score, *but* removing it from the index destroys phrase and proximity search:
``"right to privacy"`` and ``"murder /3 intention"`` need "to" to still occupy
position 2.  Dropping stop words at index time is the classic shortcut that
makes a system unable to answer its most interesting queries.

So NyayaTrace keeps them in **two** places with **two** policies:

1. **Positional index — kept.**  `analyze()` emits every token, stop words
   included, and the positional index stores them.  Phrases and the ``/k``,
   ``/s``, ``/p`` operators are therefore exact.
2. **tf-idf / BM25F vectors — excluded.**  The scorers call
   `is_scoring_stopword(term)` and skip the term.  Nothing is lost: a term with
   df ≈ N has idf ≈ 0 anyway, so skipping it is a speed win, not a ranking
   change — except that it stops a long free-text query from being dominated by
   the twenty copies of "the" in the document.

The domain list
---------------
The standard English list is not enough for a judgment corpus.  Words like
``appellant``, ``learned``, ``counsel`` and ``hon'ble`` occur in essentially
every Supreme Court judgment, so their idf is ~0 and they behave exactly like
"the" — but no off-the-shelf list contains them.  We therefore *measured* them:
`python -m nyaya.stoplist --derive` computes document frequency over the built
corpus and prints the top terms; the ones with df/N above
DOMAIN_DF_THRESHOLD and no legal discriminating power were copied into
DOMAIN_DF below **with their measured df**, so the report can show the table
rather than assert it.

Author: Sumedha (M3)
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Tuple

# ==========================================================================
# 1. Standard English stop list
# ==========================================================================
# Written out explicitly rather than loaded from nltk.corpus: the NLTK stopwords
# corpus is a download, and every build and test here must run network-free.
# This is the usual ~180-term list (van Rijsbergen / NLTK English) plus the
# auxiliary forms that show up constantly in judicial prose ("shall", "would").
STANDARD_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at
be because been before being below between both but by
can cannot could couldn't
did didn't do does doesn't doing don't down during
each few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too
under until up upon very
was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves
shall may must might also thus hence therefore however whether being
""".split())

# ==========================================================================
# 2. Measured domain stop list
# ==========================================================================
#: A term is a *candidate* domain stop word once it appears in this share of
#: documents.  0.80 is deliberately high: we only want terms whose idf is
#: genuinely ~0, not merely common legal vocabulary like "evidence" (which is
#: common but discriminating).
DOMAIN_DF_THRESHOLD = 0.80

#: term -> (document frequency, collection size N) as measured by --derive.
#: *** MEASURED, not guessed — see DF_SOURCE below. ***  Terms are the *stemmed*
#: index form, which is why "respondent" appears as "respond" and "appellant" as
#: "appel".  Terms already covered by stemming the standard English list
#: ("was"->"wa", "this"->"thi", "has"->"ha", "any"->"ani", "before"->"befor",
#: "therefore"->"therefor", "only"->"onli", "his"->"hi") are NOT repeated here.
DOMAIN_DF: Dict[str, Tuple[int, int]] = {
    # --- the words every judgment contains: idf <= 0.10, pure noise ---------
    "versu": (13878, 13878),        # df/N = 1.000
    "court": (13877, 13878),        # df/N = 1.000
    "judgment": (13741, 13878),     # df/N = 0.990
    "date": (13643, 13878),         # df/N = 0.983
    "case": (13608, 13878),         # df/N = 0.981
    "respond": (13385, 13878),      # df/N = 0.964
    "deliv": (13262, 13878),        # df/N = 0.956
    "appeal": (13096, 13878),       # df/N = 0.944
    "appel": (12992, 13878),        # df/N = 0.936
    "jurisdict": (12989, 13878),    # df/N = 0.936
    "made": (12935, 13878),         # df/N = 0.932
    "one": (12437, 13878),          # df/N = 0.896
    "view": (12435, 13878),         # df/N = 0.896
    "fact": (12425, 13878),         # df/N = 0.895
    "said": (12301, 13878),         # df/N = 0.886
    "refer": (12295, 13878),        # df/N = 0.886
    "question": (12264, 13878),     # df/N = 0.884
    "follow": (11969, 13878),       # df/N = 0.862
    "file": (11697, 13878),         # df/N = 0.843
    "time": (11606, 13878),         # df/N = 0.836
    "reason": (11496, 13878),       # df/N = 0.828
    "two": (11318, 13878),          # df/N = 0.816
    "allow": (11254, 13878),        # df/N = 0.811
    "civil": (11223, 13878),        # df/N = 0.809
    "present": (11214, 13878),      # df/N = 0.808
    "learn": (11179, 13878),        # df/N = 0.806
    "matter": (11156, 13878),       # df/N = 0.804
    "will": (11142, 13878),         # df/N = 0.803
    "taken": (11137, 13878),        # df/N = 0.802
    # --- page furniture of the S.C.R. reporter, not language at all --------
    # Every S.C.R. page prints the paragraph-locator letters A-H down the
    # margin and the bench line ends ", JJ.", so PDF extraction emits them in
    # nearly every document.  (The "SUPREME COURT REPORTS" running header that
    # topped the wave-1 list is gone: df("supreme") and df("reports") are now 0,
    # because M1's zoner strips it.  These single letters are what is left.)
    "v": (13368, 13878),            # df/N = 0.963
    "c": (13331, 13878),            # df/N = 0.961
    "j": (13261, 13878),            # df/N = 0.956
    "b": (12891, 13878),            # df/N = 0.929
    "jj": (12856, 13878),           # df/N = 0.926
    "d": (12836, 13878),            # df/N = 0.925
    "s": (12626, 13878),            # df/N = 0.910
    "e": (12574, 13878),            # df/N = 0.906
    "scr": (12533, 13878),          # df/N = 0.903
    "h": (12107, 13878),            # df/N = 0.872
    "f": (12018, 13878),            # df/N = 0.866
    "g": (11724, 13878),            # df/N = 0.845
    "r": (11257, 13878),            # df/N = 0.811
    # --- small integers: list and paragraph numbering ----------------------
    # Not language either, and their idf is ~0.02-0.10.  Larger numbers are
    # unaffected - a statute section number is far rarer than this.
    "1": (13649, 13878),            # df/N = 0.983
    "2": (13535, 13878),            # df/N = 0.975
    "3": (13090, 13878),            # df/N = 0.943
    "4": (12154, 13878),            # df/N = 0.876
    "5": (11512, 13878),            # df/N = 0.830
}

#: Terms that the measurement flagged but that we deliberately KEEP, with the
#: reason — this is the judgement call the report should show.  "evidence",
#: "held", "section", "act" are high-df but a query for them means something.
DOMAIN_KEEP_REASONS: Dict[str, str] = {
    "evidence": "high df but discriminating: 'evidence act', 'circumstantial evidence'",
    "section": "needed as a literal term when canonicalisation failed",
    "act": "part of act names the canonicaliser may not have caught",
    "held": "the holding verb — used by treatment classification",
    "order": "names a zone and an Order-VII-Rule-11 style citation",
    "right": "'right to privacy', 'fundamental right'",
    "law": "'law laid down', 'question of law'",
    "high": "'High Court' is a jurisdiction filter users type",
    "state": "half the respondents are a State",
}

#: The human-readable description of what --derive actually ran on, so the
#: numbers in DOMAIN_DF are reproducible and honestly labelled.
DF_SOURCE = (
    "df read from the built positional index over all N = 13,878 indexed "
    "judgments (PositionalIndex.df), which is the same df the scorers use. "
    "The candidate list came from a full streaming pass: "
    "python -m nyaya.stoplist --derive --source jsonl --write-lexicon. "
    "Wave-1 figures were measured on a 760-PDF sample and have been replaced; "
    "four terms that cleared 0.80 on the sample (dismiss, make, set, person) "
    "fall below it on the whole corpus and were dropped, and eight that the "
    "sample missed were added - 'versu' most notably, the stem of the "
    "'versus' in every S.C.R. case title, at df/N = 1.000."
)


def _domain_terms() -> frozenset:
    return frozenset(DOMAIN_DF)


DOMAIN_STOPWORDS = _domain_terms()

# ==========================================================================
# 3. The frozenset the scorers consult
# ==========================================================================
# Built from the *stemmed* forms too, because the index stores stemmed terms:
# "respondents" -> "respond", and a scorer looking up "respond" must still find
# it in the stop set.  Statute tokens can never be stop words.


def _stems(terms: Iterable[str]) -> set:
    try:
        from nltk.stem import PorterStemmer
    except Exception:                                   # pragma: no cover
        return set()
    ps = PorterStemmer()
    out = set()
    for t in terms:
        if t and "§" not in t and t.isalpha():
            out.add(ps.stem(t))
    return out


def _build_scoring_stopwords() -> frozenset:
    base = set(STANDARD_STOPWORDS) | set(DOMAIN_DF)
    return frozenset(base | _stems(base))


SCORING_STOPWORDS = _build_scoring_stopwords()


def is_scoring_stopword(term: str) -> bool:
    """True if ``term`` should be skipped when building a tf-idf / BM25F vector.

    It is still present in the positional index — see the module docstring.
    """
    if not term or "§" in term:
        return False
    return term in SCORING_STOPWORDS


def filter_scoring_terms(terms: Iterable[str]) -> List[str]:
    """Drop the scoring stop words — but never return an empty bag.

    A query like "appellate jurisdiction of the Supreme Court" is *entirely*
    stop words once the measured domain list is applied, and an empty vector
    would score nothing at all.  In that case the stop list is ignored for this
    query: every term has idf ~0, so the ranking is weak either way, but the
    user gets results instead of silence.  Callers (the scorers) must use this
    rather than filtering by hand.
    """
    terms = list(terms)
    kept = [t for t in terms if not is_scoring_stopword(t)]
    return kept if kept else terms


def refresh() -> None:
    """Rebuild the derived sets after DOMAIN_DF is edited (used by tests)."""
    global DOMAIN_STOPWORDS, SCORING_STOPWORDS
    DOMAIN_STOPWORDS = _domain_terms()
    SCORING_STOPWORDS = _build_scoring_stopwords()


# ==========================================================================
# 4. The derivation script:  python -m nyaya.stoplist --derive
# ==========================================================================
# It measures three things in one pass over the corpus:
#   * df of each *stemmed* term     -> the domain stop list (DOMAIN_DF)
#   * df of each *un-stemmed* term  -> _lexicon.COMMON_WORDS
#   * party-name tokens that are rare in body text -> _lexicon.PARTY_NAME_WORDS
# The last two exist so that "do not stem a case name" can be a property of the
# *term* rather than of its surrounding context (see analyze._mark_no_stem).
# Measuring them is what keeps documents and queries analysing identically.

#: A term frequent enough to be an ordinary word rather than a proper noun.
COMMON_WORD_DF = 0.02
#: A party-name token this rare in body text is a surname or a place name.
PARTY_NAME_MAX_DF = 0.005


def _sample_pdfs(limit: int) -> List[object]:
    """Deterministic sample spread across all years, not the first N files."""
    import random

    from nyaya import config

    by_year: Dict[str, List] = {}
    for p in config.RAW_PDF.rglob("*.pdf"):
        by_year.setdefault(p.parent.name, []).append(p)
    rng = random.Random(config.SAMPLE_SEED)
    years = sorted(by_year)
    if not years:
        return []
    per_year = max(1, (limit // len(years)) if limit else 10 ** 9)
    out: List = []
    for y in years:
        files = sorted(by_year[y])
        rng.shuffle(files)
        out.extend(files[:per_year])
    return out


#: Below this many records, judgments.jsonl is still being written by the
#: ingest job and a df measured on it would not mean anything.
MIN_JSONL_DOCS = 500


def _iter_corpus_texts(limit: int = 0, source: str = "auto") -> Iterable[Tuple[str, str]]:
    """Yield (doc_key, text) from the best available source.

    Preference order, so the list can be seeded before the zoned corpus lands and
    refreshed afterwards:
      1. data/processed/judgments.jsonl  - the real zoned corpus
      2. data/raw/pdf/{year}/*.pdf       - PDFs already downloaded
      3. data/raw/metadata/*.parquet     - the SCR headnote `description` field

    `source` pins one of "jsonl" / "pdf" / "meta" so a measurement is
    reproducible even while the ingest job is mid-run.
    """
    from nyaya import config
    from nyaya.utils import count_lines

    use_jsonl = source == "jsonl" or (
        source == "auto"
        and config.JUDGMENTS_JSONL.exists()
        and count_lines(config.JUDGMENTS_JSONL) >= MIN_JSONL_DOCS
    )
    if use_jsonl:
        from nyaya.utils import read_jsonl

        n = 0
        for rec in read_jsonl(config.JUDGMENTS_JSONL):
            zones = rec.get("zones") or {}
            text = "\n\n".join(str(v) for v in zones.values()) or str(rec.get("text", ""))
            yield str(rec.get("source_path", rec.get("doc_id", n))), text
            n += 1
            if limit and n >= limit:
                return
        return

    pdfs = [] if source == "meta" else _sample_pdfs(limit)
    if pdfs:
        import fitz  # pymupdf

        for p in pdfs:
            try:
                with fitz.open(p) as doc:
                    yield p.stem, "\n\n".join(pg.get_text() for pg in doc)
            except Exception as exc:                        # pragma: no cover
                print(f"  skip {p.name}: {exc}", file=sys.stderr)
        return

    import pandas as pd

    n = 0
    for f in sorted(config.RAW_METADATA.glob("*.parquet")):
        d = pd.read_parquet(f, columns=["description", "title", "path"])
        for _, row in d.iterrows():
            if len(str(row["description"])) < 200:
                continue
            yield str(row["path"]), f"{row['title']}\n\n{row['description']}"
            n += 1
            if limit and n >= limit:
                return


_PARTY_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']{3,}")


def _party_tokens() -> Tuple[set, str]:
    """Every token appearing in a party name, with the source it came from.

    Prefers the built corpus's ``parties`` zone (the real thing M1's zoner
    produced, which is what the index actually holds) and falls back to the
    petitioner/respondent columns of the raw metadata when the corpus is not
    built yet.
    """
    from nyaya import config

    out: set = set()
    if config.JUDGMENTS_JSONL.exists():
        from nyaya.utils import read_jsonl

        n = 0
        for rec in read_jsonl(config.JUDGMENTS_JSONL):
            zone = (rec.get("zones") or {}).get("parties") or ""
            blob = " ".join(
                [str(zone), str(rec.get("petitioner") or ""),
                 str(rec.get("respondent") or "")]
            )
            for w in _PARTY_WORD_RE.findall(blob):
                out.add(w.lower().strip("'"))
            n += 1
        if out:
            return out, f"parties zone of {n} built judgments"

    import pandas as pd

    for f in sorted(config.RAW_METADATA.glob("*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["petitioner", "respondent"])
        except Exception:                                   # pragma: no cover
            continue
        for col in ("petitioner", "respondent"):
            for v in d[col].dropna().astype(str):
                for w in _PARTY_WORD_RE.findall(v):
                    out.add(w.lower().strip("'"))
    return out, "petitioner/respondent columns of the raw metadata"


def derive(limit: int = 0, source: str = "auto") -> Dict[str, object]:
    """One pass over the corpus; returns every table the module needs."""
    from nyaya.analyze import _never_stem, analyze, stem_term

    df_stem: Counter = Counter()
    df_raw: Counter = Counter()
    n_docs = 0
    for _, text in _iter_corpus_texts(limit=limit, source=source):
        toks = analyze(text, stem=False)
        if len(toks) < 200:
            continue
        n_docs += 1
        raw_terms = {t.term for t in toks}
        df_raw.update(raw_terms)
        df_stem.update({t if _never_stem(t) else stem_term(t) for t in raw_terms})
        if n_docs % 250 == 0:
            print(f"  ... {n_docs} documents", file=sys.stderr)
    return {"df_stem": df_stem, "df_raw": df_raw, "n_docs": n_docs,
            "source": source}


def _format_domain_block(df_stem: Counter, n: int, top: int) -> List[str]:
    rows = []
    for term, c in df_stem.most_common(top * 8):
        if is_scoring_stopword(term) or term in DOMAIN_KEEP_REASONS:
            continue                     # already covered, or deliberately kept
        if "§" in term:
            continue
        if c / n < DOMAIN_DF_THRESHOLD:
            continue
        rows.append(f'    "{term}": ({c}, {n}),'.ljust(40) + f"# df/N = {c / n:.3f}")
        if len(rows) >= top:
            break
    return rows


def _wrap(words: List[str], width: int = 76) -> str:
    lines, cur = [], "    "
    for w in words:
        piece = f'"{w}", '
        if len(cur) + len(piece) > width:
            lines.append(cur.rstrip())
            cur = "    "
        cur += piece
    if cur.strip():
        lines.append(cur.rstrip())
    return "\n".join(lines)


def _write_lexicon(df_raw: Counter, n: int, source: str) -> str:
    from pathlib import Path

    from nyaya.analyze import stem_term

    common = sorted(t for t, c in df_raw.items()
                    if t.isalpha() and c / n >= COMMON_WORD_DF)
    party, party_src = _party_tokens()
    names = sorted(
        t for t in party
        if t.isalpha() and len(t) >= 4
        and stem_term(t) != t
        and df_raw.get(t, 0) / n < PARTY_NAME_MAX_DF
        and t not in STANDARD_STOPWORDS
    )
    path = Path(__file__).with_name("_lexicon.py")
    head = path.read_text(encoding="utf-8").split("MEASURED = ")[0]
    body = [
        head.rstrip() + "\n",
        "MEASURED = True",
        f"N_DOCS = {n}",
        f"SOURCE = {source!r}",
        "",
        f"#: {len(common)} un-stemmed terms with df/N >= {COMMON_WORD_DF}.",
        "COMMON_WORDS: FrozenSet[str] = frozenset((",
        _wrap(common),
        "))",
        "",
        f"#: {len(names)} party-name tokens with body df/N < {PARTY_NAME_MAX_DF},",
        f"#: drawn from the {party_src}.",
        "PARTY_NAME_WORDS: FrozenSet[str] = frozenset((",
        _wrap(names),
        "))",
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")
    return (f"{len(common)} common words, {len(names)} party-name tokens "
            f"(from the {party_src})")


def _main(argv: List[str] | None = None) -> int:
    from nyaya.utils import ensure_utf8_stdout

    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="derive the domain stop list by df")
    ap.add_argument("--derive", action="store_true")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="max documents (0 = all)")
    ap.add_argument("--source", choices=("auto", "jsonl", "pdf", "meta"),
                    default="auto")
    ap.add_argument("--write-lexicon", action="store_true",
                    help="regenerate nyaya/_lexicon.py from the measurement")
    args = ap.parse_args(argv)

    if not args.derive:
        print(f"standard stop words : {len(STANDARD_STOPWORDS)}")
        print(f"domain stop words   : {len(DOMAIN_DF)}  (source: {DF_SOURCE})")
        print(f"scoring stop set    : {len(SCORING_STOPWORDS)} (incl. stemmed forms)")
        for term, (df, n) in sorted(DOMAIN_DF.items(), key=lambda kv: -kv[1][0]):
            print(f"  {term:<20}{df:>7}/{n:<7} df/N = {df / n:.3f}")
        return 0

    out = derive(limit=args.limit, source=args.source)
    n = int(out["n_docs"])
    if not n:
        print("no corpus text found", file=sys.stderr)
        return 1
    df_stem: Counter = out["df_stem"]            # type: ignore[assignment]
    df_raw: Counter = out["df_raw"]              # type: ignore[assignment]

    print(f"# df measured over N = {n} documents")
    print(f"{'term':<22}{'df':>8}{'df/N':>9}  verdict")
    shown = 0
    for term, c in df_stem.most_common(600):
        if "§" in term or term in DOMAIN_DF:
            continue
        shown += 1
        if shown > args.top:
            break
        verdict = "STOP" if c / n >= DOMAIN_DF_THRESHOLD else ""
        if term in DOMAIN_KEEP_REASONS:
            verdict = "keep"
        print(f"{term:<22}{c:>8}{c / n:>9.3f}  {verdict}")

    print(f"\n# paste-ready DOMAIN_DF block (df/N >= {DOMAIN_DF_THRESHOLD}):")
    for line in _format_domain_block(df_stem, n, args.top):
        print(line)

    if args.write_lexicon:
        src = (f"{n} judgments ({args.source} source) sampled across all years "
               f"with seed {__import__('nyaya.config', fromlist=['x']).SAMPLE_SEED}")
        print("\n# wrote nyaya/_lexicon.py:", _write_lexicon(df_raw, n, src))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(_main())
