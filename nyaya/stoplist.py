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
    # --- the words every judgment contains: idf <= 0.01, pure noise ---------
    "court": (760, 760),                # df/N = 1.000
    "judgment": (755, 760),             # df/N = 0.993
    "supreme": (751, 760),              # df/N = 0.988
    "case": (742, 760),                 # df/N = 0.976
    "respond": (741, 760),              # df/N = 0.975   <- "respondent"
    "date": (738, 760),                 # df/N = 0.971
    "deliv": (724, 760),                # df/N = 0.953   <- "delivered"
    "made": (715, 760),                 # df/N = 0.941
    "appeal": (710, 760),               # df/N = 0.934
    "appel": (709, 760),                # df/N = 0.933   <- "appellant"
    "jurisdict": (707, 760),            # df/N = 0.930   <- "jurisdiction"
    "refer": (692, 760),                # df/N = 0.911
    "view": (687, 760),                 # df/N = 0.904
    "fact": (678, 760),                 # df/N = 0.892
    "one": (677, 760),                  # df/N = 0.891
    "follow": (670, 760),               # df/N = 0.882
    "question": (665, 760),             # df/N = 0.875
    "said": (657, 760),                 # df/N = 0.864
    "two": (642, 760),                  # df/N = 0.845
    "file": (641, 760),                 # df/N = 0.843
    "dismiss": (633, 760),              # df/N = 0.833
    "civil": (629, 760),                # df/N = 0.828
    "learn": (626, 760),                # df/N = 0.824   <- "learned [counsel]"
    "allow": (626, 760),                # df/N = 0.824
    "will": (622, 760),                 # df/N = 0.818
    "reason": (621, 760),               # df/N = 0.817
    "matter": (618, 760),               # df/N = 0.813
    "time": (618, 760),                 # df/N = 0.813
    "person": (616, 760),               # df/N = 0.811
    "set": (612, 760),                  # df/N = 0.805
    "make": (611, 760),                 # df/N = 0.804
    # --- page furniture of the S.C.R. reporter, not language at all --------
    # Every SCR page prints "SUPREME COURT REPORTS" in the header and the
    # paragraph-locator letters A-H down the margin, so PDF extraction emits
    # them in essentially every document.  M1's ingest could strip them; until
    # it does, stopping them is what keeps them out of every tf-idf vector.
    "scr": (750, 760),                  # df/N = 0.987
    "reports": (749, 760),              # df/N = 0.986
    "jj": (734, 760),                   # df/N = 0.966   (", JJ." in the bench line)
    "v": (759, 760),                    # df/N = 0.999   (the "v." of every title)
    "j": (726, 760),                    # df/N = 0.955
    "c": (748, 760),                    # df/N = 0.984
    "b": (744, 760),                    # df/N = 0.979
    "d": (736, 760),                    # df/N = 0.968
    "e": (727, 760),                    # df/N = 0.957
    "f": (715, 760),                    # df/N = 0.941
    "h": (710, 760),                    # df/N = 0.934
    "s": (697, 760),                    # df/N = 0.917
    "g": (691, 760),                    # df/N = 0.909
    "r": (619, 760),                    # df/N = 0.814
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
    "760 judgment PDFs, 10 per year across 1950-2025, drawn from "
    "data/raw/pdf with seed config.SAMPLE_SEED (358); text via PyMuPDF; "
    "command: python -m nyaya.stoplist --derive --source pdf --limit 760. "
    "Refresh against data/processed/judgments.jsonl once the zoned corpus is "
    "complete (--source jsonl); the ranking of these terms is not expected to "
    "move, because they are the reporter's own vocabulary."
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


def _party_tokens() -> set:
    """Every token appearing in a petitioner/respondent metadata field."""
    import pandas as pd

    from nyaya import config

    out: set = set()
    for f in sorted(config.RAW_METADATA.glob("*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["petitioner", "respondent"])
        except Exception:                                   # pragma: no cover
            continue
        for col in ("petitioner", "respondent"):
            for v in d[col].dropna().astype(str):
                for w in _PARTY_WORD_RE.findall(v):
                    out.add(w.lower().strip("'"))
    return out


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
    party = _party_tokens()
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
        f"#: {len(names)} party-field tokens with body df/N < {PARTY_NAME_MAX_DF}.",
        "PARTY_NAME_WORDS: FrozenSet[str] = frozenset((",
        _wrap(names),
        "))",
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")
    return f"{len(common)} common words, {len(names)} party-name tokens"


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
