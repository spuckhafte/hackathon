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
#: *** Measured on the sample described in DF_SOURCE below. ***
DOMAIN_DF: Dict[str, Tuple[int, int]] = {}

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
DF_SOURCE = "(not yet measured)"


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


def refresh() -> None:
    """Rebuild the derived sets after DOMAIN_DF is edited (used by tests)."""
    global DOMAIN_STOPWORDS, SCORING_STOPWORDS
    DOMAIN_STOPWORDS = _domain_terms()
    SCORING_STOPWORDS = _build_scoring_stopwords()


# ==========================================================================
# 4. The derivation script:  python -m nyaya.stoplist --derive
# ==========================================================================


def _iter_corpus_texts(limit: int = 0) -> Iterable[Tuple[str, str]]:
    """Yield (doc_key, text) from the best available source.

    Preference order, so the list can be seeded before Rakshit's corpus lands
    and refreshed afterwards:
      1. data/processed/judgments.jsonl  — the real zoned corpus
      2. data/raw/pdf/**/*.pdf           — PDFs already downloaded
      3. data/raw/metadata/*.parquet     — the SCR headnote `description` field
    """
    from nyaya import config

    if config.JUDGMENTS_JSONL.exists():
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

    pdfs = sorted(config.RAW_PDF.rglob("*.pdf"))
    if pdfs:
        import fitz  # pymupdf

        for i, p in enumerate(pdfs if not limit else pdfs[:limit]):
            try:
                with fitz.open(p) as doc:
                    yield p.stem, "\n\n".join(pg.get_text() for pg in doc)
            except Exception as exc:                    # pragma: no cover
                print(f"  skip {p.name}: {exc}", file=sys.stderr)
        return

    import pandas as pd

    n = 0
    for f in sorted(config.RAW_METADATA.glob("*.parquet")):
        d = pd.read_parquet(f, columns=["description", "title", "path"])
        for _, row in d.iterrows():
            txt = f"{row['title']}\n\n{row['description']}"
            if len(str(row["description"])) < 200:
                continue
            yield str(row["path"]), txt
            n += 1
            if limit and n >= limit:
                return


def derive(top: int = 50, limit: int = 0) -> List[Tuple[str, int, int]]:
    """Compute df over the available corpus and return the top terms."""
    from nyaya.analyze import analyze_terms

    df: Counter = Counter()
    n_docs = 0
    for _, text in _iter_corpus_texts(limit=limit):
        terms = set(analyze_terms(text, stem=True))
        if not terms:
            continue
        n_docs += 1
        df.update(terms)
    rows = [(t, c, n_docs) for t, c in df.most_common(top * 4)]
    return rows[: top * 4] if n_docs else []


def _main(argv: List[str] | None = None) -> int:
    from nyaya.utils import ensure_utf8_stdout

    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="derive the domain stop list by df")
    ap.add_argument("--derive", action="store_true")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="max documents (0 = all)")
    args = ap.parse_args(argv)

    if not args.derive:
        print(f"standard stop words : {len(STANDARD_STOPWORDS)}")
        print(f"domain stop words   : {len(DOMAIN_DF)}  (source: {DF_SOURCE})")
        print(f"scoring stop set    : {len(SCORING_STOPWORDS)} (incl. stemmed forms)")
        return 0

    rows = derive(top=args.top, limit=args.limit)
    if not rows:
        print("no corpus text found", file=sys.stderr)
        return 1
    n = rows[0][2]
    print(f"# df measured over N = {n} documents")
    print(f"{'term':<22}{'df':>8}{'df/N':>9}  candidate?")
    shown = 0
    for term, c, _ in rows:
        if term in STANDARD_STOPWORDS:
            continue
        shown += 1
        if shown > args.top:
            break
        flag = "STOP" if c / n >= DOMAIN_DF_THRESHOLD else ""
        if term in DOMAIN_KEEP_REASONS:
            flag = "keep"
        print(f"{term:<22}{c:>8}{c / n:>9.3f}  {flag}")
    print()
    print("# paste-ready DOMAIN_DF block (threshold "
          f"df/N >= {DOMAIN_DF_THRESHOLD}):")
    for term, c, _ in rows:
        if term in STANDARD_STOPWORDS or term in DOMAIN_KEEP_REASONS:
            continue
        if c / n >= DOMAIN_DF_THRESHOLD:
            print(f'    "{term}": ({c}, {n}),'.ljust(44) + f"# df/N = {c / n:.3f}")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(_main())
