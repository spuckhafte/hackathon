"""Tests for the positional operators  (Member 1, wave 3).

Two kinds of test here, and the difference matters.

**Brute force against a from-scratch token stream.**  For every operator and
every pair of terms in the fixture vocabulary, the expected answer is recomputed
by re-analysing the judgment text with ``analyze()`` and scanning the resulting
token list with a naive O(n*m) loop - no index, no postings, no skip pointers.
That is the correctness proof for the merge algorithms, which is the part I
wrote.

**Brute force against a regular expression over the raw text.**  The same
queries are answered by a regex on the *surface* words, which knows nothing
about stemming, statute canonicalisation or zone segmentation.  Where that
disagrees with the index, the disagreement is a tokenisation decision and the
test says which one - those are the cases worth explaining in the report rather
than hiding.
"""
from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

from nyaya import config
from nyaya.analyze import analyze
from nyaya.index import positional as P
from nyaya.query import proximity as prox

# --------------------------------------------------------------------------
# Fixture corpus
# --------------------------------------------------------------------------

DOCS = [
    {
        "title": "RAM SINGH versus STATE OF BIHAR",
        "bench": "A. K. Sikri and Ashok Bhushan, JJ.",
        "paras": [
            ("body", 1, "The principles of natural justice require that the "
                        "accused be heard before an adverse order is made."),
            ("body", 2, "Natural justice was denied because no notice of the "
                        "inquiry reached the appellant at all."),
            ("order", 3, "In the result the appeal is allowed and the order of "
                         "the High Court is set aside."),
        ],
    },
    {
        "title": "MOHAN LAL versus UNION OF INDIA",
        "bench": "R. F. Nariman and Sanjay Kishan Kaul, JJ.",
        "paras": [
            ("body", 1, "Anticipatory bail was granted by the High Court on the "
                        "ground that the investigation was complete."),
            ("body", 2, "The question of bail is distinct from the question of "
                        "guilt, and natural justice has no bearing on it."),
            ("order", 3, "The appeal is dismissed with no order as to costs."),
        ],
    },
    {
        "title": "KAMLA DEVI versus STATE OF KERALA",
        "bench": "Bela M. Trivedi, J.",
        "paras": [
            ("body", 1, "Bail was refused. Anticipatory relief was also refused "
                        "by the Sessions Judge in a separate order."),
            ("body", 2, "The right to privacy and the surveillance of citizens "
                        "were considered together in that judgment."),
            ("body", 3, "Privacy is a fundamental right. Surveillance without "
                        "authority of law cannot be justified."),
            ("order", 4, "Accordingly, the appeal is allowed and bail is granted."),
        ],
    },
    {
        "title": "STATE OF BIHAR versus UNION CARBIDE",
        "bench": "Abhay S. Oka and Augustine George Masih, JJ.",
        "paras": [
            ("body", 1, "The State of Bihar filed a suit against the company for "
                        "damages caused by the escape of gas."),
            ("order", 2, "The appeal is dismissed and the decree is confirmed."),
        ],
    },
]


def _record(doc_id: int, spec: dict) -> dict:
    petitioner, respondent = spec["title"].split(" versus ")
    return {
        "doc_id": doc_id,
        "source_path": f"test_{doc_id}",
        "title": spec["title"],
        "petitioner": petitioner,
        "respondent": respondent,
        "year": 2000 + doc_id,
        "judges": [],
        "bench_size": 2,
        "zones": {
            "title": spec["title"],
            "parties": f"{petitioner} {respondent}",
            "bench": spec["bench"],
            "statutes": "",
            "order": "\n\n".join(t for (z, _p, t) in spec["paras"] if z == "order"),
            "body": "\n\n".join(t for (z, _p, t) in spec["paras"] if z == "body"),
        },
        "paras": [{"id": pid, "zone": z, "text": t, "label": str(pid)}
                  for (z, pid, t) in spec["paras"]],
        "n_tokens": 0, "n_pages": 1,
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The real index plus a from-scratch token stream for every document."""
    tmp = tmp_path_factory.mktemp("prox")
    src = tmp / "judgments.jsonl"
    records = [_record(i, spec) for i, spec in enumerate(DOCS)]
    src.write_text("\n".join(json.dumps(r) for r in records) + "\n",
                   encoding="utf-8")
    out = tmp / "index"
    P.build(P.full_profile(out), src=src, processes=1, persist_statutes=False,
            verbose=False)
    idx = P.PositionalIndex.load(out)
    streams = {r["doc_id"]: _token_stream(r) for r in records}
    return idx, streams, records


# --------------------------------------------------------------------------
# The independent token stream and the naive scans
# --------------------------------------------------------------------------

class Stream:
    """One document's tokens, rebuilt from the text without touching the index."""

    def __init__(self, terms: List[str], zones: List[int], sents: List[int],
                 paras: List[int]):
        self.terms = terms
        self.zones = zones
        self.sents = sents
        self.paras = paras

    def positions_of(self, term: str) -> List[int]:
        return [i for i, t in enumerate(self.terms) if t == term]


def _token_stream(record: dict) -> Stream:
    terms: List[str] = []
    zones: List[int] = []
    sents: List[int] = []
    paras: List[int] = []
    pos = 0
    sent_no = -1
    para_no = -1
    last_skey = None
    last_pkey = None
    for zone, text, para_id in P._zone_texts(record, zoned=True,
                                             include_statutes=True):
        for t in analyze(text, zone=zone, para_id=para_id, start_pos=pos):
            skey = (zone, para_id, t.sent_id)
            if skey != last_skey:
                sent_no += 1
                last_skey = skey
            pkey = (zone, para_id)
            if pkey != last_pkey:
                para_no += 1
                last_pkey = pkey
            terms.append(t.term)
            zones.append(config.ZONE_ID[zone])
            sents.append(sent_no)
            paras.append(para_no)
            pos += 1
    return Stream(terms, zones, sents, paras)


def brute_phrase(st: Stream, words: Sequence[str]) -> bool:
    """Naive scan: is ``words`` a consecutive run inside one zone?"""
    n = len(st.terms)
    m = len(words)
    for i in range(n - m + 1):
        if st.terms[i:i + m] == list(words):
            if len(set(st.zones[i:i + m])) == 1:
                return True
    return False


def brute_window(st: Stream, a: str, b: str, k: int) -> bool:
    """Naive O(n*m) scan for |j - i| <= k, same zone, distinct positions."""
    pa = st.positions_of(a)
    pb = st.positions_of(b)
    for i in pa:
        for j in pb:
            if i != j and abs(i - j) <= k and st.zones[i] == st.zones[j]:
                return True
    return False


def brute_bucket(st: Stream, a: str, b: str, kind: str) -> bool:
    pa = st.positions_of(a)
    pb = st.positions_of(b)
    buckets = st.sents if kind == "s" else st.paras
    for i in pa:
        for j in pb:
            if i != j and buckets[i] == buckets[j] and st.zones[i] == st.zones[j]:
                return True
    return False


# --------------------------------------------------------------------------
# 1. Exhaustive brute-force agreement
# --------------------------------------------------------------------------

PROBE_TERMS = ["natur", "justic", "bail", "anticipatori", "privaci", "surveil",
               "appeal", "order", "bihar", "union", "state", "high", "court",
               "refus", "grant", "right"]


def test_phrase_agrees_with_the_naive_scan_on_every_adjacent_pair(built):
    idx, streams, _ = built
    checked = 0
    for a in PROBE_TERMS:
        for b in PROBE_TERMS:
            if a == b:
                continue
            got = set(map(int, prox.phrase_docs([a, b], idx)))
            want = {d for d, st in streams.items() if brute_phrase(st, [a, b])}
            assert got == want, f"phrase({a!r}, {b!r}): index {got} vs brute {want}"
            checked += 1
    assert checked == len(PROBE_TERMS) * (len(PROBE_TERMS) - 1)


@pytest.mark.parametrize("k", [1, 2, 5, 10, 50])
def test_window_operator_agrees_with_the_naive_scan(built, k):
    idx, streams, _ = built
    for a in PROBE_TERMS:
        for b in PROBE_TERMS:
            if a == b:
                continue
            got = set(map(int, prox.proximity_docs(None, None, [a], [b],
                                                   "n", k, idx)))
            want = {d for d, st in streams.items() if brute_window(st, a, b, k)}
            assert got == want, f"/{k}({a},{b}): index {got} vs brute {want}"


@pytest.mark.parametrize("kind", ["s", "p"])
def test_sentence_and_paragraph_operators_agree_with_the_naive_scan(built, kind):
    idx, streams, _ = built
    for a in PROBE_TERMS:
        for b in PROBE_TERMS:
            if a == b:
                continue
            got = set(map(int, prox.proximity_docs(None, None, [a], [b],
                                                   kind, 0, idx)))
            want = {d for d, st in streams.items() if brute_bucket(st, a, b, kind)}
            assert got == want, f"/{kind}({a},{b}): index {got} vs brute {want}"


def test_three_word_phrase_agrees_with_the_naive_scan(built):
    idx, streams, _ = built
    for words in (["natur", "justic", "requir"], ["right", "to", "privaci"],
                  ["state", "of", "bihar"], ["no", "order", "as"]):
        got = set(map(int, prox.phrase_docs(words, idx)))
        want = {d for d, st in streams.items() if brute_phrase(st, words)}
        assert got == want, f"phrase {words}: index {got} vs brute {want}"


# --------------------------------------------------------------------------
# 2. Operator semantics
# --------------------------------------------------------------------------

def test_phrase_is_ordered_but_the_window_operator_is_not(built):
    idx, _streams, _ = built
    assert prox.phrase_docs(["natur", "justic"], idx).size > 0
    assert prox.phrase_docs(["justic", "natur"], idx).size == 0
    fwd = prox.proximity_docs(None, None, ["natur"], ["justic"], "n", 2, idx)
    rev = prox.proximity_docs(None, None, ["justic"], ["natur"], "n", 2, idx)
    assert set(map(int, fwd)) == set(map(int, rev))


def test_the_operators_are_nested_phrase_subset_of_k_subset_of_s_subset_of_p(built):
    idx, _streams, _ = built
    a, b = "privaci", "surveil"
    phrase = set(map(int, prox.phrase_docs([a, b], idx)))
    k1 = set(map(int, prox.proximity_docs(None, None, [a], [b], "n", 1, idx)))
    k20 = set(map(int, prox.proximity_docs(None, None, [a], [b], "n", 20, idx)))
    sent = set(map(int, prox.proximity_docs(None, None, [a], [b], "s", 0, idx)))
    para = set(map(int, prox.proximity_docs(None, None, [a], [b], "p", 0, idx)))
    assert phrase <= k1 <= k20
    assert sent <= para
    assert k1 <= para


def test_a_larger_window_never_loses_a_document(built):
    idx, _streams, _ = built
    prev: set = set()
    for k in (1, 2, 3, 5, 8, 13, 50):
        got = set(map(int, prox.proximity_docs(None, None, ["bail"],
                                              ["anticipatori"], "n", k, idx)))
        assert prev <= got, f"k={k} lost a document that k<{k} had"
        prev = got


def test_a_term_is_never_its_own_neighbour(built):
    """The same position must not pair with itself, or every term would be
    within k of itself and /k would be a no-op."""
    idx, _streams, _ = built
    got = prox.proximity_docs(None, None, ["court"], ["court"], "n", 0, idx)
    assert got.size == 0


def test_an_unknown_term_makes_every_operator_empty(built):
    idx, _streams, _ = built
    assert prox.phrase_docs(["natur", "zzzznotaterm"], idx).size == 0
    assert prox.proximity_docs(None, None, ["natur"], ["zzzznotaterm"],
                               "n", 5, idx).size == 0
    assert prox.proximity_docs(None, None, ["zzzznotaterm"], ["natur"],
                               "s", 0, idx).size == 0


def test_a_single_term_phrase_is_just_that_term(built):
    idx, _streams, _ = built
    assert set(map(int, prox.phrase_docs(["bail"], idx))) == \
        set(map(int, prox.term_docs("bail", idx)))


def test_empty_input_is_empty_output(built):
    idx, _streams, _ = built
    assert prox.phrase_docs([], idx).size == 0
    assert prox.proximity_docs(None, None, [], ["bail"], "n", 3, idx).size == 0
    assert prox.proximity_docs(None, None, ["bail"], [], "n", 3, idx).size == 0


def test_an_unknown_kind_or_zone_is_an_error(built):
    idx, _streams, _ = built
    with pytest.raises(ValueError):
        prox.proximity_docs(None, None, ["a"], ["b"], "q", 1, idx)
    with pytest.raises(ValueError):
        prox.phrase_docs(["natur", "justic"], idx, zone="nosuchzone")


def test_results_are_ascending_int32_doc_ids(built):
    idx, _streams, _ = built
    for got in (prox.phrase_docs(["natur", "justic"], idx),
                prox.proximity_docs(None, None, ["bail"], ["refus"], "s", 0, idx)):
        assert got.dtype == np.int32
        assert list(got) == sorted(got)


# --------------------------------------------------------------------------
# 3. The cross-zone rule
# --------------------------------------------------------------------------

def test_a_phrase_must_not_straddle_a_zone_seam(built):
    """Every real zone seam in the fixture is checked, not a hand-picked one.

    Zones are concatenated with no gap, so the last token of one zone and the
    first of the next are consecutive integers in the position space while
    belonging to different zones.  For each such pair, the phrase may only match
    the document if that same pair *also* occurs inside one zone somewhere.
    """
    idx, streams, _ = built
    seams = 0
    for d, st in streams.items():
        for i in range(len(st.terms) - 1):
            if st.zones[i] == st.zones[i + 1]:
                continue
            seams += 1
            pair = [st.terms[i], st.terms[i + 1]]
            got = set(map(int, prox.phrase_docs(pair, idx)))
            legit = brute_phrase(st, pair)      # the same pair, inside one zone
            if not legit:
                assert d not in got, (
                    f"doc {d}: phrase {pair} matched only across the "
                    f"{config.ZONES[st.zones[i]]}/{config.ZONES[st.zones[i+1]]} seam")
    assert seams >= 8, "fixture no longer has zone seams to test"


def test_the_seam_pair_still_matches_when_it_also_occurs_inside_one_zone(built):
    """The rule rejects *seam* matches, not the words themselves.

    'bihar union' is adjacent across document 3's title/parties boundary and
    also legitimately adjacent inside its parties zone ('STATE OF BIHAR UNION
    CARBIDE'), so the document must still match - and the naive scan agrees.
    """
    idx, streams, _ = built
    got = set(map(int, prox.phrase_docs(["bihar", "union"], idx)))
    want = {d for d, s in streams.items() if brute_phrase(s, ["bihar", "union"])}
    assert got == want
    assert 3 in got


def test_zone_restriction_narrows_the_result(built):
    idx, _streams, _ = built
    anywhere = set(map(int, prox.phrase_docs(["state", "of"], idx)))
    in_title = set(map(int, prox.phrase_docs(["state", "of"], idx, zone="title")))
    in_body = set(map(int, prox.phrase_docs(["state", "of"], idx, zone="body")))
    assert in_title <= anywhere
    assert in_body <= anywhere
    assert in_title or in_body


def test_term_docs_zone_restriction_uses_the_stored_zone_tf(built):
    idx, _streams, _ = built
    all_docs = set(map(int, prox.term_docs("bihar", idx)))
    titled = set(map(int, prox.term_docs("bihar", idx, zone="title")))
    assert titled <= all_docs
    for d in titled:
        pl = idx.postings("bihar")
        assert pl.zone_tf_of(d).get("title", 0) > 0


# --------------------------------------------------------------------------
# 4. The candidate-narrowing arguments
# --------------------------------------------------------------------------

def test_narrowing_arguments_only_shrink_the_result(built):
    idx, _streams, _ = built
    full = prox.proximity_docs(None, None, ["bail"], ["refus"], "p", 0, idx)
    assert full.size
    narrowed = prox.proximity_docs(full[:1], None, ["bail"], ["refus"], "p", 0, idx)
    assert set(map(int, narrowed)) == {int(full[0])}


def test_several_terms_on_one_side_behave_as_an_or(built):
    idx, _streams, _ = built
    a = set(map(int, prox.proximity_docs(None, None, ["bail"], ["refus"],
                                        "p", 0, idx)))
    b = set(map(int, prox.proximity_docs(None, None, ["anticipatori"], ["refus"],
                                        "p", 0, idx)))
    both = set(map(int, prox.proximity_docs(None, None, ["bail", "anticipatori"],
                                           ["refus"], "p", 0, idx)))
    assert both == a | b


# --------------------------------------------------------------------------
# 5. Building blocks
# --------------------------------------------------------------------------

def test_phrase_hits_finds_the_consecutive_runs():
    a = np.array([1, 5, 9], dtype=np.int32)
    b = np.array([2, 6, 20], dtype=np.int32)
    assert list(prox.phrase_hits([a, b])) == [1, 5]
    c = np.array([3, 7], dtype=np.int32)
    assert list(prox.phrase_hits([a, b, c])) == [1, 5]


def test_window_pairs_is_a_two_pointer_walk_not_a_cross_product():
    left = np.array([0, 10, 20], dtype=np.int32)
    right = np.array([1, 11, 100], dtype=np.int32)
    assert prox.window_pairs(left, right, 1) == [(0, 1), (10, 11)]
    assert prox.window_pairs(left, right, 0) == []
    assert prox.window_pairs(left, right, 2) == [(0, 1), (10, 11)]


def test_window_pairs_can_be_asked_for_ordered_pairs_only():
    left = np.array([5], dtype=np.int32)
    right = np.array([3, 7], dtype=np.int32)
    assert prox.window_pairs(left, right, 3) == [(5, 3), (5, 7)]
    assert prox.window_pairs(left, right, 3, ordered=True) == [(5, 7)]


def test_intersect_doc_ids_orders_by_length_and_is_correct():
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    b = np.array([2, 4, 6], dtype=np.int32)
    c = np.array([4, 5], dtype=np.int32)
    assert list(prox.intersect_doc_ids([a, b, c])) == [4]
    assert prox.intersect_doc_ids([]).size == 0
    assert prox.intersect_doc_ids([a, np.empty(0, dtype=np.int32)]).size == 0


def test_phrase_occurrences_returns_the_places_not_just_the_fact(built):
    idx, streams, _ = built
    docs = prox.phrase_docs(["natur", "justic"], idx)
    d = int(docs[0])
    starts = prox.phrase_occurrences(["natur", "justic"], d, idx)
    assert starts.size >= 1
    st = streams[d]
    for s in map(int, starts):
        assert st.terms[s] == "natur" and st.terms[s + 1] == "justic"


def test_explain_proximity_reports_the_pairs_and_the_best_gap(built):
    idx, _streams, _ = built
    docs = prox.proximity_docs(None, None, ["bail"], ["refus"], "n", 5, idx)
    e = prox.explain_proximity(["bail"], ["refus"], int(docs[0]), "n", 5, idx)
    assert e is not None
    assert e.n_pairs >= 1
    assert 1 <= e.best_gap <= 5
    assert len(e.sample) <= 5
    assert prox.explain_proximity(["zzz"], ["bail"], int(docs[0]), "n", 5, idx) is None


# --------------------------------------------------------------------------
# 6. Regex-over-raw-text agreement, and the explainable mismatches
# --------------------------------------------------------------------------

def _raw_text(record: dict) -> str:
    """The judgment as a reader sees it, zone by zone - no analyser involved."""
    parts = []
    for z in config.ZONES:
        if z in ("body", "order"):
            parts.extend(p["text"] for p in record["paras"] if p["zone"] == z)
        else:
            t = record["zones"].get(z) or ""
            if t.strip():
                parts.append(t)
    return "\n".join(parts)


@pytest.mark.parametrize("surface, terms", [
    ("natural justice", ["natur", "justic"]),
    ("anticipatory bail", ["anticipatori", "bail"]),
    ("state of bihar", ["state", "of", "bihar"]),
    ("high court", ["high", "court"]),
])
def test_regex_over_raw_text_agrees_with_the_index_on_plain_phrases(
        built, surface, terms):
    """No stemming involved in these: the surface words are already the stems'
    own prefixes, so the two methods must agree exactly."""
    idx, _streams, records = built
    pat = re.compile(r"\b" + r"\s+".join(map(re.escape, surface.split())) + r"\b",
                     re.I)
    want = {r["doc_id"] for r in records if pat.search(_raw_text(r))}
    got = set(map(int, prox.phrase_docs(terms, idx)))
    assert got == want, f"{surface!r}: index {got} vs regex {want}"


def test_a_stemming_difference_is_a_real_and_explainable_mismatch(built):
    """'Privacy is a fundamental right' does not contain the surface phrase
    'privacy right', but it does contain the stems 'privaci' and 'right' in the
    same sentence - so /s finds it and a surface regex cannot.

    This is the mismatch class to explain in the report: the index matches
    *stems*, the regex matches *characters*, and the index is the one that
    answers the user's question.
    """
    idx, _streams, records = built
    pat = re.compile(r"\bprivacy\s+right\b", re.I)
    regex_hits = {r["doc_id"] for r in records if pat.search(_raw_text(r))}
    index_hits = set(map(int, prox.proximity_docs(None, None, ["privaci"],
                                                 ["right"], "s", 0, idx)))
    assert regex_hits == set()
    assert index_hits, "the stems do co-occur in a sentence"
    assert index_hits - regex_hits == index_hits


def test_the_index_never_claims_a_phrase_the_raw_text_cannot_support(built):
    """A soundness direction that must hold with no exceptions: if the index
    says a two-word phrase occurs, both surface words occur in that document."""
    idx, _streams, records = built
    by_id = {r["doc_id"]: _raw_text(r).lower() for r in records}
    for surface, terms in [("natural justice", ["natur", "justic"]),
                           ("anticipatory bail", ["anticipatori", "bail"]),
                           ("bail granted", ["bail", "grant"]),
                           ("surveillance citizens", ["surveil", "citizen"])]:
        for d in map(int, prox.phrase_docs(terms, idx)):
            for w in surface.split():
                assert w[:5] in by_id[d], f"{w!r} absent from doc {d}"
