"""Tests for the S3 positional index  (Member 1, wave 2).

Every test builds a real index with the real SPIMI code path, from a handful of
inline documents written to a temporary ``judgments.jsonl``.  No network, no
PDFs, no dependency on the 13,878-document corpus - but also no mocks: the
dictionary, ``postings.bin`` and ``docmeta.npz`` on disk are the ones the
production loader reads.

``config.SPIMI_BLOCK_DOCS`` is monkeypatched down to 2 in several tests so that
the heap merge across multiple blocks is actually exercised; with the real 600
a 6-document fixture would be a single block and the merge would never run.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from nyaya import config
from nyaya.contracts import Posting
from nyaya.index import positional as P

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

#: Deliberately plain English: the tests assert on structure and on terms whose
#: stem is stable, so they do not break when Member 3 tunes the analyser.
DOCS = [
    {
        "title": "RAM SINGH versus STATE OF BIHAR",
        "bench": "A. K. Sikri and Ashok Bhushan, JJ.",
        "paras": [
            ("body", 1, "The appellant was convicted by the trial court and the "
                        "High Court confirmed that conviction on appeal."),
            ("body", 2, "The question before this court is whether the evidence "
                        "of the single eyewitness can sustain a conviction."),
            ("order", 3, "In the result the appeal is allowed and the conviction "
                         "is set aside. No order as to costs."),
        ],
    },
    {
        "title": "MOHAN LAL versus UNION OF INDIA",
        "bench": "R. F. Nariman and Sanjay Kishan Kaul, JJ.",
        "paras": [
            ("body", 1, "The petitioner challenges the acquisition of his land "
                        "by the state government under the land acquisition law."),
            ("order", 2, "The writ petition is dismissed with no order as to costs."),
        ],
    },
    {
        "title": "KAMLA DEVI versus STATE OF KERALA",
        "bench": "Bela M. Trivedi, J.",
        "paras": [
            ("body", 1, "Bail was refused by the High Court and the appellant "
                        "renewed the bail application before this court."),
            ("body", 2, "The bail jurisprudence of this court requires the court "
                        "to balance liberty against the gravity of the offence."),
            ("order", 3, "Accordingly, the appeal is allowed and bail is granted."),
        ],
    },
    {
        "title": "STATE OF PUNJAB versus GURDEEP SINGH",
        "bench": "Abhay S. Oka and Augustine George Masih, JJ.",
        "paras": [
            ("body", 1, "The respondent was acquitted by the High Court of the "
                        "charge of murder and the state has appealed."),
            ("order", 2, "The appeal is dismissed and the acquittal is confirmed."),
        ],
    },
    {
        "title": "XYZZY CORPORATION versus FROBNICATE LIMITED",
        "bench": "A. P. Sen, J.",
        "paras": [
            ("body", 1, "A quux arbitration award was challenged on the ground "
                        "that the arbitrator had exceeded his jurisdiction."),
            ("order", 2, "The appeal is allowed in part."),
        ],
    },
    {
        "title": "SURESH KUMAR versus STATE OF GOA",
        "bench": "S. N. Variava and S. H. Kapadia, JJ.",
        "paras": [
            ("body", 1, "The accused was tried for an offence of cheating and the "
                        "trial court recorded a conviction on the documents."),
            ("order", 2, "The appeal is partly allowed and the sentence reduced."),
        ],
    },
]


def _record(doc_id: int, spec: dict) -> dict:
    """One judgments.jsonl row, in exactly the shape nyaya.zones writes."""
    petitioner, respondent = spec["title"].split(" versus ")
    paras = [{"id": pid, "zone": z, "text": t, "label": str(pid)}
             for (z, pid, t) in spec["paras"]]
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
        "paras": paras,
        "n_tokens": 0,
        "n_pages": 1,
    }


@pytest.fixture
def corpus(tmp_path):
    """A 6-document judgments.jsonl with a gap in the doc_id sequence.

    doc_ids are 0, 1, 2, 4, 5, 9: the index must be addressed by the manifest
    doc_id, not by a private row number, and 3/6/7/8 stand for the 64 manifest
    rows that failed extraction in the real corpus.
    """
    ids = [0, 1, 2, 4, 5, 9]
    src = tmp_path / "judgments.jsonl"
    with src.open("w", encoding="utf-8") as fh:
        for doc_id, spec in zip(ids, DOCS):
            fh.write(json.dumps(_record(doc_id, spec), ensure_ascii=False) + "\n")
    return src, ids


@pytest.fixture
def index(tmp_path, corpus, monkeypatch):
    """The real index, built through SPIMI with 3 blocks of 2 documents."""
    src, ids = corpus
    monkeypatch.setattr(config, "SPIMI_BLOCK_DOCS", 2)
    monkeypatch.setattr(P.config, "SPIMI_BLOCK_DOCS", 2)
    out = tmp_path / "index"
    meta = P.build(P.full_profile(out), src=src, processes=1,
                   persist_statutes=False, verbose=False)
    assert meta["spimi_blocks"] == 3, "the heap merge must actually run"
    return P.PositionalIndex.load(out), ids, meta


# --------------------------------------------------------------------------
# Record layout arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("df, npos", [(1, 1), (5, 40), (17, 300), (1000, 50_000)])
def test_record_ints_and_positions_in_are_inverses(df, npos):
    size = P.record_ints(df, npos)
    assert P.positions_in(df, size) == npos


def test_record_length_accounts_for_every_section():
    df, npos = 10, 77
    expected = (df                      # doc_ids
                + df * P.NZONES         # zone tfs
                + df + 1                # pos_offsets
                + npos                  # positions
                + P._n_skips(df))       # skip pointers
    assert P.record_ints(df, npos) == expected


def test_skip_stride_is_about_sqrt_of_df():
    assert P.skip_stride_for(10) == 0          # too short to be worth it
    assert P.skip_stride_for(100) == 10
    assert P.skip_stride_for(10_000) == 100
    assert P._n_skips(100) == 10


# --------------------------------------------------------------------------
# The contract in docs/CONTRACTS.md Section 4
# --------------------------------------------------------------------------


def test_index_exposes_the_contracted_attributes(index):
    idx, ids, _ = index
    assert idx.doc_count == len(ids)
    assert idx.doc_len.shape == (max(ids) + 1,)
    assert idx.zone_len.shape == (max(ids) + 1, len(config.ZONES))
    assert idx.doc_norm.shape == (max(ids) + 1,)
    assert idx.vocab_size > 50


def test_doc_ids_are_manifest_ids_with_gaps_left_empty(index):
    idx, ids, _ = index
    assert list(idx.indexed) == ids
    for d in ids:
        assert idx.doc_len[d] > 0
    for d in set(range(max(ids) + 1)) - set(ids):
        assert idx.doc_len[d] == 0, f"slot {d} was never indexed"


def test_doc_count_is_the_N_used_by_idf_not_the_number_of_slots(index):
    idx, ids, _ = index
    assert idx.doc_count == 6
    assert idx.n_doc_slots == 10


def test_idf_is_exactly_the_contracted_formula(index):
    idx, _ids, _ = index
    for term in ("court", "appeal", "bail"):
        d = idx.df(term)
        if not d:
            continue
        n = idx.doc_count
        assert idx.idf(term) == pytest.approx(
            math.log(1.0 + (n - d + 0.5) / (d + 0.5)))


def test_idf_of_a_term_in_every_document_is_positive(index):
    idx, _ids, _ = index
    # the point of the +0.5 smoothing: ln(N/df) would be exactly 0 here
    for term, df in idx.terms_by_df(3):
        if df == idx.doc_count:
            assert idx.idf(term) > 0
            break
    else:
        pytest.skip("no term appears in every fixture document")


def test_df_of_an_unknown_term_is_zero_and_postings_are_none(index):
    idx, _ids, _ = index
    assert idx.df("thisisnotaterm") == 0
    assert idx.postings("thisisnotaterm") is None


def test_postings_list_satisfies_the_postings_list_protocol(index):
    idx, _ids, _ = index
    pl = idx.postings("court")
    assert pl is not None
    assert pl.df == len(pl.doc_ids) == len(pl)
    assert pl.doc_ids.dtype == np.int32
    assert isinstance(pl.skip_stride(), int)
    assert isinstance(pl.zone_tf_of(int(pl.doc_ids[0])), dict)
    assert isinstance(pl.positions_of(int(pl.doc_ids[0])), np.ndarray)
    assert all(isinstance(p, Posting) for p in pl)


# --------------------------------------------------------------------------
# Postings correctness
# --------------------------------------------------------------------------


def test_doc_ids_are_ascending_even_though_they_came_from_three_blocks(index):
    idx, _ids, _ = index
    for term, _df in idx.terms_by_df(40):
        d = idx.postings(term).doc_ids
        assert np.all(np.diff(d) > 0), f"{term} postings are not ascending"


def test_positions_are_sorted_and_within_the_document(index):
    idx, ids, _ = index
    for term, _df in idx.terms_by_df(40):
        pl = idx.postings(term)
        for p in pl:
            pos = np.asarray(p.positions)
            assert np.all(np.diff(pos) > 0)
            assert pos[0] >= 0
            assert pos[-1] < idx.doc_len[p.doc_id]


def test_zone_tf_sums_to_the_number_of_positions(index):
    idx, _ids, _ = index
    for term, _df in idx.terms_by_df(60):
        pl = idx.postings(term)
        for p in pl:
            assert p.tf == len(p.positions), f"{term} in doc {p.doc_id}"


def test_zone_tf_of_omits_zero_zones_but_the_matrix_keeps_them(index):
    idx, _ids, _ = index
    pl = idx.postings("court")
    d = int(pl.doc_ids[0])
    assert all(v > 0 for v in pl.zone_tf_of(d).values())
    assert pl.zone_tf_matrix().shape == (pl.df, len(config.ZONES))
    assert pl.zone_tf_matrix().sum() == sum(pl.tf_array())


def test_positions_of_an_absent_document_is_empty(index):
    idx, _ids, _ = index
    pl = idx.postings("court")
    assert pl.positions_of(10_000).size == 0
    assert pl.zone_tf_of(10_000) == {}
    assert pl.index_of(10_000) == -1


def test_total_positions_equals_the_sum_of_document_lengths(index):
    idx, _ids, meta = index
    assert meta["total_positions"] == int(idx.doc_len.sum())


def test_every_term_of_a_document_is_retrievable_at_its_position(index):
    """The invariant that matters: position p of document d holds term t iff
    t's postings list contains d with p among its positions."""
    from nyaya.analyze import analyze

    idx, ids, _ = index
    d = ids[2]
    rec = _record(d, DOCS[2])
    # rebuild the expected stream the same way the builder does
    expected: dict[int, str] = {}
    pos = 0
    for zone, text, para_id in P._zone_texts(rec, zoned=True):
        for t in analyze(text, zone=zone, para_id=para_id, start_pos=pos):
            expected[pos] = t.term
            pos += 1
    assert pos == int(idx.doc_len[d])
    for p, term in expected.items():
        pl = idx.postings(term)
        assert pl is not None, f"{term!r} missing from the dictionary"
        assert p in set(map(int, pl.positions_of(d))), f"{term!r} at {p}"


# --------------------------------------------------------------------------
# Skip pointers
# --------------------------------------------------------------------------


def test_index_of_finds_every_document_through_the_skip_pointers(tmp_path,
                                                                monkeypatch):
    """Built wide on purpose: df must exceed 16 for skips to exist at all."""
    monkeypatch.setattr(config, "SPIMI_BLOCK_DOCS", 10)
    monkeypatch.setattr(P.config, "SPIMI_BLOCK_DOCS", 10)
    src = tmp_path / "judgments.jsonl"
    n = 60
    with src.open("w", encoding="utf-8") as fh:
        for i in range(n):
            spec = dict(DOCS[i % len(DOCS)])
            fh.write(json.dumps(_record(i * 2, spec), ensure_ascii=False) + "\n")
    out = tmp_path / "index"
    P.build(P.full_profile(out), src=src, processes=1, persist_statutes=False,
            verbose=False)
    idx = P.PositionalIndex.load(out)
    pl = idx.postings("court")
    assert pl.df > 16
    assert pl.skip_stride() == max(1, int(math.sqrt(pl.df)))
    assert pl.skips.size == P._n_skips(pl.df)
    # the skip array holds the doc_id *at* each stride boundary
    assert list(pl.skips) == list(pl.doc_ids[::pl.skip_stride()])
    for j, d in enumerate(pl.doc_ids):
        assert pl.index_of(int(d)) == j
    assert pl.index_of(int(pl.doc_ids[-1]) + 1) == -1


# --------------------------------------------------------------------------
# Document statistics
# --------------------------------------------------------------------------


def test_doc_len_equals_the_sum_of_zone_len(index):
    idx, ids, _ = index
    for d in ids:
        assert int(idx.doc_len[d]) == int(idx.zone_len[d].sum())


def test_doc_norm_is_the_lnc_vector_length_over_scoring_terms(index):
    """Recompute the cosine denominator by hand for one document."""
    idx, ids, _ = index
    from nyaya import stoplist as sl

    d = ids[0]
    tf: dict[str, int] = {}
    for term, _df in [(t, 0) for t in idx._terms]:
        pl = idx.postings(term)
        k = len(pl.positions_of(d))
        if k:
            tf[term] = k
    expected = math.sqrt(sum((1 + math.log(n)) ** 2 for t, n in tf.items()
                             if not sl.is_scoring_stopword(t)))
    assert float(idx.doc_norm[d]) == pytest.approx(expected, rel=1e-5)


def test_doc_norm_excludes_stop_words_so_it_is_smaller_than_the_full_norm(index):
    idx, ids, _ = index
    d = ids[0]
    tf: dict[str, int] = {}
    for term in idx._terms:
        k = len(idx.postings(term).positions_of(d))
        if k:
            tf[term] = k
    full = math.sqrt(sum((1 + math.log(n)) ** 2 for n in tf.values()))
    assert float(idx.doc_norm[d]) < full


def test_zone_len_puts_tokens_in_the_right_zone(index):
    idx, ids, _ = index
    zid = config.ZONE_ID
    for d in ids:
        assert idx.zone_len[d, zid["title"]] > 0
        assert idx.zone_len[d, zid["bench"]] > 0
        assert idx.zone_len[d, zid["body"]] > 0
        assert idx.zone_len[d, zid["order"]] > 0


# --------------------------------------------------------------------------
# Sentence / paragraph / zone lookup
# --------------------------------------------------------------------------


def test_sent_and_para_ids_are_non_decreasing_across_a_document(index):
    idx, ids, _ = index
    for d in ids:
        pos = np.arange(int(idx.doc_len[d]))
        s = idx.sent_ids_at(d, pos)
        p = idx.para_ids_at(d, pos)
        assert s[0] == 0 and p[0] == 0
        assert np.all(np.diff(s) >= 0)
        assert np.all(np.diff(p) >= 0)
        assert int(s[-1]) + 1 == idx.n_sentences(d)
        assert int(p[-1]) + 1 == idx.n_paragraphs(d)


def test_two_terms_in_the_same_sentence_share_a_sentence_id(index):
    idx, ids, _ = index
    d = ids[0]
    # "appellant ... convicted ... trial court" are one sentence of document 0
    a = idx.postings("appel")
    assert a is not None
    pa = a.positions_of(d)
    assert pa.size
    sa = idx.sent_ids_at(d, pa)
    c = idx.postings("convict")
    sc = idx.sent_ids_at(d, c.positions_of(d))
    assert set(map(int, sa)) & set(map(int, sc)), "no shared sentence id"


def test_zone_at_maps_positions_back_to_their_zone(index):
    idx, ids, _ = index
    zid = config.ZONE_ID
    for d in ids:
        bounds = idx.zone_bounds(d)
        assert int(bounds[-1]) == int(idx.doc_len[d])
        # the first token of the document is in the first non-empty zone
        first = int(idx.zone_at(d, [0])[0])
        assert first == zid["title"]
        last = int(idx.zone_at(d, [int(idx.doc_len[d]) - 1])[0])
        assert last == zid["body"]


def test_a_cross_zone_adjacency_is_detectable(index):
    """Zones are concatenated without a gap, so a phrase operator must reject a
    match that straddles a boundary - zone_at is how it does that."""
    idx, ids, _ = index
    d = ids[0]
    bounds = idx.zone_bounds(d)
    b = int(bounds[config.ZONE_ID["title"]])        # first position of 'parties'
    assert int(idx.zone_at(d, [b - 1])[0]) != int(idx.zone_at(d, [b])[0])


# --------------------------------------------------------------------------
# prefix_terms, for the bail! root operator
# --------------------------------------------------------------------------


def test_prefix_terms_finds_every_term_with_that_prefix(index):
    idx, _ids, _ = index
    hits = idx.prefix_terms("bail")
    assert hits
    assert all(t.startswith("bail") for t in hits)
    assert set(hits) == {t for t in idx._terms if t.startswith("bail")}


def test_prefix_terms_is_empty_for_an_unknown_prefix(index):
    idx, _ids, _ = index
    assert idx.prefix_terms("zzzzqq") == []
    assert idx.prefix_terms("") == []


def test_the_dictionary_is_stored_sorted_so_the_prefix_scan_can_bisect(index):
    idx, _ids, _ = index
    assert idx._terms == sorted(idx._terms)


# --------------------------------------------------------------------------
# The baseline index (B0)
# --------------------------------------------------------------------------


def test_baseline_index_puts_everything_in_one_body_zone(tmp_path, corpus,
                                                         monkeypatch):
    src, ids = corpus
    monkeypatch.setattr(config, "SPIMI_BLOCK_DOCS", 2)
    monkeypatch.setattr(P.config, "SPIMI_BLOCK_DOCS", 2)
    out = tmp_path / "baseline"
    P.build(P.baseline_profile(out), src=src, processes=1,
            persist_statutes=False, verbose=False)
    idx = P.PositionalIndex.load(out)
    zid = config.ZONE_ID
    for d in ids:
        assert int(idx.zone_len[d, zid["body"]]) == int(idx.doc_len[d])
        for z in ("title", "parties", "bench", "statutes", "order"):
            assert int(idx.zone_len[d, zid[z]]) == 0


def test_baseline_profile_turns_statute_canonicalisation_off():
    baseline = P.baseline_profile()
    assert baseline.canonicalise is False
    assert baseline.zoned is False
    assert baseline.stoplist == "standard"
    assert P.full_profile().canonicalise is True
    assert P.full_profile().stoplist == "legal"


def test_baseline_does_not_mint_statute_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPIMI_BLOCK_DOCS", 2)
    monkeypatch.setattr(P.config, "SPIMI_BLOCK_DOCS", 2)
    spec = {
        "title": "A versus B",
        "bench": "X, J.",
        "paras": [("body", 1, "The accused was convicted under Section 302 of "
                              "the Indian Penal Code by the trial court."),
                  ("order", 2, "The appeal is dismissed.")],
    }
    src = tmp_path / "judgments.jsonl"
    src.write_text(json.dumps(_record(0, spec)) + "\n", encoding="utf-8")

    full = tmp_path / "full"
    base = tmp_path / "base"
    P.build(P.full_profile(full), src=src, processes=1, persist_statutes=False,
            verbose=False)
    P.build(P.baseline_profile(base), src=src, processes=1,
            persist_statutes=False, verbose=False)
    fi = P.PositionalIndex.load(full)
    bi = P.PositionalIndex.load(base)
    statute_terms = [t for t in fi._terms if "§" in t]
    assert statute_terms, "the full index should canonicalise 302 IPC"
    assert not [t for t in bi._terms if "§" in t]
    # and the baseline keeps the surface number as an ordinary term
    assert bi.df("302") >= 1


def test_the_baseline_stop_list_is_the_standard_one_not_the_legal_one():
    from nyaya import stoplist as sl

    legal_only = set(sl.SCORING_STOPWORDS) - set(sl.STANDARD_STOPWORDS)
    assert legal_only, "the measured domain list should add terms"
    probe = sorted(legal_only)[0]
    assert P._is_scoring_stopword(probe, "legal") is True
    assert P._is_scoring_stopword(probe, "standard") is False


def test_a_statute_token_is_never_a_stop_word_in_either_profile():
    for sl_name in ("legal", "standard"):
        assert P._is_scoring_stopword("ipc§302", sl_name) is False


# --------------------------------------------------------------------------
# Statute zone enrichment
# --------------------------------------------------------------------------


def test_build_fills_the_statutes_zone_that_zones_py_left_empty(tmp_path,
                                                               monkeypatch):
    monkeypatch.setattr(config, "SPIMI_BLOCK_DOCS", 2)
    monkeypatch.setattr(P.config, "SPIMI_BLOCK_DOCS", 2)
    spec = {
        "title": "A versus B",
        "bench": "X, J.",
        "paras": [("body", 1, "He was convicted under Section 302 of the Indian "
                              "Penal Code and Article 21 of the Constitution "
                              "was invoked before the High Court."),
                  ("order", 2, "The appeal is dismissed.")],
    }
    rec = _record(0, spec)
    assert rec["zones"]["statutes"] == ""      # wave 1 leaves it blank
    dt = P.analyse_record(rec, P.full_profile(tmp_path))
    assert rec["zones"]["statutes"], "the statutes zone must be synthesised"
    assert dt.statute_tokens
    assert all("§" in t for t in dt.statute_tokens)
    assert dt.acts
    assert dt.zone_len[config.ZONE_ID["statutes"]] > 0


def test_persist_statutes_writes_the_tokens_back_into_the_corpus(tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(config, "SPIMI_BLOCK_DOCS", 2)
    monkeypatch.setattr(P.config, "SPIMI_BLOCK_DOCS", 2)
    spec = {
        "title": "A versus B",
        "bench": "X, J.",
        "paras": [("body", 1, "Convicted under Section 302 of the Indian Penal "
                              "Code by the trial court on the evidence."),
                  ("order", 2, "The appeal is dismissed.")],
    }
    src = tmp_path / "judgments.jsonl"
    src.write_text(json.dumps(_record(0, spec)) + "\n", encoding="utf-8")
    P.build(P.full_profile(tmp_path / "idx"), src=src, processes=1,
            keep_blocks=True, persist_statutes=True, verbose=False)
    row = json.loads(src.read_text(encoding="utf-8").strip())
    assert row["statute_tokens"], "Member 2's act: filter reads this"
    assert row["acts"]
    assert row["zones"]["statutes"]


# --------------------------------------------------------------------------
# Load errors and reporting
# --------------------------------------------------------------------------


def test_load_of_a_missing_index_tells_you_how_to_build_it(tmp_path):
    with pytest.raises(SystemExit) as exc:
        P.PositionalIndex.load(tmp_path / "nope")
    assert "build" in str(exc.value)


def test_stats_reports_real_numbers(index):
    idx, ids, _ = index
    s = idx.stats()
    assert s["documents"] == len(ids)
    assert s["vocabulary"] == idx.vocab_size
    assert s["total_positions"] == int(idx.doc_len.sum())
    assert s["total_postings"] == sum(idx.df(t) for t in idx._terms)
    assert sum(s["zone_tokens"].values()) == int(idx.doc_len.sum())
    assert s["sentences"] > 0 and s["paragraphs"] > 0


def test_terms_by_df_is_sorted_descending(index):
    idx, _ids, _ = index
    top = idx.terms_by_df(10)
    assert [d for _t, d in top] == sorted((d for _t, d in top), reverse=True)
    assert all(idx.df(t) == d for t, d in top)


def test_the_dictionary_records_the_profile_so_an_index_is_self_describing(index):
    idx, _ids, _ = index
    prof = idx.meta["profile"]
    assert prof["name"] == "full"
    assert prof["canonicalise"] is True
    assert prof["zoned"] is True
    assert idx.meta["zones"] == list(config.ZONES)
    assert idx.meta["spimi_block_docs"] == 2
