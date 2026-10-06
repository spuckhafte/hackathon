"""Tests for store S2 — the parametric index.  Network-free, inline frames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nyaya.index.parametric import (
    ParametricIndex,
    ParametricPostings,
    build,
    intersect,
    union,
)
from nyaya.metadata import MANIFEST_COLUMNS


# --------------------------------------------------------------------------
# A tiny corpus: 6 documents whose facts are easy to reason about by hand
# --------------------------------------------------------------------------
_DOCS = [
    # doc_id, year, bench(pdf), judges(pdf),            type,            disposal
    (0, 1954, 5, ["M.C. MAHAJAN", "B.K. MUKHERJEA"], "civil appeal", "allowed"),
    (1, 1973, 13, ["S.M. SIKRI", "D.Y. CHANDRACHUD"], "writ petition", "partly allowed"),
    (2, 1995, 2, ["J.S. VERMA"], "criminal appeal", "dismissed"),
    (3, 2000, 3, ["D.Y. CHANDRACHUD"], "civil appeal", "dismissed"),
    (4, 2017, 9, ["D.Y. CHANDRACHUD", "J.S. KHEHAR"], "writ petition", "allowed"),
    (5, 2023, 2, ["D.Y. CHANDRACHUD"], "", "disposed"),
]


def _manifest():
    rows = []
    for doc_id, year, _bench, _judges, _type, disposal in _DOCS:
        rows.append(
            {
                "doc_id": doc_id, "node_id": 1000 + doc_id, "year": year,
                "path": f"{year}_1_{doc_id}_9", "title": f"A{doc_id} versus B{doc_id}",
                "title_key": f"a{doc_id} b{doc_id}", "title_tokens": [f"a{doc_id}"],
                "neutral_cite": "", "neutral_key": "", "scr_cite": "", "scr_key": "",
                "decision_date": f"{year}-01-01", "disposal": disposal,
                # metadata judges are deliberately WRONG/thin here: one name only,
                # which is what the real column looks like.  The PDF must win.
                "judges": ["SOME AUTHOR"], "judge_keys": ["some author", "author"],
                "bench_size_meta": 1, "case_type": "",
            }
        )
    return pd.DataFrame(rows)[MANIFEST_COLUMNS]


def _records(acts_for=None):
    acts_for = acts_for or {}
    out = []
    for doc_id, _year, bench, judges, ctype, _disposal in _DOCS:
        out.append(
            {
                "doc_id": doc_id, "bench_size": bench, "judges": judges,
                "acts": acts_for.get(doc_id, []), "statute_tokens": [],
                "case_type": ctype, "case_no": "", "disposal": "",
                "date": "", "neutral_cite": "", "scr_cite": "",
            }
        )
    return out


@pytest.fixture
def idx():
    # acts_sidecar={} keeps the test hermetic: without it, build() reads the
    # real data/processed/statutes.jsonl off disk.
    return build(_manifest(), _records({0: ["CONST"], 1: ["CONST"], 4: ["CONST", "IPC"]}),
                 verbose=False, acts_sidecar={})


# --------------------------------------------------------------------------
# Numeric fields: range predicates via binary search
# --------------------------------------------------------------------------
def test_year_range_select(idx):
    assert idx.select("year", ">=", 2000).tolist() == [3, 4, 5]
    assert idx.select("year", ">", 2000).tolist() == [4, 5]
    assert idx.select("year", "<=", 1973).tolist() == [0, 1]
    assert idx.select("year", "<", 1954).tolist() == []
    assert idx.select("year", "=", 1995).tolist() == [2]
    assert idx.select("year", "!=", 1995).tolist() == [0, 1, 3, 4, 5]


def test_year_in_list(idx):
    assert idx.select("year", "in", [1954, 2023]).tolist() == [0, 5]


def test_select_returns_ascending_int32_like_a_postings_list(idx):
    for op, val in (("<=", 2017), (">=", 1950), ("=", 2017)):
        out = idx.select("year", op, val)
        assert out.dtype == np.int32
        assert out.tolist() == sorted(out.tolist())


def test_range_select_agrees_with_a_brute_force_scan(idx):
    """The binary search must give exactly what an O(N) scan would."""
    year = idx.column("year")
    for v in (1950, 1954, 1973, 1999, 2000, 2023, 2030):
        assert idx.select("year", ">=", v).tolist() == np.flatnonzero(year >= v).tolist()
        assert idx.select("year", "<", v).tolist() == np.flatnonzero(year < v).tolist()


def test_bench_comes_from_the_pdf_not_from_bench_size_meta(idx):
    """The whole point of merge_from_record: metadata says 1 for every doc."""
    assert idx.column("bench").tolist() == [5, 13, 2, 3, 9, 2]
    assert idx.select("bench", ">=", 5).tolist() == [0, 1, 4]   # Constitution benches
    assert idx.select("bench", "=", 13).tolist() == [1]


def test_string_values_are_coerced_for_numeric_fields(idx):
    """The CLI and the query parser both hand us strings."""
    assert idx.select("year", ">=", "2000").tolist() == [3, 4, 5]
    with pytest.raises(ValueError):
        idx.select("year", ">=", "recently")


# --------------------------------------------------------------------------
# Set and categorical fields: membership only
# --------------------------------------------------------------------------
def test_judge_set_field_matches_surname_and_full_name(idx):
    assert idx.select("judge", "=", "chandrachud").tolist() == [1, 3, 4, 5]
    assert idx.select("judge", "=", "d.y. chandrachud").tolist() == [1, 3, 4, 5]
    assert idx.select("judge", "=", "CHANDRACHUD").tolist() == [1, 3, 4, 5]   # case-fold
    assert idx.select("judge", "=", "nobody here").tolist() == []


def test_metadata_judge_is_not_indexed_when_the_pdf_supplied_one(idx):
    assert idx.select("judge", "=", "some author").tolist() == []


def test_type_and_disposal_are_categorical(idx):
    assert idx.select("type", "=", "civil appeal").tolist() == [0, 3]
    assert idx.select("type", "=", '"writ petition"').tolist() == [1, 4]   # quotes stripped
    assert idx.select("disposal", "=", "allowed").tolist() == [0, 4]
    assert idx.select("disposal", "in", ["allowed", "dismissed"]).tolist() == [0, 2, 3, 4]


def test_empty_categorical_value_is_not_indexed(idx):
    """doc 5 has no case type; it must not land under the empty-string key."""
    assert "" not in idx.dicts["type"]
    assert 5 not in idx.select("type", "in", idx.values("type")).tolist()


def test_act_field_is_populated_from_the_record(idx):
    assert idx.select("act", "=", "const").tolist() == [0, 1, 4]
    assert idx.select("act", "=", "ipc").tolist() == [4]


def test_act_field_is_empty_but_present_when_m1_has_not_filled_it():
    """Must not explode before the statutes pass lands — just select nothing."""
    bare = build(_manifest(), _records(), verbose=False, acts_sidecar={})
    assert bare.dicts["act"] == {}
    assert bare.select("act", "=", "ipc").tolist() == []


def test_range_operators_are_rejected_on_a_set_field(idx):
    """A judge has no ordering; letting `judge:>=x` through would be nonsense."""
    with pytest.raises(ValueError):
        idx.select("judge", ">=", "chandrachud")
    with pytest.raises(ValueError):
        idx.select("disposal", "<", "allowed")


def test_unknown_field_raises(idx):
    with pytest.raises(KeyError):
        idx.select("vibes", "=", "good")


def test_field_aliases(idx):
    assert idx.select("bench_size", ">=", 5).tolist() == idx.select("bench", ">=", 5).tolist()
    assert idx.select("judges", "=", "chandrachud").tolist() == \
        idx.select("judge", "=", "chandrachud").tolist()


# --------------------------------------------------------------------------
# The equivalence with postings — the reason this store composes at all
# --------------------------------------------------------------------------
def test_postings_adapter_satisfies_the_postings_read_contract(idx):
    post = idx.postings("year", ">=", 2000)
    assert isinstance(post, ParametricPostings)
    assert post.doc_ids.tolist() == [3, 4, 5]
    assert post.df == 3 == len(post)
    assert post.skip_stride() >= 1


def test_postings_adapter_refuses_to_pretend_it_has_text_statistics(idx):
    post = idx.postings("year", ">=", 2000)
    with pytest.raises(TypeError):
        post.positions_of(3)
    with pytest.raises(TypeError):
        post.zone_tf_of(3)


def test_two_filters_intersect_like_postings_lists(idx):
    """`bench:>=5 AND judge:chandrachud` is a plain sorted-array merge."""
    a = idx.select("bench", ">=", 5)
    b = idx.select("judge", "=", "chandrachud")
    assert intersect(a, b).tolist() == [1, 4]


def test_union_of_selections(idx):
    a = idx.select("year", "=", 1954)
    b = idx.select("year", "=", 2023)
    assert union([a, b]).tolist() == [0, 5]
    assert union([]).tolist() == []


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def test_round_trip_through_disk(tmp_path, idx):
    idx.save(tmp_path)
    assert (tmp_path / "parametric.json").exists()
    assert (tmp_path / "parametric_year.npy").exists()
    assert (tmp_path / "parametric_year_ord.npy").exists()

    back = ParametricIndex.load(tmp_path)
    assert back.n_docs == idx.n_docs
    for field, op, value in (("year", ">=", 2000), ("bench", ">=", 5),
                             ("judge", "=", "chandrachud"),
                             ("type", "=", "civil appeal"),
                             ("disposal", "=", "allowed"),
                             ("act", "=", "const")):
        assert back.select(field, op, value).tolist() == \
            idx.select(field, op, value).tolist(), field


def test_load_without_a_build_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ParametricIndex.load(tmp_path)


def test_saved_header_records_the_bench_provenance(idx):
    assert idx.header["bench_from_pdf"] == len(_DOCS)
    assert idx.header["records_without_pdf_bench"] == 0


# --------------------------------------------------------------------------
# A manifest document with no extracted record must not vanish
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Hardening against M1's noisy CORAM parse
# --------------------------------------------------------------------------
def test_prose_caught_by_the_coram_parse_is_not_indexed_as_a_judge():
    """Real example from the corpus: the parse ran into the surrounding text."""
    from nyaya.index.parametric import _judge_lookup_keys

    keys = _judge_lookup_keys([
        "for the respondent. the judgment of the court was delivered by CHANDRACHUD",
    ])
    assert keys == []


def test_ocr_glued_and_is_split_into_two_judges():
    """"H.R. KHANNA ANDY. V. CHANDRACHUD" is two judges, not one."""
    from nyaya.index.parametric import _judge_lookup_keys

    keys = _judge_lookup_keys(["H.R. KHANNA ANDY. V. CHANDRACHUD"])
    assert "khanna" in keys
    assert "chandrachud" in keys
    assert not any("andy" in k for k in keys)


@pytest.mark.parametrize(
    "name,ok",
    [
        ("D.Y. CHANDRACHUD", True),
        ("MEHR CHAND MAHAJAN", True),
        ("SIR SYED FAZL ALI", True),
        ("PER CHANDRACHUD", False),                    # "per" is prose
        ("THE JUDGMENT OF THE COURT", False),
        ("1980 V. CHANDRACHUD", False),                # a year is not a name
        ("A B C D E F G H", False),                    # too many words
        ("", False),
    ],
)
def test_plausible_judge(name, ok):
    from nyaya.index.parametric import _plausible_judge

    assert _plausible_judge(name) is ok


def test_noisy_judges_do_not_stop_the_clean_ones_being_indexed():
    man = _manifest()
    recs = _records()
    recs[0]["judges"] = ["PER CHANDRACHUD", "D.Y. CHANDRACHUD", "A.K. SIKRI"]
    idx = build(man, recs, verbose=False, acts_sidecar={})
    # doc 1 already has S.M. Sikri, so the surname key holds both
    assert idx.select("judge", "=", "sikri").tolist() == [0, 1]
    assert idx.select("judge", "=", "a.k. sikri").tolist() == [0]
    assert 0 in idx.select("judge", "=", "chandrachud").tolist()


def test_doc_without_a_record_keeps_its_metadata_facts():
    """53 scanned + 7 short + 2 + 2 documents have no record; they still have a year."""
    man = _manifest()
    recs = [r for r in _records() if r["doc_id"] != 2]
    idx = build(man, recs, verbose=False, acts_sidecar={})
    assert idx.header["docs_without_any_record"] == 1
    assert 2 in idx.select("year", "=", 1995).tolist()       # still filterable by year
    assert idx.column("bench")[2] == 1                       # metadata fallback, honest
    assert idx.select("judge", "=", "author").tolist() == [2]  # metadata judge fallback


# --------------------------------------------------------------------------
# acts come from M1's statutes sidecar, not from the 960 MB corpus
# --------------------------------------------------------------------------
def test_acts_are_taken_from_the_sidecar_when_the_corpus_lacks_them():
    """M1 writes a small statutes.jsonl sidecar so a locked corpus cannot lose
    the build (his 431942b).  S2 must prefer it."""
    idx = build(_manifest(), _records(), verbose=False,
                acts_sidecar={0: ["const", "ipc"], 3: ["crpc"]})
    assert idx.select("act", "=", "const").tolist() == [0]
    assert idx.select("act", "=", "ipc").tolist() == [0]
    assert idx.select("act", "=", "crpc").tolist() == [3]
    assert idx.header["docs_with_acts_from_sidecar"] == 2


def test_sidecar_overrides_stale_acts_in_the_corpus():
    idx = build(_manifest(), _records({0: ["STALE"]}), verbose=False,
                acts_sidecar={0: ["const"]})
    assert idx.select("act", "=", "stale").tolist() == []
    assert idx.select("act", "=", "const").tolist() == [0]


def test_sidecar_rows_outside_the_manifest_are_ignored():
    idx = build(_manifest(), _records(), verbose=False,
                acts_sidecar={0: ["const"], 99999: ["ipc"]})
    assert idx.select("act", "=", "ipc").tolist() == []
