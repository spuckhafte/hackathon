"""Tests for nyaya.metadata — network-free, built on small inline frames.

Everything here runs off hand-written rows that mirror the shapes actually seen
in the AWS Open Data release (including the awkward ones: supplement SCR
volumes, duplicated paths, a single-name `judge` column).  No test touches
data/ or the network.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nyaya import config
from nyaya.metadata import (
    CASE_TYPE_DOMAIN,
    DISPOSAL_DOMAIN,
    MANIFEST_COLUMNS,
    PARAMETRIC_FIELDS,
    UNIVERSE_COLUMNS,
    canonical_disposal,
    clean_judge,
    derive_case_type,
    judge_keys,
    merge_from_record,
    normalise_neutral,
    normalise_scr,
    parametric_schema,
    sample_manifest,
    split_judges,
    title_key,
    title_tokens,
    to_iso_date,
)


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("23-11-2017", "2017-11-23"),
        ("09-05-2017", "2017-05-09"),
        ("1-1-1950", "1950-01-01"),
        ("", ""),
        ("2017-11-23", ""),      # already ISO -> not the DD-MM-YYYY contract
        ("31-13-2017", ""),      # month 13
        ("garbage", ""),
    ],
)
def test_to_iso_date(raw, expected):
    assert to_iso_date(raw) == expected


# --------------------------------------------------------------------------
# Citations — the exact-lookup keys wave 2 resolves against
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2017 INSC 1136", "2017INSC1136"),
        ("2017INSC1136", "2017INSC1136"),
        ("1950 insc 39", "1950INSC39"),
        ("2017 INSC 0460", "2017INSC460"),   # leading zeros normalised away
        ("", ""),
        ("[2017] 11 S.C.R. 1029", ""),       # an SCR cite is not a neutral cite
    ],
)
def test_normalise_neutral(raw, expected):
    assert normalise_neutral(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[2017] 11 S.C.R. 1029", "SCR:2017:11:1029"),
        ("[1950] 1 S.C.R. 15", "SCR:1950:1:15"),
        ("1962 3 SCR 1", "SCR:1962:3:1"),
        ("[1959] SUPP. 1 S.C.R. 10", "SCR:1959:S1:10"),
        ("[1962] Supp. 3 S.C.R. 1", "SCR:1962:S3:1"),
        ("", ""),
        ("2017 INSC 1136", ""),
    ],
)
def test_normalise_scr(raw, expected):
    assert normalise_scr(raw) == expected


def test_supplement_volume_does_not_collide_with_regular_volume():
    """20% of the corpus lives in supplement volumes; they need their own space."""
    assert normalise_scr("[1959] SUPP. 1 S.C.R. 10") != normalise_scr("[1959] 1 S.C.R. 10")


# --------------------------------------------------------------------------
# Judges — and the bench-size warning this field exists to justify
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HON'BLE MR. JUSTICE D.Y. CHANDRACHUD*", "D.Y. CHANDRACHUD"),
        ("R.K. AGRAWAL", "R.K. AGRAWAL"),
        ("D. Y. CHANDRACHUD", "D.Y. CHANDRACHUD"),
        ("Hon'ble Dr. Justice D.Y. Chandrachud, J.", "D.Y. CHANDRACHUD"),
        ("MEHR CHAND MAHAJAN", "MEHR CHAND MAHAJAN"),
        ("", ""),
    ],
)
def test_clean_judge(raw, expected):
    assert clean_judge(raw) == expected


def test_split_judges():
    assert split_judges("R.K. AGRAWAL, ABHAY MANOHAR SAPRE") == [
        "R.K. AGRAWAL",
        "ABHAY MANOHAR SAPRE",
    ]
    assert split_judges("RANJAN GOGOI") == ["RANJAN GOGOI"]
    assert split_judges("") == []
    assert split_judges("   ") == []
    # honorifics and a trailing 'JJ.' must not become a name
    assert split_judges("MR. JUSTICE A.K. SIKRI and ASHOK BHUSHAN, JJ.") == [
        "A.K. SIKRI",
        "ASHOK BHUSHAN",
    ]


def test_split_judges_is_order_preserving_and_deduped():
    assert split_judges("A.K. SIKRI, A.K. SIKRI, ASHOK BHUSHAN") == [
        "A.K. SIKRI",
        "ASHOK BHUSHAN",
    ]


def test_judge_keys_expose_full_name_and_surname():
    keys = judge_keys(["D.Y. CHANDRACHUD", "A.S. BOPANNA"])
    assert "d.y. chandrachud" in keys          # judge:"d.y. chandrachud"
    assert "chandrachud" in keys               # judge:chandrachud
    assert "bopanna" in keys
    # a bare initial is too short to be a useful key
    assert "d.y." not in keys


# --------------------------------------------------------------------------
# Titles — the discriminative key for wave-2 Jaccard name matching
# --------------------------------------------------------------------------
def test_title_tokens_keeps_everything():
    toks = title_tokens("STATE OF BOMBAY versus NAROTHAMDAS JETHABAI AND ANOTHER")
    assert toks == ["state", "of", "bombay", "versus", "narothamdas",
                    "jethabai", "and", "another"]


def test_title_key_drops_boilerplate():
    key = title_key("GUNWANTLAL GODAWAT  versus UNION OF INDIA & ANOTHER")
    assert key.split() == ["gunwantlal", "godawat"]
    # "union of india" and "another" must not survive, or every Union case
    # looks like every other Union case under Jaccard.
    for junk in ("union", "india", "versus", "another"):
        assert junk not in key.split()


def test_title_key_never_empty_for_all_boilerplate_title():
    """An all-generic title must not collapse to '' — that would match anything."""
    key = title_key("THE STATE OF U.P. versus UNION OF INDIA & ORS.")
    assert key != ""
    assert "state" in key.split()


def test_title_key_discriminates_two_state_cases():
    a = set(title_key("STATE OF MAHARASHTRA versus RAMESH KUMAR").split())
    b = set(title_key("STATE OF KERALA versus SURESH NAIR").split())
    jaccard = len(a & b) / len(a | b)
    assert jaccard < config.CITATION_NAME_THRESHOLD


# --------------------------------------------------------------------------
# Disposal / case type — the two categorical parametric fields
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Appeal(s) allowed", "allowed"),
        ("Case Allowed", "allowed"),
        ("Case Partly allowed", "partly allowed"),
        ("Dismissed", "dismissed"),
        ("Leave Granted & Dismissed", "dismissed"),
        ("Disposed off", "disposed"),
        ("Matter referred to larger bench", "referred to larger bench"),
        ("Directions issued", "directions issued"),
        ("Remitted to Lower Court", "remitted"),
        ("Hearing Adjourned", "adjourned"),
        ("", ""),
    ],
)
def test_canonical_disposal(raw, expected):
    assert canonical_disposal(raw) == expected
    if expected:
        assert expected in DISPOSAL_DOMAIN


@pytest.mark.parametrize(
    "title,cnr,extra,expected",
    [
        ("", "", "CIVIL APPEAL NO. 238 OF 2012", "civil appeal"),
        ("", "", "Criminal Appeal No. 1 of 1999", "criminal appeal"),
        ("", "", "WRIT PETITION (CIVIL) NO. 494 OF 2012", "writ petition"),
        ("", "", "Special Leave Petition (C) No. 9 of 2004", "special leave petition"),
        ("", "", "Review Petition (Crl.) No. 1 of 2017", "review petition"),
        ("", "", "Transfer Petition (C) No. 2 of 2010", "transfer petition"),
        ("", "", "Contempt Petition (C) No. 3 of 2019", "contempt petition"),
        ("IN RE: SUO MOTU WRIT PETITION", "", "", "suo motu"),
        ("A versus B", "ESCR010000382017", "", ""),   # the common, hopeless case
    ],
)
def test_derive_case_type(title, cnr, extra, expected):
    got = derive_case_type(title, cnr, extra)
    assert got == expected
    if got:
        assert got in CASE_TYPE_DOMAIN


def test_slp_beats_bare_petition_and_suo_motu_beats_writ():
    """Pattern order matters: the specific label must win."""
    assert derive_case_type(extra="Special Leave Petition (Civil) No. 1") == \
        "special leave petition"
    assert derive_case_type(title="SUO MOTU WRIT PETITION (C) NO. 1") == "suo motu"


# --------------------------------------------------------------------------
# merge_from_record — the PDF wins, but the metadata is never destroyed
# --------------------------------------------------------------------------
def _meta_row():
    return {
        "node_id": 7,
        "judges": ["R.K. AGRAWAL"],
        "bench_size_meta": 1,
        "case_type": "",
        "neutral_cite": "2017 INSC 1136",
        "scr_cite": "[2017] 11 S.C.R. 1029",
        "decision_date": "2017-11-23",
        "disposal": "dismissed",
    }


def test_merge_prefers_pdf_bench_and_keeps_metadata_value():
    out = merge_from_record(
        _meta_row(),
        {"bench_size": 3, "judges": ["R.K. AGRAWAL", "A.M. SAPRE", "D.Y. CHANDRACHUD"],
         "case_type": "civil appeal", "case_no": "Civil Appeal No. 238 of 2012"},
    )
    assert out["bench_size"] == 3
    assert out["bench_source"] == "pdf"
    assert out["bench_size_meta"] == 1          # metadata value survives
    assert len(out["judges"]) == 3
    assert out["case_type"] == "civil appeal"
    assert out["case_no"] == "Civil Appeal No. 238 of 2012"


def test_merge_falls_back_to_metadata_when_pdf_is_empty():
    out = merge_from_record(_meta_row(), {"bench_size": 0, "judges": [], "case_type": ""})
    assert out["bench_size"] == 1
    assert out["bench_source"] == "meta"
    assert out["judges"] == ["R.K. AGRAWAL"]


def test_merge_derives_case_type_from_pdf_case_number():
    out = merge_from_record(
        _meta_row(),
        {"bench_size": 2, "case_type": "", "case_no": "WRIT PETITION (CIVIL) NO. 494 OF 2012"},
    )
    assert out["case_type"] == "writ petition"


def test_merge_accepts_a_judgment_record_dataclass():
    from nyaya.contracts import JudgmentRecord

    rec = JudgmentRecord(doc_id=0, source_path="2017_11_1029_1035", title="x",
                         bench_size=5, judges=["A", "B", "C", "D", "E"],
                         case_type="criminal appeal")
    out = merge_from_record(_meta_row(), rec)
    assert out["bench_size"] == 5
    assert out["case_type"] == "criminal appeal"


# --------------------------------------------------------------------------
# Stratified sampling — reproducible, capped, year-complete
# --------------------------------------------------------------------------
def _fake_universe(counts):
    """counts: {year: n}.  Mirrors the real universe's column set and ordering."""
    rows = []
    node = 0
    for year in sorted(counts):
        for i in range(counts[year]):
            rows.append(
                {
                    "node_id": node,
                    "year": year,
                    "path": f"{year}_1_{i:05d}",
                    "title": f"CASE {i} versus UNION OF INDIA",
                    "title_key": f"case {i}",
                    "title_tokens": ["case", str(i)],
                    "petitioner": "", "respondent": "",
                    "neutral_cite": f"{year} INSC {i}",
                    "neutral_key": f"{year}INSC{i}",
                    "scr_cite": f"[{year}] 1 S.C.R. {i}",
                    "scr_key": f"SCR:{year}:1:{i}",
                    "decision_date": f"{year}-01-01",
                    "disposal": "dismissed", "disposal_raw": "Dismissed",
                    "judges": ["A B SOMEONE"], "judge_keys": ["a b someone", "someone"],
                    "bench_size_meta": 1, "case_type": "", "cnr": "",
                    "author_judge": "", "available_languages": "ENG",
                }
            )
            node += 1
    return pd.DataFrame(rows)[UNIVERSE_COLUMNS]


def test_sample_respects_the_per_year_cap():
    uni = _fake_universe({1950: 5, 1951: 50, 1952: 7})
    man = sample_manifest(uni, per_year_cap=10, seed=config.SAMPLE_SEED)
    per_year = man.groupby("year").size().to_dict()
    assert per_year == {1950: 5, 1951: 10, 1952: 7}   # small years taken whole


def test_sample_is_reproducible_and_seed_sensitive():
    uni = _fake_universe({1990: 100, 1991: 100})
    a = sample_manifest(uni, per_year_cap=20, seed=358)["node_id"].tolist()
    b = sample_manifest(uni, per_year_cap=20, seed=358)["node_id"].tolist()
    c = sample_manifest(uni, per_year_cap=20, seed=359)["node_id"].tolist()
    assert a == b
    assert a != c


def test_manifest_ids_are_dense_and_node_ids_carry_across():
    uni = _fake_universe({1950: 5, 2000: 40})
    man = sample_manifest(uni, per_year_cap=10)
    assert man["doc_id"].tolist() == list(range(len(man)))
    assert man["node_id"].is_unique
    # the indexed subset must be a subset of the graph nodes
    assert set(man["node_id"]) <= set(uni["node_id"])
    assert list(man.columns) == MANIFEST_COLUMNS


def test_manifest_is_ordered_by_year_then_path():
    uni = _fake_universe({1950: 6, 1951: 6})
    man = sample_manifest(uni, per_year_cap=3)
    keys = list(zip(man["year"], man["path"]))
    assert keys == sorted(keys)


def test_sample_covers_every_year_present():
    uni = _fake_universe({y: 3 for y in range(1950, 1960)})
    man = sample_manifest(uni, per_year_cap=2)
    assert sorted(man["year"].unique()) == list(range(1950, 1960))


# --------------------------------------------------------------------------
# The parametric schema itself
# --------------------------------------------------------------------------
def test_every_query_field_in_the_contract_has_a_schema_entry():
    """contracts.FilterNode documents: year | bench | judge | act | type | disposal."""
    assert set(PARAMETRIC_FIELDS) == {"year", "bench", "judge", "act", "type", "disposal"}


def test_schema_kinds_and_ops_are_coherent():
    for name, f in PARAMETRIC_FIELDS.items():
        assert f.kind in ("numeric", "categorical", "set"), name
        assert f.ops, name
        if f.kind == "numeric":
            # ranges only make sense on an ordered field
            assert ">=" in f.ops and "<=" in f.ops, name
        else:
            # a set/categorical field is membership-only: no ranges
            assert ">=" not in f.ops and "<" not in f.ops, name
        if f.kind == "categorical":
            assert f.domain, name


def test_bench_field_reads_the_merged_column_not_the_metadata_one():
    """Guard the headline caveat: g(d) must never be built on bench_size_meta."""
    assert PARAMETRIC_FIELDS["bench"].column == "bench_size"
    assert PARAMETRIC_FIELDS["bench"].source == "merged"


def test_parametric_schema_is_json_serialisable():
    import json

    blob = json.dumps(parametric_schema())
    assert "bench_size" in blob and "judge_keys" in blob
