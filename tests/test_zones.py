"""Tests for nyaya.zones  (Member 1).

The fixtures below are cut-down but *verbatim* headers from the corpus, one per
typographic era, including their OCR damage (``JJ.J`` for ``JJ.]``, ``IJ.`` for
``JJ.``, ``of2012`` for ``of 2012``).  That is deliberate: the damage is the
reason the code looks the way it does, so the tests have to carry it.
"""
from __future__ import annotations

import pytest

from nyaya import config, zones

# --------------------------------------------------------------------------
# Fixtures: real headers, one per era
# --------------------------------------------------------------------------

MODERN = """[2024] 10 S.C.R. 108 : 2024 INSC 735
Vijay Singh @ Vijay Kr. Sharma
v.
The State of Bihar
(Criminal Appeal No. 1031 of 2015)
25 September 2024
[Bela M. Trivedi and Satish Chandra Sharma,* JJ.]
1.\tThe appellant challenges his conviction under section 302 of the Penal Code.
2.\tThe prosecution led the evidence of four witnesses before the trial court.
3.\tLearned counsel for the appellant submitted that the dying declaration was
not reliable and that the High Court ought to have examined it afresh.
4.\tIn the result, the appeal is allowed and the conviction is set aside.
The captioned appeals stand disposed of in terms of this judgment.
"""

REPORTS_2017 = """[2017] IO S.C.R. 104
PLASTIBLENDS INDIA LIMITED
v.
ADDL. COMMISSIONER OF INCOME TAX, MUMBAI & ANR.
(Civil Appeal No. 238 of2012)
OCTOBER 9, 2017
[A. K. SIKRI AND ASHOK BIIUSHAN, JJ.]
The question is whether depreciation had to be reduced for computing profits.
The Full Bench of the Bombay High Court answered the reference against the
assessee and we see no reason to take a different view of section 80-IA.
The appeals are accordingly dismissed with no order as to costs.
"""

SCAN_1975 = """104
BISHAN SARUP GUPTA ETC. ETC.
v.
UNION OF INDIA & ORS. ETC. ETC.
April 16, 1974
[A. N. RAY C.J., D. G. PALEKAR, K. K. MATHEW, A. ALAGIRISWAMI
AND P. N. BHAGWATI, JJ.J
The above appeals were disposed of by this Court on 16-8-72. The court set
aside the seniority list prepared by the Department and gave directions as to
how the same was to be prepared. The Government then made fresh rules under
the proviso to Article 309 of the Constitution and prepared a new list which
the petitioners now challenge as being contrary to those very directions.
We think it is in accordance with the directions given by this Court.
The writ petitions are dismissed. There shall be no order as to costs.
"""

SCAN_1965_PARENS = """DEO CHAND
v.
SHIV RAM
August 24, 1964
(RAGHUBAR DAYAL, J. R. MUDHOLKAR AND S. M. SIKRI IJ.)
Pending appeal in the Supreme Court the respondents obtained possession of the
suit property in execution of the decree in their favour, and the appellants
now ask that they be put back in possession on such terms as are just.
The appellants would be entitled to recover such mesne profits as the law
allows from the respondents for the period of their dispossession.
"""


# --------------------------------------------------------------------------
# Bench: the authoritative source of bench size
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    (MODERN, ["Bela M. Trivedi", "Satish Chandra Sharma"]),
    (REPORTS_2017, ["A. K. SIKRI", "ASHOK BIIUSHAN"]),
    (SCAN_1975, ["A. N. RAY", "D. G. PALEKAR", "K. K. MATHEW",
                 "A. ALAGIRISWAMI", "P. N. BHAGWATI"]),
    (SCAN_1965_PARENS, ["RAGHUBAR DAYAL", "J. R. MUDHOLKAR", "S. M. SIKRI"]),
])
def test_bench_is_parsed_in_every_era(text, expected):
    h = zones.parse_header(text)
    assert h.bench_source == "bracket"
    assert h.judges == expected


def test_bench_size_is_the_count_of_parsed_judges_not_the_metadata():
    # The whole point of parsing the bench out of the text: Member 2's
    # bench_size_meta is 1 on 37k of 42k rows, and ours must win.
    rec, info = zones.build_record(
        doc_id=1, source_path="1975_1_104_120", year=1975, text=SCAN_1975,
        meta_title="BISHAN SARUP GUPTA versus UNION OF INDIA",
        meta={"bench_size_meta": 1, "judges": ["A. N. RAY"]},
    )
    assert rec.bench_size == 5
    assert info["bench_source"] == "bracket"
    assert len(rec.judges) == 5


def test_metadata_bench_is_used_only_when_no_judge_line_was_found():
    rec, info = zones.build_record(
        doc_id=2, source_path="x", year=1999,
        text="SOME CASE\nv.\nANOTHER\nthe judgment text runs on for a while here.",
        meta_title="SOME CASE versus ANOTHER",
        meta={"bench_size_meta": 3, "judges": ["A", "B", "C"]},
    )
    assert info["bench_source"] == "metadata"
    assert rec.bench_size == 3


def test_author_marker_star_is_not_part_of_the_name():
    judges = zones.parse_judges("Abhay S. Oka* and Augustine George Masih, JJ.")
    assert judges == ["Abhay S. Oka", "Augustine George Masih"]
    assert all("*" not in j for j in judges)


def test_titles_are_not_counted_as_judges():
    judges = zones.parse_judges(
        "Dr Dhananjaya Y Chandrachud, CJI, J B Pardiwala and Manoj Misra, JJ.")
    assert len(judges) == 3
    assert "CJI" not in judges
    assert judges[0] == "Dr Dhananjaya Y Chandrachud"


def test_honorific_is_stripped_but_dr_is_kept():
    assert zones.parse_judges("SHRI HARILAL KANIA CJ., DAS JJ.") == \
        ["HARILAL KANIA", "DAS"]
    assert zones.parse_judges("DR. A.S. ANAND AND B.N. KIRPAL, JJ.")[0] == \
        "DR. A.S. ANAND"


def test_a_citation_in_brackets_is_never_mistaken_for_a_bench():
    text = "[1966] (3) SCR 600\nRoshan Lal v. Union of India [1968] 1 S.C.R. 185\n"
    body, src = zones.find_bench(text)
    assert (body, src) == ("", "none")


def test_a_running_footer_naming_one_judge_is_rejected():
    # "(Sen, J.)" is the 1982 volumes' page footer, not the bench.
    body, src = zones.find_bench("JEEWANTI PANDEY v. K.C. PANDEY (Sen, J.)\n")
    assert src == "none"


def test_a_genuine_single_judge_bench_is_accepted():
    body, src = zones.find_bench("[Dr. Dhananjaya Y. Chandrachud, CJI, "
                                 "J.B. Pardiwala,* J.]\n")
    assert src == "bracket"
    assert len(zones.parse_judges(body)) == 2


def test_coram_line_is_used_when_present():
    body, src = zones.find_bench("CORAM: A. K. Sikri and Ashok Bhushan, JJ.\n")
    assert src == "coram"
    assert len(zones.parse_judges(body)) == 2


# --------------------------------------------------------------------------
# Citations, case number, case type, date
# --------------------------------------------------------------------------


def test_scr_and_neutral_citations_are_read_from_the_header():
    h = zones.parse_header(MODERN)
    assert h.scr_cite == "[2024] 10 S.C.R. 108"
    assert h.neutral_cite == "2024 INSC 735"


def test_ocr_damaged_volume_number_is_repaired():
    # "[2017] IO S.C.R. 104" - the volume 10 is OCR'd as the letters I and O.
    h = zones.parse_header(REPORTS_2017)
    assert h.scr_cite == "[2017] 10 S.C.R. 104"


def test_case_number_survives_the_missing_space_in_of2012():
    h = zones.parse_header(REPORTS_2017)
    assert "Civil Appeal No. 238" in h.case_no
    assert h.case_type == "civil appeal"


@pytest.mark.parametrize("case_no, expected", [
    ("Criminal Appeal No. 1031 of 2015", "criminal appeal"),
    ("Civil Appeal No. 238 of 2012", "civil appeal"),
    ("Writ Petition (Civil) No 609 of 2024", "writ petition"),
    ("Special Leave Petition (Civil) No. 1 of 2020", "special leave petition"),
    ("Review Petition No. 5 of 2019", "review petition"),
    ("Transfer Petition No. 9 of 2018", "transfer petition"),
])
def test_case_type_is_derived_from_the_case_number(case_no, expected):
    text = f"A\nv.\nB\n({case_no})\n1 January 2020\n[X and Y, JJ.]\nbody text here.\n"
    assert zones.parse_header(text).case_type == expected


@pytest.mark.parametrize("line, iso", [
    ("25 September 2024", "2024-09-25"),
    ("OCTOBER 9, 2017", "2017-10-09"),
    ("April 16, 1974", "1974-04-16"),
    ("MARCH 19, 1990", "1990-03-19"),
])
def test_dates_parse_in_both_era_styles(line, iso):
    text = f"A\nv.\nB\n{line}\n[X and Y, JJ.]\nbody.\n"
    assert zones.parse_header(text).date == iso


# --------------------------------------------------------------------------
# Parties
# --------------------------------------------------------------------------


def test_parties_split_on_the_separator_line():
    h = zones.parse_header(MODERN)
    assert h.petitioner == "Vijay Singh @ Vijay Kr. Sharma"
    assert h.respondent == "The State of Bihar"


@pytest.mark.parametrize("raw, expected", [
    ("UNION OF INDIA AND ORS.", "Union of India"),
    ("union of india & anr.", "Union of India"),
    ("STATE OF KERALA", "State of Kerala"),
    ("State of  madhya pradesh ", "State of Madhya Pradesh"),
    ("TARA SINGH ETC. ETC.", "TARA SINGH"),
    ("UNION OF INDIA & ORS. ETC. ETC.", "Union of India"),
])
def test_party_normalisation(raw, expected):
    assert zones.normalise_party(raw) == expected


def test_split_parties_handles_versus_and_vs():
    assert zones.split_parties("A versus B") == ("A", "B")
    assert zones.split_parties("A vs B") == ("A", "B")
    assert zones.split_parties("A v. B") == ("A", "B")


def test_metadata_title_wins_for_the_parties_zone():
    # The scanned header picks up page numbers and marginal letters; the
    # metadata title is clean and already contains " versus ".
    seg = zones.segment(SCAN_1975,
                        meta_title="BISHAN SARUP GUPTA ETC. ETC. versus "
                                   "UNION OF INDIA & ORS. ETC. ETC.")
    assert not seg.header.petitioner.startswith("104")
    assert seg.header.respondent == "Union of India"


# --------------------------------------------------------------------------
# Paragraph numbering
# --------------------------------------------------------------------------


def test_printed_paragraph_numbers_are_kept_as_labels():
    seg = zones.segment(MODERN, meta_title="Vijay Singh versus The State of Bihar")
    assert seg.para_strategy == "printed"
    labels = [p.label for p in seg.paras if p.label]
    assert labels == ["1", "2", "3", "4"]
    assert [p.id for p in seg.paras] == list(range(1, len(seg.paras) + 1))


def test_printed_numbering_requires_a_real_sequence():
    # A stray "(2)" and a quoted "34." must not open paragraphs.
    body = ("of section 14, it seems to me, they would be bringing themselves\n"
            "34. was the number of the section discussed in the cited case\n"
            "2) is not a paragraph number either\n")
    assert zones._split_printed(body.splitlines()) is None


def test_printed_numbering_accepts_a_consecutive_run():
    body = "1. first para\n2. second para\n3. third para\n"
    out = zones._split_printed(body.splitlines())
    assert [lbl for lbl, _ in out] == ["1", "2", "3"]
    assert out[0][1] == "first para"


def test_fallback_segmentation_is_used_when_there_are_no_printed_numbers():
    seg = zones.segment(SCAN_1975, meta_title="A versus B")
    assert seg.para_strategy == "sentence-window"
    assert all(p.label == "" for p in seg.paras)
    assert len(seg.paras) >= 2
    assert [p.id for p in seg.paras] == list(range(1, len(seg.paras) + 1))


def test_fallback_segmentation_never_splits_mid_sentence():
    text = " ".join(f"This is sentence number {i} of the judgment." for i in range(60))
    chunks = zones._split_sentence_window(text, target_words=40)
    assert len(chunks) > 1
    for _, chunk in chunks:
        assert chunk.endswith(".")


def test_paragraph_ids_are_contiguous_and_one_based_in_both_strategies():
    for text in (MODERN, SCAN_1975, REPORTS_2017, SCAN_1965_PARENS):
        seg = zones.segment(text, meta_title="A versus B")
        assert [p.id for p in seg.paras] == list(range(1, len(seg.paras) + 1))


# --------------------------------------------------------------------------
# Order zone
# --------------------------------------------------------------------------


def test_order_zone_holds_the_disposal():
    seg = zones.segment(MODERN, meta_title="Vijay Singh versus The State of Bihar")
    assert "appeal is allowed" in seg.zones["order"].lower()
    assert seg.order_strategy == "cue"
    assert 1 <= seg.n_order_paras <= zones.ORDER_MAX_PARAS


def test_order_zone_does_not_swallow_the_argument_paragraphs():
    seg = zones.segment(MODERN, meta_title="A versus B")
    assert "Learned counsel for the appellant submitted" not in seg.zones["order"]
    assert "Learned counsel for the appellant submitted" in seg.zones["body"]


def test_a_cue_in_the_headnote_does_not_make_the_whole_document_the_order():
    # "Dismissing the appeals, the Court HELD" is printed at the TOP of every
    # report; only the tail of the judgment is eligible to be the order.
    paras = [("", "Dismissing the appeals, the Court HELD: the appeal is dismissed.")] + \
            [("", f"reasoning paragraph {i}") for i in range(20)] + \
            [("", "In the result, the appeal is dismissed with costs.")]
    idx, how = zones.find_order_span(paras)
    assert how == "cue"
    assert idx >= len(paras) - zones.ORDER_MAX_PARAS


def test_order_falls_back_to_the_tail_when_no_cue_fires():
    paras = [("", f"paragraph {i} with no disposal language at all")
             for i in range(10)]
    idx, how = zones.find_order_span(paras)
    assert how == "tail"
    assert idx == 8


def test_split_at_cue_keeps_one_sentence_of_lead_in():
    text = ("The evidence was weak. The witnesses contradicted each other. "
            "I would therefore allow the appeal and dismiss the counter-claim. "
            "Appeal allowed.")
    head, tail = zones.split_at_cue(text)
    # the cue is the reporter's "Appeal allowed."; the court's own operative
    # sentence is the one before it and must land in the order zone.
    assert "I would therefore allow the appeal" in tail
    assert "The witnesses contradicted each other" in head


def test_find_order_span_on_an_empty_document():
    assert zones.find_order_span([]) == (-1, "none")


# --------------------------------------------------------------------------
# Zone set and the statutes hook
# --------------------------------------------------------------------------


def test_segment_produces_exactly_the_configured_zones():
    seg = zones.segment(MODERN, meta_title="A versus B")
    assert set(seg.zones) == set(config.ZONES)


def test_statutes_zone_starts_empty_for_member_three_to_fill():
    seg = zones.segment(MODERN, meta_title="A versus B")
    assert seg.zones["statutes"] == ""


def test_attach_statute_zone_fills_the_synthetic_zone_and_acts():
    rec, _ = zones.build_record(doc_id=5, source_path="p", year=2024,
                               text=MODERN, meta_title="A versus B")
    zones.attach_statute_zone(rec, ["ipc§302", "const§21", "ipc§302"])
    assert rec.statute_tokens == ["ipc§302", "const§21"]   # de-duplicated
    assert rec.zones["statutes"] == "ipc§302 const§21"
    assert rec.acts == ["IPC", "CONST"]


def test_header_region_is_not_duplicated_into_the_body():
    # The bench and the citation live in their own zones; counting them again in
    # body would double-weight them under BM25F.
    seg = zones.segment(MODERN, meta_title="A versus B")
    assert "Bela M. Trivedi" in seg.zones["bench"]
    assert "Bela M. Trivedi" not in seg.zones["body"]
    assert "S.C.R. 108" not in seg.zones["body"]


# --------------------------------------------------------------------------
# build_record: the merge with Member 2's metadata
# --------------------------------------------------------------------------


def test_build_record_round_trips_through_the_contract():
    from nyaya.contracts import JudgmentRecord

    rec, info = zones.build_record(
        doc_id=42, source_path="2024_10_108_125", year=2024, text=MODERN,
        meta_title="VIJAY SINGH versus THE STATE OF BIHAR", n_pages=18,
        meta={"disposal": "allowed", "decision_date": "2024-09-25"},
    )
    again = JudgmentRecord.from_json(rec.to_json())
    assert again.doc_id == 42
    assert again.bench_size == 2
    assert again.n_pages == 18
    assert again.paras[0].id == 1
    assert set(again.zones) == set(config.ZONES)
    assert info["para_strategy"] == "printed"


def test_members_metadata_wins_when_it_is_non_empty():
    rec, _ = zones.build_record(
        doc_id=1, source_path="p", year=2024, text=MODERN, meta_title="A versus B",
        meta={"neutral_cite": "2024 INSC 999", "scr_cite": "",
              "case_type": "criminal appeal", "disposal": "allowed"},
    )
    assert rec.neutral_cite == "2024 INSC 999"      # his value wins
    assert rec.scr_cite == "[2024] 10 S.C.R. 108"   # blank -> ours fills in
    assert rec.disposal == "allowed"


def test_build_record_tolerates_nan_strings_from_pandas():
    rec, _ = zones.build_record(
        doc_id=1, source_path="p", year=2024, text=MODERN, meta_title="A versus B",
        meta={"neutral_cite": "nan", "case_no": "nan"},
    )
    assert rec.neutral_cite == "2024 INSC 735"
    assert "Criminal Appeal No. 1031" in rec.case_no


# --------------------------------------------------------------------------
# Judge-name validation  (tightened after Member 2 reported prose in `judges`)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("glued, expected", [
    ("B.N. KIRPAL ANDS. RAJENDRA BABU, JJ.",
     ["B.N. KIRPAL", "S. RAJENDRA BABU"]),
    ("P. N. BHAGWATI ANDS. MURTAZA FAZAL ALI, JJ.",
     ["P. N. BHAGWATI", "S. MURTAZA FAZAL ALI"]),
    ("K. SUBBA RAO A.ND K. N. WANCHOO, JJ.",
     ["K. SUBBA RAO", "K. N. WANCHOO"]),
])
def test_ocr_glued_conjunction_still_splits_two_judges(glued, expected):
    # "AND S." printed as "ANDS." used to read as one judge, which undercounts
    # the bench - and bench size is exactly what g(d) reads.
    assert zones.parse_judges(glued) == expected


@pytest.mark.parametrize("name", [
    "CHANDRACHUD",          # contains AND but not at a word boundary
    "DR. A.S. ANAND",
    "ANAND",
])
def test_the_glued_and_split_does_not_cut_a_real_name(name):
    assert zones.parse_judges(f"{name}, J.") == [name]


@pytest.mark.parametrize("prose", [
    "1951. March 19. The judgment of the Court was delivered by MuKHERJEA J.",
    "(with him) for the respondent. Rai Brij Raj Krishna J.",
    "a) of the Press Emergency Act. DAs J.",
    'i) stated that "The choice of a close day shall rest with the owner J.',
    "1)-that no return was filed at any stage Bose J.",
])
def test_prose_ending_in_a_judge_title_is_not_a_bench_line(prose):
    assert zones.looks_like_judge_line(prose) is False
    assert zones.find_bench(prose) == ("", "none")


@pytest.mark.parametrize("body", [
    "A. K. SIKRI AND ASHOK BHUSHAN, JJ.",
    "Bela M. Trivedi and Satish Chandra Sharma,* JJ.",
    "SHRI HARILAL KANIA C.J., SAIYID FAZL ALI, PATANJALI SASTRI, JJ.",
])
def test_a_real_bench_line_still_passes_the_prose_filter(body):
    assert zones.looks_like_judge_line(body) is True
    assert len(zones.parse_judges(body)) >= 2


def test_ocr_digit_substitutions_do_not_lose_a_real_judge():
    # The 1950s scans print l as 1 and O as 0 constantly.  Rejecting any name
    # with a digit threw away 1,826 genuine judge entries, so only a *run* of
    # digits (a year, a section number) disqualifies a name.
    for name in ("SHR1 HARILAL KANg", "SA1Ym FAzL Au", "HARILAL KANIA c . .1"):
        assert zones.is_plausible_name(name) is True
    for junk in ("1980 v. chandrachud", "Suit No. 7 of 1950"):
        assert zones.is_plausible_name(junk) is False


def test_coram_line_is_validated_like_a_bracketed_one():
    # The CORAM branch had no validation at all, which is where the worst of the
    # measured noise came from.
    assert zones.find_bench("CORAM: the judgment of the Court was delivered "
                            "by MAHAJAN J.") == ("", "none")
    body, src = zones.find_bench("CORAM: A. K. Sikri and Ashok Bhushan, JJ.")
    assert src == "coram" and len(zones.parse_judges(body)) == 2


def test_is_plausible_name_rejects_prose_and_accepts_names():
    assert zones.is_plausible_name("for the respondent") is False
    assert zones.is_plausible_name("the judgment of the court") is False
    assert zones.is_plausible_name("A. K. SIKRI") is True
    assert zones.is_plausible_name("Nongmeikapam Kotiswar Singh") is True


def test_reparse_bench_downgrades_prose_to_unknown(tmp_path):
    """A prose bench becomes 'unknown', not a judge called 'for the respondent'."""
    import json

    src = tmp_path / "judgments.jsonl"
    rows = [
        {"doc_id": 1, "bench_size": 2, "judges": ["A. K. SIKRI", "ASHOK BHUSHAN"],
         "zones": {"bench": "A. K. SIKRI AND ASHOK BHUSHAN, JJ."}},
        {"doc_id": 2, "bench_size": 2,
         "judges": ["for the respondent", "Rai Brij Raj Krishna"],
         "zones": {"bench": "with him) for the respondent. Rai Brij Raj Krishna J."}},
        {"doc_id": 3, "bench_size": 3, "judges": ["X", "Y", "Z"],
         "zones": {"bench": ""}},          # came from the metadata fallback
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    stats = zones.reparse_bench(src)
    assert stats["bench_kept"] == 1
    assert stats["bench_cleared"] == 1
    assert stats["bench_from_metadata"] == 1

    out = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()]
    assert out[0]["judges"] == ["A. K. SIKRI", "ASHOK BHUSHAN"]
    assert out[1]["judges"] == [] and out[1]["bench_size"] == 0
    assert out[1]["zones"]["bench"] == ""
    # the metadata-sourced record is left exactly as it was
    assert out[2]["judges"] == ["X", "Y", "Z"] and out[2]["bench_size"] == 3


def test_a_judges_initial_is_not_the_end_of_the_bench_line():
    """The nine-judge Puttaswamy bench opens 'JAGDISH SINGH KHEHAR, CJI,
    J. CHELAMESWAR, ...' and the second judge's initial is 'J.'.  Treating that
    as the line terminator read the largest bench in the demo as one judge."""
    head = "\n".join([
        "[2017] 10 S.C.R. 569",
        "JUSTICE K S PUTTASWAMY (RETD.),",
        "AND ANR.",
        "v.",
        "UNION OF INDIA AND ORS.",
        "(Writ Petition (Civil) No. 494 of2012)",
        "AUGUST 24, 2017",
        "[JAGDISH SINGH KHEHAR, CJI, J. CHELAMESWAR,",
        "S. A. BOBDE, R. K. AGRAWAL, R. F .. NARIMAN,",
        "ABHAY MANOHAR SAPRE, DR. D. Y. CHANDRACHUD,",
        "SANJAY KISHAN KAUL AND S. ABDUL NAZEER, JJ.]",
        "Constitution of India - Art.21 - Right to privacy",
    ])
    h = zones.parse_header(head)
    assert len(h.judges) == 9, h.judges
    assert h.judges[0] == "JAGDISH SINGH KHEHAR"
    assert h.judges[-1] == "S. ABDUL NAZEER"


def test_a_genuine_single_judge_line_still_terminates_at_its_title():
    for line, n in (("[Bela M. Trivedi, J.]", 1),
                    ("[A. K. SIKRI, J.]", 1),
                    ("[Dr. Dhananjaya Y. Chandrachud, CJI, J.B. Pardiwala,* J.]", 2)):
        body, src = zones.find_bench(line)
        assert src == "bracket", line
        assert len(zones.parse_judges(body)) == n, (line, body)
