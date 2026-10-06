"""Tests for nyaya.ingest  (Member 1).

No network and no PDFs: every fixture is a few lines of text built inline, so
the suite runs in milliseconds and still exercises the behaviour that actually
breaks - running headers, the SCR marginal letter column, hyphenation, the
language gate and the extract log.
"""
from __future__ import annotations

import csv

import pytest

from nyaya import config, ingest

# --------------------------------------------------------------------------
# URL / path layout: the contract with the public bucket
# --------------------------------------------------------------------------


def test_pdf_key_and_url_match_the_bucket_layout():
    assert ingest.pdf_key(2017, "2017_11_1029_1035") == \
        "data/pdf/year=2017/english/2017_11_1029_1035_EN.pdf"
    url = ingest.pdf_url(2017, "2017_11_1029_1035")
    assert url.startswith(config.BUCKET_URL)
    assert url.endswith("_EN.pdf")
    assert "//data" not in url.replace("https://", "")   # no double slash


def test_pdf_local_is_grouped_by_year():
    p = ingest.pdf_local(1950, "1950_1_15_25")
    assert p.parent.name == "1950"
    assert p.name == "1950_1_15_25_EN.pdf"
    assert p.parent.parent == config.RAW_PDF


# --------------------------------------------------------------------------
# Unicode normalisation
# --------------------------------------------------------------------------


def test_normalise_unicode_folds_quotes_dashes_ligatures_and_replacement_char():
    raw = "“the ofﬁce” – ‘x’�…"
    out = ingest.normalise_unicode(raw)
    assert out == '"the office" - \'x\'...'
    assert "�" not in out


def test_normalise_unicode_strips_the_control_char_in_scr_running_heads():
    # The Digital SCR PDFs embed U+0008 in the header; two copies of the same
    # header must compare equal afterwards or de-duplication silently fails.
    a = ingest.normalise_unicode("[2024] 10 S.C.R. \x08")
    b = ingest.normalise_unicode("[2024] 10 S.C.R. ")
    assert a.strip() == b.strip() == "[2024] 10 S.C.R."


# --------------------------------------------------------------------------
# Running headers and footers
# --------------------------------------------------------------------------


def _page(header: str, body: str, page_no: int) -> str:
    return f"{header}\n{body}\n{page_no}\n"


def test_repeated_lines_finds_the_running_header_but_not_a_body_line():
    pages = [
        _page("[2017] 10 S.C.R. 104", "unique body line one", 104),
        _page("[2017] 10 S.C.R. 104", "unique body line two", 105),
        _page("[2017] 10 S.C.R. 104", "unique body line three", 106),
        _page("[2017] 10 S.C.R. 104", "unique body line four", 107),
    ]
    heads = ingest.repeated_lines(pages)
    assert "[2017] 10 S.C.R. 104" in heads
    assert "unique body line one" not in heads


def test_clean_pages_removes_running_header_and_page_numbers():
    pages = [
        _page("PLASTIBLENDS INDIA LIMITED", "first page of the judgment", 104),
        _page("PLASTIBLENDS INDIA LIMITED", "second page of the judgment", 105),
        _page("PLASTIBLENDS INDIA LIMITED", "third page of the judgment", 106),
    ]
    out = ingest.clean_pages(pages)
    # kept once, on page 1, because the first band of page 1 is protected:
    # that is where zones.py reads the citation and the bench from.
    assert out.count("PLASTIBLENDS INDIA LIMITED") == 1
    assert "\n105\n" not in out and not out.endswith("106")
    for n in ("first", "second", "third"):
        assert f"{n} page of the judgment" in out


def test_first_page_header_survives_so_the_citation_is_not_lost():
    header = "[2024] 10 S.C.R. 150 : 2024 INSC 749"
    pages = [_page(header, "body one", 150)] + \
            [_page(header, f"body {i}", 150 + i) for i in range(1, 5)]
    out = ingest.clean_pages(pages)
    assert out.startswith(header)
    assert out.count(header) == 1


# --------------------------------------------------------------------------
# The SCR marginal letter column
# --------------------------------------------------------------------------


def test_strip_marginal_letters_drops_the_a_to_h_column():
    lines = ["A", "real text here", "B", "more text", "C", "D", "E",
             "F", "G", "H", "last line"]
    out = ingest.strip_marginal_letters(lines)
    assert out == ["real text here", "more text", "last line"]


def test_strip_marginal_letters_leaves_a_lone_letter_alone_when_there_is_no_column():
    # Guarded on purpose: fewer than 8 hits means this is not a margin column
    # and a genuine single-letter line (an enumerated head "A") must survive.
    lines = ["A", "some text", "more text"]
    assert ingest.strip_marginal_letters(lines) == lines


# --------------------------------------------------------------------------
# Hyphenation
# --------------------------------------------------------------------------


def test_clean_pages_repairs_end_of_line_hyphenation():
    out = ingest.clean_pages(["the convic-\ntion was set aside"])
    assert "conviction was set aside" in out
    assert "convic-" not in out


def test_clean_pages_keeps_a_real_hyphenated_compound():
    out = ingest.clean_pages(["an income-tax reference was made"])
    assert "income-tax" in out


# --------------------------------------------------------------------------
# Language gate
# --------------------------------------------------------------------------


ENGLISH_SAMPLE = (
    "The appellant was convicted under section 302 of the Indian Penal Code "
    "and the High Court confirmed the conviction. In our opinion the evidence "
    "of the prosecution witnesses is not such as can be acted upon, and we are "
    "therefore of the view that the appeal must be allowed and the conviction "
    "set aside. The appeal is accordingly allowed with no order as to costs."
)


def test_english_score_is_high_for_a_judgment():
    ascii_share, stop_rate = ingest.english_score(ENGLISH_SAMPLE)
    assert ascii_share == pytest.approx(1.0)
    # measured on the real corpus a healthy judgment lands at 0.40-0.48
    assert 0.30 <= stop_rate <= 0.60
    assert ascii_share >= ingest.ASCII_SHARE_MIN
    assert stop_rate >= ingest.STOPWORD_RATE_MIN


def test_english_score_rejects_devanagari():
    hindi = "यह अपीलकर्ता की अपील है और न्यायालय ने इसे खारिज कर दिया है।" * 5
    ascii_share, stop_rate = ingest.english_score(hindi)
    assert ascii_share < ingest.ASCII_SHARE_MIN


def test_english_score_rejects_a_word_salad_with_no_function_words():
    salad = " ".join(["xyzzy", "frobnicate", "quux", "blorp"] * 60)
    _, stop_rate = ingest.english_score(salad)
    assert stop_rate < ingest.STOPWORD_RATE_MIN


def test_english_score_is_zero_on_empty_text():
    assert ingest.english_score("") == (0.0, 0.0)


def test_word_count_counts_words_not_punctuation():
    assert ingest.word_count("the appeal is, therefore, allowed.") == 5


# --------------------------------------------------------------------------
# extract_one: statuses and the audit log
# --------------------------------------------------------------------------


def test_extract_one_reports_missing_file_without_touching_the_network(tmp_path,
                                                                      monkeypatch):
    monkeypatch.setattr(config, "RAW_PDF", tmp_path)
    monkeypatch.setattr(ingest.config, "RAW_PDF", tmp_path)
    ex = ingest.extract_one((7, 1999, "1999_2_834_856"))
    assert ex.status == "missing_file"
    assert ex.doc_id == 7 and ex.year == 1999
    assert ex.text == ""


def test_extract_one_reports_a_pdf_error_for_a_file_that_is_not_a_pdf(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(ingest.config, "RAW_PDF", tmp_path)
    fp = tmp_path / "2017" / "2017_1_1_2_EN.pdf"
    fp.parent.mkdir(parents=True)
    fp.write_bytes(b"this is not a pdf at all")
    ex = ingest.extract_one((1, 2017, "2017_1_1_2"))
    assert ex.status == "drop_pdf_error"
    assert ex.reason


def test_extraction_log_row_has_exactly_the_contracted_columns():
    ex = ingest.Extraction(3, "p", 2001, "drop_short", 2, 100, "100 words")
    assert list(ex.log_row().keys()) == ingest.LOG_FIELDS
    assert ingest.LOG_FIELDS == ["doc_id", "path", "year", "status",
                                 "n_pages", "n_words", "reason"]


def test_iter_extractions_logs_every_attempt_including_the_failures(tmp_path,
                                                                   monkeypatch):
    monkeypatch.setattr(ingest.config, "RAW_PDF", tmp_path)   # nothing on disk
    log = tmp_path / "extract_log.csv"
    tasks = [(0, 1950, "a"), (1, 1951, "b"), (2, 1952, "c")]
    got = list(ingest.iter_extractions(tasks, processes=1, log_path=log))
    assert [g.status for g in got] == ["missing_file"] * 3
    with log.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3                     # coverage is reported honestly
    assert {r["status"] for r in rows} == {"missing_file"}


# --------------------------------------------------------------------------
# No silent fallback manifest
# --------------------------------------------------------------------------


def test_load_manifest_refuses_to_invent_doc_ids(tmp_path, monkeypatch):
    missing = tmp_path / "manifest.parquet"
    monkeypatch.setattr(ingest.config, "MANIFEST_PARQUET", missing)
    with pytest.raises(SystemExit) as exc:
        ingest.load_manifest()
    assert "doc_id" in str(exc.value)
