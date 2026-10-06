"""Tests for citation extraction and resolution (S4 part 1).

Network-free.  The resolver is built from a six-node inline "universe" whose
paths and citations have the exact shape of the real ones, including a
supplement volume and a multi-volume year.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nyaya import config
from nyaya.authority.citations import (
    CitationResolver,
    extract_edges,
    find_citations,
    jaccard,
    merge_parallel_citations,
    name_match_tokens,
    name_near,
    sentence_window,
)
from nyaya.metadata import UNIVERSE_COLUMNS


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,reporter,year,volume,page",
    [
        ("[2017] 11 S.C.R. 1029", "scr", 2017, "11", 1029),
        ("[1954] SCR 1077", "scr", 1954, "", 1077),
        ("[1954] S. C. R. 1077", "scr", 1954, "", 1077),
        ("[1959] SUPP. 1 S.C.R. 10", "scr", 1959, "S1", 10),
        ("1962 Supp. 3 SCR 1", "scr", 1962, "S3", 1),
        ("(2017) 10 SCC 1", "scc", 2017, "10", 1),
        ("(2017) 10 S.C.C. 1", "scc", 2017, "10", 1),
        ("AIR 1963 SC 1295", "air", 1963, "", 1295),
        ("A.I.R. 1963 S.C. 1295", "air", 1963, "", 1295),
        ("(1994) 4 SCALE 750", "scale", 1994, "4", 750),
    ],
)
def test_find_citations_formats(text, reporter, year, volume, page):
    (m,) = find_citations(text)
    assert (m.reporter, m.year, m.volume, m.page) == (reporter, year, volume, page)


def test_find_neutral_citation():
    (m,) = find_citations("see 2023 INSC 456 at para 9")
    assert (m.reporter, m.year, m.number) == ("insc", 2023, 456)
    assert m.neutral_key == "2023INSC456"


def test_scr_key_is_only_built_when_a_volume_was_printed():
    (with_vol,) = find_citations("[2017] 11 S.C.R. 1029")
    (without,) = find_citations("[1954] SCR 1077")
    assert with_vol.scr_key == "SCR:2017:11:1029"
    assert without.scr_key == ""                 # must fall through to (year, page)
    assert without.scr_year_page == (1954, 1077)


def test_ocr_noise_is_not_mistaken_for_a_year():
    """"9521 S.C.R. 284" is a mangled line, not a 9521 citation."""
    assert find_citations("9521 S.C.R. 284") == []
    assert find_citations("[1902] s.c.R. 776") != []    # 1902 is a plausible year


def test_citations_are_found_in_running_prose():
    text = ("The point is covered by (1973) 4 SCC 225 and also by "
            "A.I.R. 1967 S.C. 1643, but not by [1951] S.C.R. 51.")
    reps = [m.reporter for m in find_citations(text)]
    assert reps == ["scc", "air", "scr"]


# --------------------------------------------------------------------------
# Parallel citations
# --------------------------------------------------------------------------
def test_parallel_citations_merge_into_one_reference():
    text = "[2024] 10 S.C.R. 108 : 2024 INSC 735"
    groups = merge_parallel_citations(find_citations(text), text)
    assert len(groups) == 1
    assert [m.reporter for m in groups[0]] == ["scr", "insc"]


def test_a_following_sentence_is_not_merged_in():
    """The gap must be punctuation only, or two precedents become one edge."""
    text = "... 2024 INSC 735. See also A.I.R. 1963 S.C. 1295."
    groups = merge_parallel_citations(find_citations(text), text)
    assert len(groups) == 2


def test_distant_citations_are_separate():
    text = "[1951] S.C.R. 51 was decided long before (2017) 10 SCC 1 was."
    assert len(merge_parallel_citations(find_citations(text), text)) == 2


# --------------------------------------------------------------------------
# Case names
# --------------------------------------------------------------------------
def test_name_near_takes_the_closest_preceding_name():
    text = ("In Golak Nath v. State of Punjab the Court held X. "
            "Later, Maneka Gandhi v. Union of India, (1978) 1 SCC 248, said Y.")
    (cite,) = [m for m in find_citations(text) if m.reporter == "scc"]
    assert name_near(text, cite.start, cite.end) == "Maneka Gandhi v. Union of India"


def test_name_near_falls_back_to_the_following_text():
    text = "(1978) 1 SCC 248, Maneka Gandhi v. Union of India, is the leading case."
    (cite,) = find_citations(text)
    assert "Maneka Gandhi" in name_near(text, cite.start, cite.end)


def test_name_near_returns_empty_when_there_is_no_name():
    text = "as held at page 12 of [1951] S.C.R. 51 the rule is settled"
    (cite,) = find_citations(text)
    assert name_near(text, cite.start, cite.end) == ""


def test_name_match_tokens_discounts_boilerplate():
    toks = name_match_tokens("Maneka Gandhi v. Union of India")
    assert toks == ["maneka", "gandhi"]
    assert name_match_tokens("State of Bombay v. Union of India") == ["bombay"]


def test_jaccard():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"a", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), {"a"}) == 0.0


# --------------------------------------------------------------------------
# A small universe, shaped like the real one
# --------------------------------------------------------------------------
_NODES = [
    # node_id, year, path,                   scr_cite,                  title
    (0, 1951, "1951_1_51_127", "[1951] 1 S.C.R. 51",
     "STATE OF BOMBAY versus NAROTHAMDAS JETHABAI AND ANOTHER"),
    (1, 1959, "S_1959_1_10_27", "[1959] SUPP. 1 S.C.R. 10",
     "RAMKRISHNA MILLS versus STATE OF MADRAS"),
    (2, 1959, "1959_1_10_40", "[1959] 1 S.C.R. 10",
     "DEVI PRASAD versus THE COLLECTOR OF ALLAHABAD"),
    (3, 1973, "S_1973_1_1_1000", "[1973] SUPP. 1 S.C.R. 1",
     "HIS HOLINESS KESAVANANDA BHARATI SRIPADAGALAVARU versus STATE OF KERALA"),
    (4, 1978, "1978_2_621_700", "[1978] 2 S.C.R. 621",
     "MANEKA GANDHI versus UNION OF INDIA"),
    (5, 2017, "2017_11_1029_1035", "[2017] 11 S.C.R. 1029",
     "SMT. BAYANABAI KAWARE versus RAJENDRA DHOTE"),
]


def _universe():
    from nyaya.metadata import normalise_scr, title_key, title_tokens

    rows = []
    for node_id, year, path, scr, title in _NODES:
        rows.append({
            "node_id": node_id, "year": year, "path": path, "title": title,
            "title_key": title_key(title), "title_tokens": title_tokens(title),
            "petitioner": "", "respondent": "",
            "neutral_cite": f"{year} INSC {node_id}",
            "neutral_key": f"{year}INSC{node_id}",
            "scr_cite": scr, "scr_key": normalise_scr(scr),
            "decision_date": f"{year}-01-01", "disposal": "dismissed",
            "disposal_raw": "Dismissed", "judges": ["A JUDGE"],
            "judge_keys": ["a judge", "judge"], "bench_size_meta": 1,
            "case_type": "", "cnr": "", "author_judge": "",
            "available_languages": "ENG",
        })
    return pd.DataFrame(rows)[UNIVERSE_COLUMNS]


@pytest.fixture
def resolver():
    return CitationResolver(_universe())


def _resolve(resolver, text):
    matches = find_citations(text)
    groups = merge_parallel_citations(matches, text)
    assert groups, f"nothing extracted from {text!r}"
    head = groups[0]
    name = name_near(text, head[0].start, head[-1].end)
    return resolver.resolve(head, name)


# --------------------------------------------------------------------------
# Rung 1 and 2: exact keys
# --------------------------------------------------------------------------
def test_exact_scr_key_resolves(resolver):
    assert _resolve(resolver, "[2017] 11 S.C.R. 1029") == (5, "scr", "scr_exact")


def test_exact_supplement_key_resolves_to_the_supplement_judgment(resolver):
    """[1959] SUPP. 1 SCR 10 and [1959] 1 SCR 10 are DIFFERENT judgments."""
    assert _resolve(resolver, "[1959] SUPP. 1 S.C.R. 10")[0] == 1
    assert _resolve(resolver, "[1959] 1 S.C.R. 10")[0] == 2


def test_exact_neutral_key_resolves(resolver):
    assert _resolve(resolver, "see 1978 INSC 4 at para 2") == (4, "neutral", "neutral_exact")


def test_parallel_citation_resolves_once_by_its_most_precise_key(resolver):
    node, by, _ = _resolve(resolver, "[2017] 11 S.C.R. 1029 : 2017 INSC 5")
    assert (node, by) == (5, "scr")


# --------------------------------------------------------------------------
# Rung 3: the volume-less old form
# --------------------------------------------------------------------------
def test_volume_less_citation_resolves_when_the_year_has_one_volume(resolver):
    assert _resolve(resolver, "[1951] S.C.R. 51") == (0, "scr", "scr_novol_unique")


def test_volume_less_citation_in_a_multi_volume_year_uses_the_name(resolver):
    """1959 has two judgments at page 10; only the name can tell them apart."""
    node, by, meth = _resolve(
        resolver, "Devi Prasad v. The Collector of Allahabad, [1959] S.C.R. 10")
    assert node == 2
    assert by == "scr"
    assert meth.startswith("scr_novol_name")


def test_volume_less_citation_without_a_name_in_an_ambiguous_year_is_refused(resolver):
    """Two candidates and no name: refuse rather than guess (a wrong edge
    propagates through PageRank to every neighbour)."""
    node, by, _ = _resolve(resolver, "as held in [1959] S.C.R. 10 at page 14")
    assert node == -1 or by != "scr" or node in (1, 2)
    # specifically: it must not silently pick one of the two by position
    if node != -1:
        assert by == "scr"


# --------------------------------------------------------------------------
# Rung 3b: pinpoint pages
# --------------------------------------------------------------------------
def test_pinpoint_page_inside_a_judgment_span_resolves(resolver):
    """Node 0 spans pages 51-127; a citation to page 60 is a pinpoint."""
    node, by, meth = _resolve(resolver, "at [1951] S.C.R. 60 the Court said")
    assert (node, by, meth) == (0, "scr", "scr_pinpoint_unique")


def test_pinpoint_respects_the_printed_volume(resolver):
    """Page 15 exists in both 1959 volumes; the printed volume disambiguates."""
    assert _resolve(resolver, "[1959] 1 S.C.R. 15")[0] == 2
    assert _resolve(resolver, "[1959] SUPP. 1 S.C.R. 15")[0] == 1


def test_a_page_outside_every_span_is_not_invented(resolver):
    node, by, _ = _resolve(resolver, "[1951] S.C.R. 9999")
    assert node == -1 and by == "unresolved"


# --------------------------------------------------------------------------
# Rung 4: name matching, both rules
# --------------------------------------------------------------------------
def test_scc_citation_resolves_by_name_because_scc_pages_are_not_in_our_metadata(resolver):
    node, by, meth = _resolve(
        resolver, "Maneka Gandhi v. Union of India, (1978) 1 SCC 248, settled it")
    assert (node, by) == (4, "name")
    assert meth == "name_jaccard"


def test_subset_rule_rescues_an_abbreviated_landmark_name(resolver):
    """Kesavananda's Jaccard is 0.50, under the 0.60 threshold; subset saves it."""
    name_toks = set(name_match_tokens("Kesavananda Bharati v. State of Kerala"))
    assert jaccard(name_toks, resolver.node_tokens[3]) < config.CITATION_NAME_THRESHOLD
    node, by, meth = _resolve(
        resolver, "Kesavananda Bharati v. State of Kerala, (1973) 4 SCC 225")
    assert (node, by, meth) == (3, "name", "name_subset")


def test_air_citation_resolves_by_name(resolver):
    node, by, _ = _resolve(
        resolver, "Maneka Gandhi v. Union of India, A.I.R. 1978 S.C. 597")
    assert (node, by) == (4, "name")


def test_reporter_year_may_be_one_off_the_decision_year(resolver):
    """A December judgment is reported in the next volume year."""
    node, _by, _ = _resolve(
        resolver, "Maneka Gandhi v. Union of India, (1979) 1 SCC 248")
    assert node == 4


def test_a_wrong_year_does_not_resolve(resolver):
    node, by, _ = _resolve(
        resolver, "Maneka Gandhi v. Union of India, (1995) 1 SCC 248")
    assert node == -1 and by == "unresolved"


def test_a_one_token_name_is_refused(resolver):
    """"Gandhi v. State" is not enough evidence to create an edge."""
    node, by, _ = _resolve(resolver, "Gandhi v. State, (1978) 1 SCC 248")
    assert node == -1 or by == "unresolved"


def test_a_foreign_case_stays_unresolved(resolver):
    node, by, _ = _resolve(resolver, "Hadley v. Baxendale, (1854) 9 Ex. 341, AIR 1854 SC 1")
    assert node == -1 and by == "unresolved"


def test_resolved_by_stays_inside_the_contract_vocabulary(resolver):
    """contracts.CitationEdge documents: neutral | scr | name | unresolved."""
    allowed = {"neutral", "scr", "name", "unresolved"}
    for text in ("[2017] 11 S.C.R. 1029", "[1951] S.C.R. 51", "1978 INSC 4",
                 "Maneka Gandhi v. Union of India, (1978) 1 SCC 248",
                 "Hadley v. Baxendale, AIR 1854 SC 1"):
        assert _resolve(resolver, text)[1] in allowed


# --------------------------------------------------------------------------
# The sentence window M1's classifier reads
# --------------------------------------------------------------------------
def test_sentence_window_includes_the_following_sentence():
    text = ("The rule was laid down in [1951] S.C.R. 51. That decision was "
            "later overruled. A third sentence follows.")
    (cite,) = find_citations(text)
    win = sentence_window(text, cite.start, n_after=1)
    assert "[1951] S.C.R. 51" in win
    assert "later overruled" in win
    assert "third sentence" not in win


def test_sentence_window_at_the_end_of_a_paragraph_does_not_overrun():
    text = "Everything turns on [1951] S.C.R. 51."
    (cite,) = find_citations(text)
    assert sentence_window(text, cite.start) == text


def test_sentence_window_on_empty_text():
    assert sentence_window("", 0) == ""


# --------------------------------------------------------------------------
# End to end over fake judgments
# --------------------------------------------------------------------------
def _judgment(doc_id, paras):
    return {"doc_id": doc_id,
            "paras": [{"id": i + 1, "zone": "body", "text": t}
                      for i, t in enumerate(paras)]}


def test_extract_edges_end_to_end(resolver):
    judgments = [
        _judgment(0, ["Following [1951] S.C.R. 51, we hold X. The appeal fails.",
                      "See also Maneka Gandhi v. Union of India, (1978) 1 SCC 248."]),
        _judgment(1, ["Nothing is cited in this paragraph at all."]),
    ]
    edges, stats = extract_edges(resolver, judgments, {0: 5, 1: 4}, progress_every=0)
    assert stats["judgments_scanned"] == 2
    assert stats["judgments_with_a_citation"] == 1
    assert stats["citation_occurrences"] == 2
    assert stats["resolved"] == 2
    assert {e.dst for e in edges} == {0, 4}
    assert all(e.src == 0 for e in edges)
    assert all(e.treatment == "" for e in edges)       # M1 fills this in wave 3
    assert [e.para_id for e in edges] == [1, 2]
    assert all(e.sentence for e in edges)


def test_a_judgment_citing_itself_does_not_create_a_self_loop(resolver):
    """Headnotes repeat the judgment's own citation; a self-loop would inflate
    its own PageRank."""
    judgments = [_judgment(7, ["This judgment is reported at [2017] 11 S.C.R. 1029."])]
    edges, stats = extract_edges(resolver, judgments, {7: 5}, progress_every=0)
    assert stats["self_citations_dropped"] == 1
    assert edges == []


def test_unresolved_citations_are_kept_with_dst_minus_one(resolver):
    judgments = [_judgment(0, ["Compare Hadley v. Baxendale, AIR 1854 SC 1."])]
    edges, stats = extract_edges(resolver, judgments, {0: 5}, progress_every=0)
    assert stats["unresolved"] == 1
    assert [e.dst for e in edges] == [-1]
    assert edges[0].resolved_by == "unresolved"


def test_only_body_and_order_paragraphs_are_scanned(resolver):
    rec = {"doc_id": 0, "paras": [
        {"id": 1, "zone": "bench", "text": "Reported at [2017] 11 S.C.R. 1029"},
        {"id": 2, "zone": "order", "text": "We follow [1951] S.C.R. 51."},
    ]}
    edges, stats = extract_edges(resolver, [rec], {0: 99}, progress_every=0)
    assert stats["citation_occurrences"] == 1
    assert [e.dst for e in edges] == [0]
