"""Tests for proximity scoring and snippets  (Member 1, wave 3).

Both modules are tested against a real index built from inline documents, so the
positions, paragraph ids and zone boundaries are the production ones.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from nyaya import config
from nyaya.index import positional as P
from nyaya.rank import proximity as RP
from nyaya.rank import snippets as SN

DOCS = [
    {   # terms adjacent -> narrow window, high proximity
        "title": "RAM SINGH versus STATE OF BIHAR",
        "bench": "A. K. Sikri and Ashok Bhushan, JJ.",
        "paras": [
            ("body", 1, "The right to privacy and the surveillance of citizens "
                        "were considered together by the Constitution Bench."),
            ("order", 2, "The appeal is allowed."),
        ],
    },
    {   # same terms, far apart, different paragraphs -> discounted
        "title": "MOHAN LAL versus UNION OF INDIA",
        "bench": "R. F. Nariman, J.",
        "paras": [
            ("body", 1, "The question of privacy arises at the threshold of this "
                        "appeal and was argued at length by learned counsel for "
                        "the petitioner over several days of hearing."),
            ("body", 2, "A quite separate question about the acquisition of land "
                        "was also raised and need not detain us here at all."),
            ("body", 3, "Only much later did the surveillance of the premises "
                        "become relevant to the controversy before this Court."),
            ("order", 4, "The appeal is dismissed."),
        ],
    },
    {   # only one of the two terms
        "title": "KAMLA DEVI versus STATE OF KERALA",
        "bench": "Bela M. Trivedi, J.",
        "paras": [
            ("body", 1, "Bail was refused by the High Court for reasons that "
                        "were recorded at length in the impugned order."),
            ("order", 2, "Accordingly the appeal is allowed and bail is granted."),
        ],
    },
    {   # a statute, written as the court writes it
        "title": "SURESH KUMAR versus STATE OF GOA",
        "bench": "S. N. Variava and S. H. Kapadia, JJ.",
        "paras": [
            ("body", 1, "The accused was convicted of murder under Section 302 "
                        "of the Indian Penal Code by the trial court."),
            ("order", 2, "The appeal is partly allowed and the sentence reduced."),
        ],
    },
]


def _record(doc_id: int, spec: dict) -> dict:
    petitioner, respondent = spec["title"].split(" versus ")
    return {
        "doc_id": doc_id, "source_path": f"t{doc_id}", "title": spec["title"],
        "petitioner": petitioner, "respondent": respondent,
        "year": 2000 + doc_id, "judges": [], "bench_size": 2,
        "zones": {
            "title": spec["title"], "parties": f"{petitioner} {respondent}",
            "bench": spec["bench"], "statutes": "",
            "order": "\n\n".join(t for (z, _p, t) in spec["paras"] if z == "order"),
            "body": "\n\n".join(t for (z, _p, t) in spec["paras"] if z == "body"),
        },
        "paras": [{"id": pid, "zone": z, "text": t, "label": str(pid)}
                  for (z, pid, t) in spec["paras"]],
        "n_tokens": 0, "n_pages": 1,
    }


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rank")
    src = tmp / "judgments.jsonl"
    src.write_text("\n".join(json.dumps(_record(i, d)) for i, d in enumerate(DOCS))
                   + "\n", encoding="utf-8")
    out = tmp / "index"
    P.build(P.full_profile(out), src=src, processes=1, persist_statutes=False,
            verbose=False)
    return P.PositionalIndex.load(out), [_record(i, d) for i, d in enumerate(DOCS)]


# --------------------------------------------------------------------------
# smallest_window: the sliding-window primitive
# --------------------------------------------------------------------------

def test_smallest_window_finds_the_narrowest_covering_span():
    pos = np.array([1, 4, 5, 9, 10, 20], dtype=np.int64)
    tid = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    omega, start, end = RP.smallest_window(pos, tid, 2)
    assert (omega, start, end) == (2, 4, 5)


def test_smallest_window_needs_every_term():
    pos = np.array([1, 2, 3], dtype=np.int64)
    tid = np.array([0, 0, 0], dtype=np.int64)
    assert RP.smallest_window(pos, tid, 2) == (0, -1, -1)
    assert RP.smallest_window(np.empty(0, dtype=np.int64),
                              np.empty(0, dtype=np.int64), 2) == (0, -1, -1)


def test_smallest_window_handles_three_terms():
    pos = np.array([0, 5, 6, 7, 30], dtype=np.int64)
    tid = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    omega, start, end = RP.smallest_window(pos, tid, 3)
    assert (omega, start, end) == (3, 5, 7)


def test_smallest_window_is_linear_not_quadratic():
    """A quadratic implementation would not finish this in a test run."""
    n = 200_000
    pos = np.arange(n, dtype=np.int64)
    tid = (pos % 3).astype(np.int64)
    omega, _s, _e = RP.smallest_window(pos, tid, 3)
    assert omega == 3


# --------------------------------------------------------------------------
# The score
# --------------------------------------------------------------------------

def test_adjacent_terms_score_higher_than_scattered_ones(index):
    idx, _recs = index
    near = RP.proximity_score(["privaci", "surveil"], 0, idx)
    far = RP.proximity_score(["privaci", "surveil"], 1, idx)
    assert near.score > far.score
    assert near.omega < far.omega
    assert near.within_paragraph
    assert not far.within_paragraph


def test_the_formula_is_min_one_q_over_omega(index):
    idx, _recs = index
    s = RP.proximity_score(["privaci", "surveil"], 0, idx)
    assert s.score == pytest.approx(min(1.0, s.n_terms / s.omega))


def test_a_cross_paragraph_match_is_discounted(index):
    idx, _recs = index
    with_rule = RP.proximity_score(["privaci", "surveil"], 1, idx)
    plain = RP.proximity_score(["privaci", "surveil"], 1, idx, per_paragraph=False)
    assert with_rule.score == pytest.approx(plain.score * RP.CROSS_PARA_DISCOUNT)
    assert plain.omega == with_rule.omega


def test_the_paragraph_rule_is_a_flag_so_the_ablation_is_free(index):
    idx, _recs = index
    for d in (0, 1):
        on = RP.proximity_score(["privaci", "surveil"], d, idx, per_paragraph=True)
        off = RP.proximity_score(["privaci", "surveil"], d, idx, per_paragraph=False)
        assert on.omega == off.omega          # same window either way
        assert on.score <= off.score + 1e-9   # the rule can only discount


def test_only_the_terms_actually_found_count_towards_q(index):
    """Otherwise a document matching 2 of 5 terms is punished twice: once by the
    text scorer, which already saw two matches, and again here."""
    idx, _recs = index
    s = RP.proximity_score(["privaci", "surveil", "zzznotaterm"], 0, idx)
    assert s.n_terms == 2
    assert s.terms_found == ["privaci", "surveil"]


def test_a_document_with_one_query_term_scores_zero(index):
    idx, _recs = index
    s = RP.proximity_score(["bail", "surveil"], 2, idx)
    assert s.n_terms == 1
    assert s.score == 0.0


def test_a_document_with_no_query_term_scores_zero(index):
    idx, _recs = index
    s = RP.proximity_score(["zzznotaterm"], 0, idx)
    assert s.score == 0.0 and s.n_terms == 0


def test_the_score_is_bounded_in_zero_one(index):
    idx, _recs = index
    for d in range(len(DOCS)):
        for terms in (["privaci", "surveil"], ["bail", "appeal"],
                      ["the", "of", "appeal"]):
            s = RP.proximity_score(terms, d, idx)
            assert 0.0 <= s.score <= 1.0


def test_a_window_wider_than_the_cap_scores_zero():
    assert RP._score_from(RP.MAX_WINDOW + 1, 5) == 0.0
    assert RP._score_from(RP.MAX_WINDOW, 5) > 0.0


def test_batch_scoring_matches_single_scoring(index):
    idx, _recs = index
    ids = list(range(len(DOCS)))
    batch = RP.proximity_scores(["privaci", "surveil"], ids, idx)
    one = [RP.proximity_score(["privaci", "surveil"], d, idx).score for d in ids]
    assert np.allclose(batch, one)


def test_explain_prints_the_formula_and_beta(index):
    idx, _recs = index
    line = RP.explain(["privaci", "surveil"], 0, idx)
    assert "smallest window" in line
    assert "prox =" in line
    assert str(config.BETA) in line


# --------------------------------------------------------------------------
# Snippets
# --------------------------------------------------------------------------

def test_a_stemmed_query_term_highlights_its_surface_word(index):
    """The measured bug: Porter maps 'privacy' to 'privaci', and 'privacy' does
    not start with 'privaci', so stem-prefix highlighting marked nothing."""
    spans = SN.term_spans("The right to privacy is fundamental.", ["privaci"])
    assert spans
    a, b = spans[0]
    assert "The right to privacy is fundamental."[a:b] == "privacy"


def test_highlight_marks_every_matching_word():
    text = "Privacy and surveillance: privacy again."
    marked, spans, _statutes = SN.highlight(text, ["privaci", "surveil"])
    assert len(spans) == 3
    # HL_OPEN and HL_CLOSE are the same marker, so three matches give six
    assert marked.count(SN.HL_OPEN) == 6
    assert marked == ("**Privacy** and **surveillance**: **privacy** again.")


def test_highlight_leaves_a_non_matching_word_alone():
    marked, _s, _st = SN.highlight("The appeal is dismissed.", ["privaci"])
    assert SN.HL_OPEN not in marked


def test_a_statute_query_highlights_the_judgments_own_wording():
    """The index holds ipc302; the judgment says 'Section 302 of the Indian Penal
    Code', and that is what has to appear on screen."""
    text = ("The accused was convicted under Section 302 of the Indian Penal "
            "Code by the trial court.")
    spans = SN.statute_spans_for(text, ["ipc§302"])
    assert spans
    a, b, token = spans[0]
    assert token == "ipc§302"
    assert "302" in text[a:b]
    assert "Penal" in text[a:b] or "Section" in text[a:b]
    marked, _s, shown = SN.highlight(text, ["ipc§302"])
    assert SN.HL_OPEN in marked
    assert shown == ["ipc§302"]


def test_a_statute_token_is_never_matched_as_a_literal_string():
    marked, _s, _st = SN.highlight("no statute here at all", ["ipc§302"])
    assert SN.HL_OPEN not in marked


def test_paragraphs_are_ranked_by_distinct_term_density(index):
    idx, _recs = index
    ranked = SN.rank_paragraphs(idx, ["privaci", "surveil"], 1, k=5)
    assert ranked
    densities = [d for _p, d, _t in ranked]
    assert densities == sorted(densities, reverse=True)
    # the paragraph with no query term must not appear
    assert all(terms for _p, _d, terms in ranked)


def test_snippets_come_only_from_the_body_and_order_zones(index):
    """A statute token lives in the synthetic `statutes` zone, whose positions
    would otherwise nominate paragraph 1 of every matching judgment."""
    idx, _recs = index
    hits = SN._para_positions(idx, ["ipc§302"], 3)
    for para, per_term in hits.items():
        for positions in per_term.values():
            zones = idx.zone_at(3, positions)
            assert set(map(int, zones)) <= set(map(int, SN._SNIPPET_ZONE_IDS))


def test_snippet_gutter_says_window_when_the_paragraph_was_invented(index):
    """Printing '¶14' for a window this project invented would be a lie, and a
    legal reader notices immediately."""
    idx, _recs = index
    snips = SN.snippets_for(0, ["privaci", "surveil"], idx, paras=None)
    # with no paragraph store there is no text, but the gutter must still be
    # honest about what it does not know
    assert all(not s.is_printed_para for s in snips)


def test_snippet_carries_the_terms_that_matched(index):
    idx, _recs = index
    ranked = SN.rank_paragraphs(idx, ["privaci", "surveil"], 0, k=1)
    assert ranked[0][2] == ["privaci", "surveil"]


def test_snippet_round_trips_to_a_dict_and_back_to_plain_text():
    s = SN.Snippet(doc_id=1, para_id=2, zone="body",
                   text=f"a {SN.HL_OPEN}b{SN.HL_CLOSE} c", gutter="w2")
    d = s.as_dict()
    assert d["gutter"] == "w2" and d["doc_id"] == 1
    assert s.plain() == "a b c"


def test_merge_spans_joins_overlaps():
    assert SN._merge_spans([(0, 3), (2, 5), (10, 12)]) == [(0, 5), (10, 12)]
    assert SN._merge_spans([]) == []


def test_the_window_is_cut_at_word_boundaries():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    a, b = SN._window_around(text, [(30, 34)], width=20)
    assert a == 0 or text[a - 1].isspace()
    assert b == len(text) or text[b].isspace()


def test_render_is_printable_and_shows_the_gutter():
    s = SN.Snippet(doc_id=1, para_id=7, zone="body", text="x", gutter="¶7",
                   density=0.5, statutes=["ipc§302"])
    out = SN.render([s])
    assert "¶7" in out and "ipc§302" in out and "0.500" in out
