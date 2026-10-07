"""Tests for the citation treatment classifier  (Member 1, wave 3).

Every fixture sentence below is a real window from ``index/citations.jsonl``,
shortened.  The three rules that are not a keyword list - the reporter's own
disposition, negation scope, and the bench rule - each have their own tests,
including tests for the two measured failure modes they were written to fix.
"""
from __future__ import annotations

import json

import pytest

from nyaya import config
from nyaya.authority import treatment as T


# --------------------------------------------------------------------------
# The reporter's disposition, which beats any cue-distance heuristic
# --------------------------------------------------------------------------

HEADNOTE = ("Its Employees [1960] 2 S.C.R. 942 referred to. "
            "Letang v. Cooper [1965] I Q.B. 232 distinguished.")


def test_each_entry_of_a_headnote_list_gets_its_own_disposition():
    """The measured bug: the nearest-cue rule read the *neighbour's*
    disposition and labelled a 'referred to' case as distinguished."""
    assert T.classify(HEADNOTE, "[1960] 2 S.C.R. 942").treatment == T.MENTIONED
    assert T.classify(HEADNOTE, "[1965] I Q.B. 232").treatment == T.DISTINGUISHED


@pytest.mark.parametrize("phrase, expected", [
    ("referred to", T.MENTIONED),
    ("cited", T.MENTIONED),
    ("relied on", T.FOLLOWED),
    ("relied upon", T.FOLLOWED),
    ("followed", T.FOLLOWED),
    ("approved", T.FOLLOWED),
    ("applied", T.FOLLOWED),
    ("distinguished", T.DISTINGUISHED),
    ("held inapplicable", T.DISTINGUISHED),
    ("explained", T.DISTINGUISHED),
    ("overruled", T.OVERRULED),
    ("not followed", T.OVERRULED),
])
def test_the_case_law_reference_table_is_read_per_citation(phrase, expected):
    window = f"2012 (2) SCR 427 referred to Para 10 2010 (10) SCR 487 {phrase} Para 11"
    lab = T.classify(window, "2010 (10) SCR 487")
    assert lab.treatment == expected
    assert lab.rule == "disposition"


def test_a_disposition_too_far_after_the_citation_is_not_used():
    filler = "x" * (T.DISPOSITION_WINDOW_CHARS + 20)
    window = f"Some Case [1960] 1 SCR 1 {filler} overruled."
    assert T.classify(window, "[1960] 1 SCR 1").rule != "disposition"


def test_a_citation_broken_across_a_line_still_finds_its_disposition():
    """The offset bug: the citation is only findable in whitespace-squeezed text,
    and the squeezed index has to be mapped back to the original window or the
    disposition scan starts in the wrong place."""
    window = "Dalbir Singh v. State of Punjab (1979) 3\nSCC 745 - referred to."
    lab = T.classify(window, "(1979) 3\nSCC 745")
    assert lab.rule == "disposition"
    assert lab.treatment == T.MENTIONED


def test_cite_span_maps_a_squeezed_match_back_to_the_window():
    window = "x y (1979) 3\nSCC 745 z"
    start, end = T.cite_span(window, "(1979) 3 SCC 745")
    assert window[start:end].replace("\n", " ") == "(1979) 3 SCC 745"


def test_cite_span_is_zero_when_the_citation_is_absent():
    assert T.cite_span("nothing here", "[1999] 1 SCR 1") == (0, 0)
    assert T.cite_span("anything", "") == (0, 0)


# --------------------------------------------------------------------------
# Negation
# --------------------------------------------------------------------------

def test_a_negated_overruling_is_not_an_overruling():
    lab = T.classify("We are unable to hold that the decision in Mathew "
                     "[1960] 1 SCR 1 stands overruled.", "[1960] 1 SCR 1")
    assert lab.treatment == T.MENTIONED
    assert lab.negated_cues


@pytest.mark.parametrize("negator", [
    "cannot be said that", "we do not hold that", "it is not the case that",
    "we decline to hold that", "there is no question of holding that",
])
def test_every_negator_shape_suppresses_the_cue(negator):
    window = f"{negator} the decision in X [1960] 1 SCR 1 is overruled."
    assert T.classify(window, "[1960] 1 SCR 1").treatment == T.MENTIONED


def test_a_negator_beyond_the_window_does_not_suppress_the_cue():
    """Measured: 'not less than twenty reported decisions were considered and
    thereafter the Court has overruled X' is a real overruling, and the 'not'
    is far away and belongs to a different clause."""
    window = ("A study of the decision makes it evident that not less than "
              "twenty reported decisions of this Court were taken into "
              "consideration and thereafter the Court has overruled the "
              "decision in Baidyanath Mahapatra v. State of Orissa "
              "(1989) 4 SCC 664, which held otherwise.")
    assert T.classify(window, "(1989) 4 SCC 664").treatment == T.OVERRULED


def test_the_negation_window_is_a_character_window_and_that_is_the_limitation():
    """The documented false negative, kept as a test so it cannot be forgotten.

    'could not be continued by his legal representatives' is part of the
    proposition the cited case held, not a negation of the 'overruled' that
    follows it - but a character window cannot tell the difference.
    """
    window = ("Naubat Rai, [1973] 3 SCR 769; a decision which was relied by "
              "petitioner in support of submission that an application lapses "
              "on the death of landlord and it could not be continued by his "
              "legal representatives overruled in Shanti Lal Thakur Das.")
    lab = T.classify(window, "[1973] 3 SCR 769")
    assert lab.treatment == T.MENTIONED      # the truth is 'overruled'
    assert lab.negated_cues, "the suppression must be visible in the explanation"


# --------------------------------------------------------------------------
# Who did what to whom
# --------------------------------------------------------------------------

def test_overruling_a_contention_is_not_overruling_a_precedent():
    lab = T.classify("This Court overruled the contention on behalf of the "
                     "respondent which was based on [1953] S.C.R. 1144.",
                     "[1953] S.C.R. 1144")
    assert lab.treatment == T.MENTIONED
    assert lab.rejected


def test_a_cited_case_that_is_the_one_doing_the_overruling_is_not_overruled():
    lab = T.classify("Later, in State of Gujarat v. Shantilal, "
                     "[1969] 3 S.C.R. 341, the Court overruled the decision "
                     "in Metal Corporation case.", "[1969] 3 S.C.R. 341")
    assert lab.treatment != T.OVERRULED


def test_overruled_by_a_cited_case_does_not_overrule_that_case():
    lab = T.classify("The decision in Pune Municipal Corporation has been "
                     "specifically overruled by the Constitution Bench in "
                     "Indore Development Authority, (2020) 8 SCC 129.",
                     "(2020) 8 SCC 129")
    assert lab.treatment != T.OVERRULED


# --------------------------------------------------------------------------
# Cue-distance caps
# --------------------------------------------------------------------------

def test_an_overruling_cue_far_from_the_citation_is_not_used():
    # the filler must not itself contain a disposition word, or the reporter's
    # disposition rule fires first and the cap is never reached
    filler = "and many other matters were argued at great length by counsel " * 4
    window = (f"Some Case v. Another, [1960] 1 SCR 1, {filler} "
              f"the earlier view stands overruled.")
    lab = T.classify(window, "[1960] 1 SCR 1")
    assert lab.treatment == T.MENTIONED
    assert any("cap" in r or "sentence" in r for r in lab.rejected)


def test_an_overruling_cue_in_another_sentence_is_not_used():
    window = ("Ltd., [1953] S.C.R. 1069 should be reconsidered. "
              "Four Judges of the Bench of seven voted to overrule it.")
    assert T.classify(window, "[1953] S.C.R. 1069").treatment == T.MENTIONED


def test_a_followed_cue_is_not_distance_capped_because_lists_share_one():
    """'A; B; C - relied on' is one disposition for a whole list, and all ten
    sampled followings were correct at distances up to 423 characters."""
    assert T.MAX_CUE_DISTANCE[T.FOLLOWED] == 0
    assert T.MAX_CUE_DISTANCE[T.OVERRULED] > 0
    names = "; ".join(f"Case {i} v. State, [19{60 + i}] {i} SCR {i * 7}"
                      for i in range(8))
    window = f"{names}; Target v. State, [1982] 2 SCR 533 - relied on."
    lab = T.classify(window, "[1960] 0 SCR 0")
    assert lab.treatment == T.FOLLOWED


# --------------------------------------------------------------------------
# The bench rule
# --------------------------------------------------------------------------

def test_a_smaller_bench_cannot_overrule_a_larger_one():
    window = "A two-judge bench purported to overrule Kesavananda [1973] 4 SCC 225."
    lab = T.classify(window, "[1973] 4 SCC 225", bench_src=2, bench_dst=13)
    assert lab.raw == T.OVERRULED
    assert lab.treatment == T.DISTINGUISHED
    assert lab.downgraded
    assert "13" in lab.explain() and "2" in lab.explain()


def test_an_equal_or_larger_bench_may_overrule():
    window = "Kesavananda [1973] 4 SCC 225, overruled."
    for src in (13, 14):
        lab = T.classify(window, "[1973] 4 SCC 225", bench_src=src, bench_dst=13)
        assert lab.treatment == T.OVERRULED
        assert not lab.downgraded


def test_the_bench_rule_abstains_when_a_bench_size_is_unknown():
    """A zero means 'not known', and the rule must not fire on a guess."""
    window = "Some Case [1960] 1 SCR 1, overruled."
    for src, dst in ((0, 13), (2, 0), (0, 0)):
        lab = T.classify(window, "[1960] 1 SCR 1", bench_src=src, bench_dst=dst)
        assert lab.treatment == T.OVERRULED, f"bench {src} vs {dst}"
        assert not lab.downgraded


def test_the_downgrade_changes_the_pagerank_weight_as_intended():
    """The point of encoding the rule in the label: 0.0 would zero the cited
    judgment's authority on the word of a bench that had no power to do it."""
    assert config.TREATMENT_WEIGHTS[T.OVERRULED] == 0.0
    assert config.TREATMENT_WEIGHTS[T.DISTINGUISHED] > 0.0
    assert config.TREATMENT_PENALTY[T.DISTINGUISHED] > \
        config.TREATMENT_PENALTY[T.OVERRULED]


# --------------------------------------------------------------------------
# Defaults, contract compliance and the file round trip
# --------------------------------------------------------------------------

def test_a_bare_citation_is_mentioned():
    lab = T.classify("In Special Reference No. 1 of 1964, [1965] 1 S.C.R. 413 "
                     "at 446, Gajendragadkar, C.J. observed as follows.",
                     "[1965] 1 S.C.R. 413")
    assert lab.treatment == T.MENTIONED
    assert lab.rule == "default"


def test_an_empty_window_is_mentioned():
    assert T.classify("", "").treatment == T.MENTIONED
    assert T.classify("   ", "[1960] 1 SCR 1").treatment == T.MENTIONED


def test_every_label_is_one_the_config_knows_how_to_weight():
    for label in (T.OVERRULED, T.DISTINGUISHED, T.FOLLOWED, T.MENTIONED):
        assert label in config.TREATMENT_WEIGHTS


def test_label_file_writes_only_contract_fields(tmp_path, monkeypatch):
    """`citations.load_edges` does `CitationEdge(**row)`, so an extra key would
    break Member 2's loader."""
    from nyaya.contracts import CitationEdge

    src = tmp_path / "citations.jsonl"
    rows = [
        {"src": 0, "dst": 1, "treatment": "", "sentence": HEADNOTE,
         "para_id": 3, "cite_string": "[1965] I Q.B. 232", "resolved_by": "scr"},
        {"src": 0, "dst": 2, "treatment": "", "sentence": HEADNOTE,
         "para_id": 3, "cite_string": "[1960] 2 S.C.R. 942", "resolved_by": "scr"},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(T, "build_bench_cache", lambda *a, **k: {
        "doc_bench": __import__("numpy").zeros(8, dtype="int16"),
        "node_bench": __import__("numpy").zeros(8, dtype="int16"),
    })
    monkeypatch.setattr(T, "TREATMENT_STATS", tmp_path / "stats.json")
    stats = T.label_file(src, verbose=False)
    assert stats["edges"] == 2
    out = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()]
    assert out[0]["treatment"] == T.DISTINGUISHED
    assert out[1]["treatment"] == T.MENTIONED
    for row in out:
        CitationEdge(**row)          # must not raise on an unexpected key


def test_dry_run_does_not_touch_the_file(tmp_path, monkeypatch):
    src = tmp_path / "citations.jsonl"
    row = {"src": 0, "dst": 1, "treatment": "", "sentence": HEADNOTE,
           "para_id": 1, "cite_string": "[1965] I Q.B. 232", "resolved_by": "scr"}
    original = json.dumps(row) + "\n"
    src.write_text(original, encoding="utf-8")
    monkeypatch.setattr(T, "build_bench_cache", lambda *a, **k: {
        "doc_bench": __import__("numpy").zeros(8, dtype="int16"),
        "node_bench": __import__("numpy").zeros(8, dtype="int16"),
    })
    T.label_file(src, dry_run=True, verbose=False)
    assert src.read_text(encoding="utf-8") == original


def test_sentence_boundaries_come_from_member_threes_splitter():
    """Not our own splitter: the sent_id values in the index were produced by
    hers, and a second splitter that disagreed would mislabel 'same sentence'."""
    from nyaya.analyze import sentences

    window = ("The appeal is allowed. Sec. 302 of the I.P.C. was not attracted. "
              "The conviction is set aside.")
    spans = T._sentence_spans(window)
    assert len(spans) == len(sentences(window))
    for _s, _e, text in spans:
        assert text in window


# --------------------------------------------------------------------------
# Direction: the cue is present but does not predicate of the cited case
# --------------------------------------------------------------------------

MANEKA = (
    "In this connection, reliance was placed on the observations of this Court "
    "in E.P. Royappa v. State of Tamil Nadu, [1974] 2 SCR 348 and Maneka Gandhi "
    "v. Union of India, [1978] 2 SCR 621 where it was held that the view that "
    "Articles 19 and 21 constitute watertight compartments has been rightly "
    "overruled."
)


def test_a_landmark_that_overruled_something_is_not_itself_overruled():
    """Maneka Gandhi is live, foundational constitutional law and is famous for
    overrul*ing*.  The cue's grammatical subject is the view Maneka rejected, so
    labelling Maneka 'overruled' would print it in the report as dead law."""
    lab = T.classify(MANEKA, "[1978] 2 SCR 621")
    assert lab.treatment != T.OVERRULED
    assert lab.treatment == T.MENTIONED


@pytest.mark.parametrize("verb", ["held", "observed", "laid down", "stated",
                                  "ruled", "found", "declared"])
def test_every_reporting_verb_shields_the_cited_case(verb):
    window = (f"Reliance was placed on X v. Y, [1978] 2 SCR 621 where it was "
              f"{verb} that the earlier view was overruled.")
    assert T.classify(window, "[1978] 2 SCR 621").treatment != T.OVERRULED


@pytest.mark.parametrize("window, cite", [
    ("Rao Shiv Bahadur Singh, [1954] S.C.R. 1098, overruled.",
     "[1954] S.C.R. 1098"),
    ("[1988] 3 SCR 73 overruled", "[1988] 3 SCR 73"),
    ("Madhya Pradesh Industries Ltd., [1966] 1 S.C.R. 466 held overruled by "
     "Bhagat Raja.", "[1966] 1 S.C.R. 466"),
    ("The case of Firm Chhotabhai, [1953] SCR 476 is not good law and has been "
     "overruled by larger benches.", "[1953] SCR 476"),
    ("The decision in ADM, Jabalpur [(1976) 2 SCC 521] about the restrictive "
     "reading of right to life stood impliedly overruled by subsequent "
     "decisions.", "(1976) 2 SCC 521"),
    ("Pune Municipal Corpn. v. Harakchand, (2014) 3 SCC 183 is hereby overruled "
     "and all other decisions following it.", "(2014) 3 SCC 183"),
    ("Krishna Veni Nagam v. Harish Nagam (2017) 4 SCC 150 - partly overruled.",
     "(2017) 4 SCC 150"),
])
def test_the_direction_rule_keeps_every_genuine_overruling(window, cite):
    """The rule must not buy precision by losing real overrulings: these seven
    are the hand-checked genuine ones from the audit sample."""
    assert T.classify(window, cite).treatment == T.OVERRULED


def test_a_transitive_cue_whose_object_is_another_case_is_not_a_treatment():
    window = ("This Court in All India Railway Recruitment Board v. K. Shyam "
              "Kumar [(2010) 6 SCC 614] distinguished Mohinder Singh Gill's "
              "case, stating that a larger public interest is involved.")
    assert T.classify(window, "(2010) 6 SCC 614").treatment != T.DISTINGUISHED


def test_but_a_cue_whose_object_is_the_cited_case_still_counts():
    """'overrule Kesavananda [1973] 4 SCC 225' also has a capital after the cue,
    and there the capitalised words ARE the cited case.  The direction of the
    citation relative to the cue is what tells the two apart."""
    lab = T.classify("A two-judge bench purported to overrule Kesavananda "
                     "[1973] 4 SCC 225.", "[1973] 4 SCC 225",
                     bench_src=2, bench_dst=13)
    assert lab.raw == T.OVERRULED
    assert lab.treatment == T.DISTINGUISHED      # the bench rule then downgrades


def test_distinguished_in_its_ordinary_english_sense_is_not_a_treatment():
    window = ("It allows a permissive inference of fact, as distinguished from "
              "a mandatory presumption, having regard to 2021 (10) SCR 1118.")
    assert T.classify(window, "2021 (10) SCR 1118").treatment != T.DISTINGUISHED


def test_edge_key_is_stable_across_a_rebuild_but_not_across_a_window_change():
    base = {"src": 1, "dst": 2, "cite_string": "[1978] 2 SCR 621",
            "para_id": 7, "sentence": "some window"}
    same_rebuilt = dict(base, treatment="overruled")      # a label is not part of it
    assert T.edge_key(base) == T.edge_key(same_rebuilt)
    # whitespace in the citation is normalised, so a re-wrapped line still matches
    assert T.edge_key(dict(base, cite_string="[1978]  2\nSCR 621")) == \
        T.edge_key(base)
    # but a different evidence window is a different judgment to make
    assert T.edge_key(dict(base, sentence="a different window")) != \
        T.edge_key(base)
