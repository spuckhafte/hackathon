"""Tests for `analyze()` — the function the whole system is downstream of.

The invariant that matters most is **document/query symmetry**: the index builder
and the query normaliser call the same function, so anything that makes them
diverge is a silent recall bug.  Several tests below exist only to pin that down.

Network-free.  Author: Sumedha (M3)
"""
from __future__ import annotations

import pytest

from nyaya import analyze as A
from nyaya.analyze import analyze, analyze_terms, clean_text, sentences
from nyaya.contracts import Token
from nyaya.stoplist import SCORING_STOPWORDS, is_scoring_stopword


# ==========================================================================
# 1. The contract
# ==========================================================================


def test_returns_contract_tokens_with_zone_and_para():
    toks = analyze("The appellant filed an appeal.", zone="title", para_id=7)
    assert toks and all(isinstance(t, Token) for t in toks)
    assert {t.zone for t in toks} == {"title"}
    assert {t.para_id for t in toks} == {7}
    assert [t.pos for t in toks] == list(range(len(toks)))


def test_positions_are_contiguous_and_include_stop_words():
    """Phrase and /k proximity are only exact if stop words occupy positions."""
    toks = analyze("the right to privacy")
    assert [t.term for t in toks] == ["the", "right", "to", "privaci"]
    assert [t.pos for t in toks] == [0, 1, 2, 3]
    # "right to privacy" is a 3-word phrase with consecutive positions
    pos = {t.term: t.pos for t in toks}
    assert pos["privaci"] - pos["right"] == 2


def test_start_pos_lets_the_builder_concatenate_paragraphs():
    a = analyze("the appellant", start_pos=0)
    b = analyze("was convicted", start_pos=len(a))
    assert [t.pos for t in a + b] == [0, 1, 2, 3]


def test_analyze_terms_matches_analyze():
    text = "Section 302 of the Indian Penal Code, 1860 was invoked."
    assert analyze_terms(text) == [t.term for t in analyze(text)]


def test_empty_and_whitespace_input():
    assert analyze("") == []
    assert analyze("   \n\n  ") == []


def test_deterministic():
    text = "The Appellant was convicted u/s 302 IPC."
    assert analyze_terms(text) == analyze_terms(text)


# ==========================================================================
# 2. Pass 0 — the character stream
# ==========================================================================


def test_clean_text_rejoins_pdf_line_wrapped_words():
    assert "appellant" in clean_text("the appel-\nlant was convicted")
    # Porter("appellant") == "appel"; what must NOT appear is the broken halves.
    terms = analyze_terms("the appel-\nlant was convicted")
    assert "appel" in terms and "lant" not in terms


def test_clean_text_normalises_smart_punctuation():
    assert clean_text("“Hon’ble”") == '"Hon\'ble"'
    assert "hon'ble" in analyze_terms("the Hon’ble Court")


def test_clean_text_preserves_paragraph_breaks():
    """They are the reset points of the act-inheritance heuristic."""
    assert "\n\n" in clean_text("para one\n\n\n\npara two")


# ==========================================================================
# 3. Pass 1 — statute canonicalisation, and that it runs first
# ==========================================================================


def test_statute_tokens_survive_tokenisation_intact():
    terms = analyze_terms("convicted under Sec. 302 of the Indian Penal Code, 1860")
    assert "ipc§302" in terms
    assert "302" not in terms


def test_canonicalise_flag_gives_the_b0_baseline():
    text = "convicted under Sec. 302 of the Indian Penal Code, 1860"
    off = analyze_terms(text, canonicalise=False)
    assert "ipc§302" not in off
    assert "302" in off and "penal" in off and "code" in off


def test_act_inheritance_flows_through_analyze():
    text = ("convicted under Section 302 of the Indian Penal Code, 1860 "
            "and also u/s 34 and S. 120B")
    terms = analyze_terms(text)
    assert {"ipc§302", "ipc§34", "ipc§120b"} <= set(terms)


def test_inherit_act_flag_is_exposed_for_the_ablation():
    text = "Section 302 IPC. He was also charged under Section 34."
    assert "ipc§34" in analyze_terms(text, inherit_act=True)
    assert "ipc§34" not in analyze_terms(text, inherit_act=False)


def test_document_and_query_agree_on_a_statute():
    doc = analyze_terms("He was convicted under Section 302 of the Indian Penal Code, 1860.")
    query = analyze_terms("S. 302 IPC murder")
    assert "ipc§302" in doc and "ipc§302" in query


# ==========================================================================
# 4. Pass 2 — tokenisation, folding, sentence splitting
# ==========================================================================


def test_case_folding_and_possessives():
    assert analyze_terms("the Appellants' plea", stem=False) == [
        "the", "appellants", "plea",
    ]
    assert analyze_terms("the Appellant's plea", stem=False) == [
        "the", "appellant", "plea",
    ]


def test_hyphenated_words_split():
    assert analyze_terms("a well-known principle", stem=False) == [
        "a", "well", "known", "principle",
    ]


def test_initials_collapse_to_one_token():
    assert "ks" in analyze_terms("K.S. Puttaswamy")
    assert "air" in analyze_terms("A.I.R. 1973 SC 1461")


def test_sentence_split_does_not_break_on_a_case_name():
    """The canonical failure case: "K.S. Puttaswamy v. Union of India" contains
    four full stops and is one sentence."""
    text = ("The Court followed K.S. Puttaswamy v. Union of India, (2017) 10 SCC 1. "
            "The appeal was dismissed.")
    assert len(sentences(text)) == 2
    assert "Puttaswamy" in sentences(text)[0]
    sent_ids = {t.term: t.sent_id for t in analyze(text)}
    assert sent_ids["puttaswamy"] == 0
    assert sent_ids["dismiss"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "Civil Appeal Nos. 238 and 239 of 2012 were heard together.",
        "Messrs. Tata Steel Ltd. and Birla Pvt. Ltd. appeared.",
        "Shah, J. and Kapur, JJ. delivered the judgment.",
        "Per Krishna Iyer, J. the appeal must fail.",
        "See Art. 21 and Sec. 302 and cl. (a) thereof.",
        "Mr. Singh for the appellant and Dr. Dhavan for the respondent.",
        "Smt. Indira Gandhi and Mrs. Sonia filed No. 4 of 1975.",
        "Reported at AIR 1973 SC 1461 and (1973) 4 SCC 225 respectively.",
        "The charge was u/s 302 r/w 34 of the Penal Code.",
    ],
)
def test_legal_abbreviations_never_end_a_sentence(text):
    assert len(sentences(text)) == 1, sentences(text)


def test_real_sentence_boundaries_are_found():
    text = "He was convicted. He appealed. The appeal was allowed."
    assert len(sentences(text)) == 3
    assert max(t.sent_id for t in analyze(text)) == 2


def test_sentence_id_is_monotone():
    text = ("First sentence here. Second one, citing Puttaswamy v. Union of India. "
            "Third and last.")
    ids = [t.sent_id for t in analyze(text)]
    assert ids == sorted(ids)
    assert ids[0] == 0


# ==========================================================================
# 5. Pass 3 — stop words: in the positional index, out of the tf-idf vector
# ==========================================================================


def test_stop_words_are_returned_by_analyze():
    toks = analyze("the right to privacy is a fundamental right")
    assert "the" in [t.term for t in toks]
    assert "to" in [t.term for t in toks]


def test_stop_words_are_excluded_from_the_scoring_bag():
    text = "the right to privacy is a fundamental right"
    positional = analyze_terms(text)
    scoring = A.analyze_scoring_terms(text)
    assert len(scoring) < len(positional)
    assert "the" in positional and "the" not in scoring
    assert "privaci" in scoring


def test_the_stop_set_contains_stemmed_forms():
    """The index stores stems, so the scorers look up stems.  Porter turns "was"
    into "wa"; if "wa" were not in the set, every judgment would carry it."""
    assert is_scoring_stopword("wa")             # Porter("was")
    assert is_scoring_stopword("the")
    assert "wa" in SCORING_STOPWORDS


def test_statute_tokens_are_never_stop_words():
    assert not is_scoring_stopword("ipc§302")
    assert not is_scoring_stopword("const§21")


def test_domain_stop_words_come_with_their_measured_df():
    """The report prints this table, so the numbers must actually be there."""
    from nyaya import stoplist

    for term, value in stoplist.DOMAIN_DF.items():
        assert isinstance(value, tuple) and len(value) == 2, term
        df, n = value
        assert 0 < df <= n, term
        assert df / n >= stoplist.DOMAIN_DF_THRESHOLD, term
        assert is_scoring_stopword(term), term


# ==========================================================================
# 6. Pass 4 — stemming and its exceptions
# ==========================================================================


def test_morphological_variants_collapse():
    for surface in ("convicted", "conviction", "convicts", "convicting"):
        assert analyze_terms(surface) == ["convict"], surface


def test_stem_flag_is_an_ablation_switch():
    assert analyze_terms("convicted", stem=True) == ["convict"]
    assert analyze_terms("convicted", stem=False) == ["convicted"]


def test_statute_tokens_are_never_stemmed():
    for tok in ("ipc§302", "const§21a", "crpc§482", "bns§103"):
        text = f"charged under {tok}"
        assert tok in analyze_terms(text), tok


def test_numbers_are_never_stemmed():
    assert analyze_terms("in 1860 and 2017") == ["in", "1860", "and", "2017"]


def test_legal_abbreviations_are_not_stemmed():
    # Porter would turn "ors" into "or" (a stop word!) and "scc" is meaningless
    # to stem.
    terms = analyze_terms("reported in AIR 1973 SC 1461 and 4 SCC 225, ors.")
    assert "air" in terms and "scc" in terms and "ors" in terms


def test_case_names_in_a_party_context_are_not_stemmed():
    """"Puttaswamy" -> "puttaswami" is pure damage: nobody types the stem."""
    terms = analyze_terms("following K.S. Puttaswamy v. Union of India")
    assert "puttaswamy" in terms
    assert "puttaswami" not in terms


def test_the_party_exception_does_not_touch_ordinary_party_vocabulary():
    """If it did, the `parties` zone would hold "committee" while a query held
    the stem "committe", and the match would be lost."""
    doc = analyze_terms("K.P. Manu v. Chairman, Scrutiny Committee for Verification")
    query = analyze_terms("scrutiny committee verification")
    for term in query:
        assert term in doc, term


def test_protected_names_work_without_a_party_context():
    """A user who types just "puttaswamy privacy" must still match."""
    assert "puttaswamy" in analyze_terms("the puttaswamy judgment on privacy")


def test_protect_names_flag_can_be_switched_off():
    on = analyze_terms("following K.S. Rajagopal v. State of Tamil Nadu",
                       protect_names=True)
    off = analyze_terms("following K.S. Rajagopal v. State of Tamil Nadu",
                        protect_names=False)
    assert "rajagopal" in on
    assert "rajagop" in off and "rajagopal" not in off


def test_stem_cache_is_used():
    A._STEM_CACHE.clear()
    analyze_terms("convicted convicting convicts")
    assert A._STEM_CACHE.get("convicted") == "convict"


# ==========================================================================
# 7. The demo must run (it is in the video)
# ==========================================================================


def test_pipeline_trace_exposes_every_pass():
    tr = A.pipeline_trace(A.DEMO_SENTENCE)
    for key in ("raw", "cleaned", "canonicalised", "tokens_folded",
                "tokens_stemmed", "statutes", "sentences", "tokens"):
        assert key in tr and tr[key] is not None
    assert "ipc§302" in tr["statutes"]
    assert len(tr["tokens_folded"]) == len(tr["tokens_stemmed"])


def test_demo_cli_runs():
    assert A._main(["--demo", "convicted u/s 302 IPC"]) == 0
