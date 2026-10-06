"""Tests for the statute canonicaliser — the project's equivalence classing.

Network-free.  Each test names the IR property it protects, because the point of
this module is not "the regex fires" but "six surface forms collapse to one
index term".

Author: Sumedha (M3)
"""
from __future__ import annotations

import pytest

from nyaya import statutes as S


# ==========================================================================
# 1. Equivalence classing: many surface forms -> one term
# ==========================================================================
IPC_302_FORMS = [
    "Section 302 of the Indian Penal Code, 1860",
    "Section 302 of the Indian Penal Code",
    "Section 302 of the Penal Code, 1860",
    "section 302 IPC",
    "S. 302 IPC",
    "s. 302, I.P.C.",
    "u/s 302 I.P.C.",
    "u/s. 302 IPC",
    "Sec. 302 IPC",
    "Sec 302 of the Penal Code",
    "Sect. 302 of the Indian Penal Code",
    "Sections 302 IPC",
    "IPC 302",
    "I.P.C. 302",
    "the Indian Penal Code, Section 302",
    "under Section 302 of the Indian Penal Code, 1860",
]


@pytest.mark.parametrize("surface", IPC_302_FORMS)
def test_every_ipc_302_surface_form_collapses_to_one_token(surface):
    assert S.extract_statutes(surface) == ["ipc§302"]


CONST_21_FORMS = [
    "Article 21 of the Constitution",
    "Article 21 of the Constitution of India",
    "Art. 21",
    "Art 21 of the Constitution",
    "article 21",
    "Articles 21",
]


@pytest.mark.parametrize("surface", CONST_21_FORMS)
def test_constitution_article_forms(surface):
    assert S.extract_statutes(surface) == ["const§21"]


@pytest.mark.parametrize(
    "surface,expected",
    [
        ("Section 302A IPC", "ipc§302a"),
        ("Section 304-B of the Indian Penal Code", "ipc§304b"),
        ("Section 304B IPC", "ipc§304b"),
        ("Section 498A IPC", "ipc§498a"),
        ("Article 21A of the Constitution", "const§21a"),
        ("Section 34(1) IPC", "ipc§34"),              # sub-clause dropped
        ("Section 376(2)(g) IPC", "ipc§376"),
        ("Section 482 of the Code of Criminal Procedure", "crpc§482"),
        ("Section 482 Cr.P.C.", "crpc§482"),
        ("u/s 482 CrPC", "crpc§482"),
        ("Section 65B of the Indian Evidence Act, 1872", "evid§65b"),
        ("Section 138 of the Negotiable Instruments Act", "ni§138"),
        ("Section 20 of the NDPS Act", "ndps§20"),
        ("section 13(1)(d) of the Prevention of Corruption Act, 1988", "pca§13"),
        ("Section 103 of the Bharatiya Nyaya Sanhita, 2023", "bns§103"),
        ("s. 482 BNSS", "bnss§482"),
        ("Section 9 of the Arbitration and Conciliation Act, 1996", "arb§9"),
        ("Section 166 of the Motor Vehicles Act", "mva§166"),
        ("Order VII Rule 11 CPC", "cpc§o7r11"),
        ("Order 7 Rule 11 of the Code of Civil Procedure", "cpc§o7r11"),
    ],
)
def test_subsections_and_other_acts(surface, expected):
    assert S.extract_statutes(surface) == [expected]


def test_section_lists_expand():
    assert S.extract_statutes("Sections 302, 307 and 34 IPC") == [
        "ipc§302", "ipc§307", "ipc§34",
    ]
    assert S.extract_statutes("Section 302/34 IPC") == ["ipc§302", "ipc§34"]
    assert S.extract_statutes("under Sections 147, 148, 149 and 302 IPC") == [
        "ipc§147", "ipc§148", "ipc§149", "ipc§302",
    ]
    assert S.extract_statutes("Sections 391 to 393 of the Companies Act, 1956") == [
        "comp§391", "comp§393",
    ]


def test_mixed_acts_in_one_sentence():
    text = "Article 136 of the Constitution of India read with Section 482 CrPC"
    assert S.extract_statutes(text) == ["const§136", "crpc§482"]


# ==========================================================================
# 2. The act-inheritance heuristic (and its reset)
# ==========================================================================


def test_bare_section_inherits_the_act_named_earlier_in_the_paragraph():
    text = ("The appellant was convicted under Section 302 of the Indian Penal "
            "Code, 1860. The Trial Court also convicted him under Section 34 "
            "and Section 120B.")
    assert S.extract_statutes(text) == ["ipc§302", "ipc§34", "ipc§120b"]


def test_inheritance_resets_at_a_paragraph_boundary():
    text = ("Section 302 of the IPC was invoked.\n"
            "\n"
            "He was also charged under Section 34.")
    # Second paragraph has no act in scope, so the bare section stays plain text.
    assert S.extract_statutes(text) == ["ipc§302"]
    assert "section 34" in S.canonicalise_statutes(text).lower()


def test_inheritance_is_switchable_so_it_can_be_measured():
    text = "Section 302 IPC. He was also charged under Section 34."
    assert S.extract_statutes(text, inherit_act=True) == ["ipc§302", "ipc§34"]
    assert S.extract_statutes(text, inherit_act=False) == ["ipc§302"]


def test_inheritance_does_not_fire_for_an_act_we_do_not_know():
    """The guard that stops "Section 4 of the Dowry Prohibition Act" from
    becoming ipc§4 just because IPC was named earlier."""
    text = "Section 498A IPC and Section 4 of the Dowry Prohibition Act"
    assert S.extract_statutes(text) == ["ipc§498a"]

    text2 = "Section 302 of the IPC and Section 11 of the Juvenile Justice Act, 2015"
    assert S.extract_statutes(text2) == ["ipc§302"]


def test_bare_section_with_no_act_anywhere_is_left_alone():
    assert S.extract_statutes("sec. 302") == []
    assert S.canonicalise_statutes("sec. 302") == "sec. 302"


def test_query_side_may_seed_a_default_act():
    """A user typing "sec. 302" means the Penal Code; a *document* saying it gets
    no guess, because guessing there would fabricate a citation."""
    assert S.QUERY_FALLBACK_ACT == "IPC"
    out = S.canonicalise_statutes("sec. 302", default_act=S.QUERY_FALLBACK_ACT)
    assert "ipc§302" in out
    assert "ipc§302" not in S.canonicalise_statutes("sec. 302")


# ==========================================================================
# 3. Robustness
# ==========================================================================


def test_canonicalisation_is_idempotent():
    once = S.canonicalise_statutes("Section 302 of the Indian Penal Code, 1860")
    twice = S.canonicalise_statutes(once)
    assert once == twice
    assert S.extract_statutes(once) == ["ipc§302"]


@pytest.mark.parametrize(
    "text",
    [
        "In 1860 the Code was enacted.",
        "the Constitution 42nd Amendment",
        "i.e. a confession was recorded",
        "There were 302 persons present in the hall.",
        "",
    ],
)
def test_no_false_positives(text):
    assert S.extract_statutes(text) == []


def test_ordinary_words_are_untouched():
    text = "The appellant, a man of 45 years, filed an appeal."
    assert S.canonicalise_statutes(text) == text


# ==========================================================================
# 4. Helpers used by the index builder and the parametric index
# ==========================================================================


def test_act_of_and_section_of():
    assert S.ACT_OF("ipc§302") == "IPC"
    assert S.ACT_OF("const§21a") == "CONST"
    assert S.SECTION_OF("ipc§304b") == "304b"
    assert S.ACT_OF("murder") == ""


def test_extract_acts_feeds_the_parametric_field():
    text = "Section 302 IPC and Article 21 of the Constitution and Section 34 IPC"
    assert S.extract_acts(text) == ["IPC", "CONST"]


def test_extract_statutes_dedupes_and_keeps_order():
    text = "Section 302 IPC, Section 34 IPC, again Section 302 IPC"
    assert S.extract_statutes(text) == ["ipc§302", "ipc§34"]


def test_every_documented_abbreviation_resolves():
    for surface, code in S.ACT_ABBREVIATIONS.items():
        assert S.act_code(surface) == code, surface


def test_pretty_is_human_readable():
    assert S.pretty("ipc§302") == "Section 302, Indian Penal Code, 1860"
    assert S.pretty("const§21").startswith("Article 21,")
    assert S.pretty("cpc§o7r11").startswith("Order 7 Rule 11,")


def test_normalise_section():
    assert S.normalise_section("304 - B") == "304b"
    assert S.normalise_section("34(1)") == "34"
    assert S.normalise_section("302A") == "302a"


# ==========================================================================
# 5. The IPC <-> BNS crosswalk (query time only)
# ==========================================================================


def test_crosswalk_is_sourced():
    assert S.CROSSWALK_SOURCES, "the crosswalk must cite where it was verified"
    assert any("keralaprisons.gov.in" in u for u in S.CROSSWALK_SOURCES)


def test_crosswalk_forward_and_reverse():
    assert S.crosswalk("ipc§302") == ["bns§103"]
    assert "ipc§302" in S.crosswalk("bns§103")
    assert S.crosswalk("crpc§482") == ["bnss§528"]
    assert S.crosswalk("evid§65b") == ["bsa§63"]


def test_crosswalk_handles_a_one_to_many_split():
    """IPC 498A became BNS 85 (offence) + BNS 86 (definition of cruelty)."""
    assert S.crosswalk("ipc§498a") == ["bns§85", "bns§86"]
    assert "not one-to-one" in S.crosswalk_note("ipc§498a").lower()


def test_crosswalk_handles_a_many_to_one_merge():
    """IPC 405-409 all collapsed into BNS 316."""
    for sec in ("405", "406", "407", "408", "409"):
        assert S.crosswalk(f"ipc§{sec}") == ["bns§316"]
    back = S.crosswalk("bns§316")
    assert set(back) == {f"ipc§{s}" for s in ("405", "406", "407", "408", "409")}


def test_omitted_sections_expand_to_nothing_but_say_why():
    assert S.crosswalk("ipc§377") == []
    assert "omitted" in S.crosswalk_note("ipc§377").lower()
    assert S.crosswalk("ipc§497") == []
    assert S.crosswalk_note("ipc§497")


def test_the_482_collision_is_flagged():
    """CrPC 482 -> BNSS 528, but BNSS 482 is anticipatory bail (old CrPC 438).
    If we silently treated 482 as 482 the system would answer the wrong question."""
    assert S.crosswalk("crpc§482") == ["bnss§528"]
    assert S.crosswalk("crpc§438") == ["bnss§482"]
    assert "collision" in S.crosswalk_note("crpc§482").lower()


def test_unmapped_tokens_return_empty_rather_than_a_guess():
    assert S.crosswalk("ipc§999") == []
    assert S.crosswalk("const§21") == []        # the Constitution was not replaced
    assert S.crosswalk("murder") == []


def test_crosswalk_table_is_well_formed():
    for old, news in S.CROSSWALK.items():
        assert S.ACT_OF(old) in ("IPC", "CRPC", "EVID"), old
        assert isinstance(news, list) and news, old
        for new in news:
            assert S.ACT_OF(new) in ("BNS", "BNSS", "BSA"), (old, new)
            assert S.SECTION_OF(new), (old, new)
    # the three code families must not be mixed up
    fam = {"IPC": "BNS", "CRPC": "BNSS", "EVID": "BSA"}
    for old, news in S.CROSSWALK.items():
        assert all(S.ACT_OF(n) == fam[S.ACT_OF(old)] for n in news), old


def test_expand_statutes_builds_the_query_expansion_map():
    out = S.expand_statutes(["ipc§302", "const§21", "ipc§498a"])
    assert out == {"ipc§302": ["bns§103"], "ipc§498a": ["bns§85", "bns§86"]}
