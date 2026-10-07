"""Tests for the Precedent Brief.

Two matter more than the rest:

* `test_a_window_citation_never_renders_as_a_paragraph_number` — a generated
  legal brief that invents "¶ 14" for a judgment with no printed numbering is
  the worst failure this system could have, and 72% of the corpus has none.
* `test_the_verifier_catches_an_unsupported_sentence` — on the extractive path
  every sentence verifies by construction, so without a deliberately
  unsupported sentence the verifier would be untested and could be vacuous.

Network-free; no model is called.  Author: Sumedha (M3)
"""
from __future__ import annotations

import json

import pytest

from nyaya import config
from nyaya.brief import build, render
from nyaya.brief.compose import (BOILERPLATE, CITE_RE, compose_extractive,
                                 llm_available, render_prompt,
                                 split_cited_sentences)
from nyaya.brief.evidence import EvidencePack, EvidenceParagraph, build_pack
from nyaya.brief.verify import cosine, vector, verify
from nyaya.index.paragraphs import ParagraphStore

_BUILT = (config.INDEX / "dictionary.json").exists() and ParagraphStore.exists()
needs_index = pytest.mark.skipif(not _BUILT, reason="index or S1 not built")


# ==========================================================================
# 1. Honest citations — the non-negotiable
# ==========================================================================


def _pack(*paras) -> EvidencePack:
    p = EvidencePack(query="q")
    p.paragraphs = list(paras)
    return p


def _para(pid="P1", text="the dying declaration may form the sole basis of "
                         "conviction if it is found reliable and voluntary",
          cite="¶ 14", kind="printed", **kw):
    return EvidenceParagraph(pid=pid, doc_id=kw.pop("doc_id", 1),
                             para_id=kw.pop("para_id", 14), zone="body",
                             text=text, score=1.0, cite=cite, cite_kind=kind,
                             **kw)


def test_a_window_citation_never_renders_as_a_paragraph_number():
    p = _para(cite="window 14 of 32", kind="window")
    assert "¶" not in p.cite
    assert "¶" not in p.label()
    assert "window 14 of 32" in p.label()


def test_a_treatment_badge_travels_with_the_paragraph():
    """A brief that quotes an overruled judgment without saying so is worse
    than no brief."""
    p = _para(badge="overruled by 2 later judgments", status="overruled")
    assert "overruled" in p.label()


@needs_index
def test_the_real_pack_cites_both_kinds_and_never_fabricates():
    b = build("dying declaration as the sole basis of conviction")
    assert b.pack and b.pack.paragraphs
    for p in b.pack.paragraphs:
        assert p.cite_kind in ("printed", "printed-split", "window")
        if p.cite_kind == "window":
            assert "¶" not in p.cite
            assert p.cite.startswith("window ")
        elif p.cite_kind == "printed":
            assert p.cite.startswith("¶")


@needs_index
def test_a_window_in_the_pack_is_announced_in_the_notes():
    b = build("dying declaration as the sole basis of conviction")
    if any(p.cite_kind == "window" for p in b.pack.paragraphs):
        assert any("window" in n for n in b.notes)


# ==========================================================================
# 2. Selection
# ==========================================================================


@needs_index
def test_the_pack_respects_both_caps():
    b = build("anticipatory bail in economic offences")
    assert len(b.pack.paragraphs) <= config.BRIEF_MAX_PARAS
    per_doc = {}
    for p in b.pack.paragraphs:
        per_doc[p.doc_id] = per_doc.get(p.doc_id, 0) + 1
    assert max(per_doc.values()) <= config.BRIEF_MAX_PARAS_PER_DOC


@needs_index
def test_the_pack_draws_on_more_than_one_judgment():
    """A brief citing one case is a quotation, not a synthesis - that is what
    the per-document cap is for."""
    b = build("dying declaration as the sole basis of conviction")
    assert len({p.doc_id for p in b.pack.paragraphs}) >= 3


@needs_index
def test_paragraphs_are_ordered_by_score():
    b = build("Section 482 CrPC quashing of criminal proceedings")
    scores = [p.score for p in b.pack.paragraphs]
    assert scores == sorted(scores, reverse=True)


def test_the_reporters_own_index_is_not_quotable():
    """"List of Keywords ..." is the S.C.R.'s index, not the Court's words."""
    assert BOILERPLATE.search("List of Keywords Dying declaration; Failure")
    assert BOILERPLATE.search("Case Law Reference: Para 8")
    assert not BOILERPLATE.search("The dying declaration may form the sole basis")


# ==========================================================================
# 3. Citation parsing — the bug that made every sentence read as uncited
# ==========================================================================


def test_sentences_are_cut_at_the_citation_not_at_the_full_stop():
    """A full stop ends the sentence BEFORE its citation, so splitting on
    sentences first left every citation attached to the next sentence."""
    text = "The first claim. [P1] The second claim. [P2]"
    out = split_cited_sentences(text)
    assert [s.pid for s in out] == ["P1", "P2"]
    assert out[0].body == "The first claim."
    assert out[1].body == "The second claim."


def test_two_sentences_sharing_one_citation_leaves_the_first_uncited():
    out = split_cited_sentences("A claim. Another claim. [P1]")
    assert [s.pid for s in out] == ["", "P1"]


def test_trailing_text_with_no_citation_is_reported():
    out = split_cited_sentences("A claim. [P1] An uncited afterthought.")
    assert [s.pid for s in out] == ["P1", ""]


def test_a_case_name_does_not_split_a_sentence():
    out = split_cited_sentences(
        "The Court followed K.S. Puttaswamy v. Union of India on privacy. [P3]")
    assert len(out) == 1 and out[0].pid == "P3"


# ==========================================================================
# 4. The verifier
# ==========================================================================


class _Idx:
    """Enough of PositionalIndex for the verifier: an idf."""

    def __init__(self, idfs):
        self._idfs = idfs

    def idf(self, term):
        return self._idfs.get(term, 3.0)


class _Ctx:
    def __init__(self, idx):
        self.index = idx


@pytest.fixture
def ctx():
    return _Ctx(_Idx({"the": 0.0, "of": 0.0, "a": 0.0}))


def test_vectors_are_unit_length(ctx):
    v = vector(["dying", "declaration", "conviction"], ctx)
    assert pytest.approx(sum(x * x for x in v.values()), rel=1e-9) == 1.0


def test_a_zero_idf_term_is_dropped(ctx):
    assert "the" not in vector(["the", "dying"], ctx)


def test_cosine_of_a_vector_with_itself_is_one(ctx):
    v = vector(["dying", "declaration"], ctx)
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_of_disjoint_vectors_is_zero(ctx):
    a = vector(["dying", "declaration"], ctx)
    b = vector(["arbitration", "award"], ctx)
    assert cosine(a, b) == 0.0


def test_the_verifier_catches_an_unsupported_sentence(ctx):
    """Without this the verifier could be vacuous: on the extractive path every
    sentence passes by construction."""
    pack = _pack(_para("P1", "the dying declaration may form the sole basis of "
                             "conviction if found reliable and voluntary"))
    brief = ("The arbitral tribunal lacked jurisdiction to award damages for "
             "breach of the concession agreement. [P1]")
    rep = verify(brief, pack, ctx)
    assert not rep.passed
    assert rep.flagged[0].cosine < config.BRIEF_COSINE_THRESHOLD
    assert "unsupported" in rep.flagged[0].reason


def test_the_verifier_catches_a_missing_citation(ctx):
    pack = _pack(_para())
    rep = verify("The dying declaration may form the sole basis of conviction.",
                 pack, ctx)
    assert not rep.passed
    assert "no citation" in rep.flagged[0].reason


def test_the_verifier_catches_a_citation_outside_the_pack(ctx):
    pack = _pack(_para("P1"))
    rep = verify("The dying declaration may form the basis of conviction. [P9]",
                 pack, ctx)
    assert not rep.passed
    assert "not in the evidence pack" in rep.flagged[0].reason


def test_the_verifier_names_the_paragraph_that_would_have_supported_it(ctx):
    pack = _pack(
        _para("P1", "the plaint was rejected under Order VII Rule 11 of the Code"),
        _para("P2", "a dying declaration may form the sole basis of conviction "
                    "where it is found to be reliable and voluntary"))
    brief = ("A dying declaration may form the sole basis of conviction where "
             "it is reliable. [P1]")
    rep = verify(brief, pack, ctx)
    assert not rep.passed
    f = rep.flagged[0]
    assert f.closest_pid == "P2"
    assert "probably the wrong citation" in f.reason


def test_support_uses_the_best_sentence_not_only_the_whole_paragraph(ctx):
    """Measured: a sentence quoted verbatim out of a long paragraph scored 0.111
    against its own source, because the paragraph vector is dominated by
    everything else in it."""
    long_para = ("The appeal arises from a conviction recorded by the trial "
                 "court and affirmed in appeal. The evidence included witness "
                 "testimony about a quarrel over land, the recovery of a weapon "
                 "and medical evidence of injuries. A dying declaration may "
                 "form the sole basis of conviction. The sentence of life "
                 "imprisonment was accordingly confirmed by the High Court "
                 "after considering the submissions of counsel on both sides.")
    pack = _pack(_para("P1", long_para))
    rep = verify("A dying declaration may form the sole basis of conviction. [P1]",
                 pack, ctx)
    s = rep.sentences[0]
    assert s.ok
    assert s.cosine_sent > s.cosine_para      # the whole paragraph dilutes it
    assert s.cosine == pytest.approx(s.cosine_sent)


# ==========================================================================
# 5. Composition and degradation
# ==========================================================================


def test_no_api_key_here_so_the_brief_is_extractive():
    ok, why = llm_available()
    assert not ok
    assert "no model API key" in why or "SDK is not installed" in why


def test_the_prompt_restricts_the_model_to_the_pack():
    from nyaya.brief.compose import SYSTEM_PROMPT

    assert "ONLY the numbered paragraphs" in SYSTEM_PROMPT
    assert "exactly one paragraph id" in SYSTEM_PROMPT
    pack = _pack(_para("P1", badge="overruled by a later judgment"))
    prompt = render_prompt(pack)
    assert "[P1]" in prompt and "TREATMENT WARNING" in prompt


@needs_index
def test_the_extractive_brief_is_verbatim_and_fully_grounded():
    b = build("dying declaration as the sole basis of conviction")
    assert b.composer == "extractive"
    assert b.text
    assert b.verification is not None
    assert b.verification.passed, [s.reason for s in b.verification.flagged]
    for s in b.verification.sentences:
        assert s.pid in b.pack.pids


@needs_index
def test_the_brief_says_which_composer_wrote_it_and_why():
    b = build("anticipatory bail in economic offences")
    assert b.composer == "extractive"
    assert any("extractive" in n or "API key" in n for n in b.notes)


@needs_index
def test_the_brief_length_is_within_the_specified_range():
    from nyaya.brief.compose import MAX_SENTENCES, MIN_SENTENCES

    b = build("Section 482 CrPC quashing of criminal proceedings")
    n = len(b.verification.sentences)
    assert MIN_SENTENCES <= n <= MAX_SENTENCES


@needs_index
def test_every_sentence_ends_with_exactly_one_citation():
    b = build("dying declaration as the sole basis of conviction")
    for s in b.verification.sentences:
        assert CITE_RE.search(s.text), s.text
        assert s.text.count("[P") == 1, s.text


def test_a_query_that_retrieves_nothing_degrades_quietly():
    b = build('zzzqqq AND act:NOSUCHACT')
    assert b.text == ""
    assert b.notes
    assert "no brief" in render(b)


@needs_index
def test_render_shows_the_sources_and_the_grounding_check():
    out = render(build("dying declaration as the sole basis of conviction"))
    for fragment in ("PRECEDENT BRIEF", "sources", "grounding check",
                     "threshold", "composer"):
        assert fragment in out


@needs_index
def test_the_brief_serialises():
    b = build("dying declaration as the sole basis of conviction")
    blob = json.dumps(b.as_json(), default=str)
    assert "paragraphs" in blob and "verification" in blob


@needs_index
def test_the_cli_brief_flag_runs(capsys):
    from nyaya import cli

    assert cli.main(["--brief", "dying declaration as the sole basis of "
                                "conviction"]) == 0
    assert "PRECEDENT BRIEF" in capsys.readouterr().out
