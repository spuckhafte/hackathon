"""Tests for S1, the paragraph store.

The interesting property is not "it reads a line back" but **citation
integrity**: only 28% of these judgments print their own paragraph numbers, and
the store must never let the other 72% be rendered as if they did.

Network-free; builds a tiny store in a tmp_path.  Author: Sumedha (M3)
"""
from __future__ import annotations

import json

import pytest

from nyaya.contracts import Para
from nyaya.index import paragraphs as P
from nyaya.index.paragraphs import ParagraphStore


def _record(doc_id: int, paras, **kw):
    return {
        "doc_id": doc_id,
        "source_path": f"doc_{doc_id}",
        "title": kw.get("title", f"Doc {doc_id}"),
        "paras": [{"id": i + 1, "zone": z, "text": t, "label": lab}
                  for i, (z, t, lab) in enumerate(paras)],
        "zones": {},
    }


@pytest.fixture
def store(tmp_path):
    """Three documents: printed numbering, window numbering, and a split."""
    rows = [
        # doc 0 — the court printed its own numbers, and one long paragraph was
        # split so the tail carries no label of its own.
        _record(0, [
            ("body", "Headnote text before any numbering.", ""),
            ("body", "The first numbered paragraph.", "1"),
            ("body", "The second numbered paragraph, part one.", "2"),
            ("body", "The second numbered paragraph, part two.", ""),
            ("order", "The appeal is dismissed.", "3"),
        ]),
        # doc 1 — no printed numbers anywhere: every para is a window.
        _record(1, [
            ("body", "Window one about privacy and dignity.", ""),
            ("body", "Window two about Section 302 of the Penal Code.", ""),
            ("order", "In the result the appeal is allowed.", ""),
        ]),
        # doc 7 — a gap in doc_id, to prove the store does not assume density.
        _record(7, [("body", "Only paragraph.", "")]),
    ]
    src = tmp_path / "judgments.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    P.build(src=src, out=tmp_path / "idx", verbose=False)
    s = ParagraphStore.load(tmp_path / "idx")
    yield s
    s.close()


# ==========================================================================
# 1. Random access
# ==========================================================================


def test_sizes(store):
    assert store.doc_count == 3
    assert store.para_count == 9
    assert store.n_paras(0) == 5
    assert store.n_paras(1) == 3
    assert store.n_paras(7) == 1


def test_get_is_one_based_and_returns_contract_paras(store):
    p = store.get(0, 2)
    assert isinstance(p, Para)
    assert p.id == 2 and p.label == "1"
    assert p.text == "The first numbered paragraph."
    assert p.zone == "body"


def test_get_out_of_range_is_none_not_an_exception(store):
    assert store.get(0, 0) is None
    assert store.get(0, 99) is None
    assert store.get(123456, 1) is None
    assert store.get(2, 1) is None            # doc_id 2 was never written
    assert store.ref(2, 1) is None


def test_sparse_doc_ids_are_handled(store):
    """doc_id 2..6 do not exist; doc 7 must still resolve."""
    assert not store.has(2)
    assert store.has(7)
    assert store.get(7, 1).text == "Only paragraph."


def test_paras_of_returns_the_whole_document_in_order(store):
    paras = store.paras_of(0)
    assert [p.id for p in paras] == [1, 2, 3, 4, 5]
    assert [p.zone for p in paras] == ["body"] * 4 + ["order"]


def test_paras_in_zone(store):
    assert [p.id for p in store.paras_in_zone(0, "order")] == [5]


def test_iter_paras_streams_everything(store):
    rows = list(store.iter_paras())
    assert len(rows) == 9
    assert {d for d, _ in rows} == {0, 1, 7}


def test_iter_paras_can_be_restricted_to_some_documents(store):
    """This is the Precedent Brief's access pattern: top-5 docs only."""
    rows = list(store.iter_paras([1, 7]))
    assert [(d, p.id) for d, p in rows] == [(1, 1), (1, 2), (1, 3), (7, 1)]


def test_text_helper(store):
    assert store.text(1, 1).startswith("Window one")
    assert store.text(1, 99) == ""


# ==========================================================================
# 2. Citation integrity — the point of the module
# ==========================================================================


def test_a_window_is_never_rendered_as_a_paragraph_number(store):
    """The 72% case.  "window 2 of 3", never "¶ 2"."""
    assert store.numbering(1) == P.WINDOW
    ref = store.ref(1, 2)
    assert ref.kind == P.WINDOW
    assert ref.is_printed is False
    assert ref.cite() == "window 2 of 3"
    assert "¶" not in ref.cite()
    assert "¶" not in ref.short()
    assert "no paragraph numbers" in ref.tooltip()


def test_a_printed_paragraph_cites_the_courts_own_label(store):
    assert store.numbering(0) == P.PRINTED
    ref = store.ref(0, 2)
    assert ref.kind == P.PRINTED
    assert ref.is_printed is True
    assert ref.cite() == "¶ 1"           # label "1", not para_id 2
    assert ref.short() == "¶1"


def test_the_tail_of_a_split_paragraph_says_cont(store):
    """para_id 4 is the second half of printed ¶ 2 — it must not claim to be a
    paragraph of its own, and it must not be silently dropped either."""
    ref = store.ref(0, 4)
    assert ref.kind == P.PRINTED_SPLIT
    assert ref.cite() == "¶ 2 (cont.)"
    assert ref.is_printed is True
    assert "split" in ref.tooltip()


def test_an_unlabelled_paragraph_before_the_numbering_starts(store):
    """The SCR headnote sits before ¶ 1 and inherits nothing."""
    ref = store.ref(0, 1)
    assert ref.kind == P.PRINTED_SPLIT
    assert ref.label == ""
    assert ref.cite() == "para 1 (unnumbered)"


def test_para_id_and_printed_label_genuinely_diverge(store):
    """If they never diverged the distinction would not be worth the code."""
    assert store.get(0, 2).label == "1"
    assert store.find_label(0, "1") == 2
    assert store.find_label(0, "2") == 3
    assert store.find_label(0, "¶ 3") == 5
    assert store.find_label(0, "99") is None


def test_find_label_on_a_window_document_finds_nothing(store):
    assert store.find_label(1, "2") is None


def test_ref_for_avoids_a_second_read(store):
    p = store.get(0, 2)
    assert store.ref_for(0, p) == store.ref(0, 2)


# ==========================================================================
# 3. The header is the honest record
# ==========================================================================


def test_header_records_the_printed_share_and_warns(store):
    h = store.header
    assert h["documents"] == 3
    assert h["paragraphs"] == 9
    assert h["docs_with_printed_numbers"] == 1
    assert h["paragraphs_with_printed_label"] == 3
    assert "Never render a window as a paragraph number" in h["note"]
    assert h["zone_counts"] == {"body": 7, "order": 2}


def test_stats_are_reportable(store):
    st = store.stats()
    assert st["documents"] == 3 and st["paragraphs"] == 9
    assert st["printed_share"] == pytest.approx(1 / 3, abs=0.01)
    assert st["resident_bytes"] < 100_000       # offsets only, never the text


def test_store_is_a_context_manager(tmp_path, store):
    with ParagraphStore.load(store.path) as s:
        assert s.get(0, 1) is not None


def test_exists_guard(tmp_path):
    assert not ParagraphStore.exists(tmp_path / "nothing")


# ==========================================================================
# 4. Against the real corpus, when it is built
# ==========================================================================


@pytest.mark.skipif(not ParagraphStore.exists(), reason="S1 not built")
def test_real_store_matches_the_corpus_and_reads_randomly():
    import random

    from nyaya import config
    from nyaya.utils import count_lines

    s = ParagraphStore.load()
    assert s.para_count == s.header["paragraphs"]
    assert s.doc_count <= count_lines(config.JUDGMENTS_JSONL)
    # the measured split that the whole citation design exists for
    assert 0.2 < s.stats()["printed_share"] < 0.4

    rng = random.Random(358)
    live = [d for d in range(s._doc_row.size) if s.has(d)]
    for _ in range(200):
        d = rng.choice(live)
        pid = rng.randint(1, s.n_paras(d))
        p = s.get(d, pid)
        assert p is not None and p.id == pid
        ref = s.ref_for(d, p)
        assert ref.kind in (P.PRINTED, P.PRINTED_SPLIT, P.WINDOW)
        if ref.kind == P.WINDOW:
            assert "¶" not in ref.cite()
    s.close()
