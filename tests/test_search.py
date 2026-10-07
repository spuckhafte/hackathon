"""Tests for `search()` and the CLI.

`search()` is the single function the evaluation harness and the user interface
both call.  The properties worth pinning are that **all six modes actually run**
(an ablation that silently falls back to one implementation measures nothing),
that the mode table and not an if-tree decides what each mode does, and that the
"why this result" arithmetic reproduces the score the system printed.

Most of these need the built index and skip without it.  Author: Sumedha (M3)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nyaya import config
from nyaya.contracts import MODES, ParsedQuery, Result
from nyaya.rank.explain import verify
from nyaya.search import MODE_SETTINGS, SearchReport, load_context, search

_BUILT = (config.INDEX / "dictionary.json").exists()
needs_index = pytest.mark.skipif(not _BUILT, reason="S3 not built")


# ==========================================================================
# 1. The mode table
# ==========================================================================


def test_every_contract_mode_has_a_row():
    assert set(MODE_SETTINGS) == set(MODES)


def test_the_modes_differ_in_exactly_the_documented_way():
    m = MODE_SETTINGS
    assert m["B0"].baseline_index and not m["S1"].baseline_index
    assert m["B0"].scorer == "lnc.ltc"
    assert m["S1"].scorer == "uniform-zone"      # full index, flat weights
    assert m["S2"].scorer == "weighted-zone"
    assert m["S2p"].scorer == "bm25f"
    assert not m["S2p"].authority and m["S3"].authority
    assert not m["S3"].treatment and m["S4"].treatment
    assert not m["S3"].proximity and m["S4"].proximity


def test_a_mode_without_authority_puts_all_the_weight_on_text():
    for name in ("B0", "S1", "S2", "S2p"):
        assert MODE_SETTINGS[name].alpha == 1.0
        assert MODE_SETTINGS[name].beta == 0.0
    assert MODE_SETTINGS["S3"].alpha == config.ALPHA
    assert MODE_SETTINGS["S4"].beta == config.BETA


def test_only_b0_reads_a_different_index():
    """S1 is the full index with flat zone weights, not a third 700 MB build."""
    assert sum(1 for s in MODE_SETTINGS.values() if s.baseline_index) == 1


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        search("privacy", mode="S9")


# ==========================================================================
# 2. End to end
# ==========================================================================


@needs_index
@pytest.mark.parametrize("mode", list(MODES))
def test_every_mode_returns_results_for_a_free_text_query(mode):
    rep = SearchReport()
    out = search("right to privacy of personal data", k=5, mode=mode,
                 explain=False, report=rep)
    assert out and all(isinstance(r, Result) for r in out)
    assert rep.path == "free-text"
    assert rep.candidates > 0
    assert rep.scorer == MODE_SETTINGS[mode].scorer
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)


@needs_index
def test_the_modes_do_not_all_return_the_same_ranking():
    """If they did, the ablation table would be measuring nothing."""
    rankings = {}
    for mode in MODES:
        out = search("right to privacy of personal data", k=5, mode=mode,
                     explain=False)
        rankings[mode] = tuple(r.doc_id for r in out)
    assert len(set(rankings.values())) >= 3


@needs_index
def test_an_operator_query_goes_through_the_planner():
    rep = SearchReport()
    out = search('privacy /p "fundamental right"', k=5, mode="S4",
                 explain=False, report=rep)
    assert rep.path == "boolean"
    assert rep.plan is not None
    assert out


@needs_index
def test_a_filters_only_query_ranks_by_authority_alone():
    """There is no text to score, so ordering it by a text score of zero would
    be arbitrary."""
    rep = SearchReport()
    out = search('act:NDPS type:"criminal appeal" disposal:allowed', k=5,
                 mode="S4", explain=False, report=rep)
    assert rep.path == "filters-only"
    assert out
    gs = load_context("S4").gscore
    assert [round(r.score, 9) for r in out] == \
        [round(gs.g(r.doc_id), 9) for r in out]


@needs_index
def test_a_malformed_query_returns_cleanly_rather_than_raising():
    rep = SearchReport()
    out = search('unbalanced "quote AND stray ) /x', k=5, mode="S4",
                 explain=False, report=rep)
    assert isinstance(out, list)
    assert rep.warnings


@needs_index
def test_an_empty_query_returns_nothing_without_raising():
    assert search("", k=5, mode="S4", explain=False) == []


@needs_index
def test_results_carry_the_metadata_the_ui_prints():
    out = search("basic structure doctrine", k=3, mode="S4", explain=False)
    r = out[0]
    assert r.title and r.year > 1900
    assert r.bench_size >= 1          # from the parametric column, not metadata
    assert r.neutral_cite or r.scr_cite


@needs_index
def test_only_the_authority_modes_carry_a_treatment_badge():
    for mode in ("B0", "S1", "S2", "S2p"):
        out = search("murder", k=5, mode=mode, explain=False)
        assert all(r.badge == "" and r.status == "ok" for r in out)


# ==========================================================================
# 3. The explanation reproduces the score
# ==========================================================================


@needs_index
@pytest.mark.parametrize("mode", ["B0", "S2", "S2p", "S4"])
def test_the_explanation_arithmetic_reproduces_the_printed_score(mode):
    out = search('privacy /p "fundamental right"', k=3, mode=mode, explain=True)
    assert out
    for r in out:
        assert r.explanation is not None
        v = verify(r.explanation)
        assert v["ok"], (mode, v)
        assert r.score == pytest.approx(r.explanation.net, abs=1e-9)


@needs_index
def test_the_explanation_names_the_scorer_that_actually_ran():
    for mode in MODES:
        out = search("dying declaration", k=1, mode=mode, explain=True)
        assert out[0].explanation.scorer == MODE_SETTINGS[mode].scorer


@needs_index
def test_a_statute_query_explains_its_crosswalk_expansion():
    out = search("S. 302 IPC dying declaration", k=1, mode="S4", explain=True)
    notes = " ".join(out[0].explanation.notes)
    assert "ipc§302 OR bns§103" in notes


# ==========================================================================
# 4. Context
# ==========================================================================


@needs_index
def test_the_context_is_cached_per_index_and_authority_weighting():
    a = load_context("S2p")
    b = load_context("S3")
    assert a.index is b.index          # same full index, loaded once
    c = load_context("B0")
    assert c.index is not a.index      # B0 is the baseline build


@needs_index
def test_the_context_bundles_the_stores_the_contract_names():
    ctx = load_context("S4")
    assert ctx.index is not None
    assert ctx.params is not None
    assert ctx.paras is not None       # S1, for snippets and the Brief
    assert ctx.gscore is not None
    assert ctx.docs is not None


# ==========================================================================
# 2b. S5 is wired into the live free-text path
# ==========================================================================

_S5 = (config.INDEX / "champions.json").exists() and (config.INDEX / "tiers.npz").exists()
needs_s5 = pytest.mark.skipif(not _S5, reason="S5 not built")


@needs_index
@needs_s5
def test_load_context_loads_s5_so_the_live_path_is_tiered():
    """The whole point of the wiring: `ctx.tiers` is no longer None.

    While it was None the measured champion-list speed-up existed only in M2's
    harness and not in the shipped system, which is the worst kind of number to
    put in a report.
    """
    ctx = load_context("S4")
    assert ctx.tiers is not None


@needs_index
@needs_s5
def test_a_free_text_query_records_which_tier_answered():
    ctx = load_context("S4")
    rep = SearchReport()
    search("basic structure of the constitution", k=10, mode="S4",
           explain=False, ctx=ctx, report=rep)
    assert rep.path == "free-text"
    assert rep.tier_trace is not None
    assert rep.tier_trace.tier in (1, 2)
    assert rep.tier_trace.reason
    # and it survives serialisation, because the UI and the report read as_json()
    assert rep.as_json()["tier_trace"]["tier"] == rep.tier_trace.tier


@needs_index
def test_a_boolean_query_has_no_tier_trace_rather_than_a_fake_one():
    """``None`` means "no tier decision was taken", not "tier 2"."""
    ctx = load_context("S4")
    rep = SearchReport()
    search('privacy AND year:>=2000', k=5, mode="S4", explain=False,
           ctx=ctx, report=rep)
    assert rep.path != "free-text"
    assert rep.tier_trace is None
    assert rep.as_json()["tier_trace"] is None


@needs_index
@needs_s5
def test_tiering_is_a_flag_not_a_fork():
    """With ``ctx.tiers = None`` the candidate set is the exhaustive union again.

    This is what lets the efficiency table report both arms from one code path:
    the tiered and the exhaustive runs differ by one attribute, not by a branch
    that could drift.
    """
    from nyaya.search import _free_text_candidates
    from nyaya.query.parser import parse

    ctx = load_context("S4")
    keep = ctx.tiers
    q = parse("basic structure of the constitution")
    terms = list(q.free_terms)
    try:
        ctx.tiers = None
        rep_off = SearchReport()
        search("basic structure of the constitution", k=10, mode="S4",
               explain=False, ctx=ctx, report=rep_off)
        exhaustive = _free_text_candidates(terms, ctx)
        assert rep_off.tier_trace is None
        assert rep_off.candidates == int(exhaustive.size)

        ctx.tiers = keep
        rep_on = SearchReport()
        search("basic structure of the constitution", k=10, mode="S4",
               explain=False, ctx=ctx, report=rep_on)
        assert rep_on.tier_trace is not None
        # a non-safe optimisation: fewer candidates, and the top-1 unchanged on
        # this query (the general fidelity cost is measured, not asserted here)
        assert rep_on.candidates < rep_off.candidates
    finally:
        ctx.tiers = keep


@needs_index
@needs_s5
def test_an_unreadable_s5_degrades_to_exhaustive_instead_of_failing(monkeypatch):
    """A corrupt champion store must not take the whole search down."""
    import nyaya.index.tiers as T
    from nyaya import search as S

    def boom(*a, **kw):
        raise ValueError("truncated tiers.npz")

    monkeypatch.setattr(T.TieredIndex, "load", classmethod(boom))
    S._CONTEXTS.clear()
    try:
        ctx = load_context("S4")
        assert ctx.tiers is None
        assert "unreadable" in getattr(ctx, "tier_note", "")
        rep = SearchReport()
        out = search("basic structure of the constitution", k=5, mode="S4",
                     explain=False, ctx=ctx, report=rep)
        assert out and any("unreadable" in w for w in rep.warnings)
    finally:
        S._CONTEXTS.clear()


@needs_index
def test_snippets_cite_paragraphs_honestly():
    from nyaya.search import snippets_for

    ctx = load_context("S4")
    out = search("dying declaration", k=3, mode="S4", explain=False)
    seen = 0
    for r in out:
        for s in snippets_for(r, ["die", "declar"], ctx, limit=1):
            seen += 1
            assert s.startswith("[")
            head = s[1:s.index("]")]
            assert head.startswith("¶") or head.startswith("window") \
                or head.startswith("para")
    assert seen            # the corpus must actually yield some evidence


# ==========================================================================
# 5. The CLI
# ==========================================================================


@needs_index
@pytest.mark.parametrize("argv", [
    ["--mode", "S4", "right to privacy of personal data", "-k", "3"],
    ["--why", "1", "--mode", "S4", 'privacy /p "fundamental right"'],
    ["--explain-plan", 'privacy /p "fundamental right" AND bench:>=5'],
    ["--analyze", "convicted u/s 302 IPC read with S. 34"],
    ["--parse", "bail! AND act:IPC year:>=2010"],
    ["--postings", "privacy", "--limit", "5"],
    ["--merge", 'privacy /p "fundamental right"', "--trace"],
    ["--mode", "S4", "privacy", "-k", "2", "--json"],
    ["--mode", "S4", "privacy", "-k", "2", "--snippets"],
])
def test_every_recorded_cli_command_exits_zero(argv, capsys):
    from nyaya import cli

    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.strip()


def test_the_cli_help_lists_the_recorded_flags(capsys):
    from nyaya import cli

    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for flag in ("--explain-plan", "--why", "--analyze", "--parse",
                 "--postings", "--merge", "--trace", "--treatment-audit",
                 "--smoke", "--mode"):
        assert flag in out


def test_an_unknown_mode_is_rejected_by_argparse():
    from nyaya import cli

    with pytest.raises(SystemExit):
        cli.main(["--mode", "S9", "privacy"])


@needs_index
def test_the_treatment_audit_fails_loudly_rather_than_faking_output(capsys):
    """Segment 17's narration is 'nothing here is a mock-up', so the flag must
    say the classifier has not landed instead of inventing rows."""
    from nyaya import cli

    rc = cli.main(["--treatment-audit", "--only-errors", "--limit", "3"])
    err = capsys.readouterr().err
    assert rc in (0, 2)
    if rc == 2:
        assert "has not landed" in err
        assert "python -m nyaya.statutes --measure" in err
