"""Tests for the corpus-statistics module.

Two things worth pinning: the **fits** must recover a known power law from
synthetic data (otherwise a plausible-looking exponent in the report means
nothing), and the df table's stop-list column must be read from
`nyaya.stoplist` live, so the report can never quote a decision the code does
not actually take.

Network-free; the fits need no index.  Author: Sumedha (M3)
"""
from __future__ import annotations

import numpy as np
import pytest

from nyaya.index import stats as S
from nyaya.index.positional import PositionalIndex


# ==========================================================================
# 1. The fits recover a law we planted
# ==========================================================================


def test_heaps_fit_recovers_a_known_exponent():
    k, b = 42.0, 0.55
    T = np.logspace(3, 8, 40)
    V = k * T ** b
    fit = S.fit_heaps(T, V)
    assert fit["b"] == pytest.approx(b, abs=1e-6)
    assert fit["k"] == pytest.approx(k, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


def test_heaps_fit_survives_noise():
    rng = np.random.default_rng(358)
    T = np.logspace(3, 8, 60)
    V = 30.0 * T ** 0.5 * rng.lognormal(0, 0.03, T.size)
    fit = S.fit_heaps(T, V)
    assert fit["b"] == pytest.approx(0.5, abs=0.02)
    assert fit["r2"] > 0.99


def test_heaps_fit_refuses_to_guess_from_two_points():
    fit = S.fit_heaps([10, 100], [5, 20])
    assert np.isnan(fit["b"])


def test_heaps_predict_round_trips():
    fit = S.fit_heaps(np.logspace(3, 7, 30), 20 * np.logspace(3, 7, 30) ** 0.6)
    assert S.heaps_predict(fit, 1e6) == pytest.approx(20 * 1e6 ** 0.6, rel=1e-6)


def test_zipf_fit_recovers_a_known_exponent():
    rank = np.arange(1, 200_001)
    cf = np.maximum(1, (1e7 / rank ** 1.0)).astype(np.int64)
    fit = S.fit_zipf(cf, rank_lo=10, rank_hi=1000)
    assert fit["s"] == pytest.approx(1.0, abs=0.02)
    assert fit["r2"] > 0.999


def test_the_zipf_fit_range_is_reported_because_it_changes_the_answer():
    """The honesty requirement: quoting s without its range is meaningless."""
    rank = np.arange(1, 200_001)
    cf = np.maximum(1, (1e7 / rank ** 1.0)).astype(np.int64)
    narrow = S.fit_zipf(cf, rank_lo=10, rank_hi=1_000)
    wide = S.fit_zipf(cf, rank_lo=10, rank_hi=100_000)
    assert narrow["rank_hi"] == 1_000 and wide["rank_hi"] == 100_000
    # the quantised tail steepens the slope - the effect the docstring describes
    assert wide["s"] > narrow["s"]


def test_log_checkpoints_span_the_range_in_log_space():
    marks = sorted(S._log_checkpoints(10_000, 30))
    assert marks[0] == 1 and marks[-1] == 10_000
    # more than a third of the checkpoints below 1% of the corpus: that is the
    # point of log spacing, and linear spacing would put none there
    assert sum(1 for m in marks if m <= 100) >= len(marks) // 3


# ==========================================================================
# 2. Tables (against the built index when it exists)
# ==========================================================================
_HAS_INDEX = (S.config.INDEX / "dictionary.json").exists()
needs_index = pytest.mark.skipif(not _HAS_INDEX, reason="S3 not built")


@pytest.fixture(scope="module")
def index():
    return PositionalIndex.load()


@needs_index
def test_zone_table_reads_the_index_rather_than_recomputing(index):
    rows = S.zone_table(index)
    from nyaya import config

    assert [r["zone"] for r in rows] == list(config.ZONES)
    assert sum(r["tokens"] for r in rows) == int(index.zone_len[index.doc_len > 0].sum())
    body = next(r for r in rows if r["zone"] == "body")
    assert body["share"] > 0.5            # the body is where the text is
    assert body["weight"] == config.ZONE_WEIGHTS["body"]


@needs_index
def test_the_df_table_reads_its_decisions_live_from_the_stop_list(index):
    from nyaya import stoplist as sl

    rows = S.df_table(index, 30)
    assert len(rows) == 30
    assert rows == sorted(rows, key=lambda r: -r["df"])
    for r in rows:
        stopped = r["decision"].startswith("stop")
        assert stopped == sl.is_scoring_stopword(r["term"]), r
        assert r["why"]


@needs_index
def test_every_measured_domain_stop_word_still_clears_the_threshold(index):
    """The table in `stoplist` is quoted in the report; it must match the index
    it was measured from."""
    from nyaya import stoplist as sl

    N = index.doc_count
    stale = [n for _, (_, n) in sl.DOMAIN_DF.items() if n != N]
    if stale:
        pytest.skip(f"the index was rebuilt (N={N}, table says {stale[0]}); "
                    "re-run `python -m nyaya.stoplist --derive --source jsonl`")
    for term, (df, n) in sl.DOMAIN_DF.items():
        assert index.df(term) == df, term
        assert df / N >= sl.DOMAIN_DF_THRESHOLD, term


@needs_index
def test_size_table_covers_the_stores_that_exist(index):
    sizes = S.size_table()
    assert "S3 postings.bin" in sizes
    assert sizes["total_bytes"] > 0


@needs_index
def test_profile_comparison_isolates_what_it_claims_to(index):
    cmp = S.profile_comparison()
    if cmp is None:
        pytest.skip("the B0 baseline index is not built")
    # canonicalisation is the whole point: B0 must hold essentially no statute
    # tokens, the full index must hold thousands
    assert cmp["statute_tokens_in_full"] > 1000
    assert cmp["statute_tokens_in_baseline"] < 50
    assert "not stemming" in cmp["note"]


# ==========================================================================
# 3. The growth scan and the ablations
# ==========================================================================


@needs_index
@pytest.mark.skipif(not (S.config.INDEX / "paragraphs.off.npz").exists(),
                    reason="S1 not built")
def test_growth_scan_produces_a_monotone_curve_and_both_ablations():
    g = S.growth_scan(limit=12, checkpoints=6, verbose=False)
    curve = g["curve"]
    assert len(curve) >= 3
    for a, b in zip(curve, curve[1:]):
        assert b["documents"] > a["documents"]
        assert b["tokens"] >= a["tokens"]
        assert b["vocab_stemmed"] >= a["vocab_stemmed"]
        assert b["vocab_unstemmed"] >= a["vocab_unstemmed"]
    # stemming can only merge terms, never create them
    assert g["vocab_stemmed"] <= g["vocab_unstemmed"]
    assert 0 <= g["stemming_ablation"]["reduction"] < 1
    # act inheritance can only find more statute mentions, never fewer
    ia = g["inherit_act_ablation"]
    assert ia["statute_tokens_on"] >= ia["statute_tokens_off"]
    assert ia["distinct_on"] >= ia["distinct_off"]


@needs_index
def test_the_growth_sample_is_drawn_across_the_whole_corpus_not_the_first_years():
    """doc_id order is year order, so an unshuffled head would be the 1950s."""
    from nyaya.index.paragraphs import ParagraphStore
    import random

    from nyaya import config

    store = ParagraphStore.load()
    live = [d for d in range(store._doc_row.size) if store.has(d)]
    store.close()
    shuffled = list(live)
    random.Random(config.SAMPLE_SEED).shuffle(shuffled)
    head = shuffled[:500]
    assert max(head) > 0.8 * max(live)      # it reaches the recent years
    assert min(head) < 0.2 * max(live)      # and the early ones


# ==========================================================================
# 4. The figures render
# ==========================================================================


def test_figures_render_without_an_index(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    growth = {
        "documents": 50, "tokens": 100_000,
        "curve": [{"documents": d, "tokens": d * 2000,
                   "vocab_stemmed": int(30 * (d * 2000) ** 0.55),
                   "vocab_unstemmed": int(34 * (d * 2000) ** 0.56)}
                  for d in (1, 2, 5, 10, 20, 50)],
    }
    p = S.plot_heaps(growth, index_vocab=500_000, index_tokens=80_000_000,
                     out=tmp_path / "h.png")
    assert p and p.exists() and p.stat().st_size > 5_000

    zones = [{"zone": "body", "tokens": 1000, "share": 0.9, "docs_with_zone": 10,
              "mean_len": 100.0, "median_len": 90, "p95_len": 200, "max_len": 300,
              "weight": 1.0, "bm25f_b": 0.75},
             {"zone": "title", "tokens": 100, "share": 0.1, "docs_with_zone": 10,
              "mean_len": 10.0, "median_len": 9, "p95_len": 15, "max_len": 20,
              "weight": 3.0, "bm25f_b": 0.3}]
    assert S.plot_zones(zones, out=tmp_path / "z.png").exists()

    df_rows = [{"term": f"t{i}", "df": 100 - i, "df_over_N": (100 - i) / 100,
                "idf": 0.1, "decision": "stop (domain)" if i % 2 else "kept",
                "why": "x"} for i in range(20)]
    assert S.plot_df_top(df_rows, out=tmp_path / "d.png", n=20).exists()


def test_plot_heaps_declines_rather_than_fitting_two_points(tmp_path):
    assert S.plot_heaps({"curve": [{"documents": 1, "tokens": 10,
                                    "vocab_stemmed": 5, "vocab_unstemmed": 6}]},
                        out=tmp_path / "h.png") is None
