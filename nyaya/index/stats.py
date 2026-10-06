"""Corpus statistics and the ablations — the figures and tables for the report.

What this module is *not*
-------------------------
It is not a second implementation of the index.  `PositionalIndex` already holds
``df``, ``idf``, ``doc_len``, ``zone_len`` and ``doc_norm``; everything here reads
them.  The only thing recomputed is the one number the index cannot hold: the
**vocabulary growth curve**, which needs |V| at a sequence of collection prefixes
and therefore a second pass over the text.  That pass does double duty as the
stemming ablation, because stemming on and off are two views of the same scan.

What it produces
----------------
======================================  =================================================
``runs/heaps.png``                      vocabulary growth vs collection size, log-log,
                                        with the fitted Heaps' law and the stemming
                                        ablation as the second curve
``runs/zipf.png``                       rank vs collection frequency, log-log, with the
                                        fitted Zipf exponent
``runs/zone_lengths.png``               tokens per zone: where the text actually is
``runs/df_top50.png``                   the 50 highest-df terms, marked with the
                                        stop-list decision taken for each
``runs/corpus_stats.json``              every number in the figures, for the report text
======================================  =================================================

Author: Sumedha (M3)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nyaya import config
from nyaya.utils import human

# --------------------------------------------------------------------------
# Figure style.  Colours are the validated two-slot categorical palette; the
# ink and grid roles are the recessive chart-chrome set.  Axis labels use
# secondary ink rather than muted, because these figures are printed.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_1 = "#2a78d6"        # categorical slot 1 (blue)
SERIES_2 = "#eb6834"        # categorical slot 2 (orange)

GROWTH_JSON = "corpus_growth.json"
STATS_JSON = "corpus_stats.json"


# ==========================================================================
# 1. The one pass we do recompute: vocabulary growth + the ablations
# ==========================================================================


def growth_scan(limit: int = 0, checkpoints: int = 40,
                verbose: bool = True) -> Dict[str, object]:
    """Stream the paragraph store and record |V| against collection size.

    Returns the Heaps' law series for **stemming on and off together**, plus the
    statute-token yield with act inheritance on and off — four measurements from
    one scan, because the expensive part is `analyze()` and all four need it.

    Reads S1 rather than ``judgments.jsonl``: it is a sequential read of the same
    text (the body and order zones, 99.5% of the indexed tokens), and it is the
    store the snippets and the Brief read, so the numbers describe what the
    system actually searches.
    """
    from nyaya.analyze import _never_stem, analyze, stem_term
    from nyaya.index.paragraphs import ParagraphStore
    from nyaya.statutes import extract_statutes

    store = ParagraphStore.load()
    live = [d for d in range(store._doc_row.size) if store.has(d)]
    # Shuffle before truncating.  doc_id order is year order, so `live[:2500]`
    # would be the 1950s-70s only and the growth curve would describe the
    # vocabulary of one era.  Heaps' law is about a random stream of text, so a
    # seeded random subset is both the faster and the correct sample.
    import random

    random.Random(config.SAMPLE_SEED).shuffle(live)
    if limit:
        live = live[:limit]
    marks = _log_checkpoints(len(live), checkpoints)

    vocab_raw: set = set()      # un-stemmed: the "stemming off" arm
    vocab_stem: set = set()     # stemmed: the "stemming on" arm
    tokens = 0
    rows: List[Dict[str, int]] = []
    stat_on = stat_off = 0
    stat_vocab_on: set = set()
    stat_vocab_off: set = set()
    t0 = time.perf_counter()

    for i, doc_id in enumerate(live, 1):
        text = "\n\n".join(p.text for p in store.paras_of(doc_id))
        toks = analyze(text, stem=False)
        tokens += len(toks)
        for t in toks:
            vocab_raw.add(t.term)
            vocab_stem.add(t.term if _never_stem(t.term) else stem_term(t.term))
        on = extract_statutes(text, inherit_act=True)
        off = extract_statutes(text, inherit_act=False)
        stat_on += len(on)
        stat_off += len(off)
        stat_vocab_on.update(on)
        stat_vocab_off.update(off)
        if i in marks:
            rows.append({"documents": i, "tokens": tokens,
                         "vocab_stemmed": len(vocab_stem),
                         "vocab_unstemmed": len(vocab_raw)})
            if verbose:
                print(f"  {i:>6} docs  {tokens:>11,} tokens  "
                      f"|V| {len(vocab_stem):>7,} stemmed / "
                      f"{len(vocab_raw):>7,} unstemmed", file=sys.stderr)
    store.close()

    out = {
        "documents": len(live),
        "tokens": tokens,
        "seconds": round(time.perf_counter() - t0, 1),
        "curve": rows,
        "vocab_stemmed": len(vocab_stem),
        "vocab_unstemmed": len(vocab_raw),
        "stemming_ablation": {
            "vocab_stem_on": len(vocab_stem),
            "vocab_stem_off": len(vocab_raw),
            "reduction": round(1 - len(vocab_stem) / max(1, len(vocab_raw)), 4),
        },
        "inherit_act_ablation": {
            "statute_tokens_on": stat_on,
            "statute_tokens_off": stat_off,
            "distinct_on": len(stat_vocab_on),
            "distinct_off": len(stat_vocab_off),
            "share_from_inheritance": round(
                1 - stat_off / max(1, stat_on), 4),
        },
    }
    return out


def _log_checkpoints(n: int, k: int) -> set:
    """Checkpoints spread evenly in *log* space — Heaps' law is a power law, so
    linear checkpoints would crowd every point into the last decade."""
    if n <= 0:
        return set()
    lo = math.log10(1)
    hi = math.log10(n)
    marks = {int(round(10 ** x)) for x in np.linspace(lo, hi, k)}
    marks.add(n)
    return {m for m in marks if 1 <= m <= n}


# ==========================================================================
# 2. Fits
# ==========================================================================


def fit_heaps(tokens: Sequence[float], vocab: Sequence[float]
              ) -> Dict[str, float]:
    """Heaps' law ``|V| = k * T^b`` by least squares on the log-log points.

    ``b`` near 0.5 and ``k`` in the tens is the textbook result for English
    prose; a legal corpus usually sits a little lower in ``b`` because the
    vocabulary is formulaic, and a little higher in ``k``.
    """
    t = np.log10(np.asarray(tokens, dtype=float))
    v = np.log10(np.asarray(vocab, dtype=float))
    keep = np.isfinite(t) & np.isfinite(v)
    t, v = t[keep], v[keep]
    if t.size < 3:
        return {"k": float("nan"), "b": float("nan"), "r2": float("nan"), "n": int(t.size)}
    b, logk = np.polyfit(t, v, 1)
    pred = logk + b * t
    ss_res = float(((v - pred) ** 2).sum())
    ss_tot = float(((v - v.mean()) ** 2).sum())
    return {"k": float(10 ** logk), "b": float(b),
            "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"), "n": int(t.size)}


def heaps_predict(fit: Dict[str, float], tokens: float) -> float:
    return fit["k"] * tokens ** fit["b"]


def fit_zipf(freqs: np.ndarray, rank_lo: int = 10, rank_hi: int = 1_000
             ) -> Dict[str, float]:
    """Zipf ``cf = C / rank^s`` by least squares on the log-log middle.

    Both ends are excluded on purpose, and the choice changes the answer, so the
    range is reported with the exponent - a Zipf slope quoted without its range
    is not a measurement:

    * the **head** (rank < 10) is the handful of function words the stop list
      removes anyway, and they sit above the line;
    * the **tail** is quantised.  63% of this vocabulary occurs exactly once, so
      beyond rank ~10,000 the points lie on the flat shelves cf = 3, 2, 1.
      Measured on this index: fitting ranks 10-1,000 gives s = 0.98 (R2 = 0.993),
      the textbook value; dragging the fit out to rank 100,000 gives s = 1.68,
      which is a measurement of the quantisation, not of Zipf's law.
    """
    f = np.sort(freqs)[::-1]
    hi = min(rank_hi, f.size)
    rank = np.arange(1, f.size + 1, dtype=float)
    sl = slice(rank_lo - 1, hi)
    x = np.log10(rank[sl])
    y = np.log10(f[sl].astype(float))
    keep = np.isfinite(x) & np.isfinite(y) & (f[sl] > 1)
    x, y = x[keep], y[keep]
    if x.size < 10:
        return {"s": float("nan"), "C": float("nan"), "r2": float("nan")}
    slope, logC = np.polyfit(x, y, 1)
    pred = logC + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"s": float(-slope), "C": float(10 ** logC),
            "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
            "rank_lo": rank_lo, "rank_hi": int(hi)}


# ==========================================================================
# 3. Tables read straight off the index
# ==========================================================================


def zone_table(index) -> List[Dict[str, object]]:
    """Per-zone length statistics from ``index.zone_len`` — not recomputed."""
    zl = index.zone_len
    live = index.doc_len > 0
    rows = []
    total = float(zl[live].sum()) or 1.0
    for i, zone in enumerate(config.ZONES):
        col = zl[live, i].astype(float)
        present = col > 0
        rows.append({
            "zone": zone,
            "tokens": int(col.sum()),
            "share": round(float(col.sum()) / total, 4),
            "docs_with_zone": int(present.sum()),
            "mean_len": round(float(col[present].mean()), 1) if present.any() else 0.0,
            "median_len": int(np.median(col[present])) if present.any() else 0,
            "p95_len": int(np.percentile(col[present], 95)) if present.any() else 0,
            "max_len": int(col.max()),
            "weight": config.ZONE_WEIGHTS.get(zone, 1.0),
            "bm25f_b": config.BM25F_B.get(zone),
        })
    return rows


def df_table(index, n: int = 50) -> List[Dict[str, object]]:
    """The top-``n`` terms by df, each marked with the stop-list decision.

    This is the table the report uses to justify the measured domain stop list:
    the decision column is read live from `nyaya.stoplist`, so the table can
    never drift from the code that actually runs.
    """
    from nyaya import stoplist as sl

    N = index.doc_count
    rows = []
    for term, df in index.terms_by_df(n):
        if term in sl.DOMAIN_DF:
            decision, why = "stop (domain)", "measured df/N >= 0.80"
        elif sl.is_scoring_stopword(term):
            decision, why = "stop (standard)", "standard English list, or its stem"
        elif term in sl.DOMAIN_KEEP_REASONS:
            decision, why = "KEPT", sl.DOMAIN_KEEP_REASONS[term]
        else:
            decision, why = "kept", "discriminating despite high df"
        rows.append({
            "term": term, "df": int(df), "df_over_N": round(df / max(1, N), 4),
            "idf": round(index.idf(term), 4), "decision": decision, "why": why,
        })
    return rows


def size_table(path: Path = config.INDEX) -> Dict[str, object]:
    """On-disk footprint of every store, with the ratio to the raw text."""
    path = Path(path)
    files = {
        "S1 paragraphs.jsonl": path / "paragraphs.jsonl",
        "S1 paragraphs.off.npz": path / "paragraphs.off.npz",
        "S3 postings.bin": path / "postings.bin",
        "S3 dictionary.json": path / "dictionary.json",
        "S3 docmeta.npz": path / "docmeta.npz",
        "S2 parametric.json": path / "parametric.json",
        "S4 citations.jsonl": path / "citations.jsonl",
        "S4 authority.json": path / "authority.json",
    }
    out: Dict[str, object] = {}
    total = 0
    for label, f in files.items():
        if f.exists():
            size = f.stat().st_size
            total += size
            out[label] = human(size)
    out["total"] = human(total)
    out["total_bytes"] = total
    if config.JUDGMENTS_JSONL.exists():
        raw = config.JUDGMENTS_JSONL.stat().st_size
        out["judgments.jsonl"] = human(raw)
        out["index_to_text_ratio"] = round(total / raw, 2)
    return out


def profile_comparison(full: Path = config.INDEX,
                       baseline: Path = None) -> Optional[Dict[str, object]]:
    """The S1-vs-B0 ablation that is already on disk.

    M1 builds both indexes, so the effect of statute canonicalisation + the
    measured stop list + zoning on vocabulary and postings is a free read.  It is
    a *bundle*, not an isolated ablation: the stemming arm is separate, because
    both profiles stem.
    """
    from nyaya.index.positional import PositionalIndex

    baseline = Path(baseline or (config.INDEX / "baseline"))
    if not (baseline / "dictionary.json").exists():
        return None
    a = PositionalIndex.load(full)
    b = PositionalIndex.load(baseline)
    sa, sb = a.stats(), b.stats()
    return {
        "full": {k: sa[k] for k in
                 ("vocabulary", "total_postings", "total_positions", "hapax_terms")},
        "baseline": {k: sb[k] for k in
                     ("vocabulary", "total_postings", "total_positions", "hapax_terms")},
        "vocabulary_delta": sa["vocabulary"] - sb["vocabulary"],
        "postings_delta": sa["total_postings"] - sb["total_postings"],
        "statute_tokens_in_full": sum(1 for t in a._terms if "§" in t),
        "statute_tokens_in_baseline": sum(1 for t in b._terms if "§" in t),
        "note": ("the baseline keeps stemming on, so this row isolates "
                 "canonicalisation + the legal stop list + zoning, not stemming"),
    }


# ==========================================================================
# 4. Figures
# ==========================================================================


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, fontweight="600", pad=12, loc="left")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=11)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=11)
    ax.tick_params(colors=INK_2, labelsize=10, length=4, width=0.8)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def plot_heaps(growth: Dict, index_vocab: Optional[int] = None,
               index_tokens: Optional[int] = None,
               out: Path = None) -> Optional[Path]:
    """Vocabulary growth, log-log, with the Heaps fit and the stemming ablation."""
    import matplotlib.pyplot as plt

    curve = growth.get("curve") or []
    if len(curve) < 3:
        return None
    T = [r["tokens"] for r in curve]
    Vs = [r["vocab_stemmed"] for r in curve]
    Vu = [r["vocab_unstemmed"] for r in curve]
    fs, fu = fit_heaps(T, Vs), fit_heaps(T, Vu)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    _style(ax, "Heaps' law: vocabulary growth over the indexed corpus",
           "collection size T (tokens, log scale)",
           "vocabulary |V| (distinct terms, log scale)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    grid = np.logspace(math.log10(min(T)), math.log10(max(T) * 1.15), 60)
    ax.plot(grid, [heaps_predict(fu, t) for t in grid], color=SERIES_2,
            linewidth=2, linestyle=(0, (5, 3)), zorder=2, alpha=0.9)
    ax.plot(grid, [heaps_predict(fs, t) for t in grid], color=SERIES_1,
            linewidth=2, linestyle=(0, (5, 3)), zorder=2, alpha=0.9)
    ax.plot(T, Vu, "o", color=SERIES_2, markersize=6.5, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.4,
            label=f"stemming OFF   |V| = {fu['k']:.1f}·T^{fu['b']:.3f}")
    ax.plot(T, Vs, "o", color=SERIES_1, markersize=6.5, zorder=5,
            markeredgecolor=SURFACE, markeredgewidth=1.4,
            label=f"stemming ON    |V| = {fs['k']:.1f}·T^{fs['b']:.3f}")

    if index_vocab and index_tokens:
        ax.plot([index_tokens], [index_vocab], "*", color=INK, markersize=16,
                zorder=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=f"the built index: {index_vocab:,} terms")
        pred = heaps_predict(fs, index_tokens)
        ax.annotate(
            f"fit extrapolates to {pred:,.0f}\n({pred / index_vocab - 1:+.0%} vs built)",
            xy=(index_tokens, index_vocab), xytext=(-12, -58),
            textcoords="offset points", ha="right", fontsize=9.5, color=INK_2,
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.9))

    leg = ax.legend(loc="upper left", frameon=False, fontsize=10,
                    labelcolor=INK_2, handletextpad=0.7)
    for t in leg.get_texts():
        t.set_color(INK_2)
    ax.text(0.99, 0.02,
            f"R² = {fs['r2']:.4f} (stemmed), {fu['r2']:.4f} (unstemmed)   "
            f"{growth['documents']:,} judgments",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=INK_2)
    return _save(fig, Path(out or config.RUNS / "heaps.png"))


def plot_zipf(index, out: Path = None) -> Path:
    """Rank vs collection frequency, log-log, with the fitted exponent."""
    import matplotlib.pyplot as plt

    cf = np.asarray(index._npos, dtype=np.int64)
    cf = cf[cf > 0]
    order = np.sort(cf)[::-1]
    rank = np.arange(1, order.size + 1)
    fit = fit_zipf(order)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    _style(ax, "Zipf's law: term rank against collection frequency",
           "rank (log scale)", "collection frequency cf (log scale)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Thin the scatter *logarithmically*: 544k points print as mud, but uniform
    # thinning deletes the head, which is the half of the curve worth seeing.
    keep = np.unique(np.concatenate([
        np.arange(0, min(400, order.size)),
        np.unique(np.logspace(math.log10(400), math.log10(order.size),
                              3000).astype(np.int64)) - 1,
    ]))
    keep = keep[(keep >= 0) & (keep < order.size)]
    ax.plot(rank[keep], order[keep], ".", color=SERIES_1, markersize=3.4,
            alpha=0.8, zorder=3)
    lo, hi = fit["rank_lo"], fit["rank_hi"]
    xs = np.logspace(math.log10(lo), math.log10(hi), 50)
    ax.plot(xs, fit["C"] / xs ** fit["s"], color=SERIES_2, linewidth=2.2,
            linestyle=(0, (5, 3)), zorder=4,
            label=(f"cf = {fit['C']:,.0f} / rank^{fit['s']:.3f}   "
                   f"(R² = {fit['r2']:.3f}, ranks "
                   f"{fit['rank_lo']:,}-{fit['rank_hi']:,})"))
    ax.axvspan(1, lo, color=GRID, alpha=0.55, zorder=1)
    ax.axvspan(hi, order.size, color=GRID, alpha=0.55, zorder=1)
    ax.annotate("head:\nfunction words\nthe stop list\nremoves", xy=(1.3, 3),
                fontsize=8.5, color=INK_2, ha="left", va="bottom")
    tail_fit = fit_zipf(freqs=order, rank_lo=lo, rank_hi=order.size)
    ax.annotate(f"tail: cf quantised at 3, 2, 1.\nFitting through it gives "
                f"s = {tail_fit['s']:.2f},\nwhich measures the quantisation,\n"
                f"not the law.",
                xy=(hi * 1.6, order[0] * 0.06), fontsize=8.5, color=INK_2,
                ha="left", va="top")
    hapax = int((order == 1).sum())
    ax.text(0.99, 0.95,
            f"{order.size:,} terms\n{hapax:,} appear once ({hapax / order.size:.0%})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK_2)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK_2)
    return _save(fig, Path(out or config.RUNS / "zipf.png"))


def plot_zones(rows: List[Dict], out: Path = None) -> Path:
    """Where the text actually is — one measure across six zones, so one hue."""
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["tokens"] > 0]
    labels = [r["zone"] for r in rows]
    tokens = [r["tokens"] for r in rows]
    order = np.argsort(tokens)
    labels = [labels[i] for i in order]
    tokens = [tokens[i] for i in order]
    rows = [rows[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    _style(ax, "Tokens per zone, and the weight each zone carries",
           "tokens in the zone across the whole corpus (log scale)", "")
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    bars = ax.barh(labels, tokens, color=SERIES_1, height=0.62, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
    for bar, r in zip(bars, rows):
        ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
                f"{r['tokens']:,}   ·   median {r['median_len']:,} tok/doc"
                f"   ·   weight {r['weight']:g}",
                va="center", fontsize=9.5, color=INK_2)
    ax.set_xlim(right=max(tokens) * 220)
    ax.tick_params(axis="y", labelsize=11)
    body = next((r for r in rows if r["zone"] == "body"), None)
    share = f"{body['share']:.0%}" if body else "most"
    ax.text(0.0, -0.19,
            f"The body zone holds {share} of the tokens; the short zones are "
            "weighted up (config.ZONE_WEIGHTS) precisely\nbecause a hit in a "
            "9-token title is worth more than a hit in a 3,644-token body. The "
            "bench zone is weight 0:\nit is filter-only, reachable through "
            "judge: but never scored.",
            transform=ax.transAxes, fontsize=9, color=INK_2, va="top")
    return _save(fig, Path(out or config.RUNS / "zone_lengths.png"))


def plot_df_top(rows: List[Dict], out: Path = None, n: int = 50) -> Path:
    """The 50 highest-df terms, coloured by the stop-list decision."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = rows[:n][::-1]
    labels = [r["term"] for r in rows]
    vals = [r["df_over_N"] for r in rows]
    stopped = [r["decision"].startswith("stop") for r in rows]
    colors = [SERIES_2 if s else SERIES_1 for s in stopped]

    fig, ax = plt.subplots(figsize=(7.2, 11.0))
    _style(ax, f"The {n} highest-df terms, and what the stop list does with each",
           "document frequency df / N", "")
    ax.grid(axis="y", visible=False)
    ax.barh(labels, vals, color=colors, height=0.72, zorder=3,
            edgecolor=SURFACE, linewidth=1.6)
    ax.set_xlim(0, 1.06)
    for y, r in enumerate(rows):
        ax.text(r["df_over_N"] + 0.012, y, f"{r['df_over_N']:.2f}",
                va="center", fontsize=8.5, color=INK_2)
    ax.tick_params(axis="y", labelsize=9)
    ax.axvline(0.80, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax.text(0.80, len(rows) + 0.4, "  domain stop threshold df/N = 0.80",
            fontsize=9, color=INK_2, va="bottom")
    legend = [Patch(facecolor=SERIES_2, label="excluded from tf-idf / BM25F "
                                              "(still in the positional index)"),
              Patch(facecolor=SERIES_1, label="kept as a scoring term")]
    # outside the axes: every bar here is near df/N = 1, so an inside legend
    # lands on top of the data whatever corner it is put in
    leg = ax.legend(handles=legend, loc="upper center", frameon=False,
                    fontsize=9.5, bbox_to_anchor=(0.5, -0.045), ncol=1,
                    handlelength=1.6, handleheight=1.1)
    for t in leg.get_texts():
        t.set_color(INK_2)
    return _save(fig, Path(out or config.RUNS / "df_top50.png"))


# ==========================================================================
# 5. Driver
# ==========================================================================


def collect(limit: int = 0, growth: bool = True, verbose: bool = True
            ) -> Dict[str, object]:
    """Everything, as one JSON-serialisable blob written to ``runs/``."""
    from nyaya.index.positional import PositionalIndex

    idx = PositionalIndex.load()
    cache = config.RUNS / GROWTH_JSON
    if growth:
        g = growth_scan(limit=limit, verbose=verbose)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(g, indent=1), encoding="utf-8")
    elif cache.exists():
        g = json.loads(cache.read_text(encoding="utf-8"))
    else:
        g = {"curve": [], "documents": 0, "tokens": 0}

    zones = zone_table(idx)
    dfs = df_table(idx, 50)
    live = idx.doc_len > 0
    out: Dict[str, object] = {
        "index": idx.stats(),
        "documents": idx.doc_count,
        "doc_len": {
            "mean": round(float(idx.doc_len[live].mean()), 1),
            "median": int(np.median(idx.doc_len[live])),
            "p05": int(np.percentile(idx.doc_len[live], 5)),
            "p95": int(np.percentile(idx.doc_len[live], 95)),
            "min": int(idx.doc_len[live].min()),
            "max": int(idx.doc_len[live].max()),
        },
        "zones": zones,
        "top_df": dfs,
        "sizes": size_table(),
        "profiles": profile_comparison(),
        "growth": {k: v for k, v in g.items() if k != "curve"},
        "heaps_stemmed": fit_heaps([r["tokens"] for r in g["curve"]],
                                   [r["vocab_stemmed"] for r in g["curve"]])
        if len(g.get("curve") or []) >= 3 else None,
        "heaps_unstemmed": fit_heaps([r["tokens"] for r in g["curve"]],
                                     [r["vocab_unstemmed"] for r in g["curve"]])
        if len(g.get("curve") or []) >= 3 else None,
        "zipf": fit_zipf(np.asarray(idx._npos, dtype=np.int64)),
    }

    figs = {}
    total_tokens = int(idx.doc_len[live].sum())
    p = plot_heaps(g, idx.vocab_size, total_tokens)
    if p:
        figs["heaps"] = str(p)
    figs["zipf"] = str(plot_zipf(idx))
    figs["zones"] = str(plot_zones(zones))
    figs["df_top50"] = str(plot_df_top(dfs))
    out["figures"] = figs

    (config.RUNS / STATS_JSON).write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def print_report(st: Dict[str, object]) -> None:
    """The text the report quotes, in the order the report needs it."""
    idx = st["index"]
    bar = "-" * 74
    print(f"\n{bar}\nCORPUS AND INDEX\n{bar}")
    print(f"  documents indexed        {st['documents']:,}")
    print(f"  tokens (positions)       {idx['total_positions']:,}")
    print(f"  vocabulary               {idx['vocabulary']:,}")
    print(f"  postings (term,doc)      {idx['total_postings']:,}")
    print(f"  terms appearing once     {idx['hapax_terms']:,} "
          f"({idx['hapax_terms'] / idx['vocabulary']:.1%} of the vocabulary)")
    dl = st["doc_len"]
    print(f"  document length          mean {dl['mean']:,} / median {dl['median']:,}"
          f" / p05 {dl['p05']:,} / p95 {dl['p95']:,} tokens")
    print(f"  paragraphs (S1)          {idx['paragraphs']:,}")
    print(f"  sentences                {idx['sentences']:,}")

    print(f"\n{bar}\nHEAPS' LAW\n{bar}")
    for arm, label in (("heaps_stemmed", "stemming ON "),
                       ("heaps_unstemmed", "stemming OFF")):
        f = st.get(arm)
        if f:
            print(f"  {label}  |V| = {f['k']:.2f} * T^{f['b']:.4f}   "
                  f"R2 = {f['r2']:.4f}  ({f['n']} checkpoints)")
    f = st.get("heaps_stemmed")
    if f:
        pred = heaps_predict(f, idx["total_positions"])
        print(f"  extrapolated to the full {idx['total_positions']:,} tokens: "
              f"{pred:,.0f} terms vs {idx['vocabulary']:,} actually built "
              f"({pred / idx['vocabulary'] - 1:+.1%})")

    z = st["zipf"]
    print(f"\n{bar}\nZIPF'S LAW\n{bar}")
    print(f"  cf = {z['C']:,.0f} / rank^{z['s']:.4f}   R2 = {z['r2']:.4f}  "
          f"(fitted over ranks {z['rank_lo']}-{z['rank_hi']:,})")

    print(f"\n{bar}\nABLATIONS\n{bar}")
    sa = (st.get("growth") or {}).get("stemming_ablation")
    if sa:
        print(f"  stemming        |V| {sa['vocab_stem_off']:,} off -> "
              f"{sa['vocab_stem_on']:,} on   "
              f"({sa['reduction']:.1%} smaller vocabulary)")
    ia = (st.get("growth") or {}).get("inherit_act_ablation")
    if ia:
        print(f"  act inheritance statute tokens {ia['statute_tokens_off']:,} off -> "
              f"{ia['statute_tokens_on']:,} on   "
              f"({ia['share_from_inheritance']:.1%} of statute mentions are only "
              "found by inheriting the act)")
        print(f"                  distinct statute terms "
              f"{ia['distinct_off']:,} -> {ia['distinct_on']:,}")
    pc = st.get("profiles")
    if pc:
        print(f"  full vs B0      vocabulary {pc['baseline']['vocabulary']:,} -> "
              f"{pc['full']['vocabulary']:,} ({pc['vocabulary_delta']:+,})")
        print(f"                  statute tokens in the dictionary "
              f"{pc['statute_tokens_in_baseline']:,} (B0) -> "
              f"{pc['statute_tokens_in_full']:,} (full)")
        print(f"                  note: {pc['note']}")

    print(f"\n{bar}\nZONES\n{bar}")
    print(f"  {'zone':<10}{'tokens':>13}{'share':>8}{'docs':>8}"
          f"{'median':>9}{'p95':>8}{'weight':>8}")
    for r in st["zones"]:
        print(f"  {r['zone']:<10}{r['tokens']:>13,}{r['share']:>8.1%}"
              f"{r['docs_with_zone']:>8,}{r['median_len']:>9,}{r['p95_len']:>8,}"
              f"{r['weight']:>8g}")

    print(f"\n{bar}\nTOP 50 TERMS BY DOCUMENT FREQUENCY\n{bar}")
    print(f"  {'#':>3} {'term':<18}{'df':>8}{'df/N':>7}{'idf':>8}  decision")
    for i, r in enumerate(st["top_df"], 1):
        print(f"  {i:>3} {r['term']:<18}{r['df']:>8,}{r['df_over_N']:>7.2f}"
              f"{r['idf']:>8.3f}  {r['decision']}")

    print(f"\n{bar}\nON DISK\n{bar}")
    for k, v in st["sizes"].items():
        if k != "total_bytes":
            print(f"  {k:<26}{v}")

    print(f"\n{bar}\nFIGURES\n{bar}")
    for k, v in (st.get("figures") or {}).items():
        print(f"  {k:<12}{v}")


def _main(argv: Optional[List[str]] = None) -> int:           # pragma: no cover
    from nyaya.utils import ensure_utf8_stdout

    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.index.stats")
    ap.add_argument("--limit", type=int, default=0,
                    help="documents in the growth scan (0 = the whole corpus)")
    ap.add_argument("--no-growth", action="store_true",
                    help="reuse the cached growth scan in runs/")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    st = collect(limit=args.limit, growth=not args.no_growth,
                 verbose=not args.quiet)
    print_report(st)
    print(f"\n  all numbers: {config.RUNS / STATS_JSON}")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(_main())
