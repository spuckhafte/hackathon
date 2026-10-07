"""The report's charts  (Member 1, wave 4).

Three figures, each answering one question a marker will ask:

1. ``trackA_table.png``      - the systems x metrics table, as a figure so the
                               report and the slide show the same numbers
2. ``trackA_ndcg.png``       - nDCG@10 by system, the ablation ladder
3. ``trackA_per_query.png``  - per-query S4 - B0, sorted, with improved / tied /
                               worsened counted

The third is the important one and it is deliberately not a single bar.  With 20
test queries a mean difference of a few points is not evidence; the distribution
of per-query differences is, and a reader can see at a glance whether the gain is
broad or comes from one query.

All three read ``runs/trackA.json``, so they cannot disagree with the tables.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from nyaya import config
from nyaya.eval import metrics as M

#: A small, colour-blind-safe set.  Order matters: it is the ablation order, so
#: the ladder reads left to right as the system gets more capable.
SYSTEM_ORDER = ("B0", "S1", "S2", "S2p", "S3", "S4")
IMPROVED = "#2b6cb0"
WORSENED = "#c53030"
TIED = "#a0aec0"
INK = "#1a202c"
GRID = "#e2e8f0"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")                 # no display on a build machine
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    })
    return plt


def _ordered(systems: Sequence[str]) -> List[str]:
    known = [s for s in SYSTEM_ORDER if s in systems]
    return known + [s for s in systems if s not in SYSTEM_ORDER]


def table_figure(report: Dict[str, Any], out: Optional[Path] = None) -> Path:
    """The systems x metrics table, rendered so the report cannot mistype it."""
    plt = _mpl()
    systems = _ordered(report["systems"])
    metrics = list(M.METRIC_NAMES)
    cells = [[f"{report['summary'][s].get(m, float('nan')):.4f}" for m in metrics]
             + [f"{report['efficiency'].get(s, {}).get('median_ms', 0):.1f}"]
             for s in systems]
    best = {m: max(report["summary"][s].get(m, 0.0) for s in systems)
            for m in metrics}

    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(systems) + 1.1))
    ax.axis("off")
    # bbox rather than loc="center": matplotlib sizes a table from its own font
    # metrics and leaves the rest of the axes empty, which printed as a small
    # table marooned in half a page of white.  The explicit bbox makes it fill
    # the axes, so the figure's height is the table's height.
    tbl = ax.table(cellText=cells, rowLabels=systems,
                   colLabels=metrics + ["median ms"],
                   cellLoc="center", bbox=[0.055, 0.0, 0.945, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j, m in enumerate(metrics):
        for i, s in enumerate(systems):
            val = report["summary"][s].get(m, float("nan"))
            if abs(val - best[m]) < 1e-9:
                tbl[(i + 1, j)].set_facecolor("#ebf4ff")
                tbl[(i + 1, j)].set_text_props(weight="bold")
    ax.set_title(f"Track A: retrieval effectiveness  "
                 f"({report['split']} split, "
                 f"{report['summary'][systems[0]]['n_queries']:.0f} queries, "
                 f"k={report['retrieve_k']}).  Best per column highlighted.",
                 fontsize=9, pad=14)
    out = Path(out or config.RUNS / "trackA_table.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def ndcg_figure(report: Dict[str, Any], out: Optional[Path] = None) -> Path:
    """nDCG@10 by system: the ablation ladder."""
    plt = _mpl()
    systems = _ordered(report["systems"])
    vals = [report["summary"][s].get("nDCG@10", 0.0) for s in systems]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    bars = ax.bar(systems, vals, color=["#cbd5e0"] * len(systems), width=0.62)
    top = max(range(len(vals)), key=lambda i: vals[i])
    bars[top].set_color(IMPROVED)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.set_ylim(0, max(1.0, max(vals) * 1.18))
    ax.set_ylabel("nDCG@10")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"nDCG@10 by system ({report['split']} split, "
                 f"{report['summary'][systems[0]]['n_queries']:.0f} queries)",
                 fontsize=9)
    out = Path(out or config.RUNS / "trackA_ndcg.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def per_query_figure(report: Dict[str, Any], metric: str = "nDCG@10",
                     out: Optional[Path] = None) -> Optional[Path]:
    """Per-query difference, sorted, so the reader sees the distribution.

    The chart the report actually needs: a mean that moves because one query
    moved a lot is not an improvement, and this is the figure that shows which
    of the two happened.
    """
    comp = next((c for c in report["comparisons"] if c["metric"] == metric), None)
    if comp is None or not comp["per_query"]:
        return None
    plt = _mpl()
    rows = comp["per_query"]
    labels = [r["qid"] for r in rows]
    deltas = [r["delta"] for r in rows]
    colours = [IMPROVED if d > 1e-9 else WORSENED if d < -1e-9 else TIED
               for d in deltas]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.bar(labels, deltas, color=colours, width=0.68)
    ax.axhline(0, color=INK, linewidth=0.9)
    bs = comp["bootstrap"]
    ax.axhline(comp["mean_a"] - comp["mean_b"], color="#2f855a",
               linestyle="--", linewidth=1.0,
               label=f"mean {comp['mean_a'] - comp['mean_b']:+.3f}")
    ax.axhspan(bs["lo"], bs["hi"], color="#2f855a", alpha=0.10,
               label=f"95% CI [{bs['lo']:+.3f}, {bs['hi']:+.3f}]")
    c = comp["counts"]
    st = comp["sign_test"]
    ax.set_ylabel(f"{metric}: {comp['a_system']} - {comp['b_system']}")
    ax.set_title(
        f"{comp['a_system']} vs {comp['b_system']} per query - "
        f"{c['improved']} improved, {c['tied']} tied, {c['worsened']} worsened; "
        f"sign test p = {st['p_value']:.4f}"
        f"{' (significant)' if st['significant_at_05'] else ' (not significant)'}",
        fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    out = Path(out or config.RUNS / "trackA_per_query.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def all_plots(report: Dict[str, Any]) -> List[Path]:
    made = [table_figure(report), ndcg_figure(report)]
    pq = per_query_figure(report)
    if pq is not None:
        made.append(pq)
    return made


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from nyaya.utils import ensure_utf8_stdout

    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m nyaya.eval.plots",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--report", default=str(config.RUNS / "trackA.json"))
    args = ap.parse_args(argv)
    path = Path(args.report)
    if not path.exists():
        raise SystemExit(f"{path} missing - run python -m nyaya.eval.run first")
    report = json.loads(path.read_text(encoding="utf-8"))
    for p in all_plots(report):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
