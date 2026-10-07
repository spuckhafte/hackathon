"""Precision of the treatment classifier on 40 hand-labelled citations  (M1).

The gold labels live in ``eval/treatment_gold.json`` - 40 edges, 10 per predicted
label, sampled with ``random.Random(358)`` and read by hand.  This script
recomputes precision per label from them, so the number in the report is
reproducible rather than asserted.

    python scripts/treatment_audit.py
    python scripts/treatment_audit.py --show-errors
    python scripts/treatment_audit.py --limitation      # the video's beat
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyaya import config                                       # noqa: E402
from nyaya.authority import treatment as T                     # noqa: E402
from nyaya.utils import ensure_utf8_stdout                     # noqa: E402

GOLD = config.EVAL_DIR / "treatment_gold.json"
LABELS = (T.OVERRULED, T.DISTINGUISHED, T.FOLLOWED, T.MENTIONED)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--limitation", action="store_true",
                    help="print the documented negation failure")
    args = ap.parse_args(argv)

    gold_doc = json.loads(GOLD.read_text(encoding="utf-8"))
    gold: Dict[int, str] = {int(k): v for k, v in gold_doc["labels"].items()}

    if args.limitation:
        lim = gold_doc["limitation_case"]
        print("=" * 76)
        print("DOCUMENTED LIMITATION - negation scope is a character window")
        for k, v in lim.items():
            print(f"  {k}: {v}")
        print("\nreproduce with:")
        print(f"  python -m nyaya.authority.treatment explain --edge {lim['edge']}")
        return 0

    bench = T.build_bench_cache()
    predicted: Dict[int, str] = {}
    rules: Dict[int, str] = {}
    windows: Dict[int, str] = {}
    for i, row in T.iter_labelled():
        if i not in gold:
            continue
        src = int(row["src"])
        dst = int(row["dst"])
        bs = int(bench["doc_bench"][src]) if src < bench["doc_bench"].size else 0
        bd = int(bench["node_bench"][dst]) if 0 <= dst < bench["node_bench"].size else 0
        lab = T.classify(row.get("sentence") or "", row.get("cite_string") or "",
                         bs, bd)
        predicted[i] = lab.treatment
        rules[i] = f"{lab.rule}/{lab.cue or '-'}@{lab.distance}"
        windows[i] = T._WS.sub(" ", row.get("sentence") or "")[:200]

    missing = sorted(set(gold) - set(predicted))
    if missing:
        print(f"WARNING: {len(missing)} gold edges not found in the edge file "
              f"(it has been rebuilt): {missing[:8]}", file=sys.stderr)

    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    confusion: Dict[str, Counter] = defaultdict(Counter)
    errors = []
    for i, truth in gold.items():
        pred = predicted.get(i)
        if pred is None:
            continue
        confusion[truth][pred] += 1
        if pred == truth:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[truth] += 1
            errors.append((i, truth, pred))

    n = len(predicted)
    correct = sum(tp.values())
    print("=" * 76)
    print(f"treatment classifier, {n} hand-labelled citation edges")
    print(f"overall accuracy: {correct}/{n} = {correct / max(n, 1):.1%}")
    print()
    print(f"{'label':<16}{'predicted':>10}{'correct':>9}{'precision':>11}"
          f"{'gold':>6}{'recall':>9}")
    for lab in LABELS:
        pred_n = tp[lab] + fp[lab]
        gold_n = tp[lab] + fn[lab]
        prec = tp[lab] / pred_n if pred_n else float("nan")
        rec = tp[lab] / gold_n if gold_n else float("nan")
        print(f"{lab:<16}{pred_n:>10}{tp[lab]:>9}{prec:>10.0%}"
              f"{gold_n:>7}{rec:>9.0%}")

    print("\nconfusion (rows = truth, columns = predicted):")
    header = "".join(f"{l[:6]:>9}" for l in LABELS)
    print(f"{'':<16}{header}")
    for truth in LABELS:
        row = "".join(f"{confusion[truth][p]:>9}" for p in LABELS)
        print(f"{truth:<16}{row}")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for i, truth, pred in errors:
            print(f"  edge {i}: truth={truth} predicted={pred}  [{rules[i]}]")
            if args.show_errors:
                print(f"      {windows[i]}")
                note = gold_doc.get("_notes", {}).get(str(i))
                if note:
                    print(f"      note: {note}")

    lim = gold_doc["limitation_case"]
    print(f"\ndocumented limitation: edge {lim['edge']} "
          f"(predicted {lim['predicted']}, truth {lim['truth']}) - "
          f"run with --limitation for the detail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
