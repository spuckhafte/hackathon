"""Small shared helpers.  No IR logic lives here."""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


def ensure_utf8_stdout() -> None:
    """Windows consoles default to cp1252; the demo prints statute tokens."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int:
    with Path(path).open("rb") as fh:
        return sum(1 for _ in fh)


@contextmanager
def timed(label: str, quiet: bool = False):
    t0 = time.perf_counter()
    yield
    if not quiet:
        print(f"[{label}] {time.perf_counter() - t0:.2f}s", file=sys.stderr)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"
