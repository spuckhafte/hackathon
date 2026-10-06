"""`AuthorityStore` — g(d), treatment status and the badge, by doc id.

M2 computes the citation graph, PageRank and g(d) and writes
``index/authority.json`` keyed by **node id** (the 37,210-document universe).
Everything else in the system speaks **doc id** (the 13,878 documents whose text
we indexed), so somebody has to own the join and the "what does this mean"
layer.  That is this file.

Why a store and not a dict
--------------------------
Three things have to happen together and are easy to get subtly wrong apart:

1. **The id join.**  ``doc_id`` and ``node_id`` are different numbering systems
   and a silent mis-join gives every result somebody else's authority score.
   `DocNodeMap` does the translation; this store never lets a caller skip it.
2. **The default.**  A judgment with no authority record gets ``g = 0`` and
   ``status = "ok"``, never a neighbour's score and never a crash.
3. **The explanation.**  g(d) is a weighted sum of a PageRank component and a
   bench-size component, multiplied by a treatment penalty.  The "why this
   result" panel has to show those three numbers and have them multiply out to
   the g it printed, so the store returns the parts, not just the total.

Author: Sumedha (M3)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from nyaya import config
from nyaya.contracts import Authority

__all__ = ["AuthorityStore", "BADGES"]

#: What each treatment status means to a reader, and how the UI should flag it.
BADGES: Dict[str, Dict[str, str]] = {
    "overruled": {
        "label": "overruled",
        "tone": "critical",
        "gloss": "a later judgment has overruled this one - do not rely on it "
                 "without reading that later case",
    },
    "distinguished": {
        "label": "frequently distinguished",
        "tone": "warning",
        "gloss": "later courts have repeatedly distinguished this case on its "
                 "facts, so its ratio may be narrower than it reads",
    },
    "ok": {"label": "", "tone": "neutral", "gloss": ""},
}


class AuthorityStore:
    """Read side of S4, addressed by ``doc_id``."""

    def __init__(self, nodes: Dict[int, Authority], mapping,
                 header: Optional[Dict[str, Any]] = None):
        self._nodes = nodes
        self._map = mapping
        self.header = header or {}

    # ---- construction ---------------------------------------------------
    @classmethod
    def load(cls, path: Path = None, mapping=None) -> "AuthorityStore":
        import json

        from nyaya.authority.pagerank import load_authority
        from nyaya.rank.net import DocNodeMap

        nodes = load_authority(path)
        header = {}
        p = Path(path or (config.INDEX / "authority.json"))
        if p.exists():
            try:
                header = json.loads(p.read_text(encoding="utf-8")).get("header", {})
            except Exception:                               # pragma: no cover
                header = {}
        return cls(nodes, mapping or DocNodeMap.from_manifest(), header)

    @classmethod
    def empty(cls) -> "AuthorityStore":
        """A store that answers 0 / "ok" for everything.

        Used by the B0 and S1/S2/S2p modes, which are defined as *not* using
        authority: handing them a null store is cleaner than scattering
        ``if mode in (...)`` through the scorer.
        """
        from nyaya.rank.net import DocNodeMap

        return cls({}, DocNodeMap([], [], 0))

    @classmethod
    def exists(cls, path: Path = None) -> bool:
        return Path(path or (config.INDEX / "authority.json")).exists()

    @property
    def nodes(self) -> Dict[int, Authority]:
        """The raw ``{node_id: Authority}`` dict.

        `nyaya.rank.net.g_of_docs` takes this directly - it is keyed by *node*
        id and does its own `DocNodeMap` translation, so it must not be handed
        this store, which is keyed by *doc* id.  Keeping both views named
        differently is what stops that mix-up.
        """
        return self._nodes

    # ---- lookups --------------------------------------------------------
    def record(self, doc_id: int) -> Optional[Authority]:
        node = self._map.node_of(int(doc_id))
        return self._nodes.get(node) if node >= 0 else None

    def g(self, doc_id: int) -> float:
        rec = self.record(doc_id)
        return float(getattr(rec, "g", 0.0)) if rec is not None else 0.0

    def status(self, doc_id: int) -> str:
        rec = self.record(doc_id)
        return str(getattr(rec, "status", "ok")) if rec is not None else "ok"

    def penalty(self, doc_id: int) -> float:
        rec = self.record(doc_id)
        return float(getattr(rec, "penalty", 1.0)) if rec is not None else 1.0

    def parts(self, doc_id: int) -> Dict[str, float]:
        """The components g(d) is built from, so the panel can show the sum.

        ``g = penalty * (G_PAGERANK_WEIGHT * pagerank_norm
                         + G_BENCH_WEIGHT * bench_norm)``

        `check` is that identity recomputed from the parts; the explanation
        panel prints it beside the stored ``g`` so a reader can see they agree.
        """
        rec = self.record(doc_id)
        if rec is None:
            return {"g": 0.0, "pagerank_norm": 0.0, "bench_norm": 0.0,
                    "penalty": 1.0, "in_degree": 0, "out_degree": 0, "check": 0.0}
        pr = float(getattr(rec, "pagerank_norm", 0.0))
        bn = float(getattr(rec, "bench_norm", 0.0))
        pen = float(getattr(rec, "penalty", 1.0))
        return {
            "g": float(getattr(rec, "g", 0.0)),
            "pagerank": float(getattr(rec, "pagerank", 0.0)),
            "pagerank_norm": pr,
            "bench_norm": bn,
            "penalty": pen,
            "in_degree": int(getattr(rec, "in_degree", 0)),
            "out_degree": int(getattr(rec, "out_degree", 0)),
            "check": pen * (config.G_PAGERANK_WEIGHT * pr
                            + config.G_BENCH_WEIGHT * bn),
        }

    def badge(self, doc_id: int) -> str:
        """A short, honest flag, or ``""`` when the case is undisturbed.

        Includes the evidence when M2 recorded any, because "overruled" with no
        citable reason is an accusation rather than a finding.
        """
        rec = self.record(doc_id)
        if rec is None:
            return ""
        status = str(getattr(rec, "status", "ok"))
        meta = BADGES.get(status, BADGES["ok"])
        if not meta["label"]:
            return ""
        ev = list(getattr(rec, "evidence", []) or [])
        by = list(getattr(rec, "overruled_by", []) or [])
        bits = [meta["label"]]
        if by:
            bits.append(f"by {len(by)} later judgment{'s' if len(by) > 1 else ''}")
        if ev:
            bits.append(f"- {str(ev[0])[:160]}")
        return " ".join(bits)

    def evidence(self, doc_id: int, limit: int = 3) -> List[str]:
        rec = self.record(doc_id)
        if rec is None:
            return []
        return [str(e) for e in (getattr(rec, "evidence", []) or [])][:limit]

    def gloss(self, doc_id: int) -> str:
        return BADGES.get(self.status(doc_id), BADGES["ok"])["gloss"]

    # ---- vectorised ------------------------------------------------------
    def g_array(self, doc_ids: np.ndarray) -> np.ndarray:
        """g(d) for a candidate array, aligned with it."""
        from nyaya.rank.net import g_of_docs

        return g_of_docs(np.asarray(doc_ids), self._nodes, self._map)

    def status_array(self, doc_ids: np.ndarray) -> List[str]:
        from nyaya.rank.net import status_of_docs

        return status_of_docs(np.asarray(doc_ids), self._nodes, self._map)

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:                               # pragma: no cover
        return f"<AuthorityStore {len(self._nodes)} nodes>"
