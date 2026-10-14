from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class KG:
    indptr: np.ndarray        # int64 [N+1]; out-edges of u are indptr[u]:indptr[u+1]
    indices: np.ndarray       # int32 [E]; destination node of each edge
    rel: np.ndarray           # int16 [E]; relation id of each edge
    node_type: np.ndarray     # int8  [N]; index into type_names
    type_names: list[str]
    relation_names: list[str]
    nodes: pd.DataFrame       # one row per node id: key, name, aliases, ...

    @property
    def num_nodes(self) -> int:
        return len(self.indptr) - 1

    @property
    def num_edges(self) -> int:
        return len(self.indices)

    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def neighbors(self, u: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.indptr[u], self.indptr[u + 1]
        return self.indices[s:e], self.rel[s:e]

    def expand(self, src: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All out-edges of the nodes in `src`, vectorised.

        Returns (edge_src, edge_dst, edge_rel). This is the inner step of the
        bottleneck DP: one call per hop instead of a Python loop over nodes.
        """
        src = np.asarray(src, dtype=np.int64)
        starts = self.indptr[src]
        counts = self.indptr[src + 1] - starts
        total = int(counts.sum())
        if total == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty
        # position of each edge = start of its block + offset inside the block
        block_begin = np.repeat(np.cumsum(counts) - counts, counts)
        pos = np.repeat(starts, counts) + (np.arange(total) - block_begin)
        return np.repeat(src, counts), self.indices[pos].astype(np.int64), self.rel[pos].astype(np.int64)

    # ------------------------------------------------------------------ io
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "csr.npz", indptr=self.indptr, indices=self.indices,
                 rel=self.rel, node_type=self.node_type)
        self.nodes.to_parquet(out / "nodes.parquet", index=False)
        meta = {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "type_names": self.type_names,
            "relation_names": self.relation_names,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, in_dir: str | Path) -> "KG":
        d = Path(in_dir)
        z = np.load(d / "csr.npz")
        meta = json.loads((d / "meta.json").read_text())
        return cls(indptr=z["indptr"], indices=z["indices"], rel=z["rel"],
                   node_type=z["node_type"], type_names=meta["type_names"],
                   relation_names=meta["relation_names"],
                   nodes=pd.read_parquet(d / "nodes.parquet"))


def build_csr(src: np.ndarray, dst: np.ndarray, rel: np.ndarray,
              num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort directed edges by (src, rel, dst) and compress into CSR arrays."""
    order = np.lexsort((dst, rel, src))
    src, dst, rel = src[order], dst[order], rel[order]
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, dst.astype(np.int32), rel.astype(np.int16)
