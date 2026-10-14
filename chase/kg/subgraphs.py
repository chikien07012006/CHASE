from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chase.kg.graph import KG

NODE_TYPES = ["entity", "cvt"]


class SubgraphStore:
    """Per-question subgraphs of one split, packed into flat arrays.

    Question i owns nodes node_offset[i]:node_offset[i+1] of the packed node
    arrays and edges edge_offset[i]:edge_offset[i+1]. Inside a question, node
    ids are local (0..n_i-1), so get(i) returns an ordinary KG.
    """

    def __init__(self, split_dir: str | Path, relation_names: list[str]):
        d = Path(split_dir)
        z = np.load(d / "csr.npz")
        self.node_offset = z["node_offset"]
        self.edge_offset = z["edge_offset"]
        self.indptr = z["indptr"]          # local edge positions, one block of n_i+1 per question
        self.indices = z["indices"]        # local node ids
        self.rel = z["rel"]
        self.node_type = z["node_type"]
        self.questions = pd.read_parquet(d / "questions.parquet")
        self._names = pd.read_parquet(d / "node_names.parquet")["names"]
        self.relation_names = relation_names

    @classmethod
    def open(cls, dataset_dir: str | Path, split: str) -> "SubgraphStore":
        d = Path(dataset_dir)
        relation_names = json.loads((d / "relations.json").read_text())
        return cls(d / split, relation_names)

    def __len__(self) -> int:
        return len(self.node_offset) - 1

    def get(self, i: int) -> KG:
        n0, n1 = self.node_offset[i], self.node_offset[i + 1]
        e0, e1 = self.edge_offset[i], self.edge_offset[i + 1]
        indptr = self.indptr[n0 + i: n1 + i + 1]   # each question stores n_i+1 entries
        return KG(indptr=indptr, indices=self.indices[e0:e1], rel=self.rel[e0:e1],
                  node_type=self.node_type[n0:n1], type_names=NODE_TYPES,
                  relation_names=self.relation_names,
                  nodes=pd.DataFrame({"name": self._names.iloc[i]}))
