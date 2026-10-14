from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from chase.kg.graph import KG
from chase.kg.subgraphs import SubgraphStore
from chase.model.text_cache import CACHE_DIRNAME


@dataclass
class GraphFeatures:
    """Frozen input features of one searchable graph, aligned with its node and relation ids.

    node:     [N, node_dim] = [text embedding ; one-hot node type ; log(1 + degree)]
    relation: [R, rel_dim]  = [text embedding ; inverse flag]

    The scorer only sees this object, so it works the same on full PrimeKG and on
    a per-question Freebase subgraph.
    """
    node: torch.Tensor
    relation: torch.Tensor

    @property
    def node_dim(self) -> int:
        return self.node.shape[1]

    @property
    def rel_dim(self) -> int:
        return self.relation.shape[1]


def _node_matrix(text: np.ndarray, node_type: np.ndarray, num_types: int, degree: np.ndarray) -> torch.Tensor:
    onehot = np.eye(num_types, dtype=np.float32)[node_type]
    logdeg = np.log1p(degree).astype(np.float32)[:, None]
    return torch.from_numpy(np.concatenate([text.astype(np.float32), onehot, logdeg], axis=1))


def _relation_matrix(cache: Path) -> torch.Tensor:
    emb = np.load(cache / "relations.npy").astype(np.float32)
    inverse = np.load(cache / "rel_inverse.npy").astype(np.float32)[:, None]
    return torch.from_numpy(np.concatenate([emb, inverse], axis=1))


def primekg_features(kg: KG, kg_dir: str | Path) -> GraphFeatures:
    cache = Path(kg_dir) / CACHE_DIRNAME
    node = _node_matrix(np.load(cache / "nodes.npy"), kg.node_type, len(kg.type_names), kg.degree())
    return GraphFeatures(node=node, relation=_relation_matrix(cache))


class SubgraphFeatureStore:
    """Per-question GraphFeatures for one Freebase split. CVT nodes get a zero text vector."""

    def __init__(self, store: SubgraphStore, data_dir: str | Path, split: str):
        cache = Path(data_dir) / CACHE_DIRNAME
        self.store = store
        self.names = np.load(cache / "names.npy", mmap_mode="r")
        self.node_emb_row = np.load(cache / split / "node_emb_row.npy")
        self.questions = torch.from_numpy(np.load(cache / split / "questions.npy").astype(np.float32))
        self.relation = _relation_matrix(cache)
        self.zero = np.zeros(self.names.shape[1], dtype=np.float16)

    def question(self, i: int) -> torch.Tensor:
        return self.questions[i]

    def graph(self, i: int, kg: KG | None = None) -> GraphFeatures:
        kg = kg if kg is not None else self.store.get(i)
        rows = self.node_emb_row[self.store.node_offset[i]: self.store.node_offset[i + 1]]
        text = np.where(rows[:, None] >= 0, self.names[np.maximum(rows, 0)], self.zero)
        node = _node_matrix(text, kg.node_type, len(kg.type_names), kg.degree())
        return GraphFeatures(node=node, relation=self.relation)
