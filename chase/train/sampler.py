from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
import torch

from chase.kg.graph import KG
from chase.model.features import GraphFeatures


# ------------------------------------------------------------------ labels from gold paths (built once)

@dataclass
class QuestionLabels:
    pos_steps: set[tuple[int, int, int]] = field(default_factory=set)        # (h, r, v) on a gold path
    gold_states: dict[int, set[int]] = field(default_factory=dict)           # hop h-1 -> nodes a gold step leaves from
    pos_stops: set[tuple[int, int]] = field(default_factory=set)             # (path length, last node)
    mid_stops: set[tuple[int, int]] = field(default_factory=set)             # (position, intermediate node)

    @property
    def reachable(self) -> bool:
        return bool(self.pos_stops)


def build_labels(gold: pd.DataFrame, indices, H: int) -> dict[int, QuestionLabels]:
    """Per question index: positives and gold states. Questions without gold paths get empty labels."""
    labels = {int(i): QuestionLabels(gold_states={h: set() for h in range(H)}) for i in indices}
    for i, nodes, rels in zip(gold["q_index"], gold["nodes"], gold["rels"]):
        lab = labels.get(int(i))
        if lab is None:
            continue
        nodes, rels = list(nodes), list(rels)
        for j, (r, v) in enumerate(zip(rels, nodes[1:]), start=1):
            lab.pos_steps.add((j, int(r), int(v)))
            lab.gold_states[j - 1].add(int(nodes[j - 1]))
            if j < len(rels):
                lab.mid_stops.add((j, int(v)))
        lab.pos_stops.add((len(rels), int(nodes[-1])))
    for lab in labels.values():
        lab.mid_stops -= lab.pos_stops
    return labels


# ------------------------------------------------------------------ per-epoch sampling

@dataclass
class QuestionSamples:
    steps: np.ndarray        # [n, 3] (h, r, v)
    step_y: np.ndarray       # [n]
    stops: np.ndarray        # [m, 2] (h, v)
    stop_y: np.ndarray       # [m]


def _rows(items, width):
    return np.array(sorted(items), dtype=np.int64).reshape(-1, width)


def sample_question(kg: KG, lab: QuestionLabels, H: int, rng: np.random.Generator,
                    k_hard: int = 32, k_easy: int = 32) -> QuestionSamples:
    """Positives from the gold paths; negatives drawn fresh every call.

    Hard negative steps: other edges leaving a gold state (at most k_hard per state).
    Easy negative steps: k_easy random edges at random hops.
    Negative stops: intermediate gold nodes, targets of hard negative steps, k_easy random (hop, node).
    """
    if not lab.reachable:
        return QuestionSamples(np.zeros((0, 3), np.int64), np.zeros(0, np.float32),
                               np.zeros((0, 2), np.int64), np.zeros(0, np.float32))
    neg_steps, neg_stops = set(), set(lab.mid_stops)
    for h in range(1, H + 1):
        for u in lab.gold_states.get(h - 1, ()):
            nbrs, rels = kg.neighbors(u)
            cand = [(h, int(r), int(v)) for r, v in zip(rels, nbrs) if (h, int(r), int(v)) not in lab.pos_steps]
            if len(cand) > k_hard:
                cand = [cand[j] for j in rng.choice(len(cand), k_hard, replace=False)]
            neg_steps.update(cand)
            neg_stops.update((hh, v) for hh, _, v in cand)
    if kg.num_edges:
        e = rng.integers(0, kg.num_edges, k_easy)
        hops = rng.integers(1, H + 1, k_easy)
        neg_steps.update(zip(hops.tolist(), kg.rel[e].astype(np.int64).tolist(), kg.indices[e].astype(np.int64).tolist()))
    neg_stops.update(zip(rng.integers(1, H + 1, k_easy).tolist(), rng.integers(0, kg.num_nodes, k_easy).tolist()))
    neg_steps -= lab.pos_steps
    neg_stops -= lab.pos_stops

    pos_s, neg_s = _rows(lab.pos_steps, 3), _rows(neg_steps, 3)
    pos_t, neg_t = _rows(lab.pos_stops, 2), _rows(neg_stops, 2)
    return QuestionSamples(
        steps=np.concatenate([pos_s, neg_s]), step_y=np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))].astype(np.float32),
        stops=np.concatenate([pos_t, neg_t]), stop_y=np.r_[np.ones(len(pos_t)), np.zeros(len(neg_t))].astype(np.float32))


# ------------------------------------------------------------------ batches

@dataclass
class StepBatch:
    q: torch.Tensor          # [n, q_dim]
    h: torch.Tensor          # [n]
    rel_feat: torch.Tensor   # [n, rel_dim]
    node_feat: torch.Tensor  # [n, node_dim]
    label: torch.Tensor      # [n]


@dataclass
class StopBatch:
    q: torch.Tensor
    h: torch.Tensor
    node_feat: torch.Tensor
    label: torch.Tensor


@dataclass
class GraphItem:
    """Everything stage 2 needs to run the soft DP on one question's subgraph."""
    q: torch.Tensor          # [q_dim]
    graph: GraphFeatures
    src: torch.Tensor        # [E] directed edges src -> dst
    dst: torch.Tensor
    rel_ids: torch.Tensor    # relations present in the subgraph
    rel_pos: torch.Tensor    # [E] position of each edge's relation in rel_ids
    topic: torch.Tensor
    gold: torch.Tensor
    answerable: torch.Tensor  # bool [N]
    reachable: bool


@dataclass
class Batch:
    step: StepBatch
    stop: StopBatch
    graphs: list[GraphItem] | None


def graph_item(kg: KG, graph: GraphFeatures, q: torch.Tensor, topic, gold, reachable: bool) -> GraphItem:
    src = np.repeat(np.arange(kg.num_nodes), kg.degree())
    rel_ids, rel_pos = np.unique(kg.rel.astype(np.int64), return_inverse=True)
    return GraphItem(q=q, graph=graph, src=torch.from_numpy(src), dst=torch.from_numpy(kg.indices.astype(np.int64)),
                     rel_ids=torch.from_numpy(rel_ids), rel_pos=torch.from_numpy(rel_pos.astype(np.int64)),
                     topic=torch.as_tensor(list(topic), dtype=torch.long),
                     gold=torch.as_tensor(list(gold), dtype=torch.long),
                     answerable=torch.from_numpy(kg.node_type == 0), reachable=reachable)


class Sampler:
    """Fresh training batches every epoch, built from gold paths and the subgraph store.

    store: SubgraphStore-like (.get(i), .questions with topic_nodes / answer_nodes).
    features: SubgraphFeatureStore-like (.graph(i, kg) -> GraphFeatures, .question(i) -> tensor).
    """

    def __init__(self, store, features, gold: pd.DataFrame, indices, H: int,
                 k_hard: int = 32, k_easy: int = 32, seed: int = 0):
        self.store, self.features, self.H = store, features, H
        self.k_hard, self.k_easy, self.seed = k_hard, k_easy, seed
        self.indices = [int(i) for i in indices]
        self.labels = build_labels(gold, self.indices, H)

    @property
    def reachable_indices(self) -> list[int]:
        return [i for i in self.indices if self.labels[i].reachable]

    def batches(self, epoch: int, batch_size: int, with_graph: bool = False) -> Iterator[Batch]:
        """Stage 1 (with_graph=False): reachable questions only. Stage 2: every question, plus GraphItems."""
        rng = np.random.default_rng((self.seed, epoch))
        pool = self.indices if with_graph else self.reachable_indices
        order = rng.permutation(pool)
        for start in range(0, len(order), batch_size):
            yield self._collate(order[start:start + batch_size], rng, with_graph)

    def _collate(self, chunk, rng, with_graph: bool) -> Batch:
        step_parts, stop_parts, graphs = [], [], []
        for i in chunk:
            i = int(i)
            kg = self.store.get(i)
            feats = self.features.graph(i, kg)
            q = self.features.question(i)
            lab = self.labels[i]
            s = sample_question(kg, lab, self.H, rng, self.k_hard, self.k_easy)
            if len(s.step_y):
                st = torch.from_numpy(s.steps)
                step_parts.append((q.expand(len(st), -1), st[:, 0], feats.relation[st[:, 1]], feats.node[st[:, 2]],
                                   torch.from_numpy(s.step_y)))
            if len(s.stop_y):
                sp = torch.from_numpy(s.stops)
                stop_parts.append((q.expand(len(sp), -1), sp[:, 0], feats.node[sp[:, 1]], torch.from_numpy(s.stop_y)))
            if with_graph:
                row = self.store.questions.iloc[i]
                graphs.append(graph_item(kg, feats, q, row["topic_nodes"], row["answer_nodes"], lab.reachable))
        return Batch(step=StepBatch(*_cat(step_parts, 5)), stop=StopBatch(*_cat(stop_parts, 4)),
                     graphs=graphs if with_graph else None)


def _cat(parts, width):
    if not parts:
        return [torch.empty(0)] * width
    return [torch.cat([p[k] for p in parts]) for k in range(width)]
