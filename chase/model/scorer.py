from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from chase.model.features import GraphFeatures


@dataclass
class ScorerConfig:
    q_dim: int
    rel_dim: int
    node_dim: int
    hidden: int = 256
    proj: int = 128
    hop_dim: int = 32
    h_max: int = 4


class Head(nn.Module):
    def __init__(self, cfg: ScorerConfig, use_relation: bool):
        super().__init__()
        self.use_relation = use_relation
        self.hop = nn.Embedding(cfg.h_max + 1, cfg.hop_dim)
        self.in_q = nn.Linear(cfg.q_dim + cfg.hop_dim, cfg.hidden)
        self.in_rel = nn.Linear(cfg.rel_dim, cfg.hidden, bias=False) if use_relation else None
        self.out = nn.Linear(cfg.hidden, 1)
        self.P = nn.Linear(cfg.q_dim + cfg.hop_dim, cfg.proj)
        self.Q = nn.Linear(cfg.node_dim, cfg.proj)

    def _qh(self, q: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([q, self.hop(h)], dim=-1)

    def mlp_part(self, q: torch.Tensor, h: torch.Tensor, rel: torch.Tensor | None = None) -> torch.Tensor:
        pre = self.in_q(self._qh(q, h))
        if self.use_relation:
            pre = pre + self.in_rel(rel)
        return self.out(F.relu(pre)).squeeze(-1)

    def query_vec(self, q: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.P(self._qh(q, h))

    def forward(self, q, h, node, rel=None) -> torch.Tensor:
        return self.mlp_part(q, h, rel) + (self.query_vec(q, h) * self.Q(node)).sum(-1)


class StateScorer(nn.Module):
    """Step head scores (hop h, relation r, node v); stop head scores (hop h, node v).

    Outputs are logits. Nonconformity score s = -log sigmoid(logit) = softplus(-logit),
    low = promising. Training uses logits; the DP uses s through bind().
    """

    def __init__(self, cfg: ScorerConfig):
        super().__init__()
        self.cfg = cfg
        self.step = Head(cfg, use_relation=True)
        self.stop = Head(cfg, use_relation=False)

    def step_logits(self, q, h, rel_feat, node_feat) -> torch.Tensor:
        return self.step(q, h, node_feat, rel_feat)

    def stop_logits(self, q, h, node_feat) -> torch.Tensor:
        return self.stop(q, h, node_feat)

    @torch.no_grad()
    def encode_graph(self, graph: GraphFeatures) -> "EncodedGraph":
        """Question-independent parts. Compute once per graph, reuse for every question on it."""
        return EncodedGraph(step_node=self.step.Q(graph.node), stop_node=self.stop.Q(graph.node),
                            step_rel=self.step.in_rel(graph.relation))

    @torch.no_grad()
    def bind(self, q: torch.Tensor, enc: "EncodedGraph", H: int) -> "BoundScorer":
        """Precompute per-question tables so that scoring any edge is a lookup plus an add."""
        hops = torch.arange(H + 1)
        qh = q.unsqueeze(0).expand(H + 1, -1)
        pre = self.step.in_q(self.step._qh(qh, hops))                                 # [H+1, hidden]
        rel_logit = self.step.out(F.relu(pre.unsqueeze(1) + enc.step_rel)).squeeze(-1)  # [H+1, R]
        step_node = self.step.query_vec(qh, hops) @ enc.step_node.T                   # [H+1, N]
        stop_logit = self.stop.mlp_part(qh, hops).unsqueeze(1) + self.stop.query_vec(qh, hops) @ enc.stop_node.T
        return BoundScorer(rel_logit.numpy(), step_node.numpy(), stop_logit.numpy())


@dataclass
class EncodedGraph:
    step_node: torch.Tensor   # [N, proj]   Q_step(node)
    stop_node: torch.Tensor   # [N, proj]   Q_stop(node)
    step_rel: torch.Tensor    # [R, hidden] relation part of the step MLP's first layer


@dataclass
class BoundScorer:
    """Numpy scoring functions for one question; the interface the DP consumes."""
    rel_logit: np.ndarray    # [H+1, R]
    step_node: np.ndarray    # [H+1, N]
    stop_logit: np.ndarray   # [H+1, N]

    def edge_logit(self, h: int, rel: np.ndarray, dst: np.ndarray) -> np.ndarray:
        return self.rel_logit[h, rel] + self.step_node[h, dst]

    def edge_score(self, h: int, src: np.ndarray, rel: np.ndarray, dst: np.ndarray) -> np.ndarray:
        # src is unused by this state-based scorer; kept so richer scorers can share the DP interface
        return np.logaddexp(0.0, -self.edge_logit(h, rel, dst))

    def stop_score(self, h: int, nodes: np.ndarray) -> np.ndarray:
        return np.logaddexp(0.0, -self.stop_logit[h, nodes])


def nonconformity(logit: torch.Tensor) -> torch.Tensor:
    return F.softplus(-logit)
