from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from chase.kg.graph import KG

INF = np.inf


class ScoreFn(Protocol):
    """What the DP needs from a scorer. BoundScorer (chase.model.scorer) implements it."""

    def edge_score(self, h: int, src: np.ndarray, rel: np.ndarray, dst: np.ndarray) -> np.ndarray: ...

    def stop_score(self, h: int, nodes: np.ndarray) -> np.ndarray: ...


class ZeroScorer:
    """Every step and stop scores 0, so the DP reduces to BFS reachability (used for rho_H)."""

    def edge_score(self, h, src, rel, dst):
        return np.zeros(len(dst))

    def stop_score(self, h, nodes):
        return np.zeros(len(nodes))


@dataclass
class DPResult:
    b: np.ndarray                    # [H+1, N] minimal surviving threshold to reach (v, h)
    cand: np.ndarray                 # [H+1, N] max(b, lam * stop) for answerable nodes, INF otherwise
    sigma: np.ndarray                # [N] min over hops of cand: smallest threshold that returns v
    best_hop: np.ndarray             # [N] hop achieving sigma, -1 if never returned
    pred_node: np.ndarray | None     # [H+1, N] predecessor on the best walk, -1 if none
    pred_rel: np.ndarray | None      # [H+1, N]
    states_expanded: int
    edges_scored: int

    def answer_set(self, tau: float) -> np.ndarray:
        """Nodes returned at threshold tau, most confident (lowest sigma) first."""
        nodes = np.flatnonzero(self.sigma <= tau)
        return nodes[np.argsort(self.sigma[nodes], kind="stable")]

    @property
    def sigma_min(self) -> float:
        """sigma of the most confident node (INF if none is returned at any threshold)."""
        return float(self.sigma.min())

    def gap_answer_set(self, gap_tau: float) -> np.ndarray:
        """Nodes whose gap sigma_v - sigma_min is <= gap_tau, most confident first."""
        return self.answer_set(self.sigma_min + gap_tau)

    def sigma_star(self, answers) -> float:
        """Nonconformity score of a question: smallest threshold at which any gold answer is returned."""
        answers = np.asarray(answers, dtype=np.int64)
        return float(self.sigma[answers].min()) if answers.size else INF

    def evidence_path(self, v: int) -> list[tuple[int, int, int]]:
        """Best walk to v as (src, rel, dst) steps from a topic entity. Needs keep_paths=True."""
        if self.pred_node is None:
            raise ValueError("run bottleneck_dp(..., keep_paths=True) to recover evidence paths")
        h = int(self.best_hop[v])
        if h < 0:
            return []
        steps = []
        while h > 0:
            u, r = int(self.pred_node[h, v]), int(self.pred_rel[h, v])
            steps.append((u, r, int(v)))
            v, h = u, h - 1
        return steps[::-1]


def bottleneck_dp(kg: KG, topic, scorer: ScoreFn, H: int, *, tau: float = INF, gap_tau: float = INF,
                  lam: float = 1.0, answer_mask: np.ndarray | None = None,
                  feasible: Callable[[int, np.ndarray], np.ndarray] | None = None,
                  keep_paths: bool = False) -> DPResult:
    """b_h(v) = min over edges (u -r-> v) of max(b_{h-1}(u), s_h(q, r, v)), over walks.

    tau = INF (calibration): no pruning; sigma is exact for every node.
    tau < INF (test): steps scoring above tau are pruned. answer_set(tau) is identical to the
    unpruned run (sigma is exact wherever sigma <= tau); only the cost changes.
    gap_tau < INF (test, gap score g_v = sigma_v - sigma_min): hop h is pruned at U + gap_tau, where U is
    the best candidate sigma found at hops < h. U >= sigma_min, so every node with g_v <= gap_tau keeps its
    best walk: sigma_min and gap_answer_set(gap_tau) are those of the unpruned run.

    answer_mask: nodes allowed as answers (e.g. not CVT, right answer type). Topic entities
    are never answers, and answers need at least one hop.
    feasible(h, nodes): states (node, h) to keep, e.g. TypeDistance feasibility.
    """
    N = kg.num_nodes
    b = np.full((H + 1, N), INF)
    pred_node = np.full((H + 1, N), -1, dtype=np.int32) if keep_paths else None
    pred_rel = np.full((H + 1, N), -1, dtype=np.int32) if keep_paths else None
    topic = np.unique(np.asarray(topic, dtype=np.int64))
    b[0, topic] = -INF
    states_expanded = edges_scored = 0
    answerable = np.ones(N, dtype=bool) if answer_mask is None else answer_mask.astype(bool).copy()
    answerable[topic] = False
    cand = np.full((H + 1, N), INF)
    best_cand = INF

    for h in range(1, H + 1):
        t = min(tau, best_cand + gap_tau)
        prev = b[h - 1]
        frontier = np.flatnonzero((prev < INF) & (prev <= t))
        if frontier.size == 0:
            break
        states_expanded += frontier.size
        src, dst, rel = kg.expand(frontier)
        if feasible is not None and dst.size:
            keep = feasible(h, dst)
            src, dst, rel = src[keep], dst[keep], rel[keep]
        if dst.size == 0:
            continue
        edges_scored += dst.size
        val = np.maximum(prev[src], scorer.edge_score(h, src, rel, dst))
        keep = val <= t
        src, dst, rel, val = src[keep], dst[keep], rel[keep], val[keep]
        if dst.size == 0:
            continue
        # min over incoming edges per destination: sort by (dst, val) and take the first of each group
        order = np.lexsort((val, dst))
        d_sorted = dst[order]
        first = np.ones(d_sorted.size, dtype=bool)
        first[1:] = d_sorted[1:] != d_sorted[:-1]
        best = order[first]
        b[h, dst[best]] = val[best]
        if keep_paths:
            pred_node[h, dst[best]] = src[best]
            pred_rel[h, dst[best]] = rel[best]
        reached = np.flatnonzero((b[h] < INF) & answerable)
        if reached.size:
            cand[h, reached] = np.maximum(b[h, reached], lam * scorer.stop_score(h, reached))
            best_cand = min(best_cand, float(cand[h].min()))

    sigma = cand.min(axis=0)
    best_hop = np.where(np.isfinite(sigma), cand.argmin(axis=0), -1).astype(np.int8)
    return DPResult(b=b, cand=cand, sigma=sigma, best_hop=best_hop, pred_node=pred_node,
                    pred_rel=pred_rel, states_expanded=states_expanded, edges_scored=edges_scored)


def reachable_within(result: DPResult, answers, H: int) -> bool:
    """With a ZeroScorer result: is some gold answer returned within <= H hops?"""
    answers = np.asarray(answers, dtype=np.int64)
    return bool(answers.size) and bool(np.isfinite(result.cand[1:H + 1, answers]).any())


def rho_freebase(data_dir: str, split: str, hops: list[int]) -> dict[int, float]:
    """rho_H on a per-question Freebase store: share of questions with an answer reachable in <= H hops."""
    from chase.kg.subgraphs import SubgraphStore
    store = SubgraphStore.open(data_dir, split)
    H_max, hit = max(hops), {H: 0 for H in hops}
    for i in range(len(store)):
        row = store.questions.iloc[i]
        kg = store.get(i)
        res = bottleneck_dp(kg, row["topic_nodes"], ZeroScorer(), H_max, answer_mask=kg.node_type == 0)
        for H in hops:
            hit[H] += reachable_within(res, row["answer_nodes"], H)
    return {H: hit[H] / len(store) for H in hops}


def main() -> None:
    ap = argparse.ArgumentParser(description="Reachability ceiling rho_H for a Freebase subgraph store")
    ap.add_argument("--data", required=True, help="e.g. data/processed/webqsp")
    ap.add_argument("--split", default="test")
    ap.add_argument("--H", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()
    t0 = time.time()
    rho = rho_freebase(args.data, args.split, args.H)
    print(f"{args.data} [{args.split}]  " + "  ".join(f"rho_{H}={v:.2%}" for H, v in rho.items())
          + f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
