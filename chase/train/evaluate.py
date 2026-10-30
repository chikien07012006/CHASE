from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from chase.conformal.split import INF, conformal_quantile, covered
from chase.search.dp import bottleneck_dp


@dataclass
class SigmaRecord:
    """Hard-DP result of one labelled question, as CHASE scores: the gap g_v = sigma_v - sigma_min(q) (see to_gap)."""
    sigma_star: float          # g*: smallest threshold that returns a gold answer (INF if none)
    sorted_sigma: np.ndarray   # score of every answerable node, sorted (finite only)
    states_expanded: int
    edges_scored: int = 0
    seconds: float = 0.0       # bind + DP wall time
    top_sigma: float = INF     # sigma of the most confident node
    top_is_gold: bool = False  # is that node a gold answer
    best_hop: int = -1         # hop of the gold answer achieving sigma_star
    topic_degree: int = 0      # largest degree among the topic entities


def to_gap(rec: SigmaRecord) -> SigmaRecord:
    """Gap score g_v = sigma_v - sigma_min(q): a raw-sigma record measured from the question's most confident node.

    sigma_min uses no labels, so g* = sigma* - sigma_min is still one exchangeable score per question.
    """
    if rec.sorted_sigma.size == 0:
        return rec
    m = rec.sorted_sigma[0]
    return replace(rec, sigma_star=rec.sigma_star - m, sorted_sigma=rec.sorted_sigma - m, top_sigma=rec.top_sigma - m)


def collect_sigmas(scorer, store, features, indices, H: int, lam: float = 1.0, tau: float = INF,
                   progress: bool = False, desc: str = "validating") -> list[SigmaRecord]:
    """DP with the current scorer on each question (Freebase stores: CVT nodes are never answers); gap records.

    tau = INF (default) runs the unpruned DP used for calibration and metrics; a finite tau is a gap threshold
    and runs the pruned search, whose answer set at tau is identical but whose cost is the real test-time cost.
    """
    from tqdm import tqdm
    scorer.eval()
    records = []
    for i in tqdm(indices, desc=desc, colour="cyan", leave=False, dynamic_ncols=True, disable=not progress):
        i = int(i)
        row = store.questions.iloc[i]
        kg = store.get(i)
        topic, gold = np.asarray(row["topic_nodes"], dtype=np.int64), np.asarray(row["answer_nodes"], dtype=np.int64)
        t0 = time.perf_counter()
        bound = scorer.bind(features.question(i), scorer.encode_graph(features.graph(i, kg)), H)
        res = bottleneck_dp(kg, topic, bound, H, lam=lam, answer_mask=kg.node_type == 0, gap_tau=tau)
        seconds = time.perf_counter() - t0
        s = res.sigma
        star = res.sigma_star(gold)
        top = int(np.argmin(s)) if np.isfinite(s).any() else -1
        best_hop = int(res.best_hop[gold[np.argmin(s[gold])]]) if np.isfinite(star) else -1
        records.append(SigmaRecord(
            sigma_star=star, sorted_sigma=np.sort(s[np.isfinite(s)]), states_expanded=res.states_expanded,
            edges_scored=res.edges_scored, seconds=seconds, top_sigma=float(s[top]) if top >= 0 else INF,
            top_is_gold=bool(top >= 0 and top in set(gold.tolist())), best_hop=best_hop,
            topic_degree=int(kg.degree()[topic].max()) if topic.size else 0))
    return [to_gap(r) for r in records]


def set_sizes(records: list[SigmaRecord], tau: float) -> np.ndarray:
    """Answer-set size of each question at threshold tau."""
    return np.array([np.searchsorted(r.sorted_sigma, tau, side="right") for r in records], dtype=float)


def two_fold_metrics(records: list[SigmaRecord], alpha: float, seed: int = 0) -> dict:
    """Calibrate on one half, measure on the other, then swap; report the average.

    Used for model selection on the validation set. APSS is INF if either fold is infeasible.
    """
    perm = np.random.default_rng(seed).permutation(len(records))
    halves = [perm[: len(perm) // 2], perm[len(perm) // 2:]]
    apss, ecr, taus = [], [], []
    for cal, test in [(halves[0], halves[1]), (halves[1], halves[0])]:
        tau = conformal_quantile([records[j].sigma_star for j in cal], alpha)
        taus.append(tau)
        if not np.isfinite(tau):
            apss.append(INF)
            ecr.append(float("nan"))
            continue
        test_recs = [records[j] for j in test]
        ecr.append(float(covered([r.sigma_star for r in test_recs], tau).mean()))
        apss.append(float(set_sizes(test_recs, tau).mean()))
    return {"apss": float(np.mean(apss)), "ecr": float(np.mean(ecr)), "tau": float(np.mean(taus)),
            "feasible": bool(np.all(np.isfinite(taus)))}
