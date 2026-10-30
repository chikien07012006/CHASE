"""CPR-style baseline with the CHASE scorer: fixed TreeG beam, then a post-hoc conformal threshold.

Retrieval follows CPR's TreeG (cpr/core.py, retrieve_candidates): from each active path, branch out to the
`branch` best neighbour steps by local score, give each new path a value, keep the `active` best paths over all
branches, and collect the tails of the active paths at every hop. With the CHASE scorer the local score is the
step score s_h(q, r, v), a path's value is its worst step, and an answer's value adds the stop score like the DP:
max(value, lam * stop). An entity's score is the best value over the retrieved paths that end at it, INF if the
beam never reaches it. With unbounded branch/active this equals the DP's sigma (tested).

Calibration rules:
  "split": the same split-conformal rule as CHASE (k-th smallest score; INF -> abstain).
  "cpr":   CPR's own rule (cpr/conformal/path_cp.py): INF scores become 1e9, tau = np.quantile at level
           min((n+1)(1-alpha)/n, 1) with linear interpolation, and an empty set is replaced by the single best
           retrieved entity. When many questions miss in the beam, tau lands near 1e9 and the whole beam is returned.
"""
from __future__ import annotations

import numpy as np

from chase.search.dp import INF
from chase.train.evaluate import SigmaRecord

CPR_MISS = 1e9


def beam_scores(kg, topic, scorer, H: int, branch: int, active: int, lam: float = 1.0,
                answer_mask: np.ndarray | None = None) -> tuple[np.ndarray, int, int]:
    """Entity scores from a TreeG beam: (score [N], states expanded, edges scored)."""
    N = kg.num_nodes
    topic = np.unique(np.asarray(topic, dtype=np.int64))
    answerable = np.ones(N, dtype=bool) if answer_mask is None else answer_mask.astype(bool).copy()
    answerable[topic] = False
    score = np.full(N, INF)
    nodes, vals = topic, np.full(topic.size, -INF)
    states = edges = 0
    for h in range(1, H + 1):
        if nodes.size == 0:
            break
        states += nodes.size
        counts = kg.indptr[nodes + 1] - kg.indptr[nodes]
        path = np.repeat(np.arange(nodes.size), counts)
        src, dst, rel = kg.expand(nodes)
        if dst.size == 0:
            break
        edges += dst.size
        local = scorer.edge_score(h, src, rel, dst)
        # branch-out: the `branch` best steps of every active path
        order = np.lexsort((local, path))
        first = np.searchsorted(path[order], np.arange(nodes.size))
        rank = np.arange(order.size) - np.repeat(first, counts)
        keep = order[rank < branch]
        val = np.maximum(vals[path[keep]], local[keep])
        # active set: the `active` best new paths over all branches
        sel = np.argsort(val, kind="stable")[:active]
        nodes, vals = dst[keep[sel]], val[sel]
        ans = answerable[nodes]
        if ans.any():
            v = nodes[ans]
            np.minimum.at(score, v, np.maximum(vals[ans], lam * scorer.stop_score(h, v)))
    return score, states, edges


def beam_record(score: np.ndarray, gold: np.ndarray, states: int, edges: int, seconds: float) -> SigmaRecord:
    finite = score[np.isfinite(score)]
    top = int(np.argmin(score)) if finite.size else -1
    return SigmaRecord(sigma_star=float(score[gold].min()) if gold.size else INF, sorted_sigma=np.sort(finite),
                       states_expanded=states, edges_scored=edges, seconds=seconds,
                       top_sigma=float(score[top]) if top >= 0 else INF,
                       top_is_gold=bool(top >= 0 and top in set(gold.tolist())))


# ------------------------------------------------------------------ CPR's own calibration rule

def cpr_quantile(scores, alpha: float) -> float:
    s = np.where(np.isfinite(scores), scores, CPR_MISS).astype(float)
    n = len(s)
    return float(np.quantile(s, min(max((n + 1) * (1 - alpha) / n, 0.0), 1.0)))


def _cpr_eval(stars: np.ndarray, sorted_sigma: list[np.ndarray], top_gold: np.ndarray, tau: np.ndarray):
    """Per (threshold, question): covered and set size under CPR's rule (empty set -> best retrieved entity)."""
    raw = np.stack([np.searchsorted(s, tau, side="right") for s in sorted_sigma], axis=-1)
    n_cand = np.array([len(s) for s in sorted_sigma])
    fallback = (raw == 0) & (n_cand > 0)
    cov = (np.isfinite(stars) & (stars <= tau[..., None])) | (fallback & top_gold)
    return cov, np.where(fallback, 1, raw)


def cpr_metrics_at(cal: list[SigmaRecord], test: list[SigmaRecord], alpha: float) -> dict:
    tau = cpr_quantile([r.sigma_star for r in cal], alpha)
    cov, size = _cpr_eval(np.array([r.sigma_star for r in test]), [r.sorted_sigma for r in test],
                          np.array([r.top_is_gold for r in test]), np.array([tau]))
    cov, size = cov[0], size[0].astype(float)
    return {"alpha": alpha, "target": 1 - alpha, "tau": tau, "feasible": True, "returns_whole_beam": tau >= CPR_MISS / 2,
            "ecr": float(cov.mean()), "apss": float(size.mean()), "size_median": float(np.median(size)),
            "empty_share": float(np.mean(size == 0))}


def cpr_resplit_samples(cal: list[SigmaRecord], test: list[SigmaRecord], alphas, n_splits: int = 1000,
                        seed: int = 0) -> dict[float, dict[str, np.ndarray]]:
    """Same permutations as chase.eval.freebase.resplit_samples, with CPR's rule."""
    recs = cal + test
    stars = np.array([r.sigma_star for r in recs])
    top_gold = np.array([r.top_is_gold for r in recs])
    n_cal, n_test = len(cal), len(test)
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(len(recs)) for _ in range(n_splits)])
    cal_scores = np.where(np.isfinite(stars), stars, CPR_MISS)[perms[:, :n_cal]]
    in_test = np.zeros((n_splits, len(recs)), dtype=bool)
    np.put_along_axis(in_test, perms[:, n_cal:], True, axis=1)
    out = {}
    for a in alphas:
        tau = np.quantile(cal_scores, min(max((n_cal + 1) * (1 - a) / n_cal, 0.0), 1.0), axis=1)
        cov, size = _cpr_eval(stars, [r.sorted_sigma for r in recs], top_gold, tau)
        out[a] = {"ecr": (cov & in_test).sum(1) / n_test, "apss": (size * in_test).sum(1) / n_test,
                  "empty": ((size == 0) & in_test).sum(1) / n_test}
    return out
