"""UaG-style baseline with the CHASE scorer: one conformal threshold per hop, plus one for stopping.

UaG has no public code; this follows its idea of calibrating each retrieval step separately. A walk is kept iff
its hop-h step scores <= tau_h for every hop and its stop score (lam * stop) <= tau_stop. CHASE instead uses one
threshold on the walk's worst step, so the difference between the two is exactly per-hop vs per-question
calibration.

Calibration scores: for each cal question take the walk that returns its best gold answer (the DP's evidence
walk) and record its component scores [s_1, ..., s_H, stop]; hops the walk does not use are NaN; a question with
no reachable answer gets INF everywhere. Component c is calibrated on the cal questions that use it, at level
alpha_c:
  "per_hop":    alpha_c = alpha for every component (no correction for the H + 1 chances to miss);
  "bonferroni": alpha_c = alpha / (H + 1). Valid by the union bound: P(miss) <= sum_c alpha_c = alpha.

Only H <= 2 is supported (all Freebase experiments use H = 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chase.conformal.split import conformal_quantile
from chase.search.dp import INF, DPResult

ALLOCATIONS = ("per_hop", "bonferroni")


def walk_components(res: DPResult, scorer, gold: np.ndarray, H: int, lam: float = 1.0) -> np.ndarray:
    """[s_1, ..., s_H, stop] of the evidence walk to the best gold answer. Needs keep_paths=True."""
    comp = np.full(H + 1, np.nan)
    if gold.size == 0 or not np.isfinite(res.sigma[gold]).any():
        comp[:] = INF
        return comp
    v = int(gold[np.argmin(res.sigma[gold])])
    steps = res.evidence_path(v)
    for h, (u, r, w) in enumerate(steps, start=1):
        comp[h - 1] = scorer.edge_score(h, np.array([u]), np.array([r]), np.array([w]))[0]
    comp[H] = lam * scorer.stop_score(len(steps), np.array([v]))[0]
    return comp


@dataclass
class PerHopTable:
    """Requirements for every way of returning a node: row i passes iff req[i, c] <= tau_c for all c."""
    req: np.ndarray        # [M, H+1]; -INF = no requirement on that component
    starts: np.ndarray     # [G] first row of each distinct node (rows sorted by node)
    gold: np.ndarray       # [G] bool: is that node a gold answer

    def evaluate(self, tau: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
        """tau [T, H+1] -> (covered [T] bool, set size [T])."""
        T = len(tau)
        cov, size = np.zeros(T, dtype=bool), np.zeros(T)
        if self.starts.size == 0:
            return cov, size
        for lo in range(0, T, chunk):
            t = tau[lo:lo + chunk]
            ok = np.ones((len(t), len(self.req)), dtype=bool)
            for c in range(self.req.shape[1]):
                ok &= self.req[None, :, c] <= t[:, None, c]
            node_ok = np.logical_or.reduceat(ok, self.starts, axis=1)
            size[lo:lo + chunk] = node_ok.sum(1)
            cov[lo:lo + chunk] = node_ok[:, self.gold].any(1)
        return cov, size


def per_hop_table(kg, topic, scorer, res: DPResult, gold: np.ndarray, H: int, lam: float = 1.0,
                  answer_mask: np.ndarray | None = None) -> PerHopTable:
    """Built from an unpruned DP (res.b[1] = best hop-1 step score into each node)."""
    if H > 2:
        raise ValueError("per-hop baseline supports H <= 2")
    topic = np.unique(np.asarray(topic, dtype=np.int64))
    answerable = np.ones(kg.num_nodes, dtype=bool) if answer_mask is None else answer_mask.astype(bool).copy()
    answerable[topic] = False
    b1 = res.b[1]
    rows = []                                                       # (node, s_1, s_2, stop)
    v1 = np.flatnonzero(np.isfinite(b1) & answerable)
    if v1.size:
        r = np.full((v1.size, 4), -INF)
        r[:, 0], r[:, 1], r[:, 3] = v1, b1[v1], lam * scorer.stop_score(1, v1)
        rows.append(r)
    if H == 2:
        u = np.flatnonzero(np.isfinite(b1))
        src, dst, rel = kg.expand(u) if u.size else (np.empty(0, np.int64),) * 3
        keep = answerable[dst]
        src, dst, rel = src[keep], dst[keep], rel[keep]
        if dst.size:
            # Pareto frontier of (s_1, s_2) per node: the stop score is the same for every hop-2 way into it
            df = pd.DataFrame({"v": dst, "t1": b1[src], "t2": scorer.edge_score(2, src, rel, dst)})
            df = df.sort_values(["v", "t1", "t2"], kind="stable")
            prev_min = df.groupby("v")["t2"].cummin().groupby(df["v"]).shift(fill_value=INF)
            df = df[df["t2"] < prev_min]
            v2 = df["v"].to_numpy()
            r = np.column_stack([v2, df["t1"].to_numpy(), df["t2"].to_numpy(), lam * scorer.stop_score(2, v2)])
            rows.append(r)
    if not rows:
        return PerHopTable(req=np.empty((0, H + 1)), starts=np.empty(0, np.int64), gold=np.empty(0, bool))
    r = np.concatenate(rows)
    r = r[np.argsort(r[:, 0], kind="stable")]
    nodes = r[:, 0].astype(np.int64)
    starts = np.flatnonzero(np.r_[True, nodes[1:] != nodes[:-1]])
    req = r[:, 1:] if H == 2 else r[:, [1, 3]]
    return PerHopTable(req=req, starts=starts, gold=np.isin(nodes[starts], gold))


def thresholds(comp: np.ndarray, alpha: float, allocation: str) -> np.ndarray:
    """[H+1] thresholds from cal component scores comp [n, H+1]; INF if any component's quantile is INF."""
    C = comp.shape[1]
    a = alpha if allocation == "per_hop" else alpha / C
    tau = np.empty(C)
    for c in range(C):
        s = comp[:, c][~np.isnan(comp[:, c])]
        tau[c] = conformal_quantile(s, a) if s.size else INF
    return tau


def metrics_at(cal_comp: np.ndarray, tables: list[PerHopTable], alpha: float, allocation: str) -> dict:
    tau = thresholds(cal_comp, alpha, allocation)
    row = {"alpha": alpha, "target": 1 - alpha, "tau": tau.tolist(), "feasible": bool(np.isfinite(tau).all())}
    if not row["feasible"]:
        return row
    res = [t.evaluate(tau[None]) for t in tables]
    cov = np.array([c[0] for c, _ in res])
    size = np.array([s[0] for _, s in res])
    row.update(ecr=float(cov.mean()), apss=float(size.mean()), size_median=float(np.median(size)),
               empty_share=float(np.mean(size == 0)))
    return row


def resplit_samples(cal_comp: np.ndarray, test_comp: np.ndarray, cal_tables: list[PerHopTable],
                    test_tables: list[PerHopTable], alphas, allocation: str, n_splits: int = 200,
                    seed: int = 0) -> dict[float, dict[str, np.ndarray]]:
    """Same permutation scheme as chase.eval.freebase.resplit_samples (fewer splits: each one needs a table pass)."""
    comp = np.concatenate([cal_comp, test_comp])
    tables = cal_tables + test_tables
    n_cal, n_test = len(cal_comp), len(test_comp)
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(len(comp)) for _ in range(n_splits)])
    in_test = np.zeros((n_splits, len(comp)), dtype=bool)
    np.put_along_axis(in_test, perms[:, n_cal:], True, axis=1)
    tau = np.stack([np.stack([thresholds(comp[perms[s, :n_cal]], a, allocation) for a in alphas])
                    for s in range(n_splits)])                                      # [S, A, C]
    S, A, C = tau.shape
    flat = tau.reshape(S * A, C)
    ok = np.isfinite(flat).all(1)
    ecr, apss, empty = np.zeros(S * A), np.zeros(S * A), np.zeros(S * A)
    w = np.repeat(in_test, A, axis=0)                                               # [S*A, P]
    for p, t in enumerate(tables):
        cov, size = t.evaluate(np.where(ok[:, None], flat, -INF))
        ecr += cov * w[:, p]
        apss += size * w[:, p]
        empty += (size == 0) * w[:, p]
    out = {}
    for i, a in enumerate(alphas):
        sl = slice(i, S * A, A)
        nan = np.where(ok[sl], 1.0, np.nan)
        out[a] = {"ecr": ecr[sl] / n_test * nan, "apss": apss[sl] / n_test * nan, "empty": empty[sl] / n_test * nan}
    return out
