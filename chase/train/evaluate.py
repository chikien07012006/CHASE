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
    # tie-breaking: candidates are ordered by (sigma, tie) lexicographically; tie = lam * stop score at the best hop
    star_tie: float = 0.0                  # tie of the gold answer achieving sigma_star
    sorted_tie: np.ndarray | None = None   # tie of each candidate, aligned with sorted_sigma (None: all 0)
    top_tie: float = 0.0                   # tie of the most confident node


def record_from_dp(res, gold: np.ndarray, *, tie_break: bool, seconds: float = 0.0, topic_degree: int = 0) -> SigmaRecord:
    """Raw-sigma record of one question. Candidates are sorted by (sigma, tie), and sigma_star / the top node are
    the lexicographic minima. tie_break=False sets every tie to 0: sigma alone, ties kept whole (node id order)."""
    s = res.sigma
    t = res.tie if tie_break else np.zeros_like(s)
    finite = np.flatnonzero(np.isfinite(s))
    order = finite[np.lexsort((t[finite], s[finite]))]
    g = gold[np.isfinite(s[gold])] if gold.size else gold
    best = int(g[np.lexsort((t[g], s[g]))[0]]) if g.size else -1
    top = int(order[0]) if order.size else -1
    return SigmaRecord(
        sigma_star=float(s[best]) if best >= 0 else INF, sorted_sigma=s[order], states_expanded=res.states_expanded,
        edges_scored=res.edges_scored, seconds=seconds, top_sigma=float(s[top]) if top >= 0 else INF,
        top_is_gold=bool(top >= 0 and top in set(gold.tolist())), best_hop=int(res.best_hop[best]) if best >= 0 else -1,
        topic_degree=topic_degree, star_tie=float(t[best]) if best >= 0 else 0.0, sorted_tie=t[order],
        top_tie=float(t[top]) if top >= 0 else 0.0)


def to_gap(rec: SigmaRecord) -> SigmaRecord:
    """Gap score g_v = sigma_v - sigma_min(q): a raw-sigma record measured from the question's most confident node.

    sigma_min uses no labels, so g* = sigma* - sigma_min is still one exchangeable score per question. The ties are
    measured from the top node's tie in the same way, so the top node is the pair (0, 0) in every question.
    """
    if rec.sorted_sigma.size == 0:
        return rec
    m = rec.sorted_sigma[0]
    rec = replace(rec, sigma_star=rec.sigma_star - m, sorted_sigma=rec.sorted_sigma - m, top_sigma=rec.top_sigma - m)
    if rec.sorted_tie is None:
        return rec
    t = rec.top_tie
    return replace(rec, star_tie=rec.star_tie - t, sorted_tie=rec.sorted_tie - t, top_tie=0.0)


class LexKey:
    """Exact scalar keys for the lexicographic order of (gap, tie) pairs.

    Built from every pair of a pool of gap records (the keys only re-encode the order). For any pair (g, t) and any
    pair (g', t') whose gap and tie both occur in the pool, key(g, t) <= key(g', t') iff (g, t) <= (g', t')
    lexicographically, and the same with the sides swapped. Thresholds are calibration pairs, which are in the pool,
    so the scalar conformal code (quantile, set sizes, re-splits) gives the lexicographic result.
    """

    def __init__(self, records: list[SigmaRecord]):
        gs, ts = [], []
        for r in records:
            gs.append(r.sorted_sigma)
            ts.append(r.sorted_tie if r.sorted_tie is not None else np.zeros(r.sorted_sigma.size))
            if np.isfinite(r.sigma_star):
                gs.append([r.sigma_star])
                ts.append([r.star_tie])
        self.g = np.unique(np.concatenate(gs + [[0.0]]))
        self.t = np.unique(np.concatenate(ts + [[0.0]]))
        self.m = 2 * self.t.size + 2

    @staticmethod
    def _code(u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """2i where x == u[i]; 2i - 1 where u[i-1] < x < u[i]: monotone, and exact on u."""
        i = np.searchsorted(u, x, side="left")
        exact = (i < u.size) & (u[np.minimum(i, u.size - 1)] == x)
        return 2 * i - (~exact).astype(np.int64)

    def key(self, g, t) -> np.ndarray:
        g, t = np.atleast_1d(np.asarray(g, dtype=float)), np.atleast_1d(np.asarray(t, dtype=float))
        ok = np.isfinite(g)
        out = np.full(g.shape, INF)
        out[ok] = (self._code(self.g, g[ok]) * self.m + self._code(self.t, t[ok]) + 1).astype(float)
        return out

    def gap(self, key: float) -> float:
        """The gap of a key (exact for pairs of the pool; otherwise the next gap of the pool, an upper bound)."""
        if not np.isfinite(key):
            return INF
        c = int(key) // self.m
        return float(self.g[min((c + 1) // 2, self.g.size - 1)])

    def apply(self, records: list[SigmaRecord]) -> list[SigmaRecord]:
        """Gap records -> records whose scores are the keys (set sizes and coverage then follow the pair order)."""
        out = []
        for r in records:
            tie = r.sorted_tie if r.sorted_tie is not None else np.zeros(r.sorted_sigma.size)
            out.append(replace(r, sigma_star=float(self.key(r.sigma_star, r.star_tie)[0]),
                               sorted_sigma=self.key(r.sorted_sigma, tie),
                               top_sigma=float(self.key(r.top_sigma, r.top_tie)[0]), sorted_tie=None, star_tie=0.0,
                               top_tie=0.0))
        return out


def collect_sigmas(scorer, store, features, indices, H: int, lam: float = 1.0, tau: float = INF,
                   progress: bool = False, desc: str = "validating", tie_break: bool = False) -> list[SigmaRecord]:
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
        records.append(record_from_dp(res, gold, tie_break=tie_break, seconds=seconds,
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
