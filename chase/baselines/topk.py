"""Conformal top-K with the CHASE scorer.

Rank every answerable node by sigma_v (ties broken by node id). The nonconformity score of a question is the
rank of its best gold answer (INF if none is reachable). K_hat = the conformal quantile of these ranks; the set
is the K_hat best nodes. Ranks are integers, so the scores tie and coverage is >= 1 - alpha but not tight.

Because the score is a rank and every candidate's "score" is its own rank, a question is a SigmaRecord with
sorted_sigma = 1..n: chase.eval.freebase.metrics_at / resplit_samples then work unchanged.
"""
from __future__ import annotations

import numpy as np

from chase.search.dp import INF, DPResult
from chase.train.evaluate import SigmaRecord


def topk_record(res: DPResult, gold: np.ndarray, seconds: float = 0.0) -> SigmaRecord:
    s = res.sigma
    cand = np.flatnonzero(np.isfinite(s))
    order = cand[np.lexsort((cand, s[cand]))]          # by sigma, then node id
    rank = np.full(len(s), INF)
    rank[order] = np.arange(1, order.size + 1)
    star = float(rank[gold].min()) if gold.size else INF
    return SigmaRecord(sigma_star=star, sorted_sigma=np.arange(1, order.size + 1, dtype=float),
                       states_expanded=res.states_expanded, edges_scored=res.edges_scored, seconds=seconds,
                       top_sigma=1.0 if order.size else INF, top_is_gold=bool(order.size and order[0] in set(gold.tolist())))
