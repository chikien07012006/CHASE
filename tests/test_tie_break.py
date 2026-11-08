import math

import numpy as np
import pytest

from chase.conformal.split import conformal_quantile, covered
from chase.search.dp import INF, bottleneck_dp
from chase.train.evaluate import LexKey, SigmaRecord, record_from_dp, set_sizes, to_gap
from tests.test_dp import H, _case


def _lex_le(a, b):
    return a[0] < b[0] or (a[0] == b[0] and a[1] <= b[1])


@pytest.mark.parametrize("seed", range(10))
def test_tie_is_the_stop_score_at_the_best_hop(seed):
    kg, scorer, topic, mask, _ = _case(seed, with_mask=True)
    res = bottleneck_dp(kg, topic, scorer, H, lam=0.8, answer_mask=mask)
    for v in range(kg.num_nodes):
        h = res.best_hop[v]
        assert res.tie[v] == (0.8 * scorer.stop[h, v] if h >= 0 else INF)


@pytest.mark.parametrize("seed", range(10))
def test_lex_key_follows_the_lexicographic_order(seed):
    rng = np.random.default_rng(seed)
    pool = [SigmaRecord(sigma_star=float(rng.integers(0, 4)), star_tie=float(rng.integers(-2, 3)),
                        sorted_sigma=np.sort(rng.integers(0, 5, 6).astype(float)), sorted_tie=rng.integers(-2, 3, 6) * 1.0,
                        states_expanded=0) for _ in range(20)]
    lex = LexKey(pool)
    pairs = [(g, t) for g in np.arange(-1, 6, 0.5) for t in np.arange(-3, 4, 0.5)]   # any pair
    grid = [(g, t) for g in lex.g for t in lex.t]                                   # thresholds: pool values
    keys = lex.key([p[0] for p in pairs], [p[1] for p in pairs])
    kgrid = lex.key([q[0] for q in grid], [q[1] for q in grid])
    for p, k in zip(pairs, keys):
        for q, kq in zip(grid, kgrid):
            assert (k <= kq) == _lex_le(p, q) and (kq <= k) == _lex_le(q, p)
    assert lex.key(INF, 0.0)[0] == INF
    for r in pool:                                              # gaps of pool pairs decode exactly
        assert lex.gap(lex.key(r.sigma_star, r.star_tie)[0]) == r.sigma_star


@pytest.mark.parametrize("seed", range(10))
def test_no_tie_break_reproduces_sigma_only_sets(seed):
    kg, scorer, topic, mask, _ = _case(seed, with_mask=True)
    res = bottleneck_dp(kg, topic, scorer, H, lam=0.8, answer_mask=mask)
    gold = np.flatnonzero(mask)[:3]
    rec = to_gap(record_from_dp(res, gold, tie_break=False))
    lex = LexKey([rec])
    keyed = lex.apply([rec])[0]
    for g in np.unique(rec.sorted_sigma):
        assert set_sizes([keyed], lex.key(g, 0.0)[0])[0] == res.gap_answer_set(g).size


@pytest.mark.parametrize("seed", range(10))
def test_tie_break_cuts_inside_a_tie_and_keeps_the_top_node(seed):
    kg, scorer, topic, mask, _ = _case(seed, with_mask=True)
    res = bottleneck_dp(kg, topic, scorer, H, lam=0.8, answer_mask=mask)
    if not np.isfinite(res.sigma_min):
        return
    rec = to_gap(record_from_dp(res, np.array([], dtype=np.int64), tie_break=True))
    assert rec.sorted_sigma[0] == 0 and rec.sorted_tie[0] == 0           # the top node is the pair (0, 0)
    lex = LexKey([rec])
    keyed = lex.apply([rec])[0]
    gaps = res.sigma - res.sigma_min
    f = np.flatnonzero(np.isfinite(res.sigma))
    top_tie = res.tie[f[np.lexsort((res.tie[f], res.sigma[f]))[0]]]
    for g, t in zip(rec.sorted_sigma, rec.sorted_tie):
        expected = sum(_lex_le((gaps[v], res.tie[v] - top_tie), (g, t)) for v in np.flatnonzero(np.isfinite(gaps)))
        assert set_sizes([keyed], lex.key(g, t)[0])[0] == expected
        assert expected >= 1


@pytest.mark.parametrize("seed", range(10))
def test_pruned_search_gives_the_same_tie_broken_sets(seed):
    kg, scorer, topic, mask, feasible = _case(seed, with_mask=True, with_feasible=True)
    full = to_gap(record_from_dp(bottleneck_dp(kg, topic, scorer, H, lam=0.8, answer_mask=mask, feasible=feasible),
                                 np.array([], dtype=np.int64), tie_break=True))
    lex = LexKey([full])
    keyed = lex.apply([full])[0]
    for g, t in zip(full.sorted_sigma, full.sorted_tie):
        key = lex.key(g, t)[0]
        res = bottleneck_dp(kg, topic, scorer, H, gap_tau=lex.gap(key), lam=0.8, answer_mask=mask, feasible=feasible)
        pruned = lex.apply([to_gap(record_from_dp(res, np.array([], dtype=np.int64), tie_break=True))])[0]
        assert set_sizes([pruned], key)[0] == set_sizes([keyed], key)[0]


def test_coverage_with_heavy_ties_is_exact():
    """Exchangeable questions whose gaps tie a lot (3 values) but whose tie scores do not: the pair score is
    continuous in effect, so coverage is 1 - alpha up to 1 / (n + 1), where the gap alone over-covers."""
    rng = np.random.default_rng(0)
    n, trials, alpha = 99, 3000, 0.25
    hit_pair, hit_gap = [], []
    for _ in range(trials):
        g = rng.integers(0, 3, n + 1).astype(float)
        t = rng.random(n + 1)
        recs = [SigmaRecord(sigma_star=g[i], star_tie=t[i], sorted_sigma=np.array([g[i]]), sorted_tie=np.array([t[i]]),
                            states_expanded=0) for i in range(n + 1)]
        lex = LexKey(recs)
        keyed = lex.apply(recs)
        tau = conformal_quantile([r.sigma_star for r in keyed[:n]], alpha)
        hit_pair.append(covered([keyed[n].sigma_star], tau)[0])
        hit_gap.append(covered([g[n]], conformal_quantile(g[:n], alpha))[0])
    k = math.ceil((n + 1) * (1 - alpha))
    assert abs(np.mean(hit_pair) - k / (n + 1)) < 0.02
    assert np.mean(hit_gap) > np.mean(hit_pair) + 0.05
