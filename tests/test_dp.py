import numpy as np
import pandas as pd
import pytest

from chase.kg.graph import KG, build_csr
from chase.search.dp import INF, ZeroScorer, bottleneck_dp, reachable_within

N, R, H = 25, 3, 3


class TableScorer:
    """Scores that depend only on (h, rel, dst) / (h, node), rounded to force ties."""

    def __init__(self, rng):
        self.step = np.round(rng.random((H + 1, R, N)), 1)
        self.stop = np.round(rng.random((H + 1, N)), 1)

    def edge_score(self, h, src, rel, dst):
        return self.step[h, rel, dst]

    def stop_score(self, h, nodes):
        return self.stop[h, nodes]


def _random_kg(rng, num_edges=55) -> KG:
    src, dst = rng.integers(0, N, num_edges), rng.integers(0, N, num_edges)
    rel = rng.integers(0, R, num_edges)
    indptr, indices, rel = build_csr(src, dst, rel, N)
    return KG(indptr=indptr, indices=indices, rel=rel, node_type=np.zeros(N, np.int8),
              type_names=["t"], relation_names=[f"r{i}" for i in range(R)],
              nodes=pd.DataFrame({"name": [str(i) for i in range(N)]}))


def _brute_force_sigma(kg, topic, scorer, lam, answer_mask, feasible):
    """Enumerate every walk of length 1..H and take the min of max(step scores, lam * stop)."""
    sigma = np.full(N, INF)

    def walk(node, h, worst):
        if h == H:
            return
        nbrs, rels = kg.neighbors(node)
        for v, r in zip(nbrs.tolist(), rels.tolist()):
            if feasible is not None and not feasible(h + 1, np.array([v]))[0]:
                continue
            w = max(worst, scorer.step[h + 1, r, v])
            if answer_mask[v] and v not in topic:
                sigma[v] = min(sigma[v], max(w, lam * scorer.stop[h + 1, v]))
            walk(v, h + 1, w)

    for t in topic:
        walk(t, 0, -INF)
    return sigma


def _case(seed, with_mask=False, with_feasible=False):
    rng = np.random.default_rng(seed)
    kg, scorer = _random_kg(rng), TableScorer(rng)
    topic = sorted(set(rng.integers(0, N, 2).tolist()))
    answer_mask = rng.random(N) < 0.7 if with_mask else np.ones(N, bool)
    blocked = rng.random((H + 1, N)) < 0.2
    feasible = (lambda h, nodes: ~blocked[h, nodes]) if with_feasible else None
    return kg, scorer, topic, answer_mask, feasible


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("with_mask,with_feasible", [(False, False), (True, False), (True, True)])
def test_sigma_matches_brute_force(seed, with_mask, with_feasible):
    kg, scorer, topic, mask, feasible = _case(seed, with_mask, with_feasible)
    lam = 0.8
    res = bottleneck_dp(kg, topic, scorer, H, lam=lam, answer_mask=mask, feasible=feasible)
    np.testing.assert_array_equal(res.sigma, _brute_force_sigma(kg, topic, scorer, lam, mask, feasible))


@pytest.mark.parametrize("seed", range(10))
def test_sigma_star_is_min_over_many_answers(seed):
    # several gold answers, some masked out, unreachable or topic entities: sigma* = min over the reachable ones
    kg, scorer, topic, mask, _ = _case(seed, with_mask=True)
    brute = _brute_force_sigma(kg, topic, scorer, 1.0, mask, None)
    res = bottleneck_dp(kg, topic, scorer, H, answer_mask=mask)
    answers = np.random.default_rng(seed).choice(N, 6, replace=False).tolist() + list(topic)
    assert res.sigma_star(answers) == brute[answers].min()
    unreachable = [v for v in range(N) if not np.isfinite(brute[v])]
    assert res.sigma_star(unreachable) == INF


@pytest.mark.parametrize("seed", range(10))
def test_pruned_run_returns_same_answer_set_and_is_nested(seed):
    kg, scorer, topic, mask, _ = _case(seed, with_mask=True)
    full = bottleneck_dp(kg, topic, scorer, H, answer_mask=mask)
    previous = set()
    for tau in np.round(np.arange(0.0, 1.05, 0.1), 1):
        pruned = bottleneck_dp(kg, topic, scorer, H, tau=tau, answer_mask=mask)
        expected = set(full.answer_set(tau).tolist())
        assert set(pruned.answer_set(tau).tolist()) == expected
        assert previous <= expected
        assert pruned.edges_scored <= full.edges_scored
        previous = expected


@pytest.mark.parametrize("seed", range(10))
def test_evidence_path_is_a_real_walk_with_score_sigma(seed):
    kg, scorer, topic, mask, _ = _case(seed, with_mask=True)
    lam = 0.8
    res = bottleneck_dp(kg, topic, scorer, H, lam=lam, answer_mask=mask, keep_paths=True)
    for v in np.flatnonzero(np.isfinite(res.sigma)):
        path = res.evidence_path(int(v))
        assert len(path) == res.best_hop[v] >= 1
        assert path[0][0] in topic and path[-1][2] == v
        worst = -INF
        for h, (u, r, w) in enumerate(path, start=1):
            nbrs, rels = kg.neighbors(u)
            assert any(x == w and y == r for x, y in zip(nbrs, rels)), "step is not an edge"
            worst = max(worst, scorer.step[h, r, w])
        assert max(worst, lam * scorer.stop[len(path), v]) == res.sigma[v]


def test_topic_is_never_an_answer():
    kg, scorer, topic, mask, _ = _case(3)
    res = bottleneck_dp(kg, topic, scorer, H, answer_mask=mask)
    assert np.isinf(res.sigma[topic]).all()


def test_edge_cases():
    kg, scorer, _, _, _ = _case(4)
    assert np.isinf(bottleneck_dp(kg, [], scorer, H).sigma).all()
    assert np.isinf(bottleneck_dp(kg, [0], scorer, H, answer_mask=np.zeros(N, bool)).sigma).all()
    assert bottleneck_dp(kg, [0], scorer, H).sigma_star([]) == INF


def test_zero_scorer_is_reachability():
    kg, _, topic, _, _ = _case(5)
    res = bottleneck_dp(kg, topic, ZeroScorer(), H)
    for target in range(N):
        if target in topic:
            continue
        frontier, seen_within = set(topic), False
        for _ in range(H):
            frontier = {int(v) for u in frontier for v in kg.neighbors(u)[0]}
            seen_within |= target in frontier
        assert reachable_within(res, [target], H) == seen_within


@pytest.mark.parametrize("seed", range(10))
def test_gap_pruned_run_returns_same_gap_set_and_sigma_min(seed):
    kg, scorer, topic, mask, feasible = _case(seed, with_mask=True, with_feasible=True)
    full = bottleneck_dp(kg, topic, scorer, H, lam=0.8, answer_mask=mask, feasible=feasible)
    for gap in np.round(np.arange(0.0, 1.05, 0.1), 1):
        pruned = bottleneck_dp(kg, topic, scorer, H, gap_tau=gap, lam=0.8, answer_mask=mask, feasible=feasible)
        assert pruned.sigma_min == full.sigma_min
        assert set(pruned.gap_answer_set(gap).tolist()) == set(full.gap_answer_set(gap).tolist())
        assert pruned.edges_scored <= full.edges_scored
        if np.isfinite(full.sigma_min):
            assert full.gap_answer_set(gap).size >= 1          # the most confident node always has gap 0
