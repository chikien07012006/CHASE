import itertools

import numpy as np
import pandas as pd
import pytest

from chase.baselines import beam, per_hop, topk
from chase.kg.graph import KG, build_csr
from chase.search.dp import INF, bottleneck_dp

N, R, H = 25, 3, 2


class TableScorer:
    """Scores that depend only on (h, rel, dst) / (h, node), rounded to force ties."""

    def __init__(self, rng):
        self.step = np.round(rng.random((H + 1, R, N)), 1)
        self.stop = np.round(rng.random((H + 1, N)), 1)

    def edge_score(self, h, src, rel, dst):
        return self.step[h, rel, dst]

    def stop_score(self, h, nodes):
        return self.stop[h, nodes]


def _random_kg(rng, num_edges=60) -> KG:
    src, dst = rng.integers(0, N, num_edges), rng.integers(0, N, num_edges)
    indptr, indices, rel = build_csr(src, dst, rng.integers(0, R, num_edges), N)
    return KG(indptr=indptr, indices=indices, rel=rel, node_type=np.zeros(N, np.int8), type_names=["t"],
              relation_names=[f"r{i}" for i in range(R)], nodes=pd.DataFrame({"name": [str(i) for i in range(N)]}))


def _case(seed):
    rng = np.random.default_rng(seed)
    kg, scorer = _random_kg(rng), TableScorer(rng)
    topic = rng.choice(N, 2, replace=False)
    mask = rng.random(N) < 0.8
    return rng, kg, scorer, topic, mask


def _walks(kg, topic, scorer, lam):
    """Every walk of 1..H hops: (end node, [step scores], lam * stop)."""
    out = []

    def go(node, steps):
        if len(steps) == H:
            return
        nbrs, rels = kg.neighbors(node)
        for v, r in zip(nbrs, rels):
            s = steps + [float(scorer.edge_score(len(steps) + 1, None, np.array([r]), np.array([v]))[0])]
            out.append((int(v), s, lam * float(scorer.stop_score(len(s), np.array([v]))[0])))
            go(v, s)

    for t in np.unique(topic):
        go(int(t), [])
    return out


@pytest.mark.parametrize("seed", range(20))
def test_unbounded_beam_equals_the_dp(seed):
    _, kg, scorer, topic, mask = _case(seed)
    res = bottleneck_dp(kg, topic, scorer, H, lam=0.7, answer_mask=mask)
    score, _, _ = beam.beam_scores(kg, topic, scorer, H, branch=10 ** 6, active=10 ** 6, lam=0.7, answer_mask=mask)
    np.testing.assert_array_equal(score, res.sigma)


@pytest.mark.parametrize("seed", range(20))
def test_narrow_beam_only_loses_paths(seed):
    _, kg, scorer, topic, mask = _case(seed)
    res = bottleneck_dp(kg, topic, scorer, H, lam=0.7, answer_mask=mask)
    score, _, _ = beam.beam_scores(kg, topic, scorer, H, branch=2, active=3, lam=0.7, answer_mask=mask)
    assert (score >= res.sigma).all()
    assert np.isfinite(score).sum() <= 2 * 3


def test_topk_rank_is_position_of_best_gold_by_sigma_then_id():
    _, kg, scorer, topic, mask = _case(3)
    res = bottleneck_dp(kg, topic, scorer, H, answer_mask=mask)
    finite = np.flatnonzero(np.isfinite(res.sigma))
    gold = finite[[2, 5]]
    rec = topk.topk_record(res, gold)
    order = sorted(finite, key=lambda v: (res.sigma[v], v))
    assert rec.sigma_star == 1 + min(order.index(g) for g in gold)
    np.testing.assert_array_equal(rec.sorted_sigma, np.arange(1, finite.size + 1))
    assert topk.topk_record(res, np.array([], dtype=np.int64)).sigma_star == INF


@pytest.mark.parametrize("seed", range(20))
def test_per_hop_table_matches_brute_force(seed):
    rng, kg, scorer, topic, mask = _case(seed)
    lam = 0.7
    res = bottleneck_dp(kg, topic, scorer, H, lam=lam, answer_mask=mask)
    answerable = mask.copy()
    answerable[topic] = False
    gold = rng.choice(N, 3, replace=False)
    table = per_hop.per_hop_table(kg, topic, scorer, res, gold, H, lam, mask)
    walks = _walks(kg, topic, scorer, lam)
    taus = rng.choice(np.round(np.arange(0, 1.05, 0.1), 1), size=(30, H + 1))
    cov, size = table.evaluate(taus)
    for t, c, s in zip(taus, cov, size):
        returned = {v for v, steps, stop in walks
                    if answerable[v] and all(x <= t[h] for h, x in enumerate(steps)) and stop <= t[H]}
        assert s == len(returned)
        assert c == bool(returned & set(gold.tolist()))


@pytest.mark.parametrize("seed", range(20))
def test_walk_components_rebuild_sigma_star(seed):
    rng, kg, scorer, topic, mask = _case(seed)
    res = bottleneck_dp(kg, topic, scorer, H, lam=0.7, answer_mask=mask, keep_paths=True)
    gold = rng.choice(N, 3, replace=False)
    comp = per_hop.walk_components(res, scorer, gold, H, lam=0.7)
    star = res.sigma_star(gold)
    if np.isfinite(star):
        assert np.nanmax(comp) == star
    else:
        assert np.isinf(comp).all()


def test_bonferroni_splits_alpha_over_components_and_skips_unused_hops():
    rng = np.random.default_rng(0)
    comp = rng.random((200, 3))
    comp[:80, 1] = np.nan                                  # 1-hop walks: no hop-2 score
    from chase.conformal.split import conformal_quantile
    tau = per_hop.thresholds(comp, 0.3, "bonferroni")
    assert tau[1] == conformal_quantile(comp[80:, 1], 0.1)
    assert tau[0] == conformal_quantile(comp[:, 0], 0.1)
    assert (per_hop.thresholds(comp, 0.3, "per_hop") <= tau).all()


def test_bonferroni_covers_and_per_hop_can_miss_in_simulation():
    """Independent component scores: a walk survives iff all three pass, so per-hop alpha gives about (1-alpha)^3."""
    rng = np.random.default_rng(1)
    alpha, n, reps = 0.2, 300, 400
    hits = {a: [] for a in per_hop.ALLOCATIONS}
    for _ in range(reps):
        cal, test = rng.random((n, 3)), rng.random((n, 3))
        for a in per_hop.ALLOCATIONS:
            hits[a].append(np.mean((test <= per_hop.thresholds(cal, alpha, a)).all(1)))
    assert np.mean(hits["bonferroni"]) >= 1 - alpha
    assert np.mean(hits["per_hop"]) == pytest.approx((1 - alpha) ** 3, abs=0.02)


def test_cpr_rule_matches_cpr_code():
    s = np.array([0.3, 0.1, INF, 0.5, 0.2])
    expected = np.quantile(np.where(np.isfinite(s), s, 1e9), min(6 * 0.7 / 5, 1.0))
    assert beam.cpr_quantile(s, 0.3) == expected
    # empty set -> the single best retrieved entity
    cal = [beam.beam_record(np.array([0.1, 0.4, INF]), np.array([0]), 0, 0, 0.0) for _ in range(9)]
    rec = beam.beam_record(np.array([0.5, 0.9, INF]), np.array([0]), 0, 0, 0.0)
    row = beam.cpr_metrics_at(cal, [rec], 0.5)
    assert row["apss"] == 1 and row["ecr"] == 1


def test_gap_record_is_sigma_measured_from_the_best_node():
    from chase.train.evaluate import SigmaRecord, set_sizes, to_gap
    rec = SigmaRecord(sigma_star=0.7, sorted_sigma=np.array([0.4, 0.7, 0.9]), states_expanded=0, top_sigma=0.4)
    g = to_gap(rec)
    assert g.sigma_star == pytest.approx(0.3) and g.top_sigma == 0
    np.testing.assert_allclose(g.sorted_sigma, [0.0, 0.3, 0.5])
    # set at gap threshold t == set at sigma threshold sigma_min + t
    assert set_sizes([g], 0.35)[0] == set_sizes([rec], 0.75)[0] == 2
    empty = SigmaRecord(sigma_star=INF, sorted_sigma=np.array([]), states_expanded=0)
    assert to_gap(empty).sigma_star == INF
