import numpy as np
import pytest

from chase.baselines.bootstrap import ChaseCurve, bootstrap, cluster_weights, holm, ratios
from chase.search.dp import INF


def _chase(rng, n=300):
    """Keyed CHASE test records: sorted integer keys per question, g* one of them (or INF), top = smallest key 0."""
    keys = [np.sort(np.r_[0, rng.integers(1, 40, rng.integers(1, 30))]).astype(float) for _ in range(n)]
    star = np.array([k[rng.integers(0, len(k))] if rng.random() < 0.9 else INF for k in keys])
    return {"chase_star": star, "chase_top": np.zeros(n), "chase_keys": np.concatenate(keys),
            "chase_offsets": np.r_[0, np.cumsum([len(k) for k in keys])]}


def _at(d, t):
    """CHASE itself at threshold t: covered flags and sizes."""
    k, o = d["chase_keys"], d["chase_offsets"]
    size = np.array([np.searchsorted(k[o[i]:o[i + 1]], t, side="right") for i in range(len(o) - 1)], dtype=float)
    return d["chase_star"] <= t, size


def _seed(rng, baseline, n=300, groups=None):
    d = _chase(rng, n)
    cov, size = baseline(d)
    d.update(methods=np.array(["B"]), alphas=np.array([0.3]), cov=cov[None, None], size=size[None, None],
             groups=np.arange(n) // 3 if groups is None else groups)
    return d


def _curve(d):
    return ChaseCurve(d["chase_star"], d["chase_keys"], d["chase_offsets"], d["chase_top"])


def test_cluster_weights_draw_whole_groups():
    groups = np.repeat(np.arange(7), [1, 3, 2, 5, 1, 4, 2])
    W = cluster_weights(groups, 200, np.random.default_rng(0))
    for g in range(7):
        assert (W[:, groups == g] == W[:, groups == g][:, :1]).all()               # a group is drawn as a whole
    per_group = np.stack([W[:, np.flatnonzero(groups == g)[0]] for g in range(7)], axis=1)
    assert (per_group.sum(1) == 7).all()                                            # G draws per resample


@pytest.mark.parametrize("seed", range(5))
def test_chase_against_itself_gives_ratio_one(seed):
    rng = np.random.default_rng(seed)
    d = _seed(rng, lambda d: _at(d, 12.0))
    W = np.vstack([np.ones(300), cluster_weights(d["groups"], 300, rng)])
    r = ratios(W, d["cov"][0, 0].astype(float), d["size"][0, 0], _curve(d))
    assert r["theta"][0] == pytest.approx(1.0)
    assert (r["theta"] <= 1 + 1e-9).all()                       # never larger: a smaller threshold may reach the ECR
    assert (r["ecr_c"] >= r["ecr_b"] - 1e-12).all()             # matched coverage is never below the baseline's


def test_doubled_sizes_are_significant():
    rng = np.random.default_rng(0)
    seeds = [_seed(rng, lambda d: (_at(d, 12.0)[0], 2 * _at(d, 12.0)[1])) for _ in range(3)]
    for s in seeds[1:]:
        s["groups"] = seeds[0]["groups"]
    (p,) = bootstrap(seeds, n_resamples=500)
    assert p["ratio"] == pytest.approx(0.5)
    assert p["ci_high"] < 1 and p["verdict"] == "CHASE smaller"


def test_same_apss_is_not_significant():
    rng = np.random.default_rng(1)
    seeds = [_seed(rng, lambda d: (_at(d, 12.0)[0], rng.permutation(_at(d, 12.0)[1]))) for _ in range(3)]
    for s in seeds[1:]:
        s["groups"] = seeds[0]["groups"]
    (p,) = bootstrap(seeds, n_resamples=500)
    assert p["ratio"] == pytest.approx(1.0)
    assert p["ci_low"] < 1 < p["ci_high"] and p["verdict"] == "n.s."


def test_levels_below_the_floor_are_not_tested():
    rng = np.random.default_rng(2)
    seeds = [_seed(rng, lambda d: (np.zeros(300, bool), np.ones(300))) for _ in range(3)]   # 0% coverage
    for s in seeds[1:]:
        s["groups"] = seeds[0]["groups"]
    (p,) = bootstrap(seeds, n_resamples=200)
    assert p["floor"] and "verdict" not in p


def test_abstaining_levels_are_skipped():
    rng = np.random.default_rng(3)
    seeds = [_seed(rng, lambda d: (np.zeros(300, bool), np.full(300, np.nan)))]
    assert bootstrap(seeds, n_resamples=50) == []


def test_holm():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_per_question_rows_average_to_the_official_metrics(tmp_path):
    """per_question.npz rows reproduce evaluate()'s official ECR / APSS for every baseline and level."""
    from chase.baselines import beam, per_hop, topk
    from chase.baselines.run import evaluate, per_question, save_per_question
    from chase.train.evaluate import LexKey
    from chase.search.dp import bottleneck_dp
    from chase.train.evaluate import record_from_dp, to_gap
    from tests.test_baselines import H, _case

    def records(seeds):
        out = {"chase": [], "topk": [], "comp": [], "table": [], "beam4": []}
        for s in seeds:
            rng, kg, scorer, topic, mask = _case(s)
            gold = rng.choice(np.flatnonzero(mask), 2, replace=False)
            res = bottleneck_dp(kg, topic, scorer, H, lam=0.8, answer_mask=mask, keep_paths=True)
            out["chase"].append(to_gap(record_from_dp(res, gold, tie_break=True)))
            out["topk"].append(topk.topk_record(res, gold))
            out["comp"].append(per_hop.walk_components(res, scorer, gold, H, 0.8))
            out["table"].append(per_hop.per_hop_table(kg, topic, scorer, res, gold, H, 0.8, mask))
            score, states, edges = beam.beam_scores(kg, topic, scorer, H, 4, 4, 0.8, mask)
            out["beam4"].append(beam.beam_record(score, gold, states, edges, 0.0))
        out["comp"] = np.stack(out["comp"])
        return out

    cal, test = records(range(40)), records(range(40, 70))
    alphas = [0.2, 0.5]
    methods = evaluate(cal, test, [4], n_splits=5, n_splits_hop=5, alphas=alphas)
    pq = per_question(cal, test, [4], alphas)
    assert set(pq) == set(methods) - {"CHASE"}
    for name, rows in pq.items():
        for (cov, size), off in zip(rows, methods[name]["official"]):
            if off["feasible"]:
                assert cov.mean() == pytest.approx(off["ecr"]) and size.mean() == pytest.approx(off["apss"])
            else:
                assert np.isnan(size).all()

    lex = LexKey(cal["chase"] + test["chase"])                                     # end to end: save, load, bootstrap
    save_per_question(tmp_path / "pq.npz", pq, lex.apply(test["chase"]), np.arange(30) // 2, alphas)
    d = dict(np.load(tmp_path / "pq.npz"))
    pts = bootstrap([d, d], n_resamples=100)
    assert pts and all(np.isfinite(p["ratio"]) and p["ci_low"] <= p["ratio"] <= p["ci_high"] + 1e-9 for p in pts)
