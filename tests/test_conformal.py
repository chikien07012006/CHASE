import math

import numpy as np
import pandas as pd
import pytest

from chase.conformal.calibrate import CalibrationProblem, CalibrationScores, calibration_scores
from chase.conformal.split import INF, conformal_quantile, covered, crc_threshold, miss_rate
from chase.kg.graph import KG, build_csr
from chase.search.dp import bottleneck_dp


# ---------------------------------------------------------------- split conformal

def test_quantile_picks_kth_smallest():
    scores = np.array([0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6])   # n = 9
    assert conformal_quantile(scores, 0.2) == 0.8                        # k = ceil(10 * 0.8) = 8
    assert conformal_quantile(scores, 0.5) == 0.5                        # k = 5


def test_quantile_infeasible_levels():
    assert conformal_quantile(np.arange(9.0), 0.05) == INF               # k = 10 > n
    assert conformal_quantile(np.array([0.1, 0.2, INF, INF]), 0.4) == INF  # k-th score is INF
    assert conformal_quantile(np.array([]), 0.1) == INF
    with pytest.raises(ValueError):
        conformal_quantile(np.arange(5.0), 1.0)


@pytest.mark.parametrize("alpha,n", [(0.1, 50), (0.2, 200), (0.3, 30)])
def test_split_conformal_coverage_is_valid_and_tight(alpha, n):
    rng = np.random.default_rng(0)
    reps = 20000
    scores = rng.random((reps, n + 1))
    hits = np.array([covered(s[-1:], conformal_quantile(s[:-1], alpha))[0] for s in scores])
    se = math.sqrt(alpha * (1 - alpha) / reps)
    assert 1 - alpha - 4 * se <= hits.mean() <= 1 - alpha + 1 / (n + 1) + 4 * se


def test_unreachable_questions_never_count_as_covered():
    # 30% of questions cannot reach an answer: asking for 90% must be reported infeasible
    rng = np.random.default_rng(1)
    s = np.where(rng.random(500) < 0.3, INF, rng.random(500))
    assert conformal_quantile(s, 0.1) == INF
    assert not covered(np.array([INF]), INF)[0]
    tau = conformal_quantile(s, 0.5)
    assert np.isfinite(tau)


# ---------------------------------------------------------------- conformal risk control

def _random_questions(rng, n, p_unreachable=0.05):
    out = []
    for _ in range(n):
        m = int(rng.integers(1, 12))
        a = rng.random(m)
        a[rng.random(m) < p_unreachable] = INF
        out.append(a)
    return out


def test_crc_returns_smallest_valid_threshold():
    rng = np.random.default_rng(2)
    qs, alpha = _random_questions(rng, 80), 0.2
    tau = crc_threshold(qs, alpha)
    n = len(qs)
    risk = lambda t: n / (n + 1) * miss_rate(qs, t).mean() + 1 / (n + 1)
    assert risk(tau) <= alpha + 1e-12
    smaller = np.unique(np.concatenate([q[np.isfinite(q)] for q in qs]))
    smaller = smaller[smaller < tau]
    assert all(risk(t) > alpha for t in smaller)


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.3])
def test_crc_controls_expected_miss_rate(alpha):
    rng = np.random.default_rng(3)
    misses = []
    for _ in range(3000):
        qs = _random_questions(rng, 61, p_unreachable=0.02)
        tau = crc_threshold(qs[:-1], alpha)
        misses.append(miss_rate(qs[-1:], tau)[0])
    assert np.mean(misses) <= alpha + 0.02


def test_crc_infeasible_when_too_many_answers_unreachable():
    qs = [np.array([INF, INF, 0.1]) for _ in range(50)]     # at best 1/3 recall
    assert crc_threshold(qs, 0.2) == INF
    with pytest.raises(ValueError):
        crc_threshold([np.array([])], 0.2)


# ---------------------------------------------------------------- glue with the DP

class _ConstScorer:
    def __init__(self, step, stop):
        self.step, self.stop = step, stop

    def edge_score(self, h, src, rel, dst):
        return self.step[dst]

    def stop_score(self, h, nodes):
        return self.stop[nodes]


def _line_kg(n=6):
    src = np.arange(n - 1)
    indptr, indices, rel = build_csr(np.concatenate([src, src + 1]), np.concatenate([src + 1, src]),
                                     np.zeros(2 * (n - 1), dtype=np.int64), n)
    return KG(indptr=indptr, indices=indices, rel=rel, node_type=np.zeros(n, np.int8), type_names=["t"],
              relation_names=["r"], nodes=pd.DataFrame({"name": [str(i) for i in range(n)]}))


def test_calibration_scores_match_dp_and_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    kg = _line_kg()
    problems, expected = [], []
    for _ in range(5):
        sc = _ConstScorer(rng.random(6), rng.random(6))
        p = CalibrationProblem(kg=kg, topic=[0], answers=[2, 3], scorer=sc, num_absent=1)
        res = bottleneck_dp(kg, [0], sc, 3)
        expected.append(res.sigma[[2, 3]].min())
        problems.append(p)
    scores = calibration_scores(problems, H=3)
    np.testing.assert_array_equal(scores.sigma_star, expected)
    assert all(len(a) == 3 and np.isinf(a[-1]) for a in scores.answer_sigma)   # absent answer counted as missed

    scores.save(tmp_path / "s.npz")
    back = CalibrationScores.load(tmp_path / "s.npz")
    np.testing.assert_array_equal(back.sigma_star, scores.sigma_star)
    for a, b in zip(back.answer_sigma, scores.answer_sigma):
        np.testing.assert_array_equal(a, b)
