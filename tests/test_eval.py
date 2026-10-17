import numpy as np
import pandas as pd

from chase.eval import plots
from chase.eval.freebase import comparison_table, metrics_at, validity_resplits
from chase.eval.reference import reference_for
from chase.train.evaluate import SigmaRecord


def _rec(star, sizes=(0.1, 0.4, 0.8), top_gold=True):
    s = np.array(sorted(sizes))
    return SigmaRecord(sigma_star=star, sorted_sigma=s, states_expanded=10, top_sigma=float(s[0]), top_is_gold=top_gold,
                       best_hop=2, topic_degree=5)


def test_metrics_at_matches_hand_computation():
    cal = [_rec(x) for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]   # n = 9, alpha 0.2 -> k = 8 -> tau = 0.8
    test = [_rec(0.3), _rec(0.85), _rec(np.inf, top_gold=False)]
    r = metrics_at(cal, test, 0.2)
    assert r["feasible"] and r["tau"] == 0.8
    assert abs(r["ecr"] - 1 / 3) < 1e-12                 # only the first question is covered
    assert r["apss"] == 3.0                               # every node (0.1, 0.4, 0.8) is <= 0.8
    assert abs(r["hit1"] - 2 / 3) < 1e-12
    assert not metrics_at(cal, test, 0.05)["feasible"]    # k = 10 > n


def test_validity_resplits_are_centred_on_the_target():
    rng = np.random.default_rng(0)
    cal = [_rec(x) for x in rng.random(300)]
    test = [_rec(x) for x in rng.random(1500)]
    summary, samples = validity_resplits(cal, test, [0.1, 0.3], n_splits=400)
    for a in (0.1, 0.3):
        s = summary[str(a)]
        assert abs(s["ecr_mean"] - (1 - a)) < 0.01
        assert abs(s["ecr_mean"] - s["theory_mean"]) < 0.01 and len(samples[a]) == 400


def test_comparison_table_marks_missing_levels():
    rows = [{"alpha": 0.1, "feasible": True, "ecr": 0.9, "apss": 12.0},
            {"alpha": 0.5, "feasible": True, "ecr": 0.5, "apss": 3.0},
            {"alpha": 0.05, "feasible": False}]
    md = comparison_table(rows, reference_for("webqsp"), rho=0.95)
    assert "CPR (reported)" in md and "**CHASE**" in md
    cpr_line = next(line for line in md.splitlines() if line.startswith("| CPR"))
    assert cpr_line.count("—") == 2 and "61.4 / 7.14" in cpr_line   # no CPR number at 0.1 or 0.05
    chase_line = next(line for line in md.splitlines() if "**CHASE**" in line)
    assert "90.0 / 12.00" in chase_line and chase_line.count("—") == 1


def test_all_figures_are_written(tmp_path):
    rng = np.random.default_rng(1)
    cal = [_rec(x) for x in rng.random(200)]
    test = [_rec(x, sizes=rng.random(30)) for x in rng.random(600)]
    alphas = [0.1, 0.3, 0.5]
    rows = []
    for a in alphas:
        r = metrics_at(cal, test, a)
        r.update(states_mean=50.0 * (1 - a), states_unpruned_mean=80.0)
        rows.append(r)
    validity, samples = validity_resplits(cal, test, alphas, n_splits=200)
    sizes = {a: np.array([np.searchsorted(t.sorted_sigma, r["tau"], side="right") for t in test], float)
             for a, r in zip(alphas, rows)}
    per_q = pd.DataFrame({"best_hop": rng.integers(1, 3, 600), "states_expanded": rng.integers(1, 500, 600),
                          "topic_degree": rng.integers(1, 3000, 600)})
    history = {"epochs": [{"stage": 1, "val": {"apss": 60.0, "ecr": 0.9}}, {"stage": 2, "val": {"apss": 40.0, "ecr": 0.91}}]}
    names = plots.make_all(tmp_path, rows, validity, samples, sizes, per_q, reference_for("webqsp"), 0.95, 0.1, history)
    assert names == ["fig1_validity_hist.png", "fig2_coverage_vs_target.png", "fig3_apss_efficiency.png",
                     "fig4_set_size_distribution.png", "fig5_cost.png", "fig6_training_curves.png"]
    assert all((tmp_path / n.replace(".png", ".pdf")).exists() for n in names)
