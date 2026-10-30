"""Official evaluation of a trained CHASE scorer on a RoG (Freebase) store.

calibrate on the cal split (unpruned DP) -> tau_hat per alpha -> metrics on the test split ->
pruned search at tau_hat for real cost -> 1000 cal/test re-splits for validity -> tables + figures.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from chase.conformal.split import INF, conformal_quantile, covered
from chase.eval import plots
from chase.eval.reference import reference_for
from chase.train.evaluate import SigmaRecord, collect_sigmas, set_sizes

ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
GREEN, YELLOW, CYAN, BOLD, DIM, RESET = "\033[32m", "\033[33m", "\033[36m", "\033[1m", "\033[2m", "\033[0m"


# ------------------------------------------------------------------ metrics (pure functions, tested)

def metrics_at(cal: list[SigmaRecord], test: list[SigmaRecord], alpha: float) -> dict:
    """tau_hat from the calibration scores, then coverage / size / Hit@1 on the test questions."""
    tau = conformal_quantile([r.sigma_star for r in cal], alpha)
    row = {"alpha": alpha, "target": 1 - alpha, "tau": tau, "feasible": bool(np.isfinite(tau))}
    if not row["feasible"]:
        return row
    sizes = set_sizes(test, tau)
    ecr = float(covered([r.sigma_star for r in test], tau).mean())
    apss = float(sizes.mean())
    row.update(ecr=ecr, apss=apss, size_median=float(np.median(sizes)), size_p90=float(np.percentile(sizes, 90)),
               empty_share=float(np.mean(sizes == 0)), coverage_efficiency=100 * ecr / apss if apss else float("nan"),
               hit1=float(np.mean([r.top_is_gold and r.top_sigma <= tau for r in test])))
    return row


def resplit_samples(cal: list[SigmaRecord], test: list[SigmaRecord], alphas, n_splits: int = 1000,
                    seed: int = 0) -> dict[float, dict[str, np.ndarray]]:
    """Pool cal + test, re-split with the same sizes n_splits times; per alpha and split: ECR, APSS, empty share.

    tau_hat comes from the calibration part of each split (NaN metrics where tau_hat = INF).
    """
    recs = cal + test
    pooled = np.array([r.sigma_star for r in recs])
    n_cal, n_test = len(cal), len(test)
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(len(pooled)) for _ in range(n_splits)])
    cal_sorted = np.sort(pooled[perms[:, :n_cal]], axis=1)
    in_test = np.zeros((n_splits, len(pooled)), dtype=bool)
    np.put_along_axis(in_test, perms[:, n_cal:], True, axis=1)
    out = {}
    for a in alphas:
        k = math.ceil((n_cal + 1) * (1 - a))
        tau = cal_sorted[:, k - 1] if k <= n_cal else np.full(n_splits, INF)
        ok = np.isfinite(tau)
        hit = np.isfinite(pooled)[None, :] & (pooled[None, :] <= tau[:, None])
        sizes = np.stack([np.searchsorted(r.sorted_sigma, tau, side="right") for r in recs], axis=1)
        out[a] = {"ecr": np.where(ok, (hit & in_test).sum(1) / n_test, np.nan),
                  "apss": np.where(ok, (sizes * in_test).sum(1) / n_test, np.nan),
                  "empty": np.where(ok, ((sizes == 0) & in_test).sum(1) / n_test, np.nan)}
    return out


def _summarise(s: dict[str, np.ndarray]) -> dict:
    ok = np.isfinite(s["ecr"])
    mean = lambda x: float(x[ok].mean()) if ok.any() else float("nan")
    return {"infeasible_share": float(np.mean(~ok)), "ecr_mean": mean(s["ecr"]), "apss_mean": mean(s["apss"]),
            "apss_std": float(s["apss"][ok].std()) if ok.any() else float("nan"), "empty_mean": mean(s["empty"])}


def validity_resplits(cal: list[SigmaRecord], test: list[SigmaRecord], alphas, n_splits: int = 1000,
                      seed: int = 0) -> tuple[dict, dict[float, np.ndarray]]:
    """Protocol P2: pool cal + test, re-split with the same sizes n_splits times, record ECR (and APSS).

    Theory for split conformal (continuous scores): coverage given the calibration set is
    Beta(k, n+1-k), k = ceil((n+1)(1-alpha)); the test ECR adds Binomial noise over n_test questions.
    """
    n_cal, n_test = len(cal), len(test)
    raw = resplit_samples(cal, test, alphas, n_splits, seed)
    summary, samples = {}, {}
    for a in alphas:
        ecr = raw[a]["ecr"]
        k = math.ceil((n_cal + 1) * (1 - a))
        ok = ecr[np.isfinite(ecr)]
        s = {"alpha": a, "target": 1 - a, "n_cal": n_cal, "n_test": n_test, "k": k,
             "infeasible_share": float(np.mean(~np.isfinite(ecr))),
             "ecr_mean": float(ok.mean()) if ok.size else float("nan"),
             "ecr_std": float(ok.std()) if ok.size else float("nan"),
             "ecr_q05": float(np.quantile(ok, 0.05)) if ok.size else float("nan"),
             "ecr_q95": float(np.quantile(ok, 0.95)) if ok.size else float("nan"),
             "below_target_share": float(np.mean(ok < 1 - a)) if ok.size else float("nan")}
        s.update({key: v for key, v in _summarise(raw[a]).items() if key in ("apss_mean", "apss_std", "empty_mean")})
        if k <= n_cal:
            beta = stats.beta(k, n_cal + 1 - k)
            s.update(theory_mean=float(beta.mean()), theory_upper=1 - a + 1 / (n_cal + 1),
                     theory_q01=float(beta.ppf(0.01)), theory_q99=float(beta.ppf(0.99)))
        summary[str(a)] = s
        samples[a] = ecr
    return summary, samples


FLOOR_TOL = 0.01


def ratio_cell(apss: float, ref_apss: float, ecr: float, ref_ecr: float) -> str:
    """APSS ratio; '(floor)' when CHASE cannot go down to the reference coverage (target below Hit@1: the most
    confident node is always returned), so its coverage is higher and the ratio compares unequal coverage."""
    if not (np.isfinite(apss) and ref_apss > 0):
        return "—"
    return f"{apss / ref_apss:.2f}" + (" (floor)" if ecr > ref_ecr + FLOOR_TOL else "")


CURVE_ALPHAS = [round(a, 2) for a in np.arange(0.02, 0.86, 0.01)]


def matched_coverage(cal: list[SigmaRecord], test: list[SigmaRecord], reference: dict, n_splits: int = 1000,
                     seed: int = 0) -> dict:
    """Compare at equal coverage without touching test labels.

    For every reported (ECR, APSS) of another method, calibrate CHASE at target 1 - alpha' = that ECR and
    report its re-split means (exact coverage on average, by the split-conformal guarantee) and the official
    split. "curve" is the same re-split summary on a dense alpha grid: APSS as a function of coverage.
    """
    targets = {round(1 - ecr / 100, 4) for table in reference.values() for ecr, _ in table.values()}
    grid = sorted(set(CURVE_ALPHAS) | targets)
    summ = {a: _summarise(s) for a, s in resplit_samples(cal, test, grid, n_splits, seed).items()}
    points = []
    for name, table in reference.items():
        for ref_alpha, (ecr, apss) in sorted(table.items()):
            a = round(1 - ecr / 100, 4)
            off = metrics_at(cal, test, a)
            points.append({"method": name, "ref_alpha": ref_alpha, "ref_ecr": ecr / 100, "ref_apss": apss, "alpha": a,
                           "official": {k: off.get(k) for k in ("feasible", "ecr", "apss", "empty_share")},
                           "resplit": summ[a]})
    return {"points": points, "curve": [{"alpha": a, **summ[a]} for a in CURVE_ALPHAS]}


def matched_table(points: list[dict]) -> str:
    """Markdown: each reported point of another method vs CHASE calibrated at that coverage."""
    lines = ["**Equal coverage** (CHASE calibrated on cal at target = the other method's ECR; re-split means)", "",
             "| Method | α | ECR % | APSS | CHASE target α' | CHASE ECR % | CHASE APSS | APSS ratio |",
             "|---|---|---|---|---|---|---|---|"]
    for p in points:
        r = p["resplit"]
        lines.append(f"| {p['method']} (reported) | {p['ref_alpha']} | {100 * p['ref_ecr']:.1f} | {p['ref_apss']:.2f} | "
                     f"{p['alpha']:.3f} | {100 * r['ecr_mean']:.1f} | {r['apss_mean']:.2f} | "
                     f"{ratio_cell(r['apss_mean'], p['ref_apss'], r['ecr_mean'], p['ref_ecr'])} |")
    return "\n".join(lines)


def comparison_table(rows: list[dict], reference: dict, rho: float) -> str:
    """Markdown table: every alpha as a column; ECR % / APSS per method; '—' where unreported or infeasible."""
    head = "| Method | " + " | ".join(f"α={r['alpha']}" for r in rows) + " |"
    sep = "|---|" + "---|" * len(rows)
    lines = ["**ECR (%) / APSS**", "", head, sep]
    for name, table in reference.items():
        cells = [f"{table[r['alpha']][0]:.1f} / {table[r['alpha']][1]:.2f}" if r["alpha"] in table else "—" for r in rows]
        lines.append(f"| {name} (reported) | " + " | ".join(cells) + " |")
    cells = [f"{100 * r['ecr']:.1f} / {r['apss']:.2f}" if r["feasible"] else "—" for r in rows]
    lines.append("| **CHASE** | " + " | ".join(cells) + " |")
    lines += ["", f"Target coverage 1−α: " + ", ".join(f"{1 - r['alpha']:.2f}" for r in rows),
              f"Reachability ceiling ρ (test): {rho:.4f}. '—' = not reported (α < 0.3), coverage failed (ECR < 1−α, as in CPR's table), or infeasible."]
    return "\n".join(lines)


# ------------------------------------------------------------------ pipeline

def run(data: str, model: str, out: str, H: int, lam: float | None, alphas, main_alpha: float,
        n_splits: int, dataset: str | None) -> None:
    from chase.data.splits import load_splits
    from chase.kg.subgraphs import SubgraphStore
    from chase.model.features import SubgraphFeatureStore
    from chase.train.train import load_scorer

    torch.set_num_threads(1)                      # single-thread CPU timing, as in the protocol
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(model)
    lam = lam if lam is not None else ck["train_config"]["lam"]
    scorer = load_scorer(model)
    dataset = dataset or Path(data).name
    splits = load_splits(data)

    def open_split(name):
        ref = splits[name]
        store = SubgraphStore.open(data, ref["store"])
        return store, SubgraphFeatureStore(store, data, ref["store"]), ref["indices"]

    t0 = time.time()
    with torch.no_grad():
        cal_store, cal_feats, cal_idx = open_split("cal")
        cal = collect_sigmas(scorer, cal_store, cal_feats, cal_idx, H, lam, progress=True, desc="calibration (cal)")
        te_store, te_feats, te_idx = open_split("test")
        test = collect_sigmas(scorer, te_store, te_feats, te_idx, H, lam, progress=True, desc="test, unpruned")
    rho = float(np.mean([np.isfinite(r.sigma_star) for r in test]))
    print(f"{CYAN}cal {len(cal)} | test {len(test)} | test reachability ρ = {rho:.4f} | lam = {lam}{RESET}")

    rows, sizes_by_alpha, per_q = [], {}, None
    states_unpruned = float(np.mean([r.states_expanded for r in test]))
    for a in alphas:
        row = metrics_at(cal, test, a)
        row["states_unpruned_mean"] = states_unpruned
        if row["feasible"]:
            with torch.no_grad():
                pruned = collect_sigmas(scorer, te_store, te_feats, te_idx, H, lam, tau=row["tau"], progress=True,
                                        desc=f"test, pruned α={a}")
            row.update(states_mean=float(np.mean([r.states_expanded for r in pruned])),
                       states_median=float(np.median([r.states_expanded for r in pruned])),
                       edges_mean=float(np.mean([r.edges_scored for r in pruned])),
                       ms_per_question=1000 * float(np.mean([r.seconds for r in pruned])),
                       pruned_matches_unpruned=bool(np.array_equal(set_sizes(pruned, row["tau"]),
                                                                   set_sizes(test, row["tau"]))))
            sizes_by_alpha[a] = set_sizes(test, row["tau"])
            if a == main_alpha:
                per_q = pd.DataFrame({
                    "qid": te_store.questions.iloc[te_idx]["qid"].to_numpy(),
                    "sigma_star": [r.sigma_star for r in test], "covered": covered([r.sigma_star for r in test], row["tau"]),
                    "set_size": sizes_by_alpha[a], "states_expanded": [r.states_expanded for r in pruned],
                    "edges_scored": [r.edges_scored for r in pruned], "ms": [1000 * r.seconds for r in pruned],
                    "best_hop": [r.best_hop for r in test], "topic_degree": [r.topic_degree for r in test]})
        rows.append(row)
        _print_row(row)

    validity, samples = validity_resplits(cal, test, alphas, n_splits)
    feasible = [r["alpha"] for r in rows if r["feasible"]]
    summary = {"dataset": dataset, "model": model, "H": H, "lam": lam, "n_cal": len(cal), "n_test": len(test),
               "rho_test": rho, "min_feasible_alpha": min(feasible) if feasible else None,
               "main_alpha": main_alpha, "rows": rows, "seconds": round(time.time() - t0, 1)}
    reference = reference_for(dataset)
    matched = matched_coverage(cal, test, reference, n_splits)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    (out_dir / "validity.json").write_text(json.dumps(validity, indent=2))
    (out_dir / "matched.json").write_text(json.dumps(matched, indent=2, default=float))
    np.savez(out_dir / "validity_ecr.npz", **{f"alpha_{a}": v for a, v in samples.items()})
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    (out_dir / "results_table.md").write_text(comparison_table(rows, reference, rho) + "\n\n"
                                              + matched_table(matched["points"]))
    if per_q is not None:
        per_q.to_parquet(out_dir / "per_question.parquet", index=False)

    history_path = Path(model).parent / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else None
    plots.make_all(out_dir / "figures", rows, validity, samples, sizes_by_alpha,
                   per_q if per_q is not None else pd.DataFrame(columns=["best_hop", "states_expanded", "topic_degree"]),
                   reference, rho, main_alpha, history)
    plots.apss_vs_coverage([matched["curve"]], reference, out_dir / "figures")

    print(f"\n{BOLD}validity over {n_splits} re-splits{RESET}")
    for a in alphas:
        v = validity[str(a)]
        print(f"  α={a:<4} ECR {v['ecr_mean']:.3f} ± {v['ecr_std']:.3f}  (target {1 - a:.2f}, "
              f"infeasible {v['infeasible_share']:.0%})")
    print(f"\n{comparison_table(rows, reference, rho)}\n\n{matched_table(matched['points'])}\n")
    print(f"{GREEN}saved to {out_dir}: metrics.json, results.csv, results_table.md, validity.json, "
          f"validity_ecr.npz, matched.json, per_question.parquet, figures/ ({len(list((out_dir / 'figures').glob('*.png')))} figures) | "
          f"{summary['seconds']}s{RESET}")


def _print_row(r: dict) -> None:
    if not r["feasible"]:
        print(f"  α={r['alpha']:<4} {YELLOW}infeasible (τ̂ = ∞){RESET}")
        return
    ok = GREEN if r["ecr"] >= r["target"] - 0.02 else YELLOW
    print(f"  α={r['alpha']:<4} τ̂ {r['tau']:.3f} | ECR {ok}{r['ecr']:.3f}{RESET} (target {r['target']:.2f}) | "
          f"APSS {BOLD}{r['apss']:.2f}{RESET} (median {r['size_median']:.0f}, p90 {r['size_p90']:.0f}) | "
          f"Hit@1 {r['hit1']:.3f} | states {r['states_mean']:.0f} | {r['ms_per_question']:.1f} ms/q "
          f"{DIM}pruned==unpruned: {r['pruned_matches_unpruned']}{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Official CHASE evaluation on a RoG (Freebase) store")
    ap.add_argument("--data", required=True, help="e.g. data/processed/webqsp")
    ap.add_argument("--model", required=True, help="e.g. runs/webqsp_seed0/best.pt")
    ap.add_argument("--out", required=True, help="e.g. runs/webqsp_seed0/eval")
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--lam", type=float, default=None, help="default: the value used in training")
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--main-alpha", type=float, default=0.1)
    ap.add_argument("--resplits", type=int, default=1000)
    ap.add_argument("--dataset", default=None, help="reference numbers to compare with (default: folder name)")
    args = ap.parse_args()
    run(args.data, args.model, args.out, args.H, args.lam, args.alphas, args.main_alpha, args.resplits, args.dataset)


if __name__ == "__main__":
    main()
