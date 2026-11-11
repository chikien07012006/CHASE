from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chase.eval.freebase import FLOOR_TOL

LEVEL = 0.05


def cluster_weights(groups: np.ndarray, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """[R, n] weights: how many times each question is drawn when R x G clusters are drawn with replacement."""
    _, g = np.unique(groups, return_inverse=True)
    G = g.max() + 1
    draws = rng.integers(0, G, size=(n_resamples, G))
    counts = np.stack([np.bincount(d, minlength=G) for d in draws])
    return counts[:, g].astype(np.float64)


class ChaseCurve:
    """CHASE on the test questions of one seed: coverage and set size as functions of the threshold."""

    def __init__(self, star: np.ndarray, keys: np.ndarray, offsets: np.ndarray, top: np.ndarray):
        fin = np.isfinite(star)
        self.order = np.flatnonzero(fin)[np.argsort(star[fin], kind="stable")]      # covered questions by g*
        self.sorted_star = star[self.order]
        self.thresholds = np.unique(self.sorted_star)
        self.size = np.stack([np.searchsorted(keys[offsets[i]:offsets[i + 1]], self.thresholds, side="right")
                              for i in range(len(star))]).astype(np.float32)       # [n, T]
        self.floor = (fin & (star <= top)).astype(np.float64)                      # covered by the top node alone

    def matched(self, W: np.ndarray, ecr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per row of W [R, n]: (ECR, APSS) of CHASE at the smallest threshold with ECR >= ecr[r]; NaN if none."""
        tot = W.sum(1)
        cum = np.cumsum(W[:, self.order], axis=1) / tot[:, None]
        j = (cum < ecr[:, None] - 1e-12).sum(1)                                     # first position reaching ecr
        ok = j < len(self.order)
        t = self.sorted_star[np.minimum(j, len(self.order) - 1)]
        col = np.searchsorted(self.thresholds, t)
        apss = np.einsum("rn,nr->r", W, self.size[:, col]) / tot
        last = np.searchsorted(self.sorted_star, t, side="right") - 1                # the ECR counts every tie at t
        got = cum[np.arange(len(j)), last]
        return np.where(ok, got, np.nan), np.where(ok, apss, np.nan)


def ratios(W: np.ndarray, cov: np.ndarray, size: np.ndarray, curve: ChaseCurve) -> dict[str, np.ndarray]:
    """One seed, one (baseline, alpha): per row of W the baseline's ECR / APSS, CHASE's matched ECR / APSS, ratio."""
    tot = W.sum(1)
    ecr_b, apss_b = W @ cov / tot, W @ size / tot
    ecr_c, apss_c = curve.matched(W, ecr_b)
    return {"ecr_b": ecr_b, "apss_b": apss_b, "ecr_c": ecr_c, "apss_c": apss_c, "theta": apss_c / apss_b}


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values (family-wise error control)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.minimum(1.0, (len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.maximum.accumulate(adj)
    return out


def two_sided_p(theta_star: np.ndarray, null: float = 1.0) -> float:
    R = len(theta_star)
    above, below = (theta_star >= null).sum(), (theta_star <= null).sum()
    return float(min(1.0, 2 * min(above + 1, below + 1) / (R + 1)))


def bootstrap(seeds: list[dict], n_resamples: int = 10000, seed: int = 0, block: int = 500) -> list[dict]:
    """seeds: loaded per_question.npz of each seed (same test set). One result per (baseline, alpha) tested."""
    groups = seeds[0]["groups"]
    n = len(groups)
    assert all(len(d["groups"]) == n and (d["groups"] == groups).all() for d in seeds), "seeds differ in test set"
    names, alphas = list(seeds[0]["methods"]), list(seeds[0]["alphas"])
    curves = [ChaseCurve(d["chase_star"], d["chase_keys"], d["chase_offsets"], d["chase_top"]) for d in seeds]
    ones = np.ones((1, n))
    points = []
    for m, name in enumerate(names):
        for a, alpha in enumerate(alphas):
            if not all(np.isfinite(d["size"][m, a]).all() for d in seeds):
                continue                                                         # abstains in some seed
            full = [ratios(ones, d["cov"][m, a].astype(float), d["size"][m, a].astype(float), c)
                    for d, c in zip(seeds, curves)]
            floor = any(f["ecr_c"][0] > f["ecr_b"][0] + FLOOR_TOL for f in full)   # CHASE cannot go this low
            if any(not np.isfinite(f["theta"][0]) for f in full):
                continue
            points.append({"method": name, "alpha": float(alpha), "m": m, "a": a, "floor": bool(floor),
                           **{k: float(np.mean([f[k][0] for f in full])) for k in ("ecr_b", "apss_b", "ecr_c", "apss_c")},
                           "ratio": float(np.mean([f["theta"][0] for f in full])),
                           "ratio_seeds": [float(f["theta"][0]) for f in full]})
    rng = np.random.default_rng(seed)
    theta_star = np.zeros((len(points), n_resamples))
    for lo in range(0, n_resamples, block):
        W = cluster_weights(groups, min(block, n_resamples - lo), rng)
        for i, p in enumerate(points):
            th = [ratios(W, d["cov"][p["m"], p["a"]].astype(float), d["size"][p["m"], p["a"]].astype(float), c)["theta"]
                  for d, c in zip(seeds, curves)]
            theta_star[i, lo:lo + len(W)] = np.mean(th, axis=0)
    tested = [i for i, p in enumerate(points) if not p["floor"]]
    adj = holm([two_sided_p(theta_star[i][np.isfinite(theta_star[i])]) for i in tested]) if tested else []
    for i, p in enumerate(points):
        t = theta_star[i][np.isfinite(theta_star[i])]
        p.update(ci_low=float(np.percentile(t, 2.5)), ci_high=float(np.percentile(t, 97.5)), p=two_sided_p(t),
                 valid_resamples=int(t.size))
        p.pop("m"), p.pop("a")
    for i, q in zip(tested, adj):
        p = points[i]
        p["p_holm"] = float(q)
        p["verdict"] = ("CHASE smaller" if p["ratio"] < 1 else "baseline smaller") if q < LEVEL else "n.s."
    return points


def table(points: list[dict], dataset: str, n_seeds: int, n_resamples: int) -> str:
    lines = [f"# Paired bootstrap: {dataset}, {n_seeds} seeds, {n_resamples} resamples", "",
             "Official test split. CHASE is matched to each baseline's ECR on the same questions (its ECR is >= the "
             "baseline's). Ratio = CHASE APSS / baseline APSS, mean over seeds (< 1: CHASE smaller). 95% CI: percentile "
             "bootstrap over clusters of test questions (the groups of the data split), same draws for both methods "
             "and all seeds. p: two-sided, H0 ratio = 1; Holm-adjusted over the tested rows of this dataset. "
             "'(floor)': the baseline's coverage is below CHASE's Hit@1 floor; not tested.", "",
             "| Baseline | α | ECR % | APSS | CHASE ECR % | CHASE APSS | ratio [95% CI] | p (Holm) | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in points:
        test = "—" if p["floor"] else (f"{p['p_holm']:.4f}" if p["p_holm"] >= 1e-4 else "<0.0001")
        lines.append(f"| {p['method']} | {p['alpha']:g} | {100 * p['ecr_b']:.1f} | {p['apss_b']:.2f} | "
                     f"{100 * p['ecr_c']:.1f} | {p['apss_c']:.2f} | {p['ratio']:.2f} [{p['ci_low']:.2f}, {p['ci_high']:.2f}] | "
                     f"{test} | {'(floor)' if p['floor'] else p['verdict']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired cluster bootstrap: CHASE vs same-scorer baselines at equal coverage")
    ap.add_argument("--runs", nargs="+", required=True, help="folders with per_question.npz, e.g. runs/webqsp_seed*/baselines")
    ap.add_argument("--out", required=True, help="e.g. runs/webqsp_baselines")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seeds = [dict(np.load(Path(d) / "per_question.npz")) for d in args.runs]
    dataset = json.loads((Path(args.runs[0]) / "baselines.json").read_text())["dataset"]
    points = bootstrap(seeds, args.resamples, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = table(points, dataset, len(seeds), args.resamples)
    (out / "bootstrap.md").write_text(md + "\n")
    (out / "bootstrap.json").write_text(json.dumps({"dataset": dataset, "runs": args.runs, "resamples": args.resamples,
                                                    "seed": args.seed, "points": points}, indent=2))
    print(md)
    print(f"\nsaved to {out}: bootstrap.md, bootstrap.json")


if __name__ == "__main__":
    main()
