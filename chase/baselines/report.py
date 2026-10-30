"""Mean ± std over seeds of the same-scorer baselines (baselines.json from chase.baselines.run, one per seed)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chase.eval import plots
from chase.eval.compare import _ms
from chase.eval.freebase import ALPHAS, FLOOR_TOL

COLOUR = {"Top-K": "#9b59b6", "CPR-style": "#eb6834", "UaG-style per-hop": "#1baf7a", "UaG-style Bonferroni": "#c9a227"}


BEAM_SHADES = ["#eb6834", "#a8401a", "#f2a07a", "#6e2a10"]


def _colour(name: str, beams: list[int]) -> str:
    if name.startswith("CPR-style"):
        b = int(name.split("B=")[1].split(",")[0])
        return BEAM_SHADES[beams.index(b) % len(BEAM_SHADES)] if b in beams else BEAM_SHADES[0]
    return next((c for k, c in COLOUR.items() if name.startswith(k)), plots.INK_2)


def _usable(p: dict) -> bool:
    """A re-split point is plotted only if no split abstained (the mean ECR is then over all splits)."""
    return np.isfinite(p["apss_mean"]) and p["apss_mean"] > 0 and p["infeasible_share"] == 0


PLOTTED = ["Top-K", "CPR-style B=32", "UaG-style Bonferroni"]   # the tables keep every variant


def equal_coverage_figure(results: list[dict], out: Path, name: str = "fig_baselines_equal_coverage",
                          methods: list[str] = PLOTTED) -> None:
    """APSS against re-split mean coverage, same scorer for every method (log scale)."""
    plt = plots.plt
    plots._style()
    curves = [r["chase_curve"] for r in results]
    keep = [i for i in range(len(curves[0])) if all(_usable(c[i]) for c in curves)]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    plots._seed_band(ax, 100 * np.mean([[c[i]["ecr_mean"] for i in keep] for c in curves], axis=0),
                     np.array([[c[i]["apss_mean"] for i in keep] for c in curves]), "CHASE")
    beams = [int(m.split("B=")[1].split(",")[0]) for m in methods if m.startswith("CPR-style")]
    for nm in methods:
        if nm not in results[0]["methods"]:
            continue
        rs = [r["methods"][nm]["resplit"] for r in results]
        alphas = [a for a in rs[0] if all(_usable(x[a]) for x in rs)]
        x = [100 * np.mean([r[a]["ecr_mean"] for r in rs]) for a in alphas]
        y = [np.mean([r[a]["apss_mean"] for r in rs]) for a in alphas]
        ax.plot(x, y, "o--" if "CPR rule" in nm else "o-", color=_colour(nm, beams), ms=3, lw=1.2, label=nm)
    ax.set_yscale("log")
    ax.set_xlabel("coverage, ECR (%) — re-split mean, levels with no abstention")
    ax.set_ylabel("APSS (log)")
    ax.set_title("Same scorer, equal coverage (lower is better)")
    ax.legend(fontsize=7.5)
    plots._save(fig, out, name)


def same_alpha_table(results: list[dict]) -> str:
    names = list(results[0]["methods"])
    lines = ["**Same α, official split**: ECR % / APSS, mean ± std over seeds. '—' = abstain in some seed.", "",
             "| α | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for i, a in enumerate(ALPHAS):
        cells = []
        for n in names:
            rows = [r["methods"][n]["official"][i] for r in results]
            cells.append(f"{_ms([100 * x['ecr'] for x in rows], '{:.1f}')} / {_ms([x['apss'] for x in rows])}"
                         if all(x["feasible"] for x in rows) else "—")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def equal_coverage_table(results: list[dict]) -> str:
    lines = ["**Equal coverage**: each baseline's re-split mean (ECR, APSS) at α; CHASE calibrated at that ECR. "
             "Ratio = CHASE APSS / baseline APSS (< 1: CHASE smaller). Mean ± std over seeds. '(floor)': the baseline's "
             "coverage is below CHASE's Hit@1 floor, so CHASE covers more.", "",
             "| Baseline | α | ECR % | APSS | CHASE ECR % | CHASE APSS | ratio |", "|---|---|---|---|---|---|---|"]
    keys = {(p["method"], p["ref_alpha"]) for p in results[0]["matched"]}
    keys = [k for k in keys if all(any((p["method"], p["ref_alpha"]) == k for p in r["matched"]) for r in results)]
    order = list(results[0]["methods"])

    for m, a in sorted(keys, key=lambda k: (order.index(k[0]), k[1])):
        pts = [next(p for p in r["matched"] if (p["method"], p["ref_alpha"]) == (m, a)) for r in results]
        apss = [p["resplit"]["apss_mean"] for p in pts]
        ecr = [p["resplit"]["ecr_mean"] for p in pts]
        floor = np.mean(ecr) > np.mean([p["ref_ecr"] for p in pts]) + FLOOR_TOL
        lines.append(f"| {m} | {a} | {_ms([100 * p['ref_ecr'] for p in pts], '{:.1f}')} | {_ms([p['ref_apss'] for p in pts])} | "
                     f"{_ms([100 * e for e in ecr], '{:.1f}')} | {_ms(apss)} | "
                     f"{_ms([c / p['ref_apss'] for c, p in zip(apss, pts)])}{' (floor)' if floor else ''} |")
    return "\n".join(lines)


def cost_table(results: list[dict]) -> str:
    lines = ["**Cost on test** (mean over seeds): ms per question, edges scored, share of questions whose gold answer "
             "is reached at all (the coverage ceiling of the method).", "",
             "| Method | ms | edges scored | gold reached |", "|---|---|---|---|"]
    for n in results[0]["methods"]:
        cs = [r["methods"][n]["cost"] for r in results]
        if cs[0]:
            lines.append(f"| {n} | {np.mean([c['ms_per_question'] for c in cs]):.1f} | "
                         f"{np.mean([c['edges_mean'] for c in cs]):.0f} | {100 * np.mean([c['reach'] for c in cs]):.1f}% |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mean ± std over seeds of the same-scorer baselines")
    ap.add_argument("--runs", nargs="+", required=True, help="folders with baselines.json, e.g. runs/webqsp_seed*/baselines")
    ap.add_argument("--out", required=True, help="e.g. runs/webqsp_baselines")
    ap.add_argument("--plot", nargs="+", default=PLOTTED, help="methods drawn in the figure besides CHASE")
    args = ap.parse_args()
    results = [json.loads((Path(d) / "baselines.json").read_text()) for d in args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Same-scorer baselines: {results[0]['dataset']}, {len(results)} seeds", "",
          "Runs: " + ", ".join(f"`{d}`" for d in args.runs), "",
          same_alpha_table(results), "", equal_coverage_table(results), "", cost_table(results), "",
          "Figure: `fig_baselines_equal_coverage.png`."]
    (out / "baselines_compare.md").write_text("\n".join(md) + "\n")
    equal_coverage_figure(results, out, methods=args.plot)
    print("\n".join(md))
    print(f"\nsaved to {out}: baselines_compare.md, fig_baselines_equal_coverage.png/pdf")


if __name__ == "__main__":
    main()
