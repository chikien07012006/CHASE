"""Multi-seed comparison with published methods, at the same alpha and at equal coverage.

Reads metrics.json, validity.json and matched.json from several eval folders (one per seed) written by
chase.eval.freebase, and writes compare.md plus two figures. Mean ± std over seeds (ddof = 1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chase.eval import plots
from chase.eval.freebase import ratio_cell
from chase.eval.reference import reference_for


def _ms(x, fmt="{:.2f}") -> str:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        return "—"
    return fmt.format(x.mean()) + (" ± " + fmt.format(x.std(ddof=1)) if len(x) > 1 else "")


def same_alpha_table(metrics: list[dict], validities: list[dict], reference: dict) -> str:
    alphas = [r["alpha"] for r in metrics[0]["rows"] if all(next(x for x in m["rows"] if x["alpha"] == r["alpha"])["feasible"]
                                                           for m in metrics)]
    lines = ["**Same α**: ECR % / APSS. CHASE: official split (mean ± std over seeds) and mean over 1000 re-splits.", "",
             "| α | target % | " + " | ".join(f"{n} (reported)" for n in reference) +
             " | CHASE ECR % (official) | CHASE APSS (official) | CHASE APSS (re-split) | CHASE empty % (official) |",
             "|---|---|" + "---|" * (len(reference) + 4)]
    for a in alphas:
        rows = [next(x for x in m["rows"] if x["alpha"] == a) for m in metrics]
        ref = [f"{t[a][0]:.1f} / {t[a][1]:.2f}" if a in t else "—" for t in reference.values()]
        lines.append(f"| {a} | {100 * (1 - a):.0f} | " + " | ".join(ref) + " | "
                     f"{_ms([100 * r['ecr'] for r in rows], '{:.1f}')} | {_ms([r['apss'] for r in rows])} | "
                     f"{_ms([v[str(a)]['apss_mean'] for v in validities])} | "
                     f"{_ms([100 * r['empty_share'] for r in rows], '{:.1f}')} |")
    return "\n".join(lines)


def equal_coverage_table(matched: list[dict]) -> str:
    lines = ["**Equal coverage**: CHASE calibrated on cal at target 1 − α' = the other method's reported ECR; "
             "CHASE numbers are means over 1000 re-splits (then mean ± std over seeds). '(floor)': the target is "
             "below CHASE's Hit@1 floor, so CHASE covers more than the other method.", "",
             "| Method | α | ECR % | APSS | CHASE α' | CHASE ECR % | CHASE APSS | CHASE empty % | APSS ratio |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, p in enumerate(matched[0]["points"]):
        r = [m["points"][i]["resplit"] for m in matched]
        apss = [x["apss_mean"] for x in r]
        lines.append(f"| {p['method']} (reported) | {p['ref_alpha']} | {100 * p['ref_ecr']:.1f} | {p['ref_apss']:.2f} | "
                     f"{p['alpha']:.3f} | {_ms([100 * x['ecr_mean'] for x in r], '{:.1f}')} | {_ms(apss)} | "
                     f"{_ms([100 * x['empty_mean'] for x in r], '{:.1f}')} | "
                     f"{ratio_cell(np.mean(apss), p['ref_apss'], np.mean([x['ecr_mean'] for x in r]), p['ref_ecr'])} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed comparison at the same alpha and at equal coverage")
    ap.add_argument("--evals", nargs="+", required=True, help="eval folders, e.g. runs/webqsp_seed*/eval")
    ap.add_argument("--out", required=True, help="e.g. runs/webqsp_compare")
    args = ap.parse_args()
    dirs = [Path(d) for d in args.evals]
    load = lambda name: [json.loads((d / name).read_text()) for d in dirs]
    metrics, validities, matched = load("metrics.json"), load("validity.json"), load("matched.json")
    reference = reference_for(metrics[0]["dataset"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# {metrics[0]['dataset']}: {len(dirs)} seeds", "", "Runs: " + ", ".join(f"`{d}`" for d in dirs), "",
          same_alpha_table(metrics, validities, reference), "", equal_coverage_table(matched), "",
          "Figures: `fig_same_alpha.png`, `fig_equal_coverage.png`."]
    (out / "compare.md").write_text("\n".join(md) + "\n")
    plots._style()
    plots.apss_vs_alpha(validities, reference, out)
    plots.apss_vs_coverage([m["curve"] for m in matched], reference, out, name="fig_equal_coverage")
    print("\n".join(md))
    print(f"\nsaved to {out}: compare.md, fig_same_alpha.png/pdf, fig_equal_coverage.png/pdf")


if __name__ == "__main__":
    main()
