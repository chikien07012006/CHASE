"""Figures for the evaluation report. Static (PNG for slides, PDF for the paper).

One fixed colour per method (validated categorical palette, slots 1-3), thin marks,
hairline solid grid, neutral reference lines with direct labels, no dual axes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METHOD_COLOUR = {"CHASE": "#2a78d6", "CPR": "#eb6834", "UaG": "#1baf7a", "CLM": "#9b59b6", "LoFreeCP": "#c9a227"}
STAGE_COLOUR = {1: "#2a78d6", 2: "#eb6834"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 6,
    })


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def _ref_line(ax, y=None, x=None, label="", where="right"):
    if y is not None:
        ax.axhline(y, color=INK_2, lw=0.9)
        xmin, xmax = ax.get_xlim()
        ax.text(xmax if where == "right" else xmin, y, f" {label}", color=INK_2, fontsize=8,
                va="bottom", ha="right" if where == "right" else "left")
    else:
        ax.axvline(x, color=INK_2, lw=0.9)
        ax.text(x, ax.get_ylim()[1], f" {label}", color=INK_2, fontsize=8, va="top", ha="left")


# ------------------------------------------------------------------ figures

def validity_histograms(validity: dict, ecr_samples: dict[float, np.ndarray], out: Path, alphas=(0.1, 0.3, 0.5)) -> None:
    """Fig. 1 (RQ1): ECR over random calibration/test re-splits vs the split-conformal theory."""
    alphas = [a for a in alphas if a in ecr_samples and np.isfinite(ecr_samples[a]).any()]
    if not alphas:
        return
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.6 * len(alphas), 3.0), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, a in zip(axes[0], alphas):
        v = validity[str(a)]
        e = ecr_samples[a][np.isfinite(ecr_samples[a])]
        bins = np.linspace(min(e.min(), v.get("theory_q01", e.min())), max(e.max(), v.get("theory_q99", e.max())), 30)
        ax.hist(e, bins=bins, density=True, color=METHOD_COLOUR["CHASE"], rwidth=0.88, alpha=0.85,
                label=f"empirical ({len(e)} splits)")
        if v["k"] <= v["n_cal"]:
            theory = rng.binomial(v["n_test"], rng.beta(v["k"], v["n_cal"] + 1 - v["k"], 200_000)) / v["n_test"]
            ax.hist(theory, bins=bins, density=True, histtype="step", color=INK, lw=1.2, label="theory")
        _ref_line(ax, x=1 - a, label=f"1−α = {1 - a:.2f}")
        ax.set_title(f"α = {a}")
        ax.set_xlabel("ECR on the test part")
        ax.set_yticks([])
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Coverage over 1000 random calibration/test splits", fontsize=11, color=INK)
    _save(fig, out, "fig1_validity_hist")


def coverage_vs_target(rows: list[dict], validity: dict, reference: dict, rho: float, out: Path) -> None:
    """Fig. 2 (RQ1/RQ3): achieved coverage against the requested level, with the reachability ceiling."""
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], color=INK_2, lw=0.9)
    ax.text(0.17, 0.12, "ECR = 1−α (exact)", color=INK_2, fontsize=8, rotation=38)
    for name, table in reference.items():
        a = sorted(table)
        ax.plot([1 - x for x in a], [table[x][0] / 100 for x in a], "o-", color=METHOD_COLOUR.get(name, INK_2),
                label=f"{name} (reported)")
    feas = [r for r in rows if r["feasible"]]
    t = np.array([1 - r["alpha"] for r in feas])
    mean = np.array([validity[str(r["alpha"])]["ecr_mean"] for r in feas])
    std = np.array([validity[str(r["alpha"])]["ecr_std"] for r in feas])
    ax.errorbar(t, mean, yerr=std, fmt="o-", color=METHOD_COLOUR["CHASE"], capsize=3,
                label="CHASE (mean ± sd, 1000 splits)")
    ax.plot(t, [r["ecr"] for r in feas], "s", mfc="none", color=METHOD_COLOUR["CHASE"], label="CHASE (official split)")
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.1, 1.0)
    _ref_line(ax, y=rho, label=f"ceiling ρ = {rho:.3f}", where="left")
    ax.set_xlabel("requested coverage 1−α")
    ax.set_ylabel("empirical coverage (ECR)")
    ax.set_title("Requested vs achieved coverage")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out, "fig2_coverage_vs_target")


def apss_and_efficiency(rows: list[dict], reference: dict, out: Path) -> None:
    """Fig. 3 (RQ2): answer-set size and coverage efficiency against alpha (two panels, one axis each)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    feas = [r for r in rows if r["feasible"]]
    al = [r["alpha"] for r in feas]
    for name, table in reference.items():
        a = sorted(table)
        a1.plot(a, [table[x][1] for x in a], "o-", color=METHOD_COLOUR.get(name, INK_2), label=f"{name} (reported)")
        a2.plot(a, [table[x][0] / table[x][1] for x in a], "o-", color=METHOD_COLOUR.get(name, INK_2),
                label=f"{name} (reported)")
    a1.plot(al, [r["apss"] for r in feas], "o-", color=METHOD_COLOUR["CHASE"], label="CHASE (mean)")
    a1.fill_between(al, [r["size_median"] for r in feas], [r["size_p90"] for r in feas],
                    color=METHOD_COLOUR["CHASE"], alpha=0.15, lw=0, label="CHASE median–p90")
    a2.plot(al, [100 * r["ecr"] / r["apss"] if r["apss"] else np.nan for r in feas], "o-",
            color=METHOD_COLOUR["CHASE"], label="CHASE")
    a1.set_yscale("log")
    a1.set_xlabel("α")
    a1.set_ylabel("answer-set size (log)")
    a1.set_title("APSS (lower is better)")
    a2.set_xlabel("α")
    a2.set_ylabel("ECR / APSS  (% per answer)")
    a2.set_title("Coverage efficiency (higher is better)")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    _save(fig, out, "fig3_apss_efficiency")


def set_size_distribution(sizes: dict[float, np.ndarray], out: Path, alphas=(0.1, 0.5)) -> None:
    """Fig. 4: how answer-set sizes are spread across questions (is APSS driven by a long tail?)."""
    alphas = [a for a in alphas if a in sizes]
    if not alphas:
        return
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.4 * len(alphas), 3.2), squeeze=False)
    for ax, a in zip(axes[0], alphas):
        s = sizes[a]
        bins = np.unique(np.round(np.logspace(0, np.log10(max(s.max(), 2) + 1), 30)))
        ax.hist(np.maximum(s, 1), bins=bins, color=METHOD_COLOUR["CHASE"], rwidth=0.88, alpha=0.85)
        ax.set_xscale("log")
        for q, lab in [(np.median(s), "median"), (np.mean(s), "mean")]:
            _ref_line(ax, x=max(q, 1), label=f"{lab} {q:.1f}")
        ax.set_title(f"α = {a}: {np.mean(s == 0):.0%} empty, {np.mean(s <= 5):.0%} ≤ 5 answers")
        ax.set_xlabel("answer-set size per question (log)")
        ax.set_ylabel("questions")
    _save(fig, out, "fig4_set_size_distribution")


def cost(rows: list[dict], per_q, main_alpha: float, out: Path) -> None:
    """Fig. 5 (RQ4): search cost against alpha, against question hop count and topic-entity degree."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))
    feas = [r for r in rows if r["feasible"]]
    axes[0].plot([r["alpha"] for r in feas], [r["states_mean"] for r in feas], "o-", color=METHOD_COLOUR["CHASE"])
    if rows and "states_unpruned_mean" in rows[0]:
        _ref_line(axes[0], y=rows[0]["states_unpruned_mean"], label="no pruning", where="left")
    axes[0].set_xlabel("α")
    axes[0].set_ylabel("states expanded per question")
    axes[0].set_title("Pruning cost vs α")
    hops = sorted(h for h in per_q["best_hop"].unique() if h > 0)
    data = [per_q.loc[per_q["best_hop"] == h, "states_expanded"].to_numpy() for h in hops]
    if data:
        bp = axes[1].boxplot(data, tick_labels=[str(h) for h in hops], showfliers=False, patch_artist=True, widths=0.5)
        for box in bp["boxes"]:
            box.set(facecolor=METHOD_COLOUR["CHASE"], alpha=0.35, edgecolor=METHOD_COLOUR["CHASE"])
        for med in bp["medians"]:
            med.set(color=INK)
    axes[1].set_xlabel("hops to the best gold answer")
    axes[1].set_ylabel("states expanded")
    axes[1].set_title(f"Cost vs question difficulty (α = {main_alpha})")
    deg = per_q["topic_degree"].clip(lower=1)
    axes[2].scatter(deg, per_q["states_expanded"].clip(lower=1), s=8, color=METHOD_COLOUR["CHASE"], alpha=0.35, lw=0)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("topic-entity degree (log)")
    axes[2].set_ylabel("states expanded (log)")
    axes[2].set_title(f"Cost vs topic-entity degree (α = {main_alpha})")
    _save(fig, out, "fig5_cost")


def training_curves(history: dict, alpha: float, out: Path) -> None:
    """Fig. 6: validation APSS and ECR per epoch for both training stages (two panels)."""
    eps = history.get("epochs", [])
    if not eps:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    x = np.arange(len(eps))
    for stage in (1, 2):
        idx = [i for i, e in enumerate(eps) if e["stage"] == stage]
        if not idx:
            continue
        lab = "stage 1 (BCE)" if stage == 1 else "stage 2 (+ conformal training)"
        a1.plot(x[idx], [eps[i]["val"]["apss"] for i in idx], "o-", ms=3.5, color=STAGE_COLOUR[stage], label=lab)
        a2.plot(x[idx], [eps[i]["val"]["ecr"] for i in idx], "o-", ms=3.5, color=STAGE_COLOUR[stage], label=lab)
    a1.set_xlabel("epoch (both stages)")
    a1.set_ylabel(f"val APSS at α = {alpha}")
    a1.set_title("Answer-set size during training")
    a2.set_ylim(max(0, 1 - alpha - 0.15), min(1, 1 - alpha + 0.15))
    _ref_line(a2, y=1 - alpha, label=f"1−α = {1 - alpha:.2f}", where="left")
    a2.set_xlabel("epoch (both stages)")
    a2.set_ylabel("val ECR")
    a2.set_title("Coverage stays at the target")
    a1.legend(fontsize=8)
    _save(fig, out, "fig6_training_curves")


def make_all(out: Path, rows, validity, ecr_samples, sizes, per_q, reference, rho, main_alpha, history=None) -> list[str]:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    validity_histograms(validity, ecr_samples, out)
    coverage_vs_target(rows, validity, reference, rho, out)
    apss_and_efficiency(rows, reference, out)
    set_size_distribution(sizes, out)
    cost(rows, per_q, main_alpha, out)
    if history:
        training_curves(history, main_alpha, out)
    return sorted(p.name for p in out.glob("*.png"))
