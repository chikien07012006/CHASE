"""Evaluate the same-scorer baselines (and CHASE) for one trained model.

One unpruned DP per question gives CHASE's sigma, the top-K ranks and the per-hop table; the TreeG beam is run
separately for each beam width. Every method is calibrated on the cal split and measured on the test split
(official split) and over random cal/test re-splits. "Equal coverage" calibrates CHASE at each baseline's
re-split mean ECR, as chase.eval.freebase.matched_coverage does for published numbers.

Writes baselines.json, baselines.md, per_question.npz (test set sizes and coverage, for chase.baselines.bootstrap)
and fig_baselines_equal_coverage.png/pdf to --out.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from chase.baselines import beam, per_hop, topk
from chase.conformal.split import conformal_quantile, covered
from chase.eval.freebase import ALPHAS, _summarise, matched_coverage, metrics_at, ratio_cell, resplit_samples
from chase.search.dp import INF, bottleneck_dp
from chase.train.evaluate import LexKey, SigmaRecord, record_from_dp, set_sizes, to_gap

GRID = [round(a, 2) for a in np.arange(0.05, 0.86, 0.05)]       # re-split grid; contains ALPHAS
CYAN, GREEN, RESET = "\033[36m", "\033[32m", "\033[0m"


def collect(scorer, store, features, indices, H: int, lam: float, beams: list[int], desc: str,
            tie_break: bool = True) -> dict:
    """Per question: CHASE / top-K / beam records, per-hop components and table."""
    from tqdm import tqdm
    out = {"chase": [], "topk": [], "comp": [], "table": [], **{f"beam{b}": [] for b in beams}}
    for i in tqdm(indices, desc=desc, colour="cyan", leave=False, dynamic_ncols=True):
        i = int(i)
        row = store.questions.iloc[i]
        kg = store.get(i)
        topic, gold = np.asarray(row["topic_nodes"], dtype=np.int64), np.asarray(row["answer_nodes"], dtype=np.int64)
        mask = kg.node_type == 0
        t0 = time.perf_counter()
        bound = scorer.bind(features.question(i), scorer.encode_graph(features.graph(i, kg)), H)
        t_bind = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = bottleneck_dp(kg, topic, bound, H, lam=lam, answer_mask=mask, keep_paths=True)
        t_dp = time.perf_counter() - t0
        out["chase"].append(to_gap(record_from_dp(res, gold, tie_break=tie_break, seconds=t_bind + t_dp)))
        out["topk"].append(topk.topk_record(res, gold, seconds=t_bind + t_dp, tie_break=tie_break))
        out["comp"].append(per_hop.walk_components(res, bound, gold, H, lam))
        out["table"].append(per_hop.per_hop_table(kg, topic, bound, res, gold, H, lam, mask))
        for b in beams:
            t0 = time.perf_counter()
            score, states, edges = beam.beam_scores(kg, topic, bound, H, b, b, lam, mask)
            out[f"beam{b}"].append(beam.beam_record(score, gold, states, edges, t_bind + time.perf_counter() - t0))
    out["comp"] = np.stack(out["comp"])
    return out


def _cost(recs: list[SigmaRecord]) -> dict:
    return {"ms_per_question": 1000 * float(np.mean([r.seconds for r in recs])),
            "states_mean": float(np.mean([r.states_expanded for r in recs])),
            "edges_mean": float(np.mean([r.edges_scored for r in recs])),
            "reach": float(np.mean([np.isfinite(r.sigma_star) for r in recs]))}


def evaluate(cal: dict, test: dict, beams: list[int], n_splits: int, n_splits_hop: int, alphas=ALPHAS) -> dict:
    """{method: {"official": [rows per alpha], "resplit": {alpha: summary}, "cost": {...}}}"""
    methods = {}
    grid = sorted(set(GRID) | set(alphas))
    scalar = {"CHASE": "chase", "Top-K": "topk", **{f"CPR-style B={b}": f"beam{b}" for b in beams}}
    for name, key in scalar.items():
        rs = resplit_samples(cal[key], test[key], grid, n_splits)
        methods[name] = {"official": [metrics_at(cal[key], test[key], a) for a in alphas],
                         "resplit": {str(a): _summarise(rs[a]) for a in grid}, "cost": _cost(test[key])}
    for b in beams:
        key = f"beam{b}"
        rs = beam.cpr_resplit_samples(cal[key], test[key], grid, n_splits)
        methods[f"CPR-style B={b}, CPR rule"] = {
            "official": [beam.cpr_metrics_at(cal[key], test[key], a) for a in alphas],
            "resplit": {str(a): _summarise(rs[a]) for a in grid}, "cost": _cost(test[key])}
    for alloc, name in [("per_hop", "UaG-style per-hop"), ("bonferroni", "UaG-style Bonferroni")]:
        rs = per_hop.resplit_samples(cal["comp"], test["comp"], cal["table"], test["table"], grid, alloc, n_splits_hop)
        methods[name] = {"official": [per_hop.metrics_at(cal["comp"], test["table"], a, alloc) for a in alphas],
                         "resplit": {str(a): _summarise(rs[a]) for a in grid}, "cost": {}}
    return methods


def per_question(cal: dict, test: dict, beams: list[int], alphas=ALPHAS) -> dict:
    """Official split, every baseline: set size and covered flag of each test question at each alpha
    (calibrated on cal as in evaluate). Infeasible levels (tau_hat = INF) get size NaN."""
    out = {}
    scalar = {"Top-K": "topk", **{f"CPR-style B={b}": f"beam{b}" for b in beams}}
    for name, key in scalar.items():
        rows = []
        for a in alphas:
            tau = conformal_quantile([r.sigma_star for r in cal[key]], a)
            ok = np.isfinite(tau)
            rows.append((covered([r.sigma_star for r in test[key]], tau) if ok else np.zeros(len(test[key]), bool),
                         set_sizes(test[key], tau) if ok else np.full(len(test[key]), np.nan)))
        out[name] = rows
    for b in beams:
        key = f"beam{b}"
        stars = np.array([r.sigma_star for r in test[key]])
        rows = []
        for a in alphas:
            cov, size = beam._cpr_eval(stars, [r.sorted_sigma for r in test[key]],
                                       np.array([r.top_is_gold for r in test[key]]),
                                       np.array([beam.cpr_quantile([r.sigma_star for r in cal[key]], a)]))
            rows.append((cov[0], size[0].astype(float)))
        out[f"CPR-style B={b}, CPR rule"] = rows
    for alloc, name in [("per_hop", "UaG-style per-hop"), ("bonferroni", "UaG-style Bonferroni")]:
        rows = []
        for a in alphas:
            tau = per_hop.thresholds(cal["comp"], a, alloc)
            if not np.isfinite(tau).all():
                rows.append((np.zeros(len(test["table"]), bool), np.full(len(test["table"]), np.nan)))
                continue
            res = [t.evaluate(tau[None]) for t in test["table"]]
            rows.append((np.array([c[0] for c, _ in res]), np.array([z[0] for _, z in res], dtype=float)))
        out[name] = rows
    return out


def save_per_question(path: Path, pq: dict, chase: list[SigmaRecord], groups: np.ndarray, alphas) -> None:
    """per_question.npz: baselines [M, A, n] and CHASE's keyed test records (ragged keys as concat + offsets)."""
    names = list(pq)
    keys = [r.sorted_sigma for r in chase]
    np.savez_compressed(
        path, methods=np.array(names), alphas=np.array(alphas, dtype=float), groups=np.asarray(groups),
        cov=np.array([[c for c, _ in pq[n]] for n in names], dtype=bool),
        size=np.array([[z for _, z in pq[n]] for n in names], dtype=np.float32),
        chase_star=np.array([r.sigma_star for r in chase]), chase_top=np.array([r.top_sigma for r in chase]),
        chase_keys=np.concatenate(keys) if keys else np.zeros(0),
        chase_offsets=np.r_[0, np.cumsum([len(k) for k in keys])])


def as_reference(methods: dict, alphas=ALPHAS) -> dict:
    """Baselines' re-split means in the {method: {alpha: (ECR %, APSS)}} form used for published numbers."""
    ref = {}
    for name, m in methods.items():
        if name == "CHASE":
            continue
        pts = {a: (100 * m["resplit"][str(a)]["ecr_mean"], m["resplit"][str(a)]["apss_mean"]) for a in alphas}
        ref[name] = {a: p for a, p in pts.items() if np.isfinite(p[1]) and p[1] > 0 and 0 < p[0] < 100}
    return ref


def _f(x, fmt="{:.2f}") -> str:
    return fmt.format(x) if x is not None and np.isfinite(x) else "—"


def tables(methods: dict, matched: list[dict], alphas=ALPHAS) -> str:
    names = list(methods)
    lines = ["**Same α, official split**: ECR % / APSS / empty %. '—' = abstain (τ̂ = ∞).", "",
             "| α | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for i, a in enumerate(alphas):
        cells = []
        for n in names:
            r = methods[n]["official"][i]
            cells.append(f"{100 * r['ecr']:.1f} / {r['apss']:.2f} / {100 * r['empty_share']:.0f}" if r["feasible"] else "—")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines += ["", "**Same α, re-split means**: ECR % / APSS (abstain share if > 0).", "",
              "| α | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for a in alphas:
        cells = []
        for n in names:
            r = methods[n]["resplit"][str(a)]
            c = f"{_f(100 * r['ecr_mean'], '{:.1f}')} / {_f(r['apss_mean'])}"
            cells.append(c + (f" ({100 * r['infeasible_share']:.0f}% abstain)" if r["infeasible_share"] > 0 else ""))
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines += ["", "**Equal coverage**: CHASE calibrated on cal at each baseline's re-split mean ECR (re-split means). "
              "ratio = CHASE APSS / baseline APSS (< 1: CHASE smaller); '(floor)': the baseline's coverage is below "
              "CHASE's Hit@1 floor, so CHASE covers more.", "",
              "| Baseline | α | ECR % | APSS | CHASE ECR % | CHASE APSS | ratio |", "|---|---|---|---|---|---|---|"]
    for p in matched:
        r = p["resplit"]
        lines.append(f"| {p['method']} | {p['ref_alpha']} | {100 * p['ref_ecr']:.1f} | {p['ref_apss']:.2f} | "
                     f"{_f(100 * r['ecr_mean'], '{:.1f}')} | {_f(r['apss_mean'])} | "
                     f"{ratio_cell(r['apss_mean'], p['ref_apss'], r['ecr_mean'], p['ref_ecr'])} |")
    lines += ["", "**Cost** (test, per question; CHASE here is the unpruned DP, its pruned cost is in chase.eval.freebase).", "",
              "| Method | ms | states | edges scored | gold reached |", "|---|---|---|---|---|"]
    for n, m in methods.items():
        c = m["cost"]
        if c:
            lines.append(f"| {n} | {c['ms_per_question']:.1f} | {c['states_mean']:.0f} | {c['edges_mean']:.0f} | {100 * c['reach']:.1f}% |")
    return "\n".join(lines)


def run(data: str, model: str, out: str, H: int, beams: list[int], n_splits: int, n_splits_hop: int,
        alphas=ALPHAS, tie_break: bool = True) -> None:
    from chase.data.splits import load_splits, seed_groups
    from chase.baselines.report import equal_coverage_figure
    from chase.kg.subgraphs import SubgraphStore
    from chase.model.features import SubgraphFeatureStore
    from chase.train.train import load_scorer

    torch.set_num_threads(1)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lam = torch.load(model)["train_config"]["lam"]
    scorer = load_scorer(model)
    splits = load_splits(data)
    t0 = time.time()
    recs = {}
    with torch.no_grad():
        for split in ("cal", "test"):
            ref = splits[split]
            store = SubgraphStore.open(data, ref["store"])
            recs[split] = collect(scorer, store, SubgraphFeatureStore(store, data, ref["store"]), ref["indices"],
                                  H, lam, beams, desc=split, tie_break=tie_break)
            if split == "test":           # bootstrap clusters: the groups used to split the data
                groups = seed_groups(store.questions["qid"].iloc[ref["indices"]].reset_index(drop=True))
    print(f"{CYAN}cal {len(recs['cal']['chase'])} | test {len(recs['test']['chase'])} | {time.time() - t0:.0f}s; "
          f"calibrating and re-splitting{RESET}")
    lex = LexKey(recs["cal"]["chase"] + recs["test"]["chase"])     # (gap, tie) pairs -> exact scalar keys
    for split in ("cal", "test"):
        recs[split]["chase"] = lex.apply(recs[split]["chase"])
    methods = evaluate(recs["cal"], recs["test"], beams, n_splits, n_splits_hop, alphas)
    matched = matched_coverage(recs["cal"]["chase"], recs["test"]["chase"], as_reference(methods, alphas), n_splits)
    result = {"dataset": Path(data).name, "model": model, "H": H, "lam": lam, "beams": beams, "alphas": alphas,
              "tie_break": "stop" if tie_break else "none",
              "n_splits": n_splits,
              "n_splits_per_hop": n_splits_hop, "methods": methods, "matched": matched["points"],
              "chase_curve": matched["curve"]}
    (out_dir / "baselines.json").write_text(json.dumps(result, indent=2, default=float))
    save_per_question(out_dir / "per_question.npz", per_question(recs["cal"], recs["test"], beams, alphas),
                      recs["test"]["chase"], groups, alphas)
    md = tables(methods, matched["points"], alphas)
    (out_dir / "baselines.md").write_text(f"# Same-scorer baselines: {result['dataset']}\n\nModel: `{model}`\n\n{md}\n")
    equal_coverage_figure([result], out_dir)
    print(md)
    print(f"\n{GREEN}saved to {out_dir}: baselines.json, baselines.md, per_question.npz, fig_baselines_equal_coverage | "
          f"{time.time() - t0:.0f}s{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Same-scorer baselines: conformal top-K, CPR-style beam, UaG-style per-hop")
    ap.add_argument("--data", required=True, help="e.g. data/processed/webqsp")
    ap.add_argument("--model", required=True, help="e.g. runs/webqsp_seed0/best.pt")
    ap.add_argument("--out", required=True, help="e.g. runs/webqsp_seed0/baselines")
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--beams", type=int, nargs="+", default=[32], help="TreeG branch = active size (CPR default 32)")
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS, help="levels of the tables (re-split grid adds them)")
    ap.add_argument("--resplits", type=int, default=1000)
    ap.add_argument("--resplits-per-hop", type=int, default=200, help="per-hop re-splits need a table pass each")
    ap.add_argument("--tie-break", choices=["stop", "none"], default="stop",
                    help="CHASE and top-K order equal sigma by the stop score (default) or keep ties whole (ablation)")
    args = ap.parse_args()
    run(args.data, args.model, args.out, args.H, args.beams, args.resplits, args.resplits_per_hop,
        [round(a, 4) for a in args.alphas], args.tie_break == "stop")


if __name__ == "__main__":
    main()
