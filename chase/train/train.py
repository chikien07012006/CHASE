from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from chase.model.scorer import ScorerConfig, StateScorer
from chase.train.evaluate import collect_sigmas, two_fold_metrics
from chase.train.losses import (alpha_eff, bce_loss, conftr_size_loss, soft_bottleneck_dp, stage2_loss)
from chase.train.sampler import Batch, Sampler


@dataclass
class TrainConfig:
    H: int = 2
    alpha: float = 0.1               # main miscoverage level of this dataset
    stage1_epochs: int = 50          # upper bounds; early stopping usually ends a stage sooner
    stage2_epochs: int = 20
    patience: int = 8                # stop a stage after this many epochs without a better val APSS
    lr1: float = 1e-3
    lr2: float = 3e-4
    weight_decay: float = 0.01
    batch1: int = 64
    batch2: int = 128
    k_hard: int = 32
    k_easy: int = 32
    beta: float = 1.0                # stop BCE weight
    mu: float = 0.5                  # BCE weight in stage 2
    T: float = 0.1                   # smooth max / min temperature
    T_c: float = 0.1                 # soft set-size temperature
    lam: float = 1.0                 # stop-score weight in sigma
    grad_clip: float = 1.0
    seed: int = 0


def _bce(scorer: StateScorer, b: Batch, beta: float) -> torch.Tensor:
    step = scorer.step_logits(b.step.q, b.step.h, b.step.rel_feat, b.step.node_feat) if len(b.step.label) else torch.empty(0)
    stop = scorer.stop_logits(b.stop.q, b.stop.h, b.stop.node_feat) if len(b.stop.label) else torch.empty(0)
    return bce_loss(step, b.step.label, stop, b.stop.label, beta)


def _soft_sigmas(scorer: StateScorer, b: Batch, cfg: TrainConfig) -> list[torch.Tensor]:
    out = []
    for g in b.graphs:
        t = scorer.tables(g.q, g.graph, cfg.H, rel_ids=g.rel_ids)
        out.append(soft_bottleneck_dp(g.graph.node.shape[0], g.src, g.dst, t.edge_scores(g.rel_pos, g.dst),
                                      t.stop_scores(), g.topic, g.answerable, cfg.H, cfg.lam, cfg.T))
    return out


def _step(scorer, opt, loss, clip):
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(scorer.parameters(), clip)
    opt.step()


def _validate(scorer, val, cfg: TrainConfig) -> dict:
    store, features, indices = val
    with torch.no_grad():
        records = collect_sigmas(scorer, store, features, indices, cfg.H, cfg.lam, progress=True)
        return two_fold_metrics(records, cfg.alpha, cfg.seed)


GREEN, YELLOW, CYAN, BOLD, RESET = "\033[32m", "\033[33m", "\033[36m", "\033[1m", "\033[0m"
STAGE_COLOUR = {1: "green", 2: "magenta"}


def _better(m: dict, best: dict | None) -> bool:
    return best is None or (m["apss"], -np.nan_to_num(m["ecr"])) < (best["apss"], -np.nan_to_num(best["ecr"]))


def train(cfg: TrainConfig, sampler: Sampler, val, scorer: StateScorer, out_dir: str | Path) -> dict:
    """Stage 1 (BCE), then stage 2 (soft set size + mu * BCE) from the best stage-1 weights.

    val = (store, features, indices): model selection by two-fold APSS at cfg.alpha with the hard DP.
    Saves stage1_best.pt, stage2_best.pt, best.pt and history.json in out_dir.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rho = len(sampler.reachable_indices) / len(sampler.indices)
    a_eff = alpha_eff(cfg.alpha, rho)
    history = {"config": asdict(cfg), "rho_train": rho, "alpha_eff": a_eff, "epochs": []}
    print(f"train questions {len(sampler.indices)} (reachable {rho:.2%}) | alpha={cfg.alpha} -> alpha_eff={a_eff:.4f}")

    def run_stage(stage: int, epochs: int, lr: float) -> dict | None:
        opt = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        best, stale = None, 0
        batch_size = cfg.batch1 if stage == 1 else cfg.batch2
        pool = sampler.indices if stage == 2 else sampler.reachable_indices
        tqdm.write(f"{BOLD}{CYAN}== stage {stage}: up to {epochs} epochs, lr {lr}, patience {cfg.patience} =={RESET}")
        for epoch in range(epochs):
            t0, losses, sizes, taus = time.time(), [], [], []
            scorer.train()
            rng = np.random.default_rng((cfg.seed, stage, epoch))
            batches = tqdm(sampler.batches(epoch + 100 * stage, batch_size, with_graph=stage == 2),
                           total=math.ceil(len(pool) / batch_size), desc=f"stage {stage} | epoch {epoch:>2}",
                           colour=STAGE_COLOUR[stage], leave=False, dynamic_ncols=True)
            for b in batches:
                bce = _bce(scorer, b, cfg.beta)
                if stage == 1:
                    loss = bce
                else:
                    sigmas = _soft_sigmas(scorer, b, cfg)
                    perm = rng.permutation(len(sigmas)).tolist()
                    size, tau = conftr_size_loss(sigmas, [g.gold for g in b.graphs], perm[: len(perm) // 2],
                                                 perm[len(perm) // 2:], a_eff, cfg.T, cfg.T_c)
                    loss = stage2_loss(size, bce, cfg.mu)
                    sizes.append(size.item())
                    taus.append(tau.item())
                _step(scorer, opt, loss, cfg.grad_clip)
                losses.append(loss.item())
                batches.set_postfix(loss=f"{np.mean(losses):.4f}")
            m = _validate(scorer, val, cfg)
            rec = {"stage": stage, "epoch": epoch, "loss": float(np.mean(losses)), "val": m,
                   "seconds": round(time.time() - t0, 1)}
            if stage == 2:
                rec.update(size_loss=float(np.mean(sizes)), tau_batch=float(np.mean(taus)))
            history["epochs"].append(rec)
            improved = _better(m, best)
            if improved:
                best, stale = {**m, "epoch": epoch}, 0
                save_checkpoint(out / f"stage{stage}_best.pt", scorer, cfg, best)
            else:
                stale += 1
            mark = f"{GREEN}* best{RESET}" if improved else f"{YELLOW}no gain {stale}/{cfg.patience}{RESET}"
            tqdm.write(f"[stage {stage} epoch {epoch:>2}] loss {rec['loss']:.4f} | val APSS {BOLD}{m['apss']:.2f}{RESET} "
                       f"ECR {m['ecr']:.3f} tau {m['tau']:.3f} | {rec['seconds']}s  {mark}")
            (out / "history.json").write_text(json.dumps(history, indent=2))
            if stale >= cfg.patience:
                tqdm.write(f"{YELLOW}early stop: stage {stage} after epoch {epoch} (best epoch {best['epoch']}){RESET}")
                break
        return best

    best1 = run_stage(1, cfg.stage1_epochs, cfg.lr1)
    history["stage1_best"] = best1
    best2 = None
    if cfg.stage2_epochs:
        scorer.load_state_dict(torch.load(out / "stage1_best.pt")["state_dict"])
        best2 = run_stage(2, cfg.stage2_epochs, cfg.lr2)
        history["stage2_best"] = best2
    winner = 2 if best2 is not None and _better(best2, best1) else 1
    ck = torch.load(out / f"stage{winner}_best.pt")
    torch.save(ck, out / "best.pt")
    history["winner"] = winner
    if best2 is not None and np.isfinite(best1["apss"]) and best1["apss"] > 0:
        history["stage2_apss_reduction"] = 1 - best2["apss"] / best1["apss"]      # gate G5 asks for >= 15%
    (out / "history.json").write_text(json.dumps(history, indent=2))
    tqdm.write(f"{BOLD}{GREEN}best: stage {winner} (epoch {ck['val']['epoch']}) | val APSS {ck['val']['apss']:.2f} "
               f"ECR {ck['val']['ecr']:.3f} -> {out / 'best.pt'}{RESET}")
    if "stage2_apss_reduction" in history:
        tqdm.write(f"stage 2 vs stage 1 APSS reduction: {history['stage2_apss_reduction']:.1%} (gate G5 needs >= 15%)")
    return history


def save_checkpoint(path: Path, scorer: StateScorer, cfg: TrainConfig, val: dict) -> None:
    torch.save({"state_dict": scorer.state_dict(), "scorer_config": asdict(scorer.cfg),
                "train_config": asdict(cfg), "val": val}, path)


def load_scorer(path: str | Path) -> StateScorer:
    ck = torch.load(path)
    scorer = StateScorer(ScorerConfig(**ck["scorer_config"]))
    scorer.load_state_dict(ck["state_dict"])
    return scorer.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the CHASE scorer on a RoG (Freebase) store")
    ap.add_argument("--data", required=True, help="e.g. data/processed/webqsp")
    ap.add_argument("--out", required=True, help="e.g. runs/webqsp_seed0")
    for f, v in asdict(TrainConfig()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()
    cfg = TrainConfig(**{f: getattr(args, f) for f in asdict(TrainConfig())})

    from chase.data.splits import load_splits
    from chase.kg.subgraphs import SubgraphStore
    from chase.model.features import SubgraphFeatureStore

    splits = load_splits(args.data)
    tr, va = splits["train"], splits["val"]
    store = SubgraphStore.open(args.data, tr["store"])
    feats = SubgraphFeatureStore(store, args.data, tr["store"])
    gold = pd.read_parquet(Path(args.data) / "gold_paths" / "train.parquet")
    sampler = Sampler(store, feats, gold, tr["indices"], cfg.H, cfg.k_hard, cfg.k_easy, cfg.seed)
    val_store = SubgraphStore.open(args.data, va["store"])
    val = (val_store, SubgraphFeatureStore(val_store, args.data, va["store"]), va["indices"])

    node_dim = feats.graph(tr["indices"][0]).node.shape[1]
    scorer = StateScorer(ScorerConfig(q_dim=feats.questions.shape[1], rel_dim=feats.relation.shape[1], node_dim=node_dim))
    train(cfg, sampler, val, scorer, args.out)


if __name__ == "__main__":
    main()
