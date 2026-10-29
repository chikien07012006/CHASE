from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_FILE = "splits.json"


def _qids(data_dir: Path, store: str) -> pd.Series:
    return pd.read_parquet(data_dir / store / "questions.parquet", columns=["qid"])["qid"]


def seed_groups(qids: pd.Series) -> np.ndarray:
    """Group id per question: CWQ questions generated from the same WebQSP question share the
    qid prefix before '_' (e.g. WebQTrn-3513_7c41...). WebQSP qids have no '_', so every
    question is its own group."""
    codes, _ = pd.factorize(qids.str.split("_").str[0])
    return codes


def _draw_groups(groups: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Mask of the questions in a random set of whole groups holding about `fraction` of the questions
    (the smallest number of groups that reaches the target)."""
    perm = np.random.default_rng(seed).permutation(groups.max() + 1)
    k = int(np.searchsorted(np.cumsum(np.bincount(groups)[perm]), round(fraction * len(groups)))) + 1
    return np.isin(groups, perm[:k])


def make_freebase_splits(data_dir: str | Path, cal_fraction: float = 0.1, seed: int = 0,
                         cal_from: str = "train") -> dict:
    """train / cal / val / test for a RoG store. test is always the official RoG test set.

    cal_from="train" (CPR's protocol): cal is a random cal_fraction of RoG train, train is the rest,
    val is the official validation set. With singleton groups (WebQSP) this is a per-question draw.
    cal_from="validation": cal is the whole official validation set; val (early stopping only) is a
    random cal_fraction of RoG train and train is the rest. Use it when train questions cluster: on CWQ,
    a cal drawn from train shares many (topic, answers) pairs with the remaining training questions
    (36% vs 27% for test), looks easier than test, and tau_hat under-covers. The official validation
    set is built like the test set (seed groups unseen in train) and matches it (seen share 23% vs 27% for test,
    2-hop reachability 0.806 vs 0.801).

    Draws are over seed groups, so no group is split between two parts.
    Each split is {"store": <RoG split folder>, "indices": <question indices in that store>}.
    """
    d = Path(data_dir)
    every = lambda store: list(range(len(_qids(d, store))))
    test = {"store": "test", "indices": every("test")}
    meta = {"seed": seed, "cal_fraction": cal_fraction, "cal_from": cal_from, "grouped_by": "qid prefix before '_'"}
    if cal_from == "train":
        in_cal = _draw_groups(seed_groups(_qids(d, "train")), cal_fraction, seed)
        return {"train": {"store": "train", "indices": np.flatnonzero(~in_cal).tolist()},
                "cal": {"store": "train", "indices": np.flatnonzero(in_cal).tolist()},
                "val": {"store": "validation", "indices": every("validation")}, "test": test, "meta": meta}
    if cal_from == "validation":
        in_val = _draw_groups(seed_groups(_qids(d, "train")), cal_fraction, seed)
        return {"train": {"store": "train", "indices": np.flatnonzero(~in_val).tolist()},
                "cal": {"store": "validation", "indices": every("validation")},
                "val": {"store": "train", "indices": np.flatnonzero(in_val).tolist()}, "test": test, "meta": meta}
    raise ValueError(f"cal_from must be 'train' or 'validation', got {cal_from!r}")


def save_splits(data_dir: str | Path, splits: dict) -> Path:
    path = Path(data_dir) / SPLIT_FILE
    path.write_text(json.dumps(splits))
    return path


def load_splits(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / SPLIT_FILE).read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Write train/cal/val/test splits for a RoG store")
    ap.add_argument("--data", required=True, help="e.g. data/processed/webqsp")
    ap.add_argument("--cal-fraction", type=float, default=0.1, help="share of RoG train held out as cal (--cal-from train) or as early-stopping val (--cal-from validation)")
    ap.add_argument("--cal-from", choices=["train", "validation"], default="train")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    splits = make_freebase_splits(args.data, args.cal_fraction, args.seed, args.cal_from)
    path = save_splits(args.data, splits)
    sizes = {k: len(v["indices"]) for k, v in splits.items() if k != "meta"}
    print(f"{sizes} (cal from {args.cal_from}, seed={args.seed}) -> {path}")


if __name__ == "__main__":
    main()
