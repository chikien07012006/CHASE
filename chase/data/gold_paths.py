from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from chase.kg.graph import KG
from chase.kg.type_distance import multi_source_bfs

Path_ = tuple[tuple[int, ...], tuple[int, ...]]     # (nodes v0..vL, relations r1..rL)


def enumerate_paths(kg: KG, topic, answers, H: int, max_prefixes: int = 20_000, seed: int = 0) -> list[Path_]:
    """All simple paths (no repeated node) of length 1..H from a topic entity to a gold answer.

    Prefixes that cannot reach an answer in the remaining hops are dropped (BFS distance from
    the answers; the graph stores both directions). If a hop still has more than max_prefixes
    prefixes, a fixed random subset is kept.
    """
    answers = np.unique(np.asarray(answers, dtype=np.int64))
    topic = np.unique(np.asarray(topic, dtype=np.int64))
    if answers.size == 0 or topic.size == 0:
        return []
    dist = multi_source_bfs(kg, answers, H)
    is_answer = np.zeros(kg.num_nodes, dtype=bool)
    is_answer[answers] = True
    rng = np.random.default_rng(seed)

    nodes = topic[dist[topic] <= H][:, None]          # [P, h+1]
    rels = np.zeros((nodes.shape[0], 0), dtype=np.int64)
    out: list[Path_] = []
    for h in range(1, H + 1):
        if nodes.shape[0] == 0:
            break
        src, dst, rel = kg.expand(np.unique(nodes[:, -1]))
        edges = pd.DataFrame({"last": src, "dst": dst, "rel": rel})
        m = pd.DataFrame({"pid": np.arange(nodes.shape[0]), "last": nodes[:, -1]}).merge(edges, on="last")
        pid, dst, rel = m["pid"].to_numpy(), m["dst"].to_numpy(), m["rel"].to_numpy()
        keep = dist[dst] <= H - h
        for c in range(nodes.shape[1]):               # simple path: the new node is not already on it
            keep &= nodes[pid, c] != dst
        pid, dst, rel = pid[keep], dst[keep], rel[keep]
        nodes = np.concatenate([nodes[pid], dst[:, None]], axis=1)
        rels = np.concatenate([rels[pid], rel[:, None]], axis=1)
        done = is_answer[dst]
        out += [(tuple(n), tuple(r)) for n, r in zip(nodes[done].tolist(), rels[done].tolist())]
        if nodes.shape[0] > max_prefixes:
            pick = np.sort(rng.choice(nodes.shape[0], max_prefixes, replace=False))
            nodes, rels = nodes[pick], rels[pick]
    return out


def select_gold(paths: list[Path_], q_emb: np.ndarray, rel_emb: np.ndarray,
                delta: float = 0.1, max_paths: int = 64) -> list[tuple[tuple[int, ...], tuple[int, ...], float]]:
    """Drop coincidental paths: keep relation patterns whose question similarity is within delta of the best.

    A pattern is the relation sequence of a path; its score is the mean cosine between the
    question embedding and the embeddings of its relations (all unit-normalised).
    """
    if not paths:
        return []
    q = q_emb.astype(np.float32)
    sim = {}
    for _, rels in paths:
        if rels not in sim:
            sim[rels] = float(np.mean(rel_emb[list(rels)].astype(np.float32) @ q))
    best = max(sim.values())
    kept = [(n, r, sim[r]) for n, r in paths if sim[r] >= best - delta]
    kept.sort(key=lambda x: (-x[2], len(x[1]), x[1], x[0]))
    return kept[:max_paths]


def build_gold_paths(data_dir: str, split: str, H: int, delta: float, max_paths: int) -> tuple[pd.DataFrame, dict]:
    from chase.data.splits import load_splits
    from chase.kg.subgraphs import SubgraphStore
    from chase.model.text_cache import CACHE_DIRNAME

    d = Path(data_dir)
    ref = load_splits(d)[split]
    store = SubgraphStore.open(d, ref["store"])
    q_emb = np.load(d / CACHE_DIRNAME / ref["store"] / "questions.npy")
    rel_emb = np.load(d / CACHE_DIRNAME / "relations.npy")

    rows, n_cand, n_pat, n_kept, no_path = [], [], [], [], 0
    for k, i in enumerate(ref["indices"]):
        q = store.questions.iloc[i]
        paths = enumerate_paths(store.get(i), q["topic_nodes"], q["answer_nodes"], H)
        gold = select_gold(paths, q_emb[i], rel_emb, delta, max_paths)
        if not gold:
            no_path += 1
            continue
        n_cand.append(len(paths))
        n_pat.append(len({r for _, r, _ in gold}))
        n_kept.append(len(gold))
        for pid, (nodes, rels, s) in enumerate(gold):
            rows.append({"q_index": i, "qid": q["qid"], "path_id": pid, "length": len(rels),
                         "nodes": list(nodes), "rels": list(rels), "pattern_sim": s})
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(ref['indices'])}")

    df = pd.DataFrame(rows)
    n = len(ref["indices"])
    stats = {
        "split": split, "store": ref["store"], "H": H, "delta": delta, "max_paths": max_paths,
        "questions": n, "with_gold": n - no_path, "with_gold_share": round((n - no_path) / n, 4),
        "candidate_paths_mean": round(float(np.mean(n_cand)), 2) if n_cand else 0.0,
        "kept_paths_mean": round(float(np.mean(n_kept)), 2) if n_kept else 0.0,
        "kept_patterns_mean": round(float(np.mean(n_pat)), 2) if n_pat else 0.0,
        "kept_share_of_candidates": round(float(np.sum(n_kept) / max(np.sum(n_cand), 1)), 4),
        "length_counts": {int(k): int(v) for k, v in df["length"].value_counts().sort_index().items()} if len(df) else {},
    }
    return df, stats


def _show(df: pd.DataFrame, data_dir: str, store_name: str, k: int) -> None:
    from chase.kg.subgraphs import SubgraphStore
    store = SubgraphStore.open(data_dir, store_name)
    for i in df["q_index"].drop_duplicates().head(k):
        q, kg = store.questions.iloc[i], store.get(int(i))
        names = kg.nodes["name"].to_numpy()
        print(f"\n[{q['qid']}] {q['question']}  ->  {list(q['a_entity'])}")
        for _, p in df[df["q_index"] == i].head(5).iterrows():
            steps = " ".join(f"-[{store.relation_names[r]}]-> {names[v]}" for r, v in zip(p["rels"], p["nodes"][1:]))
            print(f"   ({p['pattern_sim']:.3f}) {names[p['nodes'][0]]} {steps}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold paths (training labels) for a RoG store")
    ap.add_argument("--data", required=True, help="e.g. data/processed/webqsp")
    ap.add_argument("--split", default="train")
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--max-paths", type=int, default=64)
    ap.add_argument("--show", type=int, default=5, help="print this many example questions")
    args = ap.parse_args()

    t0 = time.time()
    df, stats = build_gold_paths(args.data, args.split, args.H, args.delta, args.max_paths)
    out = Path(args.data) / "gold_paths"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / f"{args.split}.parquet", index=False)
    (out / f"{args.split}_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"saved {len(df):,} paths -> {out / f'{args.split}.parquet'} ({time.time() - t0:.0f}s)")
    if args.show:
        _show(df, args.data, stats["store"], args.show)


if __name__ == "__main__":
    main()
