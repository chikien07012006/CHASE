from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chase.kg.build_freebase import NAMES_SCHEMA, SPLITS
from chase.kg.graph import KG, build_csr
from chase.kg.type_distance import TypeDistance

H = 2
FRACTIONS = (0.7, 0.15, 0.15)      # train / validation / test


class NameIndex:
    """(type, name) -> node ids. Exact primary names first; then case-folded primary names and aliases."""

    def __init__(self, kg: KG):
        self.exact: dict[tuple[str, str], list[int]] = {}
        self.folded: dict[tuple[str, str], list[int]] = {}
        for i, (t, name, aliases) in enumerate(zip(kg.nodes["type"], kg.nodes["name"], kg.nodes["aliases"])):
            self.exact.setdefault((t, name), []).append(i)
            for n in {name, *aliases}:
                ids = self.folded.setdefault((t, n.casefold()), [])
                if i not in ids:
                    ids.append(i)

    def link(self, node_type: str, name: str) -> tuple[list[int], str]:
        """Node ids and how they were found: 'exact', 'folded' or 'none'."""
        if (ids := self.exact.get((node_type, name))):
            return ids, "exact"
        if (ids := self.folded.get((node_type, name.casefold()))):
            return ids, "folded"
        return [], "none"


def load_questions(path: str | Path) -> pd.DataFrame:
    """One row per question: 2-hop questions of every record, plus 1-hop questions (deduplicated, answers merged)."""
    raw = pd.read_json(path)
    two = pd.DataFrame({"question": raw["hop2_question"], "hops": 2, "topic": raw["hop2"], "topic_type": raw["hop2_type"],
                        "answer_type": raw["target_type"], "pattern": raw["relation_hop2"], "answers": raw["answer"]})
    one = pd.DataFrame({"question": raw["hop1_question"], "hops": 1, "topic": raw["hop1"], "topic_type": raw["hop1_type"],
                        "answer_type": raw["target_type"], "pattern": raw["relation_hop1"], "answers": raw["answer"]})
    one = (one.groupby("question", sort=False)
           .agg({"hops": "first", "topic": "first", "topic_type": "first", "answer_type": "first", "pattern": "first",
                 "answers": lambda a: list(dict.fromkeys(x for lst in a for x in lst))})
           .reset_index())
    q = pd.concat([two, one], ignore_index=True)
    q["group"] = pd.factorize(q["topic_type"] + "|" + q["topic"])[0]
    q["qid"] = [f"G{g}_{k}" for k, g in enumerate(q["group"])]
    return q


def split_by_group(groups: np.ndarray, fractions=FRACTIONS, seed: int = 0) -> np.ndarray:
    """Split index (0, 1, 2) per question; whole groups, in random order, until each part holds its share."""
    perm = np.random.default_rng(seed).permutation(groups.max() + 1)
    sizes = np.bincount(groups, minlength=groups.max() + 1)[perm]
    start = (np.cumsum(sizes) - sizes) / len(groups)          # share of questions placed before each group
    part_of_group = np.empty(len(perm), dtype=np.int64)
    part_of_group[perm] = np.searchsorted(np.cumsum(fractions)[:-1], start, side="right")
    return part_of_group[groups]


def typed_subgraph(kg: KG, td: TypeDistance, topic: np.ndarray, answer_type: int):
    """Nodes (global ids, sorted) and local CSR of the walks of <= 2 hops from `topic` that can end at answer_type."""
    topic = np.unique(np.asarray(topic, dtype=np.int64))
    if topic.size == 0:
        return np.zeros(0, np.int64), np.zeros(1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32)
    s1, d1, r1 = kg.expand(topic)
    k1 = td.dist[d1, answer_type] <= 1
    s1, d1, r1 = s1[k1], d1[k1], r1[k1]
    hop1 = np.unique(d1)
    s2, d2, r2 = kg.expand(hop1)
    k2 = kg.node_type[d2] == answer_type
    s2, d2, r2 = s2[k2], d2[k2], r2[k2]
    nodes = np.unique(np.concatenate([topic, hop1, d2]))
    # both directions, as in PrimeKG and the Freebase stores (gold-path search walks back from the answers);
    # a node can be both a topic and a hop-1 node, so the same edge can come from both hops
    src, dst, rel = np.r_[s1, s2], np.r_[d1, d2], np.r_[r1, r2]
    edges = np.unique(np.stack([np.r_[src, dst], np.r_[dst, src], np.r_[rel, rel]], axis=1), axis=0)
    src, dst = np.searchsorted(nodes, edges[:, 0]), np.searchsorted(nodes, edges[:, 1])
    indptr, indices, rel = build_csr(src, dst, edges[:, 2], num_nodes=len(nodes))
    return nodes, indptr, indices, rel.astype(np.int32)


def build(raw: str, kg_dir: str, out: str, seed: int = 0) -> dict:
    t0 = time.time()
    kg, td = KG.load(kg_dir), TypeDistance.load(kg_dir)
    type_id = {t: i for i, t in enumerate(kg.type_names)}
    if td.type_names != kg.type_names:
        raise ValueError("type_dist.npz and the KG list types in a different order; rebuild type_dist")
    index = NameIndex(kg)
    q = load_questions(Path(raw) / "BioHopR.json")
    part = split_by_group(q["group"].to_numpy(), seed=seed)
    names = kg.nodes["name"].to_numpy()
    out_dir = Path(out)
    stats = {"seed": seed, "fractions": FRACTIONS, "H": H, "questions": len(q),
             "by_hops": {int(k): int(v) for k, v in q["hops"].value_counts().items()}}

    link_topic, link_answer = {"exact": 0, "folded": 0, "none": 0}, {"exact": 0, "folded": 0, "none": 0}
    for s, split in enumerate(SPLITS):
        rows = q[part == s]
        d = out_dir / split
        d.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(d / "node_names.parquet", NAMES_SCHEMA)
        indptrs, indices, rels, ntypes, questions = [], [], [], [], []
        node_offset, edge_offset = [0], [0]
        for r in rows.itertuples():
            topic, how = index.link(r.topic_type, r.topic)
            link_topic[how] += 1
            T = type_id[r.answer_type]
            nodes, ip, ix, rl = typed_subgraph(kg, td, np.array(topic, dtype=np.int64), T)
            local = {int(v): k for k, v in enumerate(nodes)}
            answers = set()
            for a in r.answers:
                ids, how = index.link(r.answer_type, a)
                link_answer[how] += 1
                answers.update(local[i] for i in ids if i in local)
            nt = (kg.node_type[nodes] != T).astype(np.int8)
            answers = sorted(a for a in answers if nt[a] == 0)
            questions.append({
                "qid": r.qid, "question": r.question, "q_entity": [r.topic], "a_entity": list(r.answers),
                "topic_nodes": [local[i] for i in topic], "answer_nodes": answers,
                "topic_in_graph": len(topic) > 0, "answer_in_graph": len(answers) > 0,
                "num_nodes": len(nodes), "num_edges": len(ix), "hops": r.hops, "answer_type": r.answer_type,
                "pattern": r.pattern, "topic_key": [kg.nodes["key"].iat[i] for i in topic]})
            indptrs.append(ip); indices.append(ix); rels.append(rl); ntypes.append(nt)
            node_offset.append(node_offset[-1] + len(nodes))
            edge_offset.append(edge_offset[-1] + len(ix))
            writer.write_table(pa.table({"names": [list(names[nodes])]}, schema=NAMES_SCHEMA))
        writer.close()
        np.savez(d / "csr.npz",
                 node_offset=np.asarray(node_offset, np.int64), edge_offset=np.asarray(edge_offset, np.int64),
                 indptr=np.concatenate(indptrs), indices=np.concatenate(indices),
                 rel=np.concatenate(rels), node_type=np.concatenate(ntypes))
        qs = pd.DataFrame(questions)
        qs.to_parquet(d / "questions.parquet", index=False)
        stats[split] = {"questions": len(qs), "topics": int(rows["group"].nunique()),
                        "answer_in_graph": round(float(qs["answer_in_graph"].mean()), 4),
                        "topic_missing": int((~qs["topic_in_graph"]).sum()),
                        "nodes_median": int(qs["num_nodes"].median()), "edges_median": int(qs["num_edges"].median()),
                        "edges_p90": int(qs["num_edges"].quantile(0.9)), "edges_max": int(qs["num_edges"].max())}
        print(f"{split}: {stats[split]}")

    stats["topic_linking"], stats["answer_linking"] = link_topic, link_answer
    (out_dir / "relations.json").write_text(json.dumps(kg.relation_names, ensure_ascii=False))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"topic linking {link_topic} | answer linking {link_answer} | {time.time() - t0:.0f}s -> {out_dir}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack BioHopR on full PrimeKG as a per-question subgraph store")
    ap.add_argument("--raw", default="data/raw/biohopr")
    ap.add_argument("--kg", default="data/processed/primekg_full", help="needs type_dist.npz (chase.kg.type_distance)")
    ap.add_argument("--out", default="data/processed/biohopr")
    ap.add_argument("--seed", type=int, default=0, help="seed of the train / validation / test split")
    args = ap.parse_args()
    build(args.raw, args.kg, args.out, args.seed)


if __name__ == "__main__":
    main()
