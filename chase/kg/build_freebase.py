from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chase.kg.graph import build_csr

CVT_PATTERN = re.compile(r"^[mg]\.[0-9a-z_]+$")
SPLITS = ["train", "validation", "test"]
NAMES_SCHEMA = pa.schema([("names", pa.list_(pa.string()))])  # fixed: some subgraphs are empty


class RelationVocab:
    """Shared across splits. Relation r gets id 2k, its inverse 2k+1."""

    def __init__(self):
        self.base: dict[str, int] = {}

    def forward_ids(self, names: list[str]) -> np.ndarray:
        ids = [self.base.setdefault(n, len(self.base)) for n in names]
        return 2 * np.asarray(ids, dtype=np.int64)

    def names(self) -> list[str]:
        out = []
        for n in self.base:      # dict keeps insertion order == base id order
            out += [n, f"{n} (inverse)"]
        return out


def _question_graph(triples, vocab: RelationVocab):
    """Local CSR of one question. Each triple is stored forward and inverse; self-loops dropped."""
    if len(triples) == 0:
        return np.zeros(1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int8), []
    h = [t[0] for t in triples]
    r = [t[1] for t in triples]
    t_ = [t[2] for t in triples]
    codes, names = pd.factorize(pd.Series(h + t_), sort=False)
    hi, ti = codes[: len(h)], codes[len(h):]
    ri = vocab.forward_ids(r)

    trip = np.unique(np.stack([hi, ri, ti], axis=1), axis=0)
    trip = trip[trip[:, 0] != trip[:, 2]]
    src = np.concatenate([trip[:, 0], trip[:, 2]])
    dst = np.concatenate([trip[:, 2], trip[:, 0]])
    rel = np.concatenate([trip[:, 1], trip[:, 1] + 1])
    indptr, indices, rel = build_csr(src, dst, rel, num_nodes=len(names))
    node_type = np.array([1 if CVT_PATTERN.match(n) else 0 for n in names], dtype=np.int8)
    return indptr, indices, rel.astype(np.int32), node_type, list(names)


def build_split(raw_dir: str, split: str, out_dir: Path, vocab: RelationVocab) -> dict:
    files = sorted(glob.glob(f"{raw_dir}/data/{split}-*.parquet"))
    out_dir.mkdir(parents=True, exist_ok=True)
    names_writer = pq.ParquetWriter(out_dir / "node_names.parquet", NAMES_SCHEMA)

    indptrs, indices, rels, ntypes = [], [], [], []
    node_offset, edge_offset = [0], [0]
    questions = []
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=200):
            for row in batch.to_pylist():
                ip, ix, rl, nt, names = _question_graph(row["graph"], vocab)
                local = {n: i for i, n in enumerate(names)}
                topic = [local[e] for e in row["q_entity"] if e in local]
                answers = sorted({local[e] for e in row["a_entity"] if e in local and nt[local[e]] == 0})
                questions.append({
                    "qid": row["id"], "question": row["question"],
                    "q_entity": list(row["q_entity"]), "a_entity": list(row["a_entity"]),
                    "topic_nodes": topic, "answer_nodes": answers,
                    "topic_in_graph": len(topic) > 0, "answer_in_graph": len(answers) > 0,
                    "num_nodes": len(names), "num_edges": len(ix),
                })
                indptrs.append(ip); indices.append(ix); rels.append(rl); ntypes.append(nt)
                node_offset.append(node_offset[-1] + len(names))
                edge_offset.append(edge_offset[-1] + len(ix))
                names_writer.write_table(pa.table({"names": [names]}, schema=NAMES_SCHEMA))
    names_writer.close()

    np.savez(out_dir / "csr.npz",
             node_offset=np.asarray(node_offset, np.int64), edge_offset=np.asarray(edge_offset, np.int64),
             indptr=np.concatenate(indptrs), indices=np.concatenate(indices),
             rel=np.concatenate(rels), node_type=np.concatenate(ntypes))
    q = pd.DataFrame(questions)
    q.to_parquet(out_dir / "questions.parquet", index=False)
    return {
        "questions": len(q),
        "answer_in_graph": round(float(q["answer_in_graph"].mean()), 4),
        "topic_missing": int((~q["topic_in_graph"]).sum()),
        "empty_graph": int((q["num_nodes"] == 0).sum()),
        "nodes_median": int(q["num_nodes"].median()),
        "edges_median": int(q["num_edges"].median()),
        "edges_total": int(q["num_edges"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="e.g. data/raw/rog_webqsp")
    ap.add_argument("--out", required=True, help="e.g. data/processed/webqsp")
    args = ap.parse_args()

    out = Path(args.out)
    vocab = RelationVocab()
    stats = {}
    for split in SPLITS:
        t0 = time.time()
        stats[split] = build_split(args.raw, split, out / split, vocab)
        print(f"{split}: {stats[split]} ({time.time() - t0:.0f}s)")
    (out / "relations.json").write_text(json.dumps(vocab.names(), ensure_ascii=False))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"{len(vocab.base)} base relations -> {2 * len(vocab.base)} directed; saved to {out}")


if __name__ == "__main__":
    main()
