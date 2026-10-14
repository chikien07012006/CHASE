from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from chase.kg.graph import KG, build_csr


def _node_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (type, source, id); the most frequent name is primary, the rest are aliases."""
    cols = ["id", "type", "name", "source"]
    both = pd.concat([
        df[["x_id", "x_type", "x_name", "x_source"]].set_axis(cols, axis=1),
        df[["y_id", "y_type", "y_name", "y_source"]].set_axis(cols, axis=1),
    ])
    both["key"] = both["type"] + "|" + both["source"] + "|" + both["id"]
    counts = both.groupby(["key", "name"], sort=False).size().rename("n").reset_index()
    counts = counts.sort_values(["key", "n"], ascending=[True, False])
    primary = counts.drop_duplicates("key")[["key", "name"]]
    aliases = (counts[counts.duplicated("key")].groupby("key")["name"]
               .agg(list).rename("aliases"))
    meta = both.drop_duplicates("key")[["key", "type", "source", "id"]]
    nodes = meta.merge(primary, on="key").merge(aliases, on="key", how="left")
    nodes["aliases"] = nodes["aliases"].apply(lambda a: a if isinstance(a, list) else [])
    return nodes.sort_values("key").reset_index(drop=True)


def build(csv_path: str) -> KG:
    t0 = time.time()
    df = pd.read_csv(csv_path, dtype=str)
    print(f"read {len(df):,} rows in {time.time() - t0:.1f}s")

    nodes = _node_table(df)
    key_to_id = pd.Series(np.arange(len(nodes)), index=nodes["key"])

    xi = key_to_id.loc[(df["x_type"] + "|" + df["x_source"] + "|" + df["x_id"]).values].to_numpy()
    yi = key_to_id.loc[(df["y_type"] + "|" + df["y_source"] + "|" + df["y_id"]).values].to_numpy()

    rel_label = df["relation"] + ": " + df["display_relation"]
    relation_names = sorted(rel_label.unique())
    rel_id = pd.Series(np.arange(len(relation_names)), index=relation_names).loc[rel_label.values].to_numpy()

    # deduplicate undirected edges (the CSV has both directions plus a few exact repeats)
    lo, hi = np.minimum(xi, yi), np.maximum(xi, yi)
    und = pd.DataFrame({"r": rel_id, "lo": lo, "hi": hi}).drop_duplicates()
    print(f"{len(und):,} unique undirected edges")

    src = np.concatenate([und["lo"].to_numpy(), und["hi"].to_numpy()])
    dst = np.concatenate([und["hi"].to_numpy(), und["lo"].to_numpy()])
    rel = np.concatenate([und["r"].to_numpy(), und["r"].to_numpy()])
    indptr, indices, rel = build_csr(src, dst, rel, num_nodes=len(nodes))

    type_names = sorted(nodes["type"].dropna().unique())
    node_type = pd.Series(np.arange(len(type_names)), index=type_names).loc[nodes["type"].values].to_numpy()

    print(f"built in {time.time() - t0:.1f}s")
    return KG(indptr=indptr, indices=indices, rel=rel, node_type=node_type.astype(np.int8),
              type_names=type_names, relation_names=relation_names, nodes=nodes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/primekg/kg_giant.csv")
    ap.add_argument("--out", default="data/processed/primekg_full")
    args = ap.parse_args()

    kg = build(args.csv)
    kg.save(args.out)

    deg = kg.degree()
    print(f"nodes {kg.num_nodes:,} | directed edges {kg.num_edges:,} | "
          f"relations {len(kg.relation_names)} | types {len(kg.type_names)}")
    print(f"degree: median {int(np.median(deg))}, p99 {int(np.percentile(deg, 99))}, max {deg.max():,}, "
          f"isolated {(deg == 0).sum()}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
