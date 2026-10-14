from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chase.kg.graph import KG

FAR = 255            
FILENAME = "type_dist.npz"


def multi_source_bfs(kg: KG, sources: np.ndarray, max_dist: int) -> np.ndarray:
    dist = np.full(kg.num_nodes, FAR, dtype=np.uint8)
    frontier = np.unique(np.asarray(sources, dtype=np.int64))
    dist[frontier] = 0
    for d in range(1, max_dist + 1):
        if frontier.size == 0:
            break
        _, nbr, _ = kg.expand(frontier)
        nbr = np.unique(nbr)
        nbr = nbr[dist[nbr] == FAR]
        dist[nbr] = d
        frontier = nbr
    return dist


@dataclass
class TypeDistance:
    dist: np.ndarray          # uint8 [N, T]; column order = type_names
    type_names: list[str]
    max_dist: int

    def type_ids(self, names: list[str]) -> np.ndarray:
        return np.array([self.type_names.index(n) for n in names], dtype=np.int64)

    def feasible(self, nodes: np.ndarray, types: np.ndarray, remaining: int) -> np.ndarray:
        if remaining > self.max_dist:
            raise ValueError(f"remaining={remaining} exceeds max_dist={self.max_dist}; rebuild with a larger max_dist")
        return self.dist[np.asarray(nodes)][:, types].min(axis=1) <= remaining

    def save(self, kg_dir: str | Path) -> Path:
        path = Path(kg_dir) / FILENAME
        np.savez(path, dist=self.dist, type_names=np.array(self.type_names), max_dist=np.array(self.max_dist))
        return path

    @classmethod
    def load(cls, kg_dir: str | Path) -> "TypeDistance":
        z = np.load(Path(kg_dir) / FILENAME)
        return cls(dist=z["dist"], type_names=[str(t) for t in z["type_names"]], max_dist=int(z["max_dist"]))


def build_type_distance(kg: KG, max_dist: int = 4) -> TypeDistance:
    if not 0 < max_dist < FAR:
        raise ValueError(f"max_dist must be in 1..{FAR - 1}")
    cols = [multi_source_bfs(kg, np.flatnonzero(kg.node_type == t), max_dist)
            for t in range(len(kg.type_names))]
    return TypeDistance(dist=np.stack(cols, axis=1), type_names=list(kg.type_names), max_dist=max_dist)


def _print_stats(td: TypeDistance) -> None:
    ks = range(td.max_dist + 1)
    print("share of nodes with d_c(v) <= k" + "".join(f"{f'k={k}':>9}" for k in ks) + f"{'far':>9}")
    for t, name in enumerate(td.type_names):
        col = td.dist[:, t]
        cells = "".join(f"{(col <= k).mean():>9.1%}" for k in ks)
        print(f"  {name:<30}{cells}{(col == FAR).mean():>9.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="data/processed/primekg_full")
    ap.add_argument("--max-dist", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    kg = KG.load(args.kg)
    td = build_type_distance(kg, args.max_dist)
    path = td.save(args.kg)
    print(f"{kg.num_nodes:,} nodes x {len(td.type_names)} types, max_dist={td.max_dist}, "
          f"built in {time.time() - t0:.1f}s -> {path}")
    _print_stats(td)


if __name__ == "__main__":
    main()
