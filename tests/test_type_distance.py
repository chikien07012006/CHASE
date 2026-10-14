import networkx as nx
import numpy as np
import pandas as pd
import pytest

from chase.kg.graph import KG, build_csr
from chase.kg.type_distance import FAR, TypeDistance, build_type_distance, multi_source_bfs

N, T = 60, 3


def _random_kg(seed: int = 0, num_edges: int = 80) -> KG:
    rng = np.random.default_rng(seed)
    u, v = rng.integers(0, N, num_edges), rng.integers(0, N, num_edges)
    keep = u != v
    u, v = u[keep], v[keep]
    src, dst = np.concatenate([u, v]), np.concatenate([v, u])
    indptr, indices, rel = build_csr(src, dst, np.zeros_like(src), N)
    return KG(indptr=indptr, indices=indices, rel=rel, node_type=rng.integers(0, T, N).astype(np.int8),
              type_names=[f"t{i}" for i in range(T)], relation_names=["r"],
              nodes=pd.DataFrame({"name": [str(i) for i in range(N)]}))


def _nx_graph(kg: KG) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(kg.num_nodes))
    src = np.repeat(np.arange(kg.num_nodes), kg.degree())
    g.add_edges_from(zip(src.tolist(), kg.indices.tolist()))
    return g


@pytest.mark.parametrize("seed", range(5))
def test_matches_networkx(seed):
    kg = _random_kg(seed)
    g = _nx_graph(kg)
    max_dist = 3
    for t in range(T):
        sources = np.flatnonzero(kg.node_type == t)
        ours = multi_source_bfs(kg, sources, max_dist)
        ref = nx.multi_source_dijkstra_path_length(g, set(sources.tolist())) if len(sources) else {}
        for v in range(N):
            d = ref.get(v)
            expected = d if d is not None and d <= max_dist else FAR
            assert ours[v] == expected, (t, v, ours[v], d)


def test_feasibility_never_drops_a_feasible_walk():
    """Lemma 1: every prefix of a walk that reaches a node of the target type within H hops is kept."""
    kg = _random_kg(1, num_edges=120)
    H, target = 3, np.array([0])
    td = build_type_distance(kg, max_dist=H)
    adj = [kg.neighbors(u)[0] for u in range(N)]

    def walks(prefix):
        yield prefix
        if len(prefix) - 1 < H:
            for w in adj[prefix[-1]]:
                yield from walks(prefix + [int(w)])

    for start in range(N):
        for walk in walks([start]):
            if kg.node_type[walk[-1]] != target[0]:
                continue
            for h, node in enumerate(walk):
                assert td.feasible(np.array([node]), target, H - h)[0]


def test_remaining_beyond_max_dist_raises():
    td = build_type_distance(_random_kg(2), max_dist=2)
    with pytest.raises(ValueError):
        td.feasible(np.array([0]), np.array([0]), remaining=3)


def test_save_load_roundtrip(tmp_path):
    td = build_type_distance(_random_kg(3), max_dist=4)
    td.save(tmp_path)
    back = TypeDistance.load(tmp_path)
    assert back.type_names == td.type_names and back.max_dist == td.max_dist
    np.testing.assert_array_equal(back.dist, td.dist)
