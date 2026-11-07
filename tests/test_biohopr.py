import numpy as np
import pandas as pd
import pytest

from chase.data.biohopr import NameIndex, split_by_group, typed_subgraph
from chase.kg.graph import KG, build_csr
from chase.kg.type_distance import build_type_distance
from chase.search.dp import bottleneck_dp

N, R, T = 40, 3, 3


class GlobalScorer:
    """Scores indexed by global node ids; `nodes` maps the ids of a subgraph back to global ids."""

    def __init__(self, rng, nodes=None):
        self.step = rng.random((3, R, N))
        self.stop = rng.random((3, N))
        self.nodes = np.arange(N) if nodes is None else nodes

    def on(self, nodes):
        s = GlobalScorer.__new__(GlobalScorer)
        s.step, s.stop, s.nodes = self.step, self.stop, nodes
        return s

    def edge_score(self, h, src, rel, dst):
        return self.step[h, rel, self.nodes[dst]]

    def stop_score(self, h, nodes):
        return self.stop[h, self.nodes[nodes]]


def _typed_kg(rng, num_edges=70) -> KG:
    """Undirected random graph (both directions stored, like PrimeKG) with T node types."""
    a, b = rng.integers(0, N, num_edges), rng.integers(0, N, num_edges)
    rel = rng.integers(0, R, num_edges)
    indptr, indices, rel = build_csr(np.r_[a, b], np.r_[b, a], np.r_[rel, rel], N)
    node_type = rng.integers(0, T, N).astype(np.int8)
    names = [f"n{i}" for i in range(N)]
    return KG(indptr=indptr, indices=indices, rel=rel, node_type=node_type, type_names=["a", "b", "c"],
              relation_names=[f"r{i}" for i in range(R)],
              nodes=pd.DataFrame({"key": names, "type": [["a", "b", "c"][t] for t in node_type], "name": names,
                                  "aliases": [[] for _ in range(N)]}))


@pytest.mark.parametrize("seed", range(30))
def test_typed_subgraph_keeps_sigma_of_every_answer_type_node(seed):
    rng = np.random.default_rng(seed)
    kg = _typed_kg(rng)
    td = build_type_distance(kg, max_dist=2)
    topic = np.unique(rng.integers(0, N, 2))
    answer_type = int(rng.integers(0, T))
    scorer = GlobalScorer(rng)

    full = bottleneck_dp(kg, topic, scorer, 2, answer_mask=kg.node_type == answer_type)
    nodes, indptr, indices, rel = typed_subgraph(kg, td, topic, answer_type)
    sub = KG(indptr=indptr, indices=indices, rel=rel, node_type=(kg.node_type[nodes] != answer_type).astype(np.int8),
             type_names=["answer", "other"], relation_names=kg.relation_names, nodes=pd.DataFrame({"name": nodes}))
    local = bottleneck_dp(sub, np.searchsorted(nodes, topic), scorer.on(nodes), 2, answer_mask=sub.node_type == 0)

    expected = np.full(N, np.inf)
    expected[nodes] = local.sigma
    assert np.allclose(full.sigma, expected)          # same finite sigma, and nothing finite left out


def test_split_by_group_keeps_groups_whole_and_matches_fractions():
    groups = np.random.default_rng(0).integers(0, 400, 5000)
    part = split_by_group(groups, (0.7, 0.15, 0.15), seed=1)
    assert all(len(set(part[groups == g])) == 1 for g in np.unique(groups))
    share = np.bincount(part, minlength=3) / len(part)
    assert np.allclose(share, [0.7, 0.15, 0.15], atol=0.02)


def test_name_index_prefers_exact_then_folded_names_and_aliases():
    kg = KG(indptr=np.zeros(4, np.int64), indices=np.zeros(0, np.int32), rel=np.zeros(0, np.int16),
            node_type=np.zeros(3, np.int8), type_names=["drug", "disease"], relation_names=[],
            nodes=pd.DataFrame({"key": ["k0", "k1", "k2"], "type": ["drug", "drug", "disease"],
                                "name": ["Aspirin", "aspirin", "Pain"], "aliases": [[], ["ASA"], []]}))
    index = NameIndex(kg)
    assert index.link("drug", "Aspirin") == ([0], "exact")
    assert index.link("drug", "ASPIRIN") == ([0, 1], "folded")
    assert index.link("drug", "asa") == ([1], "folded")
    assert index.link("disease", "Aspirin") == ([], "none")      # the type must match
