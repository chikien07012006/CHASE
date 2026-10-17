import json

import numpy as np
import pandas as pd
import torch

from chase.kg.graph import KG, build_csr
from chase.model.features import GraphFeatures
from chase.model.scorer import ScorerConfig, StateScorer
from chase.train.evaluate import SigmaRecord, set_sizes, two_fold_metrics
from chase.train.sampler import Sampler
from chase.train.train import TrainConfig, load_scorer, train


# ------------------------------------------------------------------ evaluation helpers

def test_set_sizes_and_two_fold_metrics():
    recs = [SigmaRecord(sigma_star=s, sorted_sigma=np.array([0.1, 0.2, 0.5, 0.9]), states_expanded=0)
            for s in [0.1, 0.2, 0.2, 0.5, 0.1, 0.2, 0.2, 0.5]]
    np.testing.assert_array_equal(set_sizes(recs[:2], 0.2), [2, 2])
    m = two_fold_metrics(recs, alpha=0.4)
    assert m["feasible"] and 0.0 <= m["ecr"] <= 1.0 and m["apss"] > 0
    bad = two_fold_metrics([SigmaRecord(np.inf, np.array([]), 0)] * 8, alpha=0.4)
    assert not bad["feasible"] and bad["apss"] == np.inf


# ------------------------------------------------------------------ end-to-end on a toy store

def _kg():
    edges = [(0, 1, 1), (1, 2, 3), (0, 5, 4), (0, 6, 5), (1, 7, 2), (4, 8, 5)]
    u = np.array([e[0] for e in edges]); r = np.array([e[1] for e in edges]); v = np.array([e[2] for e in edges])
    indptr, indices, rel = build_csr(np.concatenate([u, v]), np.concatenate([v, u]), np.concatenate([r, r + 10]), 6)
    node_type = np.zeros(6, np.int8)
    node_type[1] = 1
    return KG(indptr=indptr, indices=indices, rel=rel, node_type=node_type, type_names=["entity", "cvt"],
              relation_names=[], nodes=pd.DataFrame({"name": [str(i) for i in range(6)]}))


class _Store:
    def __init__(self, n):
        self.kg = _kg()
        self.questions = pd.DataFrame({"topic_nodes": [[0]] * n, "answer_nodes": [[3]] * n})

    def get(self, i):
        return self.kg


class _Features:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.node, self.rel = torch.randn(6, 7, generator=g), torch.randn(20, 5, generator=g)
        self.q = torch.randn(4, generator=g)

    def graph(self, i, kg=None):
        return GraphFeatures(node=self.node, relation=self.rel)

    def question(self, i):
        return self.q


def test_train_runs_both_stages_and_saves_checkpoints(tmp_path):
    n = 10
    gold = pd.DataFrame({"q_index": list(range(8)), "nodes": [[0, 1, 3]] * 8, "rels": [[1, 2]] * 8})
    sampler = Sampler(_Store(n), _Features(), gold, indices=range(n), H=2, k_easy=4)
    val = (_Store(n), _Features(), list(range(n)))
    scorer = StateScorer(ScorerConfig(q_dim=4, rel_dim=5, node_dim=7, hidden=16, proj=8))
    cfg = TrainConfig(alpha=0.3, stage1_epochs=2, stage2_epochs=1, batch1=4, batch2=6)

    history = train(cfg, sampler, val, scorer, tmp_path)

    assert abs(history["rho_train"] - 0.8) < 1e-12
    assert all(np.isfinite(e["loss"]) for e in history["epochs"])
    assert [e["stage"] for e in history["epochs"]] == [1, 1, 2]
    for f in ["stage1_best.pt", "stage2_best.pt", "best.pt", "history.json"]:
        assert (tmp_path / f).exists()
    assert json.loads((tmp_path / "history.json").read_text())["winner"] in (1, 2)
    restored = load_scorer(tmp_path / "best.pt")
    for a, b in zip(restored.state_dict().values(), torch.load(tmp_path / "best.pt")["state_dict"].values()):
        assert torch.equal(a, b)
