import numpy as np
import pandas as pd
import pytest
import torch

from chase.kg.graph import KG, build_csr
from chase.model.features import GraphFeatures
from chase.model.scorer import ScorerConfig, StateScorer
from chase.search.dp import bottleneck_dp
from chase.train.losses import (BIG, UNREACHABLE, alpha_eff, bce_loss, conftr_size_loss, smooth_max, smooth_min,
                                soft_bottleneck_dp, soft_wrong_set_size, stage2_loss)

N, R, H = 25, 3, 3


def _random_kg(rng, num_edges=55) -> KG:
    src, dst, rel = rng.integers(0, N, num_edges), rng.integers(0, N, num_edges), rng.integers(0, R, num_edges)
    indptr, indices, rel = build_csr(src, dst, rel, N)
    return KG(indptr=indptr, indices=indices, rel=rel, node_type=np.zeros(N, np.int8), type_names=["t"],
              relation_names=[f"r{i}" for i in range(R)], nodes=pd.DataFrame({"name": [str(i) for i in range(N)]}))


class _TableScorer:
    def __init__(self, step, stop):
        self.step, self.stop = step, stop

    def edge_score(self, h, src, rel, dst):
        return self.step[h, rel, dst]

    def stop_score(self, h, nodes):
        return self.stop[h, nodes]


def _soft_inputs(kg, step, stop):
    src = torch.from_numpy(np.repeat(np.arange(kg.num_nodes), kg.degree()))
    dst = torch.from_numpy(kg.indices.astype(np.int64))
    rel = kg.rel.astype(np.int64)
    edge_s = torch.tensor(np.stack([step[h, rel, kg.indices] for h in range(H + 1)]), dtype=torch.float64)
    return src, dst, edge_s, torch.tensor(stop, dtype=torch.float64)


# ------------------------------------------------------------------ smooth max / min

def test_smooth_max_min_converge_to_hard():
    x = torch.tensor([0.3, 0.9, 0.1, 0.5])
    assert abs(smooth_max(x, 1e-3).item() - 0.9) < 1e-2
    assert abs(smooth_min(x, 1e-3).item() - 0.1) < 1e-2
    assert smooth_max(x, 0.1).item() >= 0.9 and smooth_min(x, 0.1).item() <= 0.1   # one-sided bias


# ------------------------------------------------------------------ soft DP

@pytest.mark.parametrize("seed", range(15))
def test_soft_dp_matches_hard_dp_at_low_temperature(seed):
    rng = np.random.default_rng(seed)
    kg = _random_kg(rng)
    step, stop = rng.random((H + 1, R, N)), rng.random((H + 1, N))
    topic = sorted(set(rng.integers(0, N, 2).tolist()))
    mask = rng.random(N) < 0.8
    lam = 0.8
    hard = bottleneck_dp(kg, topic, _TableScorer(step, stop), H, lam=lam, answer_mask=mask).sigma
    src, dst, edge_s, stop_s = _soft_inputs(kg, step, stop)
    soft = soft_bottleneck_dp(N, src, dst, edge_s, stop_s, torch.tensor(topic), torch.tensor(mask), H,
                              lam=lam, T=1e-4).numpy()
    finite = np.isfinite(hard)
    np.testing.assert_array_equal(finite, soft < UNREACHABLE)
    np.testing.assert_allclose(soft[finite], hard[finite], atol=5e-3)


def test_soft_dp_gradients_are_finite_with_unreachable_nodes():
    rng = np.random.default_rng(0)
    kg = _random_kg(rng, num_edges=20)          # sparse: many unreachable nodes
    step, stop = rng.random((H + 1, R, N)), rng.random((H + 1, N))
    src, dst, edge_s, stop_s = _soft_inputs(kg, step, stop)
    edge_s.requires_grad_(True)
    stop_s.requires_grad_(True)
    sigma = soft_bottleneck_dp(N, src, dst, edge_s, stop_s, torch.tensor([0]), torch.ones(N, dtype=torch.bool), H)
    (sigma[sigma < UNREACHABLE].sum() + sigma.clamp_max(10).sum()).backward()
    assert torch.isfinite(edge_s.grad).all() and torch.isfinite(stop_s.grad).all()


def test_forbidden_edges_are_never_used():
    rng = np.random.default_rng(1)
    kg = _random_kg(rng)
    step, stop = rng.random((H + 1, R, N)), rng.random((H + 1, N))
    src, dst, edge_s, stop_s = _soft_inputs(kg, step, stop)
    sigma = soft_bottleneck_dp(N, src, dst, torch.full_like(edge_s, BIG), stop_s, torch.tensor([0]),
                               torch.ones(N, dtype=torch.bool), H)
    assert (sigma >= UNREACHABLE).all()


# ------------------------------------------------------------------ BCE

def test_bce_small_for_correct_logits_and_handles_empty():
    y = torch.tensor([1, 0, 0, 0, 0, 0])
    good = bce_loss(torch.tensor([8.0, -8, -8, -8, -8, -8]), y, torch.empty(0), torch.empty(0))
    bad = bce_loss(torch.tensor([-8.0, 8, 8, 8, 8, 8]), y, torch.empty(0), torch.empty(0))
    assert good.item() < 0.01 < bad.item()


# ------------------------------------------------------------------ conformal training

def test_alpha_eff():
    assert abs(alpha_eff(0.1, 0.95) - (1 - 0.9 / 0.95)) < 1e-12
    with pytest.raises(ValueError):
        alpha_eff(0.1, 0.8)                     # 90% overall is impossible when only 80% is reachable


def test_wrong_set_size_ignores_gold_answers():
    sigma = torch.tensor([0.1, 0.2, 0.9, BIG])
    gold = torch.tensor([0])
    base = soft_wrong_set_size(sigma, gold, torch.tensor(0.5), 0.1)
    moved = soft_wrong_set_size(torch.tensor([0.9, 0.2, 0.9, BIG]), gold, torch.tensor(0.5), 0.1)
    assert torch.isclose(base, moved)           # moving the gold answer does not change the penalty
    assert 0.9 < base.item() < 1.1              # node 1 counts ~1, node 2 ~0, node 3 (unreachable) 0


def test_conftr_loss_gradients_push_the_right_way():
    torch.manual_seed(0)
    sigmas = [torch.rand(30, requires_grad=True) for _ in range(8)]
    golds = [torch.tensor([i]) for i in range(8)]
    half1, half2 = [0, 1, 2, 3], [4, 5, 6, 7]
    loss, tau = conftr_size_loss(sigmas, golds, half1, half2, alpha_train=0.25)
    stage2_loss(loss, torch.zeros(()), mu=0.5).backward()
    assert torch.isfinite(tau) and loss.item() > 0
    for i in half2:                              # wrong nodes are pushed up (gradient of loss < 0 means raise sigma)
        g = sigmas[i].grad.clone()
        g[golds[i]] = 0
        assert (g <= 0).all()
        assert sigmas[i].grad[golds[i]].item() == 0
    assert sum(sigmas[i].grad[golds[i]].item() for i in half1) > 0   # gold answers in half1 are pushed down via tau


def test_conftr_skips_unreachable_in_half1_keeps_them_in_half2():
    reach = torch.tensor([0.2, 0.4, 0.6])
    unreach = torch.tensor([BIG, 0.3, 0.5])     # its gold (node 0) cannot be reached
    sigmas = [reach, unreach, reach.clone(), unreach.clone()]
    golds = [torch.tensor([0])] * 4
    loss, tau = conftr_size_loss(sigmas, golds, half1=[0, 1], half2=[2, 3], alpha_train=0.5)
    assert tau.item() < UNREACHABLE              # unreachable question did not push tau to BIG
    assert loss.item() > 0


# ------------------------------------------------------------------ scorer tables

def test_tables_match_bind():
    torch.manual_seed(0)
    graph = GraphFeatures(node=torch.randn(40, 9), relation=torch.randn(7, 5))
    scorer = StateScorer(ScorerConfig(q_dim=6, rel_dim=5, node_dim=9, hidden=16, proj=4)).eval()
    q = torch.randn(6)
    bound = scorer.bind(q, scorer.encode_graph(graph), H)
    rel_ids, node_ids = torch.tensor([1, 4, 6]), torch.tensor([0, 3, 7, 39])
    t = scorer.tables(q, graph, H, rel_ids=rel_ids, node_ids=node_ids)
    rel_pos, dst_pos = torch.tensor([0, 2, 1]), torch.tensor([3, 0, 2])
    with torch.no_grad():
        got = t.edge_scores(rel_pos, dst_pos).numpy()
        stop = t.stop_scores().numpy()
    for h in range(H + 1):
        np.testing.assert_allclose(got[h], bound.edge_score(h, None, rel_ids[rel_pos].numpy(),
                                                            node_ids[dst_pos].numpy()), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(stop[h], bound.stop_score(h, node_ids.numpy()), rtol=1e-5, atol=1e-5)
