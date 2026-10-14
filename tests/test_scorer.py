import numpy as np
import torch

from chase.model.features import GraphFeatures
from chase.model.scorer import ScorerConfig, StateScorer, nonconformity
from chase.model.text_cache import readable_relation

N, R, H, Q_DIM, NODE_DIM, REL_DIM = 50, 12, 3, 16, 21, 17


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    graph = GraphFeatures(node=torch.randn(N, NODE_DIM), relation=torch.randn(R, REL_DIM))
    scorer = StateScorer(ScorerConfig(q_dim=Q_DIM, rel_dim=REL_DIM, node_dim=NODE_DIM, hidden=32, proj=8)).eval()
    return graph, scorer, torch.randn(Q_DIM)


def test_bind_matches_forward():
    graph, scorer, q = _setup()
    bound = scorer.bind(q, scorer.encode_graph(graph), H)
    rng = np.random.default_rng(0)
    for h in range(1, H + 1):
        rel, dst = rng.integers(0, R, 200), rng.integers(0, N, 200)
        with torch.no_grad():
            ref = scorer.step_logits(q.expand(200, -1), torch.full((200,), h),
                                     graph.relation[rel], graph.node[dst])
            ref_stop = scorer.stop_logits(q.expand(200, -1), torch.full((200,), h), graph.node[dst])
        np.testing.assert_allclose(bound.edge_logit(h, rel, dst), ref.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(bound.stop_score(h, dst), nonconformity(ref_stop).numpy(), rtol=1e-5, atol=1e-5)


def test_scores_nonnegative_and_finite():
    graph, scorer, q = _setup(1)
    bound = scorer.bind(q, scorer.encode_graph(graph), H)
    rel, dst = np.repeat(np.arange(R), N), np.tile(np.arange(N), R)
    for h in range(1, H + 1):
        s = bound.edge_score(h, None, rel, dst)
        assert s.shape == (R * N,) and np.isfinite(s).all() and (s >= 0).all()
        assert (bound.stop_score(h, np.arange(N)) >= 0).all()


def test_softplus_stable_for_extreme_logits():
    s = nonconformity(torch.tensor([-1000.0, 0.0, 1000.0]))
    assert torch.isfinite(s).all()
    assert abs(s[1].item() - np.log(2)) < 1e-6 and s[2].item() < 1e-6


def test_gradients_reach_both_heads():
    graph, scorer, q = _setup(2)
    h = torch.tensor([1, 2])
    loss = scorer.step_logits(q.expand(2, -1), h, graph.relation[:2], graph.node[:2]).sum() \
        + scorer.stop_logits(q.expand(2, -1), h, graph.node[:2]).sum()
    loss.backward()
    for name, p in scorer.named_parameters():
        assert p.grad is not None, name


def test_readable_relation():
    assert readable_relation("people.person.parents (inverse)") == "people person parents"
    assert readable_relation("drug_protein: target") == "drug protein: target"
