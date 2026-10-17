import numpy as np
import pandas as pd
import torch

from chase.kg.graph import KG, build_csr
from chase.model.features import GraphFeatures
from chase.train.sampler import Sampler, build_labels, sample_question

H = 2


def _kg():
    #  0 -1-> 1 -2-> 3 (answer)      0 -5-> 4      0 -6-> 5      1 -7-> 2      node 1 is a CVT
    edges = [(0, 1, 1), (1, 2, 3), (0, 5, 4), (0, 6, 5), (1, 7, 2)]
    u = np.array([e[0] for e in edges]); r = np.array([e[1] for e in edges]); v = np.array([e[2] for e in edges])
    indptr, indices, rel = build_csr(np.concatenate([u, v]), np.concatenate([v, u]), np.concatenate([r, r + 10]), 6)
    node_type = np.zeros(6, np.int8)
    node_type[1] = 1
    return KG(indptr=indptr, indices=indices, rel=rel, node_type=node_type, type_names=["entity", "cvt"],
              relation_names=[], nodes=pd.DataFrame({"name": [str(i) for i in range(6)]}))


GOLD = pd.DataFrame({"q_index": [7], "nodes": [[0, 1, 3]], "rels": [[1, 2]]})


def test_labels_from_gold_paths():
    lab = build_labels(GOLD, indices=[7, 8], H=H)
    assert lab[7].pos_steps == {(1, 1, 1), (2, 2, 3)}
    assert lab[7].gold_states == {0: {0}, 1: {1}}
    assert lab[7].pos_stops == {(2, 3)} and lab[7].mid_stops == {(1, 1)}
    assert lab[7].reachable and not lab[8].reachable


def test_samples_have_correct_positives_and_disjoint_negatives():
    kg, lab = _kg(), build_labels(GOLD, [7], H)[7]
    s = sample_question(kg, lab, H, np.random.default_rng(0), k_hard=32, k_easy=8)
    pos = {tuple(x) for x, y in zip(s.steps.tolist(), s.step_y) if y == 1}
    neg = {tuple(x) for x, y in zip(s.steps.tolist(), s.step_y) if y == 0}
    assert pos == lab.pos_steps and not pos & neg
    # every hard negative leaving the topic (hop 1) is present: edges 0->4 (5) and 0->5 (6)
    assert {(1, 5, 4), (1, 6, 5)} <= neg
    stop_pos = {tuple(x) for x, y in zip(s.stops.tolist(), s.stop_y) if y == 1}
    stop_neg = {tuple(x) for x, y in zip(s.stops.tolist(), s.stop_y) if y == 0}
    assert stop_pos == {(2, 3)} and (1, 1) in stop_neg and (1, 4) in stop_neg and not stop_pos & stop_neg


def test_hard_negatives_are_capped_and_sampling_is_deterministic():
    kg, lab = _kg(), build_labels(GOLD, [7], H)[7]
    a = sample_question(kg, lab, H, np.random.default_rng(1), k_hard=1, k_easy=0)
    b = sample_question(kg, lab, H, np.random.default_rng(1), k_hard=1, k_easy=0)
    np.testing.assert_array_equal(a.steps, b.steps)
    hop1_neg = [x for x, y in zip(a.steps.tolist(), a.step_y) if y == 0 and x[0] == 1]
    assert len(hop1_neg) == 1


def test_unreachable_question_has_no_bce_samples():
    s = sample_question(_kg(), build_labels(GOLD, [8], H)[8], H, np.random.default_rng(0))
    assert len(s.step_y) == 0 and len(s.stop_y) == 0


class _Store:
    def __init__(self):
        self.kg = _kg()
        self.questions = pd.DataFrame({"topic_nodes": [[0]] * 10, "answer_nodes": [[3]] * 10})

    def get(self, i):
        return self.kg


class _Features:
    def __init__(self):
        torch.manual_seed(0)
        self.node, self.rel = torch.randn(6, 7), torch.randn(20, 5)

    def graph(self, i, kg):
        return GraphFeatures(node=self.node, relation=self.rel)

    def question(self, i):
        return torch.full((4,), float(i))


def test_sampler_batches_stage1_and_stage2():
    gold = pd.DataFrame({"q_index": [7, 9], "nodes": [[0, 1, 3]] * 2, "rels": [[1, 2]] * 2})
    sampler = Sampler(_Store(), _Features(), gold, indices=[7, 8, 9], H=H, k_easy=4)
    assert sampler.reachable_indices == [7, 9]

    stage1 = list(sampler.batches(epoch=0, batch_size=2))
    assert len(stage1) == 1 and stage1[0].graphs is None          # question 8 (no gold) is skipped
    b = stage1[0]
    n = len(b.step.label)
    assert b.step.q.shape == (n, 4) and b.step.rel_feat.shape == (n, 5) and b.step.node_feat.shape == (n, 7)
    assert b.stop.node_feat.shape == (len(b.stop.label), 7) and b.step.label.sum() == 4   # 2 questions x 2 gold steps

    stage2 = list(sampler.batches(epoch=0, batch_size=3, with_graph=True))
    g = stage2[0].graphs
    assert len(g) == 3 and sum(x.reachable for x in g) == 2       # unreachable question kept for B2
    item = g[0]
    assert item.src.shape == item.dst.shape == item.rel_pos.shape
    assert torch.equal(item.rel_ids[item.rel_pos], torch.from_numpy(_kg().rel.astype(np.int64)))
    assert not item.answerable[1]                                  # CVT can never be an answer

    again = list(sampler.batches(epoch=0, batch_size=2))[0]
    other = list(sampler.batches(epoch=1, batch_size=2))[0]
    assert torch.equal(again.step.h, b.step.h)                     # same epoch -> same samples
    assert len(other.step.label) > 0
