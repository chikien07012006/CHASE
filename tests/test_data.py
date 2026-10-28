import numpy as np
import pandas as pd
import pytest

from chase.data.gold_paths import enumerate_paths, select_gold
from chase.data.splits import load_splits, make_freebase_splits, save_splits
from chase.kg.graph import KG, build_csr


# ------------------------------------------------------------------ splits

def _fake_store(tmp_path, sizes):
    for store, n in sizes.items():
        (tmp_path / store).mkdir()
        pd.DataFrame({"qid": [f"{store}-{i}" for i in range(n)]}).to_parquet(tmp_path / store / "questions.parquet")


def test_splits_are_disjoint_and_complete(tmp_path):
    _fake_store(tmp_path, {"train": 2826, "validation": 246, "test": 1628})
    s = make_freebase_splits(tmp_path, cal_fraction=0.1, seed=0)
    train, cal = set(s["train"]["indices"]), set(s["cal"]["indices"])
    assert len(cal) == 283 and len(train) == 2543
    assert not train & cal and train | cal == set(range(2826))
    assert len(s["val"]["indices"]) == 246 and s["val"]["store"] == "validation"
    assert len(s["test"]["indices"]) == 1628 and s["test"]["store"] == "test"


def test_splits_are_deterministic_and_roundtrip(tmp_path):
    _fake_store(tmp_path, {"train": 500, "validation": 10, "test": 10})
    a, b = make_freebase_splits(tmp_path, seed=3), make_freebase_splits(tmp_path, seed=3)
    assert a == b and make_freebase_splits(tmp_path, seed=4)["cal"] != a["cal"]
    save_splits(tmp_path, a)
    assert load_splits(tmp_path) == a


def test_cal_never_shares_a_seed_question_with_train(tmp_path):
    rng = np.random.default_rng(0)
    qids = [f"WebQTrn-{g}_{j}" for g in range(400) for j in range(rng.integers(1, 15))]
    for store, q in {"train": qids, "validation": ["v"], "test": ["t"]}.items():
        (tmp_path / store).mkdir()
        pd.DataFrame({"qid": q}).to_parquet(tmp_path / store / "questions.parquet")
    s = make_freebase_splits(tmp_path, cal_fraction=0.1, seed=0)
    prefix = lambda idx: {qids[i].split("_")[0] for i in idx}
    assert not prefix(s["train"]["indices"]) & prefix(s["cal"]["indices"])
    assert sorted(s["train"]["indices"] + s["cal"]["indices"]) == list(range(len(qids)))
    n_cal = len(s["cal"]["indices"])
    assert round(0.1 * len(qids)) <= n_cal < round(0.1 * len(qids)) + 15   # overshoot < one group


def test_singleton_groups_give_the_per_question_draw(tmp_path):
    _fake_store(tmp_path, {"train": 2826, "validation": 3, "test": 3})
    perm = np.random.default_rng(0).permutation(2826)
    assert make_freebase_splits(tmp_path, seed=0)["cal"]["indices"] == sorted(perm[:283].tolist())


# ------------------------------------------------------------------ gold paths

def _kg(edges, n):
    """Undirected test graph: each (u, r, v) is stored both ways with relation ids r (forward) and r + 100."""
    u = np.array([e[0] for e in edges]); r = np.array([e[1] for e in edges]); v = np.array([e[2] for e in edges])
    indptr, indices, rel = build_csr(np.concatenate([u, v]), np.concatenate([v, u]), np.concatenate([r, r + 100]), n)
    return KG(indptr=indptr, indices=indices, rel=rel, node_type=np.zeros(n, np.int8), type_names=["t"],
              relation_names=[], nodes=pd.DataFrame({"name": [str(i) for i in range(n)]}))


#   0 -r1-> 1 -r2-> 3     pattern (1, 2)
#   0 -r3-> 2 -r4-> 3     pattern (3, 4)
#   0 -r5-> 3             pattern (5,)
#   0 -r6-> 4 -r7-> 5     dead end (5 is not an answer)
EDGES = [(0, 1, 1), (1, 2, 3), (0, 3, 2), (2, 4, 3), (0, 5, 3), (0, 6, 4), (4, 7, 5)]


def test_enumerate_finds_exactly_the_simple_paths_to_answers():
    kg = _kg(EDGES, 6)
    got = set(enumerate_paths(kg, topic=[0], answers=[3], H=2))
    assert got == {((0, 1, 3), (1, 2)), ((0, 2, 3), (3, 4)), ((0, 3), (5,))}


def test_enumerate_respects_H_and_simplicity():
    kg = _kg(EDGES, 6)
    assert set(enumerate_paths(kg, [0], [3], H=1)) == {((0, 3), (5,))}
    for nodes, _ in enumerate_paths(kg, [0], [3], H=3):
        assert len(set(nodes)) == len(nodes)              # no repeated node
    assert enumerate_paths(kg, [0], [], H=2) == [] and enumerate_paths(kg, [], [3], H=2) == []
    assert enumerate_paths(kg, [0], [0], H=2) == []       # topic itself is never an answer


def test_select_gold_keeps_patterns_close_to_the_best():
    rel_emb = np.zeros((200, 3), np.float32)
    q = np.array([1.0, 0.0, 0.0], np.float32)
    rel_emb[1] = rel_emb[2] = [1.0, 0, 0]                 # pattern (1, 2): sim 1.0
    rel_emb[3] = rel_emb[4] = [0.95, np.sqrt(1 - 0.95 ** 2), 0]  # pattern (3, 4): sim 0.95
    rel_emb[5] = [0.2, np.sqrt(1 - 0.04), 0]              # pattern (5,): sim 0.2 -> coincidental
    paths = [((0, 1, 3), (1, 2)), ((0, 2, 3), (3, 4)), ((0, 3), (5,))]
    kept = select_gold(paths, q, rel_emb, delta=0.1)
    assert [r for _, r, _ in kept] == [(1, 2), (3, 4)]
    assert [r for _, r, _ in select_gold(paths, q, rel_emb, delta=0.01)] == [(1, 2)]
    assert len(select_gold(paths * 50, q, rel_emb, delta=1.0, max_paths=64)) == 64
    assert select_gold([], q, rel_emb) == []


def test_cal_from_validation_uses_all_of_validation_and_holds_out_train_groups_for_early_stopping(tmp_path):
    rng = np.random.default_rng(1)
    train_qids = [f"WebQTrn-{g}_{j}" for g in range(300) for j in range(rng.integers(1, 15))]
    for store, q in {"train": train_qids, "validation": ["v"] * 40, "test": ["t"] * 7}.items():
        (tmp_path / store).mkdir()
        pd.DataFrame({"qid": q}).to_parquet(tmp_path / store / "questions.parquet")
    s = make_freebase_splits(tmp_path, cal_fraction=0.1, seed=0, cal_from="validation")
    assert s["cal"] == {"store": "validation", "indices": list(range(40))}
    assert s["train"]["store"] == s["val"]["store"] == "train"
    # same train / held-out partition as the CPR protocol, with the held-out part used for early stopping
    assert s["val"]["indices"] == make_freebase_splits(tmp_path, cal_fraction=0.1, seed=0)["cal"]["indices"]
    prefix = lambda idx: {train_qids[i].split("_")[0] for i in idx}
    assert not prefix(s["train"]["indices"]) & prefix(s["val"]["indices"])
    assert len(s["test"]["indices"]) == 7
    with pytest.raises(ValueError):
        make_freebase_splits(tmp_path, cal_from="test")
