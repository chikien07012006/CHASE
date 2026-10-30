from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from chase.kg.graph import KG
from chase.search.dp import INF, ScoreFn, bottleneck_dp


@dataclass
class CalibrationProblem:
    """One labelled question, ready for the DP."""
    kg: KG
    topic: Sequence[int]
    answers: Sequence[int]                 # gold answers present in the graph
    scorer: ScoreFn
    answer_mask: np.ndarray | None = None
    feasible: Callable[[int, np.ndarray], np.ndarray] | None = None
    num_absent: int = 0                    # gold answers missing from the graph (count as never found)


def answer_sigmas(problem: CalibrationProblem, H: int, lam: float = 1.0) -> np.ndarray:
    """Gap score g_a = sigma_a - sigma_min of every gold answer (INF if unreachable or absent from the graph),
    via an unpruned DP."""
    res = bottleneck_dp(problem.kg, problem.topic, problem.scorer, H, lam=lam,
                        answer_mask=problem.answer_mask, feasible=problem.feasible)
    found = res.sigma[np.asarray(problem.answers, dtype=np.int64)] - res.sigma_min
    return np.concatenate([found, np.full(problem.num_absent, INF)])


@dataclass
class CalibrationScores:
    sigma_star: np.ndarray                 # [n] per question: g*, smallest gap threshold that returns a gold answer
    answer_sigma: list[np.ndarray]         # per question: g_a of every gold answer (for CRC)

    def save(self, path: str | Path) -> None:
        offsets = np.cumsum([0] + [len(a) for a in self.answer_sigma])
        flat = np.concatenate(self.answer_sigma) if self.answer_sigma else np.empty(0)
        np.savez(path, sigma_star=self.sigma_star, answer_sigma=flat, answer_offset=offsets)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationScores":
        z = np.load(path)
        off, flat = z["answer_offset"], z["answer_sigma"]
        return cls(sigma_star=z["sigma_star"], answer_sigma=[flat[off[i]:off[i + 1]] for i in range(len(off) - 1)])


def calibration_scores(problems: Iterable[CalibrationProblem], H: int, lam: float = 1.0) -> CalibrationScores:
    """Run the unpruned DP on every calibration question and collect its scores."""
    per_answer = [answer_sigmas(p, H, lam) for p in problems]
    sigma_star = np.array([a.min() if a.size else INF for a in per_answer])
    return CalibrationScores(sigma_star=sigma_star, answer_sigma=per_answer)


def freebase_problems(store, features, scorer, indices: Iterable[int], H: int) -> Iterator[CalibrationProblem]:
    """CalibrationProblems for questions of a Freebase SubgraphStore, scored by a StateScorer.

    store: chase.kg.subgraphs.SubgraphStore; features: chase.model.features.SubgraphFeatureStore.
    CVT nodes are never answers.
    """
    for i in indices:
        row = store.questions.iloc[i]
        kg = store.get(i)
        bound = scorer.bind(features.question(i), scorer.encode_graph(features.graph(i, kg)), H)
        answers = list(row["answer_nodes"])
        yield CalibrationProblem(kg=kg, topic=list(row["topic_nodes"]), answers=answers, scorer=bound,
                                 answer_mask=kg.node_type == 0,
                                 num_absent=max(len(set(row["a_entity"])) - len(answers), 0))
