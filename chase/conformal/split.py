from __future__ import annotations

import math

import numpy as np

INF = np.inf


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def conformal_quantile(scores, alpha: float) -> float:
    """Split-conformal threshold: the k-th smallest score, k = ceil((n+1)(1-alpha)).

    Returns INF when k > n or the k-th score is INF; the level is then infeasible
    (too many calibration questions cannot reach an answer).
    """
    _check_alpha(alpha)
    s = np.sort(np.asarray(scores, dtype=float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return INF if k > len(s) else float(s[k - 1])


def covered(sigma_star, tau: float) -> np.ndarray:
    """Per question: does the answer set at threshold tau contain a gold answer?"""
    s = np.asarray(sigma_star, dtype=float)
    return np.isfinite(s) & (s <= tau)


def miss_rate(answer_sigmas: list[np.ndarray], tau: float) -> np.ndarray:
    """Per question: share of gold answers not returned at threshold tau (1 - recall)."""
    return np.array([np.mean(~(np.isfinite(a) & (a <= tau))) for a in answer_sigmas])


def crc_threshold(answer_sigmas: list[np.ndarray], alpha: float) -> float:
    """Conformal risk control for multi-answer questions.

    answer_sigmas[i] holds sigma_a for every gold answer of calibration question i
    (INF for answers that cannot be reached). Loss L_i(tau) = share of answers with
    sigma_a > tau, non-increasing in tau. Returns the smallest tau with
        n/(n+1) * mean_i L_i(tau) + 1/(n+1) <= alpha,
    which gives E[L_test(tau)] <= alpha. Returns INF if no finite tau qualifies.
    """
    _check_alpha(alpha)
    n = len(answer_sigmas)
    if n == 0:
        return INF
    if any(len(a) == 0 for a in answer_sigmas):
        raise ValueError("every calibration question needs at least one gold answer")
    budget = ((n + 1) * alpha - 1) / n            # condition rewritten as mean loss <= budget
    if budget < 0:
        return INF
    finite = [np.sort(a[np.isfinite(a)]) for a in answer_sigmas]
    candidates = np.unique(np.concatenate(finite)) if any(len(f) for f in finite) else np.empty(0)
    if candidates.size == 0:
        return INF
    found = np.zeros(candidates.size)
    for f, a in zip(finite, answer_sigmas):
        found += np.searchsorted(f, candidates, side="right") / len(a)
    mean_loss = 1.0 - found / n                    # mean loss at each candidate, non-increasing
    ok = np.flatnonzero(mean_loss <= budget + 1e-12)
    return float(candidates[ok[0]]) if ok.size else INF
