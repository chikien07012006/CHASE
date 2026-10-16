from __future__ import annotations

import torch
import torch.nn.functional as F

# Finite stand-in for "unreachable": smooth max/min stay finite and gradients never become NaN.
BIG = 1e4
UNREACHABLE = BIG / 2   # anything above this is treated as unreachable


# ------------------------------------------------------------------ stage 1

def _weighted_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE with pos_weight = #neg / #pos so both classes count equally."""
    if logits.numel() == 0:
        return logits.new_zeros(())
    labels = labels.float()
    n_pos = labels.sum()
    pos_weight = ((labels.numel() - n_pos) / n_pos.clamp_min(1.0)).clamp_min(1.0).detach()
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def bce_loss(step_logits, step_labels, stop_logits, stop_labels, beta: float = 1.0) -> torch.Tensor:
    """Stage 1: L1 = BCE(step) + beta * BCE(stop)."""
    return _weighted_bce(step_logits, step_labels) + beta * _weighted_bce(stop_logits, stop_labels)


# ------------------------------------------------------------------ smooth max / min

def smooth_max(x: torch.Tensor, T: float, dim: int = -1) -> torch.Tensor:
    """T * logsumexp(x / T); tends to max as T -> 0 (overestimates by at most T log n)."""
    return T * torch.logsumexp(x / T, dim=dim)


def smooth_min(x: torch.Tensor, T: float, dim: int = -1) -> torch.Tensor:
    return -smooth_max(-x, T, dim)


def _pair_smax(a: torch.Tensor, b: torch.Tensor, T: float) -> torch.Tensor:
    return T * torch.logaddexp(a / T, b / T)


def _segment_smin(val: torch.Tensor, dst: torch.Tensor, num_nodes: int, T: float) -> torch.Tensor:
    """Smooth min of val over the edges entering each node; BIG for nodes without incoming edges."""
    x = -val / T
    m = torch.full((num_nodes,), -float("inf"), dtype=val.dtype).scatter_reduce(0, dst, x.detach(), "amax")
    has_edge = torch.isfinite(m)
    m = torch.where(has_edge, m, torch.zeros_like(m))
    total = torch.zeros(num_nodes, dtype=val.dtype).index_add(0, dst, torch.exp(x - m[dst]))
    lse = torch.log(total.clamp_min(1e-30)) + m      # logsumexp; m is a constant shift, so detaching it is exact
    return torch.where(has_edge, -T * lse, torch.full_like(lse, BIG))


# ------------------------------------------------------------------ soft bottleneck DP

def soft_bottleneck_dp(num_nodes: int, src: torch.Tensor, dst: torch.Tensor, edge_s: torch.Tensor,
                       stop_s: torch.Tensor, topic: torch.Tensor, answerable: torch.Tensor,
                       H: int, lam: float = 1.0, T: float = 0.1) -> torch.Tensor:
    """Differentiable twin of chase.search.dp.bottleneck_dp: max -> smooth max, min -> smooth min.

    edge_s: [H+1, E] step scores of the directed edges (src -> dst). Set BIG to forbid an edge at a
            hop (e.g. type-infeasible states).
    stop_s: [H+1, N] stop scores. answerable: bool [N] (topic entities are removed here).
    Returns sigma [N]; values above UNREACHABLE mean the node is never returned.
    """
    topic = torch.as_tensor(topic, dtype=torch.long)
    b = torch.full((num_nodes,), BIG, dtype=edge_s.dtype)
    b = b.index_fill(0, topic, -BIG)
    answerable = answerable.clone()
    answerable[topic] = False
    cand = []
    for h in range(1, H + 1):
        val = _pair_smax(b[src], edge_s[h], T)
        b = _segment_smin(val, dst, num_nodes, T)
        c = _pair_smax(b, lam * stop_s[h], T)
        cand.append(torch.where(answerable, c, torch.full_like(c, BIG)))
    return smooth_min(torch.stack(cand), T, dim=0)


# ------------------------------------------------------------------ stage 2 (conformal training)

def alpha_eff(alpha: float, rho: float) -> float:
    """Miscoverage level among reachable questions that matches level alpha over all questions."""
    a = 1.0 - (1.0 - alpha) / rho
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha={alpha} is infeasible for reachability rho={rho} (alpha_eff={a:.3f})")
    return a


def soft_sigma_star(sigma: torch.Tensor, gold: torch.Tensor, T: float) -> torch.Tensor:
    """Smooth min of sigma over the gold answers of one question."""
    return smooth_min(sigma[gold], T, dim=0)


def is_reachable(sigma: torch.Tensor, gold: torch.Tensor) -> bool:
    return bool(gold.numel()) and bool((sigma[gold] < UNREACHABLE).any())


def soft_wrong_set_size(sigma: torch.Tensor, gold: torch.Tensor, tau: torch.Tensor, T_c: float) -> torch.Tensor:
    """Soft count of non-gold nodes returned at threshold tau: sum of sigmoid((tau - sigma_v) / T_c)."""
    wrong = torch.ones_like(sigma, dtype=torch.bool)
    wrong[gold] = False
    return torch.sigmoid((tau - sigma[wrong]) / T_c).sum()


def conftr_size_loss(sigmas: list[torch.Tensor], golds: list[torch.Tensor], half1: list[int], half2: list[int],
                     alpha_train: float, T: float = 0.1, T_c: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 2 size term (option A).

    half1: questions used to place the batch threshold; unreachable ones are skipped and alpha_train
           should be alpha_eff(alpha, rho_train). half2: questions whose wrong answers are penalised;
           unreachable ones are kept (every returned node is wrong there).
    Returns (L_size, tau_B).
    """
    stars = [soft_sigma_star(sigmas[i], golds[i], T) for i in half1 if is_reachable(sigmas[i], golds[i])]
    if not stars or not half2:
        zero = sigmas[0].new_zeros(())
        return zero, zero
    tau = torch.quantile(torch.stack(stars), 1.0 - alpha_train)
    sizes = torch.stack([soft_wrong_set_size(sigmas[i], golds[i], tau, T_c) for i in half2])
    return torch.log1p(sizes).mean(), tau


def stage2_loss(size_loss: torch.Tensor, bce: torch.Tensor, mu: float = 0.5) -> torch.Tensor:
    """L2 = L_size + mu * L1."""
    return size_loss + mu * bce
