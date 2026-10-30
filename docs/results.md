# CHASE results: WebQSP and CWQ, 3 seeds each

All numbers are produced at commit `debca94`. Mean ± std over seeds uses ddof = 1.

| Claim | Source |
|---|---|
| Main tables, coverage checks | `runs/{webqsp,cwq}_seed{0,1,2}/eval/` (`metrics.json`, `validity.json`, `results_table.md`) |
| Same α and equal coverage vs reported CPR / UaG | `runs/{webqsp,cwq}_compare/compare.md` |
| Same-scorer baselines | `runs/{webqsp,cwq}_seed{0,1,2}/baselines/baselines.json`, `runs/{webqsp,cwq}_baselines/baselines_compare.md` |
| Training | `runs/{webqsp,cwq}_seed{0,1,2}/history.json` (config included), `runs/{webqsp,cwq}_seed{0,1,2}.train.log` |

"CPR (reported)" and "UaG (reported)" are the numbers in the CPR paper (arXiv 2605.08077). They use their own scorer, retrieval and split. "CPR-style" and "UaG-style" are our re-implementations on the CHASE scorer (see [baselines.md](baselines.md)).

## Setup

| | WebQSP | CWQ |
|---|---|---|
| Graph | RoG-WebQSP, one Freebase subgraph per question | RoG-CWQ, same |
| H | 2 | 2 |
| Cal / test | 283 / 1,628 (CPR's protocol) | 3,519 (official validation) / 3,531 (official test) |
| 2-hop reachability, test | 0.953 | 0.801 |
| Training α | 0.1 | 0.3 (see [theory.tex](../theory.tex), §"Which levels are attainable") |
| Training | stage 1 ≤ 50 epochs, stage 2 ≤ 20 epochs, early-stopping patience 8, lr 1e-3 / 3e-4 | same |
| Seeds | 0, 1, 2. They change training only; cal and test are fixed. | same |

**Stage 2** (conformal training on the gap) cut validation APSS at the training α by 57 / 37 / 45% on WebQSP and 34 / 43 / 37% on CWQ, at the same validation ECR.

## Main tables (official split)

ECR = share of test questions whose set contains a gold answer. APSS = mean set size. Hit@1 = share of questions whose top candidate is correct. "States" = DP states visited by the pruned search / by the unpruned search.

**WebQSP**

| α | Target % | ECR % | APSS | Median size | ms / question | States (pruned / full) |
|---|---|---|---|---|---|---|
| 0.05 | 95 | 93.3 ± 0.1 | 72.34 ± 1.67 | 45.3 | 6.7 | 86 / 462 |
| 0.1 | 90 | 88.8 ± 1.3 | 22.41 ± 5.76 | 10.3 | 6.7 | 45 / 462 |
| 0.2 | 80 | 78.2 ± 1.9 | 9.08 ± 1.19 | 3.3 | 6.7 | 27 / 462 |
| 0.3 | 70 | 68.2 ± 2.1 | 4.89 ± 0.22 | 2.0 | 6.7 | 21 / 462 |
| 0.4 | 60 | 57.8 ± 1.9 | 1.87 ± 0.18 | 1.0 | 6.7 | 18 / 462 |
| 0.5–0.8 | 50–20 | 56.7 ± 2.5 (floor) | 1.60 ± 0.06 | 1.0 | 6.6 | 17 / 462 |

**CWQ**

| α | Target % | ECR % | APSS | Median size | ms / question | States (pruned / full) |
|---|---|---|---|---|---|---|
| 0.05, 0.1 | 95, 90 | abstain (τ̂ = ∞) | — | — | — | — |
| 0.2 | 80 | 79.4 ± 0.1 | 129.07 ± 1.83 | 99.0 | 7.1 | 114 / 328 |
| 0.3 | 70 | 69.2 ± 1.0 | 14.57 ± 1.00 | 8.3 | 6.8 | 35 / 328 |
| 0.4 | 60 | 60.2 ± 0.7 | 6.73 ± 0.54 | 3.3 | 6.9 | 24 / 328 |
| 0.5 | 50 | 49.8 ± 1.1 | 3.52 ± 0.41 | 2.0 | 6.7 | 18 / 328 |
| 0.6 | 40 | 39.8 ± 0.2 | 1.86 ± 0.21 | 1.0 | 6.9 | 16 / 328 |
| 0.7, 0.8 | 30, 20 | 36.5 ± 1.4 (floor) | 1.48 ± 0.22 | 1.0 | 6.9 | 15 / 328 |

- **Floor.** The most confident candidate always has gap 0, so coverage cannot fall below the share of questions with a gold answer among the top-tied candidates: 56.7% on WebQSP, 36.5% on CWQ. Hit@1 is 55.2 ± 2.4% and 32.5 ± 1.3%; the floor is higher because several candidates can tie at σ_min. Below the floor, τ̂ = 0 and CHASE over-covers (Remark "Hit@1 floor" in [theory.tex](../theory.tex)).
- **Empty sets.** 0% on WebQSP and 0.06% on CWQ, the questions with no reachable candidate at all.
- **Abstention on CWQ.** At α ≤ 0.1, fewer than k cal questions are reachable, so τ̂ = ∞. At α = 0.2 the margin is 19 questions; 23.7% of re-splits abstain.
- **Exact pruning.** For every test question and every α, the pruned search returned exactly the unpruned set (`pruned_matches_unpruned` = true in all 6 runs). It visits 3–27× fewer DP states.

## Coverage validity

| α | WebQSP official ECR (s0 / s1 / s2) | WebQSP re-split mean (5–95%) | CWQ official ECR (s0 / s1 / s2) | CWQ re-split mean (5–95%) |
|---|---|---|---|---|
| 0.1 | 89.2 / 89.7 / 87.3 | 90.2 (87.0–93.1) | abstain | abstain |
| 0.2 | 76.4 / 80.2 / 78.0 | 80.4 (76.1–84.4) | 79.4 / 79.6 / 79.3 | 79.6 (78.3–80.6) |
| 0.3 | 65.9 / 68.9 / 70.0 | 70.2 (65.2–75.0) | 69.5 / 70.0 / 68.1 | 70.0 (68.2–71.9) |
| 0.4 | 55.5 / 59.1 / 58.7 | 60.5 (56.8–65.3) | 60.3 / 60.9 / 59.4 | 60.0 (58.0–62.0) |
| 0.5 | floor | floor | 50.8 / 48.6 / 50.2 | 50.0 (48.1–51.9) |
| 0.6 | floor | floor | 39.8 / 39.6 / 39.9 | 40.0 (38.0–41.9) |

- **Re-splits:** 1,000 random re-partitions of the pooled cal and test questions, scorer fixed. They are exchangeable by construction.
- Above the floor, the re-split mean equals 1 − α to within 0.5 points on both datasets.
- **WebQSP's official split is on average 1–2 points below target** at every level above the floor. Every value is inside the re-split 5–95% band except seed 0 at α = 0.4 (55.5 vs 56.8), and inside the theory interval (Beta, 1–99%: 53.4–66.8 at α = 0.4). The cal set has only 283 questions, so one draw moves coverage by several points; the guarantee holds on average over cal draws.
- **CWQ** is on target, with the larger cal set (3,519).

## Comparison with reported CPR and UaG

**Same α.** CPR over-covers at high α: on WebQSP it reports 50.8% at a 20% target, on CWQ 42.8% at 30%. CHASE is on target above its floor. Set sizes at the same α therefore favour CHASE, so the fair comparison is at equal coverage.

**Equal coverage.** CHASE is calibrated at 1 − α′ = the other method's reported ECR. APSS is the mean over 1,000 re-splits, then over seeds. Ratio = CHASE APSS / reported APSS (< 1: CHASE smaller).

| Coverage % | WebQSP: CPR APSS | CHASE APSS | Ratio | | Coverage % | CWQ: CPR APSS | CHASE APSS | Ratio |
|---|---|---|---|---|---|---|---|---|
| 76.8 | 17.77 | 8.36 | **0.47** | | 76.6 | 23.98 | 40.89 | 1.71 |
| 67.3 | 10.10 | 4.63 | **0.46** | | 68.1 | 15.28 | 12.48 | **0.82** |
| 61.4 | 7.14 | 2.93 | **0.41** | | 58.8 | 11.40 | 5.87 | **0.51** |
| 58.3 | 5.45 | 2.20 | **0.40** | | 50.2 | 8.35 | 3.42 | **0.41** |
| 55.1 | 3.01 | 1.79 | 0.59 (floor) | | 42.8 | 6.08 | 2.15 | **0.35** |

Against UaG (reported), the ratio is 0.06–0.08 on WebQSP at 60–72% coverage and 0.02 on CWQ at 37–42%.

- CHASE's sets are 2.1–2.5× smaller than CPR's on WebQSP at 58–77% coverage, and up to 2.9× smaller on CWQ at 43–68%.
- **CHASE loses on CWQ at 76.6%.** This level needs about 96% of the reachable questions (ceiling ≈ 0.80), and the sets grow fast near the ceiling.
- "(floor)" rows: the reported coverage is below CHASE's floor, so CHASE covers more and the ratio is not at equal coverage.

Figures: [webqsp_equal_coverage.png](figures/webqsp_equal_coverage.png), [cwq_equal_coverage.png](figures/cwq_equal_coverage.png), same α: [webqsp_same_alpha.png](figures/webqsp_same_alpha.png), [cwq_same_alpha.png](figures/cwq_same_alpha.png). Per-seed figures (validity histogram, coverage vs target, set sizes, cost, training curves) are in `runs/*_seed*/eval/figures/`.

## Same-scorer baselines

Every method uses the same trained CHASE scorer, cal and test sets; only the way the set is built differs. Ratio = CHASE APSS / baseline APSS at the baseline's re-split ECR, mean over seeds.

| Baseline | WebQSP | CWQ |
|---|---|---|
| UaG-style per-hop (no guarantee) | **0.34–0.49** at 59–86% | **0.51–0.61** at 48–64% |
| UaG-style Bonferroni | **0.40–0.49** at 64–89% | **0.35** at 68% (only feasible level) |
| CPR-style B=128 | **0.43–0.84** at 60–91% | **0.65–0.85** at 40–60%; 1.03 at 70% |
| CPR-style B=32 | **0.60–0.89** at 60–85%; 1.05 at 80% | **0.76–0.91** at 40–50%; 1.05 at 60% |
| CPR-style B=8 | 1.02–1.47 at 60–76% | 1.04–1.17 at 40–50% |
| Conformal top-K | **0.65 / 0.94** at 94 / 90%; **1.62–2.14 at 66–81%** | **0.60** at 80%; 0.93 at 70%; 1.06–1.22 at 45–61% |

**Reach:** CHASE and top-K score the whole 2-hop graph and reach a gold answer for 95.3% (WebQSP) and 80.1% (CWQ) of test questions. The beams reach 76.9 / 86.0 / 92.2% (WebQSP, B = 8 / 32 / 128) and 52.9 / 66.9 / 76.0% (CWQ), so beam methods must abstain at high targets.

- **In-search DP beats beam-then-filter** under the same scorer, against wide beams (B ≥ 32) at almost every level. Narrow beams (B = 8) give smaller sets at the levels they reach, because the beam drops low-ranked distractors, but they cannot reach high coverage.
- **CHASE beats both UaG-style variants**, by 1.6–2.9×.
- **Conformal top-K is smaller at mid coverage**, by up to 2.1× on WebQSP. CHASE is smaller at high coverage. The scorer's ranking is good, but the gap is not fully comparable across questions; a fixed K̂ exploits the ranking better at mid coverage.
- **UaG-style Bonferroni is almost unusable on CWQ:** with α split three ways and about 20% of cal questions unreachable, on the official split it abstains at every α in at least one seed, and every level has abstaining re-splits in some seed, so it has no point in the figure.

Figures: [webqsp_baselines.png](figures/webqsp_baselines.png), [cwq_baselines.png](figures/cwq_baselines.png).

## Limitations seen in these runs

- **Floor at Hit@1** (56.7% WebQSP, 36.5% CWQ): CHASE cannot target lower coverage.
- **Large sets near the reachability ceiling** (CWQ at 77–80%).
- **Conformal top-K wins at mid coverage** under the same scorer.
- **Stage 2 may not have converged:** the best checkpoint is at epoch 14–19 of 20 in 4 of 6 runs.
- **WebQSP's small cal set** (283) makes the official-split coverage noisy, by ±2 points.

## Why CWQ's cal set is the official validation set

The first CWQ runs followed CPR: they drew cal as 10% of RoG train, by seed group. On the official test split, ECR was then 4–5 points **below** target at α = 0.4–0.7 for all seeds. This was outside the theory interval, at percentiles 0–1.5 of the re-splits.

**Cause.** Groups do not separate *different* WebQSP questions about the same entity with the same answers. Call a question **seen** if some training question has the same topic entities and answer set.

| | Seen share | 2-hop reachability |
|---|---|---|
| Cal drawn from train (old) | 36% | 0.787 |
| Official validation (current cal) | 23% | 0.806 |
| Test | 27% | 0.801 |

Reachable seen questions are covered far more often than unseen ones, so the old cal was easier than test and τ̂ came out too small. The official validation set is built like the test set. Using it as cal needs no test information, and the coverage table above is on target.

**Split:** `python -m chase.data.splits --data data/processed/cwq --cal-from validation`. Cal = the whole official validation set (3,519; 345 groups). Val for early stopping = 10% of RoG train, drawn by seed-question group (2,768). Train = the rest of RoG train (24,871).

**Caveat:** questions within a group are dependent, so the theory interval, which treats them as independent, is somewhat too narrow.
