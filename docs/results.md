# CHASE results: WebQSP and CWQ, 3 seeds each

All numbers use the default ordering of candidates by (gap, stop score) (`--tie-break stop`; see [theory.tex](../theory.tex), "Tie-breaking"). Mean ± std over seeds uses ddof = 1.

| Claim | Source |
|---|---|
| Main tables, coverage checks | `runs/{webqsp,cwq}_seed{0,1,2}/eval/` (`metrics.json`, `validity.json`, `results_table.md`) |
| Same α and equal coverage vs reported CPR / UaG | `runs/{webqsp,cwq}_compare/compare.md` |
| Same-scorer baselines, gap-vs-rank ablation | `runs/{webqsp,cwq}_seed{0,1,2}/baselines/baselines.json`, `runs/{webqsp,cwq}_baselines/baselines_compare.md` |
| Paired bootstrap | `runs/{webqsp,cwq,biohopr}_seed{0,1,2}/baselines/per_question.npz`, `runs/{webqsp,cwq,biohopr}_baselines/bootstrap.{md,json}` |
| Stage-2 ablation | `runs/{webqsp,cwq,biohopr}_seed{0,1,2}/eval_stage1/` (stage-1 checkpoint `stage1_best.pt`) vs `.../eval/` |
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
| 0.4 | 60 | 57.5 ± 1.8 | 1.75 ± 0.40 | 1.0 | 6.6 | 18 / 462 |
| 0.5–0.8 | 50–20 | 55.4 ± 2.3 (floor) | 1.00 ± 0.00 | 1.0 | 6.6 | 17 / 462 |

**CWQ**

| α | Target % | ECR % | APSS | Median size | ms / question | States (pruned / full) |
|---|---|---|---|---|---|---|
| 0.05, 0.1 | 95, 90 | abstain (τ̂ = ∞) | — | — | — | — |
| 0.2 | 80 | 79.4 ± 0.1 | 129.07 ± 1.83 | 99.0 | 7.1 | 114 / 328 |
| 0.3 | 70 | 69.2 ± 1.0 | 14.57 ± 1.00 | 8.3 | 6.8 | 35 / 328 |
| 0.4 | 60 | 60.2 ± 0.7 | 6.73 ± 0.54 | 3.3 | 6.9 | 24 / 328 |
| 0.5 | 50 | 49.8 ± 1.1 | 3.52 ± 0.41 | 2.0 | 6.7 | 18 / 328 |
| 0.6 | 40 | 39.8 ± 0.2 | 1.86 ± 0.21 | 1.0 | 6.9 | 16 / 328 |
| 0.7, 0.8 | 30, 20 | 32.9 ± 0.9 (floor) | 1.00 ± 0.00 | 1.0 | 6.9 | 15 / 328 |

- **Floor.** The top candidate is always returned, so coverage cannot fall below Hit@1: 55.4 ± 2.3% on WebQSP, 32.9 ± 0.9% on CWQ. Below the floor, τ̂ = (0, 0), the set is the top candidate alone, and CHASE over-covers (Remark "Hit@1 floor" in [theory.tex](../theory.tex)).
- **Empty sets.** 0% on WebQSP and 0.06% on CWQ, the questions with no reachable candidate at all.
- **Abstention on CWQ.** At α ≤ 0.1, fewer than k cal questions are reachable, so τ̂ = ∞. At α = 0.2 the margin is 19 questions; 23.7% of re-splits abstain.
- **Exact pruning.** For every test question and every α, the pruned search returned exactly the unpruned set (`pruned_matches_unpruned` = true in all 6 runs, and in the 3 BioHopR runs). It visits 3–27× fewer DP states.

## Coverage validity

| α | WebQSP official ECR (s0 / s1 / s2) | WebQSP re-split mean (5–95%) | CWQ official ECR (s0 / s1 / s2) | CWQ re-split mean (5–95%) |
|---|---|---|---|---|
| 0.1 | 89.2 / 89.7 / 87.3 | 90.2 (87.0–93.1) | abstain | abstain |
| 0.2 | 76.4 / 80.2 / 78.0 | 80.4 (76.1–84.4) | 79.4 / 79.6 / 79.3 | 79.6 (78.3–80.6) |
| 0.3 | 65.9 / 68.9 / 70.0 | 70.2 (65.2–75.0) | 69.5 / 70.0 / 68.1 | 70.0 (68.2–71.9) |
| 0.4 | 55.5 / 59.1 / 57.9 | 60.4 (55.9–65.3) | 60.3 / 60.9 / 59.4 | 60.0 (58.0–62.0) |
| 0.5 | floor | floor | 50.8 / 48.6 / 50.2 | 50.0 (48.1–51.9) |
| 0.6 | floor | floor | 39.8 / 39.6 / 39.9 | 40.0 (38.0–41.9) |

- **Re-splits:** 1,000 random re-partitions of the pooled cal and test questions, scorer fixed. They are exchangeable by construction.
- Above the floor, the re-split mean equals 1 − α to within 0.5 points on both datasets.
- **WebQSP's official split is on average 1–2 points below target** at every level above the floor. Every value is inside the re-split 5–95% band except seed 0 at α = 0.4 (55.5 vs 55.9), and inside the theory interval (Beta, 1–99%: 53.4–66.8 at α = 0.4). The cal set has only 283 questions, so one draw moves coverage by several points; the guarantee holds on average over cal draws.
- **CWQ** is on target, with the larger cal set (3,519).

## Comparison with reported CPR and UaG

**Same α.** CPR over-covers at high α: on WebQSP it reports 50.8% at a 20% target, on CWQ 42.8% at 30%. CHASE is on target above its floor. Set sizes at the same α therefore favour CHASE, so the fair comparison is at equal coverage.

**Equal coverage.** CHASE is calibrated at 1 − α′ = the other method's reported ECR. APSS is the mean over 1,000 re-splits, then over seeds. Ratio = CHASE APSS / reported APSS (< 1: CHASE smaller).

| Coverage % | WebQSP: CPR APSS | CHASE APSS | Ratio | | Coverage % | CWQ: CPR APSS | CHASE APSS | Ratio |
|---|---|---|---|---|---|---|---|---|
| 76.8 | 17.77 | 8.36 | **0.47** | | 76.6 | 23.98 | 40.89 | 1.71 |
| 67.3 | 10.10 | 4.63 | **0.46** | | 68.1 | 15.28 | 12.48 | **0.82** |
| 61.4 | 7.14 | 2.90 | **0.41** | | 58.8 | 11.40 | 5.87 | **0.51** |
| 58.3 | 5.45 | 2.03 | **0.37** | | 50.2 | 8.35 | 3.42 | **0.41** |
| 55.1 | 3.01 | 1.39 | 0.46 (floor) | | 42.8 | 6.08 | 2.15 | **0.35** |

Against UaG (reported), the ratio is 0.06–0.08 on WebQSP at 60–72% coverage and 0.01–0.02 on CWQ at 37–42%.

- CHASE's sets are 2.1–2.7× smaller than CPR's on WebQSP at 58–77% coverage, and up to 2.9× smaller on CWQ at 43–68%.
- **CHASE loses on CWQ at 76.6%.** This level needs about 96% of the reachable questions (ceiling ≈ 0.80), and the sets grow fast near the ceiling.
- "(floor)" rows: the reported coverage is below CHASE's floor, so CHASE covers more and the ratio is not at equal coverage.

Figures: [webqsp_equal_coverage.png](figures/webqsp_equal_coverage.png), [cwq_equal_coverage.png](figures/cwq_equal_coverage.png), same α: [webqsp_same_alpha.png](figures/webqsp_same_alpha.png), [cwq_same_alpha.png](figures/cwq_same_alpha.png). Per-seed figures (validity histogram, coverage vs target, set sizes, cost, training curves) are in `runs/*_seed*/eval/figures/`.

## Same-scorer baselines

Every method uses the same trained CHASE scorer, cal and test sets; only the way the set is built differs. Ratio = CHASE APSS / baseline APSS at the baseline's re-split ECR, mean over seeds.

| Baseline | WebQSP | CWQ |
|---|---|---|
| UaG-style per-hop (no guarantee) | **0.32–0.49** at 59–86% | **0.51–0.61** at 35–64% |
| UaG-style Bonferroni | **0.40–0.49** at 64–89% | **0.35** at 68% (only feasible level) |
| CPR-style B=128 | **0.42–0.84** at 60–91% | **0.65–0.85** at 40–60%; 1.03 at 70% |
| CPR-style B=32 | **0.58–0.89** at 60–85%; 1.05 at 80% | **0.76–0.91** at 40–50%; 1.05 at 60% |
| CPR-style B=8 | 0.99–1.47 at 60–76% | 1.04–1.17 at 40–50% |

**Reach:** CHASE and top-K score the whole 2-hop graph and reach a gold answer for 95.3% (WebQSP) and 80.1% (CWQ) of test questions. The beams reach 76.9 / 86.0 / 92.2% (WebQSP, B = 8 / 32 / 128) and 52.9 / 66.9 / 76.0% (CWQ), so beam methods must abstain at high targets.

- **In-search DP beats beam-then-filter** under the same scorer, against wide beams (B ≥ 32) at almost every level. Narrow beams (B = 8) give smaller sets at the levels they reach, because the beam drops low-ranked distractors, but they cannot reach high coverage.
- **CHASE beats both UaG-style variants**, by 1.6–3.1×.
- **UaG-style Bonferroni is almost unusable on CWQ:** with α split three ways and about 20% of cal questions unreachable, on the official split it abstains at every α in at least one seed, and every level has abstaining re-splits in some seed, so it has no point in the figure.

Figures: [webqsp_baselines.png](figures/webqsp_baselines.png), [cwq_baselines.png](figures/cwq_baselines.png).

### Paired bootstrap

Is the ratio larger than the sampling noise of the test questions? On the official test split, with cal and scorers fixed, the test questions are resampled with replacement 10,000 times, by cluster (the seed-question groups of the split: singletons on WebQSP, 344 groups on CWQ, 1,269 topic groups on BioHopR). Each resample is shared by both methods and all 3 seeds. On every resample, CHASE is re-calibrated to the smallest threshold whose ECR is ≥ the baseline's, and the ratio of APSS is averaged over seeds. 95% percentile CI; two-sided p for H0 ratio = 1, Holm-adjusted over the tested rows of each dataset (52 on WebQSP, 45 on CWQ, 92 on BioHopR). Rows below CHASE's Hit@1 floor (CHASE ECR > baseline ECR + 1 point) and levels where the baseline abstains in some seed are not tested. With 10,000 resamples the smallest attainable Holm p is about 0.01–0.02, so we report p < 0.05. Rows with identical sets at several α (the CPR rule saturates) are counted as separate tests, so the correction is conservative.

Rows with p < 0.05 (Holm), by direction; "n" = tested rows; ratios of the significant rows.

| Baseline | WebQSP | CWQ | BioHopR |
|---|---|---|---|
| UaG-style Bonferroni | CHASE smaller 5 / 5: **0.40–0.53** at 62–81% | abstains | abstains |
| UaG-style per-hop (no guarantee) | CHASE smaller 2 / 2: **0.27–0.51** at 57–71% | CHASE smaller 2 / 2: **0.49–0.60** at 46–62% | CHASE smaller 5 / 5: **0.20–0.48** at 89.7–93.8% |
| CPR-style B=128 | CHASE smaller 4 / 4: **0.37–0.74** at 59–88% | CHASE smaller 3 / 4: **0.64–0.85** at 39–60% | CHASE smaller 7 / 9: **0.15–0.78** at 89.0–91.9% |
| CPR-style B=128, CPR rule | CHASE smaller 5 / 9: **0.78–0.87** at 60–89% | CHASE smaller 1 / 9: **0.88** at 53% | CHASE smaller 5 / 17: **0.50–0.69** at 90.1–90.7% |
| CPR-style B=32 (both rules) | CHASE smaller 3 / 12 (0.52–0.80); baseline smaller 1 / 12 (1.13 at 75%) | CHASE smaller 5 / 12 (0.71–0.76) | CHASE smaller 1 / 23 (0.31) |
| CPR-style B=8 (both rules) | baseline smaller 7 / 11: 1.34–1.64 at 61–77% | none of 11 | none of 21 |
| Top-K (ablation) | CHASE smaller 1 / 9 (0.76 at 93%); baseline smaller 3 / 9: 1.67–2.12 at 66–79% | baseline smaller 1 / 7 (1.24 at 44%) | baseline smaller 10 / 17: 1.47–6.37 at 89.3–93.5% |

- **Significant against UaG-style and wide beams on all three datasets.** Every significant row against UaG-style and CPR-style B=128 favours CHASE.
- **Narrow beams are a trade-off, not a loss for them.** B = 8 is significantly smaller than CHASE on WebQSP at the levels it reaches and indistinguishable elsewhere; B = 32 is mostly indistinguishable. Their sets are small because the beam drops answers, which also caps their coverage (B = 8: 77% WebQSP, 53% CWQ, 91% BioHopR).
- **BioHopR rows near the floor** (baseline ECR 0.2–1 point below 89.0%) have the point estimate at the edge of the CI (e.g. 0.16 [0.16, 1.14]): resampling moves the matched CHASE threshold on and off the floor. They are not used as evidence.

## Ablations

**Gap vs rank threshold.** Conformal top-K keeps the scorer, the DP and the calibration, and replaces the gap by the rank (order by (σ, stop score, node id); calibrate the rank K̂ of the best gold answer). It is not a method of the literature; it isolates the choice of score.

| Ratio CHASE / top-K | WebQSP | CWQ |
|---|---|---|
| High coverage | **0.65 / 0.94** at 94 / 90% | **0.60** at 80%; **0.94** at 70% |
| Mid coverage | 1.60–2.16 at 66–81% | 1.08–1.22 at 45–61% |

The scorer's ranking is good, but the gap of the gold answer varies across questions; a fixed K̂ exploits the ranking better at mid coverage, the gap is smaller where high coverage is needed. Figures: [webqsp_ablation_rank.png](figures/webqsp_ablation_rank.png), [cwq_ablation_rank.png](figures/cwq_ablation_rank.png).

Under the paired bootstrap (official split, above), top-K's advantage is significant at mid coverage on WebQSP (66–79%) and BioHopR (89.3–93.5%), and on CWQ only at 44%. CHASE's advantage at high coverage is significant only on WebQSP at 93% (0.76); on CWQ at 79% (0.67) and BioHopR at 94–95% (0.83) it is not.

**Conformal training (stage 2) vs BCE only (stage 1).** The stage-1 checkpoint (`stage1_best.pt`, early-stopped) is evaluated with the same pipeline. Official split, mean over 3 seeds, ECR % / APSS:

| | α | Stage 1 (BCE) | Stage 2 (CHASE) | APSS change |
|---|---|---|---|---|
| WebQSP | 0.1 | 88.6 / 37.56 | 88.8 / 22.41 | −40% |
| | 0.2 | 77.7 / 12.37 | 78.2 / 9.08 | −27% |
| | 0.3 | 67.5 / 6.52 | 68.2 / 4.89 | −25% |
| CWQ | 0.2 | 79.2 / 162.05 | 79.4 / 129.07 | −20% |
| | 0.3 | 68.5 / 20.87 | 69.2 / 14.57 | −30% |
| | 0.4 | 59.0 / 9.36 | 60.2 / 6.73 | −28% |
| BioHopR | 0.05 | 93.5 / 111.93 | 93.3 / 45.75 | −59% |
| | 0.06 | 92.4 / 72.80 | 92.7 / 33.37 | −54% |
| | 0.07 | 91.2 / 51.03 | 91.7 / 25.80 | −49% |
| | 0.08 | 90.4 / 41.13 | 90.7 / 18.22 | −56% |
| Hit@1 | | 50.8 / 29.8 / 82.9% | 55.4 / 32.9 / 89.0% | WebQSP / CWQ / BioHopR |

Coverage is on target for both stages (the guarantee does not depend on training); stage 2 shrinks the sets at every level above the floor, most at the training α, and raises Hit@1, which lowers the floor.

**Tie-breaking.** With `--tie-break none` candidates are ordered by σ alone, and every node tied with the top candidate is returned. Wherever τ̂_g > 0 the sets are identical: a test question rarely has a node whose gap equals τ̂_g exactly. They differ only where τ̂_g = 0, at the floor and at levels just above it in some seeds (WebQSP α = 0.4: APSS 1.87 → 1.75). At the floor, APSS (official split, mean over seeds):

| | WebQSP (α ≥ 0.5) | CWQ (α ≥ 0.7) | BioHopR (α ≥ 0.105) |
|---|---|---|---|
| σ only | 1.60, ECR 56.7% | 1.48, ECR 36.5% | 4.18, ECR 89.0% |
| (gap, stop score) | **1.00**, ECR 55.4% | **1.00**, ECR 32.9% | **0.98**, ECR 89.0% |

## Limitations seen in these runs

- **Floor at Hit@1** (55.4% WebQSP, 32.9% CWQ): CHASE cannot target lower coverage.
- **Large sets near the reachability ceiling** (CWQ at 77–80%).
- **A rank threshold wins at mid coverage** under the same scorer (ablation above); the gap's advantage at high coverage is significant only on WebQSP.
- **Narrow beams (B = 8) are not beaten** at the levels they reach: smaller sets on WebQSP, indistinguishable on CWQ and BioHopR.
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
