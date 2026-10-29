# CWQ results: 3 seeds

Numbers come from `runs/cwq_seed{0,1,2}/eval/` and `runs/cwq_compare/compare.md`. Mean ± std over seeds uses ddof = 1.

## Setup and split

| Item | Value |
|---|---|
| Graph | RoG-CWQ, one Freebase subgraph per question, H = 2 |
| Train | RoG train minus the early-stopping groups (24,871) |
| Val (early stopping only) | 10% of RoG train, drawn **by seed-question group** (2,768). CWQ questions generated from the same WebQSP question share the qid prefix before `_` and stay on one side. |
| Cal | The whole official RoG validation set (3,519; 345 groups) |
| Test | Official RoG test (3,531) |
| Seeds | 0, 1, 2. They change training only; cal and test are fixed. |
| Training | α = 0.3, two stages, early-stopping patience 8. Stage 2 cut val APSS by 31.5 / 38.4 / 39.2%. |
| 2-hop reachability | cal 0.806, test 0.801 |

Command: `python -m chase.data.splits --data data/processed/cwq --cal-from validation`. Why cal is not drawn from train: see [the last section](#why-cal-is-the-official-validation-set).

## Main table (official split)

| α | Target % | ECR % | APSS | APSS, 1000 re-splits | Empty sets % |
|---|---|---|---|---|---|
| 0.2 | 80 | 79.5 ± 0.3 | 148.26 ± 26.17 | 192.39 ± 6.57 | 0.5 ± 0.1 |
| 0.3 | 70 | 69.0 ± 0.5 | 15.21 ± 0.61 | 16.01 ± 0.98 | 6.5 ± 0.7 |
| 0.4 | 60 | 60.1 ± 0.9 | 7.73 ± 0.46 | 7.36 ± 0.42 | 13.4 ± 1.1 |
| 0.5 | 50 | 50.6 ± 0.3 | 4.58 ± 0.39 | 4.27 ± 0.32 | 22.6 ± 1.2 |
| 0.6 | 40 | 40.6 ± 0.4 | 2.74 ± 0.25 | 2.58 ± 0.20 | 34.8 ± 0.4 |
| 0.7 | 30 | 29.5 ± 0.4 | 1.53 ± 0.10 | 1.53 ± 0.11 | 51.1 ± 0.6 |
| 0.8 | 20 | 18.7 ± 0.9 | 0.73 ± 0.06 | 0.80 ± 0.05 | 68.5 ± 1.2 |

- α ∈ {0.05, 0.1}: the procedure abstains (τ̂ = +∞), because fewer than k cal questions are reachable.
- α = 0.2 is just under the ceiling (k = 2,816). τ̂ = +∞ in 23.7% of re-splits, and the sets are large.
- The pruned search returned exactly the unpruned sets for every test question at every α, at about 6.7 ms per question.

## Coverage uncertainty

| α | Official ECR % (s0 / s1 / s2) | Theory 95% interval | Re-split mean % | Percentile of official ECR in re-splits |
|---|---|---|---|---|
| 0.2 | 79.3 / 79.8 / 79.4 | 78.1–81.8 | 79.6 | 21 / 38 / 22 |
| 0.3 | 68.7 / 69.5 / 68.7 | 67.9–72.1 | 70.0 | 11 / 30 / 10 |
| 0.4 | 59.2 / 61.1 / 60.2 | 57.7–62.3 | 60.0 | 24 / 81 / 54 |
| 0.5 | 50.3 / 50.9 / 50.5 | 47.7–52.3 | 50.0 | 60 / 78 / 65 |
| 0.6 | 41.0 / 40.2 / 40.5 | 37.7–42.3 | 40.0 | 79 / 56 / 69 |
| 0.7 | 29.3 / 29.3 / 30.0 | 27.9–32.2 | 30.0 | 23 / 26 / 50 |
| 0.8 | 19.0 / **17.6** / 19.4 | 18.2–21.9 | 20.0 | 14 / 0.3 / 28 |

- **Theory interval:** for a fixed cal set, coverage follows Beta(k, n+1−k) with n = 3,519. The test ECR over 3,531 questions then follows a Beta-Binomial distribution; the column gives its 2.5–97.5% quantiles.
- **Re-splits:** 1,000 random re-partitions of the pooled cal and test questions, with the scorer fixed. They are exchangeable by construction.

**Finding:** 20 of 21 cells are inside the theory interval. About one cell in 20 is expected outside a 95% interval. Percentiles are spread over 10–80, so the official split behaves like a random split. Several cells are slightly below 1 − α, for example 68.7 at α = 0.3. That is expected: the guarantee holds on average over cal draws, not for every draw.

## Comparison with CPR and UaG

CPR and UaG numbers are as reported in the CPR paper. They use a different scorer, retrieval and split, and CPR draws cal from train.

**Same α.** CPR over-covers by 7–13 points (76.6% at target 70%; 42.8% at target 30%), while CHASE is on target. Comparing set sizes at the same α therefore favours CHASE, so the fair comparison is at equal coverage.

**Equal coverage.** CHASE is calibrated on cal at 1 − α′ = the other method's reported ECR. Its APSS is the mean over 1,000 re-splits, averaged over seeds.

| Coverage % | 76.6 | 68.1 | 58.8 | 50.2 | 42.8 | 29.9 |
|---|---|---|---|---|---|---|
| CPR APSS | 23.98 | 15.28 | 11.40 | 8.35 | 6.08 | 3.39 |
| CHASE APSS | 43.81 | 13.47 | 6.83 | 4.32 | 2.94 | 1.52 |
| Ratio | 1.83 | 0.88 | 0.60 | 0.52 | 0.48 | 0.45 |

Against UaG, at 42.1 / 37.3 / 26.7% coverage, the ratio is 0.02 / 0.02 / 0.04.

- CHASE loses at 76.6%. That level is close to the reachability ceiling (about 0.80): it needs about 96% of the reachable questions.
- CHASE wins at 68% coverage and below, with sets up to 2.2× smaller than CPR's, and 25–50× smaller than UaG's.

**Caveats:**
- CHASE's small APSS partly comes from empty sets: 23.6% of questions get an empty set at CPR's 50.2% coverage. CPR does not report its empty-set rate.
- WebQSP shows the opposite pattern: CHASE wins at high coverage and loses below about 60% (`runs/webqsp_compare/compare.md`). Same-scorer baselines are needed to separate the effect of the method from that of the scorer.

## Why cal is the official validation set

The first runs (`runs/cwq_trainCal_seed*`) followed CPR: they drew cal as 10% of RoG train, by seed group. On the official test split, ECR was then 4–5 points **below** target at α = 0.4–0.7 for all seeds. This was outside the theory interval, at percentiles 0–1.5 of the re-splits.

**Cause.** Groups do not separate *different* WebQSP questions about the same entity with the same answers. Call a question **seen** if some training question has the same topic entities and answer set.

| | Seen share | 2-hop reachability |
|---|---|---|
| Cal drawn from train (old) | 36% | 0.787 |
| Official validation (new cal) | 23% | 0.806 |
| Test | 27% | 0.801 |

At α = 0.3, reachable test questions were covered 95–98% if seen and about 80% if unseen. The old cal was therefore easier than test, and τ̂ came out too small. The official validation set is built like the test set, so it matches it. Using it as cal needs no test information and fixed the shift; see the coverage table above.

**Caveat:** questions within a group are dependent, so the theory interval, which treats them as independent, is somewhat too narrow.
