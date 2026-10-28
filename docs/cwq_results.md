# CWQ results: 3 seeds

All numbers come from `runs/cwq_seed{0,1,2}/eval/{metrics,validity}.json` and `validity_ecr.npz`. Code is at commit `a9f28be`; the only later change is the `patience` default (4 → 8), which these runs set explicitly.

## Setup and split

| Item | Value |
|---|---|
| Graph | RoG-CWQ, one Freebase subgraph per question, H = 2 |
| Train / cal | RoG train split into train (22,103) and cal (2,768). Cal is about 10% of RoG train, as in CPR, drawn **by seed-question group**: all CWQ questions generated from the same WebQSP question (same qid prefix before `_`) go to the same side, so no group is shared by train and cal. |
| Validation | Official RoG validation (3,519). Used for early stopping only, never for calibration. |
| Test | Official RoG test (3,531). Test groups are disjoint from train groups. |
| Seeds | 0, 1, 2. The seed changes **training only**; all seeds share the same cal and test sets. |
| Training | α = 0.3 (α_eff = 0.119, ρ_train = 0.795), two stages, early-stopping patience 8 |
| Reachability (2 hops, any gold answer) | cal 2,177 / 2,768 = 0.787; test 0.801 |
| Abstained levels | α ∈ {0.05, 0.1, 0.2}: fewer than k cal questions are reachable (at α = 0.2, k = 2,216 > 2,177), so τ̂ = +∞ |

Stage 2 (conformal training) cut validation APSS by 25.8% / 37.7% / 45.8% (seeds 0 / 1 / 2).

## Main table (official split, mean ± std over 3 seeds)

| α | Target % | ECR % | APSS | Empty sets % | Median size | Hit@1 | ECR/APSS |
|---|---|---|---|---|---|---|---|
| 0.3 | 70 | 67.9 ± 0.4 | 14.22 ± 1.23 | 6.7 ± 0.7 | 8 | 0.312 ± 0.016 | 4.8 ± 0.4 |
| 0.4 | 60 | 55.4 ± 0.3 | 5.97 ± 0.09 | 17.6 ± 1.8 | 3 | 0.299 ± 0.018 | 9.3 ± 0.2 |
| 0.5 | 50 | 45.1 ± 0.6 | 3.35 ± 0.27 | 29.6 ± 1.2 | 2 | 0.275 ± 0.019 | 13.5 ± 1.3 |
| 0.6 | 40 | 34.9 ± 0.5 | 2.03 ± 0.19 | 43.3 ± 0.9 | 1 | 0.236 ± 0.016 | 17.3 ± 1.9 |
| 0.7 | 30 | 25.9 ± 0.3 | 1.22 ± 0.15 | 57.1 ± 1.7 | 0 | 0.189 ± 0.010 | 21.4 ± 2.6 |
| 0.8 | 20 | 17.3 ± 1.0 | 0.65 ± 0.12 | 71.2 ± 3.1 | 0 | 0.137 ± 0.002 | 26.9 ± 3.8 |

- Std uses ddof = 1.
- The pruned search returned exactly the unpruned sets for every test question at every α (`pruned_matches_unpruned`), at about 6.7 ms per question.
- CPR is not compared here, because CHASE misses the target on this split (next section).

## Coverage uncertainty

| α | Target % | Official ECR % (s0 / s1 / s2) | Re-split mean % | Re-split 5–95% | Theory 95% interval | Official ECR percentile in re-splits (s0 / s1 / s2) |
|---|---|---|---|---|---|---|
| 0.3 | 70 | 67.5 / 67.8 / 68.3 | 70.0 | 68.1–71.9 | 67.7–72.3 | 2.5 / 2.8 / 7.1 |
| 0.4 | 60 | 55.7 / 55.3 / 55.2 | 60.0 | 57.9–62.0 | 57.6–62.4 | 0.1 / 0.0 / 0.0 |
| 0.5 | 50 | 45.7 / 44.6 / 45.1 | 50.0 | 47.9–52.1 | 47.5–52.5 | 0.2 / 0.0 / 0.0 |
| 0.6 | 40 | 35.4 / 34.8 / 34.4 | 40.0 | 38.0–42.0 | 37.6–42.5 | 0.0 / 0.0 / 0.0 |
| 0.7 | 30 | 25.6 / 26.2 / 25.9 | 30.0 | 28.2–31.9 | 27.8–32.3 | 0.0 / 0.0 / 0.0 |
| 0.8 | 20 | 16.2 / 17.8 / 17.9 | 20.0 | 18.4–21.7 | 18.0–22.0 | 0.0 / 1.4 / 1.5 |

How to read the columns:
- **Re-split columns:** 1,000 random re-partitions of the pooled cal and test questions, keeping the same sizes, with the trained scorer fixed. Pooling makes cal and test exchangeable by construction, so these columns show what the procedure does when the exchangeability assumption holds. They are averaged over seeds.
- **Theory 95% interval:** for a fixed calibration set, coverage is Beta(k, n+1−k) with n = 2,768 and k = ⌈(n+1)(1−α)⌉. The test ECR over 3,531 questions then follows a Beta-Binomial distribution, and this column gives its 2.5–97.5% quantiles. It assumes cal and test are exchangeable.
- **Percentile:** where the official ECR falls among the 1,000 re-split ECRs.

**Finding:**
- Under exchangeability, the procedure is exact: the re-split mean equals 1−α at every level.
- On the official split, ECR is 4–5 points **below** the target at α = 0.4–0.7, for all three seeds. This is outside the theory interval, and at percentile 0–1.5 of the re-splits.
- At α = 0.3 it is at the lower edge of the interval.
- This is a calibration/test shift, not random fluctuation. Under CPR's rule (ECR < 1−α), these cells would count as failed coverage.

## Diagnosis: cal is easier than test

The group split removes questions generated from the same WebQSP question. It does not remove *different* WebQSP questions about the same entity with the same answers. Call a question **seen** if some training question has the same topic entities and the same answer set.

| Split | Seen share |
|---|---|
| Cal | 35.9% |
| Official test | 26.6% |
| WebQSP cal / test (for comparison) | 28.3% / 24.1% |

Test coverage at α = 0.3, counting reachable questions only:

| Seed | Seen | Unseen |
|---|---|---|
| 0 | 98.1% | 79.4% |
| 1 | 95.3% | 80.9% |
| 2 | 95.4% | 81.6% |

Reachability is the same in both groups (0.79 vs 0.80), so the difference comes from the scorer, not from the graph.

Seen questions are much easier for the scorer, and cal contains more of them than test does. τ̂ is therefore calibrated on an easier mix and comes out too small for test. This is consistent with the effect being larger now than with the earlier patience-4 model (official ECR 70.0% at α = 0.3): a better-fitted scorer separates seen from unseen questions more sharply.

**Status:** this is a measured association, not yet a verified cause.

**Possible fixes (not done; both change the split protocol, not the method):**
- **(a) Component split.** Merge seed-question groups that share a (topic, answers) pair into one component, and assign whole components to train or cal. No cal question is then seen. Since 26.6% of test questions are seen, this makes cal *harder* than test, so coverage becomes conservative (valid but above target).
- **(b) Stratified cal.** Draw cal so that its seen share matches the test share (about 27%). Coverage should then return inside the theory interval.

Either way: retrain, re-evaluate, and confirm the official ECR against the theory interval.

## Fix adopted (28 Oct): calibrate on the official validation set

The official CWQ validation set is built like the test set: its seed groups appear in neither train nor test.

| | Seen share | 2-hop reachability |
|---|---|---|
| Validation | 23% | 0.806 |
| Test | 27% | 0.801 |
| Old cal | 36% | 0.787 |

New protocol (`python -m chase.data.splits --cal-from validation`):
- **cal** = the whole validation set (3,519 questions, 345 groups);
- **val** = early stopping only, the same 10% train groups that were cal before;
- **train** = unchanged, so the gold paths are unchanged too.

This uses no test information, and it is stricter than CPR's protocol, which samples cal from train at random.

**Preview with the current models** (slightly optimistic, because they were early-stopped on the validation set). Official test ECR %, with the theory 95% interval:

| α | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|---|
| seed 0 | 79.2 | 73.2 | 67.9 | 59.6 | 50.4 | 40.5 | 29.5 | 18.7 |
| seed 1 | 79.6 | 74.5 | 70.5 | 61.1 | 51.3 | 41.1 | 30.8 | 19.6 |
| seed 2 | 79.4 | 74.2 | 69.5 | 60.2 | 50.2 | 41.1 | 28.7 | 18.4 |
| interval | 78–82 | 73–77 | 68–72 | 58–62 | 48–52 | 38–42 | 28–32 | 18–22 |

Coverage is back on target at every level, and α = 0.2 and 0.25 become certifiable.

**Caveat:** questions within a group are dependent, so the interval, which treats questions as independent, is somewhat too narrow. With half of the validation set as cal (170 groups), reachability was 0.830 by chance, which shows how much the 345 groups vary. That is why the whole validation set is used.

The models must be retrained, because the validation set was used for early stopping.
