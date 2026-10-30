# Retrieval with a coverage guarantee: UaG, CPR, conformal top-K and CHASE

This note compares how each method builds its answer set and where each one is weak. The same-scorer experiments (`chase/baselines/`) re-implement UaG, CPR and top-K with the trained CHASE scorer. All four methods then differ only in how they build the set.

## Running example and goal

Question: *"Who is the wife of the director of Titanic?"* The topic entity is **Titanic**, and the gold path has two hops:

**Titanic** → `directed_by` → **James Cameron** → `spouse` → **Suzy Amis**

Every method answers the same two questions:
1. **Retrieval:** which edges of the graph to follow.
2. **Conformal step:** which nodes to return, so that the set contains at least one correct answer with probability ≥ 1 − α.

The metrics are:
- **ECR:** the share of test questions whose set contains a gold answer.
- **APSS:** the average set size.

---

## 1. CPR: fixed beam, then a post-hoc conformal filter

### Steps

1. **Beam retrieval (TreeG), with no conformal step yet.**
   - Hop 1: score every edge leaving Titanic (`directed_by`, `starring`, `release_date`, …) and keep the **32 best paths**.
   - Hop 2: extend each kept path by its 32 best edges, giving up to 1,024 paths, and again keep the **32 best**.
   - The candidates are the tails of the paths kept at each hop, so there are at most 64 entities.
2. **Path value.** CPR scores a path from its relation chain: cosine similarity between the question and the relations, plus a relation prior, plus a learned residual (RCVNet).
3. **Calibration.** For each calibration question, the score is the value of the best retrieved path that ends at a gold answer. If no gold answer is in the beam, the score is ∞.
4. **Threshold.** In CPR's code, ∞ is replaced by 10⁹, and τ̂ is `np.quantile` of the scores at level (n+1)(1−α)/n, with linear interpolation.
5. **Prediction.** Return the entities of the retrieved paths whose value is ≤ τ̂. If this set is empty, CPR adds the single best entity.

### Weaknesses

- **The beam cuts for good.** If the path through James Cameron is not in the top 32 at hop 1, the gold answer is lost, and no threshold can bring it back. Coverage is therefore capped by the beam's recall, not by the graph.
- **No abstention.** When many calibration questions miss in the beam, the ∞ scores become 10⁹ and τ̂ lands near 10⁹. CPR then returns the whole beam instead of saying that the level cannot be certified.
- **Sets are never empty, so CPR over-covers.** At high α, the fallback pulls ECR up towards Hit@1. CPR reports 50.8% ECR at a 20% target on WebQSP.
- **The conformal step only filters what the beam already chose.** It does not control the search itself.

---

## 2. UaG: one conformal threshold per hop

UaG has no public code. This section describes the idea of the paper. The paper does not give enough detail on how α is split across steps.

### Steps

1. **Hop 1.** Score every edge leaving Titanic and keep **every** edge with score ≤ τ₁. There is no fixed width, so an easy question keeps few edges and a hard one keeps many.
2. **Hop 2.** From the kept nodes, keep every edge with score ≤ τ₂.
3. **Answer.** A reached node is returned if its stop score is ≤ τ_stop.
4. **Calibration.** Each threshold is calibrated separately on the calibration set. For example, τ₁ is set so that the gold hop-1 step passes with probability 1 − α₁, and likewise for τ₂ and τ_stop.

### Weaknesses

- **Errors add up across steps.** The gold path must pass **every** threshold. If each threshold is set at α, the chance of passing all three is only about (1 − α)³: about 0.51 at α = 0.2, against a target of 0.80.
- **Fixing this makes sets large.** To reach 1 − α overall, each step needs a smaller level, for example α/3 (Bonferroni). Every threshold is then loose and the sets grow. UaG reports APSS 98.8 at α = 0.3 on WebQSP.
- **Exchangeability is broken.** Hop-2 scores are only seen for questions whose hop-1 step survived, which is a selected subset. The per-hop guarantees therefore do not combine into an exact one.

### Same-scorer versions

- **UaG-style per-hop:** α for every component, with no correction.
- **UaG-style Bonferroni:** α/(H+1) per component. This is valid by the union bound, but conservative.

---

## 3. Conformal top-K: a fixed number of answers

### Steps

1. **Ranking.** Run the search without pruning and rank every candidate node by its score (CHASE's σ_v, ties broken by node id).
2. **Calibration.** For each calibration question, the score is the rank of its best gold answer (∞ if no gold answer is reachable).
3. **Threshold.** K̂ is the split-conformal quantile of these ranks.
4. **Prediction.** Return the top K̂ nodes for every test question.

### Weaknesses

- **The set size is the same for every question.** An easy question that needs one answer still gets K̂. At high coverage, K̂ is driven by the hardest questions.
- **Coverage is not tight.** Ranks are integers, so scores tie. Coverage is ≥ 1 − α, but it can sit well above the target.
- **The minimum coverage is Hit@1.** K̂ ≥ 1, so the set is never empty, and coverage cannot drop below the share of questions whose top-ranked node is correct. At high α, top-K therefore over-covers.
- **No pruning.** The full ranking needs the unpruned search for every question.

### What it tests

Top-K uses the **same scorer and the same ranking** as CHASE. Only the cut differs: a fixed count instead of a score threshold. It is therefore the ablation for whether an adaptive, per-question set size is needed.

---

## 4. CHASE: one threshold, applied inside the search

### Steps

1. **Step scores.** A trained scorer gives every step (hop h, relation r, node v) a score s_h(q, r, v), and every possible stop a score; lower is better.
2. **Path score = its worst step.** A path survives a threshold τ only if every step scores ≤ τ, so its cost is the maximum of its step scores (with the stop score).
3. **Bottleneck DP.** The DP b_h(v) = min over edges u → v of max(b_{h−1}(u), s_h) gives every node an exact score σ_v: the smallest threshold at which the search returns v.
   - It runs in O(H(|V| + |E|)), with no beam or cap.
4. **Gap score.** σ_min(q) is the score of the most confident candidate of the question. Every candidate gets the gap g_v = σ_v − σ_min(q), and each calibration question gets g\*(q) = σ\*(q) − σ_min(q), where σ\*(q) = min over its gold answers of σ_a (∞ if no gold answer is reachable). τ̂ is the split-conformal quantile of g\*. If too few calibration questions are reachable, τ̂ = ∞ and CHASE **abstains** at that level.
   - Why the gap: σ's level varies between questions. One absolute threshold is then too loose for questions with low scores (large sets) and too tight for questions with high scores (empty sets). Measuring every question from its own best candidate makes one threshold comparable across questions.
5. **Prediction.** Run the same DP at test time and return {v : σ_v ≤ σ_min(q) + τ̂}. Pruning happens inside the search: hop h is pruned at U + τ̂, where U is the best candidate score found at earlier hops. U ≥ σ_min, so the returned set is exactly the unpruned one. Only the cost drops.
6. **Training.** Stage 1 uses BCE on gold steps. Stage 2 is conformal training through a soft version of the DP: it places a batch threshold on g\* and minimises the number of wrong answers within that gap.

### Why coverage is exact

- A gold answer is returned ⇔ g\*(q) ≤ τ̂.
- σ_min uses no labels, so g\* is one exchangeable score per question.
- The split-conformal theorem therefore applies directly: coverage ≥ 1 − α, and ≤ 1 − α + 1/(n+1) when there are no finite ties.
- Unreachable questions count as misses.

### Weaknesses

- **Coverage cannot go below Hit@1.** The most confident candidate always has gap 0 and is always returned, so sets are never empty. When 1 − α is below Hit@1, CHASE returns the top candidate and covers more than the target (like top-K and CPR's fallback).
- **Ties at the top.** The bottleneck score is a maximum of step scores, so several candidates can share σ_min. They are all returned at gap 0.
- **Capped by reachability.** Coverage above ρ_H, the share of questions with a reachable answer, is impossible. Near the cap the sets grow fast (CWQ, ρ ≈ 0.80). This cap applies to every method, but CHASE reports the cap honestly instead of hiding it.
- **The guarantee is marginal.** It holds on average over questions, not for each question, and it needs calibration and test questions to be exchangeable.
- **It needs a trained scorer.** Set sizes are only as good as the step and stop scores.

---

## Summary

| | CPR | UaG | Conformal top-K | CHASE |
|---|---|---|---|---|
| Where the conformal step acts | After retrieval (filter) | Inside retrieval, one threshold per hop | After ranking (cut by count) | Inside retrieval, **one threshold** |
| Score per question | Best gold path value in the beam | Separate score per step | Rank of the best gold answer | Gap of the best gold answer to the best candidate (g\* = σ\* − σ_min) |
| Can lose an answer to the search | Yes (beam) | No | No | No |
| Set size | Adaptive, at most the beam | Adaptive, large | Fixed K̂ | Adaptive |
| Empty sets | Never (fallback) | Yes | Never | Never |
| Coverage | Over-covers (fallback, whole-beam return) | Adds up per hop, or conservative | ≥ 1 − α, not tight | Exact above Hit@1: 1 − α to 1 − α + 1/(n+1) |
| Abstains when the level is out of reach | No | Not specified | Yes | Yes |
