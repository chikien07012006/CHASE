<div align="center">

# CHASE

### Conformal Heuristic Search with Exact Coverage<br>for Multi-hop Evidence Retrieval on Knowledge Graphs

*Prune inside the search with one calibrated threshold, and return an answer set that contains a correct answer with probability ≥ 1 − α: valid, tight, and LLM-free.*

![status](https://img.shields.io/badge/status-research%20in%20progress-2a78d6)
![python](https://img.shields.io/badge/python-3.11-52514e)
![tests](https://img.shields.io/badge/tests-356%20passing-1baf7a)
![llm](https://img.shields.io/badge/LLM%20calls-0-52514e)

</div>

> **Status (11 Nov 2026).** The full pipeline (graphs → scorer → bottleneck DP → conformal calibration → two-stage training → official evaluation) is implemented and tested. Results on **WebQSP**, **CWQ** and **BioHopR** (3 seeds each), the comparison with reported CPR / UaG, the same-scorer baselines with a paired bootstrap, and the ablations (gap vs rank, conformal training, tie-breaking) are in [`docs/results.md`](docs/results.md) and [BioHopR](#biohopr). Next: paper.

---

## Contents

[Motivation](#motivation) · [Method](#method) · [Results](#results) · [Benchmarks](#benchmarks) · [Repository layout](#repository-layout) · [Setup](#setup) · [Reproducing](#reproducing-the-results) · [Testing](#testing) · [Roadmap](#roadmap)

---

## Motivation

Missing a drug–target or drug–drug interaction is a safety failure, not a lower score. Retrieval for such settings needs a **guarantee**, not a good average. Existing conformal KGQA methods compromise:

| Method | Where the threshold acts | Guarantee | Limitation |
|---|---|---|---|
| UaG (AAAI 2025) | Inside beam search; one threshold per hop | Per-hop calibration breaks exchangeability | Coverage can fail; Freebase only |
| CPR (arXiv 2605.08077) | Post-hoc filter after a fixed beam (TreeG 32×32) | Valid, but conservative; ignores τ̂ = ∞ | Coverage capped by beam recall; one LLM call per question |
| **CHASE** | **Inside the search, one exchangeable score per question** | **Valid and tight:** 1 − α ≤ coverage ≤ 1 − α + 1/(n+1) | Capped only by KG reachability ρ_H |

## Method

```
question + topic entities ─► state-based scorer ─► bottleneck DP, pruned at τ̂ ─► answer set + evidence walks
                                                   (σ_v ≤ σ_min(q) + τ̂)       ▲
              calibration set ─► unpruned DP ─► g*(q) = σ*(q) − σ_min(q) ─► conformal quantile ─► τ̂
```

1. **State-based scorer.** A step head scores each expansion (hop $h$, relation $r$, node $v$); a stop head scores returning $v$ at hop $h$. Each head is $\mathrm{MLP}([q;r;h]) + \langle P(q,h), Q(v)\rangle$ with frozen MiniLM inputs, and the nonconformity is $s=\mathrm{softplus}(-\text{logit})$. The logit is additive, so scoring any edge is a table lookup: 1M edges in 37 ms.
2. **Max-prefix score.** Irreversible pruning keeps a walk iff every step scores ≤ τ, so a walk's cost is its worst step.
3. **Bottleneck DP.** $b_h(v)=\min_{u\to v}\max\{b_{h-1}(u),\,s_h(q,r,v)\}$ gives, for every node, $\sigma_v$ = the smallest threshold at which the search returns $v$. It is exact, takes $O(H(|V|+|E|))$, and needs no beam or hard cap.
4. **Calibration on the gap.** $\sigma^*(q)=\min_{a\in A_q}\sigma_a$ is the smallest threshold at which the search finds a correct answer, and $\sigma_{\min}(q)$ is the score of the most confident candidate. CHASE calibrates the gap $g^*(q)=\sigma^*(q)-\sigma_{\min}(q)$ and returns $\{v:\sigma_v\le\sigma_{\min}(q)+\hat\tau\}$. $\sigma_{\min}$ uses no labels, so $g^*$ is still one exchangeable score per question and coverage is exact. Measuring every question from its own best candidate makes one threshold comparable across questions; the most confident candidate is always returned, so coverage cannot fall below Hit@1.
   **Ties.** Nodes reached through the same weak step share σ, so candidates are ordered by the pair (gap, λ·stop score at the node's best hop), lexicographically, and τ̂ is the conformal quantile of that pair. Conformal calibration needs only a total order, so the guarantee is unchanged, and at the lowest level the set is the single top candidate. The pruned search uses the best candidate found so far as an upper bound on $\sigma_{\min}$ and returns exactly the unpruned set. Levels above the KG ceiling are reported as **infeasible**.
5. **Two-stage training.** Stage 1 trains a BCE loss on steps and stops, with labels from filtered gold paths. Stage 2 extends ConfTr to graph search: a soft bottleneck DP plus a differentiable quantile on half a batch, minimising the soft number of *wrong* answers within the calibrated gap on the other half.
6. **Type feasibility (typed KGs)** and **conformal risk control** for multi-answer recall are implemented for the biomedical benchmark.

Full method and proofs: [`proposal_chase.tex`](proposal_chase.tex). The coverage theorem with unreachable questions, abstention, ties and multi-answer targets: [`theory.tex`](theory.tex).

## Results

3 seeds per dataset, H = 2, commit `debca94`. Full tables, coverage checks and sources for every number: [`docs/results.md`](docs/results.md).

**Coverage is on target.** Over 1,000 re-splits of cal and test, mean ECR equals 1 − α to within 0.5 points on both datasets. On the official split, CWQ is on target; WebQSP, with only 283 cal questions, is on average 1–2 points below, inside the theory interval. Coverage cannot fall below Hit@1 (55.4% WebQSP, 32.9% CWQ), where the set is the single top candidate, and CWQ abstains at α ≤ 0.1 because of its reachability ceiling (0.80).

**Equal coverage vs reported numbers.** APSS ratio CHASE / other (< 1: CHASE smaller):

| | WebQSP | CWQ |
|---|---|---|
| CPR (reported) | 0.37–0.47 at 58–77% | 0.35–0.82 at 43–68%; **1.71 at 77%** (near the ceiling) |
| UaG (reported) | 0.06–0.08 at 60–72% | 0.01–0.02 at 37–42% |

**Same scorer.** With every method on the trained CHASE scorer:
- CHASE beats UaG-style per-hop and Bonferroni (1.6–3.1× smaller sets) and fixed beams with post-hoc filtering at B ≥ 32 at almost every level.
- **Paired cluster bootstrap** (10,000 resamples of the test questions, Holm-adjusted): against UaG-style and CPR-style B = 128, every significant difference favours CHASE, on all three datasets. Narrow beams are not beaten: B = 8 is significantly smaller on WebQSP at the levels it reaches (≤ 77%), and indistinguishable on CWQ and BioHopR.
- Beams lose answers. B = 8 / 32 / 128 reaches a gold answer for 77 / 86 / 92% of WebQSP and 53 / 67 / 76% of CWQ questions, against 95% and 80% for CHASE.
- **Ablation, gap vs rank threshold.** Conformal top-K (same scorer, same calibration, rank instead of gap) is smaller at mid coverage (up to 2.2× on WebQSP at 66–81%), and CHASE is smaller at high coverage (≥ 90% WebQSP, 80% CWQ); under the bootstrap, only top-K's mid-coverage win and CHASE's win at 93% on WebQSP are significant. Top-K is not a method of the literature; it isolates the choice of score.
- **Ablation, conformal training.** Against the stage-1 (BCE) checkpoint at the same coverage, stage 2 shrinks APSS by 25–40% on WebQSP, 20–30% on CWQ and 49–59% on BioHopR, and raises Hit@1 (50.8 → 55.4%, 29.8 → 32.9%, 82.9 → 89.0%).

| | |
|---|---|
| ![WebQSP equal coverage](docs/figures/webqsp_equal_coverage.png) | ![CWQ equal coverage](docs/figures/cwq_equal_coverage.png) |
| ![WebQSP same scorer](docs/figures/webqsp_baselines.png) | ![CWQ same scorer](docs/figures/cwq_baselines.png) |
| ![WebQSP gap vs rank](docs/figures/webqsp_ablation_rank.png) | ![CWQ gap vs rank](docs/figures/cwq_ablation_rank.png) |

The pruned search returns exactly the unpruned set on every test question, visits 3–27× fewer DP states, and takes about 7 ms per question on a laptop CPU.

### BioHopR

3 seeds, own 70 / 15 / 15 split by topic, cal = validation (1,520 questions), training α = 0.1. No published numbers exist for BioHopR, so the same-scorer baselines are the comparison. Target: at least one correct answer.

- **Only high coverage is informative.** Hit@1 is about 89% (a question has 33 answers on average), so every α ≥ 0.105 returns the single top candidate; 3.5–4.5% of questions have no reachable answer, so α ≤ 0.045 often abstains. Results use α = 0.04–0.12 in steps of 0.005, i.e. coverage 89–95%.
- **Coverage is on target**: re-split mean ECR 95.0% at α = 0.05. The official test split is 1–2 points below, because it has fewer reachable questions than cal (95.5% vs 96.5%).
- **Same scorer, equal coverage** (APSS ratio CHASE / baseline; significant under the bootstrap against UaG-style per-hop and CPR-style B = 128, not against B = 8 / 32): UaG-style per-hop 0.11–0.38 at 90–94%; UaG-style Bonferroni 0.19 at 94% (its only feasible level); CPR-style B = 128 0.07–0.78 at 89–94%; B = 32 0.12–0.98 below 91.5% and 0.94–1.22 at 91.5–92%; B = 8 0.29–0.99 at 89–90% and 1.25–1.41 at 90.3–90.4%, and cannot exceed 91%. Beams reach a gold answer for 90.9 / 92.5 / 93.9% of questions, CHASE for 95.5%.
- **Ablation, gap vs rank threshold.** Conformal top-K is smaller at 89.6–94% (1.2–6.6×); CHASE is smaller at ≥ 94.6% (0.71–0.95). Under the bootstrap, top-K's win is significant at 89.3–93.5% (1.47–6.37×) and CHASE's is not. The gold answer usually ranks among the first few candidates, but its gap varies widely across questions, so one gap threshold must be wide.
- **Ties matter most here.** Breaking ties by the stop score shrinks the set at the lowest level from 4.18 to 0.98 nodes, with the same coverage elsewhere.

| | |
|---|---|
| ![BioHopR same scorer](docs/figures/biohopr_baselines.png) | ![BioHopR gap vs rank](docs/figures/biohopr_ablation_rank.png) |

## Benchmarks

| Dataset | Searchable KG | Role | Test | Answer in graph | ρ₁ | ρ₂ |
|---|---|---|---|---|---|---|
| WebQSP (RoG) | Freebase, one subgraph per question | Direct comparison with UaG / CPR | 1,628 | 95.64% | 66.28% | **95.33%** |
| CWQ (RoG) | Freebase, one subgraph per question | Direct comparison, compositional | 3,531 | 80.66% | 43.95% | **80.12%** |
| BioHopR | Full PrimeKG (134k nodes, 4.1M edges), one type-feasible subgraph per question | Biomedical, multi-answer (≈ 33 answers / question) | 1,518 (own split) | 95.5% | 27.6% | **95.5%** |

ρ_H is the share of questions with a gold answer reachable within H hops: an upper bound on coverage for **every** method. It stops growing at H = 2, because RoG subgraphs are 2-hop neighbourhoods.

## Repository layout

```
chase/
  kg/          graph.py (CSR KG) · build_primekg.py · build_freebase.py · subgraphs.py · type_distance.py
  model/       text_cache.py (frozen MiniLM) · features.py · scorer.py (StateScorer, BoundScorer, ScoreTables)
  search/      dp.py (bottleneck DP, evidence walks, rho_H)
  conformal/   split.py (split CP, CRC) · calibrate.py (calibration scores from the DP)
  data/        splits.py · gold_paths.py (path enumeration + coincidental-path filter) · biohopr.py (BioHopR store on PrimeKG)
  train/       losses.py (BCE, soft DP, ConfTr) · sampler.py · train.py · evaluate.py
  eval/        freebase.py (official evaluation) · compare.py (multi-seed, same α / equal coverage) · plots.py · reference.py (published baselines)
  baselines/   same-scorer baselines: conformal top-K · CPR-style beam · UaG-style per-hop · bootstrap.py (paired cluster bootstrap)
tests/         356 unit tests
docs/          results.md (all results and their sources) · baselines.md (how UaG, CPR, top-K and CHASE build their sets) · figures/
biohopr.md (BioHopR pipeline, in Vietnamese) · proposal_chase.tex · theory.tex · requirements.txt
```

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2 \
  --include "*.json" "*.txt" "model.safetensors" "1_Pooling/*"
```

Raw data goes under `data/raw/` (git-ignored): PrimeKG `kg_giant.csv` (Harvard Dataverse), RoG-WebQSP / RoG-CWQ (Hugging Face `rmanluo/*`), BioHopR (Hugging Face `knowlab-research/BioHopR`).

## Reproducing the results

WebQSP end to end. Times are for an Apple-silicon laptop, CPU only.

```bash
D=data/processed/webqsp
python -m chase.kg.build_freebase --raw data/raw/rog_webqsp --out $D          # subgraphs          ~40 s
python -m chase.model.text_cache freebase $D                                  # MiniLM cache       ~3 min
python -m chase.data.splits --data $D                                         # train/cal/val/test
python -m chase.data.gold_paths --data $D --split train --H 2                 # training labels    ~6 s
for s in 0 1 2; do
  python -m chase.train.train --data $D --out runs/webqsp_seed$s --seed $s    # two-stage training ~10 min
  python -m chase.eval.freebase --data $D --model runs/webqsp_seed$s/best.pt \
                                --out runs/webqsp_seed$s/eval                 # evaluation         ~2 min
done
python -m chase.eval.compare --evals runs/webqsp_seed{0,1,2}/eval --out runs/webqsp_compare
```

For CWQ, replace `webqsp` with `cwq`, and make two other changes:
- split with `chase.data.splits --data $D --cal-from validation`;
- pass `--alpha 0.3` to training and `--main-alpha 0.3 --dataset cwq` to evaluation.

CWQ training takes about 1.5 h per seed.

The evaluation writes these files to the `--out` folder:
- `results_table.md`, `results.csv` and `metrics.json`;
- `validity.json`, with 1,000 re-splits;
- `matched.json`, with the equal-coverage points;
- `per_question.parquet`;
- seven figures (PNG + PDF).

`compare` writes `compare.md` and the same-α and equal-coverage figures.

Same-scorer baselines, per seed and then over seeds:

```bash
for s in 0 1 2; do
  python -m chase.baselines.run --data $D --model runs/webqsp_seed$s/best.pt \
                                --out runs/webqsp_seed$s/baselines --beams 8 32 128
done
python -m chase.baselines.report --runs runs/webqsp_seed{0,1,2}/baselines --out runs/webqsp_baselines
python -m chase.baselines.bootstrap --runs runs/webqsp_seed{0,1,2}/baselines --out runs/webqsp_baselines
```

Biomedical graph (PrimeKG):

```bash
python -m chase.kg.build_primekg --csv data/raw/primekg/kg_giant.csv --out data/processed/primekg_full
python -m chase.model.text_cache primekg data/processed/primekg_full
python -m chase.kg.type_distance --kg data/processed/primekg_full
```

BioHopR: only the store differs; every later step reuses the commands above. Details, timings and the full command list: [`biohopr.md`](biohopr.md).

```bash
D=data/processed/biohopr
python -m chase.data.biohopr --out $D                                         # store on PrimeKG   ~1.5 min
python -m chase.model.text_cache freebase $D
python -m chase.data.splits --data $D --cal-from validation
python -m chase.data.gold_paths --data $D --split train --H 2
# train as above (default α = 0.1); evaluate and run the baselines on the high-coverage grid:
A=(0.04 0.045 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09 0.095 0.1 0.105 0.11 0.115 0.12)   # zsh array
python -m chase.eval.freebase --data $D --model runs/biohopr_seed0/best.pt --out runs/biohopr_seed0/eval \
                              --dataset biohopr --alphas $A
python -m chase.baselines.run --data $D --model runs/biohopr_seed0/best.pt --out runs/biohopr_seed0/baselines \
                              --beams 8 32 128 --alphas $A
```

`--alphas` sets the levels of the tables; the baseline figures are drawn over that α range only. `--tie-break none` (eval and baselines) orders candidates by σ alone, for the ablation.

`report` writes `baselines_compare.md`, `fig_baselines_equal_coverage` (CHASE vs CPR-style and UaG-style) and `fig_ablation_rank` (CHASE vs top-K). `run` also saves per-question coverage and sizes (`per_question.npz`), which `bootstrap` resamples into `bootstrap.md` / `bootstrap.json`.

## Testing

```bash
python -m pytest tests -q        # 356 tests, ~20 s
```

The key invariants are checked against brute force: the DP's σ equals enumeration of every walk; a pruned search returns exactly the unpruned answer set; the soft DP converges to the hard DP as T → 0; split-conformal and CRC thresholds meet their guarantees in simulation; `bind()` lookup tables equal the full forward pass.

## Roadmap

- [x] Full PrimeKG graph; WebQSP / CWQ subgraph stores; MiniLM caches
- [x] State-based scorer, type feasibility, bottleneck DP, reachability ceilings
- [x] Conformal calibration (split CP, CRC)
- [x] Gold paths, sampler, two-stage training (BCE + conformal training)
- [x] Official evaluation, 1000-re-split validity check, figures
- [x] WebQSP / CWQ results, 3 seeds each
- [x] Same-α and equal-coverage comparison with reported CPR / UaG; theory note
- [x] Same-scorer baselines: per-hop (UaG-style), Bonferroni, fixed beam + post-hoc (CPR-style), conformal top-K
- [x] BioHopR store (name linking, type-feasible subgraphs, topic-grouped split); 3 seeds; same-scorer baselines
- [x] Tie-breaking by the stop score (ablation: `--tie-break none`)
- [x] Paired cluster bootstrap; ablation of conformal training (stage 1 vs 2)
- [ ] BioHopR: CRC for multi-answer recall; Mondrian by hop count
- [ ] Paper

## Citation

Paper in preparation. Please contact the authors before citing.

## License

To be decided before public release.
