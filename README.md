<div align="center">

# CHASE

### Conformal Heuristic Search with Exact Coverage<br>for Multi-hop Evidence Retrieval on Knowledge Graphs

*Prune inside the search with one calibrated threshold, and return an answer set that contains a correct answer with probability ≥ 1 − α: valid, tight, and LLM-free.*

![status](https://img.shields.io/badge/status-research%20in%20progress-2a78d6)
![python](https://img.shields.io/badge/python-3.11-52514e)
![tests](https://img.shields.io/badge/tests-164%20passing-1baf7a)
![llm](https://img.shields.io/badge/LLM%20calls-0-52514e)

</div>

> **Status (29 Oct 2026).** The full pipeline (graphs → scorer → bottleneck DP → conformal calibration → two-stage training → official evaluation) is implemented and tested. Results on **WebQSP** and **CWQ** are below, with 3 seeds each. Same-scorer baselines and BioHopR are in progress.

---

## Highlights

- **Exact coverage.** Over 1000 random calibration/test re-splits, empirical coverage matches the split-conformal theory to three decimals at every α (e.g. 0.902 vs 0.901 at α = 0.1). CPR over-covers by 7–31 points, i.e. it is conservative.
- **Reaches the high-coverage regime.** CHASE certifies α = 0.1 and 0.2 on WebQSP (ECR 87.6% / 78.2%) and α = 0.2 on CWQ (79.5%). Neither CPR nor UaG reports these levels.
- **Smaller sets at equal coverage.** Compared with CPR, CHASE returns 13–33% fewer answers on WebQSP at coverage ≥ 61%, and 12–55% fewer on CWQ at coverage ≤ 68%. Compared with UaG, it returns up to 10× fewer answers on WebQSP and 25–50× fewer on CWQ.
- **Cheap and adaptive.** 6.4 ms per question on a single CPU thread with no LLM. Pruning expands ~48 states per question at α = 0.1 vs 462 without pruning.
- **Conformal training works.** Optimising a differentiable set-size loss cuts validation APSS by 38.7% over BCE-only training.

<p align="center">
  <img src="docs/figures/fig2_coverage_vs_target.png" width="48%" alt="Requested vs achieved coverage">
  <img src="docs/figures/fig1_validity_hist.png" width="50%" alt="ECR over 1000 re-splits">
</p>

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
                                                              ▲
              calibration set ─► unpruned DP ─► σ*(q) per question ─► conformal quantile ─► τ̂
```

1. **State-based scorer.** A step head scores each expansion (hop $h$, relation $r$, node $v$); a stop head scores returning $v$ at hop $h$. Each head is $\mathrm{MLP}([q;r;h]) + \langle P(q,h), Q(v)\rangle$ with frozen MiniLM inputs, and the nonconformity is $s=\mathrm{softplus}(-\text{logit})$. The logit is additive, so scoring any edge is a table lookup: 1M edges in 37 ms.
2. **Max-prefix score.** Irreversible pruning keeps a walk iff every step scores ≤ τ, so a walk's cost is its worst step.
3. **Bottleneck DP.** $b_h(v)=\min_{u\to v}\max\{b_{h-1}(u),\,s_h(q,r,v)\}$ gives, for every node, $\sigma_v$ = the smallest threshold at which the search returns $v$. It is exact, takes $O(H(|V|+|E|))$, and needs no beam or hard cap.
4. **Calibration.** $\sigma^*(q)=\min_{a\in A_q}\sigma_a$ is the smallest threshold at which the search finds a correct answer. Split-conformal quantiles of $\sigma^*$ give $\hat\tau$. Because "found" ⇔ $\sigma^*\le\tau$, coverage is tight. Levels above the KG ceiling are reported as **infeasible**.
5. **Two-stage training.** Stage 1 trains a BCE loss on steps and stops, with labels from filtered gold paths. Stage 2 extends ConfTr to graph search: a soft bottleneck DP plus a differentiable quantile on half a batch, minimising the soft number of *wrong* answers returned on the other half.
6. **Type feasibility (typed KGs)** and **conformal risk control** for multi-answer recall are implemented for the biomedical benchmark.

Full method and proofs: [`proposal_chase.tex`](proposal_chase.tex). The coverage theorem with unreachable questions, abstention, ties and multi-answer targets: [`theory.tex`](theory.tex).

## Results

Official test splits, H = 2, 3 seeds. Mean ± std over seeds. CPR and UaG numbers are as reported in the CPR paper; they use a different scorer, retrieval and split.

- **WebQSP:** cal is 283 questions drawn from train, test is 1,628 questions (same sizes as CPR).
- **CWQ:** cal is the official validation set (3,519 questions), test is 3,531 questions. A cal drawn from train under-covered on test; see [`docs/cwq_results.md`](docs/cwq_results.md).

**Coverage.** ECR (%) on the official split. "abstain" means fewer than k cal questions are reachable, so τ̂ = ∞.

| α | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|---|---|
| target | 95 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 |
| WebQSP | 92.7 ± 0.3 | 87.6 ± 0.4 | 78.2 ± 1.5 | 68.5 ± 1.7 | 60.5 ± 0.9 | 49.2 ± 1.1 | 39.1 ± 0.9 | 28.7 ± 1.8 | 18.2 ± 2.2 |
| CWQ | abstain | abstain | 79.5 ± 0.3 | 69.0 ± 0.5 | 60.1 ± 0.9 | 50.6 ± 0.3 | 40.6 ± 0.4 | 29.5 ± 0.4 | 18.7 ± 0.9 |

- Over 1,000 cal/test re-splits, the mean ECR equals 1 − α at every feasible level.
- On CWQ, 20 of 21 seed × α cells fall inside the theory 95% interval.
- CPR over-covers by 7–31 points, so comparing set sizes at the same α favours CHASE.

**Equal coverage.** CHASE is calibrated on cal at 1 − α′ = the other method's reported ECR. The table gives CHASE APSS ÷ the other method's APSS, so a value below 1 means CHASE returns smaller sets. CHASE APSS is the mean over 1,000 re-splits.

| WebQSP coverage % | 76.8 | 67.3 | 61.4 | 58.3 | 55.1 | 50.8 |
|---|---|---|---|---|---|---|
| vs CPR | **0.67** | **0.76** | **0.87** | 1.03 | 1.70 | 2.76 |

| CWQ coverage % | 76.6 | 68.1 | 58.8 | 50.2 | 42.8 | 29.9 |
|---|---|---|---|---|---|---|
| vs CPR | 1.83 | **0.88** | **0.60** | **0.52** | **0.48** | **0.45** |

- **vs UaG:** the ratio is 0.10–0.97 on WebQSP and 0.02–0.04 on CWQ.
- **Where CHASE loses:**
  - Near CWQ's reachability ceiling (about 0.80).
  - At low coverage on WebQSP, where one global threshold leaves many sets empty (34% at 51% coverage) while others stay large.
- **Full tables:** `runs/{webqsp,cwq}_compare/compare.md`, produced by `chase.eval.compare`.

<p align="center">
  <img src="docs/figures/webqsp_equal_coverage.png" width="48%" alt="WebQSP: APSS vs coverage">
  <img src="docs/figures/cwq_equal_coverage.png" width="48%" alt="CWQ: APSS vs coverage">
</p>

<p align="center">
  <img src="docs/figures/fig5_cost.png" width="90%" alt="Search cost">
</p>

<p align="center">
  <img src="docs/figures/fig6_training_curves.png" width="70%" alt="Training curves">
</p>

## Benchmarks

| Dataset | Searchable KG | Role | Test | Answer in graph | ρ₁ | ρ₂ |
|---|---|---|---|---|---|---|
| WebQSP (RoG) | Freebase, one subgraph per question | Direct comparison with UaG / CPR | 1,628 | 95.64% | 66.28% | **95.33%** |
| CWQ (RoG) | Freebase, one subgraph per question | Direct comparison, compositional | 3,531 | 80.66% | 43.95% | **80.12%** |
| BioHopR | Full PrimeKG (134k nodes, 4.1M edges) | Biomedical, multi-answer (36.65 answers / question) | own split | — | — | — |

ρ_H is the share of questions with a gold answer reachable within H hops: an upper bound on coverage for **every** method. It stops growing at H = 2, because RoG subgraphs are 2-hop neighbourhoods.

## Repository layout

```
chase/
  kg/          graph.py (CSR KG) · build_primekg.py · build_freebase.py · subgraphs.py · type_distance.py
  model/       text_cache.py (frozen MiniLM) · features.py · scorer.py (StateScorer, BoundScorer, ScoreTables)
  search/      dp.py (bottleneck DP, evidence walks, rho_H)
  conformal/   split.py (split CP, CRC) · calibrate.py (calibration scores from the DP)
  data/        splits.py · gold_paths.py (path enumeration + coincidental-path filter)
  train/       losses.py (BCE, soft DP, ConfTr) · sampler.py · train.py · evaluate.py
  eval/        freebase.py (official evaluation) · compare.py (multi-seed, same α / equal coverage) · plots.py · reference.py (published baselines)
tests/         164 unit tests
docs/          cwq_results.md · figures/ (used in this README)
proposal_chase.tex · theory.tex · requirements.txt
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

Biomedical graph (PrimeKG):

```bash
python -m chase.kg.build_primekg --csv data/raw/primekg/kg_giant.csv --out data/processed/primekg_full
python -m chase.model.text_cache primekg data/processed/primekg_full
python -m chase.kg.type_distance --kg data/processed/primekg_full
```

## Testing

```bash
python -m pytest tests -q        # 164 tests, ~10 s
```

The key invariants are checked against brute force: the DP's σ equals enumeration of every walk; a pruned search returns exactly the unpruned answer set; the soft DP converges to the hard DP as T → 0; split-conformal and CRC thresholds meet their guarantees in simulation; `bind()` lookup tables equal the full forward pass.

## Roadmap

- [x] Full PrimeKG graph; WebQSP / CWQ subgraph stores; MiniLM caches
- [x] State-based scorer, type feasibility, bottleneck DP, reachability ceilings
- [x] Conformal calibration (split CP, CRC)
- [x] Gold paths, sampler, two-stage training (BCE + conformal training)
- [x] Official evaluation, 1000-re-split validity check, figures: **WebQSP**, **CWQ** (3 seeds each)
- [x] Same-α and equal-coverage comparison with reported CPR / UaG; theory note
- [ ] Same-scorer baselines: per-hop (UaG-style), Bonferroni, fixed beam + post-hoc (CPR-style), conformal top-K
- [ ] Low-coverage regime: richer state (previous relation), non-empty sets
- [ ] Paired bootstrap
- [ ] BioHopR (entity linking, CRC); CPR re-run under identical conditions
- [ ] Ablations RQ6; paper

## Citation

Paper in preparation. Please contact the authors before citing.

## License

To be decided before public release.
