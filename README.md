<div align="center">

# CHASE

### Conformal Heuristic Search with Exact Coverage<br>for Multi-hop Evidence Retrieval on Knowledge Graphs

*Prune inside the search with one calibrated threshold, and return an answer set that contains a correct answer with probability ≥ 1 − α: valid, tight, and LLM-free.*

![status](https://img.shields.io/badge/status-research%20in%20progress-2a78d6)
![python](https://img.shields.io/badge/python-3.11-52514e)
![tests](https://img.shields.io/badge/tests-148%20passing-1baf7a)
![llm](https://img.shields.io/badge/LLM%20calls-0-52514e)

</div>

> **Status (17 Oct 2026).** The full pipeline (graphs → scorer → bottleneck DP → conformal calibration → two-stage training → official evaluation) is implemented and tested. First results on **WebQSP** are below (single seed). CWQ, BioHopR, same-scorer baselines and multi-seed runs are in progress.

---

## Highlights

- **Exact coverage.** Over 1000 random calibration/test re-splits, empirical coverage matches the split-conformal theory to three decimals at every α (e.g. 0.902 vs 0.901 at α = 0.1). CPR over-covers by 7–31 points, i.e. it is conservative.
- **Reaches the high-coverage regime.** CHASE certifies α = 0.1 and 0.2 on WebQSP (ECR 87.9% / 77.0%). Neither CPR nor UaG reports these levels.
- **Smaller sets where it matters.** At matched coverage ≥ 65%, CHASE returns **16–29% fewer answers than CPR** and **3–11× fewer than UaG**.
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

Full method, proofs and experiment plan: [`proposal_chase.tex`](proposal_chase.tex). Weekly report with the full analysis: [`reports/2026-10-17_weekly_report.md`](reports/2026-10-17_weekly_report.md).

## Results

**WebQSP**, official split (283 calibration / 1,628 test, same sizes as CPR), H = 2, single seed. ECR (%) / APSS (average number of answer entities). CPR and UaG numbers are as reported in the CPR paper; "—" = not reported.

| Method | α=0.05 | α=0.1 | α=0.2 | α=0.3 | α=0.4 | α=0.5 | α=0.6 | α=0.7 | α=0.8 |
|---|---|---|---|---|---|---|---|---|---|
| target 1−α | 95 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 |
| UaG | — | — | — | 72.3 / 98.81 | 60.4 / 32.52 | 51.8 / 14.29 | 41.8 / 8.23 | 37.4 / 5.58 | 25.6 / 1.67 |
| CPR | — | — | — | 76.8 / 17.77 | 67.3 / 10.10 | 61.4 / 7.14 | 58.3 / 5.45 | 55.1 / 3.01 | 50.8 / 1.62 |
| **CHASE** | 91.4 / 62.97† | **87.9 / 31.67** | **77.0 / 12.77** | 68.4 / 8.79 | 57.6 / 6.01 | 50.3 / 4.63 | 43.7 / 3.57 | 33.2 / 2.56 | 19.9 / 1.32 |

† α = 0.05 is at the reachability ceiling (ρ₂ = 95.33%); 40% of re-splits return τ̂ = ∞.

**Matched coverage.** CPR over-covers, so comparing at equal α favours it. Comparing at equal ECR, with CHASE interpolated:

| CPR ECR | 76.8% | 67.3% | 61.4% | 58.3% | 55.1% | 50.8% |
|---|---|---|---|---|---|---|
| CPR APSS | 17.77 | 10.10 | 7.14 | 5.45 | 3.01 | 1.62 |
| CHASE APSS | **12.7** | **8.5** | 6.9 | 6.2 | 5.5 | 4.7 |

CHASE wins in the high-coverage regime. CPR wins below ~60% coverage, where a single global threshold leaves many CHASE sets empty (33% at α = 0.5) while others stay large.

<p align="center">
  <img src="docs/figures/fig3_apss_efficiency.png" width="90%" alt="APSS and coverage efficiency vs alpha">
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

ρ_H is the share of questions with a gold answer reachable within H hops: an upper bound on coverage for **every** method. It stops growing at H = 2, because RoG subgraphs are 2-hop neighbourhoods. Full statistics: [`DATA.md`](DATA.md).

## Repository layout

```
chase/
  kg/          graph.py (CSR KG) · build_primekg.py · build_freebase.py · subgraphs.py · type_distance.py
  model/       text_cache.py (frozen MiniLM) · features.py · scorer.py (StateScorer, BoundScorer, ScoreTables)
  search/      dp.py (bottleneck DP, evidence walks, rho_H)
  conformal/   split.py (split CP, CRC) · calibrate.py (calibration scores from the DP)
  data/        splits.py · gold_paths.py (path enumeration + coincidental-path filter)
  train/       losses.py (BCE, soft DP, ConfTr) · sampler.py · train.py · evaluate.py
  eval/        freebase.py (official evaluation) · plots.py · reference.py (published baselines)
tests/         148 unit tests
docs/figures/  figures used in this README
reports/       weekly reports
proposal_chase.tex · DATA.md · requirements.txt
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

WebQSP end to end (CWQ: replace `webqsp` with `cwq`). Times are for an Apple-silicon laptop, CPU only.

```bash
D=data/processed/webqsp
python -m chase.kg.build_freebase --raw data/raw/rog_webqsp --out $D          # subgraphs          ~40 s
python -m chase.model.text_cache freebase $D                                  # MiniLM cache       ~3 min
python -m chase.data.splits --data $D                                         # train/cal/val/test
python -m chase.data.gold_paths --data $D --split train --H 2                 # training labels    ~6 s
python -m chase.train.train --data $D --out runs/webqsp_seed0                 # two-stage training ~10 min
python -m chase.eval.freebase --data $D --model runs/webqsp_seed0/best.pt \
                              --out runs/webqsp_seed0/eval                    # evaluation         ~2 min
```

The evaluation writes `results_table.md`, `results.csv`, `metrics.json`, `validity.json` (1000 re-splits), `per_question.parquet` and six figures (PNG + PDF) to the `--out` folder.

Biomedical graph (PrimeKG):

```bash
python -m chase.kg.build_primekg --csv data/raw/primekg/kg_giant.csv --out data/processed/primekg_full
python -m chase.model.text_cache primekg data/processed/primekg_full
python -m chase.kg.type_distance --kg data/processed/primekg_full
```

## Testing

```bash
python -m pytest tests -q        # 148 tests, ~10 s
```

The key invariants are checked against brute force: the DP's σ equals enumeration of every walk; a pruned search returns exactly the unpruned answer set; the soft DP converges to the hard DP as T → 0; split-conformal and CRC thresholds meet their guarantees in simulation; `bind()` lookup tables equal the full forward pass.

## Roadmap

- [x] Full PrimeKG graph; WebQSP / CWQ subgraph stores; MiniLM caches
- [x] State-based scorer, type feasibility, bottleneck DP, reachability ceilings
- [x] Conformal calibration (split CP, CRC)
- [x] Gold paths, sampler, two-stage training (BCE + conformal training)
- [x] Official evaluation, 1000-re-split validity check, figures: **WebQSP**
- [ ] Same-scorer baselines: per-hop (UaG-style), Bonferroni, fixed beam + post-hoc (CPR-style), conformal top-K
- [ ] Low-coverage regime: richer state (previous relation), non-empty sets
- [ ] Five seeds, paired bootstrap; CWQ
- [ ] BioHopR (entity linking, CRC); CPR re-run under identical conditions
- [ ] Ablations RQ6; paper

## Citation

Paper in preparation. Please contact the authors before citing.

## License

To be decided before public release.
