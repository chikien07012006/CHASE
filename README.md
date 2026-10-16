# CHASE

**Conformal Heuristic Search with Exact Coverage for Multi-hop Evidence Retrieval on Knowledge Graphs**

> **Status (16 Oct 2026):** research in progress. Data pipeline, embedding caches, the state-based scorer, the type-feasibility tables and the bottleneck DP are implemented and tested (96 tests). Reachability ceilings $\rho_H$ are measured on WebQSP and CWQ. Conformal calibration comes next. All numbers below are dataset or pipeline statistics, not experimental results.

CHASE retrieves multi-hop evidence from a knowledge graph (KG) with a **finite-sample guarantee**: the returned answer set contains a correct answer with probability at least $1-\alpha$. Pruning happens **inside the search** with **a single calibrated threshold**, and the guarantee is **tight rather than conservative**. The motivating domain is biomedicine, where missing a drug–target or drug–drug interaction is a safety failure, not just a lower score.

---

## Contents

- [Motivation](#motivation)
- [Method](#method)
- [Research questions](#research-questions)
- [Benchmarks](#benchmarks)
- [Project status](#project-status)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Reproducing the pipeline](#reproducing-the-pipeline)
- [Testing](#testing)
- [Roadmap](#roadmap)

---

## Motivation

Existing conformal KGQA methods make one of two compromises:

| Method | Where the threshold acts | Guarantee | Limitation |
|---|---|---|---|
| UaG (AAAI 2025) | Inside beam search; one threshold per hop and component | Calibration couples hops, which breaks exchangeability | Coverage can fail in practice; Freebase only |
| CPR (arXiv 2605.08077) | Post-hoc filter after a fixed-beam search (TreeG, $32\times32$) | Valid single score per question | Coverage capped by the beam's recall; one LLM call per question; Freebase only |
| **CHASE** | **Inside the search, one exchangeable score per question** | **Valid and tight:** $1-\alpha \le \text{coverage} \le 1-\alpha+\tfrac{1}{n+1}$ | Ceiling set only by what the KG can reach in $H$ hops |

## Method

1. **State-based scorer.** A question-conditioned nonconformity score $s_h(q,r,v)$ for each expansion step and $s^{\text{stop}}_h(q,v)$ for stopping (lower is more promising). Each score is a sum of a relation term and a node term, so per-question lookup tables make scoring any edge a gather plus an add.
2. **Type feasibility.** On typed KGs, $d_c(v)$, the hop distance from $v$ to the nearest node of type $c$, is precomputed. A state $(v,h)$ is kept only if an answer-typed node is still reachable in the remaining $H-h$ hops. This rule provably never removes a feasible walk.
3. **Max-prefix score.** A walk survives irreversible pruning at threshold $\tau$ iff every step scores $\le\tau$, i.e. iff $\bar s(p)=\max_i s_i\le\tau$.
4. **Bottleneck DP.** Because scores depend on the state $(v,h)$ and not on the whole path,
   $b_h(v)=\min_{u\to v}\max\{b_{h-1}(u),\,s_h(q,r,v)\}$
   gives the exact minimal surviving threshold for every node in $O(H(|V|+|E|))$. There is no beam and no hard cap.
5. **Calibration.** Each question's score $\sigma^*(q)=\min_{a\in A_q}\sigma_a(q)$ is *the smallest threshold at which the search finds an answer*. Split-conformal quantiles of $\sigma^*$ give the test-time threshold $\hat\tau$. If $1-\alpha$ exceeds what the KG can reach, CHASE reports the level as **infeasible** instead of returning a meaningless set.
6. **Multi-answer recall.** For questions with many answers, conformal risk control on per-answer bottleneck values bounds the expected miss rate.

Full method, proofs and experiment plan: [`proposal_chase.tex`](proposal_chase.tex).

## Research questions

| | Question |
|---|---|
| RQ1 | **Validity:** does empirical coverage track $1-\alpha$ over 1000 calibration/test re-splits, including small $\alpha$? |
| RQ2 | **Efficiency:** at equal $\alpha$, are CHASE's answer sets smaller and its searches cheaper than CPR, conformal top-$K$ and SubgraphRAG + conformal? |
| RQ3 | **Attainable coverage:** what is each method's coverage ceiling relative to KG reachability $\rho_H$? |
| RQ4 | **Adaptivity:** does search cost scale with question difficulty? |
| RQ5 | **Multi-answer and group coverage:** CRC on BioHopR; Mondrian by hop count and relation pattern; distribution shift (1-hop to 2-hop). |
| RQ6 | **Ablations** of every component. |

## Benchmarks

| Dataset | Searchable KG | Role | Test questions | Answer in graph (test)\* | $\rho_1$ | $\rho_2$ | $\rho_3$ | $\rho_4$ |
|---|---|---|---|---|---|---|---|---|
| WebQSP (RoG) | Freebase, one subgraph per question | Direct comparison with UaG and CPR | 1,628 | 95.64% | 66.28% | **95.33%** | 95.33% | 95.33% |
| CWQ (RoG) | Freebase, one subgraph per question | Direct comparison, 2–4 hops | 3,531 | 80.66% | 43.95% | **80.12%** | 80.12% | 80.12% |
| BioHopR | Full PrimeKG (134k nodes, 4.1M edges) | Biomedical, multi-answer (36.65 answers per question) | own split | — | — | — | — | — |

\*Share of questions whose gold answer appears anywhere in the searchable graph. $\rho_H$ is the tighter ceiling: the share of questions with a gold answer **reachable from a topic entity within $H$ hops** (walks; topic entities and CVT nodes are never answers). Both bound coverage for **every** method. Full statistics in [`DATA.md`](DATA.md).

**What the ceilings imply.**
- $\rho_H$ stops growing at $H=2$ on both datasets. RoG subgraphs are 2-hop neighbourhoods of the topic entities, so the ~20% of CWQ questions that need 3–4 hops have their answers outside the subgraph. On these subgraphs, $H>2$ adds search cost but no reachable answers.
- No method can reach $1-\alpha$ above 95.33% on WebQSP or above 80.12% on CWQ. With finite calibration sets, levels just below the ceiling are often infeasible too. In practice the feasible grid is $\alpha\ge0.1$ on WebQSP and $\alpha\ge0.25$ on CWQ.

## Project status

| Component | Proposal § | Code | Tests | Output |
|---|---|---|---|---|
| Full PrimeKG graph (CSR, no edge caps) | 5.2 | `chase/kg/graph.py`, `build_primekg.py` | manual checks (symmetry, `expand`, known facts) | `primekg_full/` (51 MB) |
| WebQSP / CWQ per-question subgraph stores | 5.2 | `chase/kg/build_freebase.py`, `subgraphs.py` | manual checks (inverse consistency, gold paths) | `webqsp/` (445 MB), `cwq/` (3.0 GB) |
| Frozen MiniLM embedding caches | 4.2 | `chase/model/text_cache.py` | — | `*/emb_minilm/` (98 MB, 419 MB, 800 MB) |
| Graph features (shared interface) | 4.2 | `chase/model/features.py` | via scorer tests | — |
| State-based scorer (step and stop heads, `bind()` fast path) | 4.2 | `chase/model/scorer.py` | `tests/test_scorer.py` (5) | — |
| Type-feasibility tables $d_c(v)$ | 4.3 | `chase/kg/type_distance.py` | `tests/test_type_distance.py` (8) | `primekg_full/type_dist.npz` (1.3 MB) |
| Threshold search and bottleneck DP (calibration $\tau=\infty$ and test $\tau=\hat\tau$ modes, evidence walks, $\rho_H$) | 4.4–4.5 | `chase/search/dp.py` | `tests/test_dp.py` (83) | $\rho_H$ for WebQSP / CWQ |
| Conformal calibration (split CP, CRC, Mondrian) | 4.6 | — | — | — |
| BioHopR linking and splits for all benchmarks | 5.2–5.3 | — | — | — |
| Gold paths, scorer training | 4.7 | — | — | — |
| Baselines | 5.4 | — | — | — |

**Pipeline measurements (Apple-silicon laptop, CPU):**

| Step | Time |
|---|---|
| Score 1M PrimeKG edges with a bound scorer | 37 ms |
| Bind one question on PrimeKG / one WebQSP question end-to-end | 9.3 ms / 7.6 ms |
| Type-distance tables for PrimeKG (10 multi-source BFS, 4 layers) | 2.3 s |
| $\rho_{1..4}$ over all test questions: WebQSP (1,628) / CWQ (3,531) | 3 s / 6 s |

**Type feasibility on PrimeKG:** share of nodes within $k$ hops of the nearest node of each type.

| Answer type | $k=0$ | $k=1$ | $k=2$ | $k=3$ | $k=4$ |
|---|---|---|---|---|---|
| drug | 5.9% | 10.8% | 44.3% | 78.0% | 89.6% |
| disease | 16.5% | 32.2% | 60.2% | 82.9% | 91.2% |
| gene/protein | 20.5% | 47.5% | 75.4% | 89.0% | 94.1% |
| effect/phenotype | 11.4% | 21.0% | 48.4% | 79.3% | 90.3% |

With $H=2$ (BioHopR), a state at hop 1 needs $d_c\le1$, which only 11–48% of nodes satisfy depending on the answer type. The rule is therefore selective at the last expansion step and loose earlier. Its effect on real search cost is measured in ablation A2.

## Repository layout

```
chase/
  kg/
    graph.py            KG class: CSR storage, vectorised frontier expansion
    build_primekg.py    full PrimeKG -> CSR
    build_freebase.py   RoG WebQSP/CWQ -> packed per-question subgraphs
    subgraphs.py        SubgraphStore: random access to per-question KGs
    type_distance.py    multi-source BFS, d_c(v) tables, feasibility mask
  model/
    text_cache.py       frozen MiniLM embeddings for nodes, relations, questions
    features.py         GraphFeatures: node / relation feature matrices per graph
    scorer.py           StateScorer (step + stop heads), EncodedGraph, BoundScorer
  search/
    dp.py               bottleneck DP, DPResult (answer sets, sigma*, evidence walks), rho_H
tests/
  test_scorer.py
  test_type_distance.py
  test_dp.py
proposal_chase.tex      method, theory and experiment plan
DATA.md                 statistics of every processed dataset
requirements.txt
```

## Setup

Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The encoder is expected at `models/all-MiniLM-L6-v2` (only `model.safetensors` and the config/tokenizer files are needed):

```bash
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2 \
  --include "*.json" "*.txt" "model.safetensors" "1_Pooling/*"
```

## Reproducing the pipeline

Raw data lives under `data/raw/` and processed data under `data/processed/`; both are git-ignored. Sources: PrimeKG `kg_giant.csv` (Harvard Dataverse), RoG-WebQSP and RoG-CWQ (Hugging Face `rmanluo/*`), BioHopR (Hugging Face `knowlab-research/BioHopR`).

```bash
# 1. graphs
python -m chase.kg.build_primekg  --csv data/raw/primekg/kg_giant.csv --out data/processed/primekg_full   # ~22 s
python -m chase.kg.build_freebase --raw data/raw/rog_webqsp --out data/processed/webqsp                   # ~40 s
python -m chase.kg.build_freebase --raw data/raw/rog_cwq    --out data/processed/cwq                      # ~15 min

# 2. frozen text embeddings
python -m chase.model.text_cache primekg  data/processed/primekg_full   # ~1 min
python -m chase.model.text_cache freebase data/processed/webqsp         # ~3 min
python -m chase.model.text_cache freebase data/processed/cwq            # ~5 min

# 3. type-feasibility tables (typed KGs only)
python -m chase.kg.type_distance --kg data/processed/primekg_full       # ~2 s

# 4. reachability ceilings rho_H (no scorer needed)
python -m chase.search.dp --data data/processed/webqsp --split test --H 1 2 3 4   # ~3 s
python -m chase.search.dp --data data/processed/cwq    --split test --H 1 2 3 4   # ~6 s
```

## Testing

```bash
python -m pytest tests -v
```

| Test file | What it checks |
|---|---|
| `test_scorer.py` | `bind()` lookup tables equal the full `forward()` pass; scores are finite and non-negative; numerically stable softplus; gradients reach both heads |
| `test_type_distance.py` | multi-source BFS matches networkx; the feasibility rule never drops a walk that reaches the answer type within $H$ hops (Lemma 1); error on out-of-range queries; save/load round trip |
| `test_dp.py` | $\sigma_v$ equals brute-force enumeration of every walk (with ties, answer masks and feasibility masks); a pruned run returns exactly the unpruned answer set; answer sets are nested in $\tau$; evidence walks are real walks whose score equals $\sigma_v$; topic entities are never answers; the zero scorer reduces to reachability |

## Roadmap

- [x] Literature review and positioning against UaG / CPR
- [x] Full PrimeKG graph (CSR, no edge caps)
- [x] WebQSP / CWQ per-question subgraph stores
- [x] Frozen MiniLM embedding caches for all graphs
- [x] State-based scorer with tested fast path
- [x] Type-feasibility tables for PrimeKG
- [x] Bottleneck DP with brute-force equivalence tests
- [x] Reachability ceilings $\rho_H$ for WebQSP and CWQ
- [ ] Conformal calibration (split CP, CRC, Mondrian) with synthetic coverage tests
- [ ] Reachability ceiling $\rho_H$ for BioHopR
- [ ] BioHopR entity linking; calibration splits for all benchmarks
- [ ] Gold paths; BCE and conformal training of the scorer
- [ ] Baselines: CPR (official code), UaG-style per-hop, Bonferroni, fixed-beam + post-hoc, conformal top-$K$, SubgraphRAG + CP
- [ ] Experiments RQ1–RQ6; paper

## Citation

Paper in preparation. Please contact the authors before citing.

## License

To be decided before public release.
