# Data statistics

All numbers were computed from the processed data in `data/processed/` (git-ignored) on 13 Oct 2026. To rebuild, see the README.

## Overview

| Dataset | KG searched | Train | Val | Test | Answer in graph (test) | Status |
|---|---|---|---|---|---|---|
| WebQSP (RoG) | per-question Freebase subgraph | 2,826 | 246 | 1,628 | 95.64% | processed |
| CWQ (RoG) | per-question Freebase subgraph | 27,639 | 3,519 | 3,531 | 80.66% | processed |
| BioHopR | full PrimeKG | — | — | — (single split, 7,633 rows) | — | raw, validated against our KG |
| Second biomedical benchmark | PrimeKG | — | — | — | — | to be selected |

"Answer in graph" is the share of questions where at least one gold answer appears in the searchable graph. It is an upper bound on coverage for **every** method. The hop-limited ceiling $\rho_H$ is lower still and will be computed by the DP.

---

## PrimeKG (full, uncapped)

Source: `kg_giant.csv`, 8,196,862 rows. Output: `data/processed/primekg_full/` (51 MB).

| Quantity | Value |
|---|---|
| Nodes | 134,437 (identified by `(type, source, id)`; no isolated nodes) |
| Undirected edges | 4,098,431 (stored both ways: 8,196,862 directed) |
| Relations | 33 = 30 relation types, with `drug_protein` split into target / enzyme / transporter / carrier |
| Node types | 10 |
| Nodes with name aliases | 63 (e.g. GOLM2 ≡ CASC4) |
| Degree | mean 61.0, median 4, p90 132, p99 969, max 17,369 |
| Highest-degree nodes | anatomy hubs: multi-cellular organism (17,369), small intestine (16,855), testis (16,743) |
| Build | 22 s, ~5 GB peak RAM |

The CSV lists every edge in both directions (verified: 100% of rows have a reverse). The node count is higher than the 129,375 in the PrimeKG paper because this is a newer Dataverse release.

**Nodes by type**

| Type | Nodes | Type | Nodes |
|---|---|---|---|
| biological_process | 28,642 | anatomy | 14,033 |
| gene/protein | 27,610 | molecular_function | 11,169 |
| disease | 22,205 | drug | 7,957 |
| effect/phenotype | 15,311 | cellular_component | 4,176 |
| pathway | 2,516 | exposure | 818 |

**Largest relations (undirected edges)**

| Relation | Edges | Relation | Edges |
|---|---|---|---|
| anatomy_protein_present | 1,518,203 | disease_protein | 92,580 |
| drug_drug | 1,336,314 | cellcomp_protein | 83,402 |
| protein_protein | 321,075 | molfunc_protein | 69,530 |
| disease_phenotype_positive | 176,383 | drug_effect | 64,784 |
| bioprocess_protein | 144,805 | contraindication | 33,984 |

Two relations hold 70% of all edges, and anatomy hubs have degree above 16k. A 2-hop expansion from a single common disease (asthma) touches 211,769 edges. This is why search is state-based, not path-enumerating.

**Do not use** `data/processed/primekg_graph/` (legacy). It keeps only 6/10 node types and 18/30 relations, and caps `drug_drug` and `anatomy_protein_present` at 500k edges each.

---

## WebQSP and CWQ (RoG release)

Output: `data/processed/{webqsp,cwq}/<split>/` with `csr.npz` (packed per-question CSR), `questions.parquet`, `node_names.parquet`; plus `relations.json` shared across splits.

| | WebQSP train | WebQSP val | WebQSP test | CWQ train | CWQ val | CWQ test |
|---|---|---|---|---|---|---|
| Questions | 2,826 | 246 | 1,628 | 27,639 | 3,519 | 3,531 |
| Answer in subgraph | 96.07% | 95.53% | **95.64%** | 79.92% | 80.93% | **80.66%** |
| Answer missing (count) | 111 | 11 | 71 | 5,550 | 671 | 683 |
| Empty subgraph / topic missing | 0 / 0 | 1 / 1 | 0 / 0 | 8 / 8 | 0 / 0 | 0 / 0 |
| Nodes per subgraph (median) | 1,447 | 1,445 | 1,530 | 1,413 | 1,481 | 1,481 |
| Directed edges per subgraph (median) | 8,124 | 8,283 | 8,700 | 7,232 | 7,784 | 8,020 |
| Directed edges total | 23.1M | 2.0M | 13.6M | 212.3M | 28.3M | 29.2M |

| | WebQSP | CWQ |
|---|---|---|
| Base relations (directed with inverses) | 6,094 (12,188) | 6,649 (13,298) |
| Max hops $H$ | 2 | 4 |
| Processed size | 445 MB | 3.0 GB |
| Build time | ~40 s | ~15 min, 5.2 GB peak RAM |

- Every question has gold answers (`a_entity` is never empty). The "missing" questions have answers that RoG's subgraph extraction did not include. A case-insensitive name match recovers none of them.
- These questions stay in calibration and test, with $\sigma^*=+\infty$. Removing them would inflate coverage. Consequence: on CWQ, no method searching these subgraphs can reach $1-\alpha>0.807$.
- CVT nodes (`m.*`, `g.*`) can be traversed but are never answers.
- CWQ split sizes match CPR's (24,875 train + 2,764 calibration = 27,639).

---

## BioHopR (Findings of ACL 2025; MIT)

| Quantity | Value |
|---|---|
| Rows | 7,633 (each row = one 2-hop question; 2,494 unique 1-hop questions) |
| Answers per question | mean 36.65, median 11, max 890 |
| Identifiers | entity **names only**, so name → PrimeKG node linking is required (aliases help) |
| Topic entity mapped by (name, type) | 7,599 / 7,633 (99.55%) |
| Answers mapped by (name, type) | 275,051 / 279,738 (98.32%) |
| Mapped answers reachable from the topic via the stated type pattern in 2 hops | 96.90% (400 sampled questions) |
| Split | none provided; planned random 50/10/15/25 (train/val/cal/test), seed 0 |

The reachability check confirms that BioHopR was built on a PrimeKG version compatible with our graph.

2-hop relation patterns:

| Pattern | Questions | Pattern | Questions |
|---|---|---|---|
| drug → gene/protein → disease | 3,029 | disease → drug → gene/protein | 381 |
| disease → drug → effect/phenotype | 949 | drug → disease → gene/protein | 321 |
| disease → gene/protein → drug | 899 | effect/phenotype → drug → disease | 215 |
| gene/protein → disease → drug | 577 | drug → disease → effect/phenotype | 213 |
| effect/phenotype → disease → drug | 546 | disease → effect/phenotype → drug | 36 |
| gene/protein → drug → disease | 462 | drug → effect/phenotype → disease | 5 |

---

## Planned split protocol

| Dataset | Train (scorer) | Calibration | Validation | Test |
|---|---|---|---|---|
| WebQSP | 2,543 | 283 (10% of train, as in CPR) | 246 (RoG val) | 1,628 |
| CWQ | 24,875 | 2,764 (10% of train) | 3,519 | 3,531 |
| BioHopR | 50% | 15% | 10% | 25% |

Calibration and test always come from the same dataset. This is required for the coverage guarantee. Validity checks also re-split calibration ∪ test 1000 times.

---

## Dropped: PrimeKGQA (ECAI 2024)

Evaluated on 13 Oct 2026 and removed. **Do not use.**

- Its node ids are internal to the authors' SPARQL endpoint. They match neither PrimeKG's `nodes.csv` (0 / 652 sampled ids) nor any node ordering of `kg_giant.csv`.
- Mapping its entity names to PrimeKG works (98.9%), but only **4.5%** of the facts its questions are built from (1,762 / 38,870 mapped test triples) exist in PrimeKG. Example: "statoconial membrane *interacts with* VPS11". In PrimeKG, statoconial membrane is an anatomy node with a single `anatomy_anatomy` edge. The id → name table used to generate the questions appears to be misaligned, so the natural-language questions describe facts that are not in the KG.
- 65% of its questions ask for a relation, not an entity.

PrimeKG's `nodes.csv` was also dropped. It groups diseases (17,080 disease nodes) while `kg_giant.csv` does not (22,205), so the two files are different releases.
