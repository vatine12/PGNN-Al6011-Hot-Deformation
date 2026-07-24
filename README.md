# PGNN-Al6011-Hot-Deformation

Physics-guided and hybrid physics/data neural networks for predicting the hot-deformation flow stress of **Al 6011-O** aluminium, with an honest focus on **generalization to unseen deformation conditions**.

## TL;DR

- A **physics-guided neural network (PGNN)** that outputs Arrhenius parameters (α, n, Q, lnA) and computes stress through the sinh-Arrhenius law is accurate *and* physically interpretable (learned activation energy Q ≈ 180 kJ/mol, matching literature).
- We show that once the physics is embedded in the *architecture*, **adding physics-based loss terms is largely redundant** — the sinh-Arrhenius form already guarantees the monotonic behaviour those losses try to enforce.
- Pure PGNN is excellent in the **hot-working regime (250–450 °C)** but **structurally fails at RT/150 °C**, where dynamic strain aging (DSA) makes stress *decrease* with strain rate — behaviour the Arrhenius form cannot represent.
- Our fix is a **regime-gated physics/data hybrid**: `σ = g·σ_physics + (1−g)·σ_data`, where a learned gate `g` decides where to trust the physics. It matches the physics model where the law holds, matches a black-box network where it breaks, and the **gate recovers the Arrhenius-validity boundary directly from data**.
- Everything is evaluated with **leave-one-condition-out / leave-one-temperature-out** cross-validation (honest extrapolation) plus multi-seed paired statistics — not the optimistic random split.

## Why this problem

In hot forming (forging, rolling, extrusion) engineers need the flow stress σ as a function of temperature *T*, strain rate ε̇, and strain ε. Classical constitutive models (Sellars–McTegart / strain-compensated Arrhenius, "SCAM") are physically interpretable but rigid and only valid in a limited window. Black-box ML is accurate in-distribution but gives no physical insight and extrapolates poorly. Physics-guided ML aims for the best of both — but *how* you inject physics (in the architecture vs in the loss) turns out to matter enormously, and is the core question of this repo.

## Method

**PGNN (physics in the architecture).** The network never outputs stress directly. It outputs four Arrhenius parameters, bounded to physical ranges by sigmoids, and a fixed differentiable Arrhenius layer computes σ:

```
(T, ln ε̇, ε) → MLP backbone → 4 heads → (α, n, Q, lnA)
             → σ = (1/α)·arcsinh[ (ε̇·exp(Q/RT) / exp(lnA))^(1/n) ]
```

**Gated hybrid (physics where valid, data where not).** Two experts — the PGNN physics expert and a free-form data expert — blended by a learned gate:

```
σ = g(T, ε̇, ε) · σ_physics  +  (1 − g) · σ_data,        g ∈ [0, 1]
```

A **two-sided gate prior** encourages `g → 1` in the hot regime (trust physics) and `g → 0` at RT/150 °C (trust data, where DSA breaks Arrhenius). The gate is not hand-set — it is *learned*, and it ends up tracking the physical validity boundary.

## Honest evaluation

The original random train/test split shuffles points *within* each stress–strain curve, so test points sit between neighbouring training points on the same curve — that is interpolation, and it inflates R² to ~0.98. We instead report:

- **LOCO** (leave-one-condition-out): hold out an entire (T, ε̇) curve, train on the other 20, predict it.
- **LOTO** (leave-one-temperature-out): hold out *all* rates at one temperature — the hardest extrapolation.
- **Random split** kept only as an interpolation reference.

All comparisons use up to 8 random seeds with **paired** differences and t-tests, because condition-to-condition variance is large.

## Key results

**LOCO (all 21 conditions held out, 8 seeds), paired Δ AARE vs pure PGNN:** hybrid **+1.13 pp (p = 0.009)**, hybrid + gate-prior **+1.48 pp (p = 0.001)**; a plain ANN *hurts* (−1.89 pp).

**Error by physical regime (AARE %, lower is better):**

| Model | Hot 250–450 °C | DSA (RT/150 °C) | LOTO | Random (interp.) |
|---|---|---|---|---|
| PGNN (physics only) | **12.2** | 35.1 | 32.1 | 7.2 |
| ANN (data only) | 18.1 | 26.8 | 32.8 | 11.2 |
| Gated hybrid | 12.0 | 30.4 | 28.1 | 7.4 |
| **Hybrid + two-sided gate prior** | 12.5 | **26.8** | **24.4** | **6.9** |

The story: **PGNN wins in the hot regime and fails at DSA; the ANN is the opposite; the gated hybrid gets both.** The two-sided hybrid matches the black box on DSA while nearly matching PGNN on hot — neither pure model can do both.

**The gate learns the physics-validity boundary.** Mean learned gate per condition rises from ~0.11 at room temperature and ~0.25 at 150 °C (mostly data expert) to ~0.66 at 250 °C and ~0.98 at 450 °C (mostly physics expert) — recovering the DSA ↔ Arrhenius transition from data alone.

**Physical interpretability retained.** The physics expert's learned activation energy Q ≈ 180 kJ/mol sits squarely in the literature band (150–185 kJ/mol; Al self-diffusion ≈ 142 kJ/mol).

## What we learned

1. **Where you put the physics matters.** Embedding Arrhenius in the architecture drives the main gain (ANN → PGNN). Adding SCAM-matching / consistency / smoothness / monotonicity *losses* is largely redundant — the architecture already enforces that physics (monotonicity is in fact structurally guaranteed by the sinh-Arrhenius form).
2. **A rigid physics model has a hard limit.** Pure PGNN cannot represent DSA (negative strain-rate sensitivity) at RT/150 °C. No loss term can fix this; only a more flexible architecture can.
3. **A regime-gated hybrid is the fix**, and its gate is an interpretable, data-driven map of *where the constitutive law is valid*.

## Repository structure

```
├── README.md                                  # this file
├── RESULTS.md                                 # detailed results & analysis
├── PROJECT_KNOWLEDGE_BASE.md                  # compact context for continuation
├── al6011_downsampled_full.xlsx               # dataset (21 conditions, 1,982 pts)
├── al6011_data_summary.xlsx                   # summary statistics
├── hot-tensil-pgnn-physics-pipeline.ipynb     # LOCO/LOTO/ablation harness (physics losses)
├── hot-tensil-pgnn-stepB-mono.ipynb           # baseline vs monotonicity, multi-seed
├── hot-tensil-pgnn-stepB2-mono-masked.ipynb   # masked mono + per-condition table
├── Code/
│   ├── hot-tensil-pgnn-v15-gated-hybrid-result.ipynb    # gated hybrid (main result)
│   ├── hot-tensil-pgnn-v16-twosided-gate-result.ipynb   # two-sided gate prior (best)
│   └── hot-tensil-pgnn-v12-masked-mono-result.ipynb     # physics-loss redundancy study
├── results/                                   # LOCO/LOTO/random/per-condition CSVs
└── Futher_analysis_results/                   # earlier figures & metrics
```

## How to run

1. Upload the dataset `al6011_downsampled_full.xlsx` and a notebook (start with `Code/hot-tensil-pgnn-v16-twosided-gate-result.ipynb`) to [Kaggle](https://www.kaggle.com/).
2. Enable the **GPU accelerator**; set `CFG.data_path` to your dataset location (a folder path is fine — the loader finds the file).
3. Use `CFG.quick = True` for a fast smoke run, then `CFG.quick = False` for the full multi-seed study.
4. Run all cells — the notebook produces the LOCO/LOTO tables, per-condition metrics, the gate map, and SCAM parameter validation, saving CSVs to the working directory.

## Requirements

Python 3.8+, PyTorch, NumPy, Pandas, Matplotlib, scikit-learn, SciPy, openpyxl.

## Status

Research in progress. Current results support a regime-gated physics/data hybrid with an interpretable validity gate; see `RESULTS.md` and `PROJECT_KNOWLEDGE_BASE.md` for the full picture and open items.

## Source code

The complete, self-contained pipelines are the two notebooks in `Code/`:
- `hot-tensil-pgnn-v15-gated-hybrid-result.ipynb` — PGNN, ANN, gated hybrid, and one-sided gate prior.
- `hot-tensil-pgnn-v16-twosided-gate-result.ipynb` — the two-sided gate prior (best model).

Each defines the config, models (`HybridPGNN`, `FreeNet`, `GatedHybrid`), training engine, LOCO/LOTO/random splits, and evaluation on its own. The other notebooks are earlier exploratory studies (physics-loss ablations) and are supplementary.

**Reproduction order:** run **v15 first** — it trains the four base models (`pgnn`, `ann`, `hybrid`, `hybrid+gp`) and saves the LOCO/LOTO/per-condition CSVs. Then run **v16** in the same environment — it trains `hybrid+gp2` (two-sided gate prior) and merges with v15's CSVs to build the combined comparison table. Each notebook trains and saves its own models independently, so both are needed for the full table.

## Authors

- **Nguyen Tran Quang Minh** — Keio University, Japan
- **Tran Ngoc Dung** — Dalian University of Technology (DUT), China
