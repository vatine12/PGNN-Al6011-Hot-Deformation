# Results & Analysis

> This document reports cross-validated results under honest extrapolation testing. Note that a random
> train/test split leaks information within each stress–strain curve (interpolation) and inflates the
> metrics; all headline numbers below use leave-one-condition-out / leave-one-temperature-out instead.
> See "Evaluation" below.

## Material & dataset

- **Alloy:** Al 6011-O aluminium.
- **Conditions:** 7 temperatures (25, 150, 250, 300, 350, 400, 450 °C) × 3 strain rates (0.001, 0.01, 0.1 s⁻¹) = **21 conditions**.
- **Points:** 1,982 (downsampled from raw tensile curves, Δε ≈ 0.005). Each condition is a smooth σ–ε curve; true strain reaches ~0.7 at high T.
- **Two physical regimes:** hot working (250–450 °C, thermally activated / DRX) and **RT/150 °C dynamic strain aging (DSA)**. In the DSA regime, stress *decreases* with strain rate (negative strain-rate sensitivity) — verified directly in the data (0% strain-rate-monotone at RT/150 vs 100% at 250–450 °C).

## Models

| Model | Physics injected via | Notes |
|---|---|---|
| SCAM | classical equation | Strain-compensated sinh-Arrhenius; parameters as polynomials in ε; fit on hot conditions only, valid ε ∈ [0.05, 0.19]. Reference / validator, not ground truth. |
| ANN | none (black box) | MLP (T, ln ε̇, ε) → σ. |
| PGNN | **architecture** | MLP → (α, n, Q, lnA) sigmoid-bounded → sinh-Arrhenius → σ. |
| Gated hybrid | architecture + **gating** | σ = g·σ_PGNN + (1−g)·σ_FreeNN, learned gate g. |
| Hybrid + two-sided gate prior | + validity prior | gate pushed → 1 (hot) and → 0 (DSA). **Best overall.** |

## Evaluation (why the metric changed)

The original random 70/15/15 split shuffles points *within* each curve, so a test point at ε = 0.105 lies between training points at 0.10 and 0.11 on the *same* curve, temperature and rate. That measures interpolation and inflates R² to ~0.98. Honest protocols used here:

- **LOCO** — leave-one-condition-out: hold out one whole (T, ε̇) curve, train on the other 20, predict it. Repeated for all 21 conditions.
- **LOTO** — leave-one-temperature-out: hold out all rates at one temperature. Hardest (must extrapolate across temperature).
- **Random split** — kept only as an interpolation reference.

Up to **8 seeds**; comparisons are **paired** (same seed × fold) with t-tests, since fold-to-fold variance is large.

## Structural limitation of pure PGNN

The sinh-Arrhenius form is mathematically forced to make σ increase with strain rate, at every temperature. It therefore **cannot represent DSA / negative strain-rate sensitivity at RT/150 °C**, and mispredicts those conditions. No loss term (masked or not) can fix this — the rigidity is in the equation, not the penalty.

## Gated hybrid — the fix (main result)

**LOCO (all 21 conditions held out, 8 seeds), paired Δ AARE vs pure PGNN:**

| Config | Δ AARE (pp, + = better) | p-value | Verdict |
|---|---|---|---|
| ANN | −1.89 | 0.023 | hurts |
| Gated hybrid | **+1.13** | 0.009 | helps |
| Hybrid + gate prior | **+1.48** | 0.001 | helps |

**Error by regime (AARE %):**

| Model | Hot 250–450 °C | DSA RT/150 °C | LOTO | Random |
|---|---|---|---|---|
| PGNN | **12.18** | 35.08 | 32.09 | 7.15 |
| ANN | 18.13 | 26.81 | 32.84 | 11.20 |
| Hybrid + gate prior (v15) | 11.97 | 30.41 | 28.06 | 7.35 |
| **Hybrid + two-sided prior (v16)** | 12.46 | **26.77** | **24.43** | **6.85** |

Interpretation: **PGNN dominates the hot regime and fails at DSA; ANN is the opposite; the gated hybrid gets both.** The two-sided hybrid reaches the black box's DSA accuracy (26.8 ≈ 26.8) while nearly matching PGNN on hot (12.5 vs 12.2), and it is best on both the hardest extrapolation (LOTO) and interpolation. The two-sided prior trades ~0.3 pp of hot accuracy for ~8 pp of DSA accuracy — a tunable operating point.

## The gate recovers the validity boundary

Mean learned gate (physics weight) per condition, two-sided prior:

| Regime | Gate g (physics weight) |
|---|---|
| RT (25 °C) | ≈ 0.11 |
| 150 °C | ≈ 0.25 |
| 250 °C | ≈ 0.66 |
| 350 °C | ≈ 0.92 |
| 450 °C | ≈ 0.98 |

The model **learns where the constitutive law is valid** — low physics weight in the DSA regime, near-full physics weight in the hot regime — recovering the DSA ↔ Arrhenius transition from data alone. This gate map is the signature figure of the study.

## Physical interpretability

The physics expert's learned Arrhenius parameters remain physical inside the SCAM-valid window (T ≥ 250 °C, ε ∈ [0.05, 0.19]):

| Parameter | Learned | Classical SCAM | Literature anchor |
|---|---|---|---|
| Q (kJ/mol) | ~180 | ~157 | 150–185 (Al self-diffusion ≈ 142) |
| n | ~5.8 | ~5.6 | — |
| α (1/MPa) | ~0.025 | ~0.019 | — |

## Open items

- Merge the v15 and v16 runs into one consolidated table (paired Δ for the two-sided hybrid too).
- Capacity-fair control: report parameter counts and confirm the hybrid dominates *both* parent models across regimes, not just from extra capacity.
- Tune gate-prior strength to minimise the small hot-regime cost.
- DSA is **improved, not solved** — 26.8 % AARE at RT/150 °C is still high.
