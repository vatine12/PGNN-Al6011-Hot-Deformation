# Project Knowledge Base — Physics-Guided ML for Al 6011-O Hot-Deformation Flow Stress

*Compact context for continuing this research. Updated after the gated-hybrid runs (v15/v16).*

## 1. Goal & who
- **Aim:** predict flow stress σ of **Al 6011-O** in hot tensile deformation as a function of T, strain rate ε̇, and true strain ε, using physics-guided ML.
- **Approach:** improve on a plain physics-guided network with a regime-gated physics/data hybrid that stays accurate across all deformation conditions, including where the Arrhenius law breaks. Python/PyTorch; Kaggle T4 GPU.

## 2. Data
- `al6011_downsampled_full.xlsx`, one sheet per condition. Columns: `T_C, T_K, strain_rate, ln_sr, eps_true, sigma_true`.
- **21 conditions = 7 T × 3 ε̇.** T = 25(RT),150,250,300,350,400,450 °C; ε̇ = 0.001,0.01,0.1 s⁻¹; ~1,982 points. Each condition is a smooth σ–ε curve (Δε≈0.005), ε up to ~0.7 at high T.
- **Two regimes:** hot working (250–450 °C, Arrhenius/DRX) and **RT/150 °C = DSA/PLC** (dynamic strain aging), qualitatively different — σ *decreases* with ε̇ there (verified: 0% strain-rate-monotone vs 100% at 250–450).

## 3. Models
- **SCAM** — classical strain-compensated sinh-Arrhenius; α,n,Q,lnA as polynomials in ε; fit on hot conditions only, valid ε∈[0.05,0.19]. Used as reference/parameter validator, not ground truth.
- **ANN** — black-box MLP baseline.
- **PGNN** — MLP → α,n,Q,lnA (sigmoid-bounded) → sinh-Arrhenius → σ. Physics embedded in architecture.
- **Gated hybrid (v15/v16, current best):** σ = g(T,ε̇,ε)·σ_PGNN + (1−g)·σ_FreeNN. A physics expert (PGNN) + a data expert (free MLP) blended by a learned gate g∈[0,1]. Variants: `hybrid` (free gate), `hybrid+gp` (one-sided gate prior→physics in hot), `hybrid+gp2` (**two-sided prior**: gate→0 at RT/150, →1 at hot).

## 4. Evaluation methodology
- **Leakage fix:** random within-curve split = interpolation (inflated R²≈0.97). Honest protocols:
  - **LOCO** (leave-one-condition-out): hold out one whole curve; here **all 21** conditions.
  - **LOTO** (leave-one-temperature-out): hold out all rates at one T (hardest).
  - Random split kept as interpolation reference.
- **Stats:** up to 8 seeds, **paired Δ** (same seed×fold) + t-test. Effect sizes small vs large fold variance → paired analysis essential.

## 5. Findings
### 5a. Structural limit of pure PGNN
- The sinh-Arrhenius form guarantees σ↑ with ε̇ and σ↓ with T, so pure PGNN **cannot represent DSA / negative strain-rate sensitivity at RT/150 °C** → those conditions are mispredicted. This motivates the gated hybrid.

### 5b. Gated hybrid fixes it (v15/v16 — the main result)
LOCO (all 21 folds, 8 seeds), paired Δ AARE vs PGNN: **hybrid +1.13 pp (p=0.009), hybrid+gp +1.48 pp (p=0.001)**; ANN −1.89 pp (hurts). Regime breakdown (AARE %):

| | HOT 250–450 | DSA RT/150 | LOTO | Random |
|---|---|---|---|---|
| PGNN | **12.18** | 35.08 | 32.09 | 7.15 |
| ANN | 18.13 | 26.81 | 32.84 | 11.20 |
| hybrid+gp (v15) | 11.97 | 30.41 | 28.06 | 7.35 |
| **hybrid+gp2 (v16)** | 12.46 | **26.77** | **24.43** | **6.85** |

- **PGNN excels in hot, fails at DSA; ANN is the opposite; the gated hybrid gets both.** hybrid+gp2 matches ANN on DSA (26.8≈26.8) while ~matching PGNN on hot (12.46 vs 12.18), best interpolation (6.85) and best LOTO (24.43).
- **Signature result — the gate learns the Arrhenius-validity boundary from data.** v16 gate: RT≈0.11, 150≈0.25 (→ data expert), 250≈0.66 → 450≈0.98 (→ physics expert). It recovers the DSA↔Arrhenius transition.
- Physics expert stays interpretable: **Q ≈ 180 kJ/mol** (lit. 150–185; Al self-diffusion ~142).
- Trade-off: two-sided prior gives up ~0.3 pp on hot to gain ~8 pp on DSA — tunable via prior strength.

## 6. Summary of current state
Pure PGNN encodes hot-deformation physics rigidly and structurally fails in the DSA regime (RT/150). The **regime-gated physics/data hybrid** matches the physics model where the constitutive law holds, matches a black box where it breaks, and learns the law's domain of validity as an interpretable gate — improving LOCO/LOTO extrapolation under multi-seed evaluation while keeping physical parameters (Q ≈ 180 kJ/mol).

## 7. Open items / next steps
- Merge v15+v16 into one consolidated results table (gp2 wasn't in v15's table) + paired Δ for gp2.
- Capacity-fair control: show hybrid dominates *both* parent models across regimes (not just more parameters).
- Confirm PGNN vs hybrid used identical data-loss setup (only architecture differs).
- Tune gate-prior strength to minimize the small hot-regime cost.
- DSA is *improved, not solved* (26.8% AARE still high).

## 8. Artifacts (repo)
- Source (self-contained): `Code/hot-tensil-pgnn-v15-gated-hybrid-result.ipynb` (PGNN, ANN, hybrid, gate prior) and `Code/hot-tensil-pgnn-v16-twosided-gate-result.ipynb` (two-sided gate, best). Run v15 first, then v16.
- Baselines: `Code/hot-tensile-ann.ipynb`, `Code/hot-tensile-scam.ipynb`.
- Data: `al6011_downsampled_full.xlsx` (21 sheets). Raw curves `tensile_data*.xlsx` gitignored (large).
- Config knobs: `CFG.quick`, `CFG.seeds`, `scam_temp_min_K=523`, `scam_eps_max=0.19`.
