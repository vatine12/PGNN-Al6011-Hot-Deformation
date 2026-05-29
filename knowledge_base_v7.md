# Knowledge Base v7 — Al 6011-O Flow Stress PGNN

**Version:** 7.0 | **Updated:** 14 May 2026

---

## 1. PROJECT IDENTITY

**Title:** A Hybrid Physics-Guided Neural Network with Learnable Arrhenius Parameters for Flow Stress Prediction of Al 6011-O Aluminum Alloy during Hot Deformation

**Student:** Trần Ngọc Dũng (陈玉勇) — 20241961056 — DUT  
**Data:** Hot tensile, Al 6011-O sheet, 7T × 3ε̇ = 21 conditions. From 祉晗 (advisor's group).

---

## 2. ARRHENIUS EQUATION

$$\dot{\varepsilon} = A[\sinh(\alpha\sigma)]^n \exp(-Q/RT)$$

Solved for σ: $\sigma = \frac{1}{\alpha}\sinh^{-1}\left[(Z/A)^{1/n}\right]$, where $Z = \dot{\varepsilon}\exp(Q/RT)$, R = 8.314 J/(mol·K).

---

## 3. SCAM BASELINE — ✅ COMPLETED

Fit α, n, Q, lnA at each strain level via sequential linear regression on 15 conditions (250–450°C). RT & 150°C excluded — DSA anomaly (at 150°C, UTS ≈ 168→160 MPa across 100× strain rate increase, near-zero strain-rate sensitivity violates Arrhenius assumption of thermally-activated deformation). 6th-order polynomials α(ε), n(ε), Q(ε), lnA(ε).

**Results (15 fitting conditions):** R² = 0.9950, RMSE = 2.00 MPa, AARE = 3.15%

**Polynomial coefficients** stored in `SCAM_complete_kaggle.py` as `SCAM_POLY_COEFFS` dict. Prediction function:
```python
def scam_sigma(eps, T_K, sr):
    α, n, Q, lnA = [np.polyval(SCAM_POLY_COEFFS[p], eps) for p in ['alpha','n','Q','lnA']]
    return (1/α) * np.arcsinh((sr * np.exp(Q/(R*T_K)) / np.exp(lnA))**(1/n))
```

**Deliverables:** `SCAM_complete_kaggle.py` (full pipeline + 5 figures + equations), `tensile_data_with_strain.xlsx` (raw + added True_Strain/True_Stress columns).

---

## 4. DATASET — ✅ COMPLETED

**Raw:** `tensile_data.xlsx` — 21 sheets. Strain column (extensometer) unreliable → use Displacement/L₀ (L₀ = 25 mm). True stress = σ_eng × (1 + ε_eng), True strain = ln(1 + ε_eng).

**Downsampled:** `al6011_downsampled_full.xlsx` — Δε = 0.005, full curve (hardening + softening), **1,982 samples**, 21 sheets + Summary. Columns: T_C, T_K, strain_rate, ln_sr, eps_true, sigma_true.

**Split:** Train 70% / Val 15% / Test 15%, stratified by (T, ε̇).

**Model inputs:** [T_K, ln(ε̇), ε_true] → StandardScaler. **Output:** σ_true (MPa).

---

## 5. MODEL ARCHITECTURE

### Black-box ANN (baseline)
```
Input(3) → Dense(64,ReLU) → Dropout(0.2) → Dense(64,ReLU) → Dropout(0.2) → Dense(32,ReLU) → Output(1 = σ)
```

### Hybrid PGNN (main model)
```
Input(3) → Dense(64,ReLU) → Dropout(0.2) → Dense(64,ReLU) → Dropout(0.2) → Dense(32,ReLU)
    → 4 sigmoid-bounded heads:
        α  = 0.005 + 0.045·sigmoid(z)    ∈ [0.005, 0.05] MPa⁻¹
        n  = 2 + 8·sigmoid(z)            ∈ [2, 10]
        Q  = 130000 + 90000·sigmoid(z)   ∈ [130k, 220k] J/mol
        lnA = 15 + 25·sigmoid(z)         ∈ [15, 40]
    → Arrhenius layer (differentiable): σ = (1/α)·arcsinh[(Z/exp(lnA))^(1/n)]
```

Sigmoid instead of clipping → gradient always flows. Same backbone for ANN and PGNN (fair comparison).

---

## 6. LOSS & λ STRATEGIES

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \cdot \mathcal{L}_{physics}$$

- $\mathcal{L}_{data}$ = MSE(σ_pred, σ_exp)
- $\mathcal{L}_{physics}$ = MSE(σ_pred, σ_SCAM) — uses polynomial SCAM, not discrete values (continuous + differentiable, no circular reasoning)

**Strategy A (Warmup + Grid):** Linear warmup 20% epochs, then hold λ_max ∈ {0.01, 0.1, 0.5, 1.0}.  
**Strategy B (Adaptive Ratio):** Auto-adjust λ per epoch. r = L_data/(λ·L_physics). r>1.2→↑λ, r<0.8→↓λ.  
**Strategy C (Gradient Norm):** λ = ‖∇L_data‖/‖∇L_physics‖. Two backward passes. Ref: Wang et al. 2021.

---

## 7. 5-MODEL COMPARISON

| # | Model | Output | Loss |
|---|---|---|---|
| 0 | SCAM | polynomial regression | N/A ✅ done |
| 1 | ANN | σ directly | L_data |
| 2 | PGNN (no λ) | α,n,Q,lnA → Arrhenius → σ | L_data |
| 3 | PGNN + λA | same | L_data + λ·L_physics |
| 4 | PGNN + λB | same | L_data + λ·L_physics |
| 5 | PGNN + λC | same | L_data + λ·L_physics |

**Metrics:** R², RMSE, AARE. Plus physical consistency check (learned Q, n, α vs literature ranges).

---

## 8. MC DROPOUT UQ

On best model. Keep dropout ON at inference, T=100 forward passes.  
μ = mean(predictions), σ_unc = std(predictions), 95% CI = μ ± 1.96·σ_unc.  
Metrics: PICP ≥ 95%, MPIW (smaller = better).

---

## 9. CURRENT STATUS

| Step | Status |
|---|---|
| Data processing + true stress/strain | ✅ |
| Downsampling (Δε=0.005, 1982 pts) | ✅ |
| SCAM fitting + polynomials + plots | ✅ |
| SCAM Kaggle script | ✅ |
| **→ Build ANN (Model 1)** | **NEXT** |
| Build PGNN (Models 2–5) | pending |
| MC Dropout UQ | pending |
| Paper writing | pending |

---

## 10. KEY FILES

| File | Contents |
|---|---|
| `tensile_data.xlsx` | Raw data, 21 sheets |
| `tensile_data_with_strain.xlsx` | Raw + True_Strain, True_Stress columns |
| `al6011_downsampled_full.xlsx` | Downsampled, 21 sheets + Summary, ready for ML |
| `SCAM_complete_kaggle.py` | Full SCAM pipeline for Kaggle |

---

## 11. MUST-READ PAPERS

**Tier 1 — Competing:** [1] DNN vs Arrhenius Al 7075 (2023) doi:10.1007/s12206-023-0114-5 | [2] BP-ANN vs SCAM Al 7075 (2021) doi:10.3390/ma14205986 | [3] ML comparison Al 7075 (2025) doi:10.3389/fmats.2025.1671753 | [4] 5 ML AA6061-T6 (2025) SSRN:5398019

**Tier 2 — Method:** [5] NN-EVP PINN (2023) arXiv:2307.04301 | [6] LSTM Al 2024 (2025) doi:10.1016/S0924-0136(25)00333-4 | [14] Wang gradient pathologies (2021) doi:10.1137/20M1318043

**Tier 3 — Reference:** [9] ANN Al 6A02 Q=168.9 kJ/mol (2017) | [10] Hot tensile AA5005 SCAM (2022) | [11] Al 6063-T5 Q=191 kJ/mol (2026) | [12] Al-Zn-Mg-Cu fitting (2024)

---

## 12. PAPER STRUCTURE

1. Introduction  
2. Related Work (SCAM / ML for hot deformation / PGNN / UQ)  
3. Methodology (architecture / loss / λ strategies / MC Dropout)  
4. Experiments (material / data processing / baselines / metrics)  
5. Results (flow behavior / SCAM / 5-model comparison / physical consistency / UQ)  
6. Discussion  
7. Conclusion

**Contributions:** (1) Hybrid PGNN with Arrhenius in architecture + loss, (2) strain-dependent parameters end-to-end, (3) systematic 5-model + 3λ comparison, (4) MC Dropout UQ for flow stress, (5) first ML on Al 6011-O hot deformation.
