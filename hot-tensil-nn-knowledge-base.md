# Project Knowledge Base — Physics-Guided ML for Al 6011-O Hot-Deformation Flow Stress

*Compact context for continuing this research in a new chat. Last updated after the masked-mono (v12) run.*

## 1. Goal & who
- **Aim:** predict flow stress σ of **Al 6011-O** during hot tensile deformation as a function of temperature T, strain rate ε̇, and true strain ε, using a **physics-guided neural network (PGNN)**.
- **Original ambition:** show that **PGNN + an added physics loss beats plain PGNN** (a PGNN/PINN-flavored contribution), targeting a **Q2 / low-Q1** materials journal (e.g. JMRT, Metals, JMEP, MSEA). First-author undergrad researcher (DUT); Python + PyTorch; runs on Kaggle (T4 GPU).

## 2. Data
- File: `al6011_downsampled_full.xlsx`, one sheet per condition. Columns: `T_C, T_K, strain_rate, ln_sr, eps_true, sigma_true`.
- **21 conditions = 7 temperatures × 3 strain rates.** T = 25(RT), 150, 250, 300, 350, 400, 450 °C. ε̇ = 0.001, 0.01, 0.1 s⁻¹. ~1,982 points total.
- Each condition is a **smooth stress–strain curve** sampled at Δε≈0.005; true strain reaches ~0.7 at high T.
- Two physical regimes: **hot working (250–450 °C)** = classic Arrhenius/DRX; **RT & 150 °C** = **DSA/PLC** (dynamic strain aging), qualitatively different.

## 3. Models
- **SCAM** — classical strain-compensated sinh-Arrhenius: ε̇ = A[sinh(ασ)]ⁿ exp(−Q/RT); α,n,Q,lnA are polynomials in ε. Fit on **hot conditions only (T≥250)**, valid ε window ≈ **[0.05, 0.19]** (uniform-elongation limit; beyond it, gauge-average true stress is compromised by necking). Serves as a *reference / parameter validator*, not ground truth.
- **ANN** — black-box MLP baseline (T,ln ε̇,ε → σ).
- **PGNN (core model)** — MLP backbone → 4 heads output **α,n,Q,lnA** (sigmoid-bounded to physical ranges) → **sinh-Arrhenius formula computes σ**. Physics is embedded in the *architecture*. `HybridPGNN` in the notebooks.
- **Physics-loss variants tried:** SCAM-matching `MSE(σ_pred, σ_SCAM)` with λ schedules (λA fixed-warmup, λB adaptive-ratio, λC gradient-norm); plus `L_consistency` (params depend on ε only), `L_smooth` (params smooth in ε), `L_mono` (σ↑ with ε̇, σ↓ with T). Multi-term weighting via fixed-λ or learned **uncertainty weighting**.

## 4. Evaluation methodology (important)
- **Original flaw:** random within-curve split = **interpolation leakage** (test points sit between train points on the same curve) → inflated R²≈0.97–0.98, and physics priors can't show value.
- **Fix — honest extrapolation protocols:**
  - **LOCO** (leave-one-condition-out): hold out one whole (T,ε̇) curve, train on the other 20, predict it. Folds restricted to the 15 hot conditions.
  - **LOTO** (leave-one-temperature-out): hold out all rates at one temperature (harder; tests T-extrapolation).
  - **Random split** kept as an interpolation reference.
- **Stats:** multi-seed (up to 8), **paired Δ** (baseline vs variant on same seed×fold) + t-test. Effect sizes are small vs large fold-to-fold variance, so paired analysis is essential.

## 5. Key findings (the crux — this is what changed the project)
1. **All hand-crafted physics losses are redundant with the architecture.** SCAM-matching (same data + same Arrhenius equation the architecture already uses), consistency, and smoothness were all **within noise** under LOCO/LOTO.
2. **Monotonicity loss is *identically zero* for this architecture.** The sinh-Arrhenius form **structurally guarantees** σ↑ with ε̇ and σ↓ with T (verified **0 / 6000** violations across random models/inputs; Q>0 bounds force it). So `L_mono ≡ 0`, and **masked vs unmasked gave byte-identical results.**
3. **The apparent "mono helps extrapolation" (+0.63 pp LOCO, +0.69 pp LOTO) was a CONFOUND**, not physics: the `mono` config used **uncertainty loss-weighting** while `baseline` used fixed weighting. The gain came from the weighting scheme; the physics term contributed zero gradient. (Random-split Δ was slightly negative → noise.)
4. **Structural limitation:** the architecture **cannot represent DSA / negative strain-rate sensitivity** at RT/150 °C (data confirms σ *decreases* with ε̇ there: 0% monotone vs 100% at 250–450 °C). No loss (masked or not) can fix RT/150 — the rigidity is in the equation, not the penalty.
5. **The architecture is the real contribution:** the big accuracy jump is **ANN → PGNN** (embedding Arrhenius), not any λ/physics-loss term. Learned parameters are **physical**: Q ≈ **175 kJ/mol** (literature ~150–185; Al self-diffusion ~142), n ≈ 6.
6. **Extrapolation ≫ harder than interpolation:** random R² ≈ 0.97 vs **LOCO R² ≈ 0.64**.

## 6. Current honest paper framing
> A physics-structured Arrhenius network encodes the hot-deformation physics so completely that **auxiliary physics losses are redundant** (monotonicity is structural; parameter smoothness is built into the strain-compensated form; SCAM-matching duplicates the same law). The **architecture (ANN→PGNN), not loss-based physics, carries the improvement.** Its built-in monotonicity drives reliable hot-regime extrapolation but **structurally precludes modeling dynamic strain aging at RT/150 °C.** Evaluated honestly via LOCO/LOTO with paired multi-seed statistics; parameters validated against classical SCAM and literature.

This is a defensible *negative-with-a-positive* result (Q2 solid, Q1-low with hardening). Q1-top would need microstructure (EBSD/TEM) — not available.

## 7. Open directions (next research)
The rigid architecture is why losses are redundant *and* why RT/150 fails. To predict accurately across **all** conditions, change the **architecture**, then physics regularization becomes non-redundant:
- **Additive residual (grey-box):** σ = σ_Arrhenius(NN₁) + Δ(NN₂); regularize Δ→0 in hot regime, free at RT/150 (captures DSA).
- **Gated mixture-of-experts:** σ = g(x)·σ_Arrhenius + (1−g)·σ_freeNN; gate learns *where physics is valid* (interpretable validity boundary).
- **More novel formulations:** physics-structured **neural ODE over strain** (dσ/dε = f_θ, captures path-dependent DSA/serrations pointwise models can't); **constitutive-law discovery** (symbolic/sparse regression for the equation + its domain of validity); **domain-of-validity learning** with UQ.
- **Caveat:** with only 21 conditions, novelty should come from the **physics formulation**, not a new NN primitive; verify novelty against literature (neural ODEs for constitutive modeling, symbolic constitutive discovery, DSA modeling) before claiming it.

## 8. Artifacts (in project folder)
- Notebooks: `hot-tensil-pgnn-physics-pipeline.ipynb` (full LOCO/LOTO/ablation harness), `hot-tensil-pgnn-stepB-mono.ipynb` (baseline-vs-mono, 8 seeds + LOTO), `hot-tensil-pgnn-stepB2-mono-masked.ipynb` (masked mono + random anchor + per-21-condition table). Executed results: `...v10/v11/v12...ipynb`.
- CSVs: `ablation_loco_raw.csv`, `stepB_loco_mono_raw.csv`, `stepB_loto_mono_raw.csv`, `stepB_random_mono_raw.csv`, `stepB_percondition_raw.csv`.
- Config knobs: `CFG.quick` (debug vs full), `CFG.seeds`, `CFG.mono_srate_hot_only`, `scam_temp_min_K=523`, `scam_eps_max=0.19`.

## 9. Immediate open question / recommended next step
Confirm the confound with a clean control: add a **`data-only + uncertainty-weighting`** config and compare (fixed baseline) vs (uncertainty baseline) vs (mono) under LOCO — expect uncertainty-baseline ≈ mono, proving the effect was weighting, not physics. Then either write up the "architecture-carries-the-physics + DSA limitation" story, or pivot to a **regime-aware hybrid architecture** to fix RT/150 and make physics guidance genuinely non-redundant.
