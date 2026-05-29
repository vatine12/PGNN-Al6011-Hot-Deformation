# PGNN-Al6011-Hot-Deformation

## What This Project Does

When aluminum is heated and stretched (hot deformation), engineers need to predict how much force the metal can withstand. This is critical for manufacturing processes like forging and rolling. Traditional models use physics equations but struggle with accuracy; pure machine learning models are accurate but act as "black boxes" — they give no insight into *why* the metal behaves the way it does.

This project combines both approaches: a neural network that is forced to learn real physical parameters (like activation energy and stress sensitivity) rather than just memorizing patterns. The result is a model that is both accurate and physically interpretable.

## How It Works

Instead of directly predicting stress, the neural network learns four physical material parameters at each deformation condition. These parameters are then plugged into the [Arrhenius equation](https://en.wikipedia.org/wiki/Arrhenius_equation) — a well-known physics formula for thermally activated processes — to compute the final stress prediction.

```
Input (temperature, strain rate, strain)
    → Neural Network backbone
    → 4 physics parameter heads (α, n, Q, lnA)
    → Arrhenius equation → predicted stress
```

The key constraint: each parameter is bounded to a physically realistic range using sigmoid activation, so the network can't "cheat" by learning nonsensical values.

## Key Results

**Prediction accuracy (test set):**

| Model | R² | RMSE (MPa) | AARE (%) |
|-------|-----|-----------|----------|
| Traditional (SCAM) | — | 69.0 | 54.0 |
| Black-box ANN | 0.939 | 13.6 | 12.2 |
| **PGNN+λA (ours)** | **0.947** | **12.6** | **7.3** |

**Physical insight — the model rediscovers known physics without being told:**
- Activation energy Q ≈ 179 kJ/mol (literature for Al 6xxx: 130–180 kJ/mol)
- Stress exponent n ≈ 5.5–8, decreasing with temperature (consistent with dislocation climb)
- Q–ln(A) compensation effect (r = −0.81), a well-documented phenomenon in hot deformation

This means the architecture is not just fitting curves — it is learning real material behavior.

## Material & Dataset

- **Alloy:** Al 6011-O aluminum
- **Test conditions:** 7 temperatures (RT–450°C) × 3 strain rates (0.001–0.1 s⁻¹) = 21 conditions
- **Data points:** 1,982 (downsampled from raw tensile tests at Δε = 0.005)
- **Split:** 70% train / 15% validation / 15% test, stratified by condition

The processed dataset (`al6011_downsampled_full.xlsx`) is included in this repository.

## Models Compared

| Model | What it is |
|-------|-----------|
| SCAM | Traditional Strain-Compensated Arrhenius Model (polynomial regression) |
| ANN | Standard neural network — predicts stress directly, no physics |
| PGNN | Physics-guided neural network — learns Arrhenius parameters, data loss only |
| PGNN+λA | PGNN with an additional physics-regularization loss (best model) |

## Repository Structure

```
├── hot-tensil-pgnn-comprehensive.ipynb   # Full pipeline: training + analysis + physical insight
├── hot-tensil-pgnn-v6.ipynb              # Core PGNN training pipeline
├── analysis_percondition_physics.py      # Standalone per-condition analysis script
├── al6011_downsampled_full.xlsx          # Processed dataset (1,982 samples)
├── al6011_data_summary.xlsx              # Data summary statistics
├── Code/
│   ├── hot-tensile-ann.ipynb             # ANN baseline notebook
│   └── hot-tensile-scam.ipynb            # SCAM baseline notebook
└── Futher_analysis_results/              # Output figures and metrics from Kaggle run
    ├── model_comparison_summary.csv
    ├── per_condition_metrics.csv
    └── fig_*.png                         # 6 analysis figures
```

## How to Run

1. Upload `hot-tensil-pgnn-comprehensive.ipynb` and `al6011_downsampled_full.xlsx` to [Kaggle](https://www.kaggle.com/)
2. Enable **GPU accelerator** in notebook settings
3. Run all cells — the notebook trains all models and generates analysis outputs

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn

## Authors

- **Nguyen Tran Quang Minh**
- **Tran Ngoc Dung** — Dalian University of Technology (DUT), China

## Citation

If you use this code, please cite our work (paper in preparation).
