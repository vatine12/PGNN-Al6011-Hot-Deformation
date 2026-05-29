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

For detailed results and analysis, see [RESULTS.md](RESULTS.md).

## Repository Structure

```
├── hot-tensil-pgnn-comprehensive.ipynb   # Full pipeline: training + analysis + physical insight
├── RESULTS.md                            # Detailed results and physical insight analysis
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
