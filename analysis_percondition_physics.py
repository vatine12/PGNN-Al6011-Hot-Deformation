"""
Per-Condition Metrics & Physical Insight Analysis
==================================================
Al 6011-O Hot Tensile — SCAM / ANN / PGNN / PGNN+λA

Outputs:
  - per_condition_metrics.csv          (R², RMSE, AARE per condition × model × split)
  - model_comparison_summary.csv       (overall metrics per model)
  - Figures (PNG):
      fig_percondition_heatmap.png     R²/RMSE/AARE heatmaps per model
      fig_percondition_bar.png         bar chart of test AARE per condition
      fig_learned_params_vs_scam.png   α, n, Q, lnA vs SCAM polynomials
      fig_param_evolution_T.png        parameter box-plots by temperature
      fig_activation_energy.png        Q distribution vs literature
      fig_correlation_matrix.png       param–param correlations
"""

from __future__ import annotations
import copy, os, random, warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Config:
    data_path: str = "al6011_downsampled_full.xlsx"
    seed: int = 42
    feature_cols: List[str] = field(default_factory=lambda: ["T_K", "ln_sr", "eps_true"])
    target_col: str = "sigma_true"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 64])
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 500
    patience: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 30
    grad_clip: float = 10.0
    warmup_frac_A: float = 0.20
    lambda_max_A: float = 0.01
    mc_passes: int = 100
    scam_temp_min_K: float = 523.0
    aare_floor: float = 5.0

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = Config()

# Quick-test mode: set QUICK_TEST=True to verify code logic with minimal training
QUICK_TEST = os.environ.get("QUICK_TEST", "0") == "1"
if QUICK_TEST:
    CFG.epochs = 5
    CFG.patience = 3
    print("*** QUICK TEST MODE — 5 epochs only ***")

# ── Plot style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

# ═══════════════════════════════════════════════════════════════════
# 1.  REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════
def seed_everything(seed: int = CFG.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()
print(f"Device: {CFG.device}")

# ═══════════════════════════════════════════════════════════════════
# 2.  DATA
# ═══════════════════════════════════════════════════════════════════
xf = pd.ExcelFile(CFG.data_path)
sheets = [s for s in xf.sheet_names if s != "Summary"]
data = pd.concat(
    [pd.read_excel(CFG.data_path, sheet_name=s).assign(condition=s) for s in sheets],
    ignore_index=True,
)
print(f"Loaded {len(data):,} pts | {data['condition'].nunique()} conditions")

X_all = data[CFG.feature_cols].values.astype(np.float32)
y_all = data[CFG.target_col].values.astype(np.float32).reshape(-1, 1)
T_K_raw = data["T_K"].values.astype(np.float32)
sr_raw  = data["strain_rate"].values.astype(np.float32)
eps_raw = data["eps_true"].values.astype(np.float32)

# ── Stratified split ─────────────────────────────────────────────
seed_everything()
train_idx, val_idx, test_idx = [], [], []
for cond in data["condition"].unique():
    idx = data[data["condition"] == cond].index.values.copy()
    np.random.shuffle(idx)
    n = len(idx)
    n_tr = int(CFG.train_ratio * n)
    n_va = int(CFG.val_ratio * n)
    train_idx.extend(idx[:n_tr])
    val_idx.extend(idx[n_tr : n_tr + n_va])
    test_idx.extend(idx[n_tr + n_va :])
print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

data["split"] = "N/A"
for idx_list, lbl in [(train_idx, "train"), (val_idx, "val"), (test_idx, "test")]:
    data.loc[idx_list, "split"] = lbl

# ── Scaling ───────────────────────────────────────────────────────
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_all[train_idx]).astype(np.float32)
y_train_s = scaler_y.fit_transform(y_all[train_idx]).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════
# 3.  SCAM (Strain-Compensated Arrhenius Model)
# ═══════════════════════════════════════════════════════════════════
R_GAS = 8.314

SCAM_POLY = {
    "alpha": np.array([ 2.990838465670e+04, -1.994069281431e+04,  5.345762450621e+03,
                        -7.350303313298e+02,  5.448368675032e+01, -2.063309588944e+00,
                         5.051186723984e-02]),
    "n":     np.array([-7.812215710077e+05,  2.343228580944e+05,  4.087839767037e+04,
                        -2.591299218747e+04,  4.249708974788e+03, -3.168428320818e+02,
                         1.508120816407e+01]),
    "Q":     np.array([-8.543584880384e+10,  5.238248618259e+10, -1.249643528684e+10,
                         1.436699072794e+09, -7.644242300049e+07,  1.055736379502e+06,
                         1.900591723054e+05]),
    "lnA":   np.array([-2.689844217277e+07,  1.687345535442e+07, -4.163261535534e+06,
                         5.066408364188e+05, -3.055325125124e+04,  7.305787733034e+02,
                         2.523144522408e+01]),
}

def scam_sigma(eps, T_K, sr):
    a   = np.polyval(SCAM_POLY["alpha"], eps)
    n   = np.polyval(SCAM_POLY["n"],     eps)
    Q   = np.polyval(SCAM_POLY["Q"],     eps)
    lnA = np.polyval(SCAM_POLY["lnA"],   eps)
    Z = sr * np.exp(Q / (R_GAS * T_K))
    return (1.0 / a) * np.arcsinh((Z / np.exp(lnA)) ** (1.0 / n))

# SCAM predictions (valid only for 250-450°C)
sigma_scam_all = scam_sigma(eps_raw, T_K_raw, sr_raw)
physics_valid = np.isfinite(sigma_scam_all) & (T_K_raw >= CFG.scam_temp_min_K)
sigma_scam_all[~physics_valid] = np.nan
sigma_scam_all = np.clip(sigma_scam_all, 0, 500)
data["pred_SCAM"] = sigma_scam_all

# Scaled SCAM for physics loss
sigma_scam_for_loss = np.where(physics_valid, sigma_scam_all, 0.0).astype(np.float32).reshape(-1, 1)
sigma_scam_scaled = scaler_y.transform(sigma_scam_for_loss).astype(np.float32)
physics_mask = physics_valid.astype(np.float32).reshape(-1, 1)

print(f"SCAM valid: {physics_valid.sum()}/{len(physics_valid)} pts")

# ═══════════════════════════════════════════════════════════════════
# 4.  MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════

# ── 4a. ANN (black-box MLP) ──────────────────────────────────────
class FlowStressANN(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = CFG.hidden_dims
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── 4b. PGNN (Arrhenius output layer) ────────────────────────────
PARAM_BOUNDS = {
    "alpha": (0.005,   0.045),
    "n":     (2.0,     8.0),
    "Q":     (130_000, 90_000),
    "lnA":   (15.0,    25.0),
}

class HybridPGNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = CFG.hidden_dims
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dims[-1], 1) for name in PARAM_BOUNDS
        })

    def forward(self, x_scaled, T_K, sr, eps):
        h = self.backbone(x_scaled)
        params = {}
        for name, (lo, rng) in PARAM_BOUNDS.items():
            params[name] = lo + rng * torch.sigmoid(self.heads[name](h))
        Z_exp = torch.clamp(params["Q"] / (R_GAS * T_K), max=80.0)
        Z     = sr * torch.exp(Z_exp)
        ratio = torch.clamp(Z / torch.exp(params["lnA"]), min=1e-10, max=1e30)
        sigma = (1.0 / params["alpha"]) * torch.arcsinh(ratio ** (1.0 / params["n"]))
        return sigma, params


# ═══════════════════════════════════════════════════════════════════
# 5.  TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _scale_sigma(raw: torch.Tensor) -> torch.Tensor:
    return (raw - scaler_y.mean_[0]) / scaler_y.scale_[0]

def make_pgnn_loader(idx_list, shuffle):
    tensors = (
        torch.tensor(scaler_X.transform(X_all[idx_list]).astype(np.float32)),
        torch.tensor(scaler_y.transform(y_all[idx_list]).astype(np.float32)),
        torch.tensor(T_K_raw[idx_list].reshape(-1, 1)),
        torch.tensor(sr_raw[idx_list].reshape(-1, 1)),
        torch.tensor(eps_raw[idx_list].reshape(-1, 1)),
        torch.tensor(sigma_scam_scaled[idx_list]),
        torch.tensor(physics_mask[idx_list]),
    )
    return DataLoader(TensorDataset(*tensors),
                      batch_size=CFG.batch_size if shuffle else 256,
                      shuffle=shuffle)

def make_ann_loader(idx_list, shuffle):
    X_s = scaler_X.transform(X_all[idx_list]).astype(np.float32)
    y_s = scaler_y.transform(y_all[idx_list]).astype(np.float32)
    return DataLoader(TensorDataset(torch.tensor(X_s), torch.tensor(y_s)),
                      batch_size=CFG.batch_size if shuffle else 256,
                      shuffle=shuffle)

# ── Train ANN ─────────────────────────────────────────────────────
def train_ann(verbose_every=100):
    seed_everything()
    model = FlowStressANN().to(CFG.device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=CFG.scheduler_factor,
        patience=CFG.scheduler_patience, min_lr=1e-6)
    criterion = nn.MSELoss()

    loader_tr = make_ann_loader(train_idx, True)
    loader_va = make_ann_loader(val_idx, False)

    best_val, best_state, wait, best_ep = float("inf"), None, 0, 0

    for ep in range(CFG.epochs):
        model.train()
        s_loss, n = 0.0, 0
        for X_b, y_b in loader_tr:
            X_b, y_b = X_b.to(CFG.device), y_b.to(CFG.device)
            loss = criterion(model(X_b), y_b)
            opt.zero_grad(); loss.backward(); opt.step()
            s_loss += loss.item() * len(X_b); n += len(X_b)

        model.eval()
        v_sum = 0.0
        with torch.no_grad():
            for X_b, y_b in loader_va:
                X_b, y_b = X_b.to(CFG.device), y_b.to(CFG.device)
                v_sum += criterion(model(X_b), y_b).item() * len(X_b)
        val_loss = v_sum / len(val_idx)
        sched.step(val_loss)

        if val_loss < best_val:
            best_val, best_ep, wait = val_loss, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        if (ep+1) % verbose_every == 0:
            print(f"  [ANN] ep {ep+1:4d}  train={s_loss/n:.6f}  val={val_loss:.6f}")
        if wait >= CFG.patience:
            print(f"  [ANN] Early stop ep {ep+1} (best: {best_ep+1})")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  [ANN] Best val={best_val:.6f} @ ep {best_ep+1}")
    return model

# ── Train PGNN (strategy none or A) ──────────────────────────────
def train_pgnn(strategy="none", lambda_max=0.01, run_name="PGNN", verbose_every=100):
    seed_everything()
    model = HybridPGNN().to(CFG.device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=CFG.scheduler_factor,
        patience=CFG.scheduler_patience, min_lr=1e-6)
    criterion = nn.MSELoss()

    loader_tr = make_pgnn_loader(train_idx, True)
    loader_va = make_pgnn_loader(val_idx, False)

    best_val, best_state, wait, best_ep = float("inf"), None, 0, 0
    warmup_A = int(CFG.warmup_frac_A * CFG.epochs)

    for ep in range(CFG.epochs):
        lam = 0.0
        if strategy == "A":
            lam = lambda_max * min(ep / warmup_A, 1.0)

        model.train()
        s_total, n = 0.0, 0
        for X_b, y_b, T_b, sr_b, eps_b, scam_b, pmask_b in loader_tr:
            X_b  = X_b.to(CFG.device);  y_b   = y_b.to(CFG.device)
            T_b  = T_b.to(CFG.device);  sr_b  = sr_b.to(CFG.device)
            eps_b = eps_b.to(CFG.device); scam_b = scam_b.to(CFG.device)
            pmask_b = pmask_b.to(CFG.device)

            sigma_raw, _ = model(X_b, T_b, sr_b, eps_b)
            sigma_s = _scale_sigma(sigma_raw)
            L_data = criterion(sigma_s, y_b)

            if strategy == "A" and lam > 0:
                n_valid = pmask_b.sum()
                if n_valid > 0:
                    L_phys = ((sigma_s - scam_b)**2 * pmask_b).sum() / n_valid
                else:
                    L_phys = torch.tensor(0.0, device=CFG.device)
                loss = L_data + lam * L_phys
            else:
                loss = L_data

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            opt.step()
            s_total += loss.item() * len(X_b); n += len(X_b)

        model.eval()
        v_sum = 0.0
        with torch.no_grad():
            for X_b, y_b, T_b, sr_b, eps_b, *_ in loader_va:
                X_b = X_b.to(CFG.device); y_b = y_b.to(CFG.device)
                T_b = T_b.to(CFG.device); sr_b = sr_b.to(CFG.device)
                eps_b = eps_b.to(CFG.device)
                s_raw, _ = model(X_b, T_b, sr_b, eps_b)
                v_sum += criterion(_scale_sigma(s_raw), y_b).item() * len(X_b)
        val_loss = v_sum / len(val_idx)
        sched.step(val_loss)

        if val_loss < best_val:
            best_val, best_ep, wait = val_loss, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        if (ep+1) % verbose_every == 0:
            print(f"  [{run_name}] ep {ep+1:4d}  total={s_total/n:.6f}  val={val_loss:.6f}  λ={lam:.4f}")
        if wait >= CFG.patience:
            print(f"  [{run_name}] Early stop ep {ep+1} (best: {best_ep+1})")
            break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  [{run_name}] Best val={best_val:.6f} @ ep {best_ep+1}")
    return model

# ═══════════════════════════════════════════════════════════════════
# 6.  PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════

def predict_ann(model, idx_list):
    model.eval()
    X_s = torch.tensor(scaler_X.transform(X_all[idx_list]).astype(np.float32), device=CFG.device)
    with torch.no_grad():
        y_s = model(X_s).cpu().numpy()
    return scaler_y.inverse_transform(y_s).flatten()

def predict_pgnn(model, idx_list):
    model.eval()
    X_s   = torch.tensor(scaler_X.transform(X_all[idx_list]).astype(np.float32), device=CFG.device)
    T_t   = torch.tensor(T_K_raw[idx_list].reshape(-1, 1), device=CFG.device)
    sr_t  = torch.tensor(sr_raw[idx_list].reshape(-1, 1),  device=CFG.device)
    eps_t = torch.tensor(eps_raw[idx_list].reshape(-1, 1), device=CFG.device)
    with torch.no_grad():
        sigma, params = model(X_s, T_t, sr_t, eps_t)
    return sigma.cpu().numpy().flatten(), {k: v.cpu().numpy().flatten() for k, v in params.items()}

def aare(y_true, y_pred, floor=CFG.aare_floor):
    m = y_true.flatten() > floor
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100)

def metrics_dict(y_true, y_pred):
    yt, yp = y_true.flatten(), y_pred.flatten()
    valid = np.isfinite(yp)
    if valid.sum() < 2:
        return {"R2": np.nan, "RMSE": np.nan, "AARE": np.nan}
    yt_v, yp_v = yt[valid], yp[valid]
    return {
        "R2":   float(r2_score(yt_v, yp_v)),
        "RMSE": float(np.sqrt(mean_squared_error(yt_v, yp_v))),
        "AARE": aare(yt_v, yp_v),
    }

# ═══════════════════════════════════════════════════════════════════
# 7.  TRAIN ALL MODELS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Training ANN …")
print("=" * 60)
ann_model = train_ann()

print("\n" + "=" * 60)
print("Training PGNN (no physics loss) …")
print("=" * 60)
pgnn_model = train_pgnn(strategy="none", run_name="PGNN")

print("\n" + "=" * 60)
print("Training PGNN+λA (λ_max=0.01, warmup) …")
print("=" * 60)
pgnn_la_model = train_pgnn(strategy="A", lambda_max=CFG.lambda_max_A, run_name="PGNN+λA")

# ── Generate all predictions on full dataset ──────────────────────
all_idx = list(range(len(data)))

data["pred_ANN"]  = predict_ann(ann_model, all_idx)
data["pred_PGNN"] = predict_pgnn(pgnn_model, all_idx)[0]
pred_la, params_la = predict_pgnn(pgnn_la_model, all_idx)
data["pred_PGNN_lA"] = pred_la

# Also get PGNN (no λ) params for comparison
_, params_nolam = predict_pgnn(pgnn_model, all_idx)

# Store learned parameters in dataframe
for pname in ["alpha", "n", "Q", "lnA"]:
    data[f"learned_{pname}_lA"] = params_la[pname]
    data[f"learned_{pname}_noL"] = params_nolam[pname]

MODEL_COLS = {
    "SCAM":     "pred_SCAM",
    "ANN":      "pred_ANN",
    "PGNN":     "pred_PGNN",
    "PGNN+λA":  "pred_PGNN_lA",
}

# ── Quick overall check ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Overall Test Metrics")
print("=" * 60)
test_data = data[data["split"] == "test"]
print(f"{'Model':<12} {'R²':>9} {'RMSE':>10} {'AARE':>9}")
print("─" * 42)
for mname, col in MODEL_COLS.items():
    yt = test_data["sigma_true"].values
    yp = test_data[col].values
    m = metrics_dict(yt, yp)
    print(f"{mname:<12} {m['R2']:>9.6f} {m['RMSE']:>9.4f} {m['AARE']:>8.2f}%")

# ═══════════════════════════════════════════════════════════════════
# 8.  PER-CONDITION METRICS (R², RMSE, AARE)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Per-Condition Metrics (Test Set)")
print("=" * 60)

rows = []
for cond in data["condition"].unique():
    sub = data[data["condition"] == cond]
    T_C = sub["T_C"].iloc[0]
    sr  = sub["strain_rate"].iloc[0]
    for sp in ["train", "val", "test"]:
        s = sub[sub["split"] == sp]
        if s.empty:
            continue
        yt = s["sigma_true"].values
        for mname, col in MODEL_COLS.items():
            yp = s[col].values
            m = metrics_dict(yt, yp)
            rows.append({
                "condition": cond, "T_C": T_C, "strain_rate": sr,
                "split": sp, "n_pts": len(s), "model": mname,
                "R2": m["R2"], "RMSE": m["RMSE"], "AARE": m["AARE"],
            })

cond_df = pd.DataFrame(rows)
cond_df.to_csv("per_condition_metrics.csv", index=False)
print("Saved: per_condition_metrics.csv")

# Print test summary
tc = cond_df[cond_df["split"] == "test"].copy()
print(f"\n{'Condition':<18} {'T°C':>5} {'SR':>8}  ", end="")
for mn in MODEL_COLS:
    print(f"  {mn:>10}", end="")
print("  (AARE %)")
print("─" * 90)

for cond in sorted(tc["condition"].unique(), key=lambda c: (
        tc[tc["condition"]==c]["T_C"].iloc[0], tc[tc["condition"]==c]["strain_rate"].iloc[0])):
    sub = tc[tc["condition"] == cond]
    T_C = sub["T_C"].iloc[0]
    sr  = sub["strain_rate"].iloc[0]
    print(f"{cond:<18} {T_C:5.0f} {sr:8.4f}  ", end="")
    for mn in MODEL_COLS:
        row = sub[sub["model"] == mn]
        if row.empty or np.isnan(row["AARE"].iloc[0]):
            print(f"  {'N/A':>10}", end="")
        else:
            print(f"  {row['AARE'].iloc[0]:>9.2f}%", end="")
    print()

# ── Summary table ────────────────────────────────────────────────
summary_rows = []
for sp in ["train", "val", "test"]:
    idx_list = {"train": train_idx, "val": val_idx, "test": test_idx}[sp]
    yt = y_all[idx_list].flatten()
    for mname, col in MODEL_COLS.items():
        yp = data.loc[idx_list, col].values
        m = metrics_dict(yt, yp)
        summary_rows.append({"split": sp, "model": mname, **m})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("model_comparison_summary.csv", index=False)
print("\nSaved: model_comparison_summary.csv")

# ═══════════════════════════════════════════════════════════════════
# 9.  FIGURE 1: PER-CONDITION HEATMAPS (Test R², RMSE, AARE)
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating figures …")

tc = cond_df[cond_df["split"] == "test"].copy()
models_to_plot = ["ANN", "PGNN", "PGNN+λA"]  # skip SCAM (missing RT–200°C)

fig, axes = plt.subplots(3, 3, figsize=(18, 16))

for col_i, mname in enumerate(models_to_plot):
    m_data = tc[tc["model"] == mname]
    for row_i, (met, cmap, vmin, vmax, fmt) in enumerate([
        ("R2",   "RdYlGn",   0.7,  1.0,  ".3f"),
        ("RMSE", "RdYlGn_r", 0,    30,   ".1f"),
        ("AARE", "RdYlGn_r", 0,    25,   ".1f"),
    ]):
        ax = axes[row_i, col_i]
        piv = m_data.pivot_table(index="T_C", columns="strain_rate", values=met)
        if piv.empty:
            ax.set_visible(False)
            continue
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{sr:.3f}" for sr in piv.columns], rotation=45, fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{T:.0f}" for T in piv.index], fontsize=8)
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                v = piv.values[i, j]
                if np.isfinite(v):
                    color = "white" if (met != "R2" and v > (vmax*0.6)) or (met == "R2" and v < 0.85) else "black"
                    ax.text(j, i, f"{v:{fmt}}", ha="center", va="center", fontsize=7, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8)
        if col_i == 0:
            ax.set_ylabel(f"{met}\nTemperature (°C)")
        if row_i == 0:
            ax.set_title(mname, fontsize=13, fontweight="bold")
        if row_i == 2:
            ax.set_xlabel("Strain Rate (s⁻¹)")

plt.suptitle("Per-Condition Metrics — Test Set", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("fig_percondition_heatmap.png")
plt.close()
print("  ✓ fig_percondition_heatmap.png")

# ═══════════════════════════════════════════════════════════════════
# 10.  FIGURE 2: BAR CHART — Test AARE per condition
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 6))

conditions_sorted = sorted(tc["condition"].unique(), key=lambda c: (
    tc[(tc["condition"]==c) & (tc["model"]=="PGNN+λA")]["AARE"].values[0]
    if len(tc[(tc["condition"]==c) & (tc["model"]=="PGNN+λA")]) > 0 else 999
), reverse=True)

x = np.arange(len(conditions_sorted))
w = 0.25
colors = {"ANN": "#e41a1c", "PGNN": "#ff7f00", "PGNN+λA": "#377eb8"}

for i, mname in enumerate(["ANN", "PGNN", "PGNN+λA"]):
    vals = []
    for cond in conditions_sorted:
        row = tc[(tc["condition"]==cond) & (tc["model"]==mname)]
        vals.append(row["AARE"].iloc[0] if len(row) > 0 and np.isfinite(row["AARE"].iloc[0]) else 0)
    ax.bar(x + i*w - w, vals, w, label=mname, color=colors[mname], alpha=0.8)

ax.set_xticks(x)
T_C_map = tc.groupby("condition")["T_C"].first()
SR_map  = tc.groupby("condition")["strain_rate"].first()
ax.set_xticklabels([f"{T_C_map[c]:.0f}°C\n{SR_map[c]:.3f}" for c in conditions_sorted],
                   fontsize=7, rotation=45, ha="right")
ax.set_ylabel("AARE (%)")
ax.set_title("Per-Condition Test AARE — Model Comparison (worst → best)")
ax.legend()
ax.axhline(10, color="gray", ls="--", alpha=0.5, label="10% threshold")
plt.tight_layout()
plt.savefig("fig_percondition_bar.png")
plt.close()
print("  ✓ fig_percondition_bar.png")

# ═══════════════════════════════════════════════════════════════════
# 11.  FIGURE 3: LEARNED PARAMS VS SCAM (Physical Insight)
# ═══════════════════════════════════════════════════════════════════
eps_smooth = np.linspace(0.05, 0.27, 200)
scam_curves = {k: np.polyval(v, eps_smooth) for k, v in SCAM_POLY.items()}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
param_info = {
    "alpha": ("α (MPa⁻¹)", 1.0),
    "n":     ("n (stress exponent)", 1.0),
    "Q":     ("Q (kJ/mol)", 1e-3),
    "lnA":   ("ln A", 1.0),
}

for ax, pname in zip(axes.flat, param_info):
    label, scale = param_info[pname]

    # Learned values colored by temperature
    sc = ax.scatter(
        eps_raw, params_la[pname] * scale,
        c=data["T_C"].values, cmap="coolwarm", s=8, alpha=0.5,
        edgecolors="none", label="PGNN+λA learned",
    )

    # SCAM polynomial (250-450°C range)
    scam_y = scam_curves[pname] * scale
    # Only plot if values are reasonable
    if pname == "alpha":
        mask_ok = (scam_y > 0) & (scam_y < 0.1)
    elif pname == "n":
        mask_ok = (scam_y > 0) & (scam_y < 20)
    elif pname == "Q":
        mask_ok = (scam_y > 50) & (scam_y < 400)
    elif pname == "lnA":
        mask_ok = (scam_y > 5) & (scam_y < 60)
    else:
        mask_ok = np.ones_like(scam_y, dtype=bool)

    if mask_ok.any():
        ax.plot(eps_smooth[mask_ok], scam_y[mask_ok], "k--", lw=2, alpha=0.7,
                label="SCAM polynomial (250–450°C)")

    ax.set_xlabel("True Strain")
    ax.set_ylabel(label)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    plt.colorbar(sc, ax=ax, label="T (°C)", shrink=0.8)

plt.suptitle("Learned Arrhenius Parameters vs SCAM — PGNN+λA", fontsize=14)
plt.tight_layout()
plt.savefig("fig_learned_params_vs_scam.png")
plt.close()
print("  ✓ fig_learned_params_vs_scam.png")

# ═══════════════════════════════════════════════════════════════════
# 12.  FIGURE 4: PARAMETER EVOLUTION WITH TEMPERATURE (Box-plots)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

temps_sorted = sorted(data["T_C"].unique())

for ax, pname in zip(axes.flat, param_info):
    label, scale = param_info[pname]
    box_data = []
    box_labels = []
    for T in temps_sorted:
        vals = data[data["T_C"] == T][f"learned_{pname}_lA"].values * scale
        box_data.append(vals)
        box_labels.append(f"{T:.0f}")

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    widths=0.6, showfliers=False)
    # Color boxes by temperature
    cmap = plt.cm.coolwarm
    for i, patch in enumerate(bp["boxes"]):
        frac = i / (len(temps_sorted) - 1) if len(temps_sorted) > 1 else 0.5
        patch.set_facecolor(cmap(frac))
        patch.set_alpha(0.7)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs Temperature", fontsize=12)
    ax.tick_params(axis="x", rotation=45)

plt.suptitle("Learned Parameter Distributions by Temperature — PGNN+λA", fontsize=14)
plt.tight_layout()
plt.savefig("fig_param_evolution_T.png")
plt.close()
print("  ✓ fig_param_evolution_T.png")

# ═══════════════════════════════════════════════════════════════════
# 13.  FIGURE 5: ACTIVATION ENERGY ANALYSIS
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

Q_learned = params_la["Q"] / 1000  # kJ/mol

# 5a. Q distribution histogram
ax = axes[0]
ax.hist(Q_learned, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
ax.axvline(142, color="red", ls="--", lw=2, label="Al self-diffusion\n(~142 kJ/mol)")
ax.axvspan(130, 160, alpha=0.15, color="red", label="Literature range\nfor Al alloys")
ax.axvline(np.median(Q_learned), color="green", ls="-", lw=2,
           label=f"Median learned\n({np.median(Q_learned):.1f} kJ/mol)")
ax.set_xlabel("Q (kJ/mol)")
ax.set_ylabel("Count")
ax.set_title("Activation Energy Distribution")
ax.legend(fontsize=8)

# 5b. Q vs temperature
ax = axes[1]
for sr_val in sorted(data["strain_rate"].unique()):
    mask = data["strain_rate"] == sr_val
    ax.scatter(data.loc[mask, "T_C"], Q_learned[mask],
               s=10, alpha=0.5, label=f"{sr_val} s⁻¹")
ax.axhline(142, color="red", ls="--", lw=1.5, alpha=0.7)
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Q (kJ/mol)")
ax.set_title("Learned Q vs Temperature")
ax.legend(fontsize=7, ncol=1, loc="best")

# 5c. Q vs strain
ax = axes[2]
sc = ax.scatter(eps_raw, Q_learned, c=data["T_C"].values,
                cmap="coolwarm", s=8, alpha=0.5)
# SCAM Q for comparison (in valid range only)
scam_Q_smooth = np.polyval(SCAM_POLY["Q"], eps_smooth) / 1000
valid_Q = (scam_Q_smooth > 50) & (scam_Q_smooth < 400)
if valid_Q.any():
    ax.plot(eps_smooth[valid_Q], scam_Q_smooth[valid_Q], "k--", lw=2, label="SCAM polynomial")
ax.axhline(142, color="red", ls="--", lw=1.5, alpha=0.5, label="Al self-diffusion")
ax.set_xlabel("True Strain")
ax.set_ylabel("Q (kJ/mol)")
ax.set_title("Learned Q vs Strain")
ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="T (°C)")

plt.suptitle("Activation Energy Analysis — PGNN+λA", fontsize=14)
plt.tight_layout()
plt.savefig("fig_activation_energy.png")
plt.close()
print("  ✓ fig_activation_energy.png")

# ═══════════════════════════════════════════════════════════════════
# 14.  FIGURE 6: PARAMETER CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (tag, suffix) in zip(axes, [("PGNN+λA", "_lA"), ("PGNN (no λ)", "_noL")]):
    param_df = pd.DataFrame({
        "α": data[f"learned_alpha{suffix}"],
        "n": data[f"learned_n{suffix}"],
        "Q (kJ/mol)": data[f"learned_Q{suffix}"] / 1000,
        "ln A": data[f"learned_lnA{suffix}"],
        "T (°C)": data["T_C"],
        "ε̇ (s⁻¹)": data["strain_rate"],
        "ε": data["eps_true"],
    })
    corr = param_df.corr()
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title(tag, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Parameter–Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.savefig("fig_correlation_matrix.png")
plt.close()
print("  ✓ fig_correlation_matrix.png")

# ═══════════════════════════════════════════════════════════════════
# 15.  PHYSICAL INSIGHT SUMMARY (printed)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHYSICAL INSIGHT SUMMARY")
print("=" * 60)

# Q analysis
Q_med = np.median(Q_learned)
Q_mean = np.mean(Q_learned)
Q_std = np.std(Q_learned)
print(f"\nActivation Energy Q:")
print(f"  Learned (PGNN+λA):  {Q_mean:.1f} ± {Q_std:.1f} kJ/mol  (median: {Q_med:.1f})")
print(f"  Al self-diffusion:  ~142 kJ/mol (literature)")
print(f"  Al 6xxx series:     130–180 kJ/mol (typical range)")
if 130 < Q_med < 180:
    print(f"  → Learned Q is within expected physical range ✓")
else:
    print(f"  → Learned Q is outside typical range — investigate")

# α analysis
alpha_vals = params_la["alpha"]
print(f"\nStress multiplier α:")
print(f"  Learned: {np.mean(alpha_vals):.4f} ± {np.std(alpha_vals):.4f} MPa⁻¹")
print(f"  Typical for Al alloys: 0.01–0.04 MPa⁻¹")

# n analysis
n_vals = params_la["n"]
print(f"\nStress exponent n:")
print(f"  Learned: {np.mean(n_vals):.2f} ± {np.std(n_vals):.2f}")
print(f"  Typical for Al alloys: 3–8")
print(f"  n > 5 suggests dislocation climb mechanism")

# Temperature dependence
print(f"\nTemperature dependence of Q:")
for T in [25, 150, 250, 350, 450]:
    mask = data["T_C"] == T
    if mask.any():
        q_at_T = Q_learned[mask]
        print(f"  T={T:3d}°C: Q = {np.mean(q_at_T):.1f} ± {np.std(q_at_T):.1f} kJ/mol")

# Comparison between PGNN and PGNN+λA parameters
print(f"\nParameter comparison (PGNN vs PGNN+λA, 250–450°C):")
hot_mask = data["T_C"] >= 250
for pname in ["alpha", "n", "Q", "lnA"]:
    v_nolam = data.loc[hot_mask, f"learned_{pname}_noL"]
    v_lam   = data.loc[hot_mask, f"learned_{pname}_lA"]
    scale = 1e-3 if pname == "Q" else 1.0
    unit = " kJ/mol" if pname == "Q" else " MPa-1" if pname == "alpha" else ""
    m1 = v_nolam.mean()*scale
    s1 = v_nolam.std()*scale
    m2 = v_lam.mean()*scale
    s2 = v_lam.std()*scale
    print(f"  {pname:<5}: PGNN = {m1:.3f}+/-{s1:.3f}  |  PGNN+lA = {m2:.3f}+/-{s2:.3f}{unit}")

print("\n" + "=" * 60)
print("All outputs saved. Done.")
print("=" * 60)
