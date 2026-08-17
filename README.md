# Explorer — Vessel Size Classification Ablation Study

Progressive architectural ablation study for radar-based vessel **SIZE** classification (large / medium / small) using real radar detections from a maritime surveillance dataset.

**Dataset:** 123,051 detections · large=96,136 / medium=18,002 / small=8,913  
**Split:** ObjID-stratified GroupShuffleSplit 80/20 (seed=42)  
**Best result:** PINN λ=0.5 at scan level — **F1-macro = 0.7273** on N=23,921 test samples

---

## Ablation Progression

Each model class adds one architectural element over the previous. The table below shows the best result per model family and which result it superseded.

| Stage | Model | Features | Window | F1 | Supersedes |
|---|---|---|---|---|---|
| 0 — Thesis | ET 4-feat (thesis) | 4 legacy features | scan | 0.4017 | — |
| 0 — Thesis | ET 5-feat (thesis) | 5 legacy features | scan | 0.5000 | — |
| 1 — XGB | XGBoost BASE-24 | 24 features | 5-min | **0.7043** | Thesis +0.20 |
| 2 — ET | ExtraTrees BASE-24 | 24 features | scan | 0.6894 | (scan baseline) |
| 3 — EDA-QJL | Binary QJL k=64 | BASE-24 + QJL(EDA-18) | 30-min | **0.7223** | XGB 30-min |
| 4 — GBT | HistGBT FULL-42 | 42 features | 30-min | **0.7237** | EDA-QJL +0.0014 |
| 5 — DualStream | FiLM cross-gating | BASE-24 | scan | 0.6300 | (did not supersede) |
| 6 — PINN | Physics-constrained NN | BASE-24, λ=0.5 | scan | **0.7273** | GBT +0.0036 |

> **Note:** GBT was the penultimate champion (F1=0.7237, 30-min window) but was not saved to disk — it is not included in `models/`. Re-run `training/train_gbt_grid.py` to reproduce it.

---

## Champion Models

Saved in `models/` — one file per superseding result:

| File | Model | F1 | Window | N-test | Notes |
|---|---|---|---|---|---|
| `xgb_champion_5min_base24_f1_0704.pkl` | XGBoost BASE-24 | 0.7043 | 5-min | 4,911 | First model to surpass thesis |
| `edaqjl_champion_30min_k512_f1_0704.pkl` | EDA-QJL binary k=512 | 0.7040 | 30-min | 945 | 30-min QJL family best (k=64 was 0.7223 but not saved) |
| `pinn_champion_scan_lam05_f1_0727.pkl` | PINN λ=0.5 | 0.7273 | scan | 23,921 | **Best overall** — complete inference package (weights + scalers + thresholds) |

### Loading the PINN inference package

```python
import pickle, torch
import torch.nn as nn

with open('models/pinn_champion_scan_lam05_f1_0727.pkl', 'rb') as f:
    pkg = pickle.load(f)

# pkg keys: model_state, sc_em, sc_kin, em_cols, kin_cols,
#           classes, thresholds, d_em, d_kin, lambda, window, f1, acc

class PINNModel(nn.Module):
    def __init__(self, d_em, d_kin, n_cls=3):
        super().__init__()
        self.em_branch = nn.Sequential(
            nn.Linear(d_em, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 32),   nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.2),
        )
        self.kin_branch = nn.Sequential(
            nn.Linear(d_kin, 96), nn.BatchNorm1d(96), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(96, 48),    nn.BatchNorm1d(48), nn.GELU(), nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(80, 48), nn.BatchNorm1d(48), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(48, 24), nn.GELU(), nn.Linear(24, n_cls),
        )
    def forward(self, x_em, x_kin):
        return self.fusion(torch.cat([self.em_branch(x_em), self.kin_branch(x_kin)], dim=1))

model = PINNModel(pkg['d_em'], pkg['d_kin'])
model.load_state_dict(pkg['model_state'])
model.eval()

# Preprocess new detections:
# X_em_raw  = df[pkg['em_cols']].fillna(0).values.astype('float32')
# X_kin_raw = df[pkg['kin_cols']].fillna(0).values.astype('float32')
# X_em  = pkg['sc_em'].transform(X_em_raw).astype('float32')
# X_kin = pkg['sc_kin'].transform(X_kin_raw).astype('float32')
# with torch.no_grad():
#     preds = model(torch.tensor(X_em), torch.tensor(X_kin)).argmax(1).numpy()
# labels = [pkg['classes'][p] for p in preds]
```

---

## Feature Sets

| Set | N | Description |
|---|---|---|
| BASE-24 | 24 | 10 EM features + 14 kinematic (SOG/COG rolling stats + displacement) |
| EDA-18 | 18 | Novel: TotalAmplitude rolling, rangeStd, azimuthStd, cogStd rolling, rgw, azw |
| FULL-42 | 42 | BASE-24 + EDA-18 |

**EM features (10):** `log_peak_rcs`, `log_total_rcs`, `rcs_conc`, `aspect_ratio`, `footprint_m2`, `SampleCount`, `size_bow_stern_component`, `size_beam_component`, `ellipse_area`, `cr_dr_ratio_c`

**Kinematic features (14):** `sog`, `measured_sog_avg_{900,1800,3600,10800}`, `measured_sog_std_{900,1800,3600,10800}`, `measured_cog_std_{900,1800,3600,10800}`, `displacement`

---

## Key Findings

1. **PINN physics constraint improves small-vessel recall dramatically** — R-small=0.858 vs 0.42–0.53 for tree models. The class-weighted CE loss + physics penalty prevents collapse to the large-vessel majority.

2. **Vessel SIZE is scan-separable; vessel TYPE (HQNN) is not** — physics constraints work per detection for SIZE because large/small vessels are physically distinct in RCS and footprint at the individual scan level. TYPE (cargo vs tanker) requires track-level aggregation.

3. **EDA features help GBT (+0.013 F1) but hurt PINN** — the dual-branch PINN architecture already extracts sufficient signal from 10 EM features; adding EDA-18 increases noise.

4. **DualStream cross-gating (FiLM) fails for tabular SIZE data** — underperforms tree baselines by 0.08–0.16 F1. Single-embedding cross-attention is not a meaningful operation for tabular detection features.

5. **Thesis physics are flawed** — see `findings/thesis_vs_explorer_comparison.md` for a detailed physics critique of the range-correction exponent and geometry features.

---

## Findings Documents

- `findings/PINN_scan_level_finding.md` — Full analysis of why scan-level PINN is the best result and why this was not discovered in prior work
- `findings/thesis_vs_explorer_comparison.md` — Physics critique and quantitative comparison against the thesis baseline

---

## Reproducing Results

```bash
# Environment
source /path/to/.venv/bin/activate   # Python 3.12, PyTorch, sklearn, xgboost, pennylane

# Stage 1 — XGB ablation
python training/train_xgb_ablation.py

# Stage 2 — ExtraTrees grid
python training/train_et_grid.py

# Stage 3 — EDA-QJL
python training/train_edaqjl.py

# Stage 4 — GBT grid
python training/train_gbt_grid.py

# Stage 5 — DualStream
python training/train_dualstream.py

# Stage 6 — PINN ablation
python training/train_pinn.py

# Save PINN inference package
python training/save_pinn_inference.py

# Thesis baseline comparison
python training/train_thesis_comparison.py

# VQC (quantum, slow on CPU)
python training/train_vqc.py
```

Data file required: `data/study_cleaned.csv` (not included — proprietary radar dataset).

---

## Results Summary

All ablation results are in `results/` as JSON. The full 60-condition comparison table spans XGB (6 conditions), ET (9), EDA-QJL (24), GBT (9), DualStream (6), and PINN (6).
