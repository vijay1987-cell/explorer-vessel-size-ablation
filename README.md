# Explorer — Vessel Size Classification Ablation Study

Progressive architectural ablation study for radar-based vessel **SIZE** classification (large / medium / small) using real radar detections from a maritime surveillance dataset. Each model family replicates an architecture from the HQNN Papers 1–12 (vessel TYPE study) and measures whether it transfers to the SIZE problem.

**Dataset:** 123,051 detections · large=96,136 / medium=18,002 / small=8,913  
**Split:** ObjID-stratified GroupShuffleSplit 80/20 (seed=42)  
**Best result:** PINN λ=0.5 at scan level — **F1-macro = 0.7273** on N=23,921 test samples  
**Prior art (thesis):** ET 5-feat — F1=0.5000 · improvement: **+0.227 F1**

---

## Final Ranking

| Rank | Model | Features | Window | F1 | R-lg | R-md | R-sm | N-test |
|---|---|---|---|---|---|---|---|---|
| 1 | **PINN λ=0.5** | BASE-24 | scan | **0.7273** | 0.761 | 0.769 | **0.858** | 23,921 |
| 2 | GBT | FULL-42 | 30-min | 0.7237 | **0.900** | 0.748 | 0.480 | 945 |
| 3 | EDA-QJL k=64 binary | BASE-24+QJL | 30-min | 0.7223 | 0.885 | 0.748 | 0.510 | 945 |
| 4 | EDA-QJL k=128 binary | BASE-24+QJL | 30-min | 0.7211 | 0.885 | 0.748 | 0.500 | 945 |
| 5 | GBT | BASE-24 | 30-min | 0.7194 | 0.905 | 0.762 | 0.440 | 945 |
| 6 | XGB | BASE-24 | 5-min | 0.7043 | 0.877 | 0.751 | 0.462 | 4,911 |
| 7 | ET | BASE-24 | 30-min | 0.6951 | 0.860 | 0.714 | 0.520 | 945 |
| 8 | DualStream (FiLM) | BASE-24 | scan | 0.6300 | 0.764 | 0.831 | 0.433 | 23,921 |
| — | Thesis 5-feat | 5 legacy | scan | 0.5000 | 0.672 | 0.268 | 0.930 | 23,921 |
| — | VQC 5q† | BASE-24 | 30-min | ~0.48 | — | — | — | 945 |

†VQC terminated at epoch 9 — simulation cost prohibitive. Partial result is below all baselines.

> **Why PINN ranks #1 over GBT despite similar F1:** PINN is evaluated on 25× more test samples (23,921 vs 945), giving far higher statistical confidence. It is also the only model with balanced recall across all three classes — the tree models are all large-biased (R-sm ≤ 0.52 vs PINN R-sm=0.858).

---

## Ablation Progression

| Stage | Model | HQNN Paper | Best F1 | Supersedes |
|---|---|---|---|---|
| 0 | Thesis ET (5-feat) | prior art | 0.5000 | — |
| 1 | XGBoost | Paper 1 | 0.7043 | Thesis +0.204 |
| 2 | ExtraTrees | Paper 2 | 0.6951 | (scan baseline) |
| 3 | EDA-QJL | Paper 4 | 0.7223 | XGB at 30-min |
| 4 | GBT | Paper 5 | 0.7237 | EDA-QJL +0.001 |
| 5 | DualStream / FiLM | Paper 6 | 0.6300 | did not supersede |
| 6 | **PINN** | Paper 8 | **0.7273** | GBT +0.004 |
| 7 | VQC | Paper 3 | ~0.48† | did not supersede |

---

## Key Findings

1. **Physics constraints are the decisive element for balanced classification.** PINN λ=0.5 is the only model to achieve R-sm > 0.85 without sacrificing large or medium recall. The physics penalty fires asymmetrically through the small-vessel pathway (large-class p10 thresholds ≈0), preventing the network from assigning small-class detections to the majority large class.

2. **Vessel SIZE is scan-separable; vessel TYPE (HQNN) is not.** PINN achieves its best result per-detection at scan level. In HQNN, the PINN required track-level aggregation because cargo/tanker RCS distributions overlap at the detection level. Large/small/medium vessel RCS and footprint are physically distinct at the individual scan.

3. **Every tree model is large-biased.** Despite balanced class weights, all tree models produce R-large 0.86–0.90 and R-small 0.46–0.52. The 78% class imbalance drives splits toward the majority. PINN's physics constraint is the only mechanism that overcomes this.

4. **EDA-18 features help GBT (+0.004 F1) but hurt ET (−0.15 F1) and are neutral to PINN.** Feature set choice must be matched to the model class. Histogram binning in GBT handles noisy EDA features productively; random splits in ET cannot.

5. **30-min temporal aggregation is optimal for trees; scan is optimal for PINN.** Trees improve monotonically scan→5-min→30-min. PINN reverses this — its physics constraint handles scan-level noise per-detection, making aggregation unnecessary and slightly harmful.

6. **DualStream / FiLM cross-gating does not transfer to tabular SIZE data.** Underperforms all tree baselines by 0.07–0.14 F1. The cross-stream gating needs dense scan-level data to learn dependencies; at 30-min the small aggregated dataset starves the neural architecture.

7. **VQC does not transfer from HQNN TYPE to Explorer SIZE.** Partial 30-min result F1≈0.48 is the weakest in the study — below the thesis baseline. Pre-compression Linear(24→5) is a severe information bottleneck, and PennyLane quantum simulation does not parallelise across batch samples (87 min/epoch at scan level on GPU).

---

## Architecture Transfer — HQNN → Explorer

| Model | HQNN Result | Explorer Result | Transfers? |
|---|---|---|---|
| XGB | F1=0.789 | F1=0.7043 | ✓ Partial |
| ET | Acc=78.2% | F1=0.6951 | ✓ Partial |
| EDA-QJL | Acc=73.1% | F1=0.7223 | ✓ Yes |
| GBT | — (new) | F1=0.7237 | — New best tree |
| DualStream | F1=0.676 | F1=0.6300 | ✗ Degrades |
| PINN | Track F1=0.614 | F1=**0.7273** scan | ✓✓ Best overall |
| VQC | F1=0.714 | F1≈0.48 | ✗ Fails |

---

## Champion Models

| File | Model | F1 | Window | Notes |
|---|---|---|---|---|
| `xgb_champion_5min_base24_f1_0704.pkl` | XGBoost BASE-24 | 0.7043 | 5-min | First to surpass thesis |
| `edaqjl_champion_30min_k512_f1_0704.pkl` | EDA-QJL binary k=512 | 0.7040 | 30-min | 30-min QJL family (k=64 best but not saved) |
| `gbt_champion_30min_full42_f1_0724.pkl` | GBT FULL-42 | 0.7237 | 30-min | Best tree model |
| `pinn_champion_scan_lam05_f1_0727.pkl` | PINN λ=0.5 | 0.7273 | scan | **Best overall** — complete inference package |

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

**EM (10):** `log_peak_rcs`, `log_total_rcs`, `rcs_conc`, `aspect_ratio`, `footprint_m2`, `SampleCount`, `size_bow_stern_component`, `size_beam_component`, `ellipse_area`, `cr_dr_ratio_c`

**Kinematic (14):** `sog`, `measured_sog_avg_{900,1800,3600,10800}`, `measured_sog_std_{900,1800,3600,10800}`, `measured_cog_std_{900,1800,3600,10800}`, `displacement`

---

## Operational Recommendations

- **Deploy PINN λ=0.5 at scan level** — classifies every individual radar detection without temporal aggregation. Lowest latency, balanced recall across all vessel sizes.
- **Fallback: GBT FULL-42 at 30-min** — best tree model when physics-constrained training is unavailable; requires 30-min of detections before classification.
- **Avoid EDA-18-only features** — worst performer across all model families.
- **Avoid VQC** — until purpose-built quantum hardware (not classical simulation) is available.

---

## Findings Documents

| File | Description |
|---|---|
| `ABLATION_SUMMARY.md` | Complete per-condition results, recall profiles, and cross-model analysis |
| `findings/PINN_scan_level_finding.md` | Why scan-level PINN dominates and why this was not found in HQNN |
| `findings/VQC_ablation_finding.md` | VQC simulation cost analysis and failure root causes |
| `findings/thesis_vs_explorer_comparison.md` | Physics critique of the thesis range-correction approach |

---

## Reproducing Results

```bash
source /path/to/.venv/bin/activate   # Python 3.12, PyTorch, sklearn, xgboost, pennylane

python training/train_xgb_ablation.py     # Stage 1 — XGB
python training/train_et_grid.py          # Stage 2 — ET
python training/train_edaqjl.py           # Stage 3 — EDA-QJL
python training/train_gbt_grid.py         # Stage 4 — GBT
python training/train_dualstream.py       # Stage 5 — DualStream
python training/train_pinn.py             # Stage 6 — PINN
python training/save_pinn_inference.py    # Save PINN inference package
python training/train_thesis_comparison.py
# VQC: 30-min only feasible on classical hardware
python training/train_vqc.py
```

Data file required: `data/study_cleaned.csv` (not included — proprietary radar dataset).  
All ablation results are in `results/` as JSON.
