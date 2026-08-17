#!/usr/bin/env python3
"""
Re-trains PINN scan lambda=0.5 (best result, F1=0.7273) and saves
a complete inference package: weights + scalers + feature lists + thresholds.

Inference package: models/pinn_scan_lam05_inference.pkl
  keys: model_state, sc_em, sc_kin, em_cols, kin_cols, classes,
        thresholds, d_em, d_kin, f1, acc
"""
import time, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

REAL_PATH = "/home/iaxiom/projects/Explorer/data/study_cleaned.csv"
MODEL_DIR  = Path("/home/iaxiom/projects/Explorer/models")
CLASSES    = ['large', 'medium', 'small']
CLASS_MAP  = {'large': 0, 'medium': 1, 'small': 2}
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED       = 42
LAMBDA     = 0.5
PHYS_RCS_IDX, PHYS_FP_IDX = 0, 4
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

EM_COLS = [
    'log_peak_rcs', 'log_total_rcs', 'rcs_conc',
    'aspect_ratio', 'footprint_m2',
    'SampleCount', 'size_bow_stern_component', 'size_beam_component',
    'ellipse_area', 'cr_dr_ratio_c',
]
KIN_COLS = [
    'sog',
    'measured_sog_avg_900',  'measured_sog_avg_1800',
    'measured_sog_avg_3600', 'measured_sog_avg_10800',
    'measured_sog_std_900',  'measured_sog_std_1800',
    'measured_sog_std_3600', 'measured_sog_std_10800',
    'measured_cog_std_900',  'measured_cog_std_1800',
    'measured_cog_std_3600', 'measured_cog_std_10800',
    'displacement',
]

class PINNModel(nn.Module):
    def __init__(self, d_em=10, d_kin=14, n_cls=3):
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
            nn.Linear(48, 24), nn.GELU(),
            nn.Linear(24, n_cls),
        )
    def forward(self, x_em, x_kin):
        return self.fusion(torch.cat([self.em_branch(x_em), self.kin_branch(x_kin)], dim=1))

def physics_loss(logits, x_em_raw, thresholds):
    probs = torch.softmax(logits, dim=1)
    rcs   = x_em_raw[:, PHYS_RCS_IDX]
    fp    = x_em_raw[:, PHYS_FP_IDX]
    viol  = (probs[:,0] * (torch.relu(thresholds['large_rcs_p10'] - rcs) +
                           torch.relu(thresholds['large_fp_p10']  - fp)) +
             probs[:,2] * (torch.relu(rcs - thresholds['small_rcs_p90']) +
                           torch.relu(fp  - thresholds['small_fp_p90'])))
    return viol.mean()

def main():
    df = pd.read_csv(REAL_PATH, low_memory=False)
    df = df[df['size_class'].isin(CLASSES)].copy()
    y   = np.array([CLASS_MAP[c] for c in df['size_class']])
    grp = df['ObjID'].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(df, y, groups=grp))
    ytr, yte = y[tr], y[te]

    avail_em  = [c for c in EM_COLS  if c in df.columns]
    avail_kin = [c for c in KIN_COLS if c in df.columns]
    df_tr, df_te = df.iloc[tr], df.iloc[te]

    sc_em  = StandardScaler().fit(df_tr[avail_em].fillna(0).values)
    sc_kin = StandardScaler().fit(df_tr[avail_kin].fillna(0).values)

    Xem_tr_raw = df_tr[avail_em].fillna(0).values.astype(np.float32)
    Xem_te_raw = df_te[avail_em].fillna(0).values.astype(np.float32)
    Xem_tr  = sc_em.transform(Xem_tr_raw).astype(np.float32)
    Xem_te  = sc_em.transform(Xem_te_raw).astype(np.float32)
    Xkin_tr = sc_kin.transform(df_tr[avail_kin].fillna(0).values).astype(np.float32)
    Xkin_te = sc_kin.transform(df_te[avail_kin].fillna(0).values).astype(np.float32)

    thresholds = {}
    for cls_name, cls_id in [('large', 0), ('small', 2)]:
        mask = (ytr == cls_id)
        rcs_v = Xem_tr_raw[mask, PHYS_RCS_IDX]
        fp_v  = Xem_tr_raw[mask, PHYS_FP_IDX]
        if cls_name == 'large':
            thresholds['large_rcs_p10'] = torch.tensor(float(np.percentile(rcs_v, 10))).to(DEVICE)
            thresholds['large_fp_p10']  = torch.tensor(float(np.percentile(fp_v,  10))).to(DEVICE)
        else:
            thresholds['small_rcs_p90'] = torch.tensor(float(np.percentile(rcs_v, 90))).to(DEVICE)
            thresholds['small_fp_p90']  = torch.tensor(float(np.percentile(fp_v,  90))).to(DEVICE)

    counts  = np.bincount(ytr, minlength=3)
    w       = torch.tensor((len(ytr) / (3 * counts)).astype(np.float32)).to(DEVICE)
    ce_loss = nn.CrossEntropyLoss(weight=w)

    model = PINNModel(d_em=len(avail_em), d_kin=len(avail_kin)).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    epochs, batch, patience, warmup_eps = 120, 512, 20, 20

    ds = TensorDataset(torch.tensor(Xem_tr), torch.tensor(Xkin_tr),
                       torch.tensor(Xem_tr_raw), torch.tensor(ytr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch, shuffle=True, pin_memory=(DEVICE.type=='cuda'))
    sched  = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4*5, epochs=epochs,
                                                  steps_per_epoch=len(loader), pct_start=0.1)

    Xem_te_t  = torch.tensor(Xem_te).to(DEVICE)
    Xkin_te_t = torch.tensor(Xkin_te).to(DEVICE)

    best_f1, best_ep, pat_cnt, best_state = 0, 0, 0, None
    t0 = time.time()
    for ep in range(1, epochs+1):
        lam_ep = LAMBDA * min(1.0, ep / warmup_eps)
        model.train()
        for xb_em, xb_kin, xb_raw, yb in loader:
            xb_em, xb_kin, xb_raw, yb = xb_em.to(DEVICE), xb_kin.to(DEVICE), xb_raw.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb_em, xb_kin)
            loss   = ce_loss(logits, yb) + lam_ep * physics_loss(logits, xb_raw, thresholds)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            preds_ep = model(Xem_te_t, Xkin_te_t).argmax(1).cpu().numpy()
        f1_ep = f1_score(yte, preds_ep, average='macro')
        if f1_ep > best_f1:
            best_f1 = f1_ep; best_ep = ep; pat_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_cnt += 1
            if pat_cnt >= patience: break
        if ep % 20 == 0 or ep == 1:
            print(f"  ep{ep:3d}  val_F1={f1_ep:.4f}  best={best_f1:.4f}@ep{best_ep}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(Xem_te_t, Xkin_te_t).argmax(1).cpu().numpy()
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    print(f"\n  Final — F1: {f1:.4f}  Acc: {acc:.4f}  Time: {time.time()-t0:.0f}s  Best ep: {best_ep}")

    # Save complete inference package
    pkg = {
        'model_state': best_state,
        'sc_em':       sc_em,
        'sc_kin':      sc_kin,
        'em_cols':     avail_em,
        'kin_cols':    avail_kin,
        'classes':     CLASSES,
        'thresholds':  {k: float(v.cpu()) for k, v in thresholds.items()},
        'd_em':        len(avail_em),
        'd_kin':       len(avail_kin),
        'lambda':      LAMBDA,
        'window':      'scan',
        'f1':          round(f1, 4),
        'acc':         round(acc, 4),
    }
    out = MODEL_DIR / 'pinn_scan_lam05_inference.pkl'
    with open(out, 'wb') as f:
        pickle.dump(pkg, f)
    print(f"  Inference package saved -> {out.name}")
    print(f"  Keys: {list(pkg.keys())}")

if __name__ == '__main__':
    main()
