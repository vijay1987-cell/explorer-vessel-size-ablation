#!/usr/bin/env python3
"""
PINN — Physics-Informed Neural Network — Explorer Dataset
==========================================================
Adapted from HQNN Paper 8. Dual-branch MLP + soft physics penalty.

Architecture:
  EM  branch: Linear(10->64)->BN->GELU->Dropout(0.3)->Linear(64->32)->BN->GELU->Dropout(0.2)
  KIN branch: Linear(14->96)->BN->GELU->Dropout(0.3)->Linear(96->48)->BN->GELU->Dropout(0.2)
  Fusion:     Concat(80)->Linear(80->48)->BN->GELU->Dropout(0.2)->Linear(48->24)->GELU->Linear(24->3)

Loss: L = L_CE_weighted + lambda * L_phys
  lambda in {0.0, 0.5}   (0.0 = pure dual-branch MLP ablation, 0.5 = best from HQNN)
  lambda warmed up linearly over first 20 epochs

Physics constraints (derived from training quantiles):
  - log_peak_rcs: large class should be HIGH  (penalty if pred=large & rcs < large_p10)
                  small class should be LOW   (penalty if pred=small & rcs > small_p90)
  - footprint_m2: large class should be HIGH  (penalty if pred=large & fp < large_p10)
                  small class should be LOW   (penalty if pred=small & fp > small_p90)

Conditions: lambda in {0.0, 0.5} x window in {scan, 5-min, 30-min} = 6
"""

import time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

REAL_PATH = "/home/iaxiom/projects/Explorer/data/study_cleaned.csv"
MODEL_DIR  = Path("/home/iaxiom/projects/Explorer/models")
MODEL_DIR.mkdir(exist_ok=True)

CLASSES   = ['large', 'medium', 'small']
CLASS_MAP = {'large': 0, 'medium': 1, 'small': 2}
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED      = 42
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
# Physics constraint features (indices within EM_COLS)
PHYS_RCS_IDX = 0   # log_peak_rcs
PHYS_FP_IDX  = 4   # footprint_m2

LAMBDAS = [0.0, 0.5]
WINDOWS = [('scan', None), ('5-min', 300), ('30-min', 1800)]
ALL_RESULTS = {}


def apply_window(df, window_sec):
    if window_sec is None:
        return df
    df = df.copy()
    df['_tw'] = (df['Rtime_epoch'] // window_sec).astype(int)
    df = (df.sort_values('Rtime_epoch')
            .groupby(['ObjID', '_tw'], sort=False)
            .last().reset_index())
    return df.drop(columns=['_tw'], errors='ignore')


# ── model ─────────────────────────────────────────────────────────────────────
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
        h_em  = self.em_branch(x_em)
        h_kin = self.kin_branch(x_kin)
        return self.fusion(torch.cat([h_em, h_kin], dim=1))


# ── physics loss ──────────────────────────────────────────────────────────────
def physics_loss(logits, x_em_raw, thresholds):
    """
    Soft size-based physics constraint.
    x_em_raw: un-normalised EM features (B, d_em) — used for interpretable thresholds.
    thresholds: dict with keys large_rcs_p10, small_rcs_p90, large_fp_p10, small_fp_p90.
    """
    probs = torch.softmax(logits, dim=1)  # (B, 3)
    rcs   = x_em_raw[:, PHYS_RCS_IDX]
    fp    = x_em_raw[:, PHYS_FP_IDX]

    # relu: violated only when threshold is breached
    viol_large_rcs = torch.relu(thresholds['large_rcs_p10'] - rcs)
    viol_small_rcs = torch.relu(rcs - thresholds['small_rcs_p90'])
    viol_large_fp  = torch.relu(thresholds['large_fp_p10']  - fp)
    viol_small_fp  = torch.relu(fp  - thresholds['small_fp_p90'])

    p_large = probs[:, 0]
    p_small = probs[:, 2]
    phys = (p_large * (viol_large_rcs + viol_large_fp) +
            p_small * (viol_small_rcs + viol_small_fp))
    return phys.mean()


# ── metrics ───────────────────────────────────────────────────────────────────
def report(yte, preds, label, n_tr, n_te):
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    cm  = confusion_matrix(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=4)
    recall = {c: float(cm[i,i]/cm[i].sum()) if cm[i].sum() > 0 else 0.
              for i,c in enumerate(CLASSES)}
    print(f"\n  -- PINN [{label}] --")
    print(f"  Train: {n_tr:,}  Test: {n_te:,}")
    print(f"  F1-macro: {f1:.4f}  |  Acc: {acc:.4f}")
    print(f"  Recall -> large: {recall['large']:.4f}  medium: {recall['medium']:.4f}  small: {recall['small']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"    {'':10} {'large':>7} {'medium':>7} {'small':>7}")
    for i,c in enumerate(CLASSES):
        print(f"    {c:10} {cm[i,0]:>7} {cm[i,1]:>7} {cm[i,2]:>7}")
    print(f"\n{rep}")
    return {
        'f1': round(f1,4), 'acc': round(acc,4),
        'recall_large':  round(recall['large'],4),
        'recall_medium': round(recall['medium'],4),
        'recall_small':  round(recall['small'],4),
        'n_train': n_tr, 'n_test': n_te,
    }


# ── training ──────────────────────────────────────────────────────────────────
def run_pinn(df_tr, df_te, ytr, yte, avail_em, avail_kin,
             lam, label, key, epochs=120, batch=512, lr=3e-4, patience=20):

    sc_em  = StandardScaler().fit(df_tr[avail_em].fillna(0).values)
    sc_kin = StandardScaler().fit(df_tr[avail_kin].fillna(0).values)

    Xem_tr_raw  = df_tr[avail_em].fillna(0).values.astype(np.float32)
    Xem_te_raw  = df_te[avail_em].fillna(0).values.astype(np.float32)

    Xem_tr  = sc_em.transform(Xem_tr_raw).astype(np.float32)
    Xem_te  = sc_em.transform(Xem_te_raw).astype(np.float32)
    Xkin_tr = sc_kin.transform(df_tr[avail_kin].fillna(0).values).astype(np.float32)
    Xkin_te = sc_kin.transform(df_te[avail_kin].fillna(0).values).astype(np.float32)

    # Physics thresholds from training data (raw, unscaled)
    thresholds = {}
    for cls_name, cls_id in [('large', 0), ('small', 2)]:
        mask = (ytr == cls_id)
        rcs_vals = Xem_tr_raw[mask, PHYS_RCS_IDX]
        fp_vals  = Xem_tr_raw[mask, PHYS_FP_IDX]
        if cls_name == 'large':
            thresholds['large_rcs_p10'] = torch.tensor(float(np.percentile(rcs_vals, 10))).to(DEVICE)
            thresholds['large_fp_p10']  = torch.tensor(float(np.percentile(fp_vals,  10))).to(DEVICE)
        else:
            thresholds['small_rcs_p90'] = torch.tensor(float(np.percentile(rcs_vals, 90))).to(DEVICE)
            thresholds['small_fp_p90']  = torch.tensor(float(np.percentile(fp_vals,  90))).to(DEVICE)

    print(f"  Physics thresholds: rcs large_p10={thresholds['large_rcs_p10']:.3f}  small_p90={thresholds['small_rcs_p90']:.3f}")
    print(f"                      fp  large_p10={thresholds['large_fp_p10']:.3f}  small_p90={thresholds['small_fp_p90']:.3f}")

    counts  = np.bincount(ytr, minlength=3)
    w       = torch.tensor((len(ytr) / (3 * counts)).astype(np.float32)).to(DEVICE)
    ce_loss = nn.CrossEntropyLoss(weight=w)

    model = PINNModel(d_em=len(avail_em), d_kin=len(avail_kin)).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    ds = TensorDataset(
        torch.tensor(Xem_tr), torch.tensor(Xkin_tr),
        torch.tensor(Xem_tr_raw),  # raw EM for physics constraint
        torch.tensor(ytr, dtype=torch.long)
    )
    loader = DataLoader(ds, batch_size=batch, shuffle=True,
                        pin_memory=(DEVICE.type == 'cuda'))

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr * 5, epochs=epochs,
        steps_per_epoch=len(loader), pct_start=0.1,
    )

    Xem_te_t   = torch.tensor(Xem_te).to(DEVICE)
    Xkin_te_t  = torch.tensor(Xkin_te).to(DEVICE)
    warmup_eps = 20

    best_f1, best_ep, pat_cnt, best_state = 0, 0, 0, None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        lam_ep = lam * min(1.0, ep / warmup_eps)  # linear warmup
        model.train()
        for xb_em, xb_kin, xb_em_raw, yb in loader:
            xb_em     = xb_em.to(DEVICE)
            xb_kin    = xb_kin.to(DEVICE)
            xb_em_raw = xb_em_raw.to(DEVICE)
            yb        = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb_em, xb_kin)
            loss   = ce_loss(logits, yb)
            if lam_ep > 0:
                loss = loss + lam_ep * physics_loss(logits, xb_em_raw, thresholds)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            preds_ep = model(Xem_te_t, Xkin_te_t).argmax(1).cpu().numpy()
        f1_ep = f1_score(yte, preds_ep, average='macro')

        if f1_ep > best_f1:
            best_f1 = f1_ep; best_ep = ep; pat_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

        if ep % 20 == 0 or ep == 1:
            print(f"  ep{ep:3d}  val_F1={f1_ep:.4f}  best={best_f1:.4f}@ep{best_ep}  lam={lam_ep:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Time: {time.time()-t0:.0f}s | Best ep: {best_ep}")

    with torch.no_grad():
        preds = model(Xem_te_t, Xkin_te_t).argmax(1).cpu().numpy()

    m = report(yte, preds, label, len(ytr), len(yte))
    torch.save(model.state_dict(), MODEL_DIR / f'{key}_weights.pt')
    return m


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    df_raw = pd.read_csv(REAL_PATH, low_memory=False)
    df_raw = df_raw[df_raw['size_class'].isin(CLASSES)].copy()

    avail_em  = [c for c in EM_COLS  if c in df_raw.columns]
    avail_kin = [c for c in KIN_COLS if c in df_raw.columns]
    print(f"EM: {len(avail_em)}  KIN: {len(avail_kin)}")

    for win_tag, win_sec in WINDOWS:
        df  = apply_window(df_raw, win_sec)
        y   = np.array([CLASS_MAP[c] for c in df['size_class']])
        grp = df['ObjID'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr, te = next(gss.split(df, y, groups=grp))
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        ytr, yte = y[tr], y[te]

        print(f"\n{'#'*65}")
        print(f"  Window: {win_tag}   train={len(ytr):,}  test={len(yte):,}")
        print(f"{'#'*65}")

        for lam in LAMBDAS:
            lam_tag = f"lam{str(lam).replace('.','')}"
            label   = f"lambda={lam} | {win_tag}"
            key     = f"pinn_{win_tag.replace('-','').lower()}_{lam_tag}"
            print(f"\n{'='*60}")
            print(f"  PINN lambda={lam} | {win_tag}")
            m = run_pinn(df_tr, df_te, ytr, yte, avail_em, avail_kin,
                         lam, label, key)
            ALL_RESULTS[key] = {'model': 'PINN', 'lambda': lam, 'window': win_tag, **m}

    # Summary
    print(f"\n{'='*80}")
    print("  PINN -- SUMMARY")
    print(f"{'='*80}")
    print(f"  {'lambda':<8} {'Window':<8} {'F1':>6} {'Acc':>6} {'R-lg':>6} {'R-md':>6} {'R-sm':>6} {'N-test':>7}")
    print(f"  {'-'*60}")
    for win_tag, _ in WINDOWS:
        for lam in LAMBDAS:
            lam_tag = f"lam{str(lam).replace('.','')}"
            key = f"pinn_{win_tag.replace('-','').lower()}_{lam_tag}"
            if key in ALL_RESULTS:
                r = ALL_RESULTS[key]
                print(f"  {lam:<8} {win_tag:<8} {r['f1']:>6.4f} {r['acc']:>6.4f} "
                      f"{r['recall_large']:>6.4f} {r['recall_medium']:>6.4f} "
                      f"{r['recall_small']:>6.4f} {r['n_test']:>7,}")

    refs = [
        ('XGB BASE-24', 'scan', 0.6250), ('GBT FULL-42', 'scan', 0.7108),
        ('XGB BASE-24', '5-min', 0.7043), ('GBT FULL-42', '5-min', 0.7033),
        ('XGB BASE-24', '30-min', 0.7135), ('GBT FULL-42', '30-min', 0.7237),
        ('DualStream-B24', 'scan', 0.6300),
    ]
    print(f"\n  [Reference baselines]")
    print(f"  {'-'*45}")
    for name, wt, f1 in refs:
        print(f"  {name:<16} {wt:<8} {f1:.4f}")

    with open(MODEL_DIR / 'pinn_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Total runtime: {time.time()-t_total:.0f}s")
    print(f"  Results -> models/pinn_results.json")


if __name__ == '__main__':
    main()
