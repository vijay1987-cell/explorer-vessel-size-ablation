#!/usr/bin/env python3
"""
PINN Deep-Dive — Explorer Dataset (scan level only)
=====================================================
Grid:
  feature_set in {BASE-24 (10+14), FULL-42 (24+18)}
  lambda      in {0.0, 0.1, 0.3, 0.5, 1.0}
  phys_pct    in {p10/p90, p25/p75}   <- threshold percentile pair
  All scan level.  Total: 2 x 5 x 2 = 20 conditions.

Architecture scales with feature set:
  BASE-24: EM(10->64->32), KIN(14->96->48), Fusion(80->48->24->3)
  FULL-42: EM(24->128->64), KIN(18->128->64), Fusion(128->64->32->3)

Physics constraint (size-based):
  log_peak_rcs — large should be HIGH, small should be LOW
  footprint_m2 — large should be HIGH, small should be LOW
  Penalty: p_large * relu(large_pLO - rcs) + p_small * relu(rcs - small_pHI)
           + same for footprint_m2

p10/p90: conservative (original) — fires only for extreme violations
p25/p75: tighter — fires more often, stronger regularisation signal
"""

import time, json, pickle, warnings
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

CLASSES  = ['large', 'medium', 'small']
CLASS_MAP = {'large': 0, 'medium': 1, 'small': 2}
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED     = 42
print(f"Device: {DEVICE}")

# Physics constraint feature indices within EM_COLS
PHYS_RCS_IDX = 0   # log_peak_rcs
PHYS_FP_IDX  = 4   # footprint_m2

EM_BASE = [
    'log_peak_rcs', 'log_total_rcs', 'rcs_conc',
    'aspect_ratio', 'footprint_m2',
    'SampleCount', 'size_bow_stern_component', 'size_beam_component',
    'ellipse_area', 'cr_dr_ratio_c',
]
KIN_BASE = [
    'sog',
    'measured_sog_avg_900',  'measured_sog_avg_1800',
    'measured_sog_avg_3600', 'measured_sog_avg_10800',
    'measured_sog_std_900',  'measured_sog_std_1800',
    'measured_sog_std_3600', 'measured_sog_std_10800',
    'measured_cog_std_900',  'measured_cog_std_1800',
    'measured_cog_std_3600', 'measured_cog_std_10800',
    'displacement',
]
EDA_EM = [
    'measured_TotalAmplitude_avg_900',  'measured_TotalAmplitude_avg_1800',
    'measured_TotalAmplitude_avg_3600', 'measured_TotalAmplitude_avg_10800',
    'measured_rangeStd_900',  'measured_rangeStd_1800',
    'measured_rangeStd_3600', 'measured_rangeStd_10800',
    'measured_azimuthStd_900',  'measured_azimuthStd_1800',
    'measured_azimuthStd_3600', 'measured_azimuthStd_10800',
    'rgw', 'azw',
]
EDA_KIN = [
    'measured_cog_stdlog_900',  'measured_cog_stdlog_1800',
    'measured_cog_stdlog_3600', 'measured_cog_stdlog_10800',
]

FEATURE_SETS = {
    'BASE-24': (EM_BASE,           KIN_BASE),
    'FULL-42': (EM_BASE + EDA_EM,  KIN_BASE + EDA_KIN),
}
LAMBDAS   = [0.0, 0.1, 0.3, 0.5, 1.0]
PHYS_PCTS = {
    'p10p90': (10, 90),
    'p25p75': (25, 75),
}
ALL_RESULTS = {}


# ── model (scales with d_em, d_kin) ──────────────────────────────────────────
def branch_dims(d_in):
    hidden = max(64, d_in * 5)
    hidden = int(round(hidden / 32)) * 32   # nearest 32
    out    = hidden // 2
    return hidden, out

class PINNModel(nn.Module):
    def __init__(self, d_em, d_kin, n_cls=3):
        super().__init__()
        h_em, o_em = branch_dims(d_em)
        h_kin, o_kin = branch_dims(d_kin)
        f_in = o_em + o_kin
        f_h  = max(f_in // 2, 32)

        self.em_branch = nn.Sequential(
            nn.Linear(d_em, h_em), nn.BatchNorm1d(h_em), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h_em, o_em), nn.BatchNorm1d(o_em), nn.GELU(), nn.Dropout(0.2),
        )
        self.kin_branch = nn.Sequential(
            nn.Linear(d_kin, h_kin), nn.BatchNorm1d(h_kin), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(h_kin, o_kin), nn.BatchNorm1d(o_kin), nn.GELU(), nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(f_in, f_h),   nn.BatchNorm1d(f_h), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(f_h,  f_h//2), nn.GELU(),
            nn.Linear(f_h//2, n_cls),
        )

    def forward(self, x_em, x_kin):
        return self.fusion(torch.cat([self.em_branch(x_em), self.kin_branch(x_kin)], dim=1))


# ── physics loss ──────────────────────────────────────────────────────────────
def physics_loss_fn(logits, x_em_raw, thr):
    probs = torch.softmax(logits, dim=1)
    rcs   = x_em_raw[:, PHYS_RCS_IDX]
    fp    = x_em_raw[:, PHYS_FP_IDX]
    viol  = (probs[:,0] * (torch.relu(thr['large_rcs'] - rcs) + torch.relu(thr['large_fp'] - fp)) +
             probs[:,2] * (torch.relu(rcs - thr['small_rcs']) + torch.relu(fp  - thr['small_fp'])))
    return viol.mean()


# ── metrics ───────────────────────────────────────────────────────────────────
def report(yte, preds, label, n_tr, n_te):
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    cm  = confusion_matrix(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=4)
    recall = {c: float(cm[i,i]/cm[i].sum()) if cm[i].sum()>0 else 0.
              for i,c in enumerate(CLASSES)}
    print(f"\n  -- {label} --")
    print(f"  F1: {f1:.4f}  Acc: {acc:.4f}  R-lg: {recall['large']:.4f}  R-md: {recall['medium']:.4f}  R-sm: {recall['small']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"    {'':10} {'large':>7} {'medium':>7} {'small':>7}")
    for i,c in enumerate(CLASSES):
        print(f"    {c:10} {cm[i,0]:>7} {cm[i,1]:>7} {cm[i,2]:>7}")
    print(f"\n{rep}")
    return {'f1':round(f1,4),'acc':round(acc,4),
            'recall_large':round(recall['large'],4),
            'recall_medium':round(recall['medium'],4),
            'recall_small':round(recall['small'],4),
            'n_train':n_tr,'n_test':n_te}


# ── single run ────────────────────────────────────────────────────────────────
def run(Xem_tr, Xkin_tr, Xem_tr_raw, ytr,
        Xem_te, Xkin_te, yte,
        thr, lam, label, key,
        epochs=150, batch=512, lr=3e-4, patience=25, warmup_eps=20):

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    counts  = np.bincount(ytr, minlength=3)
    w       = torch.tensor((len(ytr) / (3*counts)).astype(np.float32)).to(DEVICE)
    ce_loss = nn.CrossEntropyLoss(weight=w)

    model = PINNModel(d_em=Xem_tr.shape[1], d_kin=Xkin_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    ds = TensorDataset(
        torch.tensor(Xem_tr), torch.tensor(Xkin_tr),
        torch.tensor(Xem_tr_raw), torch.tensor(ytr, dtype=torch.long)
    )
    loader = DataLoader(ds, batch_size=batch, shuffle=True, pin_memory=(DEVICE.type=='cuda'))
    sched  = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr*5, epochs=epochs, steps_per_epoch=len(loader), pct_start=0.1)

    Xem_te_t  = torch.tensor(Xem_te).to(DEVICE)
    Xkin_te_t = torch.tensor(Xkin_te).to(DEVICE)

    best_f1, best_ep, pat_cnt, best_state = 0, 0, 0, None
    t0 = time.time()

    for ep in range(1, epochs+1):
        lam_ep = lam * min(1.0, ep / warmup_eps)
        model.train()
        for xb_em, xb_kin, xb_raw, yb in loader:
            xb_em  = xb_em.to(DEVICE); xb_kin = xb_kin.to(DEVICE)
            xb_raw = xb_raw.to(DEVICE); yb    = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb_em, xb_kin)
            loss   = ce_loss(logits, yb)
            if lam_ep > 0:
                loss = loss + lam_ep * physics_loss_fn(logits, xb_raw, thr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        if ep % 25 == 0 or ep == 1:
            print(f"  ep{ep:3d}  F1={f1_ep:.4f}  best={best_f1:.4f}@ep{best_ep}  lam={lam_ep:.2f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Time: {time.time()-t0:.0f}s  Best ep: {best_ep}")

    with torch.no_grad():
        preds = model(Xem_te_t, Xkin_te_t).argmax(1).cpu().numpy()

    m = report(yte, preds, label, len(ytr), len(yte))

    # Save inference package for best result later
    return m, model.state_dict()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    np.random.seed(SEED)

    df = pd.read_csv(REAL_PATH, low_memory=False)
    df = df[df['size_class'].isin(CLASSES)].copy()
    y   = np.array([CLASS_MAP[c] for c in df['size_class']])
    grp = df['ObjID'].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(df, y, groups=grp))
    ytr, yte = y[tr], y[te]
    print(f"Scan split: train={len(ytr):,}  test={len(yte):,}")

    best_f1_overall = 0
    best_pkg = None

    for feat_label, (em_cols, kin_cols) in FEATURE_SETS.items():
        avail_em  = [c for c in em_cols  if c in df.columns]
        avail_kin = [c for c in kin_cols if c in df.columns]

        sc_em  = StandardScaler().fit(df.iloc[tr][avail_em].fillna(0).values)
        sc_kin = StandardScaler().fit(df.iloc[tr][avail_kin].fillna(0).values)

        Xem_tr_raw = df.iloc[tr][avail_em].fillna(0).values.astype(np.float32)
        Xem_te_raw = df.iloc[te][avail_em].fillna(0).values.astype(np.float32)
        Xem_tr  = sc_em.transform(Xem_tr_raw).astype(np.float32)
        Xem_te  = sc_em.transform(Xem_te_raw).astype(np.float32)
        Xkin_tr = sc_kin.transform(df.iloc[tr][avail_kin].fillna(0).values).astype(np.float32)
        Xkin_te = sc_kin.transform(df.iloc[te][avail_kin].fillna(0).values).astype(np.float32)

        print(f"\n{'#'*65}")
        print(f"  Feature set: {feat_label}  EM={len(avail_em)}  KIN={len(avail_kin)}")
        print(f"{'#'*65}")

        for pct_label, (lo_pct, hi_pct) in PHYS_PCTS.items():
            # Compute thresholds from training data (raw, unscaled EM)
            thr = {}
            for cls_name, cls_id in [('large', 0), ('small', 2)]:
                mask = (ytr == cls_id)
                rcs_v = Xem_tr_raw[mask, PHYS_RCS_IDX]
                fp_v  = Xem_tr_raw[mask, PHYS_FP_IDX]
                if cls_name == 'large':
                    thr['large_rcs'] = torch.tensor(float(np.percentile(rcs_v, lo_pct))).to(DEVICE)
                    thr['large_fp']  = torch.tensor(float(np.percentile(fp_v,  lo_pct))).to(DEVICE)
                else:
                    thr['small_rcs'] = torch.tensor(float(np.percentile(rcs_v, hi_pct))).to(DEVICE)
                    thr['small_fp']  = torch.tensor(float(np.percentile(fp_v,  hi_pct))).to(DEVICE)

            print(f"\n  [{pct_label}] thresholds:")
            print(f"    large rcs_p{lo_pct}={float(thr['large_rcs']):.4f}  fp_p{lo_pct}={float(thr['large_fp']):.4f}")
            print(f"    small rcs_p{hi_pct}={float(thr['small_rcs']):.4f}  fp_p{hi_pct}={float(thr['small_fp']):.4f}")

            for lam in LAMBDAS:
                # skip p25p75 for lambda=0 (physics inactive, same as p10p90 lambda=0)
                if lam == 0.0 and pct_label == 'p25p75':
                    continue

                label = f"PINN {feat_label} {pct_label} lam={lam}"
                key   = f"pinn_{feat_label.replace('-','').lower()}_{pct_label}_lam{str(lam).replace('.','')}"
                print(f"\n{'='*60}")
                print(f"  {label}")

                m, state = run(Xem_tr, Xkin_tr, Xem_tr_raw, ytr,
                               Xem_te, Xkin_te, yte,
                               thr, lam, label, key)
                ALL_RESULTS[key] = {
                    'features': feat_label, 'phys_pct': pct_label,
                    'lambda': lam, 'window': 'scan', **m
                }

                # Track best for inference package save
                if m['f1'] > best_f1_overall:
                    best_f1_overall = m['f1']
                    best_pkg = {
                        'model_state': state,
                        'sc_em': sc_em, 'sc_kin': sc_kin,
                        'em_cols': avail_em, 'kin_cols': avail_kin,
                        'classes': CLASSES,
                        'thresholds': {k: float(v.cpu()) for k, v in thr.items()},
                        'd_em': len(avail_em), 'd_kin': len(avail_kin),
                        'lambda': lam, 'phys_pct': pct_label,
                        'features': feat_label, 'window': 'scan',
                        'f1': m['f1'], 'acc': m['acc'],
                    }
                    print(f"  *** New best: F1={best_f1_overall:.4f} [{label}] ***")

    # Save best inference package
    if best_pkg is not None:
        out = MODEL_DIR / 'pinn_deepdive_best_inference.pkl'
        with open(out, 'wb') as f:
            pickle.dump(best_pkg, f)
        print(f"\n  Best inference package saved -> {out.name}")

    # Summary
    print(f"\n{'='*80}")
    print("  PINN DEEP-DIVE -- SUMMARY (scan level)")
    print(f"{'='*80}")
    print(f"  {'Features':<10} {'PhysPct':<8} {'lam':>5} {'F1':>7} {'Acc':>7} {'R-lg':>7} {'R-md':>7} {'R-sm':>7}")
    print(f"  {'-'*65}")
    for feat_label in FEATURE_SETS:
        for pct_label in PHYS_PCTS:
            for lam in LAMBDAS:
                if lam == 0.0 and pct_label == 'p25p75':
                    continue
                key = f"pinn_{feat_label.replace('-','').lower()}_{pct_label}_lam{str(lam).replace('.','')}"
                if key in ALL_RESULTS:
                    r = ALL_RESULTS[key]
                    print(f"  {feat_label:<10} {pct_label:<8} {lam:>5} "
                          f"{r['f1']:>7.4f} {r['acc']:>7.4f} "
                          f"{r['recall_large']:>7.4f} {r['recall_medium']:>7.4f} "
                          f"{r['recall_small']:>7.4f}")

    refs = [('XGB BASE-24 scan', 0.6250), ('GBT FULL-42 scan', 0.7108),
            ('PINN BASE-24 p10p90 lam=0.5 [prev run]', 0.7273)]
    print(f"\n  [Reference baselines — scan level]")
    for name, f1 in refs:
        print(f"  {name:<40}  F1={f1:.4f}")

    with open(MODEL_DIR / 'pinn_deepdive_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Total runtime: {time.time()-t_total:.0f}s")
    print(f"  Results -> models/pinn_deepdive_results.json")


if __name__ == '__main__':
    main()
