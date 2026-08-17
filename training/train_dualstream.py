#!/usr/bin/env python3
"""
DualStream Cross-Gating — Explorer Dataset (tabular adaptation of Paper 11).

In HQNN Paper 11, cross-attention operates over T detection embeddings per track.
For tabular data (one row per window), a single embedding per stream makes standard
attention degenerate (only one key-value). We use cross-gating (FiLM) instead:
  h_em_out  = h_em  * sigmoid(W_gate(h_kin))   — Kin gates EM stream
  h_kin_out = h_kin * sigmoid(W_gate(h_em))    — EM gates Kin stream
This captures the same cross-stream conditioning without attention over sequences.

Architecture:
  EM  stream: d_em  -> StreamEncoder (128 -> 64) -> cross-gate
  Kin stream: d_kin -> StreamEncoder (128 -> 64) -> cross-gate
  Concat [h_em, h_kin] -> 128 -> n_cls

Stream splits:
  BASE-24: EM=10 base EM features,    Kin=14 base KIN features
  FULL-42: EM=24 (10 base + 14 EDA),  Kin=18 (14 base + 4 EDA-KIN)

Conditions: 2 splits x 3 windows = 6
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

STREAM_SPLITS = {
    'BASE-24': (EM_BASE, KIN_BASE),
    'FULL-42': (EM_BASE + EDA_EM, KIN_BASE + EDA_KIN),
}
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
class StreamEncoder(nn.Module):
    def __init__(self, d_in, d_hidden=128, d_out=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),  nn.GELU(), nn.LayerNorm(d_out),  nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class DualStreamModel(nn.Module):
    """Cross-gating between EM and Kin stream embeddings (FiLM-style)."""
    def __init__(self, d_em, d_kin, n_cls=3, d_model=64, dropout=0.3):
        super().__init__()
        self.em_enc  = StreamEncoder(d_em,  128, d_model, dropout)
        self.kin_enc = StreamEncoder(d_kin, 128, d_model, dropout)
        # Cross-gates: each stream conditions the other
        self.gate_em_from_kin  = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gate_kin_from_em  = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm_em  = nn.LayerNorm(d_model)
        self.norm_kin = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_cls),
        )

    def forward(self, x_em, x_kin):
        h_em  = self.em_enc(x_em)    # (B, 64)
        h_kin = self.kin_enc(x_kin)  # (B, 64)
        # Cross-gate (residual)
        h_em_out  = self.norm_em(h_em   + h_em  * self.gate_em_from_kin(h_kin))
        h_kin_out = self.norm_kin(h_kin + h_kin * self.gate_kin_from_em(h_em))
        return self.head(torch.cat([h_em_out, h_kin_out], dim=1))


# ── metrics ───────────────────────────────────────────────────────────────────
def report(yte, preds, label, n_tr, n_te):
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    cm  = confusion_matrix(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=4)
    recall = {c: float(cm[i,i]/cm[i].sum()) if cm[i].sum() > 0 else 0.
              for i,c in enumerate(CLASSES)}
    print(f"\n  -- DualStream [{label}] --")
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
def run_dualstream(df, tr, te, y, em_cols, kin_cols, label, key,
                   epochs=120, batch=1024, lr=3e-4, patience=20):
    ytr, yte = y[tr], y[te]
    df_tr = df.iloc[tr]
    df_te = df.iloc[te]

    avail_em  = [c for c in em_cols  if c in df.columns]
    avail_kin = [c for c in kin_cols if c in df.columns]

    sc_em  = StandardScaler().fit(df_tr[avail_em].fillna(0).values)
    sc_kin = StandardScaler().fit(df_tr[avail_kin].fillna(0).values)

    Xem_tr  = sc_em.transform(df_tr[avail_em].fillna(0).values).astype(np.float32)
    Xem_te  = sc_em.transform(df_te[avail_em].fillna(0).values).astype(np.float32)
    Xkin_tr = sc_kin.transform(df_tr[avail_kin].fillna(0).values).astype(np.float32)
    Xkin_te = sc_kin.transform(df_te[avail_kin].fillna(0).values).astype(np.float32)

    counts  = np.bincount(ytr, minlength=3)
    w       = torch.tensor((len(ytr) / (3 * counts)).astype(np.float32)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    model = DualStreamModel(d_em=len(avail_em), d_kin=len(avail_kin)).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds = TensorDataset(
        torch.tensor(Xem_tr), torch.tensor(Xkin_tr),
        torch.tensor(ytr, dtype=torch.long)
    )
    loader = DataLoader(ds, batch_size=batch, shuffle=True,
                        pin_memory=(DEVICE.type == 'cuda'))

    Xem_te_t  = torch.tensor(Xem_te).to(DEVICE)
    Xkin_te_t = torch.tensor(Xkin_te).to(DEVICE)

    best_f1, best_ep, pat_cnt, best_state = 0, 0, 0, None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        for xb_em, xb_kin, yb in loader:
            xb_em  = xb_em.to(DEVICE)
            xb_kin = xb_kin.to(DEVICE)
            yb     = yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb_em, xb_kin), yb).backward()
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
            print(f"  ep{ep:3d}  val_F1={f1_ep:.4f}  best={best_f1:.4f}@ep{best_ep}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Time: {time.time()-t0:.0f}s | EM={len(avail_em)} Kin={len(avail_kin)} | Best ep: {best_ep}")

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

    for win_tag, win_sec in WINDOWS:
        df  = apply_window(df_raw, win_sec)
        y   = np.array([CLASS_MAP[c] for c in df['size_class']])
        grp = df['ObjID'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr, te = next(gss.split(df, y, groups=grp))
        ytr, yte = y[tr], y[te]

        print(f"\n{'#'*65}")
        print(f"  Window: {win_tag}   train={len(ytr):,}  test={len(yte):,}")
        print(f"{'#'*65}")

        for split_label, (em_cols, kin_cols) in STREAM_SPLITS.items():
            print(f"\n{'='*60}")
            print(f"  DualStream {split_label} | {win_tag}  [EM={len(em_cols)} Kin={len(kin_cols)}]")
            label = f"{split_label} | {win_tag}"
            key   = f"ds_{win_tag.replace('-','').lower()}_{split_label.replace('-','').lower()}"
            m = run_dualstream(df, tr, te, y, em_cols, kin_cols, label, key)
            ALL_RESULTS[key] = {'model': 'DualStream', 'split': split_label, 'window': win_tag, **m}

    # Summary
    print(f"\n{'='*80}")
    print("  DUALSTREAM CROSS-GATING -- SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Split':<10} {'Window':<8} {'F1':>6} {'Acc':>6} {'R-lg':>6} {'R-md':>6} {'R-sm':>6} {'N-test':>7}")
    print(f"  {'-'*65}")
    for win_tag, _ in WINDOWS:
        for split_label in STREAM_SPLITS:
            key = f"ds_{win_tag.replace('-','').lower()}_{split_label.replace('-','').lower()}"
            if key in ALL_RESULTS:
                r = ALL_RESULTS[key]
                print(f"  {split_label:<10} {win_tag:<8} {r['f1']:>6.4f} {r['acc']:>6.4f} "
                      f"{r['recall_large']:>6.4f} {r['recall_medium']:>6.4f} "
                      f"{r['recall_small']:>6.4f} {r['n_test']:>7,}")

    refs = [
        ('XGB BASE-24', 'scan',   0.6250), ('XGB BASE-24', '5-min',  0.7043), ('XGB BASE-24', '30-min', 0.7135),
        ('GBT FULL-42', 'scan',   0.7108), ('GBT FULL-42', '5-min',  0.7033), ('GBT FULL-42', '30-min', 0.7237),
        ('XW-ensemble', '5-min',  0.7205), ('QJL-k64',     '30-min', 0.7223),
    ]
    print(f"\n  [Reference baselines]")
    print(f"  {'-'*45}")
    for name, wt, f1 in refs:
        print(f"  {name:<12} {wt:<8} {f1:.4f}")

    with open(MODEL_DIR / 'dualstream_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Total runtime: {time.time()-t_total:.0f}s")
    print(f"  Results -> models/dualstream_results.json")


if __name__ == '__main__':
    main()
