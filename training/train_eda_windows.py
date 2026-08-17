#!/usr/bin/env python3
"""
EDA Feature × Temporal Window Grid
====================================
Runs BASE-24 / EDA-18 / FULL-42  ×  scan / 5-min / 30-min
for XGB and DL ResNet — 18 conditions total.
"""

import sys, time, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample as sk_resample
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

warnings.filterwarnings('ignore')

REAL_PATH = "/home/iaxiom/projects/Explorer/data/study_cleaned.csv"
MODEL_DIR  = Path("/home/iaxiom/projects/Explorer/models")
MODEL_DIR.mkdir(exist_ok=True)

CLASSES   = ['large', 'medium', 'small']
CLASS_MAP = {'large': 0, 'medium': 1, 'small': 2}
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
XGB_DEV   = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}  |  Data: {REAL_PATH}")

# ── Feature sets ──────────────────────────────────────────────────────────────
BASE_24 = [
    'log_peak_rcs', 'log_total_rcs', 'rcs_conc',
    'aspect_ratio', 'footprint_m2',
    'SampleCount', 'size_bow_stern_component', 'size_beam_component',
    'ellipse_area', 'cr_dr_ratio_c',
    'sog',
    'measured_sog_avg_900',  'measured_sog_avg_1800',
    'measured_sog_avg_3600', 'measured_sog_avg_10800',
    'measured_sog_std_900',  'measured_sog_std_1800',
    'measured_sog_std_3600', 'measured_sog_std_10800',
    'measured_cog_std_900',  'measured_cog_std_1800',
    'measured_cog_std_3600', 'measured_cog_std_10800',
    'displacement',
]
EDA_18 = [
    'measured_TotalAmplitude_avg_900',  'measured_TotalAmplitude_avg_1800',
    'measured_TotalAmplitude_avg_3600', 'measured_TotalAmplitude_avg_10800',
    'measured_rangeStd_900',  'measured_rangeStd_1800',
    'measured_rangeStd_3600', 'measured_rangeStd_10800',
    'measured_azimuthStd_900',  'measured_azimuthStd_1800',
    'measured_azimuthStd_3600', 'measured_azimuthStd_10800',
    'rgw', 'azw',
    'measured_cog_stdlog_900',  'measured_cog_stdlog_1800',
    'measured_cog_stdlog_3600', 'measured_cog_stdlog_10800',
]
FULL_42 = BASE_24 + EDA_18

FEATURE_SETS = {
    'BASE-24': BASE_24,
    'EDA-18':  EDA_18,
    'FULL-42': FULL_42,
}

# ── Windowing ─────────────────────────────────────────────────────────────────
def apply_window(df, window_sec):
    """Stride df to one row per (ObjID, time-window). None = scan-level."""
    if window_sec is None:
        return df
    if 'Rtime_epoch' not in df.columns or 'ObjID' not in df.columns:
        return df
    df = df.copy()
    df['_tw'] = (df['Rtime_epoch'] // window_sec).astype(int)
    df = (df.sort_values('Rtime_epoch')
            .groupby(['ObjID', '_tw'], sort=False)
            .last()
            .reset_index())
    df = df.drop(columns=['_tw'], errors='ignore')
    return df


def load_windowed(feat_cols, window_sec):
    df = pd.read_csv(REAL_PATH, low_memory=False)
    df = df[df['size_class'].isin(CLASSES)].copy()
    df = apply_window(df, window_sec)
    avail = [c for c in feat_cols if c in df.columns]
    df[avail] = df[avail].fillna(0)
    X = df[avail].values.astype(np.float32)
    y = np.array([CLASS_MAP[c] for c in df['size_class']])
    groups = df['ObjID'].values if 'ObjID' in df.columns else np.arange(len(df))
    return X, y, groups, avail


def obj_split(X, y, groups, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(X, y, groups=groups))
    return tr, te


def oversample(Xtr, ytr, seed=42):
    max_n = max(np.bincount(ytr))
    parts = [sk_resample(Xtr[ytr == c], ytr[ytr == c],
                         replace=True, n_samples=max_n, random_state=seed)
             for c in np.unique(ytr)]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


# ── Metrics ───────────────────────────────────────────────────────────────────
def report_metrics(yte, preds, model_tag, feat_label, window_tag, n_train, n_test):
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    cm  = confusion_matrix(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=4)
    recall = {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0
              for i, c in enumerate(CLASSES)}
    print(f"\n  ── {model_tag} [{feat_label}] [{window_tag}] ──")
    print(f"  Train: {n_train:,}  Test: {n_test:,}")
    print(f"  F1-macro: {f1:.4f}  |  Acc: {acc:.4f}")
    print(f"  Recall → large: {recall['large']:.4f}  "
          f"medium: {recall['medium']:.4f}  small: {recall['small']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"    {'':10} {'large':>7} {'medium':>7} {'small':>7}")
    for i, c in enumerate(CLASSES):
        print(f"    {c:10} {cm[i,0]:>7} {cm[i,1]:>7} {cm[i,2]:>7}")
    print(f"\n{rep}")
    return {
        'model': model_tag, 'features': feat_label, 'window': window_tag,
        'f1': round(f1, 4), 'acc': round(acc, 4),
        'recall_large':  round(recall['large'], 4),
        'recall_medium': round(recall['medium'], 4),
        'recall_small':  round(recall['small'], 4),
        'n_train': n_train, 'n_test': n_test,
    }


# ── ResNet ────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, d=256, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, d), nn.BatchNorm1d(d),
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.block(x))


class ResNet(nn.Module):
    def __init__(self, in_dim, d=256, n_blocks=3, n_cls=3, dropout=0.2):
        super().__init__()
        self.stem   = nn.Sequential(nn.Linear(in_dim, d), nn.BatchNorm1d(d), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(d, dropout) for _ in range(n_blocks)])
        self.head   = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, d//2),
                                    nn.GELU(), nn.Linear(d//2, n_cls))
    def forward(self, x): return self.head(self.blocks(self.stem(x)))


# ── Run functions ─────────────────────────────────────────────────────────────
ALL_RESULTS = {}

def run_xgb(feat_label, feat_cols, window_tag, window_sec):
    key = f"xgb_{window_tag.replace('-','').lower()}_{feat_label.replace('-','').lower()}"
    print(f"\n{'='*60}")
    print(f"  XGB — {feat_label} | {window_tag}")
    X, y, groups, avail = load_windowed(feat_cols, window_sec)
    tr, te = obj_split(X, y, groups)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    Xb, yb = oversample(Xtr_s, ytr)
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        eval_metric='mlogloss', random_state=42,
        use_label_encoder=False, tree_method='hist', device=XGB_DEV,
    )
    clf.fit(Xb, yb, eval_set=[(Xte_s, yte)], verbose=False)
    print(f"  Training time: {time.time()-t0:.0f}s | Features: {len(avail)}")
    preds = clf.predict(Xte_s)
    m = report_metrics(yte, preds, 'XGB', feat_label, window_tag, len(ytr), len(yte))
    ALL_RESULTS[key] = m
    pkg = {'model': clf, 'scaler': sc, 'feature_cols': avail, 'classes': CLASSES}
    with open(MODEL_DIR / f'{key}.pkl', 'wb') as f:
        pickle.dump(pkg, f)
    print(f"  Saved → models/{key}.pkl")
    return m


def run_dl(feat_label, feat_cols, window_tag, window_sec,
           epochs=100, batch=1024, lr=3e-4, patience=15):
    key = f"dl_{window_tag.replace('-','').lower()}_{feat_label.replace('-','').lower()}"
    print(f"\n{'='*60}")
    print(f"  DL ResNet — {feat_label} | {window_tag}")
    X, y, groups, avail = load_windowed(feat_cols, window_sec)
    tr, te = obj_split(X, y, groups)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)

    counts = np.bincount(ytr, minlength=3)
    w = torch.tensor((len(ytr) / (3 * counts)).astype(np.float32)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    model = ResNet(in_dim=len(avail)).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds     = TensorDataset(torch.tensor(Xtr_s), torch.tensor(ytr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch, shuffle=True, pin_memory=(DEVICE.type=='cuda'))

    best_f1, best_ep, pat_cnt, best_state = 0, 0, 0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            preds_ep = model(torch.tensor(Xte_s).to(DEVICE)).argmax(1).cpu().numpy()
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
    print(f"  Training time: {time.time()-t0:.0f}s | Features: {len(avail)} | Best ep: {best_ep}")
    with torch.no_grad():
        preds = model(torch.tensor(Xte_s).to(DEVICE)).argmax(1).cpu().numpy()
    m = report_metrics(yte, preds, 'DL ResNet', feat_label, window_tag, len(ytr), len(yte))
    ALL_RESULTS[key] = m
    torch.save(model.state_dict(), MODEL_DIR / f'{key}_weights.pt')
    joblib.dump({'scaler': sc, 'feature_cols': avail, 'classes': CLASSES},
                MODEL_DIR / f'{key}_preprocessor.pkl')
    print(f"  Saved → models/{key}_weights.pt")
    return m


# ── Main ──────────────────────────────────────────────────────────────────────
WINDOWS = [
    ('scan',  None),
    ('5-min', 300),
    ('30-min', 1800),
]

def main():
    t_total = time.time()

    print("\n" + "#"*60)
    print("  MODEL FAMILY: XGB")
    print("#"*60)
    for win_tag, win_sec in WINDOWS:
        for feat_label, feat_cols in FEATURE_SETS.items():
            run_xgb(feat_label, feat_cols, win_tag, win_sec)

    print("\n" + "#"*60)
    print("  MODEL FAMILY: DL ResNet (3-block, 256-dim)")
    print("#"*60)
    for win_tag, win_sec in WINDOWS:
        for feat_label, feat_cols in FEATURE_SETS.items():
            run_dl(feat_label, feat_cols, win_tag, win_sec)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  EDA × WINDOW GRID — SUMMARY")
    print(f"{'='*80}")

    for model_tag in ['XGB', 'DL ResNet']:
        print(f"\n  {model_tag}")
        print(f"  {'Feature':<10}  {'Window':<8}  {'F1':>6}  {'Acc':>6}  {'R-lg':>6}  {'R-md':>6}  {'R-sm':>6}  {'N-test':>7}")
        print(f"  {'-'*70}")
        for win_tag, _ in WINDOWS:
            for feat_label in FEATURE_SETS:
                prefix = 'xgb' if model_tag == 'XGB' else 'dl'
                key = f"{prefix}_{win_tag.replace('-','').lower()}_{feat_label.replace('-','').lower()}"
                if key in ALL_RESULTS:
                    r = ALL_RESULTS[key]
                    print(f"  {feat_label:<10}  {win_tag:<8}  {r['f1']:>6.4f}  {r['acc']:>6.4f}  "
                          f"{r['recall_large']:>6.4f}  {r['recall_medium']:>6.4f}  "
                          f"{r['recall_small']:>6.4f}  {r['n_test']:>7,}")

    with open(MODEL_DIR / 'eda_windows_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Total runtime: {time.time()-t_total:.0f}s")
    print(f"  Results → models/eda_windows_results.json")


if __name__ == '__main__':
    main()
