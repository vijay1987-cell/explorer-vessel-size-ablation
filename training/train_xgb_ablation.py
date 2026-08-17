#!/usr/bin/env python3
"""
Explorer — XGB Full-24 Ablation (real data only)
=================================================
All conditions use radar_features_labeled.csv (NaN imputed, 123K rows).
No synthetic data. ObjID-stratified 80/20 split (seed=42) held fixed.

Ablation conditions:
  A. Scan-level      — all 24 features, standard params
  B. 5-min strided   — one row per ObjID per 5-min window
  C. EM-only         — 10 EM features only (feature ablation)
  D. KIN-only        — 14 Kinematic features only (feature ablation)
  E. Deeper trees    — max_depth=8, n_estimators=800 (capacity ablation)
  F. Low LR          — lr=0.02, n_estimators=1000 (training ablation)

Output:
  models/xgb_ablation_results.json    — all metrics + per-class recall
  models/xgb_*.pkl                    — trained model packages
"""

import time, pickle, json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, classification_report,
                              accuracy_score, confusion_matrix)
from sklearn.utils import resample as sk_resample
import xgboost as xgb
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

REAL_PATH  = "/home/iaxiom/projects/Explorer/data/study_cleaned.csv"
MODEL_DIR  = Path("/home/iaxiom/projects/Explorer/models")
MODEL_DIR.mkdir(exist_ok=True)

CLASSES   = ['large', 'medium', 'small']
CLASS_MAP = {'large': 0, 'medium': 1, 'small': 2}

EM_10 = [
    'log_peak_rcs', 'log_total_rcs', 'rcs_conc',
    'aspect_ratio', 'footprint_m2',
    'SampleCount', 'size_bow_stern_component', 'size_beam_component',
    'ellipse_area', 'cr_dr_ratio_c',
]
KIN_14 = [
    'sog',
    'measured_sog_avg_900',  'measured_sog_avg_1800',
    'measured_sog_avg_3600', 'measured_sog_avg_10800',
    'measured_sog_std_900',  'measured_sog_std_1800',
    'measured_sog_std_3600', 'measured_sog_std_10800',
    'measured_cog_std_900',  'measured_cog_std_1800',
    'measured_cog_std_3600', 'measured_cog_std_10800',
    'displacement',
]
FULL_24 = EM_10 + KIN_14
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'

ALL_RESULTS = {}


def load_df():
    df = pd.read_csv(REAL_PATH, low_memory=False)
    df = df[df['size_class'].isin(CLASSES)].copy()
    return df


def make_5min(df, feat_cols):
    """Return 5-min strided version (last row per ObjID per 5-min window)."""
    if 'Rtime_epoch' not in df.columns or 'ObjID' not in df.columns:
        return df
    df = df.copy()
    df['t_window'] = (df['Rtime_epoch'] // 300).astype(int)
    df5 = (df.sort_values('Rtime_epoch')
             .groupby(['ObjID', 't_window'], sort=False)
             .last()
             .reset_index())
    return df5


def obj_split(df, seed=42):
    """ObjID-stratified 80/20 split. Returns tr_idx, te_idx into df."""
    groups = df['ObjID'].values if 'ObjID' in df.columns else np.arange(len(df))
    y = np.array([CLASS_MAP[c] for c in df['size_class']])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, y, groups=groups))
    return tr_idx, te_idx


def oversample(Xtr, ytr, seed=42):
    max_n = max(np.bincount(ytr))
    parts = [sk_resample(Xtr[ytr == c], ytr[ytr == c],
                          replace=True, n_samples=max_n, random_state=seed)
             for c in np.unique(ytr)]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def train_xgb(Xtr, ytr, Xte, yte, n_est=600, depth=6, lr=0.04,
              subsample=0.8, colsample=0.7, min_child=3):
    Xb, yb = oversample(Xtr, ytr)
    clf = xgb.XGBClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        subsample=subsample, colsample_bytree=colsample,
        min_child_weight=min_child,
        eval_metric='mlogloss', random_state=42,
        use_label_encoder=False, tree_method='hist', device=DEVICE,
    )
    clf.fit(Xb, yb, eval_set=[(Xte, yte)], verbose=False)
    return clf


def evaluate(clf, Xte, yte, tag, feat_cols):
    preds = clf.predict(Xte)
    f1    = f1_score(yte, preds, average='macro')
    acc   = accuracy_score(yte, preds)
    cm    = confusion_matrix(yte, preds)
    rep   = classification_report(yte, preds, target_names=CLASSES, digits=4)

    # Per-class recall from confusion matrix
    recall = {}
    for i, cls in enumerate(CLASSES):
        row_sum = cm[i].sum()
        recall[cls] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0

    print(f"\n  ─── {tag} ───")
    print(f"  F1-macro: {f1:.4f}  |  Acc: {acc:.4f}")
    print(f"  Recall → large: {recall['large']:.4f}  medium: {recall['medium']:.4f}  small: {recall['small']:.4f}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"    {'':10} {'large':>7} {'medium':>7} {'small':>7}")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls:10} {cm[i,0]:>7} {cm[i,1]:>7} {cm[i,2]:>7}")
    print(f"\n  Per-class detail:")
    print(rep)

    return {'f1': round(f1, 4), 'acc': round(acc, 4),
            'recall_large': round(recall['large'], 4),
            'recall_medium': round(recall['medium'], 4),
            'recall_small': round(recall['small'], 4),
            'n_features': len(feat_cols), 'n_train': len(yte)}


def run_condition(tag, save_name, df, feat_cols, tr_idx, te_idx,
                  n_est=600, depth=6, lr=0.04):
    avail = [c for c in feat_cols if c in df.columns]
    X = df[avail].fillna(0).values.astype(np.float32)
    y = np.array([CLASS_MAP[c] for c in df['size_class']])

    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte  = y[tr_idx], y[te_idx]

    print(f"\n  {'='*55}")
    print(f"  Training: {tag}")
    print(f"  Features: {len(avail)} | Train: {len(ytr):,} | Test: {len(yte):,}")
    print(f"  Class dist train: {dict(zip(CLASSES, np.bincount(ytr, minlength=3)))}")
    print(f"  Class dist test:  {dict(zip(CLASSES, np.bincount(yte, minlength=3)))}")

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    t0 = time.time()
    clf = train_xgb(Xtr_s, ytr, Xte_s, yte,
                    n_est=n_est, depth=depth, lr=lr)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.0f}s")

    metrics = evaluate(clf, Xte_s, yte, tag, avail)
    ALL_RESULTS[save_name] = {'tag': tag, **metrics}

    pkg = {'model': clf, 'scaler': sc, 'feature_cols': avail,
           'classes': CLASSES, 'condition': save_name, 'metrics': metrics}
    with open(MODEL_DIR / f'{save_name}.pkl', 'wb') as f:
        pickle.dump(pkg, f)
    print(f"  Saved → models/{save_name}.pkl")
    return metrics


def main():
    t_total = time.time()
    print("=" * 60)
    print("  Explorer — XGB Full-24 Ablation (Real Data Only)")
    print("=" * 60)
    print(f"  Data: {REAL_PATH}")

    df_scan = load_df()
    print(f"  Scan-level rows: {len(df_scan):,}")
    print(f"  Dist: {df_scan['size_class'].value_counts().to_dict()}")

    # Fix split once on scan-level df (used for A, C, D, E, F)
    tr_scan, te_scan = obj_split(df_scan)

    # ── A: Scan-level, standard ───────────────────────────────────
    run_condition(
        tag='A  XGB scan-level · FULL-24 · standard params',
        save_name='xgb_A_scan_full24',
        df=df_scan, feat_cols=FULL_24, tr_idx=tr_scan, te_idx=te_scan,
    )

    # ── B: 5-min strided ─────────────────────────────────────────
    df_5min = make_5min(df_scan, FULL_24)
    print(f"\n  5-min strided rows: {len(df_5min):,}")
    tr_5min, te_5min = obj_split(df_5min)
    run_condition(
        tag='B  XGB 5-min strided · FULL-24 · standard params',
        save_name='xgb_B_5min_full24',
        df=df_5min, feat_cols=FULL_24, tr_idx=tr_5min, te_idx=te_5min,
    )

    # ── C: EM-only (feature ablation) ────────────────────────────
    run_condition(
        tag='C  XGB scan-level · EM-10 only',
        save_name='xgb_C_scan_em10',
        df=df_scan, feat_cols=EM_10, tr_idx=tr_scan, te_idx=te_scan,
    )

    # ── D: KIN-only (feature ablation) ───────────────────────────
    run_condition(
        tag='D  XGB scan-level · KIN-14 only',
        save_name='xgb_D_scan_kin14',
        df=df_scan, feat_cols=KIN_14, tr_idx=tr_scan, te_idx=te_scan,
    )

    # ── E: Deeper trees (capacity ablation) ──────────────────────
    run_condition(
        tag='E  XGB scan-level · FULL-24 · depth=8, n_est=800',
        save_name='xgb_E_scan_deep',
        df=df_scan, feat_cols=FULL_24, tr_idx=tr_scan, te_idx=te_scan,
        n_est=800, depth=8, lr=0.04,
    )

    # ── F: Lower LR (training ablation) ──────────────────────────
    run_condition(
        tag='F  XGB scan-level · FULL-24 · lr=0.02, n_est=1000',
        save_name='xgb_F_scan_lowlr',
        df=df_scan, feat_cols=FULL_24, tr_idx=tr_scan, te_idx=te_scan,
        n_est=1000, depth=6, lr=0.02,
    )

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print("\n" + "=" * 75)
    print("  XGB ABLATION — FINAL SUMMARY")
    print("=" * 75)
    hdr = f"  {'Condition':<45} {'F1':>6} {'Acc':>6} {'R-lg':>6} {'R-md':>6} {'R-sm':>6}"
    print(hdr)
    print(f"  {'-'*72}")
    for name, r in ALL_RESULTS.items():
        print(f"  {r['tag']:<45} "
              f"{r['f1']:>6.4f} {r['acc']:>6.4f} "
              f"{r['recall_large']:>6.4f} {r['recall_medium']:>6.4f} {r['recall_small']:>6.4f}")

    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"\n  Per-class recall legend: R-lg=large, R-md=medium, R-sm=small")

    out = MODEL_DIR / 'xgb_ablation_results.json'
    with open(out, 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Results saved → {out}")


if __name__ == '__main__':
    main()
