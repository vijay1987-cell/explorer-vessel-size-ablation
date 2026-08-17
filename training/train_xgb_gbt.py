#!/usr/bin/env python3
"""
Explorer — XGB-Full24 & GBT-Full24 ablation
============================================
Ablation conditions:
  A. XGB scan-level  (real labeled, imputed)
  B. XGB 5-min strided (real labeled, imputed)
  C. XGB Synthetic 1M
  D. XGB Combined (real + synthetic)
  E. GBT scan-level  (real labeled, imputed)
  F. GBT Synthetic 1M
"""

import time, pickle, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, accuracy_score
import xgboost as xgb
import torch

warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import (REAL_PATH, SYN_PATH, MODEL_DIR, CLASSES, CLASS_MAP,
                    FULL_24, load_real, load_synthetic, obj_split, oversample)

RESULTS = {}


def train_xgb(Xtr, ytr, Xte, yte, tag, scale=True):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    Xb, yb = oversample(Xtr_s, ytr)
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        eval_metric='mlogloss', random_state=42,
        use_label_encoder=False, tree_method='hist',
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    clf.fit(Xb, yb, eval_set=[(Xte_s, yte)], verbose=False)
    preds = clf.predict(Xte_s)
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=3)
    print(f"\n  [{tag}]  F1-macro: {f1:.4f}  Acc: {acc:.4f}")
    print(rep)
    return clf, sc, f1, acc, rep


def train_gbt(Xtr, ytr, Xte, yte, tag):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    Xb, yb = oversample(Xtr_s, ytr)
    clf = GradientBoostingClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf.fit(Xb, yb)
    preds = clf.predict(Xte_s)
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=3)
    print(f"\n  [{tag}]  F1-macro: {f1:.4f}  Acc: {acc:.4f}")
    print(rep)
    return clf, sc, f1, acc, rep


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Explorer — XGB-Full24 & GBT-Full24 Ablation")
    print("=" * 60)

    # ── Load real data ────────────────────────────────────────────
    print("\nLoading real labeled (imputed)...")
    X_real, y_real, groups, df_real = load_real(FULL_24)
    Xr_tr, Xr_te, yr_tr, yr_te = obj_split(X_real, y_real, groups)
    print(f"  Real: train={len(yr_tr):,}  test={len(yr_te):,}")

    # 5-min strided version
    df_raw = pd.read_csv(REAL_PATH, low_memory=False)
    avail24 = [c for c in FULL_24 if c in df_raw.columns]
    df_raw[avail24] = df_raw[avail24].fillna(df_raw[avail24].median())
    df_raw = df_raw[df_raw['size_class'].isin(CLASSES)]
    if 'Rtime_epoch' in df_raw.columns and 'ObjID' in df_raw.columns:
        df_raw['t_window'] = (df_raw['Rtime_epoch'] // 300).astype(int)
        df_5min = (df_raw.sort_values('Rtime_epoch')
                         .groupby(['ObjID', 't_window'], sort=False)
                         .last()
                         .reset_index())
    else:
        df_5min = df_raw.copy()
    X_5min = df_5min[avail24].values.astype(np.float32)
    y_5min = np.array([CLASS_MAP[c] for c in df_5min['size_class']])
    groups_5min = df_5min['ObjID'].values if 'ObjID' in df_5min.columns else np.arange(len(df_5min))
    X5_tr, X5_te, y5_tr, y5_te = obj_split(X_5min, y_5min, groups_5min)
    print(f"  5-min: train={len(y5_tr):,}  test={len(y5_te):,}")

    # ── Load synthetic ────────────────────────────────────────────
    print("\nLoading synthetic 1M...")
    X_syn, y_syn = load_synthetic(FULL_24)
    # Synthetic has no ObjID — simple 80/20 split
    from sklearn.model_selection import train_test_split
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn)
    print(f"  Synthetic: train={len(ys_tr):,}  test={len(ys_te):,}")

    # Combined (real scan + synthetic)
    Xc_tr = np.vstack([Xr_tr, Xs_tr])
    yc_tr = np.concatenate([yr_tr, ys_tr])

    # ── Condition A: XGB scan-level real ─────────────────────────
    clf, sc, f1, acc, rep = train_xgb(Xr_tr, yr_tr, Xr_te, yr_te,
                                       f'A  XGB scan-level real ({len(Xr_tr):,} train)')
    RESULTS['A_xgb_real_scan'] = {'f1': f1, 'acc': acc}
    pkg = {'model': clf, 'scaler': sc, 'feature_cols': avail24,
           'classes': CLASSES, 'condition': 'A_xgb_real_scan'}
    with open(MODEL_DIR / 'xgb_full24_real_scan.pkl', 'wb') as f_:
        pickle.dump(pkg, f_)

    # ── Condition B: XGB 5-min strided ───────────────────────────
    clf, sc, f1, acc, rep = train_xgb(X5_tr, y5_tr, X5_te, y5_te,
                                       f'B  XGB 5-min real ({len(X5_tr):,} train)')
    RESULTS['B_xgb_real_5min'] = {'f1': f1, 'acc': acc}
    pkg = {'model': clf, 'scaler': sc, 'feature_cols': avail24,
           'classes': CLASSES, 'condition': 'B_xgb_real_5min'}
    with open(MODEL_DIR / 'xgb_full24_real_5min.pkl', 'wb') as f_:
        pickle.dump(pkg, f_)

    # ── Condition C: XGB synthetic 1M ────────────────────────────
    clf, sc, f1, acc, rep = train_xgb(Xs_tr, ys_tr, Xs_te, ys_te,
                                       f'C  XGB synthetic 1M ({len(Xs_tr):,} train)')
    RESULTS['C_xgb_syn1M'] = {'f1': f1, 'acc': acc}
    pkg = {'model': clf, 'scaler': sc, 'feature_cols': avail24,
           'classes': CLASSES, 'condition': 'C_xgb_syn1M'}
    with open(MODEL_DIR / 'xgb_full24_synthetic1M.pkl', 'wb') as f_:
        pickle.dump(pkg, f_)

    # ── Condition D: XGB combined ─────────────────────────────────
    clf, sc, f1, acc, rep = train_xgb(Xc_tr, yc_tr, Xr_te, yr_te,
                                       f'D  XGB combined ({len(Xc_tr):,} train → real test)')
    RESULTS['D_xgb_combined'] = {'f1': f1, 'acc': acc}
    pkg = {'model': clf, 'scaler': sc, 'feature_cols': avail24,
           'classes': CLASSES, 'condition': 'D_xgb_combined'}
    with open(MODEL_DIR / 'xgb_full24_combined.pkl', 'wb') as f_:
        pickle.dump(pkg, f_)

    # ── Condition E: GBT scan-level real ─────────────────────────
    clf, sc, f1, acc, rep = train_gbt(Xr_tr, yr_tr, Xr_te, yr_te,
                                       f'E  GBT scan-level real ({len(Xr_tr):,} train)')
    RESULTS['E_gbt_real_scan'] = {'f1': f1, 'acc': acc}
    pkg = {'model': clf, 'scaler': sc, 'feature_cols': avail24,
           'classes': CLASSES, 'condition': 'E_gbt_real_scan'}
    with open(MODEL_DIR / 'gbt_full24_real_scan.pkl', 'wb') as f_:
        pickle.dump(pkg, f_)

    # ── Condition F: GBT synthetic 1M ─────────────────────────────
    clf, sc, f1, acc, rep = train_gbt(Xs_tr, ys_tr, Xs_te, ys_te,
                                       f'F  GBT synthetic 1M ({len(Xs_tr):,} train)')
    RESULTS['F_gbt_syn1M'] = {'f1': f1, 'acc': acc}
    pkg = {'model': clf, 'scaler': sc, 'feature_cols': avail24,
           'classes': CLASSES, 'condition': 'F_gbt_syn1M'}
    with open(MODEL_DIR / 'gbt_full24_synthetic1M.pkl', 'wb') as f_:
        pickle.dump(pkg, f_)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  XGB/GBT ABLATION SUMMARY")
    print("=" * 60)
    labels = {
        'A_xgb_real_scan': 'A  XGB scan-level real',
        'B_xgb_real_5min': 'B  XGB 5-min strided real',
        'C_xgb_syn1M':     'C  XGB synthetic 1M',
        'D_xgb_combined':  'D  XGB combined (real+syn → real test)',
        'E_gbt_real_scan': 'E  GBT scan-level real',
        'F_gbt_syn1M':     'F  GBT synthetic 1M',
    }
    print(f"  {'Condition':<42} {'F1-mac':>7}  {'Acc':>7}")
    print(f"  {'-'*57}")
    for k, lbl in labels.items():
        r = RESULTS[k]
        print(f"  {lbl:<42} {r['f1']:>7.4f}  {r['acc']:>7.4f}")

    with open(MODEL_DIR / 'xgb_gbt_ablation.json', 'w') as f_:
        json.dump(RESULTS, f_, indent=2)
    print(f"\n  Saved ablation results → {MODEL_DIR}/xgb_gbt_ablation.json")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
