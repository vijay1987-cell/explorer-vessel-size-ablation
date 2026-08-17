#!/usr/bin/env python3
"""
ET + XGB Ensemble Study
========================
Three ensemble strategies:
  1. Scan-level  — ET(scan) + XGB(scan), probability average, weight search
  2. 5-min-level — ET(5min) + XGB(5min), probability average, weight search
  3. Cross-window — ET trained on scan, probabilities aggregated to 5-min windows,
                    then combined with XGB(5min). Tests whether dense scan-level
                    ET signals add value to the aggregated XGB view.

All use BASE-24 features, same ObjID split (seed=42).
"""

import time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample as sk_resample
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

warnings.filterwarnings('ignore')

REAL_PATH = "/home/iaxiom/projects/Explorer/data/study_cleaned.csv"
MODEL_DIR  = Path("/home/iaxiom/projects/Explorer/models")
CLASSES    = ['large', 'medium', 'small']
CLASS_MAP  = {'large': 0, 'medium': 1, 'small': 2}
XGB_DEV    = 'cuda'

BASE_24 = [
    'log_peak_rcs','log_total_rcs','rcs_conc','aspect_ratio','footprint_m2',
    'SampleCount','size_bow_stern_component','size_beam_component',
    'ellipse_area','cr_dr_ratio_c','sog',
    'measured_sog_avg_900','measured_sog_avg_1800',
    'measured_sog_avg_3600','measured_sog_avg_10800',
    'measured_sog_std_900','measured_sog_std_1800',
    'measured_sog_std_3600','measured_sog_std_10800',
    'measured_cog_std_900','measured_cog_std_1800',
    'measured_cog_std_3600','measured_cog_std_10800',
    'displacement',
]

ALL_RESULTS = {}


# ── helpers ───────────────────────────────────────────────────────────────────
def window_df(df, window_sec):
    if window_sec is None:
        return df
    df = df.copy()
    df['_tw'] = (df['Rtime_epoch'] // window_sec).astype(int)
    df = (df.sort_values('Rtime_epoch')
            .groupby(['ObjID','_tw'], sort=False)
            .last().reset_index())
    return df.drop(columns=['_tw'], errors='ignore')


def obj_split(df, y, seed=42):
    grp = df['ObjID'].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    return next(gss.split(df, y, groups=grp))


def oversample(X, y, seed=42):
    mx = max(np.bincount(y))
    parts = [sk_resample(X[y==c], y[y==c], replace=True,
                         n_samples=mx, random_state=seed)
             for c in np.unique(y)]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def get_feats(df, avail):
    return df[avail].fillna(0).values.astype(np.float32)


def report(yte, preds, label, n_tr, n_te):
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    cm  = confusion_matrix(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=4)
    recall = {c: float(cm[i,i]/cm[i].sum()) if cm[i].sum()>0 else 0.
              for i,c in enumerate(CLASSES)}
    print(f"\n  ── {label} ──")
    print(f"  Train: {n_tr:,}  Test: {n_te:,}")
    print(f"  F1-macro: {f1:.4f}  |  Acc: {acc:.4f}")
    print(f"  Recall → large: {recall['large']:.4f}  medium: {recall['medium']:.4f}  small: {recall['small']:.4f}")
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


def best_weight_search(p_a, p_b, yte, label_a, label_b):
    """Grid-search ensemble weight w for: w*p_a + (1-w)*p_b."""
    best_f1, best_w = 0, 0.5
    for w in np.arange(0.1, 1.0, 0.1):
        p = w * p_a + (1-w) * p_b
        f1 = f1_score(yte, p.argmax(1), average='macro')
        if f1 > best_f1:
            best_f1, best_w = f1, w
    print(f"  Weight search ({label_a} vs {label_b}): best w={best_w:.1f} → F1={best_f1:.4f}")
    return best_w


def train_et(Xtr, ytr):
    return ExtraTreesClassifier(
        n_estimators=700, min_samples_leaf=2, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1,
    ).fit(Xtr, ytr)


def train_xgb(Xtr_s, ytr, Xte_s, yte):
    sc = StandardScaler()
    Xb, yb = oversample(sc.fit_transform(Xtr_s), ytr)
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        eval_metric='mlogloss', random_state=42,
        use_label_encoder=False, tree_method='hist', device=XGB_DEV,
    )
    clf.fit(Xb, yb, eval_set=[(sc.transform(Xte_s), yte)], verbose=False)
    return clf, sc


# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    df_raw = pd.read_csv(REAL_PATH, low_memory=False)
    df_raw = df_raw[df_raw['size_class'].isin(CLASSES)].copy()
    avail  = [c for c in BASE_24 if c in df_raw.columns]

    # ── Strategy 1: Scan-level ensemble ──────────────────────────────────────
    print(f"\n{'#'*65}")
    print("  STRATEGY 1 — Scan-level ET + XGB ensemble")
    print(f"{'#'*65}")

    df_s = df_raw.copy()
    y_s  = np.array([CLASS_MAP[c] for c in df_s['size_class']])
    tr_s, te_s = obj_split(df_s, y_s)
    ytr_s, yte_s = y_s[tr_s], y_s[te_s]
    Xtr_s = get_feats(df_s.iloc[tr_s], avail)
    Xte_s = get_feats(df_s.iloc[te_s], avail)

    print(f"\n  Train: {len(ytr_s):,}  Test: {len(yte_s):,}")

    # ET scan
    print("\n  Training ET (scan)...")
    t1 = time.time()
    et_s = train_et(Xtr_s, ytr_s)
    print(f"  ET trained in {time.time()-t1:.0f}s")
    p_et_s = et_s.predict_proba(Xte_s)

    # XGB scan
    print("  Training XGB (scan)...")
    t1 = time.time()
    xgb_s, sc_s = train_xgb(Xtr_s, ytr_s, Xte_s, yte_s)
    print(f"  XGB trained in {time.time()-t1:.0f}s")
    p_xgb_s = xgb_s.predict_proba(sc_s.transform(Xte_s))

    # Solo results
    m = report(yte_s, p_et_s.argmax(1),  'ET-only  (scan)', len(ytr_s), len(yte_s))
    ALL_RESULTS['et_scan_solo']  = {'strategy':'solo','model':'ET','window':'scan',**m}
    m = report(yte_s, p_xgb_s.argmax(1), 'XGB-only (scan)', len(ytr_s), len(yte_s))
    ALL_RESULTS['xgb_scan_solo'] = {'strategy':'solo','model':'XGB','window':'scan',**m}

    # Ensemble
    w1 = best_weight_search(p_et_s, p_xgb_s, yte_s, 'ET', 'XGB')
    for w, tag in [(0.5, '50/50'), (w1, f'opt({w1:.1f}/{1-w1:.1f})')]:
        p_ens = w * p_et_s + (1-w) * p_xgb_s
        m = report(yte_s, p_ens.argmax(1), f'Ensemble {tag} (scan)', len(ytr_s), len(yte_s))
        ALL_RESULTS[f'ens_scan_{tag.replace("/","_").replace("(","").replace(")","").replace(".","").replace(" ","")}'] = \
            {'strategy':'ensemble','window':'scan','weights':tag,**m}

    # ── Strategy 2: 5-min ensemble ───────────────────────────────────────────
    print(f"\n{'#'*65}")
    print("  STRATEGY 2 — 5-min ET + XGB ensemble")
    print(f"{'#'*65}")

    df_5 = window_df(df_raw, 300)
    y_5  = np.array([CLASS_MAP[c] for c in df_5['size_class']])
    tr_5, te_5 = obj_split(df_5, y_5)
    ytr_5, yte_5 = y_5[tr_5], y_5[te_5]
    Xtr_5 = get_feats(df_5.iloc[tr_5], avail)
    Xte_5 = get_feats(df_5.iloc[te_5], avail)

    print(f"\n  Train: {len(ytr_5):,}  Test: {len(yte_5):,}")

    print("\n  Training ET (5-min)...")
    t1 = time.time()
    et_5 = train_et(Xtr_5, ytr_5)
    print(f"  ET trained in {time.time()-t1:.0f}s")
    p_et_5 = et_5.predict_proba(Xte_5)

    print("  Training XGB (5-min)...")
    t1 = time.time()
    xgb_5, sc_5 = train_xgb(Xtr_5, ytr_5, Xte_5, yte_5)
    print(f"  XGB trained in {time.time()-t1:.0f}s")
    p_xgb_5 = xgb_5.predict_proba(sc_5.transform(Xte_5))

    m = report(yte_5, p_et_5.argmax(1),  'ET-only  (5-min)', len(ytr_5), len(yte_5))
    ALL_RESULTS['et_5min_solo']  = {'strategy':'solo','model':'ET','window':'5-min',**m}
    m = report(yte_5, p_xgb_5.argmax(1), 'XGB-only (5-min)', len(ytr_5), len(yte_5))
    ALL_RESULTS['xgb_5min_solo'] = {'strategy':'solo','model':'XGB','window':'5-min',**m}

    w2 = best_weight_search(p_et_5, p_xgb_5, yte_5, 'ET', 'XGB')
    for w, tag in [(0.5, '50/50'), (w2, f'opt({w2:.1f}/{1-w2:.1f})')]:
        p_ens = w * p_et_5 + (1-w) * p_xgb_5
        m = report(yte_5, p_ens.argmax(1), f'Ensemble {tag} (5-min)', len(ytr_5), len(yte_5))
        ALL_RESULTS[f'ens_5min_{tag.replace("/","_").replace("(","").replace(")","").replace(".","").replace(" ","")}'] = \
            {'strategy':'ensemble','window':'5-min','weights':tag,**m}

    # ── Strategy 3: Cross-window (ET-scan → 5-min aggregated + XGB-5min) ────
    print(f"\n{'#'*65}")
    print("  STRATEGY 3 — Cross-window: ET(scan) aggregated → XGB(5-min)")
    print(f"{'#'*65}")
    print("\n  Using ET trained on scan (Strategy 1) and XGB trained on 5-min (Strategy 2).")
    print("  For each 5-min test window: average ET scan probabilities within window,")
    print("  then combine with XGB 5-min probability.")

    # Get test ObjIDs from 5-min split
    te_objids = set(df_5.iloc[te_5]['ObjID'].unique())

    # Scan-level test rows restricted to those ObjIDs
    scan_te_mask = df_raw['ObjID'].isin(te_objids)
    df_scan_te   = df_raw[scan_te_mask].copy()
    X_scan_te    = get_feats(df_scan_te, avail)

    # ET scan probabilities for all scan-level test rows
    p_et_scan_te = et_s.predict_proba(X_scan_te)   # shape (N_scan_te, 3)
    df_scan_te   = df_scan_te.reset_index(drop=True)
    df_scan_te['_tw'] = (df_scan_te['Rtime_epoch'] // 300).astype(int)

    # For each 5-min test row, aggregate ET scan probabilities
    df_5_te = df_5.iloc[te_5].copy().reset_index(drop=True)
    df_5_te['_tw'] = (df_5_te['Rtime_epoch'] // 300).astype(int)

    p_et_cross = np.zeros((len(df_5_te), 3), dtype=np.float32)
    matched = 0
    for idx, row in df_5_te.iterrows():
        mask = (df_scan_te['ObjID'] == row['ObjID']) & (df_scan_te['_tw'] == row['_tw'])
        if mask.sum() > 0:
            p_et_cross[idx] = p_et_scan_te[mask.values].mean(axis=0)
            matched += 1
        else:
            # fallback: uniform prior
            p_et_cross[idx] = 1/3

    print(f"  Windows matched: {matched}/{len(df_5_te)} ({100*matched/len(df_5_te):.1f}%)")

    w3 = best_weight_search(p_et_cross, p_xgb_5, yte_5, 'ET-cross', 'XGB-5min')
    for w, tag in [(0.5, '50/50'), (w3, f'opt({w3:.1f}/{1-w3:.1f})')]:
        p_ens = w * p_et_cross + (1-w) * p_xgb_5
        m = report(yte_5, p_ens.argmax(1),
                   f'Cross-window Ensemble {tag}', len(ytr_5), len(yte_5))
        ALL_RESULTS[f'ens_cross_{tag.replace("/","_").replace("(","").replace(")","").replace(".","").replace(" ","")}'] = \
            {'strategy':'cross-window','window':'scan→5min','weights':tag,**m}

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  ENSEMBLE STUDY — COMPLETE SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Condition':<38} {'F1':>6} {'Acc':>6} {'R-lg':>6} {'R-md':>6} {'R-sm':>6}")
    print(f"  {'-'*70}")

    order = [
        ('et_scan_solo',       'ET solo (scan)'),
        ('xgb_scan_solo',      'XGB solo (scan)'),
        ('ens_scan_5050',      'Ensemble 50/50 (scan)'),
        (f"ens_scan_opt{str(w1).replace('.','')}{str(round(1-w1,1)).replace('.','')}",
                               f'Ensemble opt ({w1:.1f}/{1-w1:.1f}) (scan)'),
        ('et_5min_solo',       'ET solo (5-min)'),
        ('xgb_5min_solo',      'XGB solo (5-min)'),
        ('ens_5min_5050',      'Ensemble 50/50 (5-min)'),
        (f"ens_5min_opt{str(w2).replace('.','')}{str(round(1-w2,1)).replace('.','')}",
                               f'Ensemble opt ({w2:.1f}/{1-w2:.1f}) (5-min)'),
        ('ens_cross_5050',     'Cross-window 50/50'),
        (f"ens_cross_opt{str(w3).replace('.','')}{str(round(1-w3,1)).replace('.','')}",
                               f'Cross-window opt ({w3:.1f}/{1-w3:.1f})'),
    ]
    for key, lbl in order:
        if key in ALL_RESULTS:
            r = ALL_RESULTS[key]
            print(f"  {lbl:<38} {r['f1']:>6.4f} {r['acc']:>6.4f} "
                  f"{r['recall_large']:>6.4f} {r['recall_medium']:>6.4f} {r['recall_small']:>6.4f}")

    with open(MODEL_DIR / 'ensemble_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Total: {time.time()-t0:.0f}s")
    print(f"  Results → models/ensemble_results.json")


if __name__ == '__main__':
    main()
