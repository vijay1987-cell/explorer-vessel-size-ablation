#!/usr/bin/env python3
"""
HistGradientBoosting — BASE-24 / EDA-18 / FULL-42  x  scan / 5-min / 30-min (9 conditions).

Uses sklearn HistGradientBoostingClassifier (histogram-based GBT, equivalent to LightGBM).
Handles NaN natively; class_weight='balanced' — no oversampling needed.

Reference baselines:
  XGB BASE-24  — scan=0.6250  5-min=0.7043  30-min=0.7135
  Cross-window ensemble      — 5-min=0.7205
  EDA-QJL binary k=64 best  — 30-min=0.7223
"""

import time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

REAL_PATH = "/home/iaxiom/projects/Explorer/data/study_cleaned.csv"
MODEL_DIR  = Path("/home/iaxiom/projects/Explorer/models")
MODEL_DIR.mkdir(exist_ok=True)

CLASSES   = ['large', 'medium', 'small']
CLASS_MAP = {'large': 0, 'medium': 1, 'small': 2}
SEED      = 42

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

FEATURE_SETS = {'BASE-24': BASE_24, 'EDA-18': EDA_18, 'FULL-42': FULL_42}
WINDOWS      = [('scan', None), ('5-min', 300), ('30-min', 1800)]
ALL_RESULTS  = {}


def apply_window(df, window_sec):
    if window_sec is None:
        return df
    df = df.copy()
    df['_tw'] = (df['Rtime_epoch'] // window_sec).astype(int)
    df = (df.sort_values('Rtime_epoch')
            .groupby(['ObjID', '_tw'], sort=False)
            .last().reset_index())
    return df.drop(columns=['_tw'], errors='ignore')


def report(yte, preds, label, n_tr, n_te):
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    cm  = confusion_matrix(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=4)
    recall = {c: float(cm[i,i]/cm[i].sum()) if cm[i].sum() > 0 else 0.
              for i,c in enumerate(CLASSES)}
    print(f"\n  -- GBT [{label}] --")
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


def main():
    t0 = time.time()
    df_raw = pd.read_csv(REAL_PATH, low_memory=False)
    df_raw = df_raw[df_raw['size_class'].isin(CLASSES)].copy()

    for win_tag, win_sec in WINDOWS:
        df  = apply_window(df_raw, win_sec)
        y   = np.array([CLASS_MAP[c] for c in df['size_class']])
        grp = df['ObjID'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr, te = next(gss.split(df, y, groups=grp))
        ytr, yte = y[tr], y[te]

        print(f"\n{'#'*60}")
        print(f"  Window: {win_tag}   train={len(ytr):,}  test={len(yte):,}")
        print(f"{'#'*60}")

        for feat_label, feat_cols in FEATURE_SETS.items():
            avail = [c for c in feat_cols if c in df.columns]
            Xtr = df.iloc[tr][avail].fillna(np.nan).values.astype(np.float32)
            Xte = df.iloc[te][avail].fillna(np.nan).values.astype(np.float32)

            print(f"\n{'='*60}")
            print(f"  GBT -- {feat_label} | {win_tag}  [{len(avail)} features]")
            t1 = time.time()
            clf = HistGradientBoostingClassifier(
                max_iter=500, max_depth=6, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=0.1,
                class_weight='balanced', random_state=SEED,
            ).fit(Xtr, ytr)
            print(f"  Training time: {time.time()-t1:.0f}s")

            preds = clf.predict(Xte)
            m = report(yte, preds, f'{feat_label} | {win_tag}', len(ytr), len(yte))
            key = f"gbt_{win_tag.replace('-','').lower()}_{feat_label.replace('-','').lower()}"
            ALL_RESULTS[key] = {'model': 'GBT', 'features': feat_label, 'window': win_tag, **m}

    # Summary
    print(f"\n{'='*80}")
    print("  HISTGRADIENTBOOSTING GRID -- SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Feature':<10} {'Window':<8} {'F1':>6} {'Acc':>6} {'R-lg':>6} {'R-md':>6} {'R-sm':>6} {'N-test':>7}")
    print(f"  {'-'*65}")
    for win_tag, _ in WINDOWS:
        for feat_label in FEATURE_SETS:
            key = f"gbt_{win_tag.replace('-','').lower()}_{feat_label.replace('-','').lower()}"
            if key in ALL_RESULTS:
                r = ALL_RESULTS[key]
                print(f"  {feat_label:<10} {win_tag:<8} {r['f1']:>6.4f} {r['acc']:>6.4f} "
                      f"{r['recall_large']:>6.4f} {r['recall_medium']:>6.4f} "
                      f"{r['recall_small']:>6.4f} {r['n_test']:>7,}")

    refs = [
        ('BASE-24','scan',   0.6250), ('BASE-24','5-min',  0.7043), ('BASE-24','30-min', 0.7135),
        ('XW-ens', '5-min',  0.7205), ('QJL-k64','30-min', 0.7223),
    ]
    print(f"\n  [Reference baselines]")
    print(f"  {'-'*65}")
    for name, wt, f1 in refs:
        print(f"  {name:<10} {wt:<8} {f1:>6.4f}")

    with open(MODEL_DIR / 'gbt_grid_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Total: {time.time()-t0:.0f}s")
    print(f"  Results -> models/gbt_grid_results.json")


if __name__ == '__main__':
    main()
