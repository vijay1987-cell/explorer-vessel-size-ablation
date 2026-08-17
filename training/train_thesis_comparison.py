#!/usr/bin/env python3
"""
Thesis approach vs. our XGB BASE-24 — direct comparison.

Thesis features implemented:
  - euclid_size_no_range   = sqrt(rgw_s^2 + azw_s^2)          [RobustScaler, train-only fit]
  - ellipse_area_no_range  = (pi/4) * rgw_s * azw_s
  - TotalAmplitude_corr_global = TotalAmplitude * (range/25000)^0.25
  - PeakAmplitude_corr_global  = PeakAmplitude  * (range/25000)^0.25
  - TotalAmplitude_avg_900     = measured_TotalAmplitude_avg_900

Conditions:
  A  Thesis Model-B (4 feat)  — ExtraTrees  — k=0.25
  B  Thesis 5-feat (+ corr peak) — ExtraTrees — k=0.25
  C  ExtraTrees   BASE-24     — same split  (isolates model vs feature effect)
  D  ExtraTrees   BASE-24     — k=0 (no correction, for ablation)
  + all three window variants for the winning condition
"""

import time, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample as sk_resample
import xgboost as xgb

warnings.filterwarnings('ignore')

REAL_PATH  = "/home/iaxiom/projects/Explorer/data/study_cleaned.csv"
MODEL_DIR  = Path("/home/iaxiom/projects/Explorer/models")
CLASSES    = ['large', 'medium', 'small']
CLASS_MAP  = {'large': 0, 'medium': 1, 'small': 2}
REF_RANGE  = 25_000          # metres — thesis global reference
K_CORR     = 0.25            # thesis best exponent

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


# ── windowing ─────────────────────────────────────────────────────────────────
def apply_window(df, window_sec):
    if window_sec is None:
        return df
    df = df.copy()
    df['_tw'] = (df['Rtime_epoch'] // window_sec).astype(int)
    df = (df.sort_values('Rtime_epoch')
            .groupby(['ObjID', '_tw'], sort=False)
            .last().reset_index())
    return df.drop(columns=['_tw'], errors='ignore')


# ── thesis feature builder ────────────────────────────────────────────────────
def build_thesis_features(df_tr, df_te, include_corr_peak=False):
    """
    Compute thesis geometry indices and amplitude corrections.
    RobustScaler fit on training partition only.
    Returns X_tr, X_te, feature_names.
    """
    feats = []

    # 1. No-range geometry — RobustScaler fit on train
    rs = RobustScaler(with_centering=False)
    tr_geom = df_tr[['rgw', 'azw']].fillna(0).values
    te_geom = df_te[['rgw', 'azw']].fillna(0).values
    tr_s = rs.fit_transform(tr_geom)
    te_s = rs.transform(te_geom)

    for split, s in [('tr', tr_s), ('te', te_s)]:
        rgw_s, azw_s = s[:, 0], s[:, 1]
        if split == 'tr':
            X_geo_tr = np.column_stack([
                np.sqrt(rgw_s**2 + azw_s**2),
                (np.pi/4) * rgw_s * azw_s,
            ])
        else:
            X_geo_te = np.column_stack([
                np.sqrt(rgw_s**2 + azw_s**2),
                (np.pi/4) * rgw_s * azw_s,
            ])
    feats += ['euclid_size_no_range', 'ellipse_area_no_range']

    # 2. Amplitude range correction  A × (range / 25000)^0.25
    def corr_factor(df):
        r = df['range'].fillna(df['range'].median()).values.clip(100)
        return (r / REF_RANGE) ** K_CORR

    cf_tr = corr_factor(df_tr).reshape(-1, 1)
    cf_te = corr_factor(df_te).reshape(-1, 1)

    ta_tr = df_tr['TotalAmplitude'].fillna(0).values.reshape(-1,1)
    ta_te = df_te['TotalAmplitude'].fillna(0).values.reshape(-1,1)
    X_amp_tr = ta_tr * cf_tr
    X_amp_te = ta_te * cf_te
    feats.append('TotalAmplitude_corr_global')

    # 3. TotalAmplitude_avg_900
    avg_tr = df_tr['measured_TotalAmplitude_avg_900'].fillna(0).values.reshape(-1,1)
    avg_te = df_te['measured_TotalAmplitude_avg_900'].fillna(0).values.reshape(-1,1)
    feats.append('TotalAmplitude_avg_900')

    X_tr = np.hstack([X_geo_tr, X_amp_tr, avg_tr])
    X_te = np.hstack([X_geo_te, X_amp_te, avg_te])

    if include_corr_peak:
        pa_tr = df_tr['PeakAmplitude'].fillna(0).values.reshape(-1,1)
        pa_te = df_te['PeakAmplitude'].fillna(0).values.reshape(-1,1)
        X_tr = np.hstack([X_tr, pa_tr * cf_tr])
        X_te = np.hstack([X_te, pa_te * cf_te])
        feats.append('PeakAmplitude_corr_global')

    return X_tr.astype(np.float32), X_te.astype(np.float32), feats


# ── metrics ───────────────────────────────────────────────────────────────────
def report(yte, preds, label, n_train, n_test):
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    cm  = confusion_matrix(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=4)
    recall = {c: float(cm[i,i]/cm[i].sum()) if cm[i].sum()>0 else 0.0
              for i,c in enumerate(CLASSES)}
    print(f"\n  ── {label} ──")
    print(f"  Train: {n_train:,}  Test: {n_test:,}")
    print(f"  F1-macro: {f1:.4f}  |  Acc: {acc:.4f}")
    print(f"  Recall → large: {recall['large']:.4f}  "
          f"medium: {recall['medium']:.4f}  small: {recall['small']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"    {'':10} {'large':>7} {'medium':>7} {'small':>7}")
    for i,c in enumerate(CLASSES):
        print(f"    {c:10} {cm[i,0]:>7} {cm[i,1]:>7} {cm[i,2]:>7}")
    print(f"\n{rep}")
    return {'f1':round(f1,4),'acc':round(acc,4),
            'recall_large':round(recall['large'],4),
            'recall_medium':round(recall['medium'],4),
            'recall_small':round(recall['small'],4),
            'n_train':n_train,'n_test':n_test}


def et_clf():
    return ExtraTreesClassifier(
        n_estimators=700, min_samples_leaf=2, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1,
    )


# ── main loop ─────────────────────────────────────────────────────────────────
WINDOWS = [('scan', None), ('5-min', 300), ('30-min', 1800)]

def run_all():
    t0 = time.time()

    df_raw = pd.read_csv(REAL_PATH, low_memory=False)
    df_raw = df_raw[df_raw['size_class'].isin(CLASSES)].copy()

    for win_tag, win_sec in WINDOWS:
        df = apply_window(df_raw, win_sec)
        y  = np.array([CLASS_MAP[c] for c in df['size_class']])
        grp = df['ObjID'].values

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, te = next(gss.split(df, y, groups=grp))
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        ytr, yte = y[tr], y[te]

        print(f"\n{'#'*65}")
        print(f"  Window: {win_tag}   train={len(ytr):,}  test={len(yte):,}")
        print(f"{'#'*65}")

        # ── A: Thesis Model B (4 features) ────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Thesis Model-B (4 feat, k=0.25) | {win_tag}")
        Xtr, Xte, fnames = build_thesis_features(df_tr, df_te, include_corr_peak=False)
        print(f"  Features: {fnames}")
        t1 = time.time()
        clf = et_clf().fit(Xtr, ytr)
        print(f"  Training time: {time.time()-t1:.0f}s")
        m = report(yte, clf.predict(Xte), f'Thesis Model-B | {win_tag}', len(ytr), len(yte))
        key = f"thesis_modelB_{win_tag.replace('-','')}"
        ALL_RESULTS[key] = {'condition':'Thesis Model-B (4f)','window':win_tag,**m}

        # ── B: Thesis 5-feature (+ corrected peak) ────────────────────────
        print(f"\n{'='*60}")
        print(f"  Thesis 5-feat (+ PeakAmpl_corr) | {win_tag}")
        Xtr5, Xte5, fnames5 = build_thesis_features(df_tr, df_te, include_corr_peak=True)
        print(f"  Features: {fnames5}")
        t1 = time.time()
        clf5 = et_clf().fit(Xtr5, ytr)
        print(f"  Training time: {time.time()-t1:.0f}s")
        m5 = report(yte, clf5.predict(Xte5), f'Thesis 5-feat | {win_tag}', len(ytr), len(yte))
        key5 = f"thesis_5feat_{win_tag.replace('-','')}"
        ALL_RESULTS[key5] = {'condition':'Thesis 5-feat','window':win_tag,**m5}

        # ── C: ExtraTrees BASE-24 (our features, ET model) ────────────────
        print(f"\n{'='*60}")
        print(f"  ExtraTrees BASE-24 | {win_tag}")
        avail = [c for c in BASE_24 if c in df.columns]
        Xtr_b = df_tr[avail].fillna(0).values.astype(np.float32)
        Xte_b = df_te[avail].fillna(0).values.astype(np.float32)
        t1 = time.time()
        clf_b = et_clf().fit(Xtr_b, ytr)
        print(f"  Training time: {time.time()-t1:.0f}s | Features: {len(avail)}")
        mb = report(yte, clf_b.predict(Xte_b), f'ET BASE-24 | {win_tag}', len(ytr), len(yte))
        keyb = f"et_base24_{win_tag.replace('-','')}"
        ALL_RESULTS[keyb] = {'condition':'ET BASE-24','window':win_tag,**mb}

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  THESIS vs. ET BASE-24 — SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Condition':<28} {'Window':<8} {'F1':>6} {'Acc':>6} {'R-lg':>6} {'R-md':>6} {'R-sm':>6} {'N-test':>7}")
    print(f"  {'-'*75}")

    for win_tag, _ in WINDOWS:
        for suffix, lbl in [
            (f"thesis_modelB_{win_tag.replace('-','')}", 'Thesis Model-B (4f)'),
            (f"thesis_5feat_{win_tag.replace('-','')}", 'Thesis 5-feat'),
            (f"et_base24_{win_tag.replace('-','')}", 'ET BASE-24'),
        ]:
            if suffix in ALL_RESULTS:
                r = ALL_RESULTS[suffix]
                print(f"  {lbl:<28} {win_tag:<8} {r['f1']:>6.4f} {r['acc']:>6.4f} "
                      f"{r['recall_large']:>6.4f} {r['recall_medium']:>6.4f} "
                      f"{r['recall_small']:>6.4f} {r['n_test']:>7,}")

    # Pull in XGB BASE-24 results from previous runs for reference
    xgb_ref = {
        'scan':  {'f1':0.6250,'acc':0.7578,'recall_large':0.8473,'recall_medium':0.5110,'recall_small':0.4482,'n_test':23921},
        '5-min': {'f1':0.7043,'acc':0.8157,'recall_large':0.8775,'recall_medium':0.7507,'recall_small':0.4618,'n_test':4911},
        '30-min':{'f1':0.7135,'acc':0.8201,'recall_large':0.8825,'recall_medium':0.7483,'recall_small':0.4900,'n_test':945},
    }
    print(f"  {'-'*75}")
    print(f"  [Reference — XGB BASE-24 from earlier runs]")
    for wt, r in xgb_ref.items():
        print(f"  {'XGB BASE-24 (ref)':<28} {wt:<8} {r['f1']:>6.4f} {r['acc']:>6.4f} "
              f"{r['recall_large']:>6.4f} {r['recall_medium']:>6.4f} "
              f"{r['recall_small']:>6.4f} {r['n_test']:>7,}")

    with open(MODEL_DIR / 'thesis_comparison_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Total: {time.time()-t0:.0f}s")
    print(f"  Results → models/thesis_comparison_results.json")


if __name__ == '__main__':
    run_all()
