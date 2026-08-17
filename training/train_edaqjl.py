#!/usr/bin/env python3
"""
EDA-QJL Ablation — Explorer Dataset
=====================================
Architecture:
  1. sc24   = StandardScaler().fit(X24_tr)    -> X24_sc  (24 features)
  2. sc_eda = StandardScaler().fit(Xeda_tr)   -> Xeda_sc (18 features)
  3. W      = randn(k, 18); W /= ||W||_2      -> unit-norm Gaussian rows
  4. B      = sign(Xeda_sc @ W.T)             -> k binary features
  5. Xfull  = hstack([X24_sc, B])             -> 24+k -> XGB

Grid: k in {64,128,256,512} x proj in {binary,real} x window in {scan,5-min,30-min} = 24 conditions
"""

import time, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample as sk_resample
import xgboost as xgb

warnings.filterwarnings('ignore')

REAL_PATH = "/home/iaxiom/projects/Explorer/data/study_cleaned.csv"
MODEL_DIR  = Path("/home/iaxiom/projects/Explorer/models")
MODEL_DIR.mkdir(exist_ok=True)

CLASSES   = ['large', 'medium', 'small']
CLASS_MAP = {'large': 0, 'medium': 1, 'small': 2}
XGB_DEV   = 'cuda'
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

WINDOWS = [('scan', None), ('5-min', 300), ('30-min', 1800)]
K_SWEEP = [64, 128, 256, 512]
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


def oversample(X, y, seed=SEED):
    max_n = max(np.bincount(y))
    parts = [sk_resample(X[y == c], y[y == c],
                         replace=True, n_samples=max_n, random_state=seed)
             for c in np.unique(y)]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def report(yte, preds, label, n_tr, n_te):
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    cm  = confusion_matrix(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=4)
    recall = {c: float(cm[i,i]/cm[i].sum()) if cm[i].sum() > 0 else 0.
              for i,c in enumerate(CLASSES)}
    print(f"\n  -- {label} --")
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


def xgb_clf():
    return xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        eval_metric='mlogloss', random_state=SEED,
        use_label_encoder=False, tree_method='hist', device=XGB_DEV,
    )


def make_W(k, n_eda=18):
    rng = np.random.RandomState(SEED + k)
    W = rng.randn(k, n_eda).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W


def project(Xeda_sc, W, binary=True):
    proj = (Xeda_sc @ W.T).astype(np.float32)
    return np.sign(proj) if binary else proj


def main():
    t_total = time.time()

    df_raw = pd.read_csv(REAL_PATH, low_memory=False)
    df_raw = df_raw[df_raw['size_class'].isin(CLASSES)].copy()

    avail24  = [c for c in BASE_24 if c in df_raw.columns]
    avail_eda = [c for c in EDA_18  if c in df_raw.columns]
    print(f"BASE-24 available: {len(avail24)}  |  EDA-18 available: {len(avail_eda)}")

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

        X24_tr  = df.iloc[tr][avail24].fillna(0).values.astype(np.float32)
        X24_te  = df.iloc[te][avail24].fillna(0).values.astype(np.float32)
        Xeda_tr = df.iloc[tr][avail_eda].fillna(0).values.astype(np.float32)
        Xeda_te = df.iloc[te][avail_eda].fillna(0).values.astype(np.float32)

        sc24   = StandardScaler().fit(X24_tr)
        sc_eda = StandardScaler().fit(Xeda_tr)
        X24_tr_sc  = sc24.transform(X24_tr).astype(np.float32)
        X24_te_sc  = sc24.transform(X24_te).astype(np.float32)
        Xeda_tr_sc = sc_eda.transform(Xeda_tr).astype(np.float32)
        Xeda_te_sc = sc_eda.transform(Xeda_te).astype(np.float32)

        for binary in [True, False]:
            proj_tag = 'binary' if binary else 'real'
            for k in K_SWEEP:
                label = f"EDA-QJL k={k} {proj_tag} | {win_tag}"
                key   = f"edaqjl_{win_tag.replace('-','').lower()}_{proj_tag}_k{k}"
                print(f"\n{'='*60}")
                print(f"  {label}  [{24+k} features]")

                W    = make_W(k, n_eda=len(avail_eda))
                B_tr = project(Xeda_tr_sc, W, binary=binary)
                B_te = project(Xeda_te_sc, W, binary=binary)

                Xfull_tr = np.hstack([X24_tr_sc, B_tr])
                Xfull_te = np.hstack([X24_te_sc, B_te])
                Xb, yb   = oversample(Xfull_tr, ytr)

                t1  = time.time()
                clf = xgb_clf()
                clf.fit(Xb, yb, eval_set=[(Xfull_te, yte)], verbose=False)
                print(f"  Training time: {time.time()-t1:.0f}s")

                preds = clf.predict(Xfull_te)
                m = report(yte, preds, label, len(ytr), len(yte))
                ALL_RESULTS[key] = {'window': win_tag, 'proj': proj_tag, 'k': k,
                                    'n_features': 24+k, **m}

                if k == 512 and binary:
                    pkg = {
                        'sc24': sc24, 'sc_eda': sc_eda, 'W': W,
                        'k': k, 'proj': 'binary',
                        'base24_feats': avail24, 'eda_feats': avail_eda,
                        'classes': CLASSES, 'f1': m['f1'], 'acc': m['acc'],
                    }
                    fname = MODEL_DIR / f"edaqjl_{win_tag.replace('-','')}_k512_binary.pkl"
                    with open(fname, 'wb') as f:
                        pickle.dump(pkg, f)
                    print(f"  Preprocessor saved -> models/{fname.name}")

    # Summary
    print(f"\n{'='*80}")
    print("  EDA-QJL ABLATION -- SUMMARY")
    print(f"{'='*80}")
    for win_tag, _ in WINDOWS:
        print(f"\n  Window: {win_tag}")
        print(f"  {'proj':<8} {'k':>5} {'n_feat':>7} {'F1':>7} {'Acc':>7} {'R-lg':>7} {'R-md':>7} {'R-sm':>7}")
        print(f"  {'-'*62}")
        for proj_tag in ['binary', 'real']:
            for k in K_SWEEP:
                key = f"edaqjl_{win_tag.replace('-','').lower()}_{proj_tag}_k{k}"
                if key in ALL_RESULTS:
                    r = ALL_RESULTS[key]
                    print(f"  {proj_tag:<8} {k:>5} {r['n_features']:>7} "
                          f"{r['f1']:>7.4f} {r['acc']:>7.4f} "
                          f"{r['recall_large']:>7.4f} {r['recall_medium']:>7.4f} "
                          f"{r['recall_small']:>7.4f}")

    refs = {
        'scan':   {'XGB BASE-24': 0.6250, 'XGB FULL-42': 0.6430},
        '5-min':  {'XGB BASE-24': 0.7043, 'XGB FULL-42': 0.6463, 'Cross-window ens': 0.7205},
        '30-min': {'XGB BASE-24': 0.7135, 'XGB FULL-42': 0.7012},
    }
    print(f"\n  [Reference baselines]")
    for wt, rdict in refs.items():
        for name, f1 in rdict.items():
            print(f"  {wt:<7}  {name:<22}  F1={f1:.4f}")

    with open(MODEL_DIR / 'edaqjl_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Total runtime: {time.time()-t_total:.0f}s")
    print(f"  Results -> models/edaqjl_results.json")


if __name__ == '__main__':
    main()
