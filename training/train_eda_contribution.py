#!/usr/bin/env python3
"""
Explorer — EDA Feature Contribution Study
==========================================
Question: do the 18 EDA-novel features add discriminating power
beyond the base Full-24 features, across XGB, DL, and QML?

Feature sets tested:
  BASE-24  : 10 EM + 14 Kinematic (papers 1-9 standard)
  EDA-18   : 18 EDA-novel features (papers 10-11 discovery)
  FULL-42  : BASE-24 + EDA-18

Model families:
  XGB      : XGBoost classifier (scan-level, same split as earlier)
  DL       : 3-block ResNet MLP (256-dim)
  HQNN     : 8-qubit HQNN — BASE-8 vs EDA-8 (top 8 from each group)

All conditions use ObjID-stratified 80/20 split (seed=42), same split held fixed.
Per-class recall saved for cross-model comparison at the end.
"""

import sys, time, json, pickle, warnings
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

warnings.filterwarnings('ignore')

REAL_PATH = "/home/iaxiom/projects/Explorer/data/study_cleaned.csv"
MODEL_DIR = Path("/home/iaxiom/projects/Explorer/models")
MODEL_DIR.mkdir(exist_ok=True)

CLASSES   = ['large', 'medium', 'small']
CLASS_MAP = {'large': 0, 'medium': 1, 'small': 2}
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
XGB_DEV   = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Torch device: {DEVICE}")

# ── Feature sets ──────────────────────────────────────────────────────────────
BASE_24 = [
    # EM-10
    'log_peak_rcs', 'log_total_rcs', 'rcs_conc',
    'aspect_ratio', 'footprint_m2',
    'SampleCount', 'size_bow_stern_component', 'size_beam_component',
    'ellipse_area', 'cr_dr_ratio_c',
    # KIN-14
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

# Top-8 from each group for QML (8 qubits)
BASE_8 = [
    'log_peak_rcs', 'log_total_rcs', 'SampleCount', 'footprint_m2',
    'aspect_ratio', 'size_beam_component', 'size_bow_stern_component', 'ellipse_area',
]
EDA_8 = [
    'measured_TotalAmplitude_avg_3600', 'measured_TotalAmplitude_avg_10800',
    'measured_rangeStd_3600',           'measured_rangeStd_10800',
    'measured_azimuthStd_3600',         'measured_azimuthStd_10800',
    'rgw', 'azw',
]

ALL_RESULTS = {}   # key → metrics dict


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(feat_cols: list):
    df = pd.read_csv(REAL_PATH, low_memory=False)
    df = df[df['size_class'].isin(CLASSES)].copy()
    avail = [c for c in feat_cols if c in df.columns]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing features: {missing}")
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
def metrics(yte, preds, tag, feat_label):
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    cm  = confusion_matrix(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=4)
    recall = {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0
              for i, c in enumerate(CLASSES)}
    print(f"\n  ── {tag} [{feat_label}] ──")
    print(f"  F1-macro: {f1:.4f}  |  Acc: {acc:.4f}")
    print(f"  Recall → large: {recall['large']:.4f}  "
          f"medium: {recall['medium']:.4f}  small: {recall['small']:.4f}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"    {'':10} {'large':>7} {'medium':>7} {'small':>7}")
    for i, c in enumerate(CLASSES):
        print(f"    {c:10} {cm[i,0]:>7} {cm[i,1]:>7} {cm[i,2]:>7}")
    print(f"\n  Per-class detail:\n{rep}")
    return {'f1': round(f1,4), 'acc': round(acc,4),
            'recall_large':  round(recall['large'],4),
            'recall_medium': round(recall['medium'],4),
            'recall_small':  round(recall['small'],4),
            'n_features': len(feat_label.split('+'))}


# ════════════════════════════════════════════════════════════════════════════
#  XGB
# ════════════════════════════════════════════════════════════════════════════
def run_xgb(feat_cols, label, save_name):
    print(f"\n{'='*60}")
    print(f"  XGB — {label}")
    X, y, groups, avail = load_data(feat_cols)
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
    m = metrics(yte, preds, f'XGB', label)
    ALL_RESULTS[save_name] = {'model': 'XGB', 'features': label, **m}
    pkg = {'model': clf, 'scaler': sc, 'feature_cols': avail, 'classes': CLASSES}
    with open(MODEL_DIR / f'{save_name}.pkl', 'wb') as f:
        pickle.dump(pkg, f)
    print(f"  Saved → models/{save_name}.pkl")
    return m


# ════════════════════════════════════════════════════════════════════════════
#  DL — 3-block ResNet MLP
# ════════════════════════════════════════════════════════════════════════════
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


def run_dl(feat_cols, label, save_name,
           epochs=100, batch=1024, lr=3e-4, patience=15):
    print(f"\n{'='*60}")
    print(f"  DL ResNet — {label}")
    X, y, groups, avail = load_data(feat_cols)
    tr, te = obj_split(X, y, groups)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)

    # Class-weighted loss
    counts = np.bincount(ytr, minlength=3)
    w = torch.tensor((len(ytr) / (3 * counts)).astype(np.float32)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    model = ResNet(in_dim=len(avail)).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds = TensorDataset(torch.tensor(Xtr_s), torch.tensor(ytr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch, shuffle=True, pin_memory=True)

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
    m = metrics(yte, preds, 'DL ResNet', label)
    ALL_RESULTS[save_name] = {'model': 'DL_ResNet', 'features': label, **m}
    torch.save(model.state_dict(), MODEL_DIR / f'{save_name}_weights.pt')
    pre = {'scaler': sc, 'feature_cols': avail, 'classes': CLASSES,
           'n_features': len(avail)}
    joblib.dump(pre, MODEL_DIR / f'{save_name}_preprocessor.pkl')
    print(f"  Saved → models/{save_name}_weights.pt")
    return m


# ════════════════════════════════════════════════════════════════════════════
#  HQNN 8q — BASE-8 vs EDA-8
# ════════════════════════════════════════════════════════════════════════════
def run_hqnn(feat_cols, label, save_name, n_qubits=8, n_layers=3,
             epochs=60, batch=256, lr=5e-3, patience=12):
    print(f"\n{'='*60}")
    print(f"  HQNN {n_qubits}q {n_layers}L — {label}")
    try:
        import pennylane as qml
        try:
            dev = qml.device('lightning.gpu', wires=n_qubits)
            print("  Using lightning.gpu")
        except Exception:
            dev = qml.device('default.qubit', wires=n_qubits)
            print("  Using default.qubit")

        @qml.qnode(dev, interface='torch')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs[:n_qubits], wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        class HQNNModel(nn.Module):
            def __init__(self, n_feat):
                super().__init__()
                self.pre    = nn.Linear(n_feat, n_qubits) if n_feat != n_qubits else nn.Identity()
                self.qlayer = qml.qnn.TorchLayer(circuit, {'weights': (n_layers, n_qubits, 3)})
                self.head   = nn.Linear(n_qubits, 3)
            def forward(self, x):
                return self.head(self.qlayer(torch.tanh(self.pre(x)) * 3.14159))

        X, y, groups, avail = load_data(feat_cols)
        # Use 5-min strided for QML (reduces sample count, speeds training)
        df = pd.read_csv(REAL_PATH, low_memory=False)
        df = df[df['size_class'].isin(CLASSES)].copy()
        avail_df = [c for c in feat_cols if c in df.columns]
        if 'Rtime_epoch' in df.columns and 'ObjID' in df.columns:
            df['t_window'] = (df['Rtime_epoch'] // 300).astype(int)
            df = (df.sort_values('Rtime_epoch')
                    .groupby(['ObjID', 't_window'], sort=False)
                    .last().reset_index())
        X5 = df[avail_df].fillna(0).values.astype(np.float32)
        y5 = np.array([CLASS_MAP[c] for c in df['size_class']])
        g5 = df['ObjID'].values if 'ObjID' in df.columns else np.arange(len(df))
        tr, te = obj_split(X5, y5, g5)
        Xtr, Xte, ytr, yte = X5[tr], X5[te], y5[tr], y5[te]
        print(f"  5-min strided: train={len(ytr):,}  test={len(yte):,}")

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
        Xte_s = sc.transform(Xte).astype(np.float32)

        counts = np.bincount(ytr, minlength=3)
        w = torch.tensor((len(ytr) / (3 * counts)).astype(np.float32)).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        model = HQNNModel(len(avail_df)).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

        ds = TensorDataset(torch.tensor(Xtr_s), torch.tensor(ytr, dtype=torch.long))
        loader = DataLoader(ds, batch_size=batch, shuffle=True)

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
            sched.step(1 - best_f1)
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
            if ep % 15 == 0 or ep == 1:
                print(f"  ep{ep:3d}  val_F1={f1_ep:.4f}  best={best_f1:.4f}@ep{best_ep}")

        model.load_state_dict(best_state)
        model.eval()
        print(f"  Training time: {time.time()-t0:.0f}s | Best ep: {best_ep}")
        with torch.no_grad():
            preds = model(torch.tensor(Xte_s).to(DEVICE)).argmax(1).cpu().numpy()
        m = metrics(yte, preds, f'HQNN {n_qubits}q', label)
        ALL_RESULTS[save_name] = {'model': f'HQNN_{n_qubits}q', 'features': label, **m}
        torch.save(model.state_dict(), MODEL_DIR / f'{save_name}_weights.pt')
        pre = {'scaler': sc, 'feature_cols': avail_df, 'n_qubits': n_qubits,
               'n_layers': n_layers, 'classes': CLASSES}
        joblib.dump(pre, MODEL_DIR / f'{save_name}_preprocessor.pkl')
        print(f"  Saved → models/{save_name}_weights.pt")
        return m

    except ImportError:
        print("  [SKIP] PennyLane not available")
        ALL_RESULTS[save_name] = {'model': f'HQNN_{n_qubits}q', 'features': label,
                                   'f1': 0, 'acc': 0, 'recall_large': 0,
                                   'recall_medium': 0, 'recall_small': 0, 'skipped': True}
        return None


# ════════════════════════════════════════════════════════════════════════════
#  EDA discriminability analysis
# ════════════════════════════════════════════════════════════════════════════
def eda_discriminability():
    print(f"\n{'='*60}")
    print("  EDA — Feature Discriminability Analysis")
    print("  (Cohen's d and Fisher Discriminant Ratio per feature)")
    print(f"{'='*60}")
    df = pd.read_csv(REAL_PATH, low_memory=False)
    df = df[df['size_class'].isin(CLASSES)].copy()
    all_feats = [c for c in FULL_42 if c in df.columns]
    df[all_feats] = df[all_feats].fillna(0)

    rows = []
    for feat in all_feats:
        groups_f = [df[df['size_class'] == c][feat].values for c in CLASSES]
        # Fisher Discriminant Ratio (one-vs-all macro)
        grand_mean = df[feat].mean()
        grand_std  = df[feat].std() + 1e-9
        between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups_f)
        within  = sum(((g - g.mean())**2).sum() for g in groups_f)
        fdr = between / (within + 1e-9)
        # Mean Cohen's d across pairs
        pairs = [(0,1),(0,2),(1,2)]
        cd_vals = []
        for i, j in pairs:
            g1, g2 = groups_f[i], groups_f[j]
            pooled_std = np.sqrt((g1.var() + g2.var()) / 2 + 1e-9)
            cd_vals.append(abs(g1.mean() - g2.mean()) / pooled_std)
        cd_mean = np.mean(cd_vals)
        group = 'EDA-18' if feat in EDA_18 else 'BASE-24'
        rows.append({'feature': feat, 'group': group,
                     'fdr': round(fdr, 4), 'cohens_d_mean': round(cd_mean, 4)})

    results_df = pd.DataFrame(rows).sort_values('cohens_d_mean', ascending=False)
    print(f"\n  {'Feature':<45} {'Group':<10} {'FDR':>8} {'Cohen-d':>8}")
    print(f"  {'-'*75}")
    for _, row in results_df.iterrows():
        marker = " ★" if row['group'] == 'EDA-18' else "  "
        print(f"  {row['feature']:<45} {row['group']:<10} {row['fdr']:>8.4f} {row['cohens_d_mean']:>8.4f}{marker}")

    base_mean  = results_df[results_df['group'] == 'BASE-24']['cohens_d_mean'].mean()
    eda_mean   = results_df[results_df['group'] == 'EDA-18']['cohens_d_mean'].mean()
    print(f"\n  Mean Cohen's d — BASE-24: {base_mean:.4f}  |  EDA-18: {eda_mean:.4f}")
    print(f"  EDA-18 discriminability vs BASE-24: {'+' if eda_mean > base_mean else ''}{eda_mean - base_mean:.4f}")

    results_df.to_csv(MODEL_DIR / 'eda_discriminability.csv', index=False)
    print(f"\n  Saved → models/eda_discriminability.csv")
    return results_df


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    t_total = time.time()
    print("=" * 60)
    print("  Explorer — EDA Feature Contribution Study")
    print("=" * 60)
    print(f"  Device: {DEVICE}  |  Data: {REAL_PATH}")

    # ── 0. Discriminability analysis ──────────────────────────────
    eda_discriminability()

    # ── XGB ───────────────────────────────────────────────────────
    print(f"\n\n{'#'*60}")
    print("  MODEL FAMILY: XGB")
    print(f"{'#'*60}")
    run_xgb(BASE_24, 'BASE-24',  'eda_xgb_base24')
    run_xgb(EDA_18,  'EDA-18',   'eda_xgb_eda18')
    run_xgb(FULL_42, 'FULL-42',  'eda_xgb_full42')

    # ── DL ────────────────────────────────────────────────────────
    print(f"\n\n{'#'*60}")
    print("  MODEL FAMILY: DL ResNet (3-block, 256-dim)")
    print(f"{'#'*60}")
    run_dl(BASE_24, 'BASE-24', 'eda_dl_base24')
    run_dl(EDA_18,  'EDA-18',  'eda_dl_eda18')
    run_dl(FULL_42, 'FULL-42', 'eda_dl_full42')

    # ── HQNN ─────────────────────────────────────────────────────
    print(f"\n\n{'#'*60}")
    print("  MODEL FAMILY: HQNN 8q (5-min strided)")
    print(f"{'#'*60}")
    run_hqnn(BASE_8, 'BASE-8',  'eda_hqnn_base8')
    run_hqnn(EDA_8,  'EDA-8',   'eda_hqnn_eda8')

    # ── Final cross-model summary ─────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n\n{'='*75}")
    print("  EDA CONTRIBUTION — CROSS-MODEL SUMMARY")
    print(f"{'='*75}")
    hdr = f"  {'Condition':<35} {'Model':<12} {'F1':>6} {'Acc':>6} {'R-lg':>6} {'R-md':>6} {'R-sm':>6}"
    print(hdr)
    print(f"  {'-'*73}")

    order = [
        'eda_xgb_base24', 'eda_xgb_eda18', 'eda_xgb_full42',
        'eda_dl_base24',  'eda_dl_eda18',  'eda_dl_full42',
        'eda_hqnn_base8', 'eda_hqnn_eda8',
    ]
    labels = {
        'eda_xgb_base24':  'XGB  BASE-24',
        'eda_xgb_eda18':   'XGB  EDA-18 only',
        'eda_xgb_full42':  'XGB  FULL-42 (24+18)',
        'eda_dl_base24':   'DL   BASE-24',
        'eda_dl_eda18':    'DL   EDA-18 only',
        'eda_dl_full42':   'DL   FULL-42 (24+18)',
        'eda_hqnn_base8':  'HQNN BASE-8',
        'eda_hqnn_eda8':   'HQNN EDA-8',
    }
    prev_model = None
    for k in order:
        if k not in ALL_RESULTS:
            continue
        r = ALL_RESULTS[k]
        if r.get('skipped'):
            print(f"  {labels[k]:<35} {r['model']:<12} {'SKIPPED':>6}")
            continue
        model_name = r['model']
        if model_name != prev_model:
            print(f"  {'-'*73}")
            prev_model = model_name
        delta_f1 = ""
        if k.endswith('_full42'):
            base_key = k.replace('_full42', '_base24')
            if base_key in ALL_RESULTS:
                d = r['f1'] - ALL_RESULTS[base_key]['f1']
                delta_f1 = f"  Δ{d:+.4f} vs BASE"
        print(f"  {labels[k]:<35} {r['model']:<12} "
              f"{r['f1']:>6.4f} {r['acc']:>6.4f} "
              f"{r['recall_large']:>6.4f} {r['recall_medium']:>6.4f} {r['recall_small']:>6.4f}"
              f"{delta_f1}")

    print(f"\n  Total time: {elapsed/60:.1f} min")
    print(f"  Per-class recall: R-lg=large  R-md=medium  R-sm=small")

    with open(MODEL_DIR / 'eda_contribution_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Full results → models/eda_contribution_results.json")


if __name__ == '__main__':
    main()
