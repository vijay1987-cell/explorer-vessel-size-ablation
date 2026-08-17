#!/usr/bin/env python3
"""HQNN EDA contribution — BASE-8 vs EDA-8 (5-min strided, lightning.qubit)."""

import sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import joblib

warnings.filterwarnings('ignore')

REAL_PATH = "/home/iaxiom/projects/Explorer/data/study_cleaned.csv"
MODEL_DIR = Path("/home/iaxiom/projects/Explorer/models")
CLASSES   = ['large', 'medium', 'small']
CLASS_MAP = {'large': 0, 'medium': 1, 'small': 2}
DEVICE    = torch.device('cpu')   # quantum circuit must be cpu-side

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

ALL_RESULTS = {}


MAX_SAMPLES = 1500   # QML circuits are O(N) serial — cap for feasibility

def load_5min(feat_cols):
    df = pd.read_csv(REAL_PATH, low_memory=False)
    df = df[df['size_class'].isin(CLASSES)].copy()
    avail = [c for c in feat_cols if c in df.columns]
    if 'Rtime_epoch' in df.columns and 'ObjID' in df.columns:
        df['t_window'] = (df['Rtime_epoch'] // 300).astype(int)
        df = (df.sort_values('Rtime_epoch')
                .groupby(['ObjID', 't_window'], sort=False)
                .last().reset_index())
    # Stratified cap — keep class balance
    if len(df) > MAX_SAMPLES:
        per_cls = MAX_SAMPLES // 3
        df = pd.concat([
            df[df['size_class'] == c].sample(
                n=min(per_cls, (df['size_class']==c).sum()), random_state=42)
            for c in CLASSES
        ]).reset_index(drop=True)
        print(f"  [Cap] Sampled {len(df)} rows for QML feasibility")
    df[avail] = df[avail].fillna(0)
    X = df[avail].values.astype(np.float32)
    y = np.array([CLASS_MAP[c] for c in df['size_class']])
    g = df['ObjID'].values if 'ObjID' in df.columns else np.arange(len(df))
    return X, y, g, avail


def obj_split(X, y, groups, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(X, y, groups=groups))
    return tr, te


def print_metrics(yte, preds, tag):
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    cm  = confusion_matrix(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=4)
    recall = {c: float(cm[i,i]/cm[i].sum()) if cm[i].sum()>0 else 0.0
              for i, c in enumerate(CLASSES)}
    print(f"\n  ── {tag} ──")
    print(f"  F1-macro: {f1:.4f}  |  Acc: {acc:.4f}")
    print(f"  Recall → large: {recall['large']:.4f}  medium: {recall['medium']:.4f}  small: {recall['small']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"    {'':10} {'large':>7} {'medium':>7} {'small':>7}")
    for i, c in enumerate(CLASSES):
        print(f"    {c:10} {cm[i,0]:>7} {cm[i,1]:>7} {cm[i,2]:>7}")
    print(f"\n{rep}")
    return {'f1': round(f1,4), 'acc': round(acc,4),
            'recall_large': round(recall['large'],4),
            'recall_medium': round(recall['medium'],4),
            'recall_small': round(recall['small'],4)}


def run_hqnn(feat_cols, label, save_name, n_qubits=8, n_layers=3,
             epochs=60, batch=128, lr=5e-3, patience=12):
    print(f"\n{'='*60}")
    print(f"  HQNN {n_qubits}q {n_layers}L — {label}")

    X, y, groups, avail = load_5min(feat_cols)
    tr, te = obj_split(X, y, groups)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    print(f"  Train: {len(ytr):,}  Test: {len(yte):,}")
    print(f"  Class dist train: {dict(zip(CLASSES, np.bincount(ytr, minlength=3)))}")

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)

    # Use lightning.qubit (CPU, supports batching properly via adjoint diff)
    try:
        dev = qml.device('lightning.qubit', wires=n_qubits)
        print("  Backend: lightning.qubit")
    except Exception:
        dev = qml.device('default.qubit', wires=n_qubits)
        print("  Backend: default.qubit")

    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
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

    counts = np.bincount(ytr, minlength=3)
    w = torch.tensor((len(ytr) / (3 * counts)).astype(np.float32))
    loss_fn = nn.CrossEntropyLoss(weight=w)
    model = HQNNModel(len(avail))
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    ds = TensorDataset(torch.tensor(Xtr_s), torch.tensor(ytr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    best_f1, best_ep, pat_cnt, best_state = 0, 0, 0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            preds_ep = model(torch.tensor(Xte_s)).argmax(1).numpy()
        f1_ep = f1_score(yte, preds_ep, average='macro')
        sched.step(1 - f1_ep)

        if f1_ep > best_f1:
            best_f1 = f1_ep; best_ep = ep; pat_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat_cnt += 1
            if pat_cnt >= patience: break

        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:3d}  F1={f1_ep:.4f}  best={best_f1:.4f}@ep{best_ep}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Training time: {time.time()-t0:.0f}s | Best ep: {best_ep}")

    with torch.no_grad():
        preds = model(torch.tensor(Xte_s)).argmax(1).numpy()
    m = print_metrics(yte, preds, f'HQNN {n_qubits}q [{label}]')
    ALL_RESULTS[save_name] = {'model': f'HQNN_{n_qubits}q', 'features': label, **m}

    torch.save(model.state_dict(), MODEL_DIR / f'{save_name}_weights.pt')
    pre = {'scaler': sc, 'feature_cols': avail, 'n_qubits': n_qubits,
           'n_layers': n_layers, 'classes': CLASSES}
    joblib.dump(pre, MODEL_DIR / f'{save_name}_preprocessor.pkl')
    print(f"  Saved → models/{save_name}_weights.pt")
    return m


def main():
    t0 = time.time()
    print("="*60)
    print("  HQNN EDA Contribution — BASE-8 vs EDA-8")
    print("="*60)

    run_hqnn(BASE_8, 'BASE-8', 'eda_hqnn_base8')
    run_hqnn(EDA_8,  'EDA-8',  'eda_hqnn_eda8')

    print(f"\n{'='*65}")
    print("  HQNN SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Condition':<25} {'F1':>6} {'Acc':>6} {'R-lg':>6} {'R-md':>6} {'R-sm':>6}")
    print(f"  {'-'*55}")
    labels = {'eda_hqnn_base8': 'HQNN BASE-8', 'eda_hqnn_eda8': 'HQNN EDA-8'}
    for k, lbl in labels.items():
        if k in ALL_RESULTS:
            r = ALL_RESULTS[k]
            print(f"  {lbl:<25} {r['f1']:>6.4f} {r['acc']:>6.4f} "
                  f"{r['recall_large']:>6.4f} {r['recall_medium']:>6.4f} {r['recall_small']:>6.4f}")

    with open(MODEL_DIR / 'hqnn_eda_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Total: {time.time()-t0:.0f}s")
    print(f"  Results → models/hqnn_eda_results.json")


if __name__ == '__main__':
    main()
