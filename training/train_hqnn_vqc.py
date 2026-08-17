#!/usr/bin/env python3
"""
Explorer — HQNN & VQC ablation
================================
Both models use 8 EM features (8 qubits), 3 layers.
Input is 5-min strided track-level data (required for quantum models).

Ablation:
  A. HQNN 8q 3L — real 5-min (imputed)
  B. HQNN 8q 3L — synthetic 1M
  C. HQNN 10q 4L — real 5-min (capacity ablation)
  D. VQC 8q 3L  — real 5-min (imputed)
  E. VQC 8q 3L  — synthetic 1M
"""

import sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pennylane as qml

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
from common import (REAL_PATH, SYN_PATH, MODEL_DIR, CLASSES, CLASS_MAP,
                    FULL_42, HQNN_8, load_real, load_synthetic,
                    obj_split, oversample)

import joblib
RESULTS = {}
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

try:
    _dev_test = qml.device('lightning.gpu', wires=1)
    QML_DEV = 'lightning.gpu'
except Exception:
    QML_DEV = 'default.qubit'
print(f"PennyLane device: {QML_DEV}")


# ── QML models (must match HQNNClassifier from src/) ─────────────────────────
def make_hqnn(n_qubits, n_layers, n_features, n_classes=3):
    dev = qml.device(QML_DEV, wires=n_qubits)

    @qml.qnode(dev, interface='torch')
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs[:n_qubits], wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    class _HQNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(n_features, n_qubits) if n_features != n_qubits else nn.Identity()
            self.qlayer = qml.qnn.TorchLayer(
                circuit, {'weights': (n_layers, n_qubits, 3)}
            )
            self.head = nn.Linear(n_qubits, n_classes)
        def forward(self, x):
            return self.head(self.qlayer(torch.tanh(self.pre(x)) * 3.14159))

    return _HQNNModel()


def make_vqc(n_qubits, n_layers, n_classes=3):
    dev = qml.device(QML_DEV, wires=n_qubits)

    @qml.qnode(dev, interface='torch')
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    class _VQCModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(circuit, {'weights': (n_layers, n_qubits)})
            self.head   = nn.Linear(n_qubits, n_classes)
        def forward(self, x):
            return self.head(self.qlayer(x))

    return _VQCModel()


def qml_train(model, X_tr, y_tr, X_te, y_te,
              epochs=60, batch=256, lr=5e-3, patience=15):
    w_arr = (len(y_tr) / (3 * np.bincount(y_tr, minlength=3))).astype(np.float32)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w_arr).to(DEVICE))
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)

    ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    best_f1, best_ep, patience_cnt, best_state = 0, 0, 0, None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_te).to(DEVICE)).argmax(1).cpu().numpy()
        f1 = f1_score(y_te, preds, average='macro')
        sched.step(1 - f1)

        if f1 > best_f1:
            best_f1 = f1; best_ep = ep; patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

        if ep % 15 == 0 or ep == 1:
            print(f"    ep{ep:3d}  F1={f1:.4f}  best={best_f1:.4f}@ep{best_ep}")

    model.load_state_dict(best_state)
    model.eval()
    return best_f1


def evaluate(model, X_te, y_te, tag):
    with torch.no_grad():
        preds = model(torch.tensor(X_te).to(DEVICE)).argmax(1).cpu().numpy()
    f1  = f1_score(y_te, preds, average='macro')
    acc = accuracy_score(y_te, preds)
    rep = classification_report(y_te, preds, target_names=CLASSES, digits=3)
    print(f"\n  [{tag}]  F1-macro: {f1:.4f}  Acc: {acc:.4f}")
    print(rep)
    return f1, acc


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Explorer — HQNN & VQC Ablation")
    print("=" * 60)

    hqnn_feats = [c for c in HQNN_8 if True]  # all 8 features
    n_qubits, n_layers = 8, 3

    # ── Real data: 5-min strided ──────────────────────────────────
    print("\nLoading real labeled (imputed, 5-min strided)...")
    df_raw = pd.read_csv(REAL_PATH, low_memory=False)
    avail8 = [c for c in hqnn_feats if c in df_raw.columns]
    df_raw[avail8] = df_raw[avail8].fillna(df_raw[avail8].median())
    df_raw = df_raw[df_raw['size_class'].isin(CLASSES)]
    if 'Rtime_epoch' in df_raw.columns and 'ObjID' in df_raw.columns:
        df_raw['t_window'] = (df_raw['Rtime_epoch'] // 300).astype(int)
        df_5min = (df_raw.sort_values('Rtime_epoch')
                         .groupby(['ObjID', 't_window'], sort=False)
                         .last().reset_index())
    else:
        df_5min = df_raw
    X5_all = df_5min[avail8].values.astype(np.float32)
    y5_all = np.array([CLASS_MAP[c] for c in df_5min['size_class']])
    g5_all = df_5min['ObjID'].values if 'ObjID' in df_5min.columns else np.arange(len(df_5min))

    from common import obj_split
    X5_tr, X5_te, y5_tr, y5_te = obj_split(X5_all, y5_all, g5_all)
    sc5 = StandardScaler()
    X5_tr_s = sc5.fit_transform(X5_tr)
    X5_te_s = sc5.transform(X5_te)
    print(f"  Real 5-min: train={len(y5_tr):,}  test={len(y5_te):,}")

    # ── Synthetic ─────────────────────────────────────────────────
    print("\nLoading synthetic 1M...")
    X_syn, y_syn = load_synthetic(avail8)
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn)
    sc_syn = StandardScaler()
    Xs_tr_s = sc_syn.fit_transform(Xs_tr)
    Xs_te_s = sc_syn.transform(Xs_te)
    print(f"  Synthetic: train={len(ys_tr):,}  test={len(ys_te):,}")

    # ── A: HQNN 8q real ──────────────────────────────────────────
    print("\n" + "-"*40)
    print("  A  HQNN 8q 3L — real 5-min")
    model = make_hqnn(n_qubits, n_layers, n_features=len(avail8)).to(DEVICE)
    qml_train(model, X5_tr_s, y5_tr, X5_te_s, y5_te)
    f1, acc = evaluate(model, X5_te_s, y5_te, 'A  HQNN 8q 3L real 5-min')
    RESULTS['A_hqnn_real'] = {'f1': f1, 'acc': acc}
    torch.save(model.state_dict(), MODEL_DIR / 'hqnn_8q3l_real5min_weights.pt')
    pre = {'scaler': sc5, 'features': avail8, 'n_qubits': n_qubits,
           'n_layers': n_layers, 'classes': CLASSES, 'condition': 'A_hqnn_real_5min'}
    joblib.dump(pre, MODEL_DIR / 'hqnn_8q3l_real5min_preprocessor.pkl')

    # ── B: HQNN 8q synthetic ─────────────────────────────────────
    print("\n" + "-"*40)
    print("  B  HQNN 8q 3L — synthetic 1M")
    model = make_hqnn(n_qubits, n_layers, n_features=len(avail8)).to(DEVICE)
    qml_train(model, Xs_tr_s, ys_tr, Xs_te_s, ys_te, epochs=40)
    f1, acc = evaluate(model, Xs_te_s, ys_te, 'B  HQNN 8q 3L synthetic 1M')
    RESULTS['B_hqnn_syn'] = {'f1': f1, 'acc': acc}
    torch.save(model.state_dict(), MODEL_DIR / 'hqnn_8q3l_syn1M_weights.pt')
    pre = {'scaler': sc_syn, 'features': avail8, 'n_qubits': n_qubits,
           'n_layers': n_layers, 'classes': CLASSES, 'condition': 'B_hqnn_syn1M'}
    joblib.dump(pre, MODEL_DIR / 'hqnn_8q3l_syn1M_preprocessor.pkl')

    # ── C: HQNN 10q larger ───────────────────────────────────────
    print("\n" + "-"*40)
    print("  C  HQNN 10q 4L — real 5-min (capacity ablation)")
    # 10 qubits: use top-10 features from FULL-42 (add footprint_m2, rcs_conc)
    hqnn10_feats = avail8 + [c for c in ['rcs_conc', 'footprint_m2'] if c in df_5min.columns][:2]
    hqnn10_feats = hqnn10_feats[:10]
    X10_all = df_5min[hqnn10_feats].fillna(df_5min[hqnn10_feats].median()).values.astype(np.float32)
    X10_tr, X10_te, _, _ = obj_split(X10_all, y5_all, g5_all)
    sc10 = StandardScaler()
    X10_tr_s = sc10.fit_transform(X10_tr)
    X10_te_s = sc10.transform(X10_te)
    model = make_hqnn(10, 4, n_features=len(hqnn10_feats)).to(DEVICE)
    qml_train(model, X10_tr_s, y5_tr, X10_te_s, y5_te)
    f1, acc = evaluate(model, X10_te_s, y5_te, 'C  HQNN 10q 4L real 5-min')
    RESULTS['C_hqnn_10q'] = {'f1': f1, 'acc': acc}
    torch.save(model.state_dict(), MODEL_DIR / 'hqnn_10q4l_real5min_weights.pt')
    pre = {'scaler': sc10, 'features': hqnn10_feats, 'n_qubits': 10,
           'n_layers': 4, 'classes': CLASSES, 'condition': 'C_hqnn_10q4l_real_5min'}
    joblib.dump(pre, MODEL_DIR / 'hqnn_10q4l_real5min_preprocessor.pkl')

    # ── D: VQC 8q real ───────────────────────────────────────────
    print("\n" + "-"*40)
    print("  D  VQC 8q 3L — real 5-min")
    model = make_vqc(n_qubits, n_layers).to(DEVICE)
    qml_train(model, X5_tr_s, y5_tr, X5_te_s, y5_te)
    f1, acc = evaluate(model, X5_te_s, y5_te, 'D  VQC 8q 3L real 5-min')
    RESULTS['D_vqc_real'] = {'f1': f1, 'acc': acc}
    torch.save(model.state_dict(), MODEL_DIR / 'vqc_8q3l_real5min_weights.pt')
    pre = {'scaler': sc5, 'features': avail8, 'n_qubits': n_qubits,
           'n_layers': n_layers, 'classes': CLASSES, 'condition': 'D_vqc_real_5min'}
    joblib.dump(pre, MODEL_DIR / 'vqc_8q3l_real5min_preprocessor.pkl')

    # ── E: VQC 8q synthetic ──────────────────────────────────────
    print("\n" + "-"*40)
    print("  E  VQC 8q 3L — synthetic 1M")
    model = make_vqc(n_qubits, n_layers).to(DEVICE)
    qml_train(model, Xs_tr_s, ys_tr, Xs_te_s, ys_te, epochs=40)
    f1, acc = evaluate(model, Xs_te_s, ys_te, 'E  VQC 8q 3L synthetic 1M')
    RESULTS['E_vqc_syn'] = {'f1': f1, 'acc': acc}
    torch.save(model.state_dict(), MODEL_DIR / 'vqc_8q3l_syn1M_weights.pt')
    pre = {'scaler': sc_syn, 'features': avail8, 'n_qubits': n_qubits,
           'n_layers': n_layers, 'classes': CLASSES, 'condition': 'E_vqc_syn1M'}
    joblib.dump(pre, MODEL_DIR / 'vqc_8q3l_syn1M_preprocessor.pkl')

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HQNN / VQC ABLATION SUMMARY")
    print("=" * 60)
    labels = {
        'A_hqnn_real': 'A  HQNN 8q 3L real 5-min',
        'B_hqnn_syn':  'B  HQNN 8q 3L synthetic 1M',
        'C_hqnn_10q':  'C  HQNN 10q 4L real 5-min (capacity)',
        'D_vqc_real':  'D  VQC 8q 3L real 5-min',
        'E_vqc_syn':   'E  VQC 8q 3L synthetic 1M',
    }
    print(f"  {'Condition':<42} {'F1-mac':>7}  {'Acc':>7}")
    print(f"  {'-'*57}")
    for k, lbl in labels.items():
        r = RESULTS[k]
        print(f"  {lbl:<42} {r['f1']:>7.4f}  {r['acc']:>7.4f}")

    with open(MODEL_DIR / 'hqnn_vqc_ablation.json', 'w') as f_:
        json.dump(RESULTS, f_, indent=2)
    print(f"\n  Saved → {MODEL_DIR}/hqnn_vqc_ablation.json")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
