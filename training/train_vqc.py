#!/usr/bin/env python3
"""
VQC Ablation — Explorer Dataset
================================
Architecture (from HQNN Paper 3):
  Classical pre: Linear(n_feat -> n_qubits) -> Tanh -> scale to [-pi, pi]
  Quantum layer: AngleEmbedding + StronglyEntanglingLayers (L layers)
  Classical out: Linear(n_qubits -> 3)

Feature set: BASE-24 (n_feat=24) for all conditions.
n_qubits in {5, 10}, n_layers=3 (matching Paper 3 best config).
Windows: scan / 5-min / 30-min.

Also includes a Classical-MLP equivalent (same pre/post, no quantum layer)
to isolate the quantum contribution at scan level.

Note: lightning.qubit (CPU simulator) — scan level may be slow.
      Uses CrossEntropyLoss(weight=class_weights) — no oversampling.

HQNN Paper 3 reference:
  VQC (5 qubits, 3 layers): acc=72.4%, F1=71.4%  [4-class TYPE, 203 test tracks]
"""

import time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

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
WINDOWS = [('scan', None), ('5-min', 300), ('30-min', 1800)]
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


# ── VQC model ─────────────────────────────────────────────────────────────────
def _build_vqc(n_features, n_qubits, n_layers):
    try:
        dev = qml.device('lightning.qubit', wires=n_qubits)
        diff_method = 'adjoint'
    except Exception:
        dev = qml.device('default.qubit', wires=n_qubits)
        diff_method = 'backprop'

    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {'weights': (n_layers, n_qubits, 3)}
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    return qlayer


class VQCModel(nn.Module):
    def __init__(self, n_features, n_qubits=5, n_layers=3, n_cls=3):
        super().__init__()
        self.pre     = nn.Sequential(nn.Linear(n_features, n_qubits), nn.Tanh())
        self.qlayer  = _build_vqc(n_features, n_qubits, n_layers)
        self.output  = nn.Linear(n_qubits, n_cls)
        self.n_qubits = n_qubits

    def forward(self, x):
        x = self.pre(x) * np.pi
        x = self.qlayer(x)
        return self.output(x)


class ClassicalMLP(nn.Module):
    """Classical equivalent: same pre/post architecture, no quantum layer."""
    def __init__(self, n_features, n_qubits=5, n_cls=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_qubits), nn.Tanh(),
            nn.Linear(n_qubits, n_qubits),   nn.Tanh(),
            nn.Linear(n_qubits, n_cls),
        )
    def forward(self, x): return self.net(x)


# ── metrics ───────────────────────────────────────────────────────────────────
def report(yte, preds, label, n_tr, n_te):
    f1  = f1_score(yte, preds, average='macro')
    acc = accuracy_score(yte, preds)
    cm  = confusion_matrix(yte, preds)
    rep = classification_report(yte, preds, target_names=CLASSES, digits=4)
    recall = {c: float(cm[i,i]/cm[i].sum()) if cm[i].sum()>0 else 0.
              for i,c in enumerate(CLASSES)}
    print(f"\n  -- {label} --")
    print(f"  Train: {n_tr:,}  Test: {n_te:,}")
    print(f"  F1: {f1:.4f}  Acc: {acc:.4f}  R-lg: {recall['large']:.4f}  R-md: {recall['medium']:.4f}  R-sm: {recall['small']:.4f}")
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


# ── training loop ─────────────────────────────────────────────────────────────
def train_model(model, Xtr, ytr, Xte, yte, label, key,
                epochs=100, batch=512, lr=1e-2, patience=20):
    torch.manual_seed(SEED)
    counts  = np.bincount(ytr, minlength=3)
    w       = torch.tensor((len(ytr) / (3*counts)).astype(np.float32))
    loss_fn = nn.CrossEntropyLoss(weight=w)

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds     = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    Xte_t = torch.tensor(Xte)
    best_f1, best_ep, pat_cnt, best_state = 0, 0, 0, None
    t0 = time.time()

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            preds_ep = model(Xte_t).argmax(1).numpy()
        f1_ep = f1_score(yte, preds_ep, average='macro')

        if f1_ep > best_f1:
            best_f1 = f1_ep; best_ep = ep; pat_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat_cnt += 1
            if pat_cnt >= patience: break

        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - t0
            print(f"  ep{ep:3d}  F1={f1_ep:.4f}  best={best_f1:.4f}@ep{best_ep}  [{elapsed:.0f}s]")

    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.0f}s | Best ep: {best_ep}")

    with torch.no_grad():
        preds = model(Xte_t).argmax(1).numpy()
    return report(yte, preds, label, len(ytr), len(yte))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    np.random.seed(SEED)

    df_raw = pd.read_csv(REAL_PATH, low_memory=False)
    df_raw = df_raw[df_raw['size_class'].isin(CLASSES)].copy()

    for win_tag, win_sec in WINDOWS:
        df  = apply_window(df_raw, win_sec)
        y   = np.array([CLASS_MAP[c] for c in df['size_class']])
        grp = df['ObjID'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr, te = next(gss.split(df, y, groups=grp))
        ytr, yte = y[tr], y[te]

        avail = [c for c in BASE_24 if c in df.columns]
        sc = StandardScaler().fit(df.iloc[tr][avail].fillna(0).values)
        Xtr = sc.transform(df.iloc[tr][avail].fillna(0).values).astype(np.float32)
        Xte = sc.transform(df.iloc[te][avail].fillna(0).values).astype(np.float32)

        print(f"\n{'#'*65}")
        print(f"  Window: {win_tag}   train={len(ytr):,}  test={len(yte):,}  feats={len(avail)}")
        print(f"{'#'*65}")

        # ── VQC 5 qubits ──────────────────────────────────────────────────────
        for n_qubits in ([5, 10] if win_tag == 'scan' else [5]):
            label = f"VQC {n_qubits}q L=3 | {win_tag}"
            key   = f"vqc_{win_tag.replace('-','').lower()}_q{n_qubits}"
            print(f"\n{'='*60}")
            print(f"  {label}  [params: pre={len(avail)*n_qubits+n_qubits} + Q={3*n_qubits*3} + out={n_qubits*3+3}]")
            model = VQCModel(n_features=len(avail), n_qubits=n_qubits, n_layers=3)
            m = train_model(model, Xtr, ytr, Xte, yte, label, key)
            ALL_RESULTS[key] = {'model':'VQC','n_qubits':n_qubits,'n_layers':3,'window':win_tag,**m}

        # ── Classical equivalent (scan only) ──────────────────────────────────
        if win_tag == 'scan':
            label_c = f"Classical-MLP 5 | scan"
            key_c   = f"classical_mlp_scan_q5"
            print(f"\n{'='*60}")
            print(f"  {label_c}  [classical equivalent, no quantum layer]")
            model_c = ClassicalMLP(n_features=len(avail), n_qubits=5)
            m_c = train_model(model_c, Xtr, ytr, Xte, yte, label_c, key_c)
            ALL_RESULTS[key_c] = {'model':'Classical-MLP','n_qubits':5,'window':'scan',**m_c}

    # Summary
    print(f"\n{'='*80}")
    print("  VQC ABLATION -- SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Model':<22} {'Window':<8} {'F1':>6} {'Acc':>6} {'R-lg':>6} {'R-md':>6} {'R-sm':>6} {'N-test':>7}")
    print(f"  {'-'*75}")
    for key, r in ALL_RESULTS.items():
        model_tag = f"{r['model']} {r.get('n_qubits','?')}q" if r['model']=='VQC' else r['model']
        print(f"  {model_tag:<22} {r['window']:<8} {r['f1']:>6.4f} {r['acc']:>6.4f} "
              f"{r['recall_large']:>6.4f} {r['recall_medium']:>6.4f} "
              f"{r['recall_small']:>6.4f} {r['n_test']:>7,}")

    refs = [
        ('XGB BASE-24',    'scan',   0.6250),
        ('GBT FULL-42',    'scan',   0.7108),
        ('PINN lam=0.5',   'scan',   0.7273),
        ('GBT FULL-42',    '30-min', 0.7237),
        ('XW-ensemble',    '5-min',  0.7205),
    ]
    print(f"\n  [Reference baselines]")
    for name, wt, f1 in refs:
        print(f"  {name:<22} {wt:<8} {f1:.4f}")

    with open(MODEL_DIR / 'vqc_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n  Total runtime: {time.time()-t_total:.0f}s")
    print(f"  Results -> models/vqc_results.json")


if __name__ == '__main__':
    main()
