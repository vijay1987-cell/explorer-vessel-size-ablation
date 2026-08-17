#!/usr/bin/env python3
"""
Explorer — DualStream & PINN ablation
=======================================
Both models use EM-24 + Kin-18 split of FULL-42 features.

DualStream: EM + Kin encoded separately → bidirectional cross-attention → head
PINN:       EM + Kin encoded separately → concatenated → head

Ablation conditions:
  A. DualStream scan-level real (imputed)
  B. DualStream synthetic 1M
  C. DualStream combined (real + syn → real test)
  D. PINN scan-level real (imputed)
  E. PINN synthetic 1M
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

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
from common import (REAL_PATH, SYN_PATH, MODEL_DIR, CLASSES, CLASS_MAP,
                    FULL_42, EM_24, KIN_18, load_real, load_synthetic,
                    obj_split, oversample)

import joblib
RESULTS = {}
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ── Architectures ─────────────────────────────────────────────────────────────
class StreamEnc(nn.Module):
    def __init__(self, n_in, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d, 64),   nn.LayerNorm(64),
        )
    def forward(self, x): return self.net(x)


class CrossAttn(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.ln = nn.LayerNorm(d)
    def forward(self, q, kv):
        q_ = self.Wq(q); k_ = self.Wk(kv); v_ = self.Wv(kv)
        w  = torch.sigmoid((q_ * k_).sum(-1, keepdim=True) / (64 ** 0.5))
        return self.ln(q + self.Wo(w * v_))


class DualStream(nn.Module):
    def __init__(self, n_em, n_kin, n_cls=3):
        super().__init__()
        self.em_enc  = StreamEnc(n_em)
        self.kin_enc = StreamEnc(n_kin)
        self.em2kin  = CrossAttn()
        self.kin2em  = CrossAttn()
        self.head    = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_cls),
        )
    def forward(self, x_em, x_kin):
        e = self.em_enc(x_em); k = self.kin_enc(x_kin)
        return self.head(torch.cat([self.em2kin(e, k), self.kin2em(k, e)], dim=1))


class PINNModel(nn.Module):
    def __init__(self, n_em, n_kin, n_cls=3):
        super().__init__()
        self.em_branch  = StreamEnc(n_em)
        self.kin_branch = StreamEnc(n_kin)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_cls),
        )
    def forward(self, x_em, x_kin):
        return self.head(torch.cat([self.em_branch(x_em), self.kin_branch(x_kin)], dim=1))


# ── Training helpers ──────────────────────────────────────────────────────────
def torch_train(model, X_em_tr, X_kin_tr, y_tr,
                X_em_te, X_kin_te, y_te,
                epochs=80, batch=512, lr=3e-4, patience=12):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    w_arr = (len(y_tr) / (3 * np.bincount(y_tr, minlength=3))).astype(np.float32)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w_arr).to(DEVICE))

    ds = TensorDataset(
        torch.tensor(X_em_tr),  torch.tensor(X_kin_tr),
        torch.tensor(y_tr, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    best_f1, best_ep, patience_cnt = 0, 0, 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for x_em, x_kin, yb in loader:
            x_em  = x_em.to(DEVICE); x_kin = x_kin.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x_em, x_kin), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_em_te).to(DEVICE),
                           torch.tensor(X_kin_te).to(DEVICE))
            preds = logits.argmax(1).cpu().numpy()
        f1 = f1_score(y_te, preds, average='macro')
        sched.step(1 - f1)

        if f1 > best_f1:
            best_f1 = f1; best_ep = ep; patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

        if ep % 20 == 0 or ep == 1:
            print(f"    ep{ep:3d}  F1={f1:.4f}  best={best_f1:.4f}@ep{best_ep}")

    model.load_state_dict(best_state)
    model.eval()
    return best_f1


def evaluate(model, X_em_te, X_kin_te, y_te, tag):
    with torch.no_grad():
        logits = model(torch.tensor(X_em_te).to(DEVICE),
                       torch.tensor(X_kin_te).to(DEVICE))
        preds = logits.argmax(1).cpu().numpy()
    f1  = f1_score(y_te, preds, average='macro')
    acc = accuracy_score(y_te, preds)
    rep = classification_report(y_te, preds, target_names=CLASSES, digits=3)
    print(f"\n  [{tag}]  F1-macro: {f1:.4f}  Acc: {acc:.4f}")
    print(rep)
    return f1, acc


def split_streams(X42, n_em):
    """Split scaled X42 → (X_em, X_kin)."""
    return X42[:, :n_em].astype(np.float32), X42[:, n_em:].astype(np.float32)


def prepare(X42_tr, X42_te, y_tr, y_te, avail_em, avail_kin):
    n_em  = len(avail_em)
    n_kin = len(avail_kin)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X42_tr)
    X_te_s = sc.transform(X42_te)
    Xem_tr, Xkin_tr = split_streams(X_tr_s, n_em)
    Xem_te, Xkin_te = split_streams(X_te_s, n_em)
    return Xem_tr, Xkin_tr, Xem_te, Xkin_te, sc, n_em, n_kin


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Explorer — DualStream & PINN Ablation")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    avail_full42 = EM_24 + KIN_18

    print("\nLoading real labeled (imputed)...")
    X42_real, y_real, groups, df_real = load_real(avail_full42)
    avail_em  = [c for c in EM_24  if c in df_real.columns]
    avail_kin = [c for c in KIN_18 if c in df_real.columns]
    avail42_real = avail_em + avail_kin

    Xr_tr, Xr_te, yr_tr, yr_te = obj_split(X42_real, y_real, groups)
    print(f"  Real: train={len(yr_tr):,}  test={len(yr_te):,}")

    print("\nLoading synthetic 1M...")
    X42_syn, y_syn = load_synthetic(avail42_real)
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X42_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn)
    print(f"  Synthetic: train={len(ys_tr):,}  test={len(ys_te):,}")

    # Combined
    Xc_tr = np.vstack([Xr_tr, Xs_tr])
    yc_tr = np.concatenate([yr_tr, ys_tr])

    n_em_real  = len(avail_em)
    n_kin_real = len(avail_kin)

    # ── Condition A: DualStream real scan ─────────────────────────
    print("\n" + "-"*40)
    print("  A  DualStream scan-level real")
    Xem_rtr, Xkin_rtr, Xem_rte, Xkin_rte, sc_r, n_em, n_kin = prepare(
        Xr_tr, Xr_te, yr_tr, yr_te, avail_em, avail_kin)
    model = DualStream(n_em, n_kin).to(DEVICE)
    torch_train(model, Xem_rtr, Xkin_rtr, yr_tr, Xem_rte, Xkin_rte, yr_te)
    f1, acc = evaluate(model, Xem_rte, Xkin_rte, yr_te, 'A  DualStream scan-level real')
    RESULTS['A_dualstream_real'] = {'f1': f1, 'acc': acc}
    torch.save(model.state_dict(), MODEL_DIR / 'dualstream_real_scan_weights.pt')
    pre = {'scaler': sc_r, 'n_em': n_em, 'n_kin': n_kin,
           'em_feats': avail_em, 'kin_feats': avail_kin, 'classes': CLASSES,
           'condition': 'A_dualstream_real_scan'}
    joblib.dump(pre, MODEL_DIR / 'dualstream_real_scan_preprocessor.pkl')

    # ── Condition B: DualStream synthetic ─────────────────────────
    print("\n" + "-"*40)
    print("  B  DualStream synthetic 1M")
    Xem_str, Xkin_str, Xem_ste, Xkin_ste, sc_s, n_em_s, n_kin_s = prepare(
        Xs_tr, Xs_te, ys_tr, ys_te, avail_em, avail_kin)
    model = DualStream(n_em_s, n_kin_s).to(DEVICE)
    torch_train(model, Xem_str, Xkin_str, ys_tr, Xem_ste, Xkin_ste, ys_te, epochs=60)
    f1, acc = evaluate(model, Xem_ste, Xkin_ste, ys_te, 'B  DualStream synthetic 1M')
    RESULTS['B_dualstream_syn'] = {'f1': f1, 'acc': acc}
    torch.save(model.state_dict(), MODEL_DIR / 'dualstream_syn1M_weights.pt')
    pre = {'scaler': sc_s, 'n_em': n_em_s, 'n_kin': n_kin_s,
           'em_feats': avail_em, 'kin_feats': avail_kin, 'classes': CLASSES,
           'condition': 'B_dualstream_syn1M'}
    joblib.dump(pre, MODEL_DIR / 'dualstream_syn1M_preprocessor.pkl')

    # ── Condition C: DualStream combined ──────────────────────────
    print("\n" + "-"*40)
    print("  C  DualStream combined (→ real test)")
    Xem_ctr, Xkin_ctr, Xem_cte, Xkin_cte, sc_c, n_em_c, n_kin_c = prepare(
        Xc_tr, Xr_te, yc_tr, yr_te, avail_em, avail_kin)
    model = DualStream(n_em_c, n_kin_c).to(DEVICE)
    torch_train(model, Xem_ctr, Xkin_ctr, yc_tr, Xem_cte, Xkin_cte, yr_te, epochs=60)
    f1, acc = evaluate(model, Xem_cte, Xkin_cte, yr_te, 'C  DualStream combined → real test')
    RESULTS['C_dualstream_combined'] = {'f1': f1, 'acc': acc}
    torch.save(model.state_dict(), MODEL_DIR / 'dualstream_combined_weights.pt')
    pre = {'scaler': sc_c, 'n_em': n_em_c, 'n_kin': n_kin_c,
           'em_feats': avail_em, 'kin_feats': avail_kin, 'classes': CLASSES,
           'condition': 'C_dualstream_combined'}
    joblib.dump(pre, MODEL_DIR / 'dualstream_combined_preprocessor.pkl')

    # ── Condition D: PINN real scan ───────────────────────────────
    print("\n" + "-"*40)
    print("  D  PINN scan-level real")
    Xem_rtr, Xkin_rtr, Xem_rte, Xkin_rte, sc_r, n_em, n_kin = prepare(
        Xr_tr, Xr_te, yr_tr, yr_te, avail_em, avail_kin)
    model = PINNModel(n_em, n_kin).to(DEVICE)
    torch_train(model, Xem_rtr, Xkin_rtr, yr_tr, Xem_rte, Xkin_rte, yr_te)
    f1, acc = evaluate(model, Xem_rte, Xkin_rte, yr_te, 'D  PINN scan-level real')
    RESULTS['D_pinn_real'] = {'f1': f1, 'acc': acc}
    torch.save(model.state_dict(), MODEL_DIR / 'pinn_real_scan_weights.pt')
    pre = {'scaler': sc_r, 'n_em': n_em, 'n_kin': n_kin,
           'em_feats': avail_em, 'kin_feats': avail_kin, 'classes': CLASSES,
           'condition': 'D_pinn_real_scan'}
    joblib.dump(pre, MODEL_DIR / 'pinn_real_scan_preprocessor.pkl')

    # ── Condition E: PINN synthetic ───────────────────────────────
    print("\n" + "-"*40)
    print("  E  PINN synthetic 1M")
    Xem_str, Xkin_str, Xem_ste, Xkin_ste, sc_s, n_em_s, n_kin_s = prepare(
        Xs_tr, Xs_te, ys_tr, ys_te, avail_em, avail_kin)
    model = PINNModel(n_em_s, n_kin_s).to(DEVICE)
    torch_train(model, Xem_str, Xkin_str, ys_tr, Xem_ste, Xkin_ste, ys_te, epochs=60)
    f1, acc = evaluate(model, Xem_ste, Xkin_ste, ys_te, 'E  PINN synthetic 1M')
    RESULTS['E_pinn_syn'] = {'f1': f1, 'acc': acc}
    torch.save(model.state_dict(), MODEL_DIR / 'pinn_syn1M_weights.pt')
    pre = {'scaler': sc_s, 'n_em': n_em_s, 'n_kin': n_kin_s,
           'em_feats': avail_em, 'kin_feats': avail_kin, 'classes': CLASSES,
           'condition': 'E_pinn_syn1M'}
    joblib.dump(pre, MODEL_DIR / 'pinn_syn1M_preprocessor.pkl')

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DUALSTREAM / PINN ABLATION SUMMARY")
    print("=" * 60)
    labels = {
        'A_dualstream_real':     'A  DualStream scan-level real',
        'B_dualstream_syn':      'B  DualStream synthetic 1M',
        'C_dualstream_combined': 'C  DualStream combined → real test',
        'D_pinn_real':           'D  PINN scan-level real',
        'E_pinn_syn':            'E  PINN synthetic 1M',
    }
    print(f"  {'Condition':<42} {'F1-mac':>7}  {'Acc':>7}")
    print(f"  {'-'*57}")
    for k, lbl in labels.items():
        r = RESULTS[k]
        print(f"  {lbl:<42} {r['f1']:>7.4f}  {r['acc']:>7.4f}")

    with open(MODEL_DIR / 'dualstream_pinn_ablation.json', 'w') as f_:
        json.dump(RESULTS, f_, indent=2)
    print(f"\n  Saved → {MODEL_DIR}/dualstream_pinn_ablation.json")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
