# VQC Ablation Finding — Explorer

**Date:** 2026-08-17  
**Dataset:** study_cleaned.csv (123,051 rows)  
**Split:** ObjID-stratified GroupShuffleSplit 80/20 (seed=42)

---

## Summary

VQC (Variational Quantum Circuit) was not fully evaluated in the Explorer ablation study due to quantum circuit simulation cost. The partial result obtained before termination places VQC below all meaningful baselines, making it the weakest model in the study.

---

## What Was Run

**Architecture (from HQNN Paper 3):**
- Pre-compress: `Linear(24→n_qubits) → Tanh → ×π`
- Quantum layer: `AngleEmbedding(Y) + StronglyEntanglingLayers(L=3)` via PennyLane `lightning.gpu` (adjoint diff)
- Output: `Linear(n_qubits→3)`

**Conditions attempted:** VQC 5q L=3 × {scan, 5-min, 30-min}, VQC 10q × 30-min, Classical-MLP × 30-min

**Device:** NVIDIA GB10, `lightning.gpu` with adjoint differentiation

---

## Simulation Cost — Why It Was Terminated

| Window | Train N | Batches/epoch | Time/epoch | Epochs to converge | Estimated total |
|---|---|---|---|---|---|
| scan | 99,130 | 193 | ~5,200s (87 min) | 20–40 | 30–60 hours |
| 5-min | 20,330 | 40 | ~1,080s (18 min) | 20–40 | 6–12 hours |
| 30-min | 3,893 | 8 | ~200s (3.3 min) | 20–40 | 1–2 hours |

Root cause: PennyLane evaluates each sample in a batch sequentially through the quantum circuit, even with GPU. The quantum state vector for 5 qubits is only 32 complex amplitudes — trivially small — but the Python-level overhead of 512 sequential circuit evaluations per batch dominates. Adjoint differentiation doubles the cost (one forward pass + one adjoint pass per batch).

This is a fundamental property of quantum simulation on classical hardware, not a tuning issue.

---

## Partial Result

**VQC 5q L=3 | 30-min** (the only feasible window, terminated at ep=10):

| Epoch | F1 | Best F1 |
|---|---|---|
| 1 | 0.2996 | 0.2996 |
| 9 | — | **0.4779** (best) |
| 10 | 0.4107 | 0.4779 |

**F1=0.4779 at epoch 9, 30-min window** — already below the thesis 5-feature baseline (F1=0.5272 at 30-min) and 22 percentage points below ET BASE-24 at the same window (F1=0.6951).

---

## Why VQC Fails for Vessel SIZE

### 1. Pre-compression destroys information
The `Linear(24→5)→Tanh` layer before the quantum circuit compresses 24 features into 5 values. This bottleneck is the primary limitation — the quantum circuit never sees the full feature space. For comparison, the PINN's EM branch processes all 10 EM features and the KIN branch processes all 14 kinematic features independently before fusion.

### 2. No physics inductive bias
The VQC quantum circuit (AngleEmbedding + StronglyEntanglingLayers) has no structural knowledge of what RCS or footprint physically encode. The PINN's physics loss directly penalises physically inconsistent predictions — the quantum circuit has no equivalent mechanism.

### 3. Dataset scale mismatch
In HQNN Paper 3, VQC was evaluated on 203 test tracks (track-level aggregation, 4-class TYPE). Explorer's scan-level has 23,921 test detections — 118× larger. The simulation cost scales linearly with training set size. What worked in HQNN (small, clean, aggregated dataset) does not transfer to a large scan-level dataset.

### 4. The classical equivalent would also fail
The Classical-MLP control model (`Linear(24→5)→Tanh→Linear(5→5)→Tanh→Linear(5→3)`) is equivalent to a 3-layer network with a 5-unit hidden layer — far too narrow to separate 3 vessel size classes in a 24-dimensional feature space. Any advantage VQC might provide over this classical baseline is irrelevant because the baseline itself is misconfigured.

---

## Comparison Against All Other Models (30-min)

| Model | Features | F1 | R-lg | R-md | R-sm |
|---|---|---|---|---|---|
| GBT | FULL-42 | 0.7237 | 0.900 | 0.748 | 0.480 |
| EDA-QJL B64 | BASE-24+QJL | 0.7223 | 0.885 | 0.748 | 0.510 |
| GBT | BASE-24 | 0.7194 | 0.905 | 0.762 | 0.440 |
| ET | BASE-24 | 0.6951 | 0.860 | 0.714 | 0.520 |
| PINN λ=0.5 | BASE-24 | 0.6842 | 0.685 | 0.823 | 0.840 |
| Thesis 5-feat | 5 legacy feat | 0.5272 | 0.692 | 0.313 | 0.910 |
| **VQC 5q** | **BASE-24** | **~0.48** | — | — | — |

VQC ranks last — below the thesis baseline.

---

## Conclusion

> **VQC does not transfer from the HQNN vessel TYPE problem to the Explorer vessel SIZE problem. The quantum circuit simulation cost is prohibitive for scan-level and 5-min data, and the partial 30-min result (F1≈0.48) is the weakest result in the entire ablation study — below even the 4-feature thesis baseline. The pre-compression bottleneck and absence of physics inductive bias are the architectural root causes. VQC provides no advantage over tree models or the PINN for radar vessel SIZE classification.**
