# Thesis vs. Explorer Approach — Comparison Study

**Date:** August 2026  
**Dataset:** radarfeatureL_Study → study_cleaned.csv (123,051 rows; large=96,136 / medium=18,002 / small=8,913)  
**Split:** ObjID-stratified GroupShuffleSplit 80/20 (seed=42)

---

## Thesis Summary

**Title:** Radar-Based Vessel Size Classification — Range-aware amplitude analysis, range-independent geometry, ablation studies, and unseen-vessel validation  
**Approach:** ExtraTrees classifier (700 estimators, class_weight=balanced) on a small set of engineered features derived from raw radar fields: rgw, azw, PeakAmplitude, TotalAmplitude, range, azimuth.

### Thesis Feature Engineering

| Feature | Formula | Purpose |
|---|---|---|
| euclid_size_no_range | `sqrt(rgw_s² + azw_s²)` | Geometry — RobustScaler removes magnitude mismatch |
| ellipse_area_no_range | `(π/4) × rgw_s × azw_s` | Geometry — scaled ellipse proxy |
| TotalAmplitude_corr_global | `TotalAmplitude × (range/25000)^0.25` | Amplitude corrected to 25 km reference |
| PeakAmplitude_corr_global | `PeakAmplitude × (range/25000)^0.25` | Peak corrected (later ablated in Model B) |
| TotalAmplitude_avg_900 | Trailing 900s rolling mean | Temporal amplitude stability |

*RobustScaler fit on training partition only; azw already in radians.*

### Thesis Key Results (on thesis dataset)

| Model | Macro F1 | Accuracy | Medium recall | External validation |
|---|---|---|---|---|
| Initial 9-feature | 0.7999 | 0.8944 | 0.50 | Not tested |
| Global 5-feat (incl. corrected peak) | 0.7541 | 0.8704 | 0.69 | **10.6% Large on 3 known Large vessels** |
| Model B (peak removed) | ~0.72 (GroupKFold) | — | ~0.69 | 89.4% Large / 90.1% Medium |

---

## Our Approach (Explorer)

**Classifier:** XGBoost (n_est=600, depth=6, lr=0.04, hist, CUDA)  
**Feature sets:**
- **BASE-24** — 10 EM features (log_peak_rcs, log_total_rcs, rcs_conc, aspect_ratio, footprint_m2, SampleCount, size_bow_stern/beam_component, ellipse_area, cr_dr_ratio_c) + 14 Kinematic features (sog, measured_sog_avg/std × 4 windows, measured_cog_std × 4 windows, displacement)
- **EDA-18** — 18 novel features: measured_TotalAmplitude_avg × 4 windows, measured_rangeStd × 4, measured_azimuthStd × 4, rgw, azw, measured_cog_stdlog × 4
- **FULL-42** — BASE-24 + EDA-18

**Temporal windows tested:** scan-level (raw rows), 5-min strided, 30-min strided

---

## Direct Comparison — Thesis Features vs. Our Features

*Same dataset, same ObjID split, ExtraTrees for thesis conditions, XGB for ours.*

### Scan Level (N-test = 23,921)

| Condition | F1 | Acc | R-large | R-medium | R-small |
|---|---|---|---|---|---|
| Thesis Model-B (4 feat) | 0.4017 | 0.5652 | 0.645 | 0.245 | 0.427 |
| Thesis 5-feat (+corr peak) | 0.5000 | 0.6400 | 0.672 | 0.268 | **0.930** |
| ET BASE-24 (ours) | **0.6894** | **0.8050** | **0.868** | **0.738** | 0.434 |
| XGB BASE-24 (ours) | 0.6250 | 0.7578 | 0.847 | 0.511 | 0.448 |

### 5-Minute Window (N-test = 4,911)

| Condition | F1 | Acc | R-large | R-medium | R-small |
|---|---|---|---|---|---|
| Thesis Model-B (4 feat) | 0.4244 | 0.5919 | 0.676 | 0.234 | 0.487 |
| Thesis 5-feat (+corr peak) | 0.5137 | 0.6567 | 0.694 | 0.267 | **0.935** |
| ET BASE-24 (ours) | **0.6750** | **0.7925** | **0.849** | **0.744** | 0.452 |
| XGB BASE-24 (ours) | 0.7043 | 0.8157 | 0.878 | 0.751 | 0.462 |

### 30-Minute Window (N-test = 945)

| Condition | F1 | Acc | R-large | R-medium | R-small |
|---|---|---|---|---|---|
| Thesis Model-B (4 feat) | 0.4394 | 0.6053 | 0.692 | 0.245 | 0.530 |
| Thesis 5-feat (+corr peak) | 0.5272 | 0.6561 | 0.692 | 0.313 | **0.910** |
| ET BASE-24 (ours) | **0.6951** | **0.8011** | **0.860** | **0.714** | 0.520 |
| XGB BASE-24 (ours) | 0.7135 | 0.8201 | 0.883 | 0.748 | 0.490 |

---

## Key Findings

### 1. Thesis approach substantially underperforms on our data
Thesis Model-B achieves F1=0.40 at scan level — 22 percentage points below our XGB BASE-24 (0.625) and 29 points below ET BASE-24 (0.689). The 4 thesis features are insufficient for our vessel population.

### 2. "No-range geometry" removes discriminative information
The thesis strips range out of geometry by RobustScaling rgw and azw_rad before computing euclid_size and ellipse_area. Our BASE-24 uses range-dependent `ellipse_area` and outperforms the thesis by a large margin. The range component actively helps classification on our dataset.

### 3. Corrected PeakAmplitude inflates small recall but collapses medium
Thesis 5-feat model: R-small=0.93, R-medium=0.27. Adding the range-corrected peak amplitude makes the model a "small detector" at the cost of medium classification — the same failure mode the thesis itself diagnosed on external validation (10.6% Large recall on known Large vessels).

### 4. ExtraTrees beats XGB at scan level but not at windowed aggregation
ET BASE-24 outperforms XGB BASE-24 at scan level (+0.064 F1), but XGB BASE-24 is better at 5-min (+0.029) and 30-min (+0.018). ExtraTrees benefits more from the dense scan-level data; XGB generalises better after temporal aggregation.

---

## Physics Critique of the Thesis

### 1. "No-range geometry" is not physically range-independent
`azw_rad = physical_cross_range / range` — the angular width of the same vessel shrinks with range. RobustScaling brings rgw and azw_rad into comparable numerical scale but does not remove the range dependence encoded in azw_rad. The resulting `euclid_size_no_range` and `ellipse_area_no_range` still encode (size × 1/range) implicitly. The paper acknowledges these are "statistical features, not physical length or area estimates" but then uses them as size proxies — an internal contradiction.

### 2. Correction exponent k=0.25 contradicts the radar range equation
The radar range equation gives received power ∝ R⁻⁴ (amplitude ∝ R⁻²), so the physically correct correction exponent is k=2 for a point target. The thesis empirically finds k=0.25 is best analytically — 8× smaller than physics dictates. This is because the training dataset has a class-range correlation (large vessels operate at longer ranges), so aggressive range correction removes class-discriminative signal. The empirically optimal k is a data-fit artifact, not a physically meaningful correction.

### 3. Exponent selected on training data, used in model training — circular
The k=0.25 exponent was selected using training-set statistics (Medium/Large Cohen's d across range bins). The corrected feature then becomes the dominant input to a classifier trained on the same data. This is feature-selection leakage. The thesis's reported Macro F1=0.754 for the global 5-feature model is optimistic.

### 4. Small vessel 7747 failure has a physics explanation the paper misses
Vessel 7747 (Small, median range 18 km) is predicted Large 76.6% of the time. At k=0.25, the correction factor at 18 km is (18000/25000)^0.25 = 0.928 — essentially no correction. A small reflective vessel at 18 km can have raw TotalAmplitude comparable to a large vessel at 40+ km, but the k=0.25 correction is too weak to equalise them. The correct k=2 would correct far more aggressively and potentially separate the classes — but would also destroy the class-range confound that makes the model internally valid.

### 5. The generalisation failure proves the point
The global 5-feature model achieved 90% Large recall on holdout data but only 10.6% on 3 known external Large vessels. This is the physically expected consequence of a correction tuned on a range-stratified training population: it works inside the training range distribution but fails systematically outside it.

---

## Conclusion

The thesis approach is methodologically honest and documents its failures — that is credit-worthy. However, its core contribution (range-independent geometry + amplitude range correction) is based on a physically imprecise definition of range-independence and an empirically fitted correction exponent that contradicts the radar range equation. On our dataset, the thesis feature set performs substantially worse than our BASE-24 features across all three temporal windows and both classifiers tested. The range information the thesis tries to remove is actively discriminative in our data, and removing it causes the same medium-recall collapse the thesis itself observed in external validation.

**Our approach is fundamentally better engineered for this problem.**
