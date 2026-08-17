#!/usr/bin/env python3
"""
Explorer — Data Pipeline
========================
Loads real labeled radar data + 1M synthetic, imputes NaN via STATIONID
group statistics (median-fill → global-median fallback), engineers all
required features, and saves cleaned CSVs to ../data/.

Outputs
-------
  data/real_labeled_imputed.csv   — 123K rows, NaN imputed
  data/synthetic_1M_ref.txt       — path pointer (no copy, file is 2+ GB)
  data/pipeline_report.txt        — imputation counts + feature summary
"""

import os, time
from pathlib import Path

import numpy as np
import pandas as pd

REAL_PATH  = "/home/iaxiom/projects/Explorer/data/radarfeatureL_Study.csv"
SYN_PATH   = "/home/iaxiom/projects/Research/Radar Datasets/radar_features_synthetic_1M.csv"
OUT_DIR    = Path(__file__).parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# ── Feature definitions ───────────────────────────────────────────────────────
EM_10 = [
    'log_peak_rcs', 'log_total_rcs', 'rcs_conc',
    'aspect_ratio', 'footprint_m2',
    'SampleCount', 'size_bow_stern_component', 'size_beam_component',
    'ellipse_area', 'cr_dr_ratio_c',
]
KIN_14 = [
    'sog',
    'measured_sog_avg_900',  'measured_sog_avg_1800',
    'measured_sog_avg_3600', 'measured_sog_avg_10800',
    'measured_sog_std_900',  'measured_sog_std_1800',
    'measured_sog_std_3600', 'measured_sog_std_10800',
    'measured_cog_std_900',  'measured_cog_std_1800',
    'measured_cog_std_3600', 'measured_cog_std_10800',
    'displacement',
]
EDA_EM_14 = [
    'measured_TotalAmplitude_avg_900',  'measured_TotalAmplitude_avg_1800',
    'measured_TotalAmplitude_avg_3600', 'measured_TotalAmplitude_avg_10800',
    'measured_rangeStd_900',  'measured_rangeStd_1800',
    'measured_rangeStd_3600', 'measured_rangeStd_10800',
    'measured_azimuthStd_900',  'measured_azimuthStd_1800',
    'measured_azimuthStd_3600', 'measured_azimuthStd_10800',
    'rgw', 'azw',
]
EDA_KIN_4 = [
    'measured_cog_stdlog_900',  'measured_cog_stdlog_1800',
    'measured_cog_stdlog_3600', 'measured_cog_stdlog_10800',
]
FULL_24 = EM_10 + KIN_14
EDA_18  = EDA_EM_14 + EDA_KIN_4
FULL_42 = FULL_24 + EDA_18
HQNN_8  = [
    'log_peak_rcs', 'log_total_rcs', 'SampleCount', 'footprint_m2',
    'aspect_ratio', 'size_beam_component', 'size_bow_stern_component', 'ellipse_area',
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived columns. Safe to call on synthetic (columns already present)."""
    d = df.copy()

    # RCS log features
    if 'PeakAmplitude' in d.columns:
        d['log_peak_rcs']  = np.log1p(d['PeakAmplitude'].clip(lower=0))
        d['log_total_rcs'] = np.log1p(d['TotalAmplitude'].clip(lower=0))
        d['rcs_conc']      = d['PeakAmplitude'] / (d['TotalAmplitude'] + 1e-6)

    # Geometry
    if 'az_extent_m' in d.columns:
        az = d['az_extent_m']
    elif 'cross_range_extent' in d.columns:
        az = d['cross_range_extent'] * 75.0
    else:
        az = pd.Series(np.nan, index=d.index)

    dr = d.get('down_range_extent', pd.Series(np.nan, index=d.index))
    d['aspect_ratio'] = az / (dr + 1e-3)
    d['footprint_m2'] = np.log1p((az * dr).clip(lower=0))

    if 'ellipse_area' not in d.columns or d['ellipse_area'].isna().all():
        d['ellipse_area'] = np.pi * (az / 2) * (dr / 2) / 10_000

    # cr_dr_ratio alias
    if 'cr_dr_ratio' in d.columns:
        d['cr_dr_ratio_c'] = d['cr_dr_ratio'].clip(upper=15)
    elif 'cr_dr_ratio_c' not in d.columns:
        d['cr_dr_ratio_c'] = az / (dr + 1e-3)

    # sog
    if 'RSog' in d.columns and 'sog' not in d.columns:
        d['sog'] = d['RSog'].clip(lower=0)

    return d


def grouped_impute(df: pd.DataFrame, group_col: str, cols: list) -> pd.DataFrame:
    """Fill NaN in numeric cols using group median, then global median fallback."""
    report = {}
    for col in cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        n_before = df[col].isna().sum()
        if n_before == 0:
            continue
        grp_median = df.groupby(group_col)[col].transform('median')
        df[col] = df[col].fillna(grp_median)
        n_after_grp = df[col].isna().sum()
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)
        n_after_global = df[col].isna().sum()
        report[col] = {
            'before': int(n_before),
            'after_group_fill': int(n_after_grp),
            'after_global_fill': int(n_after_global),
        }
    return df, report


def normalise_size_class(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.replace({
        'large vessel': 'large', 'medium vessel': 'medium', 'small vessel': 'small',
    })


def load_real_labeled() -> pd.DataFrame:
    print(f"Loading real labeled: {REAL_PATH}")
    df = pd.read_csv(REAL_PATH, low_memory=False)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Identify group column for imputation
    group_col = None
    for c in ['STATIONID', 'StationID', 'RUserID', 'UserID', 'ObjID']:
        if c in df.columns and df[c].notna().sum() > 0:
            group_col = c
            break
    print(f"  Group column for imputation: {group_col}")

    # Report NaN before
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
    print(f"  NaN columns ({len(nan_cols)}):")
    for col, cnt in nan_cols.items():
        print(f"    {col}: {cnt:,}")

    # Feature engineering (before imputation so derived cols can also be imputed)
    df = engineer_features(df)

    # Impute using group
    impute_cols = list(nan_cols.index) + ['ellipse_area', 'cr_dr_ratio_c', 'aspect_ratio',
                                           'footprint_m2', 'log_peak_rcs', 'log_total_rcs',
                                           'rcs_conc', 'sog']
    impute_cols = [c for c in dict.fromkeys(impute_cols) if c in df.columns]
    if group_col:
        df, imp_report = grouped_impute(df, group_col, impute_cols)
    else:
        imp_report = {}
        print("  No group column — using global median fallback only")
        for col in impute_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

    # Normalise label
    if 'size_class' in df.columns:
        df['size_class'] = normalise_size_class(df['size_class'])
        valid = df['size_class'].isin(['large', 'medium', 'small'])
        df = df[valid].copy()
        print(f"  After label filter: {len(df):,} rows")
        print(f"  Class distribution: {df['size_class'].value_counts().to_dict()}")

    nan_remaining = df[impute_cols].isna().sum().sum()
    print(f"  NaN remaining after imputation: {nan_remaining}")

    return df, imp_report, group_col


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Explorer — Data Pipeline")
    print("=" * 60)

    # ── Real labeled ─────────────────────────────────────────────
    df_real, imp_report, group_col = load_real_labeled()
    out_real = OUT_DIR / "study_cleaned.csv"
    df_real.to_csv(out_real, index=False)
    print(f"\n  Saved → {out_real}  ({len(df_real):,} rows)")

    # ── Synthetic 1M ref ─────────────────────────────────────────
    syn_ref = OUT_DIR / "synthetic_1M_ref.txt"
    syn_ref.write_text(f"Synthetic 1M dataset path:\n{SYN_PATH}\n\nRows: 999,999\nLabel: size_class (large/medium/small)\nNaN: none\nNote: No MMSI/STATIONID — use directly.\n")
    print(f"  Saved → {syn_ref}")

    # ── Feature summary ──────────────────────────────────────────
    report_lines = [
        "Explorer Data Pipeline Report",
        "=" * 50,
        f"Real labeled: {len(df_real):,} rows",
        f"  Group col used for imputation: {group_col}",
        "",
        "Imputation summary:",
    ]
    for col, info in imp_report.items():
        report_lines.append(
            f"  {col}: {info['before']} NaN → {info['after_group_fill']} (after group) → {info['after_global_fill']} (after global)"
        )
    report_lines += [
        "",
        "Feature sets:",
        f"  EM-10      : {EM_10}",
        f"  KIN-14     : {KIN_14}",
        f"  EDA-EM-14  : {EDA_EM_14}",
        f"  EDA-KIN-4  : {EDA_KIN_4}",
        f"  FULL-24    : 24 features",
        f"  FULL-42    : 42 features",
        f"  HQNN-8     : {HQNN_8}",
        "",
        "Output files:",
        f"  {out_real}",
        f"  {syn_ref}",
        "",
        f"Elapsed: {time.time()-t0:.1f}s",
    ]
    report_path = OUT_DIR / "pipeline_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"  Saved → {report_path}")
    print(f"\n  Done in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
