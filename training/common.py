"""Shared helpers for Explorer training scripts."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample as sk_resample

REAL_PATH  = "/home/iaxiom/projects/Explorer/data/study_cleaned.csv"
SYN_PATH   = "/home/iaxiom/projects/Research/Radar Datasets/radar_features_synthetic_1M.csv"
MODEL_DIR  = Path("/home/iaxiom/projects/Explorer/models")
MODEL_DIR.mkdir(exist_ok=True)

CLASSES    = ['large', 'medium', 'small']
CLASS_MAP  = {'large': 0, 'medium': 1, 'small': 2}

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
FULL_24  = EM_10 + KIN_14
EDA_18   = EDA_EM_14 + EDA_KIN_4
FULL_42  = FULL_24 + EDA_18
EM_24    = EM_10 + EDA_EM_14      # DualStream EM stream
KIN_18   = KIN_14 + EDA_KIN_4    # DualStream Kin stream
HQNN_8   = [
    'log_peak_rcs', 'log_total_rcs', 'SampleCount', 'footprint_m2',
    'aspect_ratio', 'size_beam_component', 'size_bow_stern_component', 'ellipse_area',
]


def load_real(cols: list, group_col: str = 'ObjID') -> tuple:
    df = pd.read_csv(REAL_PATH, low_memory=False)
    avail = [c for c in cols if c in df.columns]
    df = df[avail + [group_col, 'size_class']].copy()
    df[avail] = df[avail].fillna(df[avail].median())
    df = df[df['size_class'].isin(CLASSES)]
    y = np.array([CLASS_MAP[c] for c in df['size_class']])
    groups = df[group_col].values if group_col in df.columns else np.arange(len(df))
    return df[avail].values.astype(np.float32), y, groups, df


def load_synthetic(cols: list) -> tuple:
    df = pd.read_csv(SYN_PATH, low_memory=False)
    avail = [c for c in cols if c in df.columns]
    df = df[avail + ['size_class']].copy()
    df = df[df['size_class'].isin(CLASSES)]
    y = np.array([CLASS_MAP[c] for c in df['size_class']])
    return df[avail].values.astype(np.float32), y


def obj_split(X, y, groups, test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups=groups))
    return X[tr], X[te], y[tr], y[te]


def oversample(X_tr, y_tr, seed=42):
    max_n = max(np.bincount(y_tr))
    parts = [
        sk_resample(X_tr[y_tr == c], y_tr[y_tr == c],
                    replace=True, n_samples=max_n, random_state=seed)
        for c in np.unique(y_tr)
    ]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
