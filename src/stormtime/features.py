"""Cleaning flags, physics features and causal imputation. Nothing here uses
future information, and nothing deletes a physically valid extreme."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# Soft bounds: used ONLY to flag values for inspection, never to delete them.
SOFT_BOUNDS = {"speed": (150.0, 2000.0), "density": (0.0, 150.0),
               "b_avg": (0.0, 100.0), "bz_gsm": (-100.0, 100.0), "pressure": (0.0, 100.0)}


def select_clean(df: pd.DataFrame, label_ver: pd.Series) -> pd.DataFrame:
    """Keep the working columns, attach label_version and soft-bound flags."""
    out = df[C.KEEP + ["t_avail"]].copy()
    out["label_version"] = label_ver.reindex(out.index).values
    for col, (lo, hi) in SOFT_BOUNDS.items():
        out[f"flag_{col}_range"] = ((out[col] < lo) | (out[col] > hi)).astype("int8")
    return out


def add_physics(df: pd.DataFrame, b: float = C.DST_STAR_B, c: float = C.DST_STAR_C) -> pd.DataFrame:
    """Derived physics columns. Units: V km/s, B nT, E/VBs mV/m, P nPa, Dst nT."""
    out = df.copy()
    v, bz, by = out["speed"], out["bz_gsm"], out["by_gsm"]
    out["vbs"] = v * np.maximum(-bz, 0.0) * 1e-3                       # = max(E_y, 0)
    bt = np.hypot(by, bz)
    clock = np.arctan2(by, bz)
    out["clock_angle"] = clock
    out["newell"] = (v ** (4.0 / 3.0)) * (bt ** (2.0 / 3.0)) * (np.abs(np.sin(clock / 2.0)) ** (8.0 / 3.0))
    out["sqrt_p"] = np.sqrt(out["pressure"])
    out["dst_star"] = out["dst"] - b * out["sqrt_p"] + c
    return out


def causal_impute(df: pd.DataFrame, cols, max_ffill: int) -> pd.DataFrame:
    """Forward-fill at most `max_ffill` hours (never backward), and add
    `<col>_miss` (1 if originally missing) and `<col>_age` (hours since last
    valid value, capped at 48). max_ffill=0 gives the mask-only policy."""
    out = df.copy()
    for c_ in cols:
        s = df[c_]
        valid = s.notna()
        grp = valid.cumsum()
        age = (~valid).astype(int).groupby(grp).cumsum().clip(upper=48)
        out[c_] = s.ffill(limit=max_ffill) if max_ffill > 0 else s
        out[c_ + "_miss"] = (~valid).astype("int8")
        out[c_ + "_age"] = age.astype("int16")
    return out


def impute_report(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Missingness per column and per label tier (for the Data tab)."""
    rows = []
    for c_ in cols:
        rows.append({"column": c_, "missing_frac": float(df[c_].isna().mean()),
                     "n_missing": int(df[c_].isna().sum())})
    return pd.DataFrame(rows).set_index("column")


def gap_lengths(s: pd.Series) -> np.ndarray:
    """Lengths (hours) of consecutive-NaN runs in a series."""
    m = s.isna().to_numpy()
    if not m.any():
        return np.array([], dtype=int)
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return ends - starts
