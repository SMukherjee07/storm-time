"""Metrics: pointwise, regime-wise, skill vs reference, classification."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .events import dst_class


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def pearson(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def skill(y, p, p_ref):
    """MSE skill score vs a reference forecast: 1 - MSE/MSE_ref (1 perfect, 0 = ref)."""
    return float(1.0 - np.mean((y - p) ** 2) / np.mean((y - p_ref) ** 2))


def regime_masks(y):
    """quiet: Dst > -30 ; storm: Dst <= -50 (moderate or stronger)."""
    return {"all": np.ones(len(y), bool), "quiet": y > -30.0, "storm": y <= -50.0}


def table(y, preds: dict, ref: str = "persistence") -> pd.DataFrame:
    """Metrics for several forecasters on a COMMON evaluation set."""
    rows = []
    masks = regime_masks(y)
    for name, p in preds.items():
        r = {"model": name, "n": len(y),
             "RMSE": rmse(y, p), "MAE": mae(y, p), "r": pearson(y, p),
             "skill_vs_" + ref: skill(y, p, preds[ref]) if ref in preds else np.nan}
        for reg in ("quiet", "storm"):
            m = masks[reg]
            r[f"RMSE_{reg}"] = rmse(y[m], p[m]) if m.sum() > 1 else np.nan
            r[f"n_{reg}"] = int(m.sum())
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


def confusion(y_true, y_pred) -> pd.DataFrame:
    """Rows = observed class, columns = predicted class (regression-derived)."""
    a, b = dst_class(y_true), dst_class(y_pred)
    m = np.zeros((4, 4), dtype=int)
    np.add.at(m, (a, b), 1)
    return pd.DataFrame(m, index=[f"obs {n}" for n in C.CLASS_NAMES],
                        columns=[f"pred {n}" for n in C.CLASS_NAMES])
