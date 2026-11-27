"""Storm-event detection under a fixed, documented rule."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def find_events(dst: np.ndarray, thr: float, merge_gap_h: int = C.EVENT_MERGE_GAP_H) -> pd.DataFrame:
    """Contiguous hours with Dst <= thr; episodes separated by fewer than
    `merge_gap_h` hours are merged. NaN hours are never 'below threshold'."""
    dst = np.asarray(dst, dtype=float)
    below = np.nan_to_num(dst, nan=np.inf) <= thr
    idx = np.flatnonzero(below)
    cols = ["start_idx", "end_idx", "min_dst", "min_idx", "hours_below", "span_h"]
    if len(idx) == 0:
        return pd.DataFrame(columns=cols)
    cut = np.flatnonzero(np.diff(idx) > merge_gap_h) + 1
    groups = np.split(idx, cut)
    rows = []
    for g in groups:
        s, e = int(g[0]), int(g[-1])
        seg = dst[s:e + 1]
        k = int(np.nanargmin(seg))
        rows.append((s, e, float(seg[k]), s + k, int(len(g)), e - s + 1))
    return pd.DataFrame(rows, columns=cols)


def event_mask(n: int, events: pd.DataFrame, buffer_h: int = C.EVENT_BUFFER_H) -> np.ndarray:
    """Boolean mask over n hourly indices covering each event +/- buffer_h."""
    m = np.zeros(n, dtype=bool)
    for s, e in zip(events["start_idx"], events["end_idx"]):
        m[max(0, int(s) - buffer_h): min(n, int(e) + buffer_h + 1)] = True
    return m


def dst_class(x: np.ndarray) -> np.ndarray:
    """0 quiet (> -30), 1 weak (-50,-30], 2 moderate (-100,-50], 3 intense (<= -100)."""
    x = np.asarray(x, dtype=float)
    a, b, c = C.CLASS_EDGES
    return (x <= a).astype(int) + (x <= b).astype(int) + (x <= c).astype(int)
