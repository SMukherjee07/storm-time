"""Chronological splits with strict, window-aware purging, plus a test-set guard."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def split_ids(t_start: pd.DatetimeIndex) -> np.ndarray:
    """Integer split id per hour (index into SPLIT_ORDER), -1 outside every split."""
    sid = np.full(len(t_start), -1, dtype=np.int8)
    for k, name in enumerate(C.SPLIT_ORDER):
        s, e = C.SPLITS[name]
        sid[(t_start >= pd.Timestamp(s)) & (t_start <= pd.Timestamp(e))] = k
    return sid


def make_sample_index(t_start: pd.DatetimeIndex, L: int = C.LOOKBACK_MAX,
                      H: int = C.HORIZON_MAX) -> pd.DataFrame:
    """One row per candidate forecast origin i (the last COMPLETED hourly record).
    An origin is kept only if its whole span [i-L+1, i+H] lies inside ONE split
    (splits are contiguous, so equal ids at both ends imply no crossing).
    This purges L+H-1 hours at each boundary; inputs never straddle a split."""
    n = len(t_start)
    sid = split_ids(t_start)
    i = np.arange(n)
    lo, hi = np.clip(i - L + 1, 0, n - 1), np.clip(i + H, 0, n - 1)
    ok = (i - L + 1 >= 0) & (i + H < n) & (sid >= 0) & (sid[lo] == sid) & (sid[hi] == sid)
    out = pd.DataFrame({"i": i[ok], "t_origin": t_start[ok] + pd.Timedelta(hours=1),
                        "split": [C.SPLIT_ORDER[k] for k in sid[ok]]})
    return out.reset_index(drop=True)


def get_split(samples: pd.DataFrame, name: str, allow_test: bool = False) -> pd.DataFrame:
    """Return the origins of one split. Locked splits need allow_test=True,
    which is only ever passed in the final evaluation notebook."""
    if name in C.LOCKED_SPLITS and not allow_test:
        raise PermissionError(f"split '{name}' is locked until final evaluation "
                              f"(pass allow_test=True in the final notebook only)")
    return samples[samples["split"] == name]
