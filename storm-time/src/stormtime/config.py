"""Single source of truth for constants. Anything marked VERIFY must be checked
against the cited paper/page before it is used in a reported result."""
from __future__ import annotations

import re

# ----------------------------------------------------------------- data source
OMNI_BASE = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"
OMNI_DOC_URL = "https://omniweb.gsfc.nasa.gov/html/ow_data.html"
FIRST_YEAR = 1995
LAST_YEAR = 2026  # partial year; 2026 is an optional extra check, never trained on

# FORTRAN format of one OMNI2 hourly record (57 words), from the OMNIWeb docs.
FORMAT = ("2I4,I3,I5,2I3,2I4,14F6.1,F9.0,F6.1,F6.0,2F6.1,F6.3,F6.2,F9.0,F6.1,F6.0,"
          "2F6.1,F6.3,2F7.2,F6.1,I3,I4,I6,I5,F10.2,5F9.2,I3,I4,2F6.1,2I6,F5.1,F9.6,F7.4")

# (column name, fill value or None) for each of the 57 words, in order.
WORDS = [
    ("year", None), ("doy", None), ("hour", None), ("bartels", 9999),
    ("imf_sc", 99), ("plasma_sc", 99), ("n_imf", 999), ("n_plasma", 999),
    ("b_avg", 999.9), ("b_vec", 999.9), ("b_lat_gse", 999.9), ("b_lon_gse", 999.9),
    ("bx_gse", 999.9), ("by_gse", 999.9), ("bz_gse", 999.9), ("by_gsm", 999.9),
    ("bz_gsm", 999.9), ("sig_b_avg", 999.9), ("sig_b", 999.9), ("sig_bx", 999.9),
    ("sig_by", 999.9), ("sig_bz", 999.9), ("temp", 9999999.0), ("density", 999.9),
    ("speed", 9999.0), ("flow_lon", 999.9), ("flow_lat", 999.9), ("na_np", 9.999),
    ("pressure", 99.99), ("sig_t", 9999999.0), ("sig_n", 999.9), ("sig_v", 9999.0),
    ("sig_phi", 999.9), ("sig_theta", 999.9), ("sig_ratio", 9.999),
    ("e_field", 999.99), ("beta", 999.99), ("mach_alfven", 999.9),
    ("kp10", 99), ("sunspot", 999), ("dst", 99999), ("ae", 9999),
    ("pf_gt1", 999999.99), ("pf_gt2", 99999.99), ("pf_gt4", 99999.99),
    ("pf_gt10", 99999.99), ("pf_gt30", 99999.99), ("pf_gt60", 99999.99),
    ("pf_flag", None), ("ap", 999), ("f107", 999.9), ("pcn", 999.9),
    ("al", 99999), ("au", 99999), ("mach_ms", 99.9), ("lyman_alpha", 0.999999),
    ("qi", 9.9999),
]
assert len(WORDS) == 57

_TOKEN = re.compile(r"(\d*)([IF])(\d+)(?:\.(\d+))?")


def fortran_widths(fmt: str = FORMAT) -> list[int]:
    """Expand a FORTRAN FORMAT string such as '2I4,I3,14F6.1' into field widths."""
    widths: list[int] = []
    for tok in fmt.split(","):
        m = _TOKEN.fullmatch(tok.strip())
        if not m:
            raise ValueError(f"cannot parse format token {tok!r}")
        rep = int(m.group(1)) if m.group(1) else 1
        widths.extend([int(m.group(3))] * rep)
    return widths


def fortran_decimals(fmt: str = FORMAT) -> list:
    """Decimals per field (None for integer fields)."""
    out: list = []
    for tok in fmt.split(","):
        m = _TOKEN.fullmatch(tok.strip())
        rep = int(m.group(1)) if m.group(1) else 1
        out.extend([int(m.group(4)) if m.group(4) is not None else None] * rep)
    return out


# Columns kept in the clean table (raw files keep all 57 words).
KEEP = [
    "imf_sc", "plasma_sc", "n_imf", "n_plasma",
    "b_avg", "bx_gse", "by_gsm", "bz_gsm", "temp", "density", "speed", "pressure",
    "e_field", "beta", "mach_alfven", "kp10", "ap", "ae", "al", "au", "f107",
    "sunspot", "dst",
]
# Solar-wind / IMF inputs that get causal imputation + masks.
INPUT_COLS = ["b_avg", "by_gsm", "bz_gsm", "density", "speed", "pressure", "temp"]

# --------------------------------------------------------------------- splits
# Hour-start timestamps, inclusive on both ends, UTC. Chronological and contiguous.
SPLITS = {
    "train": ("1995-01-01 00:00", "2015-12-31 23:00"),
    "val":   ("2016-01-01 00:00", "2018-12-31 23:00"),   # tuning, early stopping, stacker folds
    "calib": ("2019-01-01 00:00", "2020-12-31 23:00"),   # conformal calibration ONLY
    "test":  ("2021-01-01 00:00", "2025-12-31 23:00"),   # touched once, at the end
    "extra": ("2026-01-01 00:00", "2026-12-31 23:00"),   # optional partial-year check
}
SPLIT_ORDER = ["train", "val", "calib", "test", "extra"]
LOCKED_SPLITS = {"test", "extra"}  # require allow_test=True

LOOKBACK_MAX = 96   # h, longest input window any model may use
HORIZONS = (1, 3, 6, 12)
PRIMARY_HORIZON = 3
HORIZON_MAX = max(HORIZONS)

# Storm classes on Dst (nT), boundaries inclusive on the stronger side:
#   quiet: > -30 ; weak: (-50, -30] ; moderate: (-100, -50] ; intense: <= -100
CLASS_EDGES = (-30.0, -50.0, -100.0)
CLASS_NAMES = ["quiet", "weak", "moderate", "intense"]

# Event rule (documented in PROTOCOL.md): contiguous hours with Dst <= threshold,
# episodes merged if separated by fewer than EVENT_MERGE_GAP_H hours.
EVENT_MERGE_GAP_H = 24
EVENT_BUFFER_H = 72

# -------------------------------------------------------------------- physics
# Pressure correction Dst* = Dst - b*sqrt(Pdyn) + c.
# VERIFY against O'Brien & McPherron (2000), JGR 105(A4), 7707-7719.
DST_STAR_B = 7.26
DST_STAR_C = 11.0
# Published empirical injection/decay forms (nT/h and h; VBs in mV/m).
# VERIFY against the same paper before reporting.
OM2000_Q_COEF = 4.4
OM2000_Q_EC = 0.49
OM2000_TAU_A, OM2000_TAU_B, OM2000_TAU_C = 2.40, 9.74, 4.69
