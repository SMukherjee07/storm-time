"""Test fixtures ONLY. Tiny hand-formatted OMNI-style rows used to unit-test the parser and
file handling. Nothing in the package, notebooks or results imports this module."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from stormtime import config as C


def format_row(values: dict) -> str:
    widths, decs = C.fortran_widths(), C.fortran_decimals()
    parts = []
    for (name, fill), w, d in zip(C.WORDS, widths, decs):
        v = values.get(name, fill if fill is not None else 0)
        parts.append(f"{int(round(v)):>{w}d}" if d is None else f"{v:{w}.{d}f}")
    return "".join(parts)


def synthetic_text(years=(1995, 1996), seed=0, missing_frac=0.02):
    rng = np.random.default_rng(seed)
    lines = []
    for y in years:
        nd = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        for doy in range(1, nd + 1):
            for h in range(24):
                v = {"year": y, "doy": doy, "hour": h,
                     "speed": float(np.clip(rng.normal(420, 80), 250, 900)),
                     "density": float(np.clip(rng.normal(6, 3), 0.5, 40)),
                     "bz_gsm": float(rng.normal(0, 3)), "by_gsm": float(rng.normal(0, 3)),
                     "b_avg": float(abs(rng.normal(6, 2))), "pressure": 2.0,
                     "dst": float(rng.normal(-10, 15))}
                v["e_field"] = -v["speed"] * v["bz_gsm"] * 1e-3
                if rng.random() < missing_frac:
                    v["speed"] = 9999.0; v["bz_gsm"] = 999.9
                lines.append(format_row(v))
    return "\n".join(lines) + "\n"
