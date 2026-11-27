import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest

from stormtime import config as C, data as D, features as F, events as E, splits as S, baselines as B, metrics as M
from helpers import format_row, synthetic_text


def test_format_widths():
    w = C.fortran_widths()
    assert len(w) == 57 and sum(w) == 343


def test_parse_roundtrip_and_fill():
    row = format_row({"year": 2003, "doy": 302, "hour": 7, "speed": 800.0, "bz_gsm": -25.5,
                      "dst": -150, "pressure": 20.0, "density": 12.3})
    assert len(row) == 343
    df = D.parse_omni_text(row + "\n")
    assert df.loc[0, "year"] == 2003 and df.loc[0, "doy"] == 302 and df.loc[0, "hour"] == 7
    assert df.loc[0, "speed"] == 800.0 and df.loc[0, "bz_gsm"] == -25.5 and df.loc[0, "dst"] == -150
    assert np.isnan(df.loc[0, "temp"])        # default row uses the fill value
    assert np.isnan(df.loc[0, "ae"])


def test_availability_parser():
    txt = """
    1963-01-01 (001) - 2020-12-31 (366)   Dst Final indices
    2021-01-01 (001) - 2026-09-30 (273)   Provisional Dst indices
    2026-10-01 (274) - 2026-11-20 (324)   Quick Look Dst indices
    """
    a = D.parse_availability(txt)
    assert a["final"] == ("1963-01-01", "2020-12-31")
    assert a["provisional"][0] == "2021-01-01" and a["quicklook"][0] == "2026-10-01"
    t = pd.DatetimeIndex(["2019-05-01", "2024-05-11", "2026-10-02"])
    assert list(D.label_version(t, a)) == ["final", "provisional", "quicklook"]


def test_time_and_continuity():
    df = D.parse_omni_text(synthetic_text())
    df = D.add_time(df)
    v = D.validate_hourly(df)
    assert v["duplicates"] == 0 and v["monotonic"] and v["non_hourly_steps"] == 0
    assert v["years_with_unexpected_row_count"] == {}
    assert (df["t_avail"] - df.index == pd.Timedelta(hours=1)).all()


def test_vbs_matches_efield():
    df = D.add_time(D.parse_omni_text(synthetic_text(missing_frac=0.0)))
    f = F.add_physics(df)
    ok = f["vbs"].notna() & f["e_field"].notna()
    assert np.allclose(f.loc[ok, "vbs"], np.maximum(f.loc[ok, "e_field"], 0), atol=0.05)  # inputs are rounded to 0.1 nT, 1 km/s


def test_causal_impute_no_lookahead():
    s = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, 9.0, 10.0])
    a = F.causal_impute(pd.DataFrame({"x": s}), ["x"], max_ffill=2)
    assert list(a["x"].iloc[:3]) == [1.0, 1.0, 1.0] and a["x"].iloc[3:5].isna().all()
    s2 = s.copy(); s2.iloc[5:] = [-99.0, -99.0]      # change the FUTURE
    b = F.causal_impute(pd.DataFrame({"x": s2}), ["x"], max_ffill=2)
    assert a["x"].iloc[:5].equals(b["x"].iloc[:5]) or a["x"].iloc[:5].fillna(-1).equals(b["x"].iloc[:5].fillna(-1))
    assert list(a["x_age"]) == [0, 1, 2, 3, 4, 0, 0]


def test_find_events_merge_and_nan():
    dst = np.full(200, -5.0)
    dst[10:15] = -120; dst[20:22] = -110       # gap of 5 h -> merged
    dst[100:103] = -130; dst[130:132] = -105   # gap of 27 h -> separate
    dst[50] = np.nan
    ev = E.find_events(dst, -100, merge_gap_h=24)
    assert len(ev) == 3
    assert ev.loc[0, "start_idx"] == 10 and ev.loc[0, "end_idx"] == 21 and ev.loc[0, "min_dst"] == -120
    m = E.event_mask(200, ev, buffer_h=2)
    assert m[8] and m[23] and not m[50]


def test_dst_class_boundaries():
    x = np.array([-29.9, -30, -49.9, -50, -99.9, -100, -300])
    assert list(E.dst_class(x)) == [0, 1, 1, 2, 2, 3, 3]


def test_sample_index_never_crosses_splits():
    t = pd.date_range("1995-01-01", "2026-06-30 23:00", freq="h")
    si = S.make_sample_index(t)
    L, H = C.LOOKBACK_MAX, C.HORIZON_MAX
    sid = S.split_ids(t)
    for name, k in zip(C.SPLIT_ORDER, range(5)):
        idx = si.loc[si["split"] == name, "i"].to_numpy()
        assert len(idx) > 0
        assert (sid[idx - L + 1] == k).all() and (sid[idx + H] == k).all()
    # purged hours at a boundary: last train origin leaves H hours inside train
    last_train = si.loc[si["split"] == "train", "i"].max()
    assert last_train + H == np.flatnonzero(sid == 0).max()   # the last kept origin ends exactly at the split edge
    with pytest.raises(PermissionError):
        S.get_split(si, "test")
    assert len(S.get_split(si, "test", allow_test=True)) > 0


def test_exp_step_matches_numeric_ode():
    q, tau, d0 = -12.0, 7.0, -30.0
    d, n = d0, 100000
    dt = 1.0 / n
    for _ in range(n):                         # fine Euler as independent check
        d += dt * (q - d / tau)
    assert B.exp_step(d0, q, tau) == pytest.approx(d, abs=1e-3)


def test_fit_burton_recovers_parameters():
    rng = np.random.default_rng(1)
    n = 40000
    vbs = np.maximum(rng.gamma(1.2, 1.0, n) * (rng.random(n) < 0.6), 0)
    true = dict(d=5.0, ec=0.6, tau=9.0)
    dstar = np.zeros(n)
    for t in range(n - 1):
        dstar[t + 1] = B.exp_step(dstar[t], B.q_burton(vbs[t + 1], true["d"], true["ec"]), true["tau"])
    dstar += rng.normal(0, 0.5, n)
    fit = B.fit_burton(dstar, vbs, np.arange(n - 1))
    assert fit["d"] == pytest.approx(5.0, rel=0.1) and fit["ec"] == pytest.approx(0.6, abs=0.1)
    assert fit["tau"] == pytest.approx(9.0, rel=0.1)


def test_oracle_one_step_equals_exp_step_and_forecast_frozen():
    rng = np.random.default_rng(2)
    n = 500
    dst = rng.normal(-20, 15, n); p = np.full(n, 3.0); vbs = np.maximum(rng.normal(1, 1, n), 0)
    prm = dict(d=4.4, ec=0.49, tau=8.0)
    i = np.arange(10, 400)
    o = B.forecast_physics(dst, p, vbs, i, 1, "burton", prm, mode="oracle")
    exp = B.star_to_dst(B.exp_step(B.dst_to_star(dst[i], p[i]), B.q_burton(vbs[i + 1], 4.4, 0.49), 8.0), p[i + 1])
    assert np.allclose(o, exp)
    f3 = B.forecast_physics(dst, p, vbs, i, 3, "burton", prm, mode="forecast")
    # forecast mode must not depend on any future record
    vbs2 = vbs.copy(); vbs2[11:] += 50; dst2 = dst.copy(); dst2[11:] += 50
    f3b = B.forecast_physics(dst2, p, vbs2, i[:1], 3, "burton", prm, mode="forecast")
    assert f3b[0] == pytest.approx(f3[0])


def test_metrics_confusion_and_skill():
    y = np.array([-10., -60., -120., -40.]); p = np.array([-12., -55., -90., -45.])
    cm = M.confusion(y, p)
    assert cm.values.sum() == 4 and cm.loc["obs intense", "pred moderate"] == 1
    assert M.skill(y, y, p) == pytest.approx(1.0)


def test_ingest_local_and_all_years(tmp_path):
    # tiny hand-formatted fixture rows: unit test of file handling only, not a dataset
    rows = {y: [format_row({"year": y, "doy": d, "hour": h, "speed": 400.0, "dst": -10})
                for d in (1, 2) for h in range(24)] for y in (1995, 1996)}
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "omni2_1995.dat").write_text("\n".join(rows[1995]) + "\n")
    assert D.ingest_local_dat(raw) == [1995]
    assert (raw / "omni2_1995.dat.gz").exists()
    big = tmp_path / "omni2_all_years.dat"
    big.write_text("\n".join(rows[1995] + rows[1996]) + "\n")
    assert D.ingest_all_years_file(big, tmp_path / "out") == [1995, 1996]
    df = D.load_year(tmp_path / "out", 1996)
    assert len(df) == 48 and (df["speed"] == 400).all()
