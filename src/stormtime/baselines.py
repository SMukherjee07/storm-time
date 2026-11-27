"""Reference forecasters: persistence, climatology, and Burton-type physics.

Conventions (see PROTOCOL.md)
- Origin index i = last COMPLETED hourly record. Horizon h targets record i+h.
- Physics state is pressure-corrected Dst*. Q (nT/h, <= 0) and tau (h) are held
  constant over each 1-h step, so the update is exact:
      D*(t+dt) = D*(t) exp(-dt/tau) + Q tau (1 - exp(-dt/tau)).
- forecast mode: solar-wind inputs frozen at record i; Dst* -> Dst conversion
  uses the pressure of record i (persistence). Same information as any other model.
- oracle mode (DIAGNOSTIC ONLY): uses observed inputs for i+1..i+h and observed
  pressure at i+h. Never compare it with the forecast-mode results as a forecast.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import config as C


# ------------------------------------------------------------------ transforms
def dst_to_star(dst, p, b=C.DST_STAR_B, c=C.DST_STAR_C):
    return dst - b * np.sqrt(p) + c


def star_to_dst(dstar, p, b=C.DST_STAR_B, c=C.DST_STAR_C):
    return dstar + b * np.sqrt(p) - c


def exp_step(dstar, q, tau, dt: float = 1.0):
    """Exact solution of dD/dt = Q - D/tau for constant Q, tau over dt."""
    e = np.exp(-dt / tau)
    return dstar * e + q * tau * (1.0 - e)


# ------------------------------------------------------------ injection / decay
def q_burton(vbs, d, ec):
    return -d * np.maximum(vbs - ec, 0.0)


def q_om2000(vbs):
    return -C.OM2000_Q_COEF * np.maximum(vbs - C.OM2000_Q_EC, 0.0)


def tau_om2000(vbs):
    return C.OM2000_TAU_A * np.exp(C.OM2000_TAU_B / (C.OM2000_TAU_C + vbs))


def rollout(dstar0, vbs_seq, q_fn, tau_fn):
    """Integrate h steps. dstar0: (n,), vbs_seq: (n,h). Returns (n,h) D* at steps 1..h."""
    n, h = vbs_seq.shape
    out = np.empty((n, h))
    d = dstar0.copy()
    for k in range(h):
        v = vbs_seq[:, k]
        d = exp_step(d, q_fn(v), tau_fn(v))
        out[:, k] = d
    return out


# ------------------------------------------------------------------- fitting
def fit_burton(dstar, vbs, train_pairs, x0=(4.4, 0.49, 8.0)):
    """Least-squares fit of (d, Ec, tau) on one-step transitions t -> t+1 using
    the concurrent record's VBs (the data are time-shifted to magnetospheric
    arrival). `train_pairs`: indices t where both t and t+1 are inside TRAIN."""
    t = np.asarray(train_pairs)
    d0, d1, v1 = dstar[t], dstar[t + 1], vbs[t + 1]
    ok = np.isfinite(d0) & np.isfinite(d1) & np.isfinite(v1)
    d0, d1, v1 = d0[ok], d1[ok], v1[ok]

    def resid(th):
        d, ec, tau = th
        return exp_step(d0, q_burton(v1, d, ec), tau) - d1

    sol = least_squares(resid, x0, bounds=([0.05, 0.0, 1.0], [40.0, 3.0, 60.0]),
                        loss="soft_l1", f_scale=5.0)
    return {"d": float(sol.x[0]), "ec": float(sol.x[1]), "tau": float(sol.x[2]),
            "n_fit": int(ok.sum()), "rmse_onestep": float(np.sqrt(np.mean(resid(sol.x) ** 2)))}


# -------------------------------------------------------------- forecasters
def forecast_physics(dst, pressure, vbs, origins, h, method, params=None, mode="forecast"):
    """Dst forecast h hours ahead for each origin index. method: 'burton' | 'om2000'."""
    i = np.asarray(origins)
    dstar0 = dst_to_star(dst[i], pressure[i])
    if method == "burton":
        q_fn = lambda v: q_burton(v, params["d"], params["ec"])
        tau_fn = lambda v: np.full_like(v, params["tau"], dtype=float)
    elif method == "om2000":
        q_fn, tau_fn = q_om2000, tau_om2000
    else:
        raise ValueError(method)
    if mode == "forecast":
        seq = np.repeat(vbs[i][:, None], h, axis=1)
        p_conv = pressure[i]
    elif mode == "oracle":
        seq = vbs[i[:, None] + np.arange(1, h + 1)[None, :]]
        p_conv = pressure[i + h]
    else:
        raise ValueError(mode)
    return star_to_dst(rollout(dstar0, seq, q_fn, tau_fn)[:, -1], p_conv)


def forecast_persistence(dst, origins, h):
    return dst[np.asarray(origins)]


def forecast_climatology(dst_train_mean: float, n: int):
    return np.full(n, dst_train_mean)


def common_eval_mask(dst, pressure, vbs, origins, h):
    """True where EVERY forecaster (incl. oracle diagnostic) can be scored on the
    same sample: observed target, observed state/pressure at the origin and at the
    target, and finite VBs for records i..i+h."""
    i = np.asarray(origins)
    ok = np.isfinite(dst[i]) & np.isfinite(dst[i + h]) & np.isfinite(pressure[i]) & np.isfinite(pressure[i + h])
    for k in range(0, h + 1):
        ok &= np.isfinite(vbs[i + k])
    return ok
