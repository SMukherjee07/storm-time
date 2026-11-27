# Locked experimental protocol (v1, written before any model result)

Changes after this point must be logged in the "Amendments" section with a date and reason.

## 1. Question
Can a neural network constrained by a simple model of Earth's storm-time magnetic response
(Burton-type ring-current equation) predict storm strength more reliably than networks that
learn from historical patterns alone, especially for storms unlike those seen in training?

## 2. Data and availability rule
- Source: NASA OMNI2 hourly, annual files 1995-2026, snapshot with SHA-256 manifest.
- OMNI hour label h is the average over [h, h+1). A record is usable only at t_avail = t_start + 1 h.
- A forecast origin i is the last COMPLETED record. Horizon h targets the Dst record i+h.
- This is **retrospective forecasting under a stated availability assumption**, not a live warning system.
- Dst label tiers (final / provisional / quick-look) are read from the saved OMNIWeb documentation
  snapshot and stored per row as `label_version`.

## 3. Splits (chronological, UTC hour-start, inclusive)
| split | from | to | use |
|---|---|---|---|
| train | 1995-01-01 | 2015-12-31 | fit parameters |
| val | 2016-01-01 | 2018-12-31 | hyperparameters, early stopping, stacker out-of-fold folds |
| calib | 2019-01-01 | 2020-12-31 | conformal calibration only |
| test | 2021-01-01 | 2025-12-31 | final evaluation, touched once |
| extra | 2026-01-01 | latest | optional partial-year check |

Purge: an origin is kept only if its whole span [i-96+1, i+12] lies inside ONE split
(L = 96 h max lookback, H = 12 h max horizon). Code: `splits.make_sample_index`.
`splits.get_split(..., "test")` raises unless `allow_test=True` (final notebook only).

## 4. Targets and horizons
Primary endpoint: Dst(t+3 h). Secondary: 1, 6, 12 h.
Classes (regression-derived, nT): quiet > -30; weak (-50, -30]; moderate (-100, -50]; intense <= -100.

## 5. Inputs (identical information for every model)
Completed-hour history of: |B|, By_GSM, Bz_GSM, density, speed, pressure, temperature, Dst,
plus missing-value masks and time-since-valid. Derived: VBs, Newell coupling, sqrt(P), Dst*.
Causal imputation only (forward fill <= 3 h, never backward); the target is never imputed.
Values are flagged, never deleted, for being extreme.

## 6. Pressure and physics conventions
Dst* = Dst - b*sqrt(Pdyn) + c (constants VERIFY in `config.py`). Forecast mode converts back with the
last observed pressure (persistence). Oracle-input runs are labelled **diagnostic** everywhere.
Hybrid state update: D*(t+dt) = D* exp(-dt/tau) + Q tau (1 - exp(-dt/tau)).

## 7. Extreme-storm experiment (rule fixed before results)
Event = contiguous hours with Dst <= threshold, episodes merged if separated by < 24 h.
Held-out training events (threshold chosen from event counts in train; recorded in
`results/protocol_choices.json`): every training window whose input span or target overlaps
an event +/- 72 h is removed; the continuous time index is preserved.
"Stronger than anything in training" is claimed only if test-event minima actually exceed the training minima.

## 8. Evaluation
RMSE, MAE, r, MSE-skill vs persistence (per horizon), quiet vs storm errors, 4-class confusion matrix,
event detection (POD, FAR, CSI at -50 and -100 nT; predicted/observed episodes match if they overlap within +/-6 h),
event peak-magnitude and peak-timing error, parameter counts, training time.
Uncertainty: paired event bootstrap for event metrics, 7-day block bootstrap for hourly metrics,
observed conformal coverage reported overall and during storms (no nominal-coverage promise).

## 9. Model selection
Fixed configurations first; then a declared Optuna budget (about 30 trials per shortlisted model) on `val` only.
Seeds: 5 for finalists. Final refit on train+val, then one evaluation on test.
Stacker: ridge on rolling-origin out-of-fold predictions, preprocessing fitted inside each fold;
compared with the simple average and the best single model.

## Amendments
(none yet)
