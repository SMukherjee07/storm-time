"""Generates notebooks/01..03 (run once; notebooks are the deliverable)."""
import nbformat as nbf
from pathlib import Path

SETUP = '''import os, sys, json, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path.cwd()
while not (ROOT / "src" / "stormtime").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
assert (ROOT / "src" / "stormtime").exists(), "Run inside the cloned repo (see README)"
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from stormtime import config as C, data as D, features as F, events as E, splits as S, baselines as B, metrics as M
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
RAW, CLEAN, FIG, RES = ROOT/"data/raw", ROOT/"data/clean", ROOT/"figures", ROOT/"results"
for p in (RAW, CLEAN, FIG, RES): p.mkdir(parents=True, exist_ok=True)
print("repo:", ROOT)'''

def nb(cells, path):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(n, path)

# ------------------------------------------------------------------ 01
nb01 = [
("md", """# 01 · Acquire NASA OMNI2 hourly data
**Goal:** a verifiable raw snapshot: every annual file, its SHA-256, retrieval time, and the OMNIWeb documentation page that defines the Dst label tiers.

On Colab first run: `!git clone https://github.com/<YOU>/storm-time.git; %cd storm-time; !pip -q install pyarrow`"""),
("code", SETUP),
("code", '''YEARS = list(range(C.FIRST_YEAR, C.LAST_YEAR + 1))
doc = D.fetch_doc_snapshot(RAW)          # saves the OMNIWeb documentation page (provenance)
print(json.dumps(doc["availability"], indent=1))'''),
("md", "**Check the printed availability table against the live page** (https://omniweb.gsfc.nasa.gov/html/ow_data.html). It sets `label_version` for every row, so it must not be hard-coded."),
("md", """### Option B: if Colab cannot reach NASA (real files only)
Download `omni2_YYYY.dat` (or the single `omni2_all_years.dat`) from https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/ in your browser, upload to `data/raw/`, then run the cell below. Years already present are not downloaded again."""),
("code", '''# Option B only: uncomment what applies
# print(D.ingest_local_dat(RAW))                                   # omni2_YYYY.dat files placed in data/raw
# print(D.ingest_all_years_file(RAW / "omni2_all_years.dat", RAW, YEARS))   # one big file'''),
("code", '''records = D.download_all(YEARS, RAW)
D.write_manifest(records, doc, RAW / "manifest.json")
man = pd.DataFrame(records)[["year", "n_lines", "bytes_raw", "sha256_raw", "http_last_modified", "retrieved_utc"]]
man.tail(8)'''),
("code", '''raw = D.add_time(D.load_raw(RAW, YEARS))
val = D.validate_hourly(raw)
print(json.dumps(val, indent=1))
assert val["duplicates"] == 0 and val["monotonic"] and val["non_hourly_steps"] == 0, "time axis problem: paste this output back"
# loud parse sanity checks: a wrong column layout would break at least one of these
assert raw["hour"].between(0, 23).all() and raw["doy"].between(1, 366).all(), "year/doy/hour columns misparsed"
assert 300 < raw["speed"].median() < 600, f"speed median {raw['speed'].median()} looks wrong: column layout?"
assert -40 < raw["dst"].median() < 0, f"Dst median {raw['dst'].median()} looks wrong: column layout?"
assert abs(raw["bz_gsm"].median()) < 2, "Bz median should be near 0"
print("parse sanity checks passed")'''),
("md", "### Coverage and fill decoding\nFraction of each key word that is missing after decoding per-word fill values, and where each stream actually ends."),
("code", '''key = ["b_avg", "bz_gsm", "speed", "density", "pressure", "dst", "kp10", "ae"]
cov = pd.DataFrame({"missing_frac": raw[key].isna().mean().round(4),
                    "first_valid": [raw[k].first_valid_index() for k in key],
                    "last_valid": [raw[k].last_valid_index() for k in key]})
cov'''),
("code", '''# sanity: value ranges after fill decoding (a leaked fill value would show up here)
raw[["speed", "density", "bz_gsm", "pressure", "dst"]].describe().T[["min", "50%", "max"]]'''),
("code", '''json.dump({"validation": val, "coverage": cov.astype(str).to_dict()}, open(RES / "validation_01.json", "w"), indent=1)
print("saved", RES / "validation_01.json")'''),
("md", """### Commit
```
git add data/raw results/validation_01.json && git commit -m "OMNI raw snapshot + manifest" && git push
```
Send me the printed `validation` JSON and the `cov` table if anything looks off (row counts per year, non-hourly steps, a leaked fill value)."""),
]
nb(nb01, "notebooks/01_acquire_omni.ipynb")

# ------------------------------------------------------------------ 02
nb02 = [
("md", """# 02 · Clean, audit and engineer
**Rules (PROTOCOL.md §5):** decode fills only · flag, never delete extremes · causal imputation only · target never imputed.
**EDA discipline:** Dst-based plots use data up to 2020 only. The test period's labels are not inspected."""),
("code", SETUP),
("code", '''man = json.load(open(RAW / "manifest.json"))
avail = {k: tuple(v) for k, v in man["documentation"]["availability"].items()}
YEARS = [r["year"] for r in man["files"]]
raw = D.add_time(D.load_raw(RAW, YEARS))
last_obs = raw[["speed", "bz_gsm", "dst"]].dropna(how="all").index.max()   # OMNI pads the current year with fill rows
print("last hour with any observation:", last_obs, "| trimming", int((raw.index > last_obs).sum()), "trailing fill rows")
raw = raw.loc[:last_obs]
lv = D.label_version(raw.index, avail)
base = F.select_clean(raw, lv)
print(base.shape, "| label tiers:", base["label_version"].value_counts().to_dict())'''),
("md", "## 1 · Missingness (inputs only; no label values used)"),
("code", '''cols = C.INPUT_COLS + ["dst"]
yr = base.index.year
miss = pd.DataFrame({c: base[c].isna().groupby(yr).mean() for c in cols}).T
fig, ax = plt.subplots(figsize=(11, 3.2))
im = ax.imshow(miss.values, aspect="auto", cmap="magma_r", vmin=0, vmax=0.5)
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols); ax.set_xticks(range(0, len(miss.columns), 3))
ax.set_xticklabels(miss.columns[::3]); ax.grid(False); plt.colorbar(im, label="fraction missing")
ax.set_title("Missingness by variable and year"); plt.tight_layout(); plt.savefig(FIG / "02_missingness_heatmap.png"); plt.show()'''),
("code", '''fig, axs = plt.subplots(1, 3, figsize=(11, 3))
for ax, c in zip(axs, ["dst", "speed", "bz_gsm"]):
    g = F.gap_lengths(base[c])
    if len(g): ax.hist(g, bins=np.logspace(0, np.log10(max(g.max(), 10)), 30)); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{c}: gap lengths (h)\\n{len(g)} gaps, longest {g.max() if len(g) else 0} h")
plt.tight_layout(); plt.savefig(FIG / "02_gap_lengths.png"); plt.show()'''),
("md", "## 2 · Distributions and soft-bound flags (flag only: valid extremes are kept)"),
("code", '''dev = base.loc[:"2020-12-31"]
fig, axs = plt.subplots(1, 4, figsize=(13, 3))
for ax, c in zip(axs, ["speed", "density", "bz_gsm", "dst"]):
    ax.hist(dev[c].dropna(), bins=80); ax.set_title(c); ax.set_yscale("log")
for thr in C.CLASS_EDGES: axs[3].axvline(thr, color="r", lw=.8)
plt.tight_layout(); plt.savefig(FIG / "02_distributions.png"); plt.show()
flags = {c: int(base[c].sum()) for c in base.columns if c.startswith("flag_")}
print("flagged (NOT removed):", flags)'''),
("md", "## 3 · Consistency checks against OMNI's own derived fields"),
("code", '''f0 = F.add_physics(base)
ok = f0["vbs"].notna() & f0["e_field"].notna()
d = (f0.loc[ok, "vbs"] - np.maximum(f0.loc[ok, "e_field"], 0)).abs()
print(f"VBs vs max(E_field,0): median |diff| {d.median():.4f}, 99.9% {d.quantile(.999):.3f} mV/m (n={ok.sum()})")
p_re = 2.0e-6 * base["density"] * base["speed"] ** 2
okp = p_re.notna() & base["pressure"].notna()
rel = ((base.loc[okp, "pressure"] - p_re[okp]) / base.loc[okp, "pressure"].clip(lower=.1)).abs()
print(f"OMNI pressure vs 2e-6*N*V^2: median rel diff {rel.median():.3f} (alpha particles explain a few % excess)")
dev_dst = base.loc[:"2020-12-31"]
print("Spearman(Kp, -Dst) on <=2020:", round(dev_dst[["kp10", "dst"]].dropna().corr(method="spearman").iloc[0, 1] * -1, 3))'''),
("md", "## 4 · Imputation policies: what does each rule remove, and does it hit storms harder than quiet times?"),
("code", '''core = ["speed", "bz_gsm", "pressure"]
dst_dev = base["dst"]
storm = (dst_dev <= -50) & (base.index <= "2020-12-31")
quiet = (dst_dev > -30) & (base.index <= "2020-12-31")
rows = []
for name, k in [("mask only", 0), ("ffill<=3h", 3), ("ffill<=6h", 6)]:
    imp = F.causal_impute(base, core, k)
    complete = imp[core].notna().all(axis=1)
    rows.append({"policy": name, "complete_all_hours": complete.mean(), "complete_storm_hours": complete[storm].mean(),
                 "complete_quiet_hours": complete[quiet].mean(), "n_storm_hours": int(storm.sum())})
pol = pd.DataFrame(rows).set_index("policy").round(4); pol'''),
("md", "Default policy: **ffill ≤ 3 h** (a hypothesis, not an optimum; compared again as an ablation later). Storm vs quiet completeness above shows whether cleaning removes storm data disproportionately."),
("code", '''pol.to_csv(RES / "imputation_policies.csv")
imp = F.causal_impute(base, C.INPUT_COLS, 3)
clean = F.add_physics(imp)
n_valid_dstar = int(clean.loc[:"2020-12-31", "dst_star"].notna().sum())
print("clean:", clean.shape, "| hours with valid Dst* (<=2020):", n_valid_dstar)'''),
("code", '''def save_parquet(df, path):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "float64": out[c] = out[c].astype("float32")
    out["label_version"] = out["label_version"].astype(str)
    out.to_parquet(path, compression="zstd")
save_parquet(base, CLEAN / "omni_hourly_base.parquet")
save_parquet(clean, CLEAN / "omni_hourly_clean.parquet")
report = {"rows": int(len(clean)), "start": str(clean.index[0]), "end": str(clean.index[-1]),
          "label_tiers": clean["label_version"].astype(str).value_counts().to_dict(),
          "flags_not_removed": flags, "imputation_default": "ffill<=3h, causal, masks + age columns",
          "policies": pol.to_dict(), "columns": list(clean.columns)}
json.dump(report, open(RES / "cleaning_report.json", "w"), indent=1, default=str)
print({p.name: f"{p.stat().st_size/1e6:.1f} MB" for p in CLEAN.glob("*.parquet")})'''),
("md", "## 5 · Storm context (pre-2021 only)"),
("code", '''dev = clean.loc[:"2020-12-31"]
ev = E.find_events(dev["dst"].to_numpy(), -100)
top = ev.sort_values("min_dst").head(4)
fig, axs = plt.subplots(len(top), 1, figsize=(10, 2.3 * len(top)), sharex=False)
axs = np.atleast_1d(axs)
for ax, (_, r) in zip(axs, top.iterrows()):
    s, e = int(r.start_idx) - 48, int(r.end_idx) + 72
    seg = dev.iloc[max(s, 0):e]
    ax.plot(seg.index, seg["dst"], color="k", label="Dst"); ax.set_ylabel("Dst (nT)")
    a2 = ax.twinx(); a2.plot(seg.index, seg["vbs"], color="tab:orange", alpha=.7); a2.set_ylabel("VBs (mV/m)", color="tab:orange"); a2.grid(False)
    ax.set_title(f"min Dst {r.min_dst:.0f} nT at {dev.index[int(r.min_idx)]:%Y-%m-%d %H:%M}", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "02_top_storms_pre2021.png"); plt.show()'''),
("md", "## 6 · Spot-check against the official source\nPre-2021 storms only. Compare these minima with WDC Kyoto's own Dst listing (https://wdc.kugi.kyoto-u.ac.jp/dst_final/) for the same hours; any mismatch means a parsing or timing error."),
("code", '''windows = {"Halloween 2003": ("2003-10-28", "2003-11-02"), "Nov 2003": ("2003-11-19", "2003-11-23"),
           "St Patrick's Day 2015": ("2015-03-16", "2015-03-19"), "Sep 2017": ("2017-09-06", "2017-09-10")}
rows = []
for k, (a, b_) in windows.items():
    seg = clean.loc[a:b_, "dst"]
    rows.append({"storm": k, "min_Dst_nT": seg.min(), "at_UTC_hour_start": seg.idxmin(), "label_version": clean.loc[seg.idxmin(), "label_version"]})
pd.DataFrame(rows).set_index("storm")'''),
("md", """### Commit
```
git add data/clean results figures && git commit -m "Clean OMNI table + EDA" && git push
```"""),
]
nb(nb02, "notebooks/02_clean_eda.ipynb")

# ------------------------------------------------------------------ 03
nb03 = [
("md", """# 03 · Splits, storm events, and physics/persistence baselines
Test (2021–2025) and extra (2026) are **locked**: nothing below evaluates on them. All numbers are on `val` (2016–2018)."""),
("code", SETUP),
("code", '''df = pd.read_parquet(CLEAN / "omni_hourly_clean.parquet")
df = df.astype({c: "float64" for c in df.columns if str(df[c].dtype) == "float32"})
t = df.index
dst, P, vbs = df["dst"].to_numpy(), df["pressure"].to_numpy(), df["vbs"].to_numpy()
samples = S.make_sample_index(t)
rows = []
for name in C.SPLIT_ORDER:
    s, e = C.SPLITS[name]
    nh = int(((t >= pd.Timestamp(s)) & (t <= pd.Timestamp(e))).sum())
    rows.append({"split": name, "from": s[:10], "to": e[:10], "hours": nh, "origins_after_purge": int((samples.split == name).sum())})
pd.DataFrame(rows).set_index("split")'''),
("code", '''try:
    S.get_split(samples, "test")
except PermissionError as e:
    print("guard works ->", e)'''),
("md", "## 1 · Storm events by split (train/val/calib only)\nEvent rule: contiguous hours with Dst ≤ threshold, merged if separated by < 24 h."),
("code", '''dst_nan = dst.copy()
rows = []
for thr in (-50, -100, -150, -250):
    for name in ("train", "val", "calib"):
        s, e = C.SPLITS[name]; m = (t >= pd.Timestamp(s)) & (t <= pd.Timestamp(e))
        ev = E.find_events(np.where(m, dst_nan, np.nan), thr)
        rows.append({"threshold": thr, "split": name, "n_events": len(ev), "deepest_nT": ev.min_dst.min() if len(ev) else np.nan})
evt = pd.DataFrame(rows).pivot(index="threshold", columns="split", values=["n_events", "deepest_nT"]); evt'''),
("code", '''# Suggest an extreme-storm threshold from TRAIN counts only; override by hand if you disagree.
n_train = {thr: int(evt.loc[thr, ("n_events", "train")]) for thr in (-100, -150, -250)}
EXTREME_THRESHOLD = -150 if n_train[-150] >= 8 else -100
print("train event counts:", n_train, "-> EXTREME_THRESHOLD =", EXTREME_THRESHOLD)
json.dump({"extreme_threshold": EXTREME_THRESHOLD, "train_event_counts": n_train,
           "rule": "Dst<=thr, merge gap <24h, buffer +/-72h", "locked_before_model_results": True},
          open(RES / "protocol_choices.json", "w"), indent=1)'''),
("code", '''s, e = C.SPLITS["train"]; mtr = (t >= pd.Timestamp(s)) & (t <= pd.Timestamp(e))
ev_tr = E.find_events(np.where(mtr, dst, np.nan), EXTREME_THRESHOLD)
ev_tr["min_time"] = t[ev_tr.min_idx.astype(int)]
excl = E.event_mask(len(t), ev_tr, C.EVENT_BUFFER_H)
print(f"Held-out-event experiment would remove {int((excl & mtr).sum())} of {int(mtr.sum())} train hours ({(excl & mtr).sum()/mtr.sum():.1%}) from {len(ev_tr)} events")
ev_tr.sort_values("min_dst").head(10)[["min_time", "min_dst", "span_h"]]'''),
("md", "## 2 · Baselines on `val`\n**persistence** · **climatology** · **Burton-fit** (constants fitted on train, exact exponential update) · **O'Brien–McPherron 2000 published forms** (constants VERIFY in `config.py`). *Oracle* rows are diagnostics, not forecasts."),
("code", '''dstar = B.dst_to_star(dst, P)
tr_i = samples.loc[samples.split == "train", "i"].to_numpy()
pairs = tr_i[tr_i + 1 < len(t)]
fit = B.fit_burton(dstar, vbs, pairs)
print("Burton fit (train):", {k: round(v, 3) if isinstance(v, float) else v for k, v in fit.items()})
json.dump(fit, open(RES / "burton_fit.json", "w"), indent=1)
clim = float(np.nanmean(dst[mtr]))'''),
("code", '''va = samples.loc[samples.split == "val", "i"].to_numpy()
all_rows, preds_store = [], {}
for h in C.HORIZONS:
    ok = B.common_eval_mask(dst, P, vbs, va, h)
    o = va[ok]; y = dst[o + h]
    preds = {"persistence": B.forecast_persistence(dst, o, h),
             "climatology": B.forecast_climatology(clim, len(o)),
             "burton_fit": B.forecast_physics(dst, P, vbs, o, h, "burton", fit),
             "om2000_published": B.forecast_physics(dst, P, vbs, o, h, "om2000"),
             "burton_fit [oracle, diagnostic]": B.forecast_physics(dst, P, vbs, o, h, "burton", fit, mode="oracle"),
             "om2000 [oracle, diagnostic]": B.forecast_physics(dst, P, vbs, o, h, "om2000", mode="oracle")}
    tab = M.table(y, preds); tab.insert(0, "horizon_h", h); all_rows.append(tab.reset_index())
    preds_store[h] = (o, y, preds)
    print(f"h={h}: {len(o)} of {len(va)} val origins scored ({len(o)/len(va):.1%})")
res = pd.concat(all_rows).set_index(["horizon_h", "model"]).round(3)
res.to_csv(RES / "baselines_val.csv"); res'''),
("code", '''fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
for m_ in res.index.get_level_values("model").unique():
    if m_ == "climatology": continue
    sub = res.xs(m_, level="model")
    ls = "--" if "oracle" in m_ else "-"
    axs[0].plot(sub.index, sub.RMSE, ls, marker="o", label=m_)
    axs[1].plot(sub.index, sub["skill_vs_persistence"], ls, marker="o")
    axs[2].plot(sub.index, sub.RMSE_storm, ls, marker="o")
axs[0].set_title("RMSE (nT), val"); axs[1].set_title("MSE skill vs persistence"); axs[2].set_title("RMSE in storm hours (Dst<=-50)")
for a in axs: a.set_xlabel("horizon (h)"); a.set_xticks(C.HORIZONS)
axs[1].axhline(0, color="k", lw=.7); axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "03_baselines_val.png"); plt.show()'''),
("code", '''# largest validation event at h=3, plotted at TARGET time
h = C.PRIMARY_HORIZON
ev_va = E.find_events(np.where((t >= pd.Timestamp(C.SPLITS["val"][0])) & (t <= pd.Timestamp(C.SPLITS["val"][1])), dst, np.nan), -100)
r = ev_va.sort_values("min_dst").iloc[0]
lo, hi = int(r.start_idx) - 72, int(r.end_idx) + 96
o_all = np.arange(lo, hi - h)
ok = B.common_eval_mask(dst, P, vbs, o_all, h); o = o_all[ok]
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(t[o + h], dst[o + h], "k", lw=2, label="observed Dst")
ax.plot(t[o + h], B.forecast_persistence(dst, o, h), label="persistence", alpha=.8)
ax.plot(t[o + h], B.forecast_physics(dst, P, vbs, o, h, "burton", fit), label="Burton-fit (forecast)")
ax.plot(t[o + h], B.forecast_physics(dst, P, vbs, o, h, "om2000"), label="OM2000 (forecast)")
ax.plot(t[o + h], B.forecast_physics(dst, P, vbs, o, h, "burton", fit, mode="oracle"), "--", label="Burton-fit [oracle, diagnostic]")
ax.set_title(f"Largest val storm, {h}-h forecasts (min {r.min_dst:.0f} nT at {t[int(r.min_idx)]:%Y-%m-%d %H:%M})"); ax.set_ylabel("Dst (nT)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "03_val_storm_h3.png"); plt.show()'''),
("code", '''o, y, preds = preds_store[C.PRIMARY_HORIZON]
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, name in zip(axs, ["persistence", "burton_fit"]):
    cm = M.confusion(y, preds[name]); ax.imshow(cm.values, cmap="Blues"); ax.grid(False)
    ax.set_xticks(range(4)); ax.set_xticklabels(C.CLASS_NAMES, rotation=30); ax.set_yticks(range(4)); ax.set_yticklabels(C.CLASS_NAMES)
    for (i, j), v in np.ndenumerate(cm.values): ax.text(j, i, v, ha="center", va="center", color="w" if v > cm.values.max() / 2 else "k", fontsize=8)
    ax.set_title(f"{name}, h=3, val"); ax.set_xlabel("predicted class"); ax.set_ylabel("observed class")
plt.tight_layout(); plt.savefig(FIG / "03_confusion_h3_baselines.png"); plt.show()
M.confusion(y, preds["burton_fit"])'''),
("code", '''print("PRIMARY ENDPOINT (h=3, val): ")
print(res.xs(3, level="horizon_h")[["RMSE", "MAE", "r", "skill_vs_persistence", "RMSE_storm"]])'''),
("md", """### Commit
```
git add results figures && git commit -m "Splits, events, baselines" && git push
```
Paste back: the split table, the event table, `Burton fit (train)`, and the `PRIMARY ENDPOINT` table. Those fix the numbers every neural model must beat."""),
]
nb(nb03, "notebooks/03_splits_events_baselines.ipynb")
print("notebooks written")
