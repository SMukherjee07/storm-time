# Data card: NASA OMNI2 hourly (1995-2026)

**Source:** NASA GSFC/SPDF OMNIWeb, `https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/` (annual `omni2_YYYY.dat`).
Dst, AE, AL, AU are from WDC for Geomagnetism, Kyoto (non-commercial use; cite DOI 10.17593/14515-74000).
Acknowledgement requested by SPDF: "GSFC/SPDF OMNIWeb".

**Snapshot:** see `data/raw/manifest.json` (retrieval time, HTTP Last-Modified, SHA-256 of every file) and
`data/raw/ow_data_snapshot.html` (documentation page used for label tiers).
OMNI is not a "hard" product and can be revised, so results refer to this snapshot.

**Timing:** hourly means time-shifted to the bow-shock nose; record labelled h covers [h, h+1) UTC.

**Label tiers:** final / provisional / quick-look Dst, stored per row as `label_version`.

**Fill values:** decoded per word to NaN (`config.WORDS`). Hours are never dropped: missing hours remain as rows.

**Cleaning decisions:** see `notebooks/02_clean_eda.ipynb` and `results/cleaning_report.json`
(rows affected by each rule, split by storm and quiet hours).

**Known limitations**
- Quick-look/provisional Dst can differ from final values; test years use provisional labels.
- Retrospective data: hourly averages are not available at the start of the hour (availability rule in PROTOCOL.md).
- Extreme storms are rare; event-level metrics and bootstrap intervals are used.

**Files**
| file | contents |
|---|---|
| `data/raw/omni2_YYYY.dat.gz` | untouched annual files (gzip) |
| `data/clean/omni_hourly_base.parquet` | parsed, flagged, no imputation |
| `data/clean/omni_hourly_clean.parquet` | + causal imputation (ffill <= 3 h), masks, physics features |
