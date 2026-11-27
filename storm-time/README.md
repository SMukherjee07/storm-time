# Storm-Time: can physics help a neural network forecast geomagnetic storms?

Hybrid physics + neural forecasting of the Dst index (NASA OMNI2), built for the CSCI neural-networks class project.
Protocol: see [PROTOCOL.md](PROTOCOL.md). Data: see [DATA_CARD.md](DATA_CARD.md).

## Run (Colab)
```python
!git clone https://github.com/<YOUR_USERNAME>/storm-time.git
%cd storm-time
!pip -q install pyarrow
```
Then run notebooks in order: `01_acquire_omni` -> `02_clean_eda` -> `03_splits_events_baselines`.
Commit `data/raw/*.gz`, `data/raw/manifest.json`, `data/clean/*.parquet`, `results/`, `figures/` after each notebook.

## Layout
`src/stormtime/` library (data, features, events, splits, baselines, metrics) · `notebooks/` · `tests/` (`pytest -q`) ·
`data/` · `results/` · `figures/`

## Real data only
Everything in `data/` comes from NASA OMNI / WDC Kyoto. If Colab cannot reach NASA, use Option B in notebook 01
(download the files in your browser from https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/ and upload them).
`tests/` contains tiny hand-formatted fixtures that exist only to unit-test the parser and file handling; no notebook or
result uses them.
