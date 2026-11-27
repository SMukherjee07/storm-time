"""Acquisition, parsing and provenance for NASA OMNI2 hourly data."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------- availability
_AVAIL_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})\s*\(\d{3}\)\s*-\s*(\d{4}-\d{2}-\d{2})\s*\(\d{3}\)\s+"
    r"(Dst Final|Provisional Dst|Quick Look Dst|IMF|Plasma)", re.IGNORECASE)


def parse_availability(doc_text: str) -> dict:
    """Extract {tier: (start, end)} for Dst label tiers (and IMF/Plasma coverage)
    from the OMNIWeb documentation page, so label_version is never hard-coded."""
    out: dict = {}
    for start, end, name in _AVAIL_RE.findall(doc_text):
        key = {"dst final": "final", "provisional dst": "provisional",
               "quick look dst": "quicklook", "imf": "imf", "plasma": "plasma"}[name.lower()]
        out.setdefault(key, (start, end))
    return out


def label_version(t_start: pd.DatetimeIndex, avail: dict) -> pd.Series:
    """Label status of Dst for each hour: final / provisional / quicklook / unknown."""
    v = pd.Series("unknown", index=t_start, dtype="object")
    for tier in ("final", "provisional", "quicklook"):
        if tier in avail:
            s, e = avail[tier]
            m = (t_start >= pd.Timestamp(s)) & (t_start < pd.Timestamp(e) + pd.Timedelta(days=1))
            v[m] = tier
    return v.astype("category")


# ------------------------------------------------------------------- download
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def omni_urls(year: int) -> list[str]:
    return [f"{C.OMNI_BASE}omni2_{year}.dat", f"{C.OMNI_BASE}OMNI2_{year}.DAT"]


def download_year(year: int, raw_dir: Path, session=None, retries: int = 3,
                  timeout: int = 90, force: bool = False) -> dict:
    """Download one annual OMNI2 file, store gzip-compressed, return a manifest record."""
    import requests
    session = session or requests.Session()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    gz_path = raw_dir / f"omni2_{year}.dat.gz"
    if gz_path.exists() and not force:
        raw = gzip.decompress(gz_path.read_bytes())
        return _record(year, "cached", raw, gz_path, None, None)
    last_err = None
    for url in omni_urls(year):
        for attempt in range(retries):
            try:
                r = session.get(url, timeout=timeout)
                if r.status_code == 200 and len(r.content) > 1000:
                    gz_path.write_bytes(gzip.compress(r.content, compresslevel=9))
                    return _record(year, url, r.content, gz_path,
                                   r.headers.get("Last-Modified"), r.status_code)
                last_err = f"{url} -> HTTP {r.status_code}"
                break  # try next URL spelling
            except Exception as e:  # network hiccup: retry with backoff
                last_err = f"{url} -> {e!r}"
                time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download OMNI2 {year}: {last_err}")


def _record(year, url, raw: bytes, gz_path: Path, last_modified, status) -> dict:
    return {
        "year": year, "url": url, "http_status": status,
        "http_last_modified": last_modified,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes_raw": len(raw), "sha256_raw": sha256_bytes(raw),
        "n_lines": raw.count(b"\n"), "file": gz_path.name,
        "sha256_gz": sha256_bytes(gz_path.read_bytes()),
    }


def download_all(years, raw_dir: Path, force: bool = False, verbose: bool = True) -> list[dict]:
    """Download missing years; years already present (downloaded earlier or placed by hand)
    keep their original provenance from manifest.json when it exists."""
    import requests
    raw_dir = Path(raw_dir)
    prior = {}
    mpath = raw_dir / "manifest.json"
    if mpath.exists():
        prior = {r["year"]: r for r in json.loads(mpath.read_text()).get("files", [])}
    s = requests.Session()
    recs = []
    for y in years:
        rec = download_year(y, raw_dir, session=s, force=force)
        if rec["url"] == "cached":
            old = prior.get(y)
            if old and old.get("sha256_raw") == rec["sha256_raw"]:
                rec = old                                  # unchanged file: keep original retrieval record
            elif old:
                print(f"WARNING {y}: file changed since manifest was written; recorded as modified")
                rec["url"] = "cached (checksum differs from earlier manifest)"
        recs.append(rec)
        if verbose:
            print(f"{y}: {rec['n_lines']:>6} lines  {rec['bytes_raw']/1e6:5.2f} MB  {rec['sha256_raw'][:12]}  ({rec['url'][:40]})")
    return recs


def ingest_local_dat(raw_dir: Path, years=None) -> list[int]:
    """Option B: you downloaded omni2_YYYY.dat files by hand and put them in raw_dir.
    Compress them to omni2_YYYY.dat.gz (originals untouched). Provenance is recorded
    as 'manual file' with the file's modification time in the manifest."""
    raw_dir = Path(raw_dir)
    done = []
    for f in sorted(raw_dir.glob("omni2_*.dat")):
        y = int(re.search(r"omni2_(\d{4})\.dat$", f.name).group(1))
        if years is not None and y not in years:
            continue
        gz = raw_dir / f"{f.name}.gz"
        if not gz.exists():
            gz.write_bytes(gzip.compress(f.read_bytes(), compresslevel=9))
        done.append(y)
    return done


def ingest_all_years_file(path: Path, raw_dir: Path, years=None) -> list[int]:
    """Option B2: split NASA's single omni2_all_years.dat into annual gz files."""
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    by_year: dict[int, list[bytes]] = {}
    with open(path, "rb") as fh:
        for line in fh:
            if len(line.strip()) < 20:
                continue
            by_year.setdefault(int(line[:4]), []).append(line if line.endswith(b"\n") else line + b"\n")
    out = []
    for y, ls in sorted(by_year.items()):
        if years is not None and y not in years:
            continue
        (raw_dir / f"omni2_{y}.dat.gz").write_bytes(gzip.compress(b"".join(ls), compresslevel=9))
        out.append(y)
    return out


def fetch_doc_snapshot(raw_dir: Path) -> dict:
    """Save the OMNIWeb documentation page used to set label tiers (provenance)."""
    import requests
    raw_dir = Path(raw_dir)
    r = requests.get(C.OMNI_DOC_URL, timeout=60)
    r.raise_for_status()
    (raw_dir / "ow_data_snapshot.html").write_bytes(r.content)
    return {"url": C.OMNI_DOC_URL, "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": sha256_bytes(r.content), "availability": parse_availability(r.text)}


def write_manifest(records: list[dict], doc: dict, path: Path) -> None:
    Path(path).write_text(json.dumps({"snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                      "documentation": doc, "files": records}, indent=2))


# --------------------------------------------------------------------- parsing
def parse_omni_text(text: str) -> pd.DataFrame:
    """Parse OMNI2 fixed-width text into a DataFrame with all 57 named words.
    Per-word fill values are decoded to NaN. Rows are NOT dropped."""
    widths = C.fortran_widths()
    names = [n for n, _ in C.WORDS]
    df = pd.read_fwf(io.StringIO(text), widths=widths, names=names, header=None)
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64")
    for name, fill in C.WORDS:
        if fill is not None:
            df.loc[np.isclose(df[name].to_numpy(), fill, rtol=0, atol=1e-9), name] = np.nan
    return df


def load_year(raw_dir: Path, year: int) -> pd.DataFrame:
    raw = gzip.decompress((Path(raw_dir) / f"omni2_{year}.dat.gz").read_bytes()).decode("ascii", "replace")
    return parse_omni_text(raw)


def load_raw(raw_dir: Path, years) -> pd.DataFrame:
    return pd.concat([load_year(raw_dir, y) for y in years], ignore_index=True)


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    """UTC (naive) hour-START timestamp. OMNI hour label h = average over [h, h+1):
    the record is only usable at t_avail = t_start + 1 h."""
    y = df["year"].astype(int)
    t0 = pd.to_datetime(y.astype(str) + "-01-01")
    t = t0 + pd.to_timedelta(df["doy"].astype(int) - 1, unit="D") + pd.to_timedelta(df["hour"].astype(int), unit="h")
    out = df.copy()
    out.index = pd.DatetimeIndex(t, name="t_start")
    out["t_avail"] = out.index + pd.Timedelta(hours=1)
    return out


def validate_hourly(df: pd.DataFrame) -> dict:
    """Continuity checks on a time-indexed OMNI table."""
    idx = df.index
    d = np.diff(idx.values).astype("timedelta64[h]").astype(int)
    per_year = pd.Series(idx.year).value_counts().sort_index()
    exp = {int(y): (8784 if pd.Timestamp(year=int(y), month=12, day=31).dayofyear == 366 else 8760)
           for y in per_year.index}
    short = {int(y): int(n) for y, n in per_year.items() if n != exp[int(y)]}
    return {
        "n_rows": int(len(df)), "start": str(idx[0]), "end": str(idx[-1]),
        "duplicates": int(idx.duplicated().sum()),
        "monotonic": bool(idx.is_monotonic_increasing),
        "non_hourly_steps": int((d != 1).sum()),
        "years_with_unexpected_row_count": short,
    }
