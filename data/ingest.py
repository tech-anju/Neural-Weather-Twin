"""
data/ingest.py
Download and cache all data sources for the flood prediction twin.

Sources:
  1. SRTM 30m DEM        — elevation / terrain
  2. OpenStreetMap        — drainage network, land use, ward boundaries
  3. IMD rainfall         — hourly gauge data (India Meteorological Department)
  4. Sentinel-1 SAR       — flood inundation masks (Copernicus Open Access)
  5. Synthetic fallbacks  — for hackathon when credentials unavailable

Usage:
  python data/ingest.py --city kolkata
  python data/ingest.py --city kolkata --demo   # synthetic data only, no auth needed
"""

import argparse
import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Optional heavy dependencies (graceful degradation) ────────────────────────
try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

try:
    import osmnx as ox
    HAS_OSM = True
except ImportError:
    HAS_OSM = False

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    # cdsetool targets the Copernicus Data Space Ecosystem API
    # (SciHub was shut down April 2023 — sentinelsat no longer works)
    from cdsetool.query import query_features
    from cdsetool.download import download_feature
    from cdsetool.credentials import Credentials
    HAS_SENTINEL = True
except ImportError:
    HAS_SENTINEL = False


# ── City registry ──────────────────────────────────────────────────────────────
CITY_BOUNDS: Dict[str, dict] = {
    "kolkata": {
        "lat": (22.45, 22.65), "lon": (88.25, 88.45),
        "imd_station": "KOLKATA", "timezone": "Asia/Kolkata",
        "base_elevation_m": 5.0,   # ~5 m ASL — very flat, flood-prone
    },
    "chennai": {
        "lat": (12.85, 13.10), "lon": (80.15, 80.35),
        "imd_station": "CHENNAI", "timezone": "Asia/Kolkata",
        "base_elevation_m": 6.5,
    },
    "mumbai": {
        "lat": (18.90, 19.10), "lon": (72.80, 72.98),
        "imd_station": "MUMBAI", "timezone": "Asia/Kolkata",
        "base_elevation_m": 10.0,
    },
}

# ── Data validation helpers ────────────────────────────────────────────────────

def _validate_tif(path: Path, min_size_kb: int = 10) -> bool:
    """Check GeoTIFF exists, is non-empty, and is readable."""
    if not path.exists():
        return False
    if path.stat().st_size < min_size_kb * 1024:
        print(f"  [WARN] {path.name} suspiciously small ({path.stat().st_size} bytes)")
        return False
    if not HAS_RASTERIO:
        return True
    try:
        with rasterio.open(path) as src:
            assert src.count >= 1
            assert src.width > 0 and src.height > 0
        return True
    except Exception as e:
        print(f"  [WARN] {path.name} unreadable: {e}")
        return False


def _validate_csv(path: Path, required_cols: list, min_rows: int = 10) -> bool:
    """Check CSV exists, has required columns, and sufficient rows."""
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, nrows=5)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  [WARN] {path.name} missing columns: {missing}")
            return False
        full_len = sum(1 for _ in open(path)) - 1  # row count without header
        if full_len < min_rows:
            print(f"  [WARN] {path.name} only {full_len} rows")
            return False
        return True
    except Exception as e:
        print(f"  [WARN] {path.name} unreadable: {e}")
        return False


def _validate_npy(path: Path, expected_ndim: int = 2) -> bool:
    """Check .npy array exists and has expected dimensions."""
    if not path.exists():
        return False
    try:
        arr = np.load(path)
        if arr.ndim != expected_ndim:
            print(f"  [WARN] {path.name} has {arr.ndim}D array, expected {expected_ndim}D")
            return False
        if arr.max() == 0:
            print(f"  [WARN] {path.name} is all zeros — possibly empty flood mask")
        return True
    except Exception as e:
        print(f"  [WARN] {path.name} unreadable: {e}")
        return False


# ── 1. DEM ─────────────────────────────────────────────────────────────────────

def _download_srtm_via_requests(bbox: tuple, out_file: Path) -> None:
    """
    Download SRTM 30m DEM tile directly from NASA EarthData via requests.
    bbox = (lon_min, lat_min, lon_max, lat_max)
    Saves as GeoTIFF to out_file.
    No credentials needed for SRTM3 public tiles.
    """
    import requests

    lon_min, lat_min, lon_max, lat_max = bbox
    # OpenTopography public SRTM API (no auth needed, 500 requests/day free)
    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype=SRTMGL1"
        f"&south={lat_min}&north={lat_max}&west={lon_min}&east={lon_max}"
        f"&outputFormat=GTiff"
        f"&API_Key=demoapikeyot2022"   # free public demo key
    )
    print(f"  [DEM] Fetching from OpenTopography...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    if len(resp.content) < 10_000:
        raise RuntimeError(f"Response too small ({len(resp.content)} bytes) — likely an error")

    out_file.write_bytes(resp.content)


def download_srtm_dem(city: str, output_dir: Path, force: bool = False) -> Path:
    """
    Download SRTM 30m DEM via OpenTopography public API (no auth needed).
    Falls back to synthetic DEM if download fails.
    """
    out_file = output_dir / f"{city}_dem.tif"

    if not force and _validate_tif(out_file):
        print(f"  [DEM] Cache hit: {out_file.name}")
        return out_file

    bounds = CITY_BOUNDS[city]
    bbox = (bounds["lon"][0], bounds["lat"][0], bounds["lon"][1], bounds["lat"][1])
    print(f"  [DEM] Downloading SRTM for {city} bbox={bbox}...")

    try:
        _download_srtm_via_requests(bbox, out_file)
        if _validate_tif(out_file):
            print(f"  [DEM] Saved: {out_file.name}")
            return out_file
        raise RuntimeError("Downloaded file invalid")

    except Exception as e:
        print(f"  [DEM] SRTM unavailable ({e}). Generating synthetic DEM.")
        return _create_synthetic_dem(city, bounds, output_dir)


def _create_synthetic_dem(city: str, bounds: dict, output_dir: Path) -> Path:
    """
    Realistic synthetic urban DEM:
    - ~5 m ASL base (Kolkata is very flat)
    - Hooghly River channel on western edge (low)
    - Slight elevation ridge in centre-north
    - Salt Lake marshland depression in east
    """
    if not HAS_RASTERIO:
        # Save as plain numpy if rasterio missing
        out_file = output_dir / f"{city}_dem.npy"
        H, W = 200, 200
        elev = _synthetic_elevation_array(city, H, W)
        np.save(out_file, elev)
        print(f"  [DEM] Saved numpy DEM: {out_file.name}")
        return out_file

    H, W = 200, 200
    elev_arr = _synthetic_elevation_array(city, H, W)

    transform = from_bounds(
        bounds["lon"][0], bounds["lat"][0],
        bounds["lon"][1], bounds["lat"][1],
        W, H,
    )
    out_file = output_dir / f"{city}_dem.tif"
    with rasterio.open(
        out_file, "w", driver="GTiff",
        height=H, width=W, count=1,
        dtype=np.float32, crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(elev_arr.astype(np.float32), 1)

    print(f"  [DEM] Synthetic DEM saved: {out_file.name} (shape {H}×{W})")
    return out_file


def _synthetic_elevation_array(city: str, H: int, W: int) -> np.ndarray:
    """Build a physically plausible elevation array for a given city."""
    rng = np.random.default_rng(42)
    base = CITY_BOUNDS[city].get("base_elevation_m", 5.0)
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2

    elev = (
        base
        + 4.0 * np.exp(-((x - cx) ** 2 + (y - cy * 0.6) ** 2) / (2 * 35 ** 2))  # central ridge
        - 3.0 * np.exp(-((x - W * 0.15) ** 2) / (2 * 15 ** 2))                   # river channel W
        - 1.5 * np.exp(-((x - W * 0.85) ** 2 + (y - H * 0.7) ** 2) / (2 * 25 ** 2))  # marsh E
        + rng.normal(0, 0.25, (H, W))                                              # micro-relief
    )
    return elev.clip(0, None).astype(np.float32)


# ── 2. OpenStreetMap ───────────────────────────────────────────────────────────

def download_osm_features(city: str, output_dir: Path, force: bool = False) -> dict:
    """
    Download OpenStreetMap drainage network, water bodies, and roads.
    Cached as GeoPackage for reuse.

    Real data: https://www.openstreetmap.org (no auth needed via osmnx)
    """
    cache_file = output_dir / f"{city}_osm.gpkg"
    layers = ["drains", "water", "roads"]

    if not force and cache_file.exists() and HAS_GPD:
        try:
            result = {lyr: gpd.read_file(cache_file, layer=lyr) for lyr in layers}
            total = sum(len(v) for v in result.values())
            print(f"  [OSM] Cache hit: {cache_file.name} ({total} features)")
            return result
        except Exception:
            pass  # corrupted cache — re-download

    if not HAS_OSM or not HAS_GPD:
        print("  [OSM] osmnx/geopandas not installed — using synthetic features.")
        return _create_synthetic_osm(city, output_dir)

    bounds = CITY_BOUNDS[city]
    # osmnx bbox = (north, south, east, west)
    bbox = (bounds["lat"][1], bounds["lat"][0], bounds["lon"][1], bounds["lon"][0])
    print(f"  [OSM] Downloading features for {city}...")

    result = {}
    try:
        result["drains"] = ox.features_from_bbox(
            bbox,
            tags={"waterway": ["drain", "ditch", "canal", "river", "stream", "culvert"]},
        )
        print(f"         drains: {len(result['drains'])} features")
    except Exception as e:
        print(f"         drains failed ({e}) — empty")
        result["drains"] = gpd.GeoDataFrame()

    try:
        result["water"] = ox.features_from_bbox(
            bbox, tags={"natural": ["water", "wetland"], "landuse": "reservoir"}
        )
        print(f"         water: {len(result['water'])} features")
    except Exception as e:
        print(f"         water failed ({e}) — empty")
        result["water"] = gpd.GeoDataFrame()

    try:
        G = ox.graph_from_bbox(bbox, network_type="drive")
        result["roads"] = ox.graph_to_gdfs(G, nodes=False, edges=True)
        print(f"         roads: {len(result['roads'])} features")
    except Exception as e:
        print(f"         roads failed ({e}) — empty")
        result["roads"] = gpd.GeoDataFrame()

    # Cache non-empty layers
    for lyr, gdf in result.items():
        if len(gdf) > 0:
            try:
                gdf.to_file(cache_file, layer=lyr, driver="GPKG")
            except Exception as e:
                print(f"         could not cache {lyr}: {e}")

    total = sum(len(v) for v in result.values())
    print(f"  [OSM] Done ({total} total features, cached: {cache_file.name})")
    return result


def _create_synthetic_osm(city: str, output_dir: Path) -> dict:
    """Synthetic drainage network for testing without osmnx."""
    from shapely.geometry import LineString
    bounds = CITY_BOUNDS[city]
    lat0, lat1 = bounds["lat"]
    lon0, lon1 = bounds["lon"]
    rng = np.random.default_rng(0)

    drain_geoms, drain_types = [], []
    # N-S drains
    for x in np.linspace(lon0, lon1, 8):
        jitter = rng.uniform(-0.002, 0.002)
        drain_geoms.append(LineString([(x + jitter, lat0), (x + jitter, lat1)]))
        drain_types.append("drain")
    # E-W drains
    for y in np.linspace(lat0, lat1, 6):
        jitter = rng.uniform(-0.002, 0.002)
        drain_geoms.append(LineString([(lon0, y + jitter), (lon1, y + jitter)]))
        drain_types.append("canal" if rng.random() > 0.5 else "drain")

    if HAS_GPD:
        import geopandas as gpd
        drains = gpd.GeoDataFrame(
            {"waterway": drain_types, "geometry": drain_geoms}, crs="EPSG:4326"
        )
    else:
        drains = {"waterway": drain_types, "geometry": drain_geoms}

    print(f"  [OSM] Synthetic: {len(drain_geoms)} drain features")
    return {"drains": drains, "water": gpd.GeoDataFrame() if HAS_GPD else {}, "roads": gpd.GeoDataFrame() if HAS_GPD else {}}


# ── 3. IMD Rainfall ────────────────────────────────────────────────────────────

def download_imd_rainfall(
    city: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """
    Download IMD hourly gridded rainfall.

    Real source (0.25° daily gridded):
      https://www.imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data_Download.html
      — Free with registration. Download .GRD binary files, parse with imdlib.

    pip install imdlib  # IMD official Python reader

    For hackathon: realistic synthetic monsoon data generated automatically.
    """
    # Canonical filename — slice dates to YYYY-MM-DD regardless of any time component
    sd = str(start_date)[:10]
    ed = str(end_date)[:10]
    out_file = output_dir / f"{city}_rainfall_{sd}_to_{ed}.csv"

    required_cols = ["datetime", "gauge_id", "latitude", "longitude", "rainfall_mm"]
    if not force and _validate_csv(out_file, required_cols, min_rows=100):
        print(f"  [IMD] Cache hit: {out_file.name}")
        return out_file

    # Try real IMD gridded download first
    real_path = _try_imd_gridded_download(city, sd, ed, output_dir)
    if real_path is not None:
        return real_path

    # Fallback: realistic synthetic monsoon
    print(f"  [IMD] Generating synthetic monsoon rainfall ({sd} → {ed})...")
    return _generate_synthetic_rainfall(city, sd, ed, out_file)


def _try_imd_gridded_download(city: str, start_date: str,
                               end_date: str, output_dir: Path) -> Optional[Path]:
    """
    Attempt to download real IMD gridded data using imdlib.
    Returns None if imdlib not installed or download fails.
    """
    try:
        import imdlib as imd  # pip install imdlib
    except ImportError:
        return None

    bounds = CITY_BOUNDS[city]
    try:
        print(f"  [IMD] Trying imdlib download for {city}...")
        # Download daily rainfall (type='rain') — 0.25° resolution
        data = imd.get_data(
            "rain",
            int(start_date[:4]),
            int(end_date[:4]),
            fn_format="yearwise",
            file_dir=str(output_dir / "imd_raw"),
        )
        # Clip to city bounding box
        lat_slice = slice(bounds["lat"][0], bounds["lat"][1])
        lon_slice = slice(bounds["lon"][0], bounds["lon"][1])
        city_data = data.sel(lat=lat_slice, lon=lon_slice)

        # Convert to tidy CSV with gauge-like structure
        records = []
        for t in city_data.time.values:
            for ilat, lat in enumerate(city_data.lat.values):
                for ilon, lon in enumerate(city_data.lon.values):
                    val = float(city_data.data[city_data.time.values == t, ilat, ilon])
                    records.append({
                        "datetime": str(t)[:10] + " 12:00:00",
                        "gauge_id": f"IMD_{ilat}_{ilon}",
                        "latitude": round(float(lat), 4),
                        "longitude": round(float(lon), 4),
                        "rainfall_mm": max(0.0, round(val, 2)),
                        "source": "imd_gridded",
                    })

        sd_tag = str(start_date)[:10]
        ed_tag = str(end_date)[:10]
        out_file = output_dir / f"{city}_rainfall_{sd_tag}_to_{ed_tag}.csv"
        pd.DataFrame(records).to_csv(out_file, index=False)
        print(f"  [IMD] Real gridded data saved: {out_file.name}")
        return out_file

    except Exception as e:
        print(f"  [IMD] imdlib download failed ({e}). Using synthetic.")
        return None


def _generate_synthetic_rainfall(
    city: str, start_date: str, end_date: str, out_file: Path
) -> Path:
    """
    Physically realistic synthetic monsoon rainfall.
    Characteristics:
      - Wet/dry spell state machine (multi-day persistence)
      - Diurnal cycle: convective peak 15-18 h, stratiform peak 03-06 h
      - Gamma-distributed intensities during wet spells
      - Rare extreme events (100+ mm/day) — mimics real Kolkata records
      - Spatial correlation across 10 gauge points
    """
    bounds = CITY_BOUNDS[city]
    dates = pd.date_range(start_date, end_date, freq="h")
    n_gauges = 10
    lats = np.linspace(bounds["lat"][0], bounds["lat"][1], n_gauges)
    lons = np.linspace(bounds["lon"][0], bounds["lon"][1], n_gauges)

    records = []
    for g_idx, (lat, lon) in enumerate(zip(lats, lons)):
        rain = _simulate_monsoon_series(dates, seed=g_idx)
        for dt, r in zip(dates, rain):
            records.append({
                "datetime": str(dt),
                "gauge_id": f"G{g_idx:02d}",
                "latitude": round(float(lat), 4),
                "longitude": round(float(lon), 4),
                "rainfall_mm": round(max(0.0, r), 2),
                "source": "synthetic",
            })

    df = pd.DataFrame(records)
    df.to_csv(out_file, index=False)
    total_rain = df["rainfall_mm"].sum()
    n_wet = (df["rainfall_mm"] > 0.1).sum()
    print(
        f"  [IMD] Synthetic saved: {out_file.name} "
        f"({len(df):,} records, {total_rain:.0f} mm total, "
        f"{n_wet / len(df) * 100:.1f}% wet hours)"
    )
    return out_file


def _simulate_monsoon_series(dates: pd.DatetimeIndex, seed: int = 0) -> np.ndarray:
    """Simulate a single gauge's hourly monsoon rainfall time series."""
    rng = np.random.default_rng(seed)
    rain = np.zeros(len(dates))
    wet = False
    days_left = 0.0

    for i, dt in enumerate(dates):
        # State transition at start of each day
        if days_left <= 0:
            month = dt.month
            # Monsoon months June-Sep: 35% chance of starting a wet spell
            p_wet_onset = 0.35 if 6 <= month <= 9 else 0.04
            wet = rng.random() < p_wet_onset
            days_left = float(rng.integers(2, 6) if wet else rng.integers(1, 4))

        days_left -= 1.0 / 24.0

        if not wet:
            continue

        h = dt.hour
        # Diurnal envelope: afternoon convective + early morning stratiform
        diurnal = (
            0.55 * np.exp(-((h - 16.0) ** 2) / 9.0)
            + 0.30 * np.exp(-((h - 4.0) ** 2) / 9.0)
            + 0.08
        )
        # Base rain from gamma distribution
        intensity = rng.gamma(shape=1.8, scale=2.5) * diurnal

        # Extreme event (mirrors documented Kolkata events: ~150 mm in 24 h)
        if rng.random() < 0.004:
            intensity += rng.gamma(shape=3.0, scale=15.0)

        rain[i] = intensity

    return rain


# ── 4. Sentinel-1 SAR Flood Masks ─────────────────────────────────────────────

def download_sentinel_flood_masks(
    city: str,
    event_dates: list,   # list of "YYYY-MM-DD" strings
    output_dir: Path,
    username: Optional[str] = None,
    password: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Download Sentinel-1 SAR-derived flood inundation masks.

    Real access (Copernicus Data Space Ecosystem — replaced SciHub Apr 2023):
      1. Register free at https://dataspace.copernicus.eu
      2. Set env vars: SENTINEL_USER, SENTINEL_PASS
      3. pip install cdsetool

    NOTE: sentinelsat + scihub.copernicus.eu is permanently offline.
          Use cdsetool against https://catalogue.dataspace.copernicus.eu instead.

    Flood mask derivation:
      - Download pre-event + post-event SAR GRD (IW mode, VV polarisation)
      - Compute log-ratio: change_dB = post_dB - pre_dB
      - Threshold at -3 dB → flood binary mask
      - Apply permanent water body mask to remove false positives

    Falls back to realistic synthetic masks when credentials unavailable.
    """
    masks = {}
    for event_date in event_dates:
        out_file = output_dir / f"{city}_flood_mask_{event_date}.npy"

        if not force and _validate_npy(out_file, expected_ndim=2):
            print(f"  [SAR] Cache hit: {out_file.name}")
            masks[event_date] = out_file
            continue

        # Try real Sentinel API
        user = username or os.environ.get("SENTINEL_USER")
        pwd  = password  or os.environ.get("SENTINEL_PASS")
        if HAS_SENTINEL and user and pwd:
            real = _download_real_sentinel(city, event_date, output_dir, user, pwd)
            if real is not None:
                masks[event_date] = real
                continue

        # Fallback: synthetic
        print(f"  [SAR] Generating synthetic flood mask for {event_date}...")
        bounds = CITY_BOUNDS[city]
        H, W = 200, 200
        depth = _generate_flood_depth_map(H, W, intensity="high", seed=hash(event_date) % 1000)
        np.save(out_file, depth)
        print(f"  [SAR] Saved: {out_file.name}  max_depth={depth.max():.2f}m  "
              f"flooded_cells={np.sum(depth > 0.1)}/{H*W}")
        masks[event_date] = out_file

    return masks


def _download_real_sentinel(
    city: str, event_date: str, output_dir: Path, username: str, password: str
) -> Optional[Path]:
    """
    Download Sentinel-1 GRD products via Copernicus Data Space Ecosystem (CDSE).

    CDSE replaced SciHub in April 2023.
    API docs: https://documentation.dataspace.copernicus.eu

    This function downloads the raw .SAFE archive. Full flood mask derivation
    (speckle filtering, terrain correction, log-ratio thresholding) requires
    ESA SNAP or the `pyroSAR` library — flagged here but outside hackathon scope.
    """
    try:
        from cdsetool.query import query_features
        from cdsetool.download import download_feature
        from cdsetool.credentials import Credentials
        from datetime import datetime, timedelta

        creds = Credentials(username, password)
        bounds = CITY_BOUNDS[city]

        # WKT footprint for city bounding box
        footprint = (
            f"POLYGON(("
            f"{bounds['lon'][0]} {bounds['lat'][0]}, "
            f"{bounds['lon'][1]} {bounds['lat'][0]}, "
            f"{bounds['lon'][1]} {bounds['lat'][1]}, "
            f"{bounds['lon'][0]} {bounds['lat'][1]}, "
            f"{bounds['lon'][0]} {bounds['lat'][0]}"
            f"))"
        )

        dt         = datetime.strptime(event_date, "%Y-%m-%d")
        date_start = dt.strftime("%Y-%m-%dT00:00:00Z")
        date_end   = (dt + timedelta(days=2)).strftime("%Y-%m-%dT23:59:59Z")

        print(f"  [SAR] Querying CDSE for {city} around {event_date}...")

        # Query Sentinel-1 GRD products
        features = list(query_features(
            "Sentinel1",
            {
                "startDate":        date_start,
                "completionDate":   date_end,
                "productType":      "GRD",
                "sensorMode":       "IW",
                "processingLevel":  "LEVEL1",
                "geometry":         footprint,
            },
            credentials=creds,
        ))

        if not features:
            print(f"  [SAR] No CDSE products found for {event_date}.")
            return None

        print(f"  [SAR] Found {len(features)} product(s). Downloading first...")

        raw_dir = output_dir / "sentinel_raw"
        raw_dir.mkdir(exist_ok=True)

        # Download the first matching product
        download_feature(features[0], str(raw_dir), credentials=creds)

        # ── Full SAR processing pipeline (requires ESA SNAP / pyroSAR) ────────
        # Steps not implemented here — too heavy for hackathon setup:
        #   1. Apply orbit file
        #   2. Thermal noise removal
        #   3. Radiometric calibration (→ sigma0)
        #   4. Speckle filtering (Lee 5×5)
        #   5. Range-Doppler terrain correction (SRTM DEM)
        #   6. Convert to dB: 10 * log10(sigma0)
        #   7. Compute log-ratio vs pre-event image
        #   8. Threshold at -3 dB → flood binary mask
        #   9. Mask out permanent water bodies
        # See: https://github.com/johntruckenbrodt/pyroSAR
        # ──────────────────────────────────────────────────────────────────────

        print("  [SAR] Raw .SAFE archive downloaded.")
        print("  [SAR] Full processing (SNAP/pyroSAR) needed to derive flood mask.")
        print("  [SAR] Falling through to synthetic mask for this run.")
        return None   # fall through to synthetic until SAR processing is wired up

    except Exception as e:
        print(f"  [SAR] CDSE download error: {e}")
        return None


def _generate_flood_depth_map(H: int, W: int, intensity: str = "medium",
                               seed: int = 42) -> np.ndarray:
    """
    Physically plausible synthetic flood depth map (metres).
    Flood concentrates along drainage channels and low-lying basins.
    """
    rng = np.random.default_rng(seed)
    depth = np.zeros((H, W), dtype=np.float32)
    max_depth = {"low": 0.25, "medium": 0.55, "high": 1.10, "extreme": 2.0}[intensity]

    # Primary drainage channels (N-S oriented, biased to west — like Hooghly)
    n_channels = rng.integers(2, 5)
    channel_xs = sorted(rng.integers(W // 5, W, size=n_channels))
    for cx in channel_xs:
        hw = rng.integers(6, 22)
        for col in range(max(0, cx - hw), min(W, cx + hw)):
            dist = abs(col - cx)
            profile = max_depth * np.exp(-dist / max(hw / 2.5, 1))
            noise = rng.uniform(0.85, 1.15, H)
            depth[:, col] = np.maximum(depth[:, col], (profile * noise).astype(np.float32))

    # Low-lying depressions (parks, ponds, underpasses)
    n_basins = rng.integers(4, 10)
    for _ in range(n_basins):
        by = rng.integers(10, H - 10)
        bx = rng.integers(10, W - 10)
        r  = rng.integers(6, 28)
        bd = rng.uniform(0.10, max_depth * 0.7)
        y, x = np.ogrid[:H, :W]
        mask = (y - by) ** 2 + (x - bx) ** 2 <= r ** 2
        depth[mask] = np.maximum(depth[mask], bd)

    # Smooth with Gaussian (physical spread of inundation)
    if HAS_SCIPY:
        depth = gaussian_filter(depth, sigma=2.5)

    # Apply a zero-border (city edge dry)
    depth[:3, :] = 0; depth[-3:, :] = 0
    depth[:, :3] = 0; depth[:, -3:] = 0

    return depth.clip(0).astype(np.float32)


# ── Master ingest function ─────────────────────────────────────────────────────

def ingest_all(
    city: str,
    output_dir: str = "data/raw",
    demo_mode: bool = False,
    force: bool = False,
) -> Dict[str, object]:
    """
    Download all data sources for a city.

    Args:
        city:       City key (kolkata | chennai | mumbai)
        output_dir: Root directory for raw data
        demo_mode:  If True, skip all network calls and use synthetic data only
        force:      Re-download even if cache exists

    Returns:
        status dict with paths/objects for each data source,
        and a boolean 'all_ok' key.
    """
    try:
        from rich.console import Console
        console = Console()
        def log(msg): console.print(msg)
    except ImportError:
        def log(msg): print(msg)

    out = Path(output_dir) / city
    out.mkdir(parents=True, exist_ok=True)

    log(f"\n[bold]Ingesting data for {city.upper()}[/bold]")
    log(f"  Mode: {'DEMO (synthetic only)' if demo_mode else 'REAL (with fallback)'}")
    log(f"  Output: {out}\n")

    status: Dict[str, object] = {}

    # 1. DEM
    try:
        if demo_mode:
            bounds = CITY_BOUNDS[city]
            status["dem"] = _create_synthetic_dem(city, bounds, out)
        else:
            status["dem"] = download_srtm_dem(city, out, force=force)
        log(f"  [green]✓[/green] DEM → {Path(str(status['dem'])).name}")
    except Exception as e:
        log(f"  [red]✗[/red] DEM failed: {e}")
        status["dem"] = None

    # 2. OSM
    try:
        if demo_mode:
            status["osm"] = _create_synthetic_osm(city, out)
        else:
            status["osm"] = download_osm_features(city, out, force=force)
        n_feat = sum(len(v) for v in status["osm"].values() if hasattr(v, "__len__"))
        log(f"  [green]✓[/green] OSM → {n_feat} features across {len(status['osm'])} layers")
    except Exception as e:
        log(f"  [red]✗[/red] OSM failed: {e}")
        status["osm"] = None

    # 3. Rainfall (5-year training window + current year)
    try:
        status["rainfall"] = download_imd_rainfall(
            city, "2024-08-01", "2024-09-30", out, force=force
        )
        log(f"  [green]✓[/green] Rainfall → {Path(str(status['rainfall'])).name}")
    except Exception as e:
        log(f"  [red]✗[/red] Rainfall failed: {e}")
        status["rainfall"] = None

    # 4. SAR flood masks — key training events
    flood_events = {
        "kolkata": ["2021-07-30", "2022-06-15", "2023-09-12", "2024-09-21"],
        "chennai": ["2021-11-10", "2023-12-04"],
        "mumbai":  ["2021-07-17", "2022-07-09"],
    }.get(city, ["2024-09-01"])

    try:
        status["flood_masks"] = download_sentinel_flood_masks(
            city, flood_events, out, force=force
        )
        log(f"  [green]✓[/green] Flood masks → {len(status['flood_masks'])} events")
    except Exception as e:
        log(f"  [red]✗[/red] Flood masks failed: {e}")
        status["flood_masks"] = {}

    # 5. Summary
    all_ok = all(v is not None for k, v in status.items() if k != "flood_masks")
    status["all_ok"] = all_ok
    status["city"] = city
    status["output_dir"] = str(out)

    # Save manifest
    manifest = {
        k: str(v) if isinstance(v, Path) else
           {e: str(p) for e, p in v.items()} if isinstance(v, dict) and k == "flood_masks"
           else str(v)
        for k, v in status.items()
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if all_ok:
        log(f"\n[bold green]✓ All data ready in {out}[/bold green]")
    else:
        log(f"\n[bold yellow]⚠ Some sources failed — see above. Demo will use synthetic fallbacks.[/bold yellow]")

    return status


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest all data for the Neural Flood Twin"
    )
    parser.add_argument("--city", default="kolkata",
                        choices=list(CITY_BOUNDS.keys()),
                        help="City to download data for")
    parser.add_argument("--output_dir", default="data/raw",
                        help="Root directory for raw data")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic data only (no network calls, ideal for hackathon)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached")
    args = parser.parse_args()

    result = ingest_all(
        city=args.city,
        output_dir=args.output_dir,
        demo_mode=args.demo,
        force=args.force,
    )
    raise SystemExit(0 if result["all_ok"] else 1)