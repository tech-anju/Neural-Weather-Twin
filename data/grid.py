"""
data/grid.py
Build the city spatial grid for the Neural Weather Twin.

Rasterizes a city's bounding box into 50m × 50m cells and computes
per-cell terrain and drainage features used as model inputs.

Features computed per cell:
  1. elevation_m          — terrain height from SRTM DEM
  2. slope_deg            — terrain slope (finite difference on DEM)
  3. flow_accumulation    — upstream drainage area (pysheds D8)
  4. drain_density        — OSM drain/canal line length per cell (m/m²)
  5. impervious_fraction  — fraction of cell that is paved/built (OSM roads + buildings)
  6. manning_n            — Manning's roughness coefficient (land-use based)

Output:
  data/processed/<city>/grid_features.npy   — float32 [H, W, 6]
  data/processed/<city>/grid_meta.json      — cell size, bounds, CRS, shape
  data/processed/<city>/grid_preview.png    — 6-panel feature map for inspection

Usage:
  python data/grid.py --city kolkata
  python data/grid.py --city kolkata --resolution 50 --output_dir data/processed
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Optional dependencies (graceful degradation) ──────────────────────────────
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.features import rasterize as rio_rasterize
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from pysheds.grid import Grid as ShedsGrid
    HAS_PYSHEDS = True
except ImportError:
    HAS_PYSHEDS = False

try:
    import geopandas as gpd
    from shapely.geometry import box, LineString, Polygon
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

try:
    from scipy.ndimage import gaussian_filter, uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── City registry ──────────────────────────────────────────────────────────────
CITY_BOUNDS: Dict[str, dict] = {
    "kolkata": {
        "lat": (22.45, 22.65), "lon": (88.25, 88.45),
        "base_elev_m": 5.0,
        "utm_epsg": 32645,           # UTM zone 45N — for metric distance ops
    },
    "chennai": {
        "lat": (12.85, 13.10), "lon": (80.15, 80.35),
        "base_elev_m": 6.5,
        "utm_epsg": 32644,
    },
    "mumbai": {
        "lat": (18.90, 19.10), "lon": (72.80, 72.98),
        "base_elev_m": 10.0,
        "utm_epsg": 32643,
    },
}

# Manning's n lookup by OSM land-use / surface type
# Reference: Chow (1959) Open-Channel Hydraulics
MANNING_N: Dict[str, float] = {
    "residential":   0.040,
    "commercial":    0.035,
    "industrial":    0.030,
    "road_asphalt":  0.013,
    "road_concrete": 0.012,
    "road_unpaved":  0.050,
    "park":          0.050,
    "water":         0.030,
    "wetland":       0.100,
    "farmland":      0.060,
    "forest":        0.120,
    "bare_ground":   0.025,
    "default":       0.035,   # urban default
}


# ── Grid dimensions ───────────────────────────────────────────────────────────

def compute_grid_shape(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution_m: int = 50,
) -> Tuple[int, int, float, float]:
    """
    Compute grid dimensions (H rows, W cols) for a given bounding box
    and target cell size in metres.

    Uses the equirectangular approximation — accurate enough for
    city-scale bounding boxes (< 50 km).

    Returns:
        H, W, dy_deg, dx_deg
    """
    lat_mid = (lat_min + lat_max) / 2.0
    # metres per degree at this latitude
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_mid))

    dy_deg = resolution_m / m_per_deg_lat
    dx_deg = resolution_m / m_per_deg_lon

    H = max(1, int(np.ceil((lat_max - lat_min) / dy_deg)))
    W = max(1, int(np.ceil((lon_max - lon_min) / dx_deg)))
    return H, W, dy_deg, dx_deg


# ── Feature 1 & 2: Elevation + Slope ─────────────────────────────────────────

def compute_elevation_and_slope(
    dem_path: Optional[Path],
    city: str,
    H: int, W: int,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution_m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DEM and compute per-cell elevation (m) and slope (degrees).

    Slope uses a 3×3 finite-difference Horn (1981) algorithm — the same
    method used by GDAL and ArcGIS. Works on the raw elevation array so
    no extra library is needed.

    Returns:
        elevation: float32 [H, W]  — metres above sea level
        slope:     float32 [H, W]  — degrees (0° = flat, 90° = vertical)
    """
    base_elev = CITY_BOUNDS[city]["base_elev_m"]

    # ── Try loading real DEM ──────────────────────────────────────────────────
    if dem_path is not None and dem_path.exists() and HAS_RASTERIO:
        try:
            with rasterio.open(dem_path) as src:
                # Reproject/resample to target grid if needed
                from rasterio.enums import Resampling
                from rasterio.warp import reproject, calculate_default_transform

                transform_out = from_bounds(lon_min, lat_min, lon_max, lat_max, W, H)
                elev_out = np.zeros((H, W), dtype=np.float32)

                reproject(
                    source=rasterio.band(src, 1),
                    destination=elev_out,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform_out,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.bilinear,
                )

            elevation = np.where(elev_out == src.nodata if hasattr(src, 'nodata') and src.nodata else elev_out == -9999,
                                 base_elev, elev_out).astype(np.float32)
            print(f"  [Grid/Elev] Loaded from DEM: min={elevation.min():.1f}m max={elevation.max():.1f}m")

        except Exception as e:
            print(f"  [Grid/Elev] DEM load failed ({e}). Using synthetic elevation.")
            elevation = _synthetic_elevation(city, H, W, base_elev)
    else:
        print("  [Grid/Elev] No DEM found. Using synthetic elevation.")
        elevation = _synthetic_elevation(city, H, W, base_elev)

    # ── Horn (1981) slope algorithm ───────────────────────────────────────────
    slope = _horn_slope(elevation, resolution_m)
    print(f"  [Grid/Slope] mean={slope.mean():.2f}° max={slope.max():.2f}°")
    return elevation, slope


def _synthetic_elevation(city: str, H: int, W: int, base_elev: float) -> np.ndarray:
    """
    Physically plausible synthetic DEM.
    Kolkata-like: ~5 m ASL, flat, Hooghly river channel on western edge,
    Salt Lake depression in east, slight ridge in north.
    """
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2

    elev = (
        base_elev
        + 4.0 * np.exp(-((x - cx) ** 2 + (y - cy * 0.6) ** 2) / (2 * (H * 0.18) ** 2))
        - 3.0 * np.exp(-((x - W * 0.12) ** 2) / (2 * (W * 0.07) ** 2))   # river W
        - 1.5 * np.exp(-((x - W * 0.82) ** 2 + (y - H * 0.72) ** 2) / (2 * (H * 0.12) ** 2))  # marsh E
        + rng.normal(0, 0.20, (H, W))
    ).clip(0).astype(np.float32)
    return elev


def _horn_slope(elevation: np.ndarray, cell_size_m: float) -> np.ndarray:
    """
    Horn (1981) slope algorithm: 3×3 weighted finite difference.
    dz/dx = [(c3+2c6+c9) - (c1+2c4+c7)] / (8 * cellsize)
    dz/dy = [(c7+2c8+c9) - (c1+2c2+c3)] / (8 * cellsize)
    slope  = arctan(sqrt((dz/dx)² + (dz/dy)²))
    """
    # Pad with edge replication so output has same shape as input
    e = np.pad(elevation, 1, mode="edge")

    dzdx = (
        (e[:-2, 2:] + 2 * e[1:-1, 2:] + e[2:, 2:]) -
        (e[:-2, :-2] + 2 * e[1:-1, :-2] + e[2:, :-2])
    ) / (8.0 * cell_size_m)

    dzdy = (
        (e[2:, :-2] + 2 * e[2:, 1:-1] + e[2:, 2:]) -
        (e[:-2, :-2] + 2 * e[:-2, 1:-1] + e[:-2, 2:])
    ) / (8.0 * cell_size_m)

    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    return np.degrees(slope_rad).astype(np.float32)


# ── Feature 3: Flow Accumulation ──────────────────────────────────────────────

def compute_flow_accumulation(
    dem_path: Optional[Path],
    elevation: np.ndarray,
    city: str,
    H: int, W: int,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> np.ndarray:
    """
    Compute D8 flow accumulation (number of upstream cells) using pysheds.
    Falls back to a fast numpy approximation when pysheds unavailable.

    High accumulation values mark natural drainage paths — cells where
    flood water concentrates.

    Returns:
        flow_acc: float32 [H, W] — normalised to [0, 1]
    """
    if HAS_PYSHEDS and dem_path is not None and dem_path.exists():
        try:
            grid = ShedsGrid.from_raster(str(dem_path))
            dem_data = grid.read_raster(str(dem_path))

            # Standard hydrological conditioning
            pit_filled   = grid.fill_pits(dem_data)
            depression_filled = grid.fill_depressions(pit_filled)
            flat_resolved = grid.resolve_flats(depression_filled)

            # D8 flow direction + accumulation
            fdir = grid.flowdir(flat_resolved)
            acc  = grid.accumulation(fdir)

            # Resample to target grid size
            from scipy.ndimage import zoom
            acc_arr = np.array(acc).astype(np.float64)
            zoom_y  = H / acc_arr.shape[0]
            zoom_x  = W / acc_arr.shape[1]
            acc_resized = zoom(acc_arr, (zoom_y, zoom_x), order=1)

            # Log-normalise (accumulation spans many orders of magnitude)
            acc_log = np.log1p(acc_resized)
            acc_norm = (acc_log / (acc_log.max() + 1e-8)).astype(np.float32)
            print(f"  [Grid/FlowAcc] pysheds D8 computed, max_acc={int(acc_resized.max())}")
            return acc_norm

        except Exception as e:
            print(f"  [Grid/FlowAcc] pysheds failed ({e}). Using numpy approximation.")

    # ── Fast numpy approximation ──────────────────────────────────────────────
    # Flow accumulation proxy: smooth inverse elevation
    # (low areas receive more upstream flow than high areas).
    # Not hydrologically precise but sufficient for model features.
    print("  [Grid/FlowAcc] Using numpy approximation (install pysheds for D8).")
    elev_inv  = elevation.max() - elevation              # invert: low = high value
    sigma     = max(H, W) * 0.04                         # ~4% of grid width
    if HAS_SCIPY:
        smoothed = gaussian_filter(elev_inv.astype(np.float64), sigma=sigma)
    else:
        # Manual box blur as fallback
        k = max(3, int(sigma * 2) | 1)
        pad = k // 2
        padded = np.pad(elev_inv, pad, mode="edge")
        smoothed = np.zeros_like(elev_inv, dtype=np.float64)
        for di in range(k):
            for dj in range(k):
                smoothed += padded[di:di+H, dj:dj+W]
        smoothed /= k * k

    norm = (smoothed / (smoothed.max() + 1e-8)).astype(np.float32)
    return norm


# ── Feature 4: Drain Density ──────────────────────────────────────────────────

def compute_drain_density(
    osm_data: Optional[dict],
    H: int, W: int,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    resolution_m: int,
) -> np.ndarray:
    """
    Compute drain density per cell: total drain/canal line length (metres)
    divided by cell area (m²), giving units of m⁻¹.

    Higher values indicate well-drained cells; lower values indicate cells
    where water is more likely to pond.

    Returns:
        drain_density: float32 [H, W] — normalised to [0, 1]
    """
    density = np.zeros((H, W), dtype=np.float32)

    if not HAS_GPD or osm_data is None:
        return _synthetic_drain_density(H, W)

    drains_gdf = osm_data.get("drains")
    if drains_gdf is None or (hasattr(drains_gdf, "__len__") and len(drains_gdf) == 0):
        return _synthetic_drain_density(H, W)

    try:
        # Cell size in degrees
        dy = (lat_max - lat_min) / H
        dx = (lon_max - lon_min) / W
        cell_area_m2 = resolution_m ** 2

        # Only keep line/multiline geometries
        lines = drains_gdf[drains_gdf.geometry.geom_type.isin(
            ["LineString", "MultiLineString"]
        )].copy()

        if len(lines) == 0:
            return _synthetic_drain_density(H, W)

        # Approximate length per cell by clipping each line to each cell
        # For speed: rasterize the drain network at 1m resolution, then
        # aggregate to 50m cells.
        # Simplified approach: count drain pixels per coarse cell.
        from shapely.geometry import box as sbox

        for row in range(H):
            for col in range(W):
                cell_lon_min = lon_min + col * dx
                cell_lon_max = cell_lon_min + dx
                cell_lat_min = lat_min + row * dy
                cell_lat_max = cell_lat_min + dy
                cell_geom = sbox(cell_lon_min, cell_lat_min, cell_lon_max, cell_lat_max)

                clipped = lines.clip(cell_geom)
                if len(clipped) == 0:
                    continue

                # Sum line lengths in degrees → convert to metres
                total_len_deg = clipped.geometry.length.sum()
                # 1 degree ≈ 111320 m at equator (approximate for drain lengths)
                total_len_m   = total_len_deg * 111_320
                density[row, col] = total_len_m / cell_area_m2

        # Normalise
        dmax = density.max()
        if dmax > 0:
            density /= dmax
        print(f"  [Grid/Drain] Computed from OSM: {(density > 0).sum()} non-zero cells")

    except Exception as e:
        print(f"  [Grid/Drain] OSM processing failed ({e}). Using synthetic.")
        density = _synthetic_drain_density(H, W)

    return density


def _synthetic_drain_density(H: int, W: int) -> np.ndarray:
    """
    Synthetic drain density: grid-like pattern of drains
    (N-S and E-W channels at regular intervals) with random variation.
    """
    rng   = np.random.default_rng(7)
    grid  = np.zeros((H, W), dtype=np.float32)

    # N-S drains every ~10% of width
    for cx in range(W // 10, W, W // 10):
        hw = max(2, W // 50)
        for c in range(max(0, cx - hw), min(W, cx + hw)):
            grid[:, c] += np.exp(-abs(c - cx) / max(1, hw / 2)) * rng.uniform(0.6, 1.0, H)

    # E-W drains every ~12% of height
    for ry in range(H // 12, H, H // 12):
        hw = max(2, H // 60)
        for r in range(max(0, ry - hw), min(H, ry + hw)):
            grid[r, :] += np.exp(-abs(r - ry) / max(1, hw / 2)) * rng.uniform(0.5, 0.9, W)

    grid += rng.uniform(0, 0.05, (H, W))    # background noise
    dmax  = grid.max()
    return (grid / dmax if dmax > 0 else grid).astype(np.float32)


# ── Feature 5: Impervious Fraction ───────────────────────────────────────────

def compute_impervious_fraction(
    osm_data: Optional[dict],
    H: int, W: int,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> np.ndarray:
    """
    Fraction of each cell covered by impervious surfaces (roads, buildings,
    car parks). Impervious cells generate more runoff and flood faster.

    Returns:
        impervious: float32 [H, W] — values in [0, 1]
    """
    if not HAS_GPD or not HAS_RASTERIO or osm_data is None:
        return _synthetic_impervious(H, W)

    roads_gdf = osm_data.get("roads")
    if roads_gdf is None or (hasattr(roads_gdf, "__len__") and len(roads_gdf) == 0):
        return _synthetic_impervious(H, W)

    try:
        from rasterio.transform import from_bounds as fb
        from rasterio.features import rasterize as rz

        transform = fb(lon_min, lat_min, lon_max, lat_max, W, H)

        # Roads → buffered lines (road width ~10-15m ≈ 0.00009°)
        road_lines = roads_gdf[roads_gdf.geometry.geom_type.isin(
            ["LineString", "MultiLineString"]
        )]
        if len(road_lines) == 0:
            return _synthetic_impervious(H, W)

        road_geoms = [
            (geom.buffer(0.00009), 1)
            for geom in road_lines.geometry
            if geom is not None and not geom.is_empty
        ]

        if not road_geoms:
            return _synthetic_impervious(H, W)

        road_mask = rz(
            road_geoms,
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype=np.float32,
        )

        # Smooth slightly (roads affect neighbouring cells via runoff)
        if HAS_SCIPY:
            road_mask = gaussian_filter(road_mask.astype(np.float64), sigma=0.8).astype(np.float32)

        # Clamp to [0, 1]
        impervious = np.clip(road_mask + 0.15, 0, 1).astype(np.float32)  # +0.15 urban baseline
        print(f"  [Grid/Imperv] mean={impervious.mean():.2f} from OSM roads")
        return impervious

    except Exception as e:
        print(f"  [Grid/Imperv] OSM processing failed ({e}). Using synthetic.")
        return _synthetic_impervious(H, W)


def _synthetic_impervious(H: int, W: int) -> np.ndarray:
    """
    Synthetic impervious fraction for urban area:
    - Dense centre (~0.75-0.85)
    - Suburban ring (~0.50-0.65)
    - Peri-urban edge (~0.30-0.45)
    - River/park patches (~0.10-0.20)
    """
    rng  = np.random.default_rng(13)
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2

    # Radial gradient: dense centre, lower at edges
    r    = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max(cx, cy)
    base = np.clip(0.80 - 0.45 * r, 0.15, 0.85)

    # River channel (low imperviousness)
    river_x = W * 0.15
    river   = np.exp(-((x - river_x) ** 2) / (2 * (W * 0.04) ** 2)) * 0.55
    base    = np.clip(base - river, 0.05, 0.95)

    # Add small-scale noise
    noise   = rng.normal(0, 0.04, (H, W))
    result  = np.clip(base + noise, 0.05, 0.95).astype(np.float32)

    if HAS_SCIPY:
        result = gaussian_filter(result.astype(np.float64), sigma=1.5).astype(np.float32)
    return result


# ── Feature 6: Manning's n ────────────────────────────────────────────────────

def compute_manning_n(
    impervious: np.ndarray,
    osm_data: Optional[dict],
    H: int, W: int,
    lat_min: float = 0.0, lat_max: float = 1.0,
    lon_min: float = 0.0, lon_max: float = 1.0,
) -> np.ndarray:
    """
    Estimate Manning's roughness coefficient per cell.

    Manning's n controls how fast water flows across a surface:
      - Low n (0.013) = smooth asphalt → fast flow, less ponding
      - High n (0.100) = wetland/dense vegetation → slow flow, more ponding

    Derived from impervious fraction as a proxy for surface type.
    Values from Chow (1959) Open-Channel Hydraulics.

    Returns:
        manning_n: float32 [H, W] — values in [0.012, 0.12]
    """
    # Linear interpolation between "fully paved" and "natural ground"
    n_paved   = MANNING_N["road_asphalt"]    # 0.013
    n_natural = MANNING_N["farmland"]        # 0.060
    n_water   = MANNING_N["water"]           # 0.030

    manning = (
        impervious * n_paved +
        (1.0 - impervious) * n_natural
    ).astype(np.float32)

    # If OSM water bodies available, override those cells with water n
    if HAS_GPD and HAS_RASTERIO and osm_data is not None:
        water_gdf = osm_data.get("water")
        if water_gdf is not None and hasattr(water_gdf, "__len__") and len(water_gdf) > 0:
            try:
                from rasterio.features import rasterize as rz
                from rasterio.transform import from_bounds as fb
                # FIX: transform was missing — rasterio mapped geometry coords
                # directly to pixel space, producing a blank mask every time.
                transform = fb(lon_min, lat_min, lon_max, lat_max, W, H)
                polys = water_gdf[
                    water_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
                ]
                if len(polys) > 0:
                    water_mask = rz(
                        [(g, 1) for g in polys.geometry if g and not g.is_empty],
                        out_shape=(H, W),
                        transform=transform,
                        fill=0,
                        dtype=np.uint8,
                    )
                    manning = np.where(water_mask == 1, n_water, manning)
            except Exception:
                pass

    print(f"  [Grid/Manning] mean_n={manning.mean():.4f} range=[{manning.min():.3f}, {manning.max():.3f}]")
    return manning


# ── Stack and save ────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "elevation_m",
    "slope_deg",
    "flow_accumulation",
    "drain_density",
    "impervious_fraction",
    "manning_n",
]


def build_grid(
    city: str,
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    resolution_m: int = 50,
) -> Tuple[np.ndarray, dict]:
    """
    Build the full spatial feature grid for a city.

    Args:
        city:          City key (kolkata | chennai | mumbai)
        raw_dir:       Root directory containing raw data from ingest.py
        output_dir:    Root directory to save processed grid
        resolution_m:  Target cell size in metres (default 50m)

    Returns:
        features: float32 numpy array [H, W, 6]
        meta:     dict with grid metadata (bounds, shape, feature names)
    """
    bounds = CITY_BOUNDS[city]
    lat_min, lat_max = bounds["lat"]
    lon_min, lon_max = bounds["lon"]

    raw_city  = Path(raw_dir) / city
    out_city  = Path(output_dir) / city
    out_city.mkdir(parents=True, exist_ok=True)

    print(f"\n[Grid] Building {resolution_m}m grid for {city.upper()}")
    print(f"       bbox: lat [{lat_min}, {lat_max}]  lon [{lon_min}, {lon_max}]")

    # ── Grid shape ─────────────────────────────────────────────────────────────
    H, W, dy_deg, dx_deg = compute_grid_shape(lat_min, lat_max, lon_min, lon_max, resolution_m)
    print(f"       shape: {H} rows × {W} cols  ({H * W:,} cells)")

    # ── Load raw data ──────────────────────────────────────────────────────────
    dem_path  = raw_city / f"{city}_dem.tif"
    if not dem_path.exists():
        dem_path = raw_city / f"{city}_dem.npy"   # numpy fallback from ingest
    if not dem_path.exists():
        dem_path = None

    # OSM data
    osm_cache = raw_city / f"{city}_osm.gpkg"
    osm_data  = None
    if HAS_GPD and osm_cache.exists():
        try:
            osm_data = {
                "drains": gpd.read_file(osm_cache, layer="drains"),
                "water":  gpd.read_file(osm_cache, layer="water"),
                "roads":  gpd.read_file(osm_cache, layer="roads"),
            }
            print(f"  [Grid] OSM loaded from {osm_cache.name}")
        except Exception as e:
            print(f"  [Grid] OSM load failed ({e}). Using synthetic features.")

    # ── Compute features ───────────────────────────────────────────────────────
    elevation, slope = compute_elevation_and_slope(
        dem_path, city, H, W, lat_min, lat_max, lon_min, lon_max, resolution_m
    )

    flow_acc = compute_flow_accumulation(
        dem_path, elevation, city, H, W, lat_min, lat_max, lon_min, lon_max
    )

    drain_density = compute_drain_density(
        osm_data, H, W, lat_min, lat_max, lon_min, lon_max, resolution_m
    )

    impervious = compute_impervious_fraction(
        osm_data, H, W, lat_min, lat_max, lon_min, lon_max
    )

    manning = compute_manning_n(impervious, osm_data, H, W, lat_min, lat_max, lon_min, lon_max)

    # ── Per-feature normalisation for model input ─────────────────────────────
    # Elevation: z-score (preserves relative height differences)
    elev_mean = elevation.mean()
    elev_std  = max(elevation.std(), 0.1)
    elev_norm = ((elevation - elev_mean) / elev_std).astype(np.float32)

    # Slope: clip at 45° then normalise to [0, 1]
    slope_norm = (np.clip(slope, 0, 45) / 45.0).astype(np.float32)

    # Flow accumulation already normalised by compute_flow_accumulation
    # Drain density already normalised to [0, 1]
    # Impervious already in [0, 1]
    # Manning n: normalise to [0, 1] between min/max values
    n_min, n_max = MANNING_N["road_asphalt"], MANNING_N["wetland"]
    manning_norm = ((manning - n_min) / (n_max - n_min)).clip(0, 1).astype(np.float32)

    # ── Stack into [H, W, 6] array ─────────────────────────────────────────────
    features = np.stack([
        elev_norm,       # 0: elevation (z-scored)
        slope_norm,      # 1: slope [0, 1]
        flow_acc,        # 2: flow accumulation [0, 1]
        drain_density,   # 3: drain density [0, 1]
        impervious,      # 4: impervious fraction [0, 1]
        manning_norm,    # 5: Manning's n [0, 1]
    ], axis=-1).astype(np.float32)

    print(f"\n  [Grid] Feature array shape: {features.shape}")
    for i, name in enumerate(FEATURE_NAMES):
        f = features[:, :, i]
        print(f"         {name:25s}  min={f.min():.3f}  mean={f.mean():.3f}  max={f.max():.3f}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "city":           city,
        "resolution_m":   resolution_m,
        "shape":          [H, W, 6],
        "lat_min":        lat_min, "lat_max": lat_max,
        "lon_min":        lon_min, "lon_max": lon_max,
        "dy_deg":         dy_deg,  "dx_deg":  dx_deg,
        "crs":            "EPSG:4326",
        "feature_names":  FEATURE_NAMES,
        "normalisation": {
            "elevation_m":        {"method": "z-score", "mean": float(elev_mean), "std": float(elev_std)},
            "slope_deg":          {"method": "clip_then_minmax", "clip_max": 45.0},
            "flow_accumulation":  {"method": "log1p_then_minmax"},
            "drain_density":      {"method": "minmax"},
            "impervious_fraction":{"method": "none"},
            "manning_n":          {"method": "minmax", "min": n_min, "max": n_max},
        },
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    features_path = out_city / "grid_features.npy"
    meta_path     = out_city / "grid_meta.json"
    raw_elev_path = out_city / "elevation_raw.npy"

    np.save(features_path, features)
    np.save(raw_elev_path, elevation)   # keep raw elevation for physics loss
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  [Grid] Saved:")
    print(f"         {features_path}")
    print(f"         {meta_path}")
    print(f"         {raw_elev_path}")

    # ── Optional preview plot ─────────────────────────────────────────────────
    preview_path = out_city / "grid_preview.png"
    _save_preview(features, elevation, FEATURE_NAMES, city, resolution_m, preview_path)

    return features, meta


# ── Preview visualisation ─────────────────────────────────────────────────────

def _save_preview(
    features: np.ndarray,
    elevation_raw: np.ndarray,
    feature_names: list,
    city: str,
    resolution_m: int,
    save_path: Path,
):
    """Save a 2×3 grid of feature maps as a PNG for quick inspection."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        cmaps = ["terrain", "YlOrRd", "Blues", "Greens", "RdYlGn_r", "cool"]
        titles = [
            f"Elevation (z-scored)\nraw mean={elevation_raw.mean():.1f}m",
            "Slope [0–45°]",
            "Flow Accumulation\n(log-normalised)",
            "Drain Density\n(normalised)",
            "Impervious Fraction\n[0=natural, 1=paved]",
            "Manning's n\n[0=smooth, 1=rough]",
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            f"Spatial Feature Grid — {city.capitalize()}  ({resolution_m}m resolution)\n"
            f"Shape: {features.shape[0]}×{features.shape[1]} cells",
            fontsize=13, y=1.01,
        )

        for idx, ax in enumerate(axes.flat):
            if idx >= features.shape[2]:
                ax.axis("off")
                continue
            im = ax.imshow(features[:, :, idx], cmap=cmaps[idx], origin="upper",
                           vmin=features[:, :, idx].min(),
                           vmax=features[:, :, idx].max())
            ax.set_title(titles[idx], fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.2f")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [Grid] Preview saved: {save_path}")

    except Exception as e:
        print(f"  [Grid] Preview generation failed ({e}) — skipping.")


# ── Loading helper ────────────────────────────────────────────────────────────

def load_grid(city: str, processed_dir: str = "data/processed") -> Tuple[np.ndarray, dict]:
    """
    Load a pre-built grid from disk.

    Returns:
        features: float32 [H, W, 6]
        meta:     dict with grid metadata
    """
    base = Path(processed_dir) / city
    features_path = base / "grid_features.npy"
    meta_path     = base / "grid_meta.json"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Grid not found at {features_path}. "
            f"Run: python data/grid.py --city {city}"
        )

    features = np.load(features_path)
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"[Grid] Loaded {city} grid: {features.shape} from {features_path}")
    return features, meta


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build spatial feature grid for Neural Weather Twin"
    )
    parser.add_argument("--city", default="kolkata",
                        choices=list(CITY_BOUNDS.keys()))
    parser.add_argument("--resolution", type=int, default=50,
                        help="Cell size in metres (default 50)")
    parser.add_argument("--raw_dir",    default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    features, meta = build_grid(
        city=args.city,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        resolution_m=args.resolution,
    )

    print(f"\n[Grid] Done. Array shape: {features.shape}")
    print(f"       Open data/processed/{args.city}/grid_preview.png to inspect features.")