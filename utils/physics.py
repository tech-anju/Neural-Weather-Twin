"""
utils/physics.py
Standalone physics utilities for the Neural Weather Twin.

Distinct from models/pinn.py (which embeds physics inside the training loss),
this module provides:

  1. Physical constants and unit conversions
  2. Hydraulic parameter lookup tables (Manning's n, soil infiltration)
  3. Saint-Venant equation solvers (numpy, for validation and visualisation)
  4. Froude number and flow regime classification
  5. Rainfall-runoff transformation (Green-Ampt infiltration)
  6. Physics sanity checks for model predictions
  7. Theoretical flood depth estimators (for synthetic data generation)

These functions are used by:
  - data/grid.py        → Manning's n lookup from OSM land use
  - data/dataset.py     → rainfall unit conversion
  - evaluate.py         → physics sanity checks on model output
  - scripts/kolkata_demo.py → theoretical baseline comparisons
  - app/inference.py    → rainfall-to-runoff for live forecasting

All numpy-based (no PyTorch) — can be called anywhere without GPU.

References:
  Chow, V.T. (1959). Open-Channel Hydraulics. McGraw-Hill.
  Green, W.H. & Ampt, G.A. (1911). Studies on soil physics.
    Journal of Agricultural Science, 4(1), 1-24.
  Bates, P.D. et al. (2010). A simple inertial formulation of the
    shallow water equations. Journal of Hydrology.
"""

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Physical Constants
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicalConstants:
    """Standard physical constants used in hydraulic calculations."""

    GRAVITY          = 9.81        # m/s²  — gravitational acceleration
    WATER_DENSITY    = 998.2       # kg/m³ — at 20°C
    KINEMATIC_VISC   = 1.004e-6   # m²/s  — water at 20°C
    EARTH_RADIUS_M   = 6_371_000  # m     — mean Earth radius

    # Degree-to-metre conversion at Kolkata latitude (~22.5°N)
    KOLKATA_LAT      = 22.5
    M_PER_DEG_LAT    = 111_320.0  # m/degree (nearly constant)
    M_PER_DEG_LON    = 111_320.0 * np.cos(np.radians(KOLKATA_LAT))  # ≈ 102_900 m/deg

    # Unit conversions
    MM_PER_HOUR_TO_M_PER_SEC = 1.0 / 3_600_000   # mm/h → m/s
    M_PER_SEC_TO_MM_PER_HOUR = 3_600_000.0         # m/s → mm/h

    # Flood depth thresholds
    WET_DRY_THRESHOLD = 0.001    # m — cells below this are "dry"
    PEDESTRIAN_RISK   = 0.30     # m — ankle-to-knee depth, walking impaired
    VEHICLE_RISK      = 0.45     # m — cars stall, SUVs affected
    STRUCTURAL_RISK   = 0.90     # m — structural damage to buildings


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Manning's n Lookup Tables
# ═══════════════════════════════════════════════════════════════════════════════

# Manning's roughness coefficient n by surface type
# Source: Chow (1959), Table 5-2; updated with urban values from ASCE
MANNING_N_TABLE: Dict[str, float] = {
    # ── Paved / impervious ───────────────────────────────────────────────────
    "asphalt_smooth":    0.011,
    "asphalt_road":      0.013,
    "concrete_channel":  0.012,
    "concrete_road":     0.014,
    "brick":             0.016,
    "cobblestone":       0.025,
    "gravel_road":       0.029,
    "unpaved_road":      0.050,

    # ── Urban land use ───────────────────────────────────────────────────────
    "commercial":        0.030,
    "residential_dense": 0.035,
    "residential_low":   0.045,
    "industrial":        0.028,
    "parking_lot":       0.015,

    # ── Drainage channels ────────────────────────────────────────────────────
    "concrete_drain":    0.013,
    "brick_drain":       0.017,
    "earth_drain":       0.028,
    "grass_channel":     0.030,
    "riprap_channel":    0.035,

    # ── Natural surfaces ─────────────────────────────────────────────────────
    "mowed_grass":       0.030,
    "dense_grass":       0.050,
    "sparse_vegetation": 0.035,
    "cultivated_land":   0.050,
    "rice_paddy":        0.070,
    "light_forest":      0.080,
    "dense_forest":      0.120,
    "mangrove":          0.140,

    # ── Water bodies ─────────────────────────────────────────────────────────
    "river_clean":       0.025,
    "river_weedy":       0.035,
    "canal_clean":       0.022,
    "floodplain":        0.060,
    "wetland":           0.100,
    "tidal_flat":        0.025,

    # ── Kolkata-specific ─────────────────────────────────────────────────────
    "kolkata_road":      0.014,   # mix of asphalt + potholes
    "kolkata_slum":      0.055,   # narrow lanes, debris, poor drainage
    "kolkata_park":      0.045,   # maidan-type open grass
    "hooghly_river":     0.030,
    "salt_lake":         0.040,   # reclaimed marshland

    # ── OSM tags → Manning's n ───────────────────────────────────────────────
    "residential":       0.040,   # OSM landuse=residential
    "commercial_osm":    0.030,   # OSM landuse=commercial
    "industrial_osm":    0.028,   # OSM landuse=industrial
    "park_osm":          0.045,   # OSM leisure=park
    "water_osm":         0.030,   # OSM natural=water
    "farmland_osm":      0.055,   # OSM landuse=farmland
    "forest_osm":        0.100,   # OSM landuse=forest
    "default_urban":     0.035,   # fallback for unlabelled urban
}


def get_manning_n(surface_type: str, fallback: float = 0.035) -> float:
    """
    Lookup Manning's n for a surface type string.
    Case-insensitive. Returns fallback if not found.

    Args:
        surface_type: Surface description string
        fallback:     Value if key not in table

    Returns:
        Manning's n coefficient
    """
    key = surface_type.lower().replace(" ", "_").replace("-", "_")
    if key in MANNING_N_TABLE:
        return MANNING_N_TABLE[key]
    # Try partial match
    for table_key in MANNING_N_TABLE:
        if key in table_key or table_key in key:
            return MANNING_N_TABLE[table_key]
    return fallback


def manning_n_from_osm_tags(tags: dict) -> float:
    """
    Derive Manning's n from an OpenStreetMap feature tag dict.

    Args:
        tags: OSM tag dict e.g. {"landuse": "residential", "surface": "asphalt"}

    Returns:
        Manning's n
    """
    # Priority: surface tag > landuse tag > highway tag > default
    surface  = tags.get("surface", "")
    landuse  = tags.get("landuse", "")
    waterway = tags.get("waterway", "")
    highway  = tags.get("highway", "")
    natural  = tags.get("natural", "")

    if surface in ("asphalt", "paved"):
        return MANNING_N_TABLE["asphalt_road"]
    if surface in ("concrete",):
        return MANNING_N_TABLE["concrete_road"]
    if surface in ("unpaved", "gravel", "dirt"):
        return MANNING_N_TABLE["unpaved_road"]
    if waterway in ("canal", "drain", "ditch"):
        return MANNING_N_TABLE["earth_drain"]
    if waterway in ("river", "stream"):
        return MANNING_N_TABLE["river_clean"]
    if natural in ("water", "wetland"):
        return MANNING_N_TABLE["wetland"]
    if landuse in ("residential",):
        return MANNING_N_TABLE["residential"]
    if landuse in ("commercial",):
        return MANNING_N_TABLE["commercial_osm"]
    if landuse in ("industrial",):
        return MANNING_N_TABLE["industrial_osm"]
    if landuse in ("forest",):
        return MANNING_N_TABLE["forest_osm"]
    if landuse in ("farmland", "farm"):
        return MANNING_N_TABLE["farmland_osm"]
    if highway:
        return MANNING_N_TABLE["asphalt_road"]

    return MANNING_N_TABLE["default_urban"]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Hydraulic Calculations
# ═══════════════════════════════════════════════════════════════════════════════

def froude_number(
    velocity: Union[float, np.ndarray],
    depth: Union[float, np.ndarray],
    g: float = PhysicalConstants.GRAVITY,
) -> Union[float, np.ndarray]:
    """
    Froude number: Fr = U / sqrt(g·h)

    Classifies flow regime:
      Fr < 1  → subcritical (tranquil) — typical urban flooding
      Fr = 1  → critical
      Fr > 1  → supercritical (rapid) — unusual for urban floods

    The diffusive wave approximation in pinn.py is valid only for Fr << 1.
    If Fr > 0.5 anywhere, consider switching to full Saint-Venant.

    Args:
        velocity: Flow speed (m/s)
        depth:    Water depth (m)
        g:        Gravitational acceleration

    Returns:
        Froude number (dimensionless)
    """
    depth_safe = np.maximum(depth, PhysicalConstants.WET_DRY_THRESHOLD)
    return np.abs(velocity) / np.sqrt(g * depth_safe)


def manning_velocity(
    depth: Union[float, np.ndarray],
    slope: Union[float, np.ndarray],
    n: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Manning's equation: U = (1/n) × R_h^(2/3) × S^(1/2)

    Wide-channel approximation: R_h ≈ depth.
    Valid when width >> depth (true for most urban flood sheets).

    Args:
        depth: Water depth (m)
        slope: Bed slope (dimensionless, m/m)
        n:     Manning's roughness coefficient

    Returns:
        Flow velocity (m/s)
    """
    depth_safe = np.maximum(depth, PhysicalConstants.WET_DRY_THRESHOLD)
    slope_safe = np.maximum(np.abs(slope), 1e-6)
    return (1.0 / n) * (depth_safe ** (2.0 / 3.0)) * np.sqrt(slope_safe)


def flood_discharge(
    depth: Union[float, np.ndarray],
    velocity: Union[float, np.ndarray],
    width: float = 50.0,
) -> Union[float, np.ndarray]:
    """
    Volumetric discharge Q = h × U × w  (m³/s per metre width)

    Args:
        depth:    Water depth (m)
        velocity: Flow velocity (m/s)
        width:    Channel/cell width (m), default 50m = one grid cell

    Returns:
        Discharge Q (m³/s)
    """
    return depth * velocity * width


def critical_depth(
    discharge_per_width: Union[float, np.ndarray],
    g: float = PhysicalConstants.GRAVITY,
) -> Union[float, np.ndarray]:
    """
    Critical depth: h_c = (q²/g)^(1/3)
    where q = Q/width = discharge per unit width (m²/s)

    At critical depth, Fr = 1. Useful for checking model outputs.
    """
    return (discharge_per_width ** 2 / g) ** (1.0 / 3.0)


def wave_celerity(
    depth: Union[float, np.ndarray],
    g: float = PhysicalConstants.GRAVITY,
) -> Union[float, np.ndarray]:
    """
    Shallow water wave celerity: c = sqrt(g·h)

    This is the speed at which a flood wave propagates.
    Used to estimate maximum forecast horizon validity.
    """
    return np.sqrt(g * np.maximum(depth, PhysicalConstants.WET_DRY_THRESHOLD))


def courant_number(
    velocity: Union[float, np.ndarray],
    depth: Union[float, np.ndarray],
    dt: float = 3600.0,
    dx: float = 50.0,
    g: float = PhysicalConstants.GRAVITY,
) -> Union[float, np.ndarray]:
    """
    Courant-Friedrichs-Lewy (CFL) number:
      C = (|U| + c) × dt / dx

    Explicit numerical schemes require C ≤ 1 for stability.
    Our model is implicit (neural network), so CFL > 1 is OK,
    but very large values indicate physically unrealistic predictions.

    Guideline: C > 10 suggests the model may be predicting
    unrealistically fast flood propagation.

    Args:
        velocity: Flow speed (m/s)
        depth:    Water depth (m)
        dt:       Timestep (s), default 3600 = 1 hour
        dx:       Cell size (m), default 50m

    Returns:
        CFL number (dimensionless)
    """
    c = wave_celerity(depth, g)
    return (np.abs(velocity) + c) * dt / dx


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Rainfall-Runoff: Green-Ampt Infiltration
# ═══════════════════════════════════════════════════════════════════════════════

# Green-Ampt soil parameters by soil type
# Source: Rawls et al. (1983). Journal of Hydraulic Engineering
SOIL_PARAMS: Dict[str, dict] = {
    "sand":          {"Ks": 117.8, "psi": 49.5,  "theta": 0.417, "fc": 0.045},
    "loamy_sand":    {"Ks": 29.9,  "psi": 61.3,  "theta": 0.401, "fc": 0.075},
    "sandy_loam":    {"Ks": 10.9,  "psi": 110.1, "theta": 0.412, "fc": 0.114},
    "loam":          {"Ks": 3.4,   "psi": 88.9,  "theta": 0.434, "fc": 0.179},
    "silt_loam":     {"Ks": 6.5,   "psi": 166.8, "theta": 0.486, "fc": 0.174},
    "clay_loam":     {"Ks": 1.0,   "psi": 208.8, "theta": 0.465, "fc": 0.255},
    "clay":          {"Ks": 0.3,   "psi": 316.3, "theta": 0.475, "fc": 0.335},
    # Urban soils (compacted, reduced infiltration)
    "urban_sandy":   {"Ks": 5.0,   "psi": 80.0,  "theta": 0.35,  "fc": 0.10},
    "urban_clay":    {"Ks": 0.5,   "psi": 250.0, "theta": 0.40,  "fc": 0.28},
    "kolkata_soil":  {"Ks": 1.5,   "psi": 180.0, "theta": 0.45,  "fc": 0.25},
    # Fully impervious (roads, rooftops)
    "impervious":    {"Ks": 0.0,   "psi": 0.0,   "theta": 0.0,   "fc": 0.0},
}


def green_ampt_infiltration(
    rainfall_mm_hr: Union[float, np.ndarray],
    cumulative_infiltration_mm: Union[float, np.ndarray],
    soil_type: str = "kolkata_soil",
    dt_hours: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Green-Ampt infiltration model.

    Estimates how much rainfall infiltrates into soil vs runs off.
    Key for urban flood modelling: impervious surfaces have near-zero
    infiltration, causing rapid runoff.

    Green-Ampt equation:
      f(t) = Ks × (1 + (psi × dθ) / F(t))
      where:
        f    = infiltration rate (mm/hr)
        Ks   = saturated hydraulic conductivity (mm/hr)
        psi  = wetting front suction head (mm)
        dθ   = (porosity - initial moisture) deficit
        F(t) = cumulative infiltration at time t (mm)

    Args:
        rainfall_mm_hr:              Rainfall rate (mm/hr)
        cumulative_infiltration_mm:  Cumulative infiltration so far (mm)
        soil_type:                   Key from SOIL_PARAMS
        dt_hours:                    Timestep in hours

    Returns:
        infiltration_mm:  Infiltration this timestep (mm)
        runoff_mm:        Runoff this timestep (mm)
    """
    params = SOIL_PARAMS.get(soil_type, SOIL_PARAMS["kolkata_soil"])
    Ks     = params["Ks"]    # mm/hr
    psi    = params["psi"]   # mm
    dtheta = params["theta"] - params["fc"]

    rainfall_mm_hr     = np.atleast_1d(np.asarray(rainfall_mm_hr, dtype=np.float64))
    cumul              = np.atleast_1d(np.asarray(cumulative_infiltration_mm, dtype=np.float64))

    # Green-Ampt infiltration rate
    cumul_safe  = np.maximum(cumul, 1e-6)
    infil_rate  = Ks * (1.0 + (psi * dtheta) / cumul_safe)  # mm/hr

    # Actual infiltration = min(potential, available rainfall)
    potential_infil = infil_rate * dt_hours                  # mm per timestep
    actual_infil    = np.minimum(potential_infil, rainfall_mm_hr * dt_hours)
    actual_infil    = np.maximum(actual_infil, 0.0)

    runoff = np.maximum(rainfall_mm_hr * dt_hours - actual_infil, 0.0)

    return actual_infil.squeeze(), runoff.squeeze()


def runoff_coefficient(
    impervious_fraction: Union[float, np.ndarray],
    slope_deg: Union[float, np.ndarray] = 1.0,
    rainfall_intensity_mm_hr: float = 25.0,
) -> Union[float, np.ndarray]:
    """
    Rational method runoff coefficient C.

    Q = C × i × A   (Rational formula for peak discharge)

    Simplified empirical formula combining:
      - Imperviousness (dominant factor for urban areas)
      - Slope (steeper = more runoff)
      - Rainfall intensity

    Args:
        impervious_fraction:       0 (fully natural) to 1 (fully paved)
        slope_deg:                 Terrain slope in degrees
        rainfall_intensity_mm_hr:  Rainfall rate

    Returns:
        C: Runoff coefficient in [0, 1]
    """
    # Base coefficient from imperviousness
    C_base = 0.05 + 0.85 * impervious_fraction

    # Slope correction: steeper terrain → slightly higher C
    slope_factor = 1.0 + 0.003 * np.clip(slope_deg, 0, 30)

    # Intensity correction: higher rainfall → higher effective C
    intensity_factor = 1.0 + 0.001 * np.clip(rainfall_intensity_mm_hr - 10, 0, 100)

    return np.clip(C_base * slope_factor * intensity_factor, 0.0, 1.0)


def rainfall_to_runoff(
    rainfall_mm: Union[float, np.ndarray],
    impervious_fraction: Union[float, np.ndarray],
    slope_deg: Union[float, np.ndarray] = 1.0,
) -> Union[float, np.ndarray]:
    """
    Convert rainfall depth (mm) to runoff depth (mm) using
    the Rational method runoff coefficient.

    Simple proxy used when Green-Ampt soil parameters unavailable.

    Returns:
        runoff_mm: Runoff depth in mm
    """
    C = runoff_coefficient(impervious_fraction, slope_deg,
                           rainfall_intensity_mm_hr=float(np.mean(rainfall_mm)))
    return C * rainfall_mm


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Saint-Venant Solver (NumPy — for validation / visualisation)
# ═══════════════════════════════════════════════════════════════════════════════

def saint_venant_continuity_residual(
    h_t0:     np.ndarray,   # [H, W] depth at t
    h_t1:     np.ndarray,   # [H, W] depth at t+1
    u:        np.ndarray,   # [H, W] x-velocity (m/s)
    v:        np.ndarray,   # [H, W] y-velocity (m/s)
    rainfall: np.ndarray,   # [H, W] rainfall rate (m/s)
    dx: float = 50.0,
    dy: float = 50.0,
    dt: float = 3600.0,
) -> np.ndarray:
    """
    Continuity equation residual (numpy version for validation).

    R = ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y - rainfall

    Perfect model → R ≈ 0 everywhere.
    Large R indicates physical inconsistency in predictions.

    Returns:
        residual: [H, W] — should be near zero for valid predictions
    """
    # Time derivative
    dhdt = (h_t1 - h_t0) / dt

    # Flux divergence using central differences with edge replication
    hu   = h_t0 * u
    hv   = h_t0 * v

    # ∂(hu)/∂x
    hu_pad = np.pad(hu, ((0,0),(1,1)), mode="edge")
    dhu_dx = (hu_pad[:, 2:] - hu_pad[:, :-2]) / (2 * dx)

    # ∂(hv)/∂y
    hv_pad = np.pad(hv, ((1,1),(0,0)), mode="edge")
    dhv_dy = (hv_pad[2:, :] - hv_pad[:-2, :]) / (2 * dy)

    return dhdt + dhu_dx + dhv_dy - rainfall


def diffusive_wave_velocity(
    h:        np.ndarray,   # [H, W] water depth (m)
    z:        np.ndarray,   # [H, W] bed elevation (m)
    n:        np.ndarray,   # [H, W] Manning's n
    dx: float = 50.0,
    dy: float = 50.0,
    min_depth: float = PhysicalConstants.WET_DRY_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate velocity field from depth + terrain using diffusive wave.

    Diffusive wave: flow driven by water surface slope (bed + free surface).
    Standard approximation for slow urban flooding (Fr << 1).

    Returns:
        u: [H, W] x-velocity (m/s)
        v: [H, W] y-velocity (m/s)
    """
    eps      = 1e-8
    wet      = (h > min_depth).astype(np.float64)
    h_safe   = np.maximum(h, min_depth)

    # Water surface elevation: η = z + h
    eta     = z + h

    # Water surface slopes
    eta_pad_x = np.pad(eta, ((0,0),(1,1)), mode="edge")
    eta_pad_y = np.pad(eta, ((1,1),(0,0)), mode="edge")
    detadx    = (eta_pad_x[:, 2:] - eta_pad_x[:, :-2]) / (2 * dx)
    detady    = (eta_pad_y[2:, :] - eta_pad_y[:-2, :]) / (2 * dy)

    slope_mag = np.sqrt(detadx**2 + detady**2 + eps)

    # Manning velocity magnitude
    speed = (1.0 / n) * (h_safe ** (2.0 / 3.0)) * np.sqrt(slope_mag)
    speed = speed * wet

    # Velocity components — negative sign: flow in downslope direction
    u = -speed * detadx / slope_mag
    v = -speed * detady / slope_mag

    return u, v


def estimate_flood_depth_peak(
    rainfall_mm:         float,
    area_m2:             float,
    impervious_fraction: float,
    drain_capacity_m3_s: float = 10.0,
    duration_hours:      float = 3.0,
) -> float:
    """
    Estimate peak flood depth for a drainage basin using water balance.

    Simple analytical estimate:
      inflow  = rainfall × area × runoff_coefficient
      outflow = drainage_capacity × duration
      storage = inflow - outflow
      depth   = storage / area

    Used to:
      - Validate model predictions (should be in same ballpark)
      - Generate synthetic events with realistic depths
      - Quick sanity check in the Streamlit demo

    Args:
        rainfall_mm:         Total rainfall depth (mm)
        area_m2:             Basin area (m²)
        impervious_fraction: 0-1, fraction of paved surface
        drain_capacity_m3_s: Drainage system capacity (m³/s)
        duration_hours:      Storm duration (hours)

    Returns:
        peak_depth_m: Estimated peak flood depth (metres)
    """
    # Runoff volume
    C            = runoff_coefficient(impervious_fraction)
    inflow_m3    = C * (rainfall_mm / 1000) * area_m2

    # Drainage outflow
    outflow_m3   = drain_capacity_m3_s * duration_hours * 3600

    # Net storage
    storage_m3   = max(0.0, inflow_m3 - outflow_m3)

    # Average depth over basin
    avg_depth_m  = storage_m3 / area_m2

    # Peak depth ≈ 2 × average (non-uniform distribution)
    peak_depth_m = 2.0 * avg_depth_m

    return float(np.clip(peak_depth_m, 0, 5.0))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Unit Conversions
# ═══════════════════════════════════════════════════════════════════════════════

def mm_per_hour_to_m_per_sec(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """mm/h → m/s."""
    return value * PhysicalConstants.MM_PER_HOUR_TO_M_PER_SEC


def m_per_sec_to_mm_per_hour(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """m/s → mm/h."""
    return value * PhysicalConstants.M_PER_SEC_TO_MM_PER_HOUR


def degrees_to_metres(lat_deg: float, lon_deg: float, ref_lat: float) -> Tuple[float, float]:
    """
    Convert geographic coordinate offsets (degrees) to metres.
    Uses flat-Earth approximation valid for city-scale areas.

    Args:
        lat_deg: Latitude offset in degrees
        lon_deg: Longitude offset in degrees
        ref_lat: Reference latitude (degrees) for longitude scaling

    Returns:
        (dy_m, dx_m): Offsets in metres
    """
    dy_m = lat_deg * PhysicalConstants.M_PER_DEG_LAT
    dx_m = lon_deg * PhysicalConstants.M_PER_DEG_LAT * np.cos(np.radians(ref_lat))
    return dy_m, dx_m


def depth_to_volume(
    depth_map: np.ndarray,
    cell_size_m: float = 50.0,
) -> float:
    """
    Convert depth map (metres) to total water volume (m³).

    Args:
        depth_map:   [H, W] flood depth in metres
        cell_size_m: Grid cell size (metres)

    Returns:
        volume_m3: Total water volume in cubic metres
    """
    cell_area = cell_size_m ** 2
    return float(np.sum(np.maximum(depth_map, 0)) * cell_area)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Physics Sanity Checks
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsSanityChecker:
    """
    Validate model predictions against physical constraints.

    Used by evaluate.py to flag physically implausible predictions
    before they reach the alert system. Catches model failure modes
    like water appearing from nowhere, negative depths, or supersonic flow.

    Args:
        dx, dy:          Grid cell size (metres)
        dt:              Timestep (seconds)
        max_froude:      Maximum plausible Froude number (default 0.5)
        max_depth_m:     Maximum plausible flood depth in metres
        max_velocity_ms: Maximum plausible flow velocity in m/s
        min_depth:       Wetting/drying threshold
    """

    def __init__(
        self,
        dx: float = 50.0,
        dy: float = 50.0,
        dt: float = 3600.0,
        max_froude: float = 0.5,
        max_depth_m: float = 5.0,
        max_velocity_ms: float = 3.0,
        min_depth: float = PhysicalConstants.WET_DRY_THRESHOLD,
    ):
        self.dx              = dx
        self.dy              = dy
        self.dt              = dt
        self.max_froude      = max_froude
        self.max_depth_m     = max_depth_m
        self.max_velocity_ms = max_velocity_ms
        self.min_depth       = min_depth

    def check(
        self,
        predictions: np.ndarray,    # [T_out, H, W]
        elevation: np.ndarray,      # [H, W]
        rainfall: np.ndarray,       # [T_out, H, W] in mm
        manning_n: Optional[np.ndarray] = None,   # [H, W]
        verbose: bool = True,
    ) -> Dict[str, object]:
        """
        Run all physics checks on model predictions.

        Returns:
            report: dict with check results and failure details
        """
        T_out, H, W = predictions.shape
        g           = PhysicalConstants.GRAVITY
        results     = {}

        if manning_n is None:
            manning_n = np.full((H, W), 0.035)

        # ── Check 1: No negative depths ───────────────────────────────────
        neg_frac = float((predictions < 0).mean())
        results["no_negative_depths"] = {
            "pass":    neg_frac == 0,
            "value":   neg_frac,
            "detail":  f"{neg_frac*100:.2f}% cells with negative depth",
        }

        # ── Check 2: Depth within plausible range ─────────────────────────
        max_pred = float(predictions.max())
        results["depth_range"] = {
            "pass":   max_pred <= self.max_depth_m,
            "value":  max_pred,
            "detail": f"max predicted depth = {max_pred:.2f}m (limit {self.max_depth_m}m)",
        }

        # ── Check 3: Froude number check (per timestep) ───────────────────
        max_fr_all = 0.0
        for t in range(T_out):
            h   = predictions[t]
            dzdx, dzdy = self._bed_slope(elevation)
            u, v = diffusive_wave_velocity(h, elevation, manning_n,
                                           self.dx, self.dy)
            speed = np.sqrt(u**2 + v**2)
            fr    = froude_number(speed, h, g)
            wet   = h > self.min_depth
            if wet.any():
                max_fr_all = max(max_fr_all, float(fr[wet].max()))

        results["froude_number"] = {
            "pass":   max_fr_all <= self.max_froude,
            "value":  max_fr_all,
            "detail": f"max Froude = {max_fr_all:.3f} (limit {self.max_froude})",
        }

        # ── Check 4: Mass conservation (continuity residual) ──────────────
        max_continuity = 0.0
        for t in range(T_out - 1):
            h0   = predictions[t]
            h1   = predictions[t + 1]
            rain = mm_per_hour_to_m_per_sec(rainfall[t])
            dzdx, dzdy = self._bed_slope(elevation)
            u, v = diffusive_wave_velocity(h0, elevation, manning_n,
                                           self.dx, self.dy)
            res  = saint_venant_continuity_residual(
                h0, h1, u, v, rain, self.dx, self.dy, self.dt
            )
            wet = h0 > self.min_depth
            if wet.any():
                max_continuity = max(max_continuity,
                                     float(np.abs(res[wet]).mean()))

        results["mass_conservation"] = {
            "pass":   max_continuity < 1e-3,
            "value":  max_continuity,
            "detail": f"mean |continuity residual| = {max_continuity:.2e} m/s",
        }

        # ── Check 5: Flood consistent with rainfall ───────────────────────
        total_rain_mm  = float(rainfall.sum(axis=0).mean())
        total_flood_mm = float(predictions.max(axis=0).mean() * 1000)
        # Flood depth should not exceed total rainfall ×2 (some accumulation ok)
        results["rainfall_flood_consistency"] = {
            "pass":   total_flood_mm <= total_rain_mm * 2.0 + 50,
            "value":  total_flood_mm / max(total_rain_mm, 1),
            "detail": f"mean flood={total_flood_mm:.1f}mm vs rain={total_rain_mm:.1f}mm",
        }

        # ── Check 6: No flood on elevated terrain ─────────────────────────
        high_terrain   = elevation > (elevation.mean() + 2 * elevation.std())
        flood_on_high  = float(predictions[:, high_terrain].mean())
        results["no_flood_on_hills"] = {
            "pass":   flood_on_high < 0.05,
            "value":  flood_on_high,
            "detail": f"mean depth on high terrain = {flood_on_high:.3f}m",
        }

        # ── Summary ───────────────────────────────────────────────────────
        n_pass    = sum(1 for r in results.values() if r["pass"])
        n_total   = len(results)
        all_pass  = n_pass == n_total
        results["summary"] = {
            "pass":   all_pass,
            "score":  f"{n_pass}/{n_total}",
            "detail": "all physics checks passed" if all_pass
                      else f"{n_total - n_pass} check(s) failed",
        }

        if verbose:
            print(f"\n[PhysicsSanityChecker] {results['summary']['score']} checks passed")
            for name, res in results.items():
                if name == "summary":
                    continue
                icon = "✓" if res["pass"] else "✗"
                print(f"  {icon}  {name:<35} {res['detail']}")

        return results

    def _bed_slope(self, elevation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Central difference bed slopes."""
        e_padx = np.pad(elevation, ((0,0),(1,1)), mode="edge")
        e_pady = np.pad(elevation, ((1,1),(0,0)), mode="edge")
        dzdx   = (e_padx[:, 2:] - e_padx[:, :-2]) / (2 * self.dx)
        dzdy   = (e_pady[2:, :] - e_pady[:-2, :]) / (2 * self.dy)
        return dzdx, dzdy


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    ok, fail = [], []

    def check(cond, msg, val=""):
        s = f"{'✓' if cond else '✗'}  {msg}"
        if val: s += f"  [{val}]"
        (ok if cond else fail).append(s)

    print("=" * 60)
    print("  utils/physics.py — smoke test")
    print("=" * 60)

    # Constants
    check(PhysicalConstants.GRAVITY == 9.81, "GRAVITY = 9.81")
    check(abs(PhysicalConstants.M_PER_DEG_LON - 102_900) < 2000,
          "M_PER_DEG_LON reasonable for Kolkata",
          f"{PhysicalConstants.M_PER_DEG_LON:.0f}")

    # Manning's n lookup
    n_asphalt = get_manning_n("asphalt_road")
    n_forest  = get_manning_n("forest")
    n_default = get_manning_n("unknown_surface_xyz")
    check(n_asphalt < n_forest,     "Asphalt n < forest n",
          f"{n_asphalt:.3f} < {n_forest:.3f}")
    check(n_default == 0.035,       "Unknown surface → default 0.035")

    # OSM tag lookup
    n_osm = manning_n_from_osm_tags({"landuse": "residential", "surface": "asphalt"})
    check(n_osm == MANNING_N_TABLE["asphalt_road"], "OSM asphalt → correct n")

    # Hydraulic calcs
    Fr = froude_number(0.5, 0.3)
    check(0 < Fr < 1, f"Froude number subcritical", f"Fr={Fr:.3f}")

    U = manning_velocity(0.3, 0.001, 0.035)
    check(U > 0, f"Manning velocity > 0", f"U={U:.3f} m/s")

    CFL = courant_number(U, 0.3)
    check(CFL > 0, f"CFL number computed", f"CFL={CFL:.2f}")

    # Rainfall → runoff
    C_paved = runoff_coefficient(0.9)
    C_grass = runoff_coefficient(0.1)
    check(C_paved > C_grass, "Paved C > grass C",
          f"{C_paved:.2f} > {C_grass:.2f}")

    infil, runoff = green_ampt_infiltration(25.0, 10.0, "kolkata_soil")
    check(infil >= 0 and runoff >= 0, "Green-Ampt returns non-negative",
          f"infil={infil:.2f}mm runoff={runoff:.2f}mm")
    check(abs(infil + runoff - 25.0) < 0.01, "infil + runoff = rainfall")

    # Unit conversions
    v_ms = mm_per_hour_to_m_per_sec(3600.0)
    check(abs(v_ms - 1.0) < 1e-9, "3600 mm/h = 1 m/s")

    dy_m, dx_m = degrees_to_metres(0.01, 0.01, 22.5)
    check(dy_m > 1000 and dx_m > 900, "Degree → metre conversion",
          f"dy={dy_m:.0f}m dx={dx_m:.0f}m")

    # Depth → volume
    depth = np.ones((200, 200)) * 0.5
    vol   = depth_to_volume(depth, cell_size_m=50)
    check(abs(vol - 0.5 * 200 * 200 * 50**2) < 1,
          "depth_to_volume correct", f"{vol:.0f} m³")

    # Saint-Venant continuity (numpy)
    H, W  = 20, 20
    h0    = np.ones((H, W)) * 0.3
    h1    = np.ones((H, W)) * 0.3   # steady state → residual ≈ 0
    u_arr = np.zeros((H, W))
    v_arr = np.zeros((H, W))
    rain  = np.zeros((H, W))
    res   = saint_venant_continuity_residual(h0, h1, u_arr, v_arr, rain)
    check(np.abs(res).max() < 1e-10, "Continuity residual = 0 at steady state",
          f"max={np.abs(res).max():.2e}")

    # Diffusive wave velocity
    elev  = np.linspace(0, 1, H)[:, np.newaxis] * np.ones((H, W))
    n_arr = np.full((H, W), 0.035)
    u_dw, v_dw = diffusive_wave_velocity(h0, elev, n_arr)
    check(u_dw.shape == (H, W), "diffusive_wave_velocity shape OK")
    check(v_dw.max() <= 0, "Flow in downslope direction (v ≤ 0)",
          f"v max={v_dw.max():.3f}")

    # Peak flood depth estimate
    d = estimate_flood_depth_peak(
        rainfall_mm=100, area_m2=1e6,
        impervious_fraction=0.7, drain_capacity_m3_s=5.0
    )
    check(0 < d < 5.0, f"Peak depth estimate reasonable", f"{d:.3f}m")

    # Sanity checker
    preds   = np.random.rand(3, 20, 20) * 0.3
    elev2   = np.random.rand(20, 20) * 5
    rainfall_arr = np.random.rand(3, 20, 20) * 20
    checker = PhysicsSanityChecker()
    report  = checker.check(preds, elev2, rainfall_arr, verbose=False)
    check("summary" in report,      "PhysicsSanityChecker returns report")
    check("score" in report["summary"], "Report has score field")

    # Report
    print(f"\n{'='*55}")
    print(f"  PASSED: {len(ok)}   FAILED: {len(fail)}")
    print(f"{'='*55}\n")
    if fail:
        print("FAILURES:")
        for f in fail: print(f"  {f}")
        print()
    print("PASSED:")
    for o in ok: print(f"  {o}")