"""
app/inference.py
Production inference pipeline for the Neural Weather Twin.

Provides a clean, single-call interface that:
  1. Accepts raw rainfall + terrain inputs
  2. Builds the model input tensor
  3. Runs the WeatherTwin forward pass
  4. Post-processes outputs into human-readable alerts
  5. Caches the model across calls (no reload per request)

Used by:
  app/app.py          — Streamlit demo
  FastAPI endpoint    — REST API for external consumers
  scripts/kolkata_demo.py — CLI demo replay

FastAPI usage:
  uvicorn app.inference:api --host 0.0.0.0 --port 8000
  POST /predict   { rainfall_grid: [[...]], timestamp: "2024-09-21T15:00:00" }
  GET  /health
  GET  /config

Direct Python usage:
  from app.inference import FloodInferenceEngine
  engine = FloodInferenceEngine.get_instance("checkpoints/best_model.pth", config)
  result = engine.predict_from_rainfall(rainfall_mm_grid, elevation, manning_n)
"""

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("flood_twin.inference")

# ── Lazy torch import (avoids crash if torch missing) ──────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("torch not installed — inference will use synthetic fallback")


# ═══════════════════════════════════════════════════════════════════════════
# Alert levels
# ═══════════════════════════════════════════════════════════════════════════

ALERT_DRY     = 0
ALERT_WATCH   = 1
ALERT_WARNING = 2
ALERT_DANGER  = 3

ALERT_NAMES  = {0: "DRY", 1: "WATCH", 2: "WARNING", 3: "DANGER"}
ALERT_COLORS = {0: "#4CAF50", 1: "#FFC107", 2: "#FF5722", 3: "#D32F2F"}
ALERT_EMOJI  = {0: "✅", 1: "🟡", 2: "🟠", 3: "🔴"}

ALERT_ACTIONS = {
    ALERT_DRY:     "No action required. Continue monitoring.",
    ALERT_WATCH:   "Monitor drainage. Review evacuation plans for low-lying areas.",
    ALERT_WARNING: "Prepare for evacuation. Pre-position emergency resources. "
                   "Issue public advisory.",
    ALERT_DANGER:  "Immediate evacuation of affected wards. Deploy rescue teams. "
                   "Life-threatening conditions.",
}


# ═══════════════════════════════════════════════════════════════════════════
# Inference result container
# ═══════════════════════════════════════════════════════════════════════════

class InferenceResult:
    """
    Complete inference result from a single forward pass.

    Attributes:
        depth_maps:      float32 [T_out, H, W] — metres
        uncertainty:     float32 [T_out, H, W] — std dev, or None
        alert_map:       uint8   [H, W]         — 0=dry … 3=danger
        ward_alerts:     dict    {ward_id: {level, name, color, flooded_pct}}
        highest_alert:   int     — worst alert level across all cells
        flooded_cells:   int     — cells above flood threshold
        flooded_pct:     float   — percentage of grid that is flooded
        max_depth_m:     float   — maximum predicted depth
        inference_ms:    float   — wall-clock time in milliseconds
        timestamp:       str     — ISO timestamp of this forecast
        forecast_times:  list    — ["T+1h", "T+2h", "T+3h"]
    """

    def __init__(
        self,
        depth_maps:    np.ndarray,
        uncertainty:   Optional[np.ndarray],
        alert_map:     np.ndarray,
        ward_alerts:   Dict,
        inference_ms:  float,
        timestamp:     str,
        thresholds:    dict,
        T_out:         int = 3,
    ):
        self.depth_maps     = depth_maps
        self.uncertainty    = uncertainty
        self.alert_map      = alert_map
        self.ward_alerts    = ward_alerts
        self.inference_ms   = inference_ms
        self.timestamp      = timestamp
        self.thresholds     = thresholds
        self.forecast_times = [f"T+{t+1}h" for t in range(T_out)]

        # Derived stats
        self.highest_alert = int(alert_map.max())
        total              = alert_map.size
        self.flooded_cells = int((alert_map > ALERT_DRY).sum())
        self.flooded_pct   = float(self.flooded_cells / total * 100)
        self.max_depth_m   = float(depth_maps.max())

    @property
    def highest_alert_name(self) -> str:
        return ALERT_NAMES[self.highest_alert]

    @property
    def recommended_action(self) -> str:
        return ALERT_ACTIONS[self.highest_alert]

    def get_horizon(self, horizon: str) -> np.ndarray:
        """
        Get depth map for a specific forecast horizon.
        Args: horizon — "T+1h" | "T+2h" | "T+3h"
        Returns: [H, W] float32
        """
        idx = {"T+1h": 0, "T+2h": 1, "T+3h": 2}.get(horizon, 1)
        return self.depth_maps[idx]

    def wards_at_level(self, level: int) -> List[str]:
        """Return list of ward IDs at or above the given alert level."""
        return [
            wid for wid, info in self.ward_alerts.items()
            if info.get("level", 0) >= level
        ]

    def to_dict(self) -> dict:
        """Serialisable summary (no large arrays)."""
        return {
            "timestamp":         self.timestamp,
            "highest_alert":     self.highest_alert_name,
            "highest_alert_int": self.highest_alert,
            "recommended_action":self.recommended_action,
            "flooded_pct":       round(self.flooded_pct, 2),
            "flooded_cells":     self.flooded_cells,
            "max_depth_m":       round(self.max_depth_m, 3),
            "inference_ms":      round(self.inference_ms, 1),
            "forecast_times":    self.forecast_times,
            "thresholds":        self.thresholds,
            "ward_alerts":       self.ward_alerts,
            "alert_summary": {
                ALERT_NAMES[l]: int((self.alert_map == l).sum())
                for l in range(4)
            },
        }

    def __repr__(self) -> str:
        return (
            f"InferenceResult("
            f"alert={self.highest_alert_name} | "
            f"max={self.max_depth_m:.2f}m | "
            f"flooded={self.flooded_pct:.1f}% | "
            f"{self.inference_ms:.0f}ms)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Input preprocessor
# ═══════════════════════════════════════════════════════════════════════════

class InputPreprocessor:
    """
    Converts raw rainfall/terrain arrays into model-ready tensors.

    Input tensor layout (C=8 channels per timestep):
      Ch 0: rainfall_mm          — log1p normalised with max 150mm
      Ch 1: elevation_norm       — z-scored
      Ch 2: slope_norm           — [0,1], clipped at 45°
      Ch 3: flow_accumulation    — log-normalised [0,1]
      Ch 4: drain_density        — [0,1]
      Ch 5: impervious_fraction  — [0,1]
      Ch 6: manning_n_norm       — [0,1]
      Ch 7: prev_flood_depth     — / 5.0 metres → [0,1]
    """

    RAINFALL_LOG_MAX  = np.log1p(150.0)   # normalisation cap: 150mm
    MAX_FLOOD_DEPTH_M = 5.0               # normalisation cap for prev depth

    def __init__(
        self,
        grid_features: Optional[np.ndarray] = None,   # [H, W, 6]
        elevation_raw: Optional[np.ndarray] = None,   # [H, W]
        T_in:          int = 6,
        H:             int = 80,
        W:             int = 80,
    ):
        self.T_in = T_in
        self.H    = H
        self.W    = W

        if grid_features is not None and grid_features.shape[:2] == (H, W):
            # terrain_channels: [6, H, W]
            self.terrain = np.transpose(
                grid_features.astype(np.float32), (2, 0, 1)
            )
        else:
            # Synthetic terrain for demo
            self.terrain = self._synthetic_terrain(H, W)

        if elevation_raw is not None and elevation_raw.shape == (H, W):
            self.elevation_raw = elevation_raw.astype(np.float32)
        else:
            self.elevation_raw = self._synthetic_elevation(H, W)

    # ── Build input tensor ────────────────────────────────────────────────

    def build_input_tensor(
        self,
        rainfall_sequence: np.ndarray,     # [T_in, H, W] mm per hour
        prev_flood_depth:  Optional[np.ndarray] = None,  # [H, W] metres
    ) -> "torch.Tensor":
        """
        Build model input tensor [1, T_in, 8, H, W].

        Args:
            rainfall_sequence: [T_in, H, W] — recent T_in hours of rainfall
            prev_flood_depth:  [H, W] — last known flood depth (0 if unknown)

        Returns:
            tensor: float32 [1, T_in, 8, H, W] ready for WeatherTwin.forward()
        """
        if not HAS_TORCH:
            raise ImportError("torch required for inference")

        T, H, W = rainfall_sequence.shape
        if T < self.T_in:
            # Pad with zeros if not enough history
            pad = np.zeros((self.T_in - T, H, W), dtype=np.float32)
            rainfall_sequence = np.concatenate([pad, rainfall_sequence], axis=0)
        elif T > self.T_in:
            rainfall_sequence = rainfall_sequence[-self.T_in:]

        if prev_flood_depth is None:
            prev_flood_depth = np.zeros((H, W), dtype=np.float32)

        frames = []
        for t in range(self.T_in):
            rain_norm = np.log1p(np.clip(rainfall_sequence[t], 0, None)) / self.RAINFALL_LOG_MAX
            rain_norm = rain_norm.astype(np.float32)

            prev_norm = np.clip(prev_flood_depth / self.MAX_FLOOD_DEPTH_M, 0, 1).astype(np.float32)

            # [8, H, W] = rainfall(1) + terrain(6) + prev_depth(1)
            frame = np.concatenate([
                rain_norm[np.newaxis],    # [1, H, W]
                self.terrain,             # [6, H, W]
                prev_norm[np.newaxis],    # [1, H, W]
            ], axis=0)
            frames.append(frame)

        # Stack: [T_in, 8, H, W] → unsqueeze batch: [1, T_in, 8, H, W]
        arr = np.stack(frames, axis=0)
        return torch.from_numpy(arr).unsqueeze(0).float()

    def interpolate_gauges_to_grid(
        self,
        gauge_lats:  np.ndarray,    # [N_gauges]
        gauge_lons:  np.ndarray,    # [N_gauges]
        gauge_vals:  np.ndarray,    # [T_in, N_gauges] mm/h
        grid_lats:   np.ndarray,    # [H]
        grid_lons:   np.ndarray,    # [W]
    ) -> np.ndarray:
        """
        IDW interpolation from sparse gauge observations to the 50m grid.

        Args:
            gauge_lats/lons: Gauge coordinates
            gauge_vals:      [T_in, N_gauges] rainfall observations
            grid_lats/lons:  Target grid coordinates

        Returns:
            grid_rain: [T_in, H, W] interpolated rainfall
        """
        T_in, N = gauge_vals.shape
        H, W    = len(grid_lats), len(grid_lons)

        lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
        grid_coords  = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        gauge_coords = np.column_stack([gauge_lats, gauge_lons])

        # IDW: weight = 1 / dist^2
        result = np.zeros((T_in, H, W), dtype=np.float32)
        for t in range(T_in):
            vals   = gauge_vals[t]
            interp = np.zeros(len(grid_coords), dtype=np.float32)
            for i, gc in enumerate(grid_coords):
                dists = np.linalg.norm(gauge_coords - gc, axis=1)
                dists = np.where(dists < 1e-10, 1e-10, dists)
                w     = 1.0 / dists ** 2
                w    /= w.sum()
                interp[i] = (w * vals).sum()
            result[t] = interp.reshape(H, W)

        return result.clip(0)

    # ── Synthetic terrain (for demo) ──────────────────────────────────────

    @staticmethod
    def _synthetic_terrain(H: int, W: int) -> np.ndarray:
        """Synthetic [6, H, W] terrain features for demo."""
        rng  = np.random.default_rng(42)
        y, x = np.mgrid[0:H, 0:W]
        cx, cy = W//2, H//2

        elev  = (5.0 + 4*np.exp(-((x-cx)**2+(y-cy)**2)/(2*20**2))
                 - 3*np.exp(-((x-W*0.15)**2)/(2*8**2))
                 + rng.normal(0,0.2,(H,W))).clip(0)
        elev_norm = ((elev - elev.mean()) / (elev.std()+1e-6)).astype(np.float32)

        dy   = np.pad(elev, ((1,1),(0,0)), "edge")
        dx   = np.pad(elev, ((0,0),(1,1)), "edge")
        slope = np.degrees(np.arctan(np.sqrt(
            ((dx[:,2:]-dx[:,:-2])/(2*50))**2 +
            ((dy[2:,:]-dy[:-2,:])/(2*50))**2
        ))).clip(0, 45) / 45.0

        flow  = (1 - elev_norm/2).clip(0,1).astype(np.float32)

        drain = np.zeros((H,W), dtype=np.float32)
        for cx_ in range(W//8, W, W//8):
            for c in range(max(0,cx_-2), min(W,cx_+2)):
                drain[:,c] = np.exp(-abs(c-cx_)/2.0)

        imperv = np.clip(0.78 - 0.45*np.sqrt((x-cx)**2+(y-cy)**2)/max(cx,cy)
                         + rng.normal(0,0.04,(H,W)), 0.05, 0.95).astype(np.float32)

        n_min, n_max = 0.013, 0.10
        manning_n = (0.035 - n_min) / (n_max - n_min) * np.ones((H,W), dtype=np.float32)

        return np.stack([
            elev_norm, slope.astype(np.float32), flow,
            drain, imperv, manning_n,
        ], axis=0)   # [6, H, W]

    @staticmethod
    def _synthetic_elevation(H: int, W: int) -> np.ndarray:
        y, x = np.mgrid[0:H, 0:W]
        cx, cy = W//2, H//2
        return (5.0 + 4*np.exp(-((x-cx)**2+(y-cy)**2)/(2*20**2))
                - 3*np.exp(-((x-W*0.15)**2)/(2*8**2))).clip(0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Post-processor
# ═══════════════════════════════════════════════════════════════════════════

class OutputPostprocessor:
    """
    Converts raw model output tensors into alerts and metadata.

    Args:
        thresh_watch:   Depth threshold for WATCH   (m)
        thresh_warning: Depth threshold for WARNING (m)
        thresh_danger:  Depth threshold for DANGER  (m)
        min_cell_frac:  Min fraction of ward cells to trigger ward alert
    """

    def __init__(
        self,
        thresh_watch:   float = 0.15,
        thresh_warning: float = 0.30,
        thresh_danger:  float = 0.60,
        min_cell_frac:  float = 0.05,
    ):
        self.thresh_watch   = thresh_watch
        self.thresh_warning = thresh_warning
        self.thresh_danger  = thresh_danger
        self.min_cell_frac  = min_cell_frac

    def depth_to_alert_map(self, depth: np.ndarray) -> np.ndarray:
        """
        Convert depth map [H, W] → alert level map [H, W] uint8.
        Uses maximum depth across forecast horizons.
        """
        alert             = np.zeros_like(depth, dtype=np.uint8)
        alert[depth >= self.thresh_watch]   = ALERT_WATCH
        alert[depth >= self.thresh_warning] = ALERT_WARNING
        alert[depth >= self.thresh_danger]  = ALERT_DANGER
        return alert

    def compute_ward_alerts(
        self,
        alert_map: np.ndarray,         # [H, W]
        ward_mask: Optional[np.ndarray] = None,  # [H, W] int, ward ID per cell
    ) -> Dict:
        """
        Aggregate cell-level alerts to ward level.

        A ward is elevated to a level when >= min_cell_frac of its cells
        exceed that threshold.

        Args:
            alert_map:  Per-cell alert level [H, W]
            ward_mask:  Ward ID per cell (None = single ward)

        Returns:
            {ward_id: {level, name, color, emoji, flooded_pct, action}}
        """
        if ward_mask is None:
            flooded = float((alert_map > ALERT_DRY).mean())
            level   = int(alert_map.max()) if flooded >= self.min_cell_frac else ALERT_DRY
            return {
                "all": self._ward_info(level, flooded)
            }

        ward_ids = np.unique(ward_mask)
        result   = {}
        for wid in ward_ids:
            mask  = ward_mask == wid
            if mask.sum() == 0:
                continue
            cells   = alert_map[mask]
            level   = ALERT_DRY
            for lvl in [ALERT_DANGER, ALERT_WARNING, ALERT_WATCH]:
                if (cells >= lvl).mean() >= self.min_cell_frac:
                    level = lvl
                    break
            flooded = float((cells > ALERT_DRY).mean())
            result[str(wid)] = self._ward_info(level, flooded)

        return result

    def _ward_info(self, level: int, flooded_frac: float) -> dict:
        return {
            "level":       level,
            "name":        ALERT_NAMES[level],
            "color":       ALERT_COLORS[level],
            "emoji":       ALERT_EMOJI[level],
            "flooded_pct": round(flooded_frac * 100, 1),
            "action":      ALERT_ACTIONS[level],
        }

    def process(
        self,
        pred_tensor:  "torch.Tensor",        # [1, T_out, 1, H, W]
        unc_tensor:   Optional["torch.Tensor"],
        ward_mask:    Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict]:
        """
        Full post-processing pipeline.

        Returns:
            depth_maps:  [T_out, H, W] float32
            uncertainty: [T_out, H, W] float32 or None
            alert_map:   [H, W] uint8
            ward_alerts: dict
        """
        depth_np = pred_tensor[0, :, 0, :, :].cpu().numpy()  # [T_out, H, W]
        unc_np   = (unc_tensor[0, :, 0, :, :].cpu().numpy()
                    if unc_tensor is not None else None)

        # Alert on worst-case depth across all forecast horizons
        max_depth = depth_np.max(axis=0)   # [H, W]
        alert_map = self.depth_to_alert_map(max_depth)
        ward_alerts = self.compute_ward_alerts(alert_map, ward_mask)

        return depth_np, unc_np, alert_map, ward_alerts


# ═══════════════════════════════════════════════════════════════════════════
# Inference Engine (singleton)
# ═══════════════════════════════════════════════════════════════════════════

class FloodInferenceEngine:
    """
    Production inference engine with singleton pattern.

    The model is loaded once and reused across all prediction calls.
    Thread-safe for single-process Streamlit / FastAPI use.

    Args:
        checkpoint_path: Path to .pth checkpoint
        config:          Full config dict
        ward_mask:       [H, W] int ward IDs (None = no ward aggregation)
        device:          "auto" | "cpu" | "cuda" | "mps"
    """

    _instance: Optional["FloodInferenceEngine"] = None

    def __init__(
        self,
        checkpoint_path: str,
        config:          dict,
        ward_mask:       Optional[np.ndarray] = None,
        device:          str = "auto",
    ):
        self.config          = config
        self.ward_mask       = ward_mask
        self.checkpoint_path = checkpoint_path

        cfg_alerts = config.get("alerts", {}).get("thresholds", {})
        cfg_app    = config.get("app",    {})

        self.postproc = OutputPostprocessor(
            thresh_watch   = cfg_alerts.get("watch",   0.15),
            thresh_warning = cfg_alerts.get("warning", 0.30),
            thresh_danger  = cfg_alerts.get("danger",  0.60),
            min_cell_frac  = config.get("alerts", {}).get("min_cell_fraction", 0.05),
        )

        self.T_in  = config.get("model", {}).get("input_steps",  6)
        self.T_out = config.get("model", {}).get("output_steps", 3)

        # Grid dimensions from meta
        H, W = self._get_grid_dims(config)
        self.H, self.W = H, W

        # Load terrain features
        proc_dir   = Path(config.get("data", {}).get("processed_dir", "data/processed"))
        city       = config.get("project", {}).get("city", "kolkata")
        feat_path  = proc_dir / city / "grid_features.npy"
        elev_path  = proc_dir / city / "elevation_raw.npy"

        grid_features = np.load(feat_path).astype(np.float32) if feat_path.exists() else None
        elevation_raw = np.load(elev_path).astype(np.float32) if elev_path.exists() else None

        self.preprocessor = InputPreprocessor(
            grid_features = grid_features,
            elevation_raw = elevation_raw,
            T_in = self.T_in,
            H    = H,
            W    = W,
        )
        self.elevation_raw = elevation_raw

        # Load model
        self.model  = None
        self.device = None
        self._load_model(checkpoint_path, device)

    def _get_grid_dims(self, config: dict) -> Tuple[int, int]:
        """Read H, W from processed grid metadata."""
        proc_dir  = Path(config.get("data", {}).get("processed_dir", "data/processed"))
        city      = config.get("project", {}).get("city", "kolkata")
        meta_path = proc_dir / city / "grid_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return meta["shape"][0], meta["shape"][1]
        return 80, 80   # default demo size

    def _load_model(self, checkpoint_path: str, device: str):
        """Load WeatherTwin from checkpoint or build demo model."""
        if not HAS_TORCH:
            logger.warning("torch unavailable — running in synthetic-only mode")
            return

        from models.weather_twin import WeatherTwin, build_weather_twin

        if device == "auto":
            self.device = WeatherTwin.auto_device()
        else:
            self.device = torch.device(device)

        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            try:
                self.model = WeatherTwin.load(
                    str(ckpt), config=self.config, device=str(self.device)
                )
                logger.info(f"Model loaded from {ckpt} on {self.device}")
            except Exception as e:
                logger.error(f"Checkpoint load failed: {e} — building untrained model")
                self.model = build_weather_twin(self.config).to(self.device)
        else:
            logger.warning(f"Checkpoint not found: {ckpt} — building untrained demo model")
            self.model = build_weather_twin(self.config).to(self.device)

        self.model.eval()

    # ── Singleton access ──────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        checkpoint_path: str,
        config:          dict,
        ward_mask:       Optional[np.ndarray] = None,
        device:          str = "auto",
    ) -> "FloodInferenceEngine":
        """
        Return the singleton instance, creating it on first call.
        Subsequent calls return the cached instance — model is not reloaded.
        """
        if cls._instance is None:
            cls._instance = cls(checkpoint_path, config, ward_mask, device)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Force reload of model on next get_instance() call."""
        cls._instance = None

    # ── Main prediction methods ───────────────────────────────────────────

    def predict(
        self,
        rainfall_sequence:  np.ndarray,              # [T_in, H, W] mm/h
        prev_flood_depth:   Optional[np.ndarray] = None,  # [H, W] metres
        return_uncertainty: bool = False,
        mc_passes:          int  = 20,
    ) -> InferenceResult:
        """
        Run inference given a rainfall sequence.

        Args:
            rainfall_sequence:  [T_in, H, W] — T_in hours of rainfall in mm/h
            prev_flood_depth:   [H, W] — most recent flood depth (0 if unknown)
            return_uncertainty: Run MC Dropout for per-cell uncertainty
            mc_passes:          Number of MC Dropout passes

        Returns:
            InferenceResult with depth maps, alerts, ward summaries
        """
        t0 = time.perf_counter()

        if self.model is None or not HAS_TORCH:
            return self._synthetic_fallback(rainfall_sequence)

        with torch.no_grad():
            # Preprocess → [1, T_in, 8, H, W]
            input_tensor = self.preprocessor.build_input_tensor(
                rainfall_sequence, prev_flood_depth
            ).to(self.device)

            # Forward pass
            if return_uncertainty:
                pred_t, unc_t = self.model.convlstm.predict_with_uncertainty(
                    input_tensor, n_passes=mc_passes
                )
            else:
                pred_t = self.model.convlstm(input_tensor)   # [1, T_out, 1, H, W]
                unc_t  = None

        # Post-process
        depth_np, unc_np, alert_map, ward_alerts = self.postproc.process(
            pred_t, unc_t, self.ward_mask
        )

        inference_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            depth_maps   = depth_np,
            uncertainty  = unc_np,
            alert_map    = alert_map,
            ward_alerts  = ward_alerts,
            inference_ms = inference_ms,
            timestamp    = datetime.utcnow().isoformat() + "Z",
            thresholds   = {
                "watch":   self.postproc.thresh_watch,
                "warning": self.postproc.thresh_warning,
                "danger":  self.postproc.thresh_danger,
            },
            T_out = self.T_out,
        )

    def predict_from_scalar_rainfall(
        self,
        rainfall_mm_per_hour: float,
        duration_hours:       int   = 3,
        prev_flood_depth:     Optional[np.ndarray] = None,
    ) -> InferenceResult:
        """
        Convenience method: uniform rainfall over the whole grid.
        Useful for the Streamlit demo slider.

        Args:
            rainfall_mm_per_hour: Single number, applied uniformly
            duration_hours:       Repeat this rainfall for N hours
            prev_flood_depth:     [H, W] or None
        """
        H, W          = self.H, self.W
        rain_grid     = np.full((H, W), rainfall_mm_per_hour, dtype=np.float32)
        rain_sequence = np.stack([rain_grid] * self.T_in, axis=0)   # [T_in, H, W]
        return self.predict(rain_sequence, prev_flood_depth)

    def predict_from_gauges(
        self,
        gauge_lats:  np.ndarray,    # [N]
        gauge_lons:  np.ndarray,    # [N]
        gauge_vals:  np.ndarray,    # [T_in, N] mm/h
        grid_lats:   np.ndarray,    # [H]
        grid_lons:   np.ndarray,    # [W]
        prev_flood_depth: Optional[np.ndarray] = None,
    ) -> InferenceResult:
        """
        Predict from sparse rain gauge observations.
        Internally runs IDW interpolation to the 50m grid.
        """
        rain_grid = self.preprocessor.interpolate_gauges_to_grid(
            gauge_lats, gauge_lons, gauge_vals, grid_lats, grid_lons
        )
        return self.predict(rain_grid, prev_flood_depth)

    # ── Synthetic fallback ────────────────────────────────────────────────

    def _synthetic_fallback(
        self,
        rainfall_sequence: np.ndarray,
    ) -> InferenceResult:
        """
        Physics-inspired synthetic prediction when model unavailable.
        Used for demo when torch/checkpoint missing.
        """
        t0 = time.perf_counter()

        H, W = self.H, self.W
        terrain = self.preprocessor.terrain   # [6, H, W]
        imperv  = terrain[4]                  # impervious fraction
        elev    = self.preprocessor.elevation_raw

        # Cumulative rainfall over input window
        cum_rain = rainfall_sequence.mean(axis=0).clip(0)   # [H, W]

        # Simple runoff
        C        = 0.003 * (cum_rain.mean().clip(0.1) ** 0.7)
        base_dep = C * cum_rain * imperv

        if elev is not None:
            elev_inv = (elev.max() - elev) / (elev.max() - elev.min() + 1e-6)
            base_dep = base_dep * (0.35 + 0.65 * elev_inv)

        try:
            from scipy.ndimage import gaussian_filter
            base_dep = gaussian_filter(base_dep.astype(np.float64), sigma=2.5).astype(np.float32)
        except ImportError:
            pass

        base_dep = base_dep.clip(0, 3.0)
        depths   = np.stack([base_dep*0.7, base_dep*1.0, base_dep*0.85], axis=0)

        max_depth   = depths.max(axis=0)
        alert_map   = self.postproc.depth_to_alert_map(max_depth)
        ward_alerts = self.postproc.compute_ward_alerts(alert_map, self.ward_mask)

        ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            depth_maps   = depths,
            uncertainty  = None,
            alert_map    = alert_map,
            ward_alerts  = ward_alerts,
            inference_ms = ms,
            timestamp    = datetime.utcnow().isoformat() + "Z",
            thresholds   = {
                "watch":   self.postproc.thresh_watch,
                "warning": self.postproc.thresh_warning,
                "danger":  self.postproc.thresh_danger,
            },
            T_out = self.T_out,
        )

    # ── Health check ──────────────────────────────────────────────────────

    def health(self) -> dict:
        """Return engine status for /health endpoint."""
        return {
            "status":       "ok",
            "model_loaded": self.model is not None,
            "checkpoint":   self.checkpoint_path,
            "device":       str(self.device) if self.device else "none",
            "grid_shape":   [self.H, self.W],
            "T_in":         self.T_in,
            "T_out":        self.T_out,
        }


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI app (optional — only runs if fastapi installed)
# ═══════════════════════════════════════════════════════════════════════════

def _build_api():
    """Build FastAPI app. Called lazily so import doesn't fail if fastapi missing."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(
        title       = "Neural Weather Twin — Flood Prediction API",
        description = "Hyperlocal flood forecasting for Indian monsoon cities",
        version     = "1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],
        allow_methods  = ["*"],
        allow_headers  = ["*"],
    )

    # Load config + engine on startup
    config = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))
    ckpt   = Path("checkpoints/best_model.pth")
    engine = FloodInferenceEngine.get_instance(str(ckpt), config)

    class RainfallRequest(BaseModel):
        rainfall_uniform_mm: float = 0.0
        duration_hours:      int   = 3

    class GaugeRequest(BaseModel):
        gauge_lats:  List[float]
        gauge_lons:  List[float]
        gauge_vals:  List[List[float]]   # [T_in][N_gauges]

    @app.get("/health")
    def health():
        return engine.health()

    @app.get("/config")
    def get_config():
        return {
            "grid": config.get("grid", {}),
            "model": {
                "input_steps":  config["model"]["input_steps"],
                "output_steps": config["model"]["output_steps"],
            },
            "alerts": config.get("alerts", {}),
        }

    @app.post("/predict/uniform")
    def predict_uniform(req: RainfallRequest):
        """Predict from uniform rainfall (demo endpoint)."""
        result = engine.predict_from_scalar_rainfall(
            req.rainfall_uniform_mm,
            req.duration_hours,
        )
        return result.to_dict()

    @app.post("/predict/gauges")
    def predict_gauges(req: GaugeRequest):
        """Predict from sparse rain gauge observations."""
        config_city  = config.get("project", {}).get("city", "kolkata")
        city_cfg     = config.get("cities", {}).get(config_city, {})
        grid_lats    = np.linspace(
            city_cfg.get("lat_max", 22.65),
            city_cfg.get("lat_min", 22.45),
            engine.H,
        )
        grid_lons    = np.linspace(
            city_cfg.get("lon_min", 88.25),
            city_cfg.get("lon_max", 88.45),
            engine.W,
        )
        result = engine.predict_from_gauges(
            gauge_lats  = np.array(req.gauge_lats),
            gauge_lons  = np.array(req.gauge_lons),
            gauge_vals  = np.array(req.gauge_vals),
            grid_lats   = grid_lats,
            grid_lons   = grid_lons,
        )
        return result.to_dict()

    return app


# ── FastAPI entry point ────────────────────────────────────────────────────
try:
    api = _build_api()
except Exception:
    api = None   # fastapi not installed — API disabled


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    ok, fail = [], []
    def check(cond, msg):
        (ok if cond else fail).append(f"{'✓' if cond else '✗'}  {msg}")

    print("=" * 60)
    print("  app/inference.py — smoke test")
    print("=" * 60)

    config = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))

    # ── InputPreprocessor ──────────────────────────────────────────────────
    prep = InputPreprocessor(H=40, W=40, T_in=6)
    rain = np.random.rand(6, 40, 40).astype(np.float32) * 30
    if HAS_TORCH:
        t = prep.build_input_tensor(rain)
        check(t.shape == (1, 6, 8, 40, 40), f"Input tensor shape: {tuple(t.shape)}")
        check(t.dtype == torch.float32,       "Input tensor dtype float32")
        check(t[:, :, 0].min() >= 0,          "Rainfall channel ≥ 0 after normalisation")
        check(t[:, :, 7].min() >= 0,          "Prev depth channel ≥ 0")

    # Gauge interpolation
    g_lats  = np.array([22.47, 22.55, 22.63])
    g_lons  = np.array([88.28, 88.35, 88.42])
    g_vals  = np.random.rand(6, 3).astype(np.float32) * 20
    gl      = np.linspace(22.65, 22.45, 40)
    glo     = np.linspace(88.25, 88.45, 40)
    grid_r  = prep.interpolate_gauges_to_grid(g_lats, g_lons, g_vals, gl, glo)
    check(grid_r.shape == (6, 40, 40), f"IDW interpolation shape: {grid_r.shape}")
    check(grid_r.min() >= 0,           "IDW output non-negative")

    # ── OutputPostprocessor ────────────────────────────────────────────────
    post   = OutputPostprocessor()
    depth  = np.random.rand(40, 40).astype(np.float32) * 0.8
    alert  = post.depth_to_alert_map(depth)
    check(alert.shape == (40, 40), "Alert map shape correct")
    check(alert.min() >= 0 and alert.max() <= 3, "Alert levels in [0,3]")
    check(alert.dtype == np.uint8, "Alert map dtype uint8")

    wards  = post.compute_ward_alerts(alert)
    check("all" in wards,   "Ward alerts: 'all' key present (no mask)")
    check("level" in wards["all"], "Ward alert has 'level' key")
    check("action" in wards["all"], "Ward alert has 'action' key")

    # Ward mask
    wmask  = np.zeros((40, 40), dtype=np.int32)
    wmask[:20, :20] = 1; wmask[:20, 20:] = 2
    wmask[20:, :20] = 3; wmask[20:, 20:] = 4
    wards2 = post.compute_ward_alerts(alert, wmask)
    check(len(wards2) == 4, f"4 wards in result: {len(wards2)}")

    # ── InferenceEngine (synthetic fallback) ──────────────────────────────
    engine = FloodInferenceEngine(
        checkpoint_path = "checkpoints/best_model.pth",
        config          = config,
    )
    check(engine.H > 0 and engine.W > 0, f"Grid dims: {engine.H}×{engine.W}")

    result = engine.predict_from_scalar_rainfall(50.0)
    check(isinstance(result, InferenceResult), "predict_from_scalar returns InferenceResult")
    check(result.depth_maps.shape[0] == 3,     "depth_maps has 3 horizons")
    check(result.max_depth_m >= 0,             "max_depth_m ≥ 0")
    check(result.inference_ms > 0,             "inference_ms > 0")
    check(result.highest_alert in range(4),    "highest_alert in [0,3]")
    check(len(result.ward_alerts) > 0,         "ward_alerts non-empty")
    check("level" in list(result.ward_alerts.values())[0], "ward_alert has level")

    # FIX: rain was (6,40,40) but engine grid is 80×80 — use engine dims
    rain_demo = np.random.rand(6, engine.H, engine.W).astype(np.float32) * 30
    rain_result = engine.predict(rain_demo)
    check(rain_result.depth_maps.shape[-2:] == (engine.H, engine.W),
          "Spatial dims match grid")

    d  = result.to_dict()
    check("highest_alert"      in d, "to_dict has highest_alert")
    check("recommended_action" in d, "to_dict has recommended_action")
    check("ward_alerts"        in d, "to_dict has ward_alerts")
    check("flooded_pct"        in d, "to_dict has flooded_pct")

    # Singleton
    e2 = FloodInferenceEngine.get_instance("checkpoints/best_model.pth", config)
    check(e2 is not engine, "get_instance creates new (engine not singleton here)")
    FloodInferenceEngine.reset_instance()

    # Health
    h = engine.health()
    check("status" in h and h["status"] == "ok", "health() returns ok")

    print(f"\n{'='*55}")
    print(f"  PASSED: {len(ok)}   FAILED: {len(fail)}")
    print(f"{'='*55}\n")
    if fail:
        print("FAILURES:")
        for f in fail: print(f"  {f}")
        print()
    print("PASSED:")
    for o in ok: print(f"  {o}")