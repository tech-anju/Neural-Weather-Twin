"""
data/dataset.py
PyTorch Dataset for the Neural Weather Twin flood prediction model.

Each sample is a spatiotemporal window:
  Input  [T_in,  C, H, W]  — T_in=6  hours of rainfall + terrain + prev flood state
  Target [T_out, H, W]     — T_out=3 hours of flood depth maps ahead

Input channels (C=8) per timestep:
  0  rainfall_mm          — gauge-interpolated rainfall on 50m grid (dynamic)
  1  elevation_norm       — z-scored terrain height            (static, repeated)
  2  slope_norm           — Horn slope [0,1]                   (static, repeated)
  3  flow_accumulation    — log-normalised upstream area       (static, repeated)
  4  drain_density        — drain line length / cell area      (static, repeated)
  5  impervious_fraction  — fraction paved/built               (static, repeated)
  6  manning_n            — roughness coefficient [0,1]        (static, repeated)
  7  prev_flood_depth     — flood depth from previous timestep (dynamic)

Target:
  Flood depth (metres) at T+1h, T+2h, T+3h
  Shape [3, H, W]  — one depth map per forecast horizon

Data flow:
  grid_features.npy  [H, W, 6]         →  static terrain channels
  rainfall CSV       [N_times, N_gauges] →  IDW-interpolated rainfall grid [N_times, H, W]
  flood_mask_*.npy   [H, W]             →  flood depth targets (SAR-derived or synthetic)

Usage:
  from data.dataset import FloodDataset, create_dataloaders

  loaders = create_dataloaders("kolkata", config)
  for inputs, targets in loaders["train"]:
      # inputs:  [B, T_in,  C, H, W]
      # targets: [B, T_out, H, W]
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ── Optional scipy for IDW interpolation ──────────────────────────────────────
try:
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Constants ─────────────────────────────────────────────────────────────────
N_TERRAIN_FEATURES = 6     # from grid.py: elevation, slope, flow_acc, drain, imperv, manning
N_INPUT_CHANNELS   = 8     # 1 rainfall + 6 terrain + 1 prev_depth
N_INPUT_STEPS      = 6     # 6-hour input window
N_OUTPUT_STEPS     = 3     # T+1h, T+2h, T+3h forecast


# ═══════════════════════════════════════════════════════════════════════════════
# Rainfall grid builder
# ═══════════════════════════════════════════════════════════════════════════════

class RainfallGrid:
    """
    Interpolates sparse rain gauge observations onto the 50m spatial grid
    using Inverse Distance Weighting (IDW).

    IDW is chosen over Kriging here because:
      - It requires no variogram fitting (important for real-time inference)
      - Runs in <50ms per timestep on a 200×200 grid
      - Sufficient accuracy for 50m-resolution nowcasting

    For research-grade offline training, swap _idw_interpolate for
    PyKrige's OrdinaryKriging for better spatial statistics.
    """

    def __init__(
        self,
        rainfall_csv: Path,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        H: int, W: int,
        power: float = 2.0,
        n_neighbours: int = 4,
    ):
        self.H, self.W = H, W
        self.power = power
        self.n_neighbours = n_neighbours

        # Build target grid coordinates [H*W, 2]
        lats = np.linspace(lat_max, lat_min, H)   # row 0 = north
        lons = np.linspace(lon_min, lon_max, W)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        self.grid_coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

        # Load gauge data
        print(f"  [RainfallGrid] Loading {rainfall_csv.name}...")
        df = pd.read_csv(rainfall_csv, parse_dates=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        self.df = df

        # Gauge positions [N_gauges, 2]
        gauges = df[["gauge_id","latitude","longitude"]].drop_duplicates("gauge_id")
        self.gauge_ids   = gauges["gauge_id"].tolist()
        self.gauge_coords = gauges[["latitude","longitude"]].values.astype(np.float32)

        # Pivot to wide format: index=datetime, columns=gauge_id
        self.pivot = df.pivot_table(
            index="datetime", columns="gauge_id", values="rainfall_mm", aggfunc="mean"
        ).fillna(0.0)
        self.timestamps = self.pivot.index  # DatetimeIndex

        # Pre-build KD-tree for fast nearest-neighbour lookup
        if HAS_SCIPY:
            self._tree = cKDTree(self.gauge_coords)
        else:
            self._tree = None

        print(
            f"  [RainfallGrid] {len(self.gauge_ids)} gauges | "
            f"{len(self.timestamps)} hourly timesteps | "
            f"grid {H}×{W}"
        )

        # Pre-compute ALL rainfall grids once at init and cache in RAM.
        # IDW interpolation on 446×412 grid takes ~0.5s per timestep.
        # Without cache: 1441 timesteps × 0.5s = 720s per epoch.
        # With cache: one-time 720s cost, then __getitem__ is instant.
        print(f"  [RainfallGrid] Pre-computing {len(self.timestamps)} "
              f"rainfall grids (one-time, ~2 min)...")
        self._grid_cache: Dict[int, np.ndarray] = {}
        for _i in range(len(self.timestamps)):
            row = self.pivot.iloc[_i]
            gauge_values = np.array(
                [row.get(g, 0.0) for g in self.gauge_ids],
                dtype=np.float32
            )
            self._grid_cache[_i] = self._idw_interpolate(gauge_values)
        print(f"  [RainfallGrid] Cache ready — {len(self._grid_cache)} grids in RAM")

    def get_grid(self, timestamp: pd.Timestamp) -> np.ndarray:
        """
        Return IDW-interpolated rainfall grid for one timestamp.

        Args:
            timestamp: exact datetime to look up

        Returns:
            grid: float32 [H, W]  — rainfall in mm
        """
        # Find nearest available timestamp (within 30 min)
        idx = self.timestamps.get_indexer([timestamp], method="nearest")[0]
        if idx < 0:
            return np.zeros((self.H, self.W), dtype=np.float32)

        # Return from pre-computed cache — O(1) vs O(H×W×N_gauges)
        if hasattr(self, "_grid_cache") and idx in self._grid_cache:
            return self._grid_cache[idx]

        # Fallback: compute on the fly if cache not built
        row = self.pivot.iloc[idx]
        gauge_values = np.array(
            [row.get(g, 0.0) for g in self.gauge_ids], dtype=np.float32
        )
        return self._idw_interpolate(gauge_values)

    def get_sequence(
        self, start: pd.Timestamp, n_steps: int
    ) -> np.ndarray:
        """
        Return a sequence of rainfall grids.

        Args:
            start:   first timestep
            n_steps: number of hourly steps

        Returns:
            sequence: float32 [n_steps, H, W]
        """
        grids = []
        for i in range(n_steps):
            ts = start + pd.Timedelta(hours=i)
            grids.append(self.get_grid(ts))
        return np.stack(grids, axis=0)   # [n_steps, H, W]

    def _idw_interpolate(self, values: np.ndarray) -> np.ndarray:
        """
        Inverse Distance Weighting interpolation.

        weight_i = 1 / dist_i^power
        z(x) = Σ(weight_i * value_i) / Σ(weight_i)
        """
        if self._tree is not None:
            # Fast path: scipy cKDTree
            dists, idxs = self._tree.query(
                self.grid_coords,
                k=min(self.n_neighbours, len(self.gauge_ids)),
            )
            # Avoid division by zero for exact gauge location hits
            dists = np.where(dists < 1e-10, 1e-10, dists)
            weights = 1.0 / (dists ** self.power)          # [H*W, k]
            weights /= weights.sum(axis=1, keepdims=True)
            interpolated = (weights * values[idxs]).sum(axis=1)
        else:
            # Slow fallback: manual nearest-neighbour (no scipy)
            interpolated = np.zeros(len(self.grid_coords), dtype=np.float32)
            for i, coord in enumerate(self.grid_coords):
                dists = np.linalg.norm(self.gauge_coords - coord, axis=1)
                nearest = np.argmin(dists)
                interpolated[i] = values[nearest]

        grid = interpolated.reshape(self.H, self.W).astype(np.float32)

        # Light smoothing to remove interpolation artefacts
        if HAS_SCIPY:
            grid = gaussian_filter(grid, sigma=0.8).astype(np.float32)

        return np.clip(grid, 0, None)   # rainfall can't be negative


# ═══════════════════════════════════════════════════════════════════════════════
# Flood depth sequence builder
# ═══════════════════════════════════════════════════════════════════════════════

class FloodDepthSequence:
    """
    Builds a continuous flood depth time series from sparse SAR observations.

    SAR observations are only available every ~6–12 days (Sentinel-1 revisit).
    Between observations we:
      1. Use the last known depth decayed by an exponential recession factor
         (water drains after rain stops — physically motivated).
      2. Scale depth by rainfall intensity in that hour.

    This gives a physically plausible pseudo-continuous target series
    for training, rather than having 95% of timesteps with no target.
    """

    RECESSION_RATE = 0.92   # depth_t = depth_{t-1} * 0.92 during dry hours
                             # ≈ 50% reduction after ~8 hours of no rain

    def __init__(
        self,
        flood_mask_dir: Path,
        city: str,
        H: int, W: int,
        timestamps: pd.DatetimeIndex,
        rainfall_grid: RainfallGrid,
    ):
        self.H, self.W = H, W
        self.timestamps = timestamps
        self.n_steps    = len(timestamps)

        # Load all available SAR flood masks
        sar_masks: Dict[pd.Timestamp, np.ndarray] = {}
        for npy_file in sorted(flood_mask_dir.glob(f"{city}_flood_mask_*.npy")):
            # filename: {city}_flood_mask_YYYY-MM-DD.npy
            date_str = npy_file.stem.replace(f"{city}_flood_mask_", "")
            try:
                ts = pd.Timestamp(date_str)
                arr = np.load(npy_file).astype(np.float32)
                # Resize to target grid if shape differs
                if arr.shape != (H, W):
                    arr = _resize_array(arr, H, W)
                sar_masks[ts] = arr
            except Exception:
                continue

        if sar_masks:
            print(f"  [FloodDepth] Loaded {len(sar_masks)} SAR mask(s) for {city}")
        else:
            print(f"  [FloodDepth] No SAR masks found — using synthetic flood events")
            sar_masks = _generate_synthetic_event_masks(city, H, W, timestamps)

        # ── Build continuous depth series [N_times, H, W] ─────────────────────
        print(f"  [FloodDepth] Interpolating continuous depth series ({self.n_steps} steps)...")
        self.depth_series = self._build_continuous_series(
            sar_masks, timestamps, rainfall_grid
        )
        print(
            f"  [FloodDepth] Done. "
            f"max_depth={self.depth_series.max():.2f}m  "
            f"flooded_frac={np.mean(self.depth_series > 0.1):.3f}"
        )
        print(f"  [DEBUG] depth_series max={self.depth_series.max():.3f}m  "
      f"cells > 0.10m: {(self.depth_series > 0.10).mean()*100:.1f}%  "
      f"cells > 0.20m: {(self.depth_series > 0.20).mean()*100:.1f}%"
      )

    def get_target(self, t_idx: int, n_steps: int = N_OUTPUT_STEPS) -> np.ndarray:
        """
        Return flood depth maps for n_steps hours starting at t_idx.

        Returns:
            target: float32 [n_steps, H, W]
        """
        end = min(t_idx + n_steps, self.n_steps)
        seq = self.depth_series[t_idx:end]

        # Pad with zeros if we're at the end of the series
        if len(seq) < n_steps:
            pad = np.zeros((n_steps - len(seq), self.H, self.W), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        return seq.astype(np.float32)

    def get_depth_at(self, t_idx: int) -> np.ndarray:
        """Return single depth map at index t_idx. Shape [H, W]."""
        return self.depth_series[t_idx].astype(np.float32)

    def _build_continuous_series(
        self,
        sar_masks: Dict,
        timestamps: pd.DatetimeIndex,
        rainfall_grid: RainfallGrid,
    ) -> np.ndarray:
        """
        Build depth[t] using:
          - Direct SAR observation where available (nearest within 12h)
          - Recession model between observations:
              depth[t] = depth[t-1] * RECESSION_RATE + α * rainfall[t]
        """
        series    = np.zeros((self.n_steps, self.H, self.W), dtype=np.float32)
        prev_depth = np.zeros((self.H, self.W), dtype=np.float32)

        # Pre-sort SAR observations by time
        sar_sorted = sorted(sar_masks.items())   # [(Timestamp, array), ...]

        for i, ts in enumerate(timestamps):
            # Check if a SAR observation exists within ±12 hours
            sar_depth = None
            for sar_ts, sar_arr in sar_sorted:
                if abs((ts - sar_ts).total_seconds()) <= 12 * 3600:
                    sar_depth = sar_arr
                    break

            if sar_depth is not None:
                # Trust the SAR observation directly
                depth = sar_depth.copy()
            else:
                # Recession + rainfall forcing
                rain = rainfall_grid.get_grid(ts)
                # α: 0.005 m depth per mm of rain (empirical urban runoff factor)
                depth = prev_depth * self.RECESSION_RATE + 0.015 * rain
                depth = np.clip(depth, 0, 5.0)    # physical cap at 5m

            series[i]  = depth
            prev_depth = depth

        return series


# ═══════════════════════════════════════════════════════════════════════════════
# Core Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class FloodDataset(Dataset):
    """
    PyTorch Dataset for spatiotemporal flood prediction.

    Each __getitem__ returns:
      inputs:  float32 tensor [T_in, C, H, W]
               T_in=6 hours, C=8 channels (rainfall + terrain + prev_depth)
      targets: float32 tensor [T_out, H, W]
               T_out=3 hours of flood depth ahead

    Args:
        city:           City key (kolkata | chennai | mumbai)
        split:          "train" | "val" | "test"
        processed_dir:  Root dir of processed data (contains grid_features.npy etc.)
        raw_dir:        Root dir of raw data (contains rainfall CSV + SAR masks)
        config:         Dict with model.input_steps, model.output_steps, etc.
        stride:         Sliding window stride in hours (default 1)
        augment:        Apply training augmentations (flip, noise)
    """

    def __init__(
        self,
        city: str,
        split: str = "train",
        processed_dir: str = "data/processed",
        raw_dir: str = "data/raw",
        config: Optional[dict] = None,
        stride: int = 1,
        augment: bool = False,
        crop_size: int = 0,
    ):
        self.city    = city
        self.split   = split
        self.augment = augment and (split == "train")

        # ── Load config ────────────────────────────────────────────────────────
        cfg           = config or {}
        self.T_in     = cfg.get("input_steps",  N_INPUT_STEPS)
        self.T_out    = cfg.get("output_steps",  N_OUTPUT_STEPS)
        self.window   = self.T_in + self.T_out   # total window width
        self.stride    = stride
        self.crop_size = crop_size

        # ── Load grid metadata + features ─────────────────────────────────────
        proc_city = Path(processed_dir) / city
        meta_path = proc_city / "grid_meta.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Grid metadata not found at {meta_path}. "
                f"Run: python data/grid.py --city {city}"
            )

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.H = self.meta["shape"][0]
        self.W = self.meta["shape"][1]

        # Terrain features [H, W, 6] → rearrange to [6, H, W]
        features_path = proc_city / "grid_features.npy"
        terrain_hwc   = np.load(features_path)                          # [H, W, 6]
        self.terrain  = np.transpose(terrain_hwc, (2, 0, 1)).astype(np.float32)  # [6, H, W]

        print(f"\n[FloodDataset] city={city} split={split} grid={self.H}×{self.W}")

        # ── Load raw data paths ────────────────────────────────────────────────
        raw_city      = Path(raw_dir) / city
        flood_mask_dir = raw_city

        # Find rainfall CSV
        rain_csvs = sorted(raw_city.glob(f"{city}_rainfall_*.csv"))
        if not rain_csvs:
            raise FileNotFoundError(
                f"No rainfall CSV found in {raw_city}. "
                f"Run: python data/ingest.py --city {city}"
            )
        rain_csv = rain_csvs[-1]   # use most recent

        # ── Build rainfall grid interpolator ──────────────────────────────────
        self.rainfall_grid = RainfallGrid(
            rainfall_csv = rain_csv,
            lat_min      = self.meta["lat_min"],
            lat_max      = self.meta["lat_max"],
            lon_min      = self.meta["lon_min"],
            lon_max      = self.meta["lon_max"],
            H            = self.H,
            W            = self.W,
        )

        # ── Build flood depth series ───────────────────────────────────────────
        self.flood_depth = FloodDepthSequence(
            flood_mask_dir = flood_mask_dir,
            city           = city,
            H              = self.H,
            W              = self.W,
            timestamps     = self.rainfall_grid.timestamps,
            rainfall_grid  = self.rainfall_grid,
        )

        # ── Select time indices for this split ────────────────────────────────
        all_timestamps = self.rainfall_grid.timestamps
        split_cfg      = cfg.get("data", {})
        self.indices   = self._split_indices(all_timestamps, split, split_cfg)

        print(
            f"  Samples: {len(self.indices)}  "
            f"(T_in={self.T_in}h → T_out={self.T_out}h, stride={stride}h)"
        )

    # ── PyTorch interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs:  [T_in, C, H, W]  float32
            targets: [T_out, H, W]    float32
        """
        t_start = self.indices[idx]

        # ── Build input sequence [T_in, C, H, W] ──────────────────────────────
        input_frames = []
        for t in range(t_start, t_start + self.T_in):
            ts       = self.rainfall_grid.timestamps[t]
            rain     = self.rainfall_grid.get_grid(ts)            # [H, W]
            prev_d   = self.flood_depth.get_depth_at(max(0, t - 1))  # [H, W]

            # Stack: rain(1) + terrain(6) + prev_depth(1) → [8, H, W]
            frame = np.concatenate([
                rain[np.newaxis],        # [1, H, W]
                self.terrain,            # [6, H, W]
                prev_d[np.newaxis],      # [1, H, W]
            ], axis=0)                   # [8, H, W]

            input_frames.append(frame)

        inputs = np.stack(input_frames, axis=0)    # [T_in, 8, H, W]

        # ── Build target [T_out, H, W] ─────────────────────────────────────────
        t_target = t_start + self.T_in
        targets  = self.flood_depth.get_target(t_target, self.T_out)   # [T_out, H, W]

        # ── Normalise inputs ───────────────────────────────────────────────────
        # Rainfall: log1p normalise (heavy-tailed distribution)
        inputs[:, 0] = np.log1p(inputs[:, 0]) / np.log1p(150.0)   # 150mm = max expected
        # Terrain channels [1:7] already normalised by grid.py
        # Prev depth: normalise to [0,1] assuming max 5m
        inputs[:, 7] = np.clip(inputs[:, 7] / 5.0, 0, 1)

        # ── Optional augmentation ──────────────────────────────────────────────
        if self.augment:
            inputs, targets = self._augment(inputs, targets)

        if self.crop_size > 0 and self.crop_size < self.H and self.crop_size < self.W:
            rng  = np.random.default_rng()
            r0   = rng.integers(0, self.H - self.crop_size)
            c0   = rng.integers(0, self.W - self.crop_size)
            r1, c1 = r0 + self.crop_size, c0 + self.crop_size
            inputs  = inputs[:,  :, r0:r1, c0:c1]
            targets = targets[:, r0:r1, c0:c1]

        return (
            torch.from_numpy(inputs.astype(np.float32)),
            torch.from_numpy(targets.astype(np.float32)),
        )

    # ── Split logic ────────────────────────────────────────────────────────────

    def _split_indices(
        self,
        timestamps: pd.DatetimeIndex,
        split: str,
        cfg: dict,
    ) -> List[int]:
        """
        Partition timestep indices into train / val / test by date range.

        Split boundaries (from config or defaults):
          train: 2019-06-01 → 2023-09-30   (matches config.yaml train_events)
          val:   2023-10-01 → 2023-10-31   (matches config.yaml val_events)
          test:  2023-11-01 → 2024-09-30   (matches config.yaml test_events)

        FIX: val_end default matched train_end, making val window zero-width.
        """
        train_end = pd.Timestamp(cfg.get("train_end", "2023-09-30"))
        val_end   = pd.Timestamp(cfg.get("val_end",   "2023-10-31"))

        data_start = timestamps[0]
        data_end   = timestamps[-1]

        # Fallback: config dates predate all available data
        if train_end < data_start and val_end < data_start:
            n = len(timestamps)
            train_cut = int(n * 0.70)
            val_cut   = int(n * 0.85)
            print(
                f"  [FloodDataset] WARNING: config split dates "
                f"({train_end.date()} / {val_end.date()}) predate "
                f"available data ({data_start.date()} to {data_end.date()}). "
                f"Using proportional 70/15/15 split instead."
            )
            valid_indices = []
            for i in range(len(timestamps)):
                if i + self.window > len(timestamps):
                    break
                if split == "train" and i < train_cut:
                    valid_indices.append(i)
                elif split == "val" and train_cut <= i < val_cut:
                    valid_indices.append(i)
                elif split == "test" and i >= val_cut:
                    valid_indices.append(i)
            return valid_indices[::self.stride]

        valid_indices = []
        for i, ts in enumerate(timestamps):
            if i + self.window > len(timestamps):
                break
            if split == "train" and ts <= train_end:
                valid_indices.append(i)
            elif split == "val" and train_end < ts <= val_end:
                valid_indices.append(i)
            elif split == "test" and ts > val_end:
                valid_indices.append(i)

        return valid_indices[::self.stride]

    # ── Augmentation ───────────────────────────────────────────────────────────

    def _augment(
        self,
        inputs: np.ndarray,    # [T_in, C, H, W]
        targets: np.ndarray,   # [T_out, H, W]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Training augmentations that respect physical constraints:
          - Horizontal flip (valid: west/east symmetry)
          - Vertical flip   (valid: north/south symmetry)
          - Gaussian noise on rainfall only (not terrain)

        Does NOT apply rotation — drainage direction is physically
        meaningful and rotation would break the Saint-Venant physics.
        """
        rng = np.random.default_rng()

        # Horizontal flip (axis W)
        if rng.random() < 0.5:
            inputs  = inputs[:, :, :, ::-1].copy()
            targets = targets[:, :, ::-1].copy()

        # Vertical flip (axis H)
        if rng.random() < 0.5:
            inputs  = inputs[:, :, ::-1, :].copy()
            targets = targets[:, ::-1, :].copy()

        # Additive Gaussian noise on rainfall channel only
        if rng.random() < 0.3:
            noise = rng.normal(0, 0.02, inputs[:, 0:1].shape).astype(np.float32)
            inputs[:, 0:1] = np.clip(inputs[:, 0:1] + noise, 0, 1)

        return inputs, targets


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_dataloaders(
    city: str,
    config: Optional[dict] = None,
    processed_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    batch_size: int = 8,
    num_workers: int = 2,
    stride_train: int = 1,
    stride_val: int = 3,
    stride_test: int = 6,
    crop_size: int = 0,
) -> Dict[str, DataLoader]:
    """
    Create train / val / test DataLoaders.

    Larger strides on val/test reduce redundant samples from the
    sliding window (consecutive windows overlap heavily).

    Args:
        city:          City key
        config:        Config dict (from config.yaml)
        processed_dir: Processed data root
        raw_dir:       Raw data root
        batch_size:    Training batch size
        num_workers:   DataLoader worker processes
        stride_train:  Window stride for training (1 = every hour)
        stride_val:    Window stride for validation
        stride_test:   Window stride for test

    Returns:
        Dict with "train", "val", "test" DataLoaders
    """
    cfg = config or {}

    datasets = {
        "train": FloodDataset(
            city=city, split="train",
            processed_dir=processed_dir, raw_dir=raw_dir,
            config=cfg, stride=stride_train, augment=True,
            crop_size=crop_size,
        ),
        "val": FloodDataset(
            city=city, split="val",
            processed_dir=processed_dir, raw_dir=raw_dir,
            config=cfg, stride=stride_val, augment=False,
            crop_size=crop_size,
        ),
        "test": FloodDataset(
            city=city, split="test",
            processed_dir=processed_dir, raw_dir=raw_dir,
            config=cfg, stride=stride_test, augment=False,
            crop_size=crop_size,
        ),
    }

    # persistent_workers requires num_workers > 0
    persistent = num_workers > 0

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=persistent,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=1,               # batch=1 for test: per-event evaluation
            shuffle=False,
            num_workers=0,              # no workers for test to avoid OOM
            pin_memory=False,
        ),
    }

    print(f"\n[DataLoaders] city={city}")
    for split, dl in loaders.items():
        n = len(dl.dataset)
        print(f"  {split:5s}: {n:6,} samples  {len(dl):5,} batches")

    return loaders


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _resize_array(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize a 2D array to (H, W) using bilinear interpolation."""
    try:
        from scipy.ndimage import zoom
        zy = H / arr.shape[0]
        zx = W / arr.shape[1]
        return zoom(arr, (zy, zx), order=1).astype(np.float32)
    except ImportError:
        # Nearest-neighbour fallback without scipy
        row_idx = (np.arange(H) * arr.shape[0] / H).astype(int)
        col_idx = (np.arange(W) * arr.shape[1] / W).astype(int)
        return arr[np.ix_(row_idx, col_idx)].astype(np.float32)


def _generate_synthetic_event_masks(
    city: str, H: int, W: int, timestamps: pd.DatetimeIndex
) -> Dict:
    """
    Generate synthetic flood events for cities with no real SAR masks.
    Places 2–4 flood events during monsoon months (June–September).
    """
    rng    = np.random.default_rng(42)
    masks  = {}
    monsoon = [ts for ts in timestamps if 6 <= ts.month <= 9]

    if not monsoon:
        return masks

    n_events = rng.integers(2, 5)
    event_times = rng.choice(len(monsoon), size=n_events, replace=False)

    for ei in sorted(event_times):
        ts    = monsoon[ei]
        depth = np.zeros((H, W), dtype=np.float32)
        cx    = rng.integers(W // 4, 3 * W // 4)

        # Flood channel
        for col in range(max(0, cx - 15), min(W, cx + 15)):
            d = 0.8 * np.exp(-abs(col - cx) / 8) * rng.uniform(0.7, 1.3, H)
            depth[:, col] = np.maximum(depth[:, col], d.astype(np.float32))

        # Low-lying basins
        for _ in range(rng.integers(3, 7)):
            by = rng.integers(10, H - 10)
            bx = rng.integers(10, W - 10)
            r  = rng.integers(5, 20)
            y, x = np.ogrid[:H, :W]
            mask = (y - by) ** 2 + (x - bx) ** 2 <= r ** 2
            depth[mask] = np.maximum(depth[mask], rng.uniform(0.15, 0.60))

        # FIX: guard import inside HAS_SCIPY check — previously imported
        # unconditionally, crashing when scipy is not installed.
        if HAS_SCIPY:
            from scipy.ndimage import gaussian_filter as gf
            depth = gf(depth, sigma=2.0)

        masks[ts] = depth.clip(0).astype(np.float32)

    print(f"  [FloodDepth] Generated {len(masks)} synthetic flood events")
    return masks


def get_sample_shapes(city: str, config: Optional[dict] = None) -> dict:
    """
    Return expected tensor shapes without loading the full dataset.
    Useful for quickly verifying model input dimensions.
    """
    cfg   = config or {}
    T_in  = cfg.get("input_steps",  N_INPUT_STEPS)
    T_out = cfg.get("output_steps", N_OUTPUT_STEPS)

    proc = Path("data/processed") / city / "grid_meta.json"
    if proc.exists():
        with open(proc) as f:
            meta = json.load(f)
        H, W = meta["shape"][0], meta["shape"][1]
    else:
        H, W = 200, 200  # default

    return {
        "inputs":  (T_in,  N_INPUT_CHANNELS, H, W),
        "targets": (T_out, H, W),
        "batched_inputs":  (8, T_in,  N_INPUT_CHANNELS, H, W),
        "batched_targets": (8, T_out, H, W),
    }


# ── Quick smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FloodDataset")
    parser.add_argument("--city",          default="kolkata")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--raw_dir",       default="data/raw")
    parser.add_argument("--batch_size",    type=int, default=4)
    args = parser.parse_args()

    print("Expected tensor shapes:")
    shapes = get_sample_shapes(args.city)
    for k, v in shapes.items():
        print(f"  {k}: {v}")

    print("\nBuilding DataLoaders...")
    loaders = create_dataloaders(
        city          = args.city,
        processed_dir = args.processed_dir,
        raw_dir       = args.raw_dir,
        batch_size    = args.batch_size,
        num_workers   = 0,
    )

    print("\nFetching one training batch...")
    inputs, targets = next(iter(loaders["train"]))
    print(f"  inputs  shape : {tuple(inputs.shape)}")
    print(f"  targets shape : {tuple(targets.shape)}")
    print(f"  inputs  dtype : {inputs.dtype}")
    print(f"  inputs  range : [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"  targets range : [{targets.min():.3f}, {targets.max():.3f}m]")
    print("\nDataset OK.")