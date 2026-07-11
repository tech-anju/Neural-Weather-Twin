"""
models/weather_twin.py
Neural Weather Twin — full model assembly.

Combines ConvLSTM encoder-decoder with Physics-Informed loss into a
single class that handles the complete forward pass, loss computation,
inference, checkpointing, and alert generation.

This is the top-level model class used by train.py, evaluate.py,
and app/inference.py.

Architecture summary:
  Input  [B, T_in=6, C=8, H, W]
    ↓
  FloodConvLSTM (encoder-decoder)
    ↓
  Predictions [B, T_out=3, 1, H, W]
    ↓
  PINNWrapper (physics-informed loss during training)
    ↓
  Alert map [B, H, W] — WATCH / WARNING / DANGER per cell

Inference outputs:
  depth_maps:   float32 [T_out, H, W]   — flood depth in metres
  uncertainty:  float32 [T_out, H, W]   — std dev from MC Dropout
  alert_map:    int8    [H, W]           — 0=dry 1=watch 2=warning 3=danger
  alert_wards:  dict    {ward_id: level} — ward-level alert summary

Usage:
  from models.weather_twin import WeatherTwin, build_weather_twin

  # Training
  twin = build_weather_twin(config)
  loss, loss_dict = twin.training_step(inputs, targets, elevation, manning_n)

  # Inference
  result = twin.predict(inputs, elevation, return_uncertainty=True)

  # Save / load
  twin.save(path)
  twin = WeatherTwin.load(path, config)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.convlstm import FloodConvLSTM, build_convlstm
from models.pinn import PINNWrapper, build_pinn


# ── Alert level constants ──────────────────────────────────────────────────────
ALERT_DRY     = 0
ALERT_WATCH   = 1
ALERT_WARNING = 2
ALERT_DANGER  = 3

ALERT_NAMES = {
    ALERT_DRY:     "DRY",
    ALERT_WATCH:   "WATCH",
    ALERT_WARNING: "WARNING",
    ALERT_DANGER:  "DANGER",
}

ALERT_COLORS = {
    ALERT_DRY:     "#4CAF50",   # green
    ALERT_WATCH:   "#FFC107",   # amber
    ALERT_WARNING: "#FF5722",   # deep orange
    ALERT_DANGER:  "#D32F2F",   # red
}


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction result container
# ═══════════════════════════════════════════════════════════════════════════════

class FloodPrediction:
    """
    Container for a single inference result.

    Attributes:
        depth_maps:      float32 [T_out, H, W] — predicted depth (metres)
        uncertainty:     float32 [T_out, H, W] — std dev (metres), or None
        alert_map:       int8    [H, W]         — per-cell alert level (0-3)
        alert_summary:   dict    {level_name: cell_count}
        forecast_times:  list    of strings e.g. ["T+1h","T+2h","T+3h"]
        inference_ms:    float   — wall-clock inference time in milliseconds
        city:            str     — city name
        thresholds:      dict    — alert depth thresholds used
    """

    def __init__(
        self,
        depth_maps:     np.ndarray,
        uncertainty:    Optional[np.ndarray],
        alert_map:      np.ndarray,
        forecast_times: List[str],
        inference_ms:   float,
        city:           str,
        thresholds:     dict,
    ):
        self.depth_maps     = depth_maps
        self.uncertainty    = uncertainty
        self.alert_map      = alert_map
        self.forecast_times = forecast_times
        self.inference_ms   = inference_ms
        self.city           = city
        self.thresholds     = thresholds

        # Summary statistics
        total = alert_map.size
        self.alert_summary = {
            "DRY":     int((alert_map == ALERT_DRY).sum()),
            "WATCH":   int((alert_map == ALERT_WATCH).sum()),
            "WARNING": int((alert_map == ALERT_WARNING).sum()),
            "DANGER":  int((alert_map == ALERT_DANGER).sum()),
            "total_cells": total,
            "flooded_pct": float(100 * (alert_map > ALERT_DRY).sum() / total),
            "max_depth_m": float(depth_maps.max()),
            "mean_depth_m": float(depth_maps[depth_maps > 0.01].mean())
                            if (depth_maps > 0.01).any() else 0.0,
        }

    @property
    def highest_alert(self) -> int:
        """Highest alert level across the entire grid."""
        return int(self.alert_map.max())

    @property
    def highest_alert_name(self) -> str:
        return ALERT_NAMES[self.highest_alert]

    def to_dict(self) -> dict:
        """Serialisable summary for API / JSON response."""
        return {
            "city":           self.city,
            "highest_alert":  self.highest_alert_name,
            "alert_summary":  self.alert_summary,
            "forecast_times": self.forecast_times,
            "inference_ms":   round(self.inference_ms, 1),
            "thresholds":     self.thresholds,
            "max_depth_m":    self.alert_summary["max_depth_m"],
            "flooded_pct":    round(self.alert_summary["flooded_pct"], 2),
        }

    def __repr__(self) -> str:
        return (
            f"FloodPrediction({self.city} | "
            f"alert={self.highest_alert_name} | "
            f"max_depth={self.alert_summary['max_depth_m']:.2f}m | "
            f"flooded={self.alert_summary['flooded_pct']:.1f}% | "
            f"{self.inference_ms:.0f}ms)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Weather Twin
# ═══════════════════════════════════════════════════════════════════════════════

class WeatherTwin(nn.Module):
    """
    Neural Weather Twin — complete flood prediction system.

    Wraps FloodConvLSTM + PINNWrapper with:
      - Clean training_step / predict interface
      - Alert map generation
      - Monte Carlo Dropout uncertainty
      - Checkpoint save/load with full config
      - Device management (CPU / CUDA / MPS)
      - Parameter count and model summary

    Args:
        convlstm:        FloodConvLSTM encoder-decoder
        pinn:            PINNWrapper (model + physics loss)
        config:          Full config dict
        city:            City name (for metadata)
    """

    def __init__(
        self,
        convlstm: FloodConvLSTM,
        pinn:     PINNWrapper,
        config:   dict,
        city:     str = "kolkata",
    ):
        super().__init__()

        # pinn already wraps convlstm as pinn.model.
        # Register only pinn to avoid double weight storage in state_dict.
        self.pinn = pinn
        self.config   = config
        self.city     = city

        # Alert thresholds from config
        alert_cfg       = config.get("alerts", {}).get("thresholds", {})
        self.thresh_watch   = alert_cfg.get("watch",   0.15)
        self.thresh_warning = alert_cfg.get("warning", 0.30)
        self.thresh_danger  = alert_cfg.get("danger",  0.60)

        self.T_out = config.get("model", {}).get("output_steps", 3)
        self.T_in  = config.get("model", {}).get("input_steps",  6)

        # Training state
        self._epoch        = 0
        self._best_metric  = 0.0
        self._train_losses: List[dict] = []

    @property
    def convlstm(self) -> 'FloodConvLSTM':
        """Convenience accessor — avoids double registration in state_dict."""
        return self.pinn.model

    # ── Training interface ─────────────────────────────────────────────────────

    def training_step(
        self,
        inputs:    torch.Tensor,           # [B, T_in, 8, H, W]
        targets:   torch.Tensor,           # [B, T_out, H, W]
        elevation: torch.Tensor,           # [B, H, W]
        manning_n: Optional[torch.Tensor], # [B, H, W] or None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step: forward pass + compute total loss.

        Args:
            inputs:    Input sequence tensor
            targets:   Ground-truth flood depth maps
            elevation: Raw terrain elevation (metres, un-normalised)
            manning_n: Per-cell Manning's n or None (uses config default)

        Returns:
            loss:      Scalar loss tensor (call .backward() on this)
            loss_dict: Dict of all loss components as Python floats
        """
        self.train()

        # Forward pass
        predictions = self.convlstm(inputs)   # [B, T_out, 1, H, W]

        # PINN loss
        total_loss, loss_dict = self.pinn.compute_loss(
            predictions = predictions,
            targets     = targets,
            inputs      = inputs,
            elevation   = elevation,
            manning_n   = manning_n,
        )

        # Convert to Python floats for logging
        loss_log = {k: float(v) for k, v in loss_dict.items()}
        return total_loss, loss_log

    def update_physics_weight(self, epoch: int):
        """
        Call at start of each epoch to ramp physics loss weight.
        Uses linear warmup: 0.1 → 1.0 over first 10 epochs.
        """
        warmup = self.config.get("training", {}).get("warmup_epochs", 5)
        weight = PINNWrapper.physics_weight_schedule(epoch, warmup_epochs=warmup)
        self.pinn.set_physics_weight(weight)
        self._epoch = epoch
        return weight

    # ── Inference interface ────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        inputs:             torch.Tensor,           # [B, T_in, 8, H, W] or [T_in, 8, H, W]
        elevation:          Optional[torch.Tensor], # [B, H, W] or [H, W]
        return_uncertainty: bool = True,
        mc_passes:          int  = 20,
    ) -> FloodPrediction:
        """
        Run full inference and return a FloodPrediction result.

        Args:
            inputs:             Input sequence (batch or single sample)
            elevation:          Terrain elevation for alert context
            return_uncertainty: Run MC Dropout for uncertainty maps
            mc_passes:          Number of MC Dropout passes

        Returns:
            FloodPrediction object with depth maps, uncertainty, alert map
        """
        self.eval()
        t0 = time.perf_counter()

        # Handle unbatched input
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(0)      # [1, T_in, C, H, W]

        device = next(self.parameters()).device
        inputs = inputs.to(device)

        # Predictions
        if return_uncertainty:
            mean_pred, uncertainty = self.convlstm.predict_with_uncertainty(
                inputs, n_passes=mc_passes
            )
            # mean_pred:   [1, T_out, 1, H, W]
            # uncertainty: [1, T_out, 1, H, W]
            depth_np = mean_pred[0, :, 0].cpu().numpy()     # [T_out, H, W]
            unc_np   = uncertainty[0, :, 0].cpu().numpy()   # [T_out, H, W]
        else:
            pred   = self.convlstm(inputs)                   # [1, T_out, 1, H, W]
            depth_np = pred[0, :, 0].cpu().numpy()
            unc_np   = None

        inference_ms = (time.perf_counter() - t0) * 1000

        # Alert map: worst-case across all forecast horizons
        max_depth = depth_np.max(axis=0)    # [H, W]
        alert_map = self._depth_to_alert(max_depth)

        forecast_times = [f"T+{t+1}h" for t in range(self.T_out)]

        return FloodPrediction(
            depth_maps     = depth_np,
            uncertainty    = unc_np,
            alert_map      = alert_map,
            forecast_times = forecast_times,
            inference_ms   = inference_ms,
            city           = self.city,
            thresholds     = {
                "watch":   self.thresh_watch,
                "warning": self.thresh_warning,
                "danger":  self.thresh_danger,
            },
        )

    def predict_batch(
        self,
        inputs_batch: torch.Tensor,   # [B, T_in, C, H, W]
        elevation:    torch.Tensor,   # [B, H, W]
    ) -> torch.Tensor:
        """
        Fast batched inference — returns raw depth tensors.
        Used by evaluate.py for metrics computation.

        Returns:
            predictions: [B, T_out, H, W]
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            preds = self.convlstm(inputs_batch.to(device))   # [B, T_out, 1, H, W]
        return preds[:, :, 0, :, :]   # [B, T_out, H, W]

    # ── Alert generation ───────────────────────────────────────────────────────

    def _depth_to_alert(self, depth: np.ndarray) -> np.ndarray:
        """
        Convert depth map (metres) to integer alert level per cell.

        Level 0 (DRY):     depth < watch threshold
        Level 1 (WATCH):   watch  ≤ depth < warning
        Level 2 (WARNING): warning ≤ depth < danger
        Level 3 (DANGER):  depth ≥ danger threshold

        Args:
            depth: [H, W] float32 — max flood depth in metres

        Returns:
            alert: [H, W] uint8 — alert level per cell
        """
        alert = np.zeros_like(depth, dtype=np.uint8)
        alert[depth >= self.thresh_watch]   = ALERT_WATCH
        alert[depth >= self.thresh_warning] = ALERT_WARNING
        alert[depth >= self.thresh_danger]  = ALERT_DANGER
        return alert

    def get_ward_alerts(
        self,
        alert_map:  np.ndarray,       # [H, W]
        ward_mask:  Optional[np.ndarray] = None,  # [H, W] int — ward IDs
        min_frac:   float = 0.05,     # min fraction of ward cells flooded
    ) -> Dict[str, dict]:
        """
        Aggregate cell-level alerts to ward level.

        A ward is issued an alert if at least min_frac of its cells
        exceed the threshold. The ward alert level = highest level
        affecting ≥ min_frac of its cells.

        Args:
            alert_map:  Per-cell alert level [H, W]
            ward_mask:  Ward ID per cell [H, W] — None = single ward
            min_frac:   Minimum fraction of cells for alert trigger

        Returns:
            ward_alerts: {ward_id: {"level": int, "name": str,
                                    "flooded_pct": float}}
        """
        if ward_mask is None:
            # Treat entire grid as one ward
            flooded = float((alert_map > ALERT_DRY).mean())
            level   = int(alert_map.max()) if flooded >= min_frac else ALERT_DRY
            return {
                "ward_1": {
                    "level":       level,
                    "name":        ALERT_NAMES[level],
                    "color":       ALERT_COLORS[level],
                    "flooded_pct": round(flooded * 100, 1),
                }
            }

        ward_ids = np.unique(ward_mask)
        ward_alerts = {}
        for wid in ward_ids:
            mask  = ward_mask == wid
            if mask.sum() == 0:
                continue
            cells  = alert_map[mask]
            level  = ALERT_DRY
            for lvl in [ALERT_DANGER, ALERT_WARNING, ALERT_WATCH]:
                if (cells >= lvl).mean() >= min_frac:
                    level = lvl
                    break
            ward_alerts[str(wid)] = {
                "level":       level,
                "name":        ALERT_NAMES[level],
                "color":       ALERT_COLORS[level],
                "flooded_pct": round(float((cells > ALERT_DRY).mean()) * 100, 1),
            }
        return ward_alerts

    # ── Model info ─────────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_p = sum(p.numel() for p in self.convlstm.encoder.parameters())
        decoder_p = sum(p.numel() for p in self.convlstm.decoder.parameters())
        return {
            "total":     total,
            "trainable": trainable,
            "encoder":   encoder_p,
            "decoder":   decoder_p,
        }

    def summary(self) -> str:
        p      = self.count_parameters()
        thresh = f"WATCH>{self.thresh_watch}m / WARNING>{self.thresh_warning}m / DANGER>{self.thresh_danger}m"
        return (
            f"\n{'='*60}\n"
            f"  Neural Weather Twin — {self.city.capitalize()}\n"
            f"{'='*60}\n"
            f"  Parameters : {p['total']:>12,}  total\n"
            f"               {p['trainable']:>12,}  trainable\n"
            f"               {p['encoder']:>12,}  encoder\n"
            f"               {p['decoder']:>12,}  decoder\n"
            f"  Input      : [{self.T_in}, 8, H, W]  (6h × 8ch)\n"
            f"  Output     : [{self.T_out}, 1, H, W]  (T+1h … T+{self.T_out}h)\n"
            f"  Alerts     : {thresh}\n"
            f"{'='*60}"
        )

    # ── Checkpoint ─────────────────────────────────────────────────────────────

    def save(
        self,
        path: str,
        optimizer_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ):
        """
        Save full model checkpoint.

        Saves:
          - model state dict
          - config used to build the model
          - training epoch and best metric
          - optionally: optimizer state and eval metrics

        Args:
            path:            Save path (.pth)
            optimizer_state: Optional optimizer state_dict for resuming
            metrics:         Optional eval metrics dict to record
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict":     self.state_dict(),
            "config":               self.config,
            "city":                 self.city,
            "epoch":                self._epoch,
            "best_metric":          self._best_metric,
            "model_version":        "1.0.0",
            "architecture":         "WeatherTwin-ConvLSTM-PINN",
        }
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if metrics is not None:
            checkpoint["metrics"] = {
                k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float, torch.Tensor))
            }

        torch.save(checkpoint, path)

        # Also save human-readable metadata alongside checkpoint
        meta_path = str(path).replace(".pth", "_meta.json")
        meta = {
            "city":         self.city,
            "epoch":        self._epoch,
            "best_metric":  float(self._best_metric),
            "architecture": "WeatherTwin-ConvLSTM-PINN",
            "parameters":   self.count_parameters(),
        }
        if metrics:
            meta["metrics"] = {k: float(v) for k, v in metrics.items()
                               if isinstance(v, (int, float))}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [WeatherTwin] Saved: {path}  (epoch {self._epoch})")

    @classmethod
    def load(
        cls,
        path:   str,
        config: Optional[dict] = None,
        device: str = "auto",
        strict: bool = True,
    ) -> "WeatherTwin":
        """
        Load WeatherTwin from checkpoint.

        Args:
            path:   Path to .pth checkpoint
            config: Config dict — if None, uses config embedded in checkpoint
            device: "auto" | "cpu" | "cuda" | "mps"
            strict: Whether to require exact key match in state dict

        Returns:
            Loaded WeatherTwin in eval mode
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        map_loc = torch.device(device)
        ckpt    = torch.load(path, map_location=map_loc, weights_only=False)

        cfg  = config or ckpt.get("config", {})
        city = ckpt.get("city", "kolkata")

        twin = build_weather_twin(cfg, city=city)
        twin.load_state_dict(ckpt["model_state_dict"], strict=strict)
        twin._epoch       = ckpt.get("epoch", 0)
        twin._best_metric = ckpt.get("best_metric", 0.0)
        twin = twin.to(map_loc)
        twin.eval()

        print(
            f"  [WeatherTwin] Loaded: {path}\n"
            f"                city={city}  epoch={twin._epoch}  "
            f"best_metric={twin._best_metric:.4f}  device={device}"
        )
        return twin

    # ── Device helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def auto_device() -> torch.device:
        """Return the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def to_best_device(self) -> "WeatherTwin":
        """Move model to best available device and return self."""
        device = self.auto_device()
        print(f"  [WeatherTwin] Using device: {device}")
        return self.to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_weather_twin(config: dict, city: str = "kolkata") -> WeatherTwin:
    """
    Build a complete WeatherTwin from config dict.

    Args:
        config: Full config dict (from config.yaml)
        city:   City key for metadata

    Returns:
        WeatherTwin ready for training or inference
    """
    # Build ConvLSTM
    convlstm = build_convlstm(config["model"])

    # Build PINN wrapper (wraps convlstm + physics loss)
    pinn = build_pinn(config, convlstm_model=convlstm)

    # Assemble WeatherTwin
    twin = WeatherTwin(
        convlstm = convlstm,
        pinn     = pinn,
        config   = config,
        city     = city,
    )

    return twin


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yaml, tempfile, os

    print("=" * 60)
    print("  WeatherTwin — smoke test")
    print("=" * 60)

    torch.manual_seed(42)
    B, T_in, T_out = 2, 6, 3
    H, W = 50, 50

    # Load config
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        print("  Config loaded from config.yaml")
    except FileNotFoundError:
        cfg = {
            "model":    {"input_steps": 6, "output_steps": 3,
                         "convlstm": {"input_channels": 8,
                                      "hidden_channels": [64, 64, 32],
                                      "kernel_size": 3, "dropout": 0.2},
                         "decoder":  {"forecast_steps": 3}},
            "physics":  {"gravity": 9.81, "dt": 3600, "min_depth": 0.001,
                         "manning_default": 0.035, "lambda_continuity": 0.5,
                         "lambda_momentum": 0.3, "lambda_boundary": 0.1},
            "alerts":   {"thresholds": {"watch": 0.15,
                                        "warning": 0.30, "danger": 0.60}},
            "training": {"warmup_epochs": 5},
            "evaluation": {"flood_threshold_m": 0.20},
        }
        print("  Using default config")

    # ── Build ─────────────────────────────────────────────────────────────
    twin = build_weather_twin(cfg, city="kolkata")
    print(twin.summary())

    # ── Training step ─────────────────────────────────────────────────────
    inputs    = torch.randn(B, T_in, 8, H, W)
    targets   = torch.rand(B, T_out, H, W) * 0.5
    elevation = torch.rand(B, H, W) * 5.0

    twin.train()
    loss, loss_dict = twin.training_step(inputs, targets, elevation, None)
    assert loss.item() >= 0
    print(f"\n✓  training_step OK  total={loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"   {k:<22} {v:.4f}")

    # ── Physics weight schedule ────────────────────────────────────────────
    print("\n  Physics weight schedule:")
    for ep in [0, 3, 5, 10]:
        w = twin.update_physics_weight(ep)
        print(f"   epoch {ep:2d} → weight={w:.3f}")

    # ── Predict ───────────────────────────────────────────────────────────
    single_input = torch.randn(T_in, 8, H, W)
    result = twin.predict(single_input, elevation=None,
                          return_uncertainty=True, mc_passes=5)
    print(f"\n✓  predict OK:  {result}")
    assert result.depth_maps.shape  == (T_out, H, W)
    assert result.uncertainty.shape == (T_out, H, W)
    assert result.alert_map.shape   == (H, W)
    assert result.alert_map.max()   <= ALERT_DANGER

    # Alert breakdown
    print(f"\n  Alert summary:")
    for k, v in result.alert_summary.items():
        print(f"   {k:<20} {v}")

    # ── Ward alerts ───────────────────────────────────────────────────────
    ward_alerts = twin.get_ward_alerts(result.alert_map)
    print(f"\n✓  Ward alerts: {ward_alerts}")

    # ── Batch predict ─────────────────────────────────────────────────────
    batch_in  = torch.randn(B, T_in, 8, H, W)
    batch_elv = torch.rand(B, H, W) * 5.0
    batch_out = twin.predict_batch(batch_in, batch_elv)
    assert batch_out.shape == (B, T_out, H, W)
    print(f"✓  predict_batch OK: {tuple(batch_out.shape)}")

    # ── to_dict ───────────────────────────────────────────────────────────
    result_dict = result.to_dict()
    assert "highest_alert" in result_dict
    assert "flooded_pct" in result_dict
    print(f"✓  to_dict OK: {list(result_dict.keys())}")

    # ── Save / Load ───────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "twin_test.pth")
        twin.save(ckpt_path, metrics={"csi": 0.72, "auc": 0.88})
        loaded = WeatherTwin.load(ckpt_path, config=cfg)
        loaded_out = loaded.predict_batch(batch_in, batch_elv)
        assert loaded_out.shape == (B, T_out, H, W)
        # Outputs should be identical (same weights)
        assert torch.allclose(batch_out, loaded_out, atol=1e-5), \
            "Loaded model produces different outputs!"
        print(f"✓  save/load OK — outputs identical after reload")

    print("\n  All checks passed ✓")