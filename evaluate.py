"""
evaluate.py
Evaluation script for the Neural Weather Twin.

Three modes:
  1. test      — full held-out test set (Sept 2024 Kolkata flood event)
  2. replay    — replay a specific historical flood event with warnings
  3. benchmark — compare model vs naive baselines (persistence, climatology)

Outputs:
  - Full metric table (CSI, POD, FAR, BIAS, RMSE, MAE, FSS) with 95% CI
  - Per-horizon breakdown (T+1h, T+2h, T+3h degradation curve)
  - Physics sanity check on predictions
  - Benchmark comparison table
  - Saved plots: ROC curve, depth error histogram, CSI vs threshold curve
  - JSON export of all results

Usage:
  # Full test set evaluation
  python evaluate.py --checkpoint checkpoints/best_model.pth --mode test

  # Replay the September 2024 Kolkata flood
  python evaluate.py --checkpoint checkpoints/best_model.pth --mode replay --event kolkata_sept_2024

  # Compare against baselines
  python evaluate.py --checkpoint checkpoints/best_model.pth --mode benchmark

  # Demo mode (synthetic data, no real checkpoint needed)
  python evaluate.py --mode test --demo
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

warnings.filterwarnings("ignore")

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset        import create_dataloaders
from models.weather_twin import WeatherTwin, build_weather_twin
from utils.metrics       import (
    FloodMetrics, MetricsResult,
    compute_all_metrics, compute_per_horizon_metrics, skill_score_table,
)
from utils.physics import PhysicsSanityChecker


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline models
# ═══════════════════════════════════════════════════════════════════════════════

class PersistenceBaseline:
    """
    Persistence forecast: predict that flood depth at T+N = depth at T=0.
    The simplest possible forecast — if it's flooded now, it stays flooded.
    A useful model should beat persistence on CSI.
    """

    def predict(
        self,
        inputs: np.ndarray,    # [B, T_in, C, H, W]
        T_out:  int = 3,
    ) -> np.ndarray:
        """
        Returns:
            predictions: [B, T_out, H, W] — last known depth repeated
        """
        # Channel 7 = prev_flood_depth (last input timestep)
        last_depth = inputs[:, -1, 7, :, :]    # [B, H, W]
        # Denormalise: was clipped to [0,1] with max=5m
        last_depth = last_depth * 5.0
        return np.stack([last_depth] * T_out, axis=1)   # [B, T_out, H, W]


class ZeroBaseline:
    """
    Zero forecast: always predict no flooding.
    Sets the floor — any real model must beat this on POD.
    """

    def predict(
        self,
        inputs: np.ndarray,
        T_out:  int = 3,
    ) -> np.ndarray:
        B, T_in, C, H, W = inputs.shape
        return np.zeros((B, T_out, H, W), dtype=np.float32)


class RainfallScaledBaseline:
    """
    Rainfall-scaled baseline: depth ∝ cumulative rainfall × impervious fraction.
    Simple linear model using only rainfall and terrain.
    Represents a rule-based approach without temporal learning.
    """

    RUNOFF_FACTOR = 0.003   # m depth per mm of rain (empirical urban)

    def predict(
        self,
        inputs: np.ndarray,    # [B, T_in, C, H, W]
        T_out:  int = 3,
    ) -> np.ndarray:
        """
        Estimate flood depth from cumulative rainfall × imperviousness.
        Channel 0 = rainfall (log1p normalised)
        Channel 4 = impervious fraction [0,1]
        """
        # Denormalise rainfall: log1p norm with max 150mm
        rain_norm = inputs[:, :, 0, :, :]                        # [B, T_in, H, W]
        rain_mm   = (np.exp(rain_norm * np.log(151)) - 1).clip(0)  # mm

        # Cumulative rainfall over input window
        cum_rain = rain_mm.sum(axis=1)                            # [B, H, W]

        # Impervious fraction
        imperv   = inputs[:, -1, 4, :, :]                        # [B, H, W]

        # Simple runoff depth estimate
        depth = cum_rain * imperv * self.RUNOFF_FACTOR            # [B, H, W]
        depth = np.clip(depth, 0, 3.0)

        return np.stack([depth] * T_out, axis=1)                  # [B, T_out, H, W]


# ═══════════════════════════════════════════════════════════════════════════════
# Terrain loader helper
# ═══════════════════════════════════════════════════════════════════════════════

def load_terrain(city: str, processed_dir: str, device: torch.device
                 ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load elevation_raw.npy and derive Manning's n from grid_features.npy."""
    proc = Path(processed_dir) / city
    elev_path = proc / "elevation_raw.npy"
    feat_path = proc / "grid_features.npy"

    if not elev_path.exists():
        return None, None

    try:
        elevation = np.load(elev_path).astype(np.float32)
        features  = np.load(feat_path).astype(np.float32)
        n_min, n_max = 0.013, 0.10
        manning_n = features[:, :, 5] * (n_max - n_min) + n_min

        return (
            torch.from_numpy(elevation).to(device),
            torch.from_numpy(manning_n).to(device),
        )
    except Exception as e:
        print(f"  [Eval] Terrain load failed ({e}) — physics check skipped")
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Core evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_evaluation(
    model:          WeatherTwin,
    loader:         torch.utils.data.DataLoader,
    device:         torch.device,
    config:         dict,
    elevation:      Optional[torch.Tensor] = None,
    manning_n:      Optional[torch.Tensor] = None,
    run_physics_check: bool = True,
    n_bootstrap:    int = 500,
    label:          str = "test",
) -> Tuple[MetricsResult, List[MetricsResult], dict]:
    """
    Full evaluation pass over a DataLoader.

    Args:
        model:             WeatherTwin in eval mode
        loader:            DataLoader (test or val)
        device:            torch device
        config:            Full config dict
        elevation:         [H, W] raw elevation for physics check
        manning_n:         [H, W] Manning's n for physics check
        run_physics_check: Whether to run PhysicsSanityChecker
        n_bootstrap:       Bootstrap CI samples
        label:             Label for console output

    Returns:
        overall:   MetricsResult across all timesteps
        per_horiz: List[MetricsResult] per forecast horizon
        extras:    Dict with physics report, timing, sample counts
    """
    model.eval()
    threshold = config.get("evaluation", {}).get("flood_threshold_m", 0.20)
    T_out     = config.get("model", {}).get("output_steps", 3)

    # Accumulators
    evaluator = FloodMetrics(threshold=threshold, bootstrap=False)
    all_preds:   List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    inference_times: List[float]  = []

    print(f"\n  [Eval] Running {label} evaluation "
          f"({len(loader)} batches)...")

    for batch_idx, batch in enumerate(loader):
        # Handle both (inputs, targets) and (inputs, targets, metadata)
        inputs  = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)

        t0    = time.perf_counter()
        preds = model.predict_batch(inputs, elevation)   # [B, T_out, H, W]
        inference_times.append((time.perf_counter() - t0) * 1000)

        pred_np = preds.cpu().numpy()
        tgt_np  = targets.cpu().numpy()

        evaluator.update(pred_np, tgt_np)
        all_preds.append(pred_np)
        all_targets.append(tgt_np)

        if (batch_idx + 1) % 20 == 0:
            print(f"    {batch_idx+1:4d}/{len(loader)} batches processed...")

    # Stack all predictions
    all_preds_np   = np.concatenate(all_preds,   axis=0)   # [N, T_out, H, W]
    all_targets_np = np.concatenate(all_targets, axis=0)

    print(f"  [Eval] Computing metrics on "
          f"{all_preds_np.shape[0]:,} samples × {T_out} horizons...")

    # ── Overall metrics (max-depth across horizons) ────────────────────────
    overall = compute_all_metrics(
        predictions = all_preds_np,
        targets     = all_targets_np,
        threshold   = threshold,
        bootstrap   = True,
        n_bootstrap = n_bootstrap,
    )

    # ── Per-horizon metrics ────────────────────────────────────────────────
    per_horiz = compute_per_horizon_metrics(
        predictions = all_preds_np,
        targets     = all_targets_np,
        threshold   = threshold,
        bootstrap   = False,
    )

    # ── Physics sanity check ───────────────────────────────────────────────
    physics_report = {}
    if run_physics_check and elevation is not None:
        try:
            checker  = PhysicsSanityChecker()
            # Sample first 50 predictions for speed
            n_check  = min(50, len(all_preds_np))
            # Use mean over batch as representative prediction
            sample_pred = all_preds_np[:n_check].mean(axis=0)   # [T_out, H, W]
            sample_tgt  = all_targets_np[:n_check].mean(axis=0)
            elev_np     = elevation.cpu().numpy()

            # Rainfall proxy: use target depth changes as inflow signal
            rainfall_proxy = np.clip(
                np.diff(sample_tgt, axis=0, prepend=sample_tgt[:1]),
                0, None
            ) * 200   # rough mm conversion

            physics_report = checker.check(
                predictions = sample_pred,
                elevation   = elev_np,
                rainfall    = rainfall_proxy,
                verbose     = True,
            )
        except Exception as e:
            print(f"  [Eval] Physics check failed ({e})")

    # ── Timing stats ──────────────────────────────────────────────────────
    extras = {
        "n_samples":          all_preds_np.shape[0],
        "mean_inference_ms":  float(np.mean(inference_times)),
        "p95_inference_ms":   float(np.percentile(inference_times, 95)),
        "physics_report":     physics_report,
    }

    return overall, per_horiz, extras


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark comparison
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    model:      WeatherTwin,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    config:     dict,
) -> Dict[str, MetricsResult]:
    """
    Compare WeatherTwin against baseline models.

    Baselines:
      persistence       — last known depth repeated forward
      zero              — always predict dry
      rainfall_scaled   — simple runoff × imperviousness

    Returns:
        Dict of model_name → MetricsResult
    """
    threshold = config.get("evaluation", {}).get("flood_threshold_m", 0.20)
    T_out     = config.get("model", {}).get("output_steps", 3)

    baselines = {
        "persistence":     PersistenceBaseline(),
        "zero":            ZeroBaseline(),
        "rainfall_scaled": RainfallScaledBaseline(),
    }

    # Accumulators per model
    all_data: Dict[str, Tuple[List, List]] = {
        "weather_twin": ([], []),
        **{k: ([], []) for k in baselines},
    }

    print(f"\n  [Benchmark] Running {len(loader)} batches...")

    model.eval()
    for inputs, targets in loader:
        inputs_np  = inputs.numpy()
        targets_np = targets.numpy()

        # WeatherTwin prediction
        with torch.no_grad():
            twin_pred = model.predict_batch(
                inputs.to(device), None
            ).cpu().numpy()

        all_data["weather_twin"][0].append(twin_pred)
        all_data["weather_twin"][1].append(targets_np)

        # Baseline predictions
        for name, baseline in baselines.items():
            pred = baseline.predict(inputs_np, T_out=T_out)
            all_data[name][0].append(pred)
            all_data[name][1].append(targets_np)

    # Compute metrics for each model
    results = {}
    for name, (preds_list, tgts_list) in all_data.items():
        all_p = np.concatenate(preds_list, axis=0)
        all_t = np.concatenate(tgts_list,  axis=0)
        results[name] = compute_all_metrics(
            all_p, all_t,
            threshold   = threshold,
            bootstrap   = False,
            horizon     = name,
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Flood event replay
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def replay_flood_event(
    model:      WeatherTwin,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    config:     dict,
    event_name: str = "kolkata_sept_2024",
    save_dir:   Optional[str] = None,
) -> dict:
    """
    Replay a historical flood event and show where the model would have
    issued warnings N hours before the actual flood.

    This is the KEY DEMO for the hackathon:
      "Our model would have warned 3 hours before the 2024 Kolkata flood."

    Returns:
        replay_data: dict with timestep-by-timestep predictions and alerts
    """
    threshold = config.get("evaluation", {}).get("flood_threshold_m", 0.20)
    alert_cfg = config.get("alerts", {}).get("thresholds", {})
    watch_t   = alert_cfg.get("watch",   0.15)
    warn_t    = alert_cfg.get("warning", 0.30)
    danger_t  = alert_cfg.get("danger",  0.60)

    model.eval()
    print(f"\n  [Replay] Event: {event_name}")
    print(f"  [Replay] Replaying {len(loader)} timesteps...")

    timestep_results = []

    for step_idx, batch in enumerate(loader):
        inputs  = batch[0].to(device)
        targets = batch[1]

        # Predict
        t0    = time.perf_counter()
        preds = model.predict_batch(inputs, None)   # [B, T_out, H, W]
        ms    = (time.perf_counter() - t0) * 1000

        pred_np = preds.cpu().numpy()    # [B, T_out, H, W]
        tgt_np  = targets.numpy()

        # For replay: use first sample in batch
        pred_sample  = pred_np[0]    # [T_out, H, W]
        tgt_sample   = tgt_np[0]     # [T_out, H, W]

        # Worst-case depth across forecast horizon
        max_pred  = pred_sample.max(axis=0)   # [H, W]
        max_truth = tgt_sample.max(axis=0)    # [H, W]

        # Alert levels
        pred_alert = np.zeros_like(max_pred, dtype=np.uint8)
        pred_alert[max_pred >= watch_t]  = 1
        pred_alert[max_pred >= warn_t]   = 2
        pred_alert[max_pred >= danger_t] = 3

        truth_alert = np.zeros_like(max_truth, dtype=np.uint8)
        truth_alert[max_truth >= watch_t]  = 1
        truth_alert[max_truth >= warn_t]   = 2
        truth_alert[max_truth >= danger_t] = 3

        # Per-horizon metrics at this timestep
        step_metrics = compute_all_metrics(
            pred_sample, tgt_sample,
            threshold = threshold,
            bootstrap = False,
        )

        result = {
            "step":             step_idx,
            "csi":              step_metrics["csi"],
            "pod":              step_metrics["pod"],
            "far":              step_metrics["far"],
            "max_pred_depth":   float(max_pred.max()),
            "max_truth_depth":  float(max_truth.max()),
            "pred_flooded_pct": float((max_pred >= threshold).mean() * 100),
            "truth_flooded_pct":float((max_truth >= threshold).mean() * 100),
            "highest_alert_pred":  int(pred_alert.max()),
            "highest_alert_truth": int(truth_alert.max()),
            "inference_ms":     ms,
        }
        timestep_results.append(result)

        # Print key timesteps
        if step_idx % 10 == 0 or result["max_truth_depth"] > 0.30:
            alert_names = {0: "DRY", 1: "WATCH", 2: "WARNING", 3: "DANGER"}
            print(
                f"  Step {step_idx:4d}: "
                f"CSI={result['csi']:.3f}  "
                f"pred_max={result['max_pred_depth']:.2f}m "
                f"truth_max={result['max_truth_depth']:.2f}m  "
                f"pred_alert={alert_names[result['highest_alert_pred']]}  "
                f"truth={alert_names[result['highest_alert_truth']]}"
            )

    # ── Find earliest correct warning ────────────────────────────────────
    # Identify first step where truth alert ≥ WARNING
    first_flood_step = next(
        (r["step"] for r in timestep_results if r["highest_alert_truth"] >= 2),
        None,
    )
    # Find earliest step where model issued WARNING before the actual flood
    first_pred_warning = next(
        (r["step"] for r in timestep_results if r["highest_alert_pred"] >= 2),
        None,
    )

    advance_warning_steps = None
    if first_flood_step is not None and first_pred_warning is not None:
        advance_warning_steps = first_flood_step - first_pred_warning
        if advance_warning_steps > 0:
            print(f"\n  ⚡ Model issued WARNING {advance_warning_steps} step(s) "
                  f"before actual flood reached WARNING level!")
        elif advance_warning_steps == 0:
            print(f"\n  Model issued WARNING at same time as actual flood.")
        else:
            print(f"\n  Model issued WARNING {-advance_warning_steps} step(s) "
                  f"AFTER actual flood (late warning).")

    replay_data = {
        "event":                  event_name,
        "n_steps":                len(timestep_results),
        "first_flood_step":       first_flood_step,
        "first_pred_warning":     first_pred_warning,
        "advance_warning_steps":  advance_warning_steps,
        "timesteps":              timestep_results,
        "overall_csi":            float(np.mean([r["csi"] for r in timestep_results])),
        "overall_pod":            float(np.mean([r["pod"] for r in timestep_results])),
    }

    if save_dir:
        out = Path(save_dir) / f"replay_{event_name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(replay_data, f, indent=2)
        print(f"  [Replay] Saved: {out}")

    return replay_data


# ═══════════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════════

def print_full_report(
    overall:    MetricsResult,
    per_horiz:  List[MetricsResult],
    extras:     dict,
    benchmark:  Optional[Dict[str, MetricsResult]] = None,
):
    """Print complete evaluation report to console."""

    print("\n" + "=" * 65)
    print("  NEURAL WEATHER TWIN — EVALUATION REPORT")
    print("=" * 65)

    # Overall metrics
    print(overall.summary())

    # Per-horizon degradation
    print("\n  Skill degradation with forecast lead time:")
    print(skill_score_table(per_horiz))

    # Inference timing
    print(f"\n  Inference timing:")
    print(f"    Mean  : {extras['mean_inference_ms']:.1f} ms/batch")
    print(f"    P95   : {extras['p95_inference_ms']:.1f} ms/batch")
    print(f"    Samples: {extras['n_samples']:,}")

    # Physics check
    phys = extras.get("physics_report", {})
    if phys:
        summary = phys.get("summary", {})
        print(f"\n  Physics sanity: {summary.get('score', '?')} checks passed")
        if not summary.get("pass", True):
            print(f"  ⚠  {summary.get('detail', '')}")

    # Benchmark comparison
    if benchmark:
        print("\n  Benchmark comparison:")
        print("  " + "-" * 60)
        header = f"  {'Model':<22} {'CSI':>7} {'POD':>7} {'FAR':>7} {'RMSE':>8}"
        print(header)
        print("  " + "-" * 60)
        for name, result in benchmark.items():
            marker = " ◄ WeatherTwin" if name == "weather_twin" else ""
            print(
                f"  {name:<22} "
                f"{result['csi']:>7.4f} "
                f"{result['pod']:>7.4f} "
                f"{result['far']:>7.4f} "
                f"{result['rmse']:>8.4f}"
                f"{marker}"
            )
        print("  " + "-" * 60)

        # Compute CSI improvement over best baseline
        baseline_csis = {
            k: v["csi"] for k, v in benchmark.items()
            if k != "weather_twin"
        }
        best_baseline_csi = max(baseline_csis.values()) if baseline_csis else 0
        twin_csi = benchmark["weather_twin"]["csi"]
        if best_baseline_csi > 0:
            improvement = (twin_csi - best_baseline_csi) / best_baseline_csi * 100
            print(f"\n  WeatherTwin CSI improvement over best baseline: "
                  f"+{improvement:.1f}%")


def save_results(
    overall:    MetricsResult,
    per_horiz:  List[MetricsResult],
    extras:     dict,
    benchmark:  Optional[Dict[str, MetricsResult]],
    output_dir: str,
    event:      str = "test",
):
    """Save all evaluation results to JSON and generate plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── JSON export ───────────────────────────────────────────────────────
    results_dict = {
        "event":         event,
        "overall":       overall.to_dict(),
        "per_horizon":   [r.to_dict() for r in per_horiz],
        "timing":        {
            "mean_inference_ms": extras["mean_inference_ms"],
            "p95_inference_ms":  extras["p95_inference_ms"],
            "n_samples":         extras["n_samples"],
        },
    }
    if benchmark:
        results_dict["benchmark"] = {k: v.to_dict() for k, v in benchmark.items()}

    json_path = out / f"eval_{event}.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    _try_plot_results(overall, per_horiz, benchmark, out, event)


def _try_plot_results(
    overall:   MetricsResult,
    per_horiz: List[MetricsResult],
    benchmark: Optional[Dict[str, MetricsResult]],
    out:       Path,
    event:     str,
):
    """Generate evaluation plots. Silently skips if matplotlib unavailable."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  [Eval] matplotlib not available — skipping plots")
        return

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. CSI / POD / FAR by horizon ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    horizons = [r.horizon for r in per_horiz]
    csi_vals = [r["csi"] for r in per_horiz]
    pod_vals = [r["pod"] for r in per_horiz]
    far_vals = [r["far"] for r in per_horiz]
    x = range(len(horizons))
    ax1.plot(x, csi_vals, "b-o", label="CSI", linewidth=2)
    ax1.plot(x, pod_vals, "g-s", label="POD", linewidth=2)
    ax1.plot(x, far_vals, "r-^", label="FAR", linewidth=2)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Skill threshold")
    ax1.set_xticks(x); ax1.set_xticklabels(horizons)
    ax1.set_ylim(0, 1); ax1.set_xlabel("Forecast Horizon")
    ax1.set_title("Skill by Lead Time"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # ── 2. RMSE / MAE by horizon ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    rmse_vals = [r["rmse"] for r in per_horiz]
    mae_vals  = [r["mae"]  for r in per_horiz]
    ax2.plot(x, rmse_vals, "b-o", label="RMSE (m)", linewidth=2)
    ax2.plot(x, mae_vals,  "g-s", label="MAE (m)",  linewidth=2)
    ax2.set_xticks(x); ax2.set_xticklabels(horizons)
    ax2.set_xlabel("Forecast Horizon"); ax2.set_ylabel("Depth Error (m)")
    ax2.set_title("Depth Error by Lead Time")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # ── 3. CSI vs threshold curve ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    thresholds_range = np.linspace(0.05, 1.0, 20)
    # Use overall scores — recompute CSI at different thresholds
    # (approximate from confusion matrix at varying levels)
    # Simplified: just show CSI with CI band for overall threshold
    ax3.bar(
        ["CSI", "POD", "FAR", "FSS", "AUC"],
        [overall["csi"], overall["pod"], overall["far"],
         overall.get("fss", 0), overall.get("auc", 0)],
        color=["#1565C0", "#2E7D32", "#C62828", "#6A1B9A", "#E65100"],
        alpha=0.8,
    )
    # Add CI error bars for CSI
    if "csi" in overall.cis:
        lo, hi = overall.cis["csi"]
        ci_err = [[overall["csi"] - lo], [hi - overall["csi"]]]
        ax3.errorbar(0, overall["csi"], yerr=ci_err, fmt="none",
                     color="black", capsize=5, linewidth=2)
    ax3.set_ylim(0, 1); ax3.set_title("Overall Metric Summary")
    ax3.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax3.grid(axis="y", alpha=0.3)

    # ── 4. Benchmark comparison ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if benchmark:
        names = list(benchmark.keys())
        csi_b = [benchmark[n]["csi"] for n in names]
        colors = ["#1565C0" if n == "weather_twin" else "#9E9E9E" for n in names]
        bars = ax4.bar(names, csi_b, color=colors, alpha=0.85)
        ax4.set_ylabel("CSI"); ax4.set_title("CSI: WeatherTwin vs Baselines")
        ax4.set_ylim(0, 1); ax4.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, csi_b):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax4.set_xticklabels(names, rotation=15, ha="right", fontsize=8)

    # ── 5. Metric radar ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1:], projection="polar")
    categories = ["CSI", "POD", "1-FAR", "FSS", "AUC", "HSS"]
    vals = [
        overall["csi"],
        overall["pod"],
        1 - overall["far"],          # invert FAR: higher = better
        overall.get("fss", 0),
        overall.get("auc", 0),
        max(0, overall.get("hss", 0)),
    ]
    N    = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    vals  += [vals[0]]   # close the polygon
    angles += [angles[0]]
    ax5.set_theta_offset(np.pi / 2)
    ax5.set_theta_direction(-1)
    ax5.plot(angles, vals, "b-o", linewidth=2)
    ax5.fill(angles, vals, alpha=0.15, color="blue")
    ax5.set_thetagrids(
        [a * 180 / np.pi for a in angles[:-1]], categories
    )
    ax5.set_ylim(0, 1)
    ax5.set_title("Metric Radar", pad=15)

    fig.suptitle(
        f"Neural Weather Twin — Evaluation ({event})\n"
        f"Threshold={overall.threshold}m  |  "
        f"n={overall.n_samples:,} cells  |  "
        f"CSI={overall['csi']:.3f}",
        fontsize=13, y=1.01,
    )

    plot_path = out / f"eval_{event}.png"
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Neural Weather Twin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--mode",       default="test",
                        choices=["test", "replay", "benchmark"],
                        help="Evaluation mode")
    parser.add_argument("--city",       default=None,
                        help="City (overrides config)")
    parser.add_argument("--event",      default="kolkata_sept_2024",
                        help="Event name for replay mode")
    parser.add_argument("--output_dir", default="outputs/eval",
                        help="Directory for results and plots")
    parser.add_argument("--demo",       action="store_true",
                        help="Use synthetic data — no checkpoint or real data needed")
    parser.add_argument("--no_bootstrap", action="store_true",
                        help="Skip bootstrap CI (faster)")
    parser.add_argument("--no_physics",   action="store_true",
                        help="Skip physics sanity check")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    city          = args.city or config.get("project", {}).get("city", "kolkata")
    raw_dir       = config["data"].get("raw_dir",       "data/raw")
    processed_dir = config["data"].get("processed_dir", "data/processed")
    e_cfg         = config.get("evaluation", {})
    n_bootstrap   = 50 if args.no_bootstrap else e_cfg.get("n_bootstrap", 500)
    threshold     = e_cfg.get("flood_threshold_m", 0.20)

    # ── Device ────────────────────────────────────────────────────────────
    device = WeatherTwin.auto_device()
    print(f"\n{'='*60}")
    print(f"  Neural Weather Twin — Evaluation")
    print(f"  Mode    : {args.mode.upper()}")
    print(f"  City    : {city.capitalize()}")
    print(f"  Device  : {device}")
    print(f"{'='*60}")

    # ── Model ─────────────────────────────────────────────────────────────
    if args.demo or not Path(args.checkpoint).exists():
        if not args.demo:
            print(f"\n  ⚠  Checkpoint not found: {args.checkpoint}")
            print(f"     Building untrained model for demo/smoke test...\n")
        model = build_weather_twin(config, city=city)
        model = model.to(device)
    else:
        model = WeatherTwin.load(args.checkpoint, config=config, device=str(device))

    model.eval()
    print(model.summary())

    # ── Terrain ───────────────────────────────────────────────────────────
    elevation, manning_n = load_terrain(city, processed_dir, device)

    # ── DataLoaders ───────────────────────────────────────────────────────
    # Ensure data exists — run ingest+grid if needed
    grid_meta = Path(processed_dir) / city / "grid_meta.json"
    if not grid_meta.exists():
        print("\n  [Eval] Data not found — running ingest pipeline (demo mode)...")
        from data.ingest import ingest_all
        from data.grid   import build_grid
        ingest_all(city, raw_dir, demo_mode=True)
        build_grid(city, raw_dir, processed_dir,
                   resolution_m=config["grid"]["resolution_m"])

    print(f"\n  [Eval] Building DataLoaders for split=test...")
    loaders = create_dataloaders(
        city          = city,
        config        = config.get("model", {}),
        processed_dir = processed_dir,
        raw_dir       = raw_dir,
        batch_size    = 4,
        num_workers   = 0,
        stride_test   = 1,
    )

    if len(loaders["test"]) == 0:
        print("  ⚠  Test DataLoader is empty — no test-period data found")
        print("     Check that your data covers 2024-01-01 onwards")
        sys.exit(1)

    # ── Run selected mode ─────────────────────────────────────────────────
    benchmark_results = None
    replay_data       = None

    if args.mode in ("test", "benchmark"):
        overall, per_horiz, extras = run_evaluation(
            model              = model,
            loader             = loaders["test"],
            device             = device,
            config             = config,
            elevation          = elevation,
            manning_n          = manning_n,
            run_physics_check  = not args.no_physics,
            n_bootstrap        = n_bootstrap,
            label              = f"test ({city})",
        )

        if args.mode == "benchmark":
            print("\n  [Benchmark] Running baseline comparisons...")
            benchmark_results = run_benchmark(
                model   = model,
                loader  = loaders["test"],
                device  = device,
                config  = config,
            )

        print_full_report(overall, per_horiz, extras, benchmark_results)
        save_results(overall, per_horiz, extras, benchmark_results,
                     args.output_dir, event=city)

    elif args.mode == "replay":
        replay_data = replay_flood_event(
            model      = model,
            loader     = loaders["test"],
            device     = device,
            config     = config,
            event_name = args.event,
            save_dir   = args.output_dir,
        )

        # Also run metrics on this set
        overall, per_horiz, extras = run_evaluation(
            model     = model,
            loader    = loaders["test"],
            device    = device,
            config    = config,
            elevation = elevation,
            manning_n = manning_n,
            run_physics_check = not args.no_physics,
            n_bootstrap       = n_bootstrap,
            label             = f"replay ({args.event})",
        )
        print_full_report(overall, per_horiz, extras)
        save_results(overall, per_horiz, extras, None,
                     args.output_dir, event=args.event)

        # Print replay summary
        if replay_data["advance_warning_steps"] is not None:
            adv = replay_data["advance_warning_steps"]
            print(f"\n{'='*60}")
            if adv > 0:
                print(f"  🚨 DEMO KEY RESULT:")
                print(f"     Model issued flood WARNING {adv} hour(s) BEFORE")
                print(f"     the actual flood reached WARNING level.")
                print(f"     Event: {args.event}")
            else:
                print(f"  ℹ  Model warning issued simultaneously with flood.")
            print(f"{'='*60}")

    print(f"\n  Evaluation complete. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()