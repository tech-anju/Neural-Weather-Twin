"""
utils/metrics.py
Evaluation metrics for the Neural Weather Twin flood prediction model.

All 7 metrics required by config.yaml are implemented:
  csi    — Critical Success Index       (primary metric)
  pod    — Probability of Detection
  far    — False Alarm Ratio
  bias   — Frequency Bias
  rmse   — Root Mean Square Error of depth (m)
  mae    — Mean Absolute Error of depth (m)
  fss    — Fractions Skill Score

Plus additional diagnostics:
  fbias  — Fractional bias
  hss    — Heidke Skill Score
  ets    — Equitable Threat Score (Gilbert Skill Score)
  auc    — Area Under ROC Curve
  depth_percentile_errors — per-percentile depth accuracy

Bootstrap confidence intervals (95%) computed for all metrics.

Metric reference:
  Wilks (2011). Statistical Methods in the Atmospheric Sciences.
  Roberts & Lean (2008). Scale-selective verification: Fractions Skill Score.
  Hogan & Mason (2012). Deterministic forecasts of binary events.

Usage:
  from utils.metrics import FloodMetrics, compute_all_metrics

  # Single prediction vs truth
  result = compute_all_metrics(
      predictions,    # [T_out, H, W] float32 — depth in metres
      targets,        # [T_out, H, W] float32 — depth in metres
      threshold=0.20, # flood/no-flood threshold
  )
  print(result.summary())

  # Batched evaluation across entire test set
  evaluator = FloodMetrics(threshold=0.20, bootstrap=True)
  for preds, tgts in test_loader:
      evaluator.update(preds.numpy(), tgts.numpy())
  final = evaluator.compute()
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsResult:
    """
    Container for all computed metrics with optional confidence intervals.

    Attributes:
        scores:  Dict of metric_name → float value
        cis:     Dict of metric_name → (lower, upper) 95% CI tuple
        n_samples: Number of grid cells used in computation
        threshold: Flood/no-flood depth threshold (m)
    """

    def __init__(
        self,
        scores:    Dict[str, float],
        cis:       Optional[Dict[str, Tuple[float, float]]] = None,
        n_samples: int = 0,
        threshold: float = 0.20,
        horizon:   Optional[str] = None,
    ):
        self.scores    = scores
        self.cis       = cis or {}
        self.n_samples = n_samples
        self.threshold = threshold
        self.horizon   = horizon   # e.g. "T+1h", "T+2h", "T+3h"

    def __getitem__(self, key: str) -> float:
        return self.scores[key]

    def get(self, key: str, default: float = float("nan")) -> float:
        return self.scores.get(key, default)

    def summary(self, width: int = 58) -> str:
        """Pretty-printed metric summary table."""
        ci = self.cis
        horizon_str = f"  Horizon    : {self.horizon}" if self.horizon else ""
        lines = [
            "=" * width,
            f"  Neural Weather Twin — Evaluation Metrics",
            f"  Threshold  : {self.threshold}m (flood / no-flood)",
            f"  Samples    : {self.n_samples:,} grid cells",
        ]
        if horizon_str:
            lines.append(horizon_str)
        lines += ["=" * width, ""]

        groups = {
            "Categorical (flood detection)": ["csi", "pod", "far", "bias", "hss", "ets"],
            "Continuous (depth accuracy)":   ["rmse", "mae", "fbias"],
            "Spatial skill":                 ["fss"],
            "Ranking":                       ["auc"],
        }

        for group, keys in groups.items():
            lines.append(f"  {group}")
            for k in keys:
                if k not in self.scores:
                    continue
                v   = self.scores[k]
                ci_ = ci.get(k)
                ci_str = f"  (95% CI: {ci_[0]:.3f}–{ci_[1]:.3f})" if ci_ else ""
                lines.append(f"    {k.upper():<8} {v:>8.4f}{ci_str}")
            lines.append("")

        lines.append("=" * width)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Flat dict for TensorBoard / JSON logging."""
        d = dict(self.scores)
        for k, (lo, hi) in self.cis.items():
            d[f"{k}_ci_lo"] = lo
            d[f"{k}_ci_hi"] = hi
        return d

    def __repr__(self) -> str:
        csi  = self.scores.get("csi", float("nan"))
        pod  = self.scores.get("pod", float("nan"))
        far  = self.scores.get("far", float("nan"))
        rmse = self.scores.get("rmse", float("nan"))
        return (
            f"MetricsResult("
            f"CSI={csi:.3f} POD={pod:.3f} FAR={far:.3f} RMSE={rmse:.3f}m"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Contingency table
# ═══════════════════════════════════════════════════════════════════════════════

def contingency_table(
    pred:      np.ndarray,   # float — predicted depth (m)
    obs:       np.ndarray,   # float — observed depth (m)
    threshold: float = 0.20,
) -> Dict[str, int]:
    """
    Build 2×2 contingency table for binary flood/no-flood classification.

    Predicted    Observed-YES   Observed-NO
    ─────────────────────────────────────────
    YES (flood)  TP (hits)      FP (false alarms)
    NO  (dry)    FN (misses)    TN (correct negatives)

    Args:
        pred:      Predicted depth array (any shape)
        obs:       Observed depth array (same shape)
        threshold: Depth threshold for binary classification (m)

    Returns:
        dict with TP, FP, FN, TN counts
    """
    pred_bin = (pred >= threshold).astype(bool)
    obs_bin  = (obs  >= threshold).astype(bool)

    TP = int(( pred_bin &  obs_bin).sum())
    FP = int(( pred_bin & ~obs_bin).sum())
    FN = int((~pred_bin &  obs_bin).sum())
    TN = int((~pred_bin & ~obs_bin).sum())

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}


# ═══════════════════════════════════════════════════════════════════════════════
# Individual metric functions
# ═══════════════════════════════════════════════════════════════════════════════

def csi(TP: int, FP: int, FN: int, **_) -> float:
    """
    Critical Success Index (Threat Score).
    CSI = TP / (TP + FP + FN)

    Range: [0, 1]. Higher is better.
    Ignores correct negatives (TN) — important when dry cells dominate.
    This is the PRIMARY metric for flood forecasting.

    CSI = 0: model catches nothing or predicts everything as flooded
    CSI = 1: perfect flood extent prediction
    """
    denom = TP + FP + FN
    return TP / denom if denom > 0 else 0.0


def pod(TP: int, FN: int, **_) -> float:
    """
    Probability of Detection (Recall / Hit Rate).
    POD = TP / (TP + FN)

    Range: [0, 1]. Higher is better.
    Answers: "Of all cells that actually flooded, what fraction did we predict?"
    Critical for emergency management — missed floods are costly.
    """
    denom = TP + FN
    return TP / denom if denom > 0 else 0.0


def far(FP: int, TP: int, **_) -> float:
    """
    False Alarm Ratio.
    FAR = FP / (TP + FP)

    Range: [0, 1]. LOWER is better.
    Answers: "Of all cells we predicted as flooded, what fraction was wrong?"
    High FAR → model cries wolf, reduces public trust in alerts.
    """
    denom = TP + FP
    return FP / denom if denom > 0 else 0.0


def frequency_bias(TP: int, FP: int, FN: int, **_) -> float:
    """
    Frequency Bias (BIAS).
    BIAS = (TP + FP) / (TP + FN)

    Range: [0, ∞). Perfect = 1.0.
    BIAS > 1: over-forecasting (too many flood predictions)
    BIAS < 1: under-forecasting (too few flood predictions)
    """
    denom = TP + FN
    return (TP + FP) / denom if denom > 0 else 0.0


def heidke_skill_score(TP: int, FP: int, FN: int, TN: int, **_) -> float:
    """
    Heidke Skill Score (HSS).
    HSS = 2(TP·TN - FP·FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))

    Range: [-1, 1]. Perfect = 1. No skill = 0. Worse than random = negative.
    Accounts for chance agreement (unlike CSI).
    """
    numer = 2 * (TP * TN - FP * FN)
    denom = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    return numer / denom if denom > 0 else 0.0


def equitable_threat_score(TP: int, FP: int, FN: int, TN: int, **_) -> float:
    """
    Equitable Threat Score (ETS / Gilbert Skill Score).
    ETS = (TP - TP_r) / (TP + FP + FN - TP_r)
    TP_r = (TP + FP)(TP + FN) / (TP + FP + FN + TN)  [random hits]

    Range: [-1/3, 1]. Perfect = 1. No skill = 0.
    More robust than CSI for rare events (which floods are).
    """
    total = TP + FP + FN + TN
    if total == 0:
        return 0.0
    TP_r  = (TP + FP) * (TP + FN) / total
    denom = TP + FP + FN - TP_r
    return (TP - TP_r) / denom if denom != 0 else 0.0


def rmse(
    pred: np.ndarray,
    obs:  np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Root Mean Square Error of flood depth (metres).

    If mask provided, only compute over masked cells.
    Typical use: mask = flooded cells only (obs > threshold).

    RMSE = sqrt(mean((pred - obs)²))
    """
    if mask is not None:
        pred = pred[mask]
        obs  = obs[mask]
    if pred.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def mae(
    pred: np.ndarray,
    obs:  np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Mean Absolute Error of flood depth (metres).
    MAE = mean(|pred - obs|)
    Less sensitive to outlier extreme flood events than RMSE.
    """
    if mask is not None:
        pred = pred[mask]
        obs  = obs[mask]
    if pred.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - obs)))


def fractional_bias(
    pred: np.ndarray,
    obs:  np.ndarray,
) -> float:
    """
    Fractional Bias (FBIAS).
    FBIAS = 2 × (mean(pred) - mean(obs)) / (mean(pred) + mean(obs))

    Range: [-2, 2]. Perfect = 0.
    Positive: model over-predicts depth.
    Negative: model under-predicts depth.
    """
    mean_pred = float(np.mean(pred))
    mean_obs  = float(np.mean(obs))
    denom     = mean_pred + mean_obs
    return 2 * (mean_pred - mean_obs) / denom if denom > 0 else 0.0


def fractions_skill_score(
    pred:      np.ndarray,   # [H, W] or [T, H, W]
    obs:       np.ndarray,
    threshold: float = 0.20,
    scales:    List[int] = None,
) -> float:
    """
    Fractions Skill Score (FSS).
    Roberts & Lean (2008). MWR.

    FSS = 1 - MSE_f / MSE_ref
    where MSE_f   = mean((fraction_pred - fraction_obs)²) at scale n
          MSE_ref = mean(fraction_pred²) + mean(fraction_obs²)  [worst case]

    Unlike CSI, FSS rewards spatial near-misses (flood predicted 1 cell off).
    This is essential for high-resolution urban flood forecasting.

    FSS = 1   → perfect
    FSS = 0.5 → skillful (standard threshold)
    FSS = 0   → no spatial skill

    Args:
        pred:      Predicted depth [H, W] or [T, H, W]
        obs:       Observed depth [H, W] or [T, H, W]
        threshold: Flood threshold (m)
        scales:    List of neighbourhood sizes to average over (default: [1,3,5,9,17])

    Returns:
        FSS averaged over provided scales
    """
    if scales is None:
        scales = [1, 3, 5, 9, 17]

    # Flatten time dimension if present
    if pred.ndim == 3:
        pred = pred.max(axis=0)
        obs  = obs.max(axis=0)

    pred_bin = (pred >= threshold).astype(np.float64)
    obs_bin  = (obs  >= threshold).astype(np.float64)

    fss_scores = []
    for scale in scales:
        if scale == 1:
            f_pred = pred_bin
            f_obs  = obs_bin
        else:
            # Neighbourhood fraction via uniform box filter
            f_pred = _box_filter(pred_bin, scale)
            f_obs  = _box_filter(obs_bin,  scale)

        mse_f   = np.mean((f_pred - f_obs) ** 2)
        mse_ref = np.mean(f_pred ** 2) + np.mean(f_obs ** 2)
        fss_s   = 1.0 - mse_f / mse_ref if mse_ref > 0 else 1.0
        fss_scores.append(fss_s)

    return float(np.mean(fss_scores))


def _box_filter(field: np.ndarray, size: int) -> np.ndarray:
    """
    Fast 2D uniform box filter (neighbourhood fraction).
    Equivalent to running mean over size×size window.
    Uses cumulative sum trick: O(H×W) regardless of window size.
    """
    if field.ndim != 2:
        field = field.reshape(field.shape[-2], field.shape[-1]) \
                if field.ndim >= 2 else field.reshape(1, -1)
    H, W   = field.shape
    pad    = size // 2
    padded = np.pad(field, pad, mode="constant", constant_values=0)
    # Cumulative sum
    cs     = padded.cumsum(axis=0).cumsum(axis=1)
    # Sum in windows using integral image
    cs_pad = np.pad(cs, ((1,0),(1,0)), mode="constant", constant_values=0)
    out    = (
        cs_pad[size:, size:] - cs_pad[:-size, size:]
        - cs_pad[size:, :-size] + cs_pad[:-size, :-size]
    )
    # Normalise by window area
    count = size * size
    return (out / count)[:H, :W]


def auc_roc(
    pred_depth: np.ndarray,
    obs_depth:  np.ndarray,
    threshold:  float = 0.20,
    n_thresholds: int = 100,
) -> float:
    """
    Area Under the ROC Curve for flood classification.

    Computed by sweeping prediction thresholds from 0 to max predicted depth.
    AUC = 1.0 → perfect discrimination
    AUC = 0.5 → no skill (random)
    AUC < 0.5 → anti-correlated (predictions inverted)

    Uses trapezoidal integration over TPR-FPR curve.
    """
    obs_bin    = (obs_depth >= threshold)
    max_pred   = pred_depth.max()
    thresholds = np.linspace(0, max_pred + 1e-6, n_thresholds)[::-1]

    tprs, fprs = [1.0], [1.0]

    for t in thresholds:
        pred_bin = (pred_depth >= t)
        tp = int(( pred_bin &  obs_bin).sum())
        fp = int(( pred_bin & ~obs_bin).sum())
        fn = int((~pred_bin &  obs_bin).sum())
        tn = int((~pred_bin & ~obs_bin).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    tprs.append(0.0); fprs.append(0.0)
    # Sort by FPR ascending for correct trapezoid integration
    pairs = sorted(zip(fprs, tprs))
    fprs_s = [p[0] for p in pairs]
    tprs_s = [p[1] for p in pairs]
    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(np.clip(trapz_fn(tprs_s, fprs_s), 0.0, 1.0))


def depth_percentile_errors(
    pred:      np.ndarray,
    obs:       np.ndarray,
    percentiles: List[float] = None,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Absolute error at key depth percentiles.
    Useful for understanding model accuracy at different flood severity levels.

    Returns errors at: p50 (median), p75, p90, p95, p99
    """
    if percentiles is None:
        percentiles = [50, 75, 90, 95, 99]

    if mask is not None:
        pred = pred[mask]
        obs  = obs[mask]

    if pred.size < 10:
        return {f"p{int(p)}": float("nan") for p in percentiles}

    errors = np.abs(pred - obs)
    result = {}
    for p in percentiles:
        result[f"p{int(p)}"] = float(np.percentile(errors, p))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    pred:         np.ndarray,
    obs:          np.ndarray,
    threshold:    float = 0.20,
    n_bootstrap:  int   = 500,
    ci_level:     float = 0.95,
    seed:         int   = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap 95% confidence intervals for all metrics.

    Resamples (pred, obs) pairs with replacement n_bootstrap times.
    Returns (lower, upper) bounds at ci_level confidence.

    Args:
        pred:        Predicted depth (flattened or any shape)
        obs:         Observed depth (same shape)
        threshold:   Flood threshold (m)
        n_bootstrap: Number of resampling iterations
        ci_level:    Confidence level (default 0.95 → 95% CI)
        seed:        Random seed for reproducibility

    Returns:
        Dict of metric → (lower_ci, upper_ci)
    """
    rng      = np.random.default_rng(seed)
    pred_flat = pred.ravel()
    obs_flat  = obs.ravel()
    N        = len(pred_flat)

    boot_scores: Dict[str, List[float]] = {
        "csi": [], "pod": [], "far": [], "bias": [],
        "rmse": [], "mae": [], "hss": [], "ets": [], "fss": [],
    }

    alpha     = 1.0 - ci_level
    lo_pct    = alpha / 2 * 100
    hi_pct    = (1.0 - alpha / 2) * 100

    for _ in range(n_bootstrap):
        idx  = rng.integers(0, N, size=N)
        p_b  = pred_flat[idx]
        o_b  = obs_flat[idx]

        ct = contingency_table(p_b, o_b, threshold)
        boot_scores["csi"].append(csi(**ct))
        boot_scores["pod"].append(pod(**ct))
        boot_scores["far"].append(far(**ct))
        boot_scores["bias"].append(frequency_bias(**ct))
        boot_scores["hss"].append(heidke_skill_score(**ct))
        boot_scores["ets"].append(equitable_threat_score(**ct))
        boot_scores["rmse"].append(rmse(p_b, o_b))
        boot_scores["mae"].append(mae(p_b, o_b))

        # FSS on 2D sub-sample (reshape to square-ish)
        side = max(3, int(np.sqrt(N)) // 4)
        if len(p_b) >= side * side:
            p2 = p_b[:side*side].reshape(side, side)
            o2 = o_b[:side*side].reshape(side, side)
            boot_scores["fss"].append(fractions_skill_score(p2, o2, threshold))

    cis = {}
    for k, vals in boot_scores.items():
        if not vals:
            continue
        arr = np.array(vals)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        cis[k] = (float(np.percentile(arr, lo_pct)),
                  float(np.percentile(arr, hi_pct)))
    return cis


# ═══════════════════════════════════════════════════════════════════════════════
# Main compute function
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    predictions:  np.ndarray,           # [T_out, H, W] or [H, W]
    targets:      np.ndarray,           # [T_out, H, W] or [H, W]
    threshold:    float = 0.20,
    bootstrap:    bool  = True,
    n_bootstrap:  int   = 500,
    horizon:      Optional[str] = None,
    fss_scales:   Optional[List[int]] = None,
) -> MetricsResult:
    """
    Compute all flood forecast evaluation metrics.

    Args:
        predictions:  Predicted depth maps [T_out, H, W] or [H, W]
        targets:      Ground-truth depth maps (same shape)
        threshold:    Flood/no-flood classification threshold (m)
        bootstrap:    Whether to compute 95% CIs (slower)
        n_bootstrap:  Bootstrap iterations (500 is standard)
        horizon:      Forecast horizon label e.g. "T+2h" (for display)
        fss_scales:   Neighbourhood scales for FSS (default [1,3,5,9,17])

    Returns:
        MetricsResult with all scores and optional CIs
    """
    pred = np.asarray(predictions, dtype=np.float64)
    obs  = np.asarray(targets,     dtype=np.float64)

    # Collapse time dimension: use worst-case (max depth) across horizons
    # Collapse to 2D [H, W] — handle all input shapes
    if pred.ndim == 4:        # [N, T_out, H, W] — batch from evaluate.py
        pred_2d = pred.max(axis=(0, 1))
        obs_2d  = obs.max(axis=(0, 1))
    elif pred.ndim == 3:      # [T_out, H, W] — single sample
        pred_2d = pred.max(axis=0)
        obs_2d  = obs.max(axis=0)
    elif pred.ndim == 2:      # [H, W] — already 2D
        pred_2d, obs_2d = pred, obs
    else:                     # [N] — flat from FloodMetrics.compute()
        pred_2d = pred.reshape(1, -1)
        obs_2d  = obs.reshape(1, -1)

    pred_flat = pred_2d.ravel()
    obs_flat  = obs_2d.ravel()

    # ── Contingency table ──────────────────────────────────────────────────
    ct = contingency_table(pred_flat, obs_flat, threshold)
    TP, FP, FN, TN = ct["TP"], ct["FP"], ct["FN"], ct["TN"]

    # ── Flooded-cell mask for depth metrics ────────────────────────────────
    flooded_mask = (obs_flat >= threshold)

    # ── Compute all metrics ────────────────────────────────────────────────
    scores = {
        # Categorical
        "csi":   csi(TP, FP, FN),
        "pod":   pod(TP, FN),
        "far":   far(FP, TP),
        "bias":  frequency_bias(TP, FP, FN),
        "hss":   heidke_skill_score(TP, FP, FN, TN),
        "ets":   equitable_threat_score(TP, FP, FN, TN),
        # Depth accuracy (over all cells)
        "rmse":  rmse(pred_flat, obs_flat),
        "mae":   mae(pred_flat,  obs_flat),
        "fbias": fractional_bias(pred_flat, obs_flat),
        # Depth accuracy (flooded cells only)
        "rmse_flooded": rmse(pred_flat, obs_flat, flooded_mask),
        "mae_flooded":  mae(pred_flat,  obs_flat, flooded_mask),
        # Spatial skill
        "fss":   fractions_skill_score(pred_2d, obs_2d, threshold, fss_scales),
        # Ranking
        "auc":   auc_roc(pred_flat, obs_flat, threshold),
        # Confusion matrix counts
        "TP": float(TP), "FP": float(FP),
        "FN": float(FN), "TN": float(TN),
        # Summary
        "n_flooded_obs":  float(TP + FN),
        "n_flooded_pred": float(TP + FP),
        "flood_prevalence": float((TP + FN) / max(len(obs_flat), 1)),
    }

    # ── Depth percentile errors ───────────────────────────────────────────
    pct_errors = depth_percentile_errors(
        pred_flat, obs_flat, mask=flooded_mask
    )
    scores.update({f"depth_err_{k}": v for k, v in pct_errors.items()})

    # ── Bootstrap CI ──────────────────────────────────────────────────────
    cis = {}
    if bootstrap:
        cis = bootstrap_ci(
            pred_flat, obs_flat, threshold,
            n_bootstrap=n_bootstrap,
        )

    return MetricsResult(
        scores    = scores,
        cis       = cis,
        n_samples = len(pred_flat),
        threshold = threshold,
        horizon   = horizon,
    )


def compute_per_horizon_metrics(
    predictions: np.ndarray,   # [T_out, H, W]
    targets:     np.ndarray,   # [T_out, H, W]
    threshold:   float = 0.20,
    bootstrap:   bool  = False,   # False for per-horizon (speed)
    n_bootstrap: int   = 200,
) -> List[MetricsResult]:
    """
    Compute metrics separately for each forecast horizon.

    Returns a list of MetricsResult, one per horizon:
      [T+1h result, T+2h result, T+3h result]

    Useful for understanding how skill degrades with lead time.
    """
    results = []
    T_out   = predictions.shape[0]

    for t in range(T_out):
        result = compute_all_metrics(
            predictions = predictions[t],
            targets     = targets[t],
            threshold   = threshold,
            bootstrap   = bootstrap,
            n_bootstrap = n_bootstrap,
            horizon     = f"T+{t+1}h",
        )
        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Batched evaluator (accumulates over test set)
# ═══════════════════════════════════════════════════════════════════════════════

class FloodMetrics:
    """
    Accumulates predictions and targets across the full test set,
    then computes final metrics in one batch.

    Designed for use in evaluate.py:

        evaluator = FloodMetrics(threshold=0.20, bootstrap=True)

        for preds, tgts in test_loader:
            evaluator.update(preds.numpy(), tgts.numpy())

        result = evaluator.compute()
        print(result.summary())

    Args:
        threshold:   Flood/no-flood depth threshold (m)
        bootstrap:   Compute 95% confidence intervals
        n_bootstrap: Bootstrap iterations
    """

    def __init__(
        self,
        threshold:   float = 0.20,
        bootstrap:   bool  = True,
        n_bootstrap: int   = 500,
    ):
        self.threshold   = threshold
        self.bootstrap   = bootstrap
        self.n_bootstrap = n_bootstrap
        self._preds: List[np.ndarray] = []
        self._tgts:  List[np.ndarray] = []

    def update(
        self,
        predictions: np.ndarray,   # [B, T_out, H, W] or [T_out, H, W]
        targets:     np.ndarray,   # same shape
    ):
        """Accumulate a batch of predictions."""
        self._preds.append(np.asarray(predictions, dtype=np.float32))
        self._tgts.append(np.asarray(targets,      dtype=np.float32))

    def reset(self):
        """Clear accumulated predictions."""
        self._preds.clear()
        self._tgts.clear()

    def compute(self) -> MetricsResult:
        """
        Compute metrics over all accumulated predictions.

        Returns:
            MetricsResult with final scores and CIs
        """
        if not self._preds:
            raise RuntimeError("No predictions accumulated. Call update() first.")

        all_pred = np.concatenate([p.reshape(-1) for p in self._preds])
        all_tgt  = np.concatenate([t.reshape(-1) for t in self._tgts])

        # Collapse to flat arrays — metrics computed over all cells × batches × horizons
        return compute_all_metrics(
            predictions = all_pred.reshape(1, 1, -1)[:, 0, :],
            targets     = all_tgt.reshape(1, 1, -1)[:, 0, :],
            threshold   = self.threshold,
            bootstrap   = self.bootstrap,
            n_bootstrap = self.n_bootstrap,
        )

    def compute_per_horizon(self) -> List[MetricsResult]:
        """
        Compute per-horizon metrics.
        Requires all batches to have the same T_out dimension.
        """
        if not self._preds:
            raise RuntimeError("No predictions accumulated.")

        # Stack along batch dimension
        all_pred = np.concatenate(self._preds, axis=0)   # [N, T_out, H, W]
        all_tgt  = np.concatenate(self._tgts,  axis=0)

        # Flatten spatial dims: [N, T_out, H*W] — reshape BOTH pred and tgt
        B = all_pred.shape[0]
        if all_pred.ndim == 4:   # [N, T_out, H, W]
            spatial  = all_pred.shape[-1] * all_pred.shape[-2]
            all_pred = all_pred.reshape(B, -1, spatial)
            all_tgt  = all_tgt.reshape(B, -1, spatial)
        # all_pred / all_tgt now [N, T_out, H*W]

        T_out    = all_pred.shape[1]
        results  = []
        for t in range(T_out):
            p_t = all_pred[:, t, :].ravel()
            o_t = all_tgt[:, t, :].ravel()
            r = compute_all_metrics(
                p_t.reshape(1, -1)[0], o_t.reshape(1, -1)[0],
                threshold   = self.threshold,
                bootstrap   = False,
                horizon     = f"T+{t+1}h",
            )
            results.append(r)
        return results

    @property
    def n_batches(self) -> int:
        return len(self._preds)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def skill_score_table(
    results:   List[MetricsResult],
    labels:    Optional[List[str]] = None,
) -> str:
    """
    Format multiple MetricsResult objects as a comparison table.

    Args:
        results:  List of MetricsResult (e.g. per horizon or per model)
        labels:   Row labels (default: horizon label from each result)

    Returns:
        Formatted string table
    """
    if labels is None:
        labels = [r.horizon or f"Run {i+1}" for i, r in enumerate(results)]

    metrics = ["csi", "pod", "far", "bias", "rmse", "mae", "fss"]
    header  = f"{'':15s}" + "".join(f"{m.upper():>10s}" for m in metrics)
    sep     = "-" * len(header)

    lines = [sep, header, sep]
    for label, r in zip(labels, results):
        row = f"{label:<15s}"
        for m in metrics:
            v = r.scores.get(m, float("nan"))
            row += f"{v:>10.4f}"
        lines.append(row)
    lines.append(sep)
    return "\n".join(lines)


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
    print("  utils/metrics.py — smoke test")
    print("=" * 60)

    np.random.seed(42)
    H, W, T = 50, 50, 3

    # Perfect prediction
    obs_perfect  = np.random.rand(H, W) * 0.8
    pred_perfect = obs_perfect.copy()
    ct_perf      = contingency_table(pred_perfect, obs_perfect, threshold=0.20)
    check(csi(**ct_perf) == 1.0,   "Perfect CSI = 1.0")
    check(pod(**ct_perf) == 1.0,   "Perfect POD = 1.0")
    check(far(**ct_perf) == 0.0,   "Perfect FAR = 0.0")
    check(frequency_bias(**ct_perf) == 1.0, "Perfect BIAS = 1.0")
    check(rmse(pred_perfect, obs_perfect) < 1e-10, "Perfect RMSE ≈ 0")
    check(mae(pred_perfect,  obs_perfect) < 1e-10, "Perfect MAE ≈ 0")

    # Zero prediction (model predicts everything dry)
    pred_zeros = np.zeros_like(obs_perfect)
    ct_zero    = contingency_table(pred_zeros, obs_perfect, threshold=0.20)
    check(csi(**ct_zero) == 0.0,   "All-dry CSI = 0.0")
    check(pod(**ct_zero) == 0.0,   "All-dry POD = 0.0")

    # Random prediction
    pred_rand = np.random.rand(H, W) * 0.8
    obs_rand  = np.random.rand(H, W) * 0.8
    ct_rand   = contingency_table(pred_rand, obs_rand, threshold=0.20)
    check(0 <= csi(**ct_rand) <= 1,   "Random CSI in [0,1]")
    check(0 <= pod(**ct_rand) <= 1,   "Random POD in [0,1]")
    check(0 <= far(**ct_rand) <= 1,   "Random FAR in [0,1]")

    # HSS and ETS
    hss_val = heidke_skill_score(**ct_rand)
    ets_val = equitable_threat_score(**ct_rand)
    check(-1 <= hss_val <= 1, f"HSS in [-1,1]", f"{hss_val:.3f}")
    check(-1 <= ets_val <= 1, f"ETS in [-1,1]", f"{ets_val:.3f}")

    # RMSE and MAE ordering
    rmse_all  = rmse(pred_rand, obs_rand)
    mae_all   = mae(pred_rand,  obs_rand)
    check(rmse_all >= mae_all,   "RMSE ≥ MAE (always)", f"{rmse_all:.3f} ≥ {mae_all:.3f}")
    check(rmse_all >= 0,         "RMSE ≥ 0")

    # FSS
    fss_val = fractions_skill_score(pred_rand, obs_rand, threshold=0.20)
    check(0 <= fss_val <= 1,  f"FSS in [0,1]", f"{fss_val:.4f}")
    fss_perf = fractions_skill_score(obs_perfect, obs_perfect, threshold=0.20)
    check(abs(fss_perf - 1.0) < 1e-9, "Perfect FSS = 1.0", f"{fss_perf:.4f}")

    # AUC
    auc_val = auc_roc(pred_rand, obs_rand)
    check(0 <= auc_val <= 1,  f"AUC in [0,1]", f"{auc_val:.4f}")

    # Box filter
    field = np.random.rand(20, 20)
    bf = _box_filter(field, size=3)
    check(bf.shape == (20, 20), "Box filter shape preserved")
    check(bf.min() >= 0 and bf.max() <= 1, "Box filter values in [0,1]")

    # Depth percentile errors
    pct = depth_percentile_errors(pred_rand, obs_rand)
    check("p50" in pct and "p95" in pct, "Percentile errors computed")
    check(pct["p95"] >= pct["p50"], "p95 ≥ p50 errors")

    # compute_all_metrics (3D)
    preds_3d = np.random.rand(T, H, W) * 0.8
    tgts_3d  = np.random.rand(T, H, W) * 0.8
    result   = compute_all_metrics(preds_3d, tgts_3d, bootstrap=False)
    for m in ["csi","pod","far","bias","rmse","mae","fss","auc"]:
        check(m in result.scores, f"compute_all_metrics has '{m}'")
    check(result.n_samples == H * W, f"n_samples = H×W", f"{result.n_samples}")

    # Per-horizon
    horizon_results = compute_per_horizon_metrics(preds_3d, tgts_3d, bootstrap=False)
    check(len(horizon_results) == T, f"Per-horizon: {T} results")
    check(horizon_results[0].horizon == "T+1h", "Horizon label T+1h")

    # Bootstrap CI (fast — 50 samples)
    result_ci = compute_all_metrics(
        preds_3d, tgts_3d, bootstrap=True, n_bootstrap=50
    )
    check("csi" in result_ci.cis, "Bootstrap CI computed for CSI")
    lo, hi = result_ci.cis["csi"]
    check(lo <= result_ci["csi"] <= hi, "CSI within its CI",
          f"{lo:.3f} ≤ {result_ci['csi']:.3f} ≤ {hi:.3f}")

    # FloodMetrics accumulator
    evaluator = FloodMetrics(threshold=0.20, bootstrap=False)
    for _ in range(3):
        evaluator.update(
            np.random.rand(2, T, H, W),
            np.random.rand(2, T, H, W),
        )
    check(evaluator.n_batches == 3, "FloodMetrics accumulated 3 batches")
    final = evaluator.compute()
    check("csi" in final.scores, "FloodMetrics.compute() works")

    # Summary string
    summary_str = result.summary()
    check("CSI" in summary_str,  "summary() contains CSI")
    check("RMSE" in summary_str, "summary() contains RMSE")

    # Skill table
    table = skill_score_table(horizon_results)
    check("T+1h" in table and "T+3h" in table, "skill_score_table formats horizons")

    # to_dict
    d = result.to_dict()
    check("csi" in d and "rmse" in d, "to_dict() has csi and rmse")

    print(f"\n{'='*55}")
    print(f"  PASSED: {len(ok)}   FAILED: {len(fail)}")
    print(f"{'='*55}\n")
    if fail:
        print("FAILURES:")
        for f in fail: print(f"  {f}")
        print()
    print("PASSED:")
    for o in ok: print(f"  {o}")

    # Show example output
    print()
    print(result.summary())