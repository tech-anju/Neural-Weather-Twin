"""
train.py
Training script for the Neural Weather Twin flood prediction model.

Full training pipeline:
  1. Load config and set up logging / checkpointing
  2. Build spatial grid and DataLoaders
  3. Build WeatherTwin (ConvLSTM + PINN)
  4. Train with:
       - Physics loss warmup schedule (0.1 → 1.0 over warmup_epochs)
       - Mixed precision (FP16) via torch.cuda.amp
       - Gradient clipping
       - Cosine LR schedule with linear warmup
       - Early stopping on val CSI
       - TensorBoard logging
  5. Evaluate best checkpoint on val set every epoch
  6. Save best model + training curves

Usage:
  # Full training (GPU recommended)
  python train.py --config config.yaml --city kolkata

  # Quick smoke test (CPU, 2 epochs, small grid)
  python train.py --config config.yaml --city kolkata --smoke_test

  # Resume from checkpoint
  python train.py --config config.yaml --resume checkpoints/best_model.pth

  # Demo mode — synthetic data, no real downloads needed
  python train.py --config config.yaml --demo
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

warnings.filterwarnings("ignore")

# ── PyTorch ────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast   # torch.cuda.amp deprecated in 2.x
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

# ── Project modules ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset   import create_dataloaders
from models.weather_twin import WeatherTwin, build_weather_twin
from models.pinn         import PINNWrapper
from utils.metrics       import FloodMetrics


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducible training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (slight speed cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# Optimizer and scheduler
# ═══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model: WeatherTwin, config: dict) -> torch.optim.Optimizer:
    """
    AdamW with differential learning rates:
      - Encoder backbone: lr × 0.1  (pretrained features, train slowly)
      - Decoder + output head: lr × 1.0
      - Physics-related params: lr × 0.5

    Differential LRs prevent the encoder from forgetting spatial patterns
    learned in early epochs when physics loss starts dominating.
    """
    lr = config["training"]["learning_rate"]
    wd = config["training"]["weight_decay"]

    encoder_params = list(model.convlstm.encoder.parameters())
    decoder_params = list(model.convlstm.decoder.parameters())
    proj_params    = list(model.convlstm.input_proj.parameters())

    # Collect all remaining params not in the above groups
    encoder_ids = {id(p) for p in encoder_params}
    decoder_ids = {id(p) for p in decoder_params + proj_params}
    other_params = [
        p for p in model.parameters()
        if id(p) not in encoder_ids | decoder_ids
    ]

    param_groups = [
        {"params": encoder_params,  "lr": lr * 0.1,  "name": "encoder"},
        {"params": decoder_params + proj_params, "lr": lr, "name": "decoder"},
        {"params": other_params,     "lr": lr * 0.5,  "name": "other"},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config:    dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Cosine annealing with linear warmup.

    Warmup: LR ramps from lr/10 → lr over warmup_epochs
    Cosine: LR decays from lr → lr/1000 over remaining epochs
    """
    epochs        = config["training"]["epochs"]
    warmup_epochs = config["training"]["warmup_epochs"]

    warmup = LinearLR(
        optimizer,
        start_factor = 0.1,
        end_factor   = 1.0,
        total_iters  = warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max   = max(1, epochs - warmup_epochs),
        eta_min = config["training"]["learning_rate"] * 1e-3,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [warmup_epochs],
    )
    return scheduler


# ═══════════════════════════════════════════════════════════════════════════════
# Terrain utilities (for physics loss)
# ═══════════════════════════════════════════════════════════════════════════════

def load_terrain_tensors(
    city:          str,
    processed_dir: str,
    device:        torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load raw elevation and Manning's n tensors for physics loss.

    These are static — loaded once and kept on device.

    Returns:
        elevation: float32 [H, W]  — raw metres (un-normalised)
        manning_n: float32 [H, W]  — per-cell roughness coefficient
    """
    proc = Path(processed_dir) / city

    elev_path = proc / "elevation_raw.npy"
    feat_path = proc / "grid_features.npy"
    meta_path = proc / "grid_meta.json"

    if not elev_path.exists() or not meta_path.exists():
        print(f"  [Train] Terrain files not found in {proc} — physics loss will use defaults")
        return None, None

    try:
        elevation = np.load(elev_path).astype(np.float32)
        features  = np.load(feat_path).astype(np.float32)   # [H, W, 6]

        # Manning's n is channel index 5 in features (see grid.py FEATURE_NAMES)
        # It was normalised to [0,1] — denormalise back to real n values
        # n_min = 0.013 (asphalt), n_max = 0.10 (wetland) from grid.py
        n_min, n_max = 0.013, 0.10
        manning_n = features[:, :, 5] * (n_max - n_min) + n_min

        elev_t  = torch.from_numpy(elevation).to(device)
        mann_t  = torch.from_numpy(manning_n).to(device)

        print(f"  [Train] Terrain loaded: elevation {elevation.shape} "
              f"min={elevation.min():.1f}m max={elevation.max():.1f}m")
        return elev_t, mann_t

    except Exception as e:
        print(f"  [Train] Terrain load failed ({e}) — physics uses defaults")
        return None, None


def expand_terrain_for_batch(
    terrain:    Optional[torch.Tensor],
    batch_size: int,
    target_h:   Optional[int] = None,
    target_w:   Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Expand [H_full, W_full] static tensor to [B, H, W] for batch processing.

    When crop_size is active, inputs/targets are spatially cropped but the
    terrain tensors are still full-grid size. We centre-crop them here to
    match the actual spatial dims of the batch.
    """
    if terrain is None:
        return None
    t = terrain   # [H_full, W_full]
    if target_h is not None and target_w is not None:
        H_full, W_full = t.shape
        if H_full != target_h or W_full != target_w:
            # Centre-crop to match batch spatial size
            r0 = (H_full - target_h) // 2
            c0 = (W_full - target_w) // 2
            t  = t[r0:r0 + target_h, c0:c0 + target_w]
    return t.unsqueeze(0).expand(batch_size, -1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# Early stopping
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stop training when validation CSI stops improving.

    Args:
        patience:  Epochs to wait after last improvement
        min_delta: Minimum improvement to count as progress
        mode:      "max" for CSI/AUC, "min" for loss/RMSE
    """

    def __init__(
        self,
        patience:  int   = 12,
        min_delta: float = 1e-4,
        mode:      str   = "max",
    ):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = float("-inf") if mode == "max" else float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, metric: float) -> bool:
        """
        Returns True if training should stop.
        """
        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True

        return self.triggered

    @property
    def message(self) -> str:
        return (
            f"Early stopping triggered after {self.patience} epochs "
            f"without improvement (best={self.best:.4f})"
        ) if self.triggered else (
            f"Patience: {self.counter}/{self.patience} "
            f"(best={self.best:.4f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:        WeatherTwin,
    loader:       torch.utils.data.DataLoader,
    optimizer:    torch.optim.Optimizer,
    scaler:       Optional[GradScaler],
    device:       torch.device,
    elevation:    Optional[torch.Tensor],
    manning_n:    Optional[torch.Tensor],
    grad_clip:    float,
    log_every:    int,
    epoch:        int,
    writer:       Optional[SummaryWriter],
) -> Dict[str, float]:
    """
    Single training epoch.

    Returns:
        Dict of averaged loss components for this epoch
    """
    model.train()
    running = {}
    n_batches = 0
    t_start = time.perf_counter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(device,  non_blocking=True)   # [B, T_in, C, H, W]
        targets = targets.to(device, non_blocking=True)   # [B, T_out, H, W]
        B       = inputs.shape[0]

        # Expand static terrain maps to batch size
        # Pass target H,W so terrain is centre-cropped when crop_size is active
        _, _, _, H_in, W_in = inputs.shape   # [B, T_in, C, H, W]
        elev_b = expand_terrain_for_batch(elevation, B, H_in, W_in)
        mann_b = expand_terrain_for_batch(manning_n, B, H_in, W_in)

        optimizer.zero_grad()

        # ── Forward pass (with optional mixed precision) ───────────────────
        if scaler is not None:
            with autocast("cuda"):
                loss, loss_dict = model.training_step(
                    inputs, targets, elev_b, mann_b
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = model.training_step(
                inputs, targets, elev_b, mann_b
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # ── Accumulate loss stats ──────────────────────────────────────────
        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + float(v)
        n_batches += 1

        # ── Batch-level logging ────────────────────────────────────────────
        global_step = epoch * len(loader) + batch_idx
        if (batch_idx + 1) % log_every == 0:
            elapsed = time.perf_counter() - t_start
            it_s    = n_batches / elapsed
            print(
                f"  [{epoch+1}][{batch_idx+1:4d}/{len(loader)}] "
                f"loss={loss_dict.get('total', 0):.4f}  "
                f"data={loss_dict.get('data', 0):.4f}  "
                f"physics={loss_dict.get('physics', 0):.4f}  "
                f"{it_s:.1f} it/s"
            )
            if writer:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train_batch/{k}", v, global_step)

    # Average over epoch
    avg = {k: v / max(n_batches, 1) for k, v in running.items()}
    return avg


@torch.no_grad()
def validate(
    model:      WeatherTwin,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    elevation:  Optional[torch.Tensor],
    manning_n:  Optional[torch.Tensor],
    threshold:  float = 0.10,
) -> Tuple[Dict[str, float], float]:
    """
    Validation loop.

    Returns:
        loss_dict: averaged validation losses
        csi:       validation CSI (primary metric for early stopping)
    """
    model.eval()
    evaluator = FloodMetrics(threshold=threshold, bootstrap=False)
    running   = {}
    n_batches = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        B       = inputs.shape[0]

        _, _, _, H_in, W_in = inputs.shape
        elev_b = expand_terrain_for_batch(elevation, B, H_in, W_in)
        mann_b = expand_terrain_for_batch(manning_n, B, H_in, W_in)

        # Predictions
        predictions = model.convlstm(inputs)   # [B, T_out, 1, H, W]

        # Validation loss (no gradient)
        _, loss_dict = model.pinn.compute_loss(
            predictions = predictions,
            targets     = targets,
            inputs      = inputs,
            elevation   = elev_b if elev_b is not None
                          else torch.zeros(B, inputs.shape[-2], inputs.shape[-1],
                                           device=device),
            manning_n   = mann_b,
        )

        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + float(v)
        n_batches += 1

        # Accumulate for metrics
        pred_np = predictions[:, :, 0, :, :].cpu().numpy()  # [B, T_out, H, W]
        tgt_np  = targets.cpu().numpy()
        evaluator.update(pred_np, tgt_np)

    avg      = {k: v / max(n_batches, 1) for k, v in running.items()}
    metrics  = evaluator.compute()
    val_csi  = metrics.scores.get("csi", 0.0)

    # Add key metrics to loss dict for logging
    for m in ["csi", "pod", "far", "rmse", "mae", "fss"]:
        avg[f"val_{m}"] = metrics.scores.get(m, 0.0)

    return avg, val_csi


# ═══════════════════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    config:     dict,
    city:       str,
    resume:     Optional[str] = None,
    demo_mode:  bool = False,
    smoke_test: bool = False,
) -> WeatherTwin:
    """
    Main training function.

    Args:
        config:     Full config dict
        city:       City key (kolkata | chennai | mumbai)
        resume:     Path to checkpoint to resume from
        demo_mode:  Use synthetic data (no real downloads needed)
        smoke_test: 2 epochs on small subset for CI / quick check

    Returns:
        Best WeatherTwin model loaded from checkpoint
    """
    t_cfg  = config["training"]
    e_cfg  = config.get("evaluation", {})

    # ── Seed ────────────────────────────────────────────────────────────────
    # Reduce CUDA memory fragmentation — critical for 4GB GPUs
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    set_seed(config.get("project", {}).get("seed", 42))

    # ── Device ──────────────────────────────────────────────────────────────
    device = WeatherTwin.auto_device()
    print(f"\n{'='*60}")
    print(f"  Neural Weather Twin — Training")
    print(f"  City    : {city.capitalize()}")
    print(f"  Device  : {device}")
    print(f"  Mode    : {'SMOKE TEST' if smoke_test else 'DEMO' if demo_mode else 'FULL'}")
    print(f"{'='*60}\n")

    # ── Directories ─────────────────────────────────────────────────────────
    raw_dir       = config["data"].get("raw_dir",       "data/raw")
    processed_dir = config["data"].get("processed_dir", "data/processed")
    ckpt_dir      = Path(t_cfg.get("checkpoint_dir", "checkpoints"))
    log_dir       = Path(t_cfg.get("log_dir",         "logs"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True,  exist_ok=True)

    # ── Ingest + grid (if not already done) ───────────────────────────────
    grid_meta = Path(processed_dir) / city / "grid_meta.json"
    if not grid_meta.exists():
        print(f"  [Train] Grid not found — running data pipeline first...\n")
        from data.ingest import ingest_all
        from data.grid   import build_grid

        ingest_all(city, raw_dir, demo_mode=demo_mode)
        build_grid(city, raw_dir, processed_dir,
                   resolution_m=config["grid"]["resolution_m"])
    else:
        print(f"  [Train] Using existing grid: {grid_meta}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    epochs     = 2 if smoke_test else t_cfg["epochs"]
    # smoke_test: batch=1 and crop_size=112 keeps 4GB GPU well within limits
    batch_size = 1 if smoke_test else t_cfg["batch_size"]
    # crop_size: smoke_test=112, demo=112 (safe for 4GB GPU), full=from config
    _cfg_crop  = config["training"].get("crop_size", 0)
    crop_size  = 112 if (smoke_test or demo_mode) else _cfg_crop

    print(f"\n  [Train] Building DataLoaders...")
    # Windows multiprocessing spawn fails pickling large datasets — use 0 workers
    import platform as _platform
    _nw = 0 if (smoke_test or _platform.system() == "Windows") else 2
    loaders = create_dataloaders(
        city          = city,
        config        = config.get("model", {}),
        processed_dir = processed_dir,
        raw_dir       = raw_dir,
        batch_size    = batch_size,
        num_workers   = _nw,
        stride_train  = 6 if smoke_test else 1,
        stride_val    = 6 if smoke_test else 3,
        crop_size     = crop_size,
    )

    if len(loaders["train"]) == 0:
        raise RuntimeError(
            "Training DataLoader is empty. "
            "Run: python data/ingest.py --city kolkata --demo"
        )

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\n  [Train] Building WeatherTwin...")
    if resume:
        print(f"  [Train] Resuming from: {resume}")
        model = WeatherTwin.load(resume, config=config)
    else:
        model = build_weather_twin(config, city=city)

    model = model.to(device)
    print(model.summary())

    # ── Static terrain tensors ────────────────────────────────────────────
    elevation, manning_n = load_terrain_tensors(city, processed_dir, device)
    if elevation is None:
        print("  [Train] ⚠ WARNING: No terrain data — physics loss will be 0.0 "
              "every epoch. Run `python data/ingest.py --city kolkata --demo` first.")
    else:
        print("  [Train] ✅ Terrain loaded — physics loss is ACTIVE")

    # ── Optimizer + scheduler ────────────────────────────────────────────
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(loaders["train"]))

    # ── Mixed precision scaler ────────────────────────────────────────────
    use_amp   = t_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler    = GradScaler("cuda") if use_amp else None
    if use_amp:
        print(f"  [Train] Mixed precision (FP16) enabled")

    # ── TensorBoard writer ────────────────────────────────────────────────
    run_name = f"{city}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer   = SummaryWriter(log_dir / run_name)
    print(f"  [Train] TensorBoard: tensorboard --logdir {log_dir}\n")

    # ── Early stopping ────────────────────────────────────────────────────
    early_stop = EarlyStopping(
        patience  = t_cfg.get("early_stopping_patience", 12),
        min_delta = 1e-4,
        mode      = "max",
    )

    grad_clip  = t_cfg.get("gradient_clip", 1.0)
    log_every  = t_cfg.get("log_every", 10)
    threshold  = e_cfg.get("flood_threshold_m", 0.20)
    best_csi   = 0.0
    best_ckpt  = str(ckpt_dir / "best_model.pth")
    train_log: List[dict] = []

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  Starting training: {epochs} epochs  |  "
          f"{len(loaders['train'])} train batches  |  "
          f"{len(loaders['val'])} val batches")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        ep_start = time.perf_counter()

        # ── Physics weight ramp ────────────────────────────────────────────
        phys_weight = model.update_physics_weight(epoch)

        # ── Train ─────────────────────────────────────────────────────────
        print(f"\nEpoch {epoch+1}/{epochs}  "
              f"(physics_weight={phys_weight:.3f}  "
              f"lr={optimizer.param_groups[1]['lr']:.2e})")

        train_losses = train_one_epoch(
            model      = model,
            loader     = loaders["train"],
            optimizer  = optimizer,
            scaler     = scaler,
            device     = device,
            elevation  = elevation,
            manning_n  = manning_n,
            grad_clip  = grad_clip,
            log_every  = log_every,
            epoch      = epoch,
            writer     = writer,
        )

        # ── Validate ──────────────────────────────────────────────────────
        val_losses, val_csi = validate(
            model     = model,
            loader    = loaders["val"],
            device    = device,
            elevation = elevation,
            manning_n = manning_n,
            threshold = threshold,
        )

        # ── LR step ──────────────────────────────────────────────────────
        scheduler.step()

        # ── Log ──────────────────────────────────────────────────────────
        ep_time = time.perf_counter() - ep_start

        # TensorBoard
        for k, v in train_losses.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_losses.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("train/lr",            optimizer.param_groups[1]["lr"], epoch)
        writer.add_scalar("train/physics_weight", phys_weight, epoch)

        # Console summary
        print(
            f"\n  ── Epoch {epoch+1} summary ──────────────────────────────\n"
            f"  Train: loss={train_losses.get('total', 0):.4f}  "
            f"data={train_losses.get('data', 0):.4f}  "
            f"physics={train_losses.get('physics', 0):.4f}\n"
            f"  Val:   loss={val_losses.get('total', 0):.4f}  "
            f"CSI={val_csi:.4f}  "
            f"POD={val_losses.get('val_pod', 0):.4f}  "
            f"FAR={val_losses.get('val_far', 0):.4f}  "
            f"RMSE={val_losses.get('val_rmse', 0):.4f}m\n"
            f"  Time:  {ep_time:.1f}s  |  {early_stop.message}"
        )

        # Training log record
        log_record = {
            "epoch":        epoch + 1,
            "physics_weight": phys_weight,
            "lr":           optimizer.param_groups[1]["lr"],
            **{f"train_{k}": v for k, v in train_losses.items()},
            **val_losses,
            "time_s":       ep_time,
        }
        train_log.append(log_record)

        # ── Save best checkpoint ──────────────────────────────────────────
        if val_csi > best_csi:
            best_csi       = val_csi
            model._best_metric = best_csi
            model.save(
                path            = best_ckpt,
                optimizer_state = optimizer.state_dict(),
                metrics         = {
                    "val_csi":  val_csi,
                    "val_pod":  val_losses.get("val_pod", 0),
                    "val_far":  val_losses.get("val_far", 0),
                    "val_rmse": val_losses.get("val_rmse", 0),
                },
            )
            print(f"  ✓  New best CSI={best_csi:.4f} → saved to {best_ckpt}")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            periodic_path = str(ckpt_dir / f"epoch_{epoch+1:03d}.pth")
            model._epoch  = epoch + 1
            model.save(periodic_path)

        # ── Early stopping check ──────────────────────────────────────────
        if early_stop.step(val_csi):
            print(f"\n  ⚡ {early_stop.message}")
            break

    # ── Save training log ─────────────────────────────────────────────────
    log_path = ckpt_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    writer.close()

    # ── Final summary ─────────────────────────────────────────────────────
    actual_epochs = len(train_log)
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Epochs trained : {actual_epochs}")
    print(f"  Best val CSI   : {best_csi:.4f}")
    print(f"  Best checkpoint: {best_ckpt}")
    print(f"  Training log   : {log_path}")
    print(f"{'='*60}\n")

    # ── Load and return best model ────────────────────────────────────────
    # Guard: if val CSI never improved (e.g. smoke_test with 2 epochs),
    # no checkpoint was saved. Save the final model state now so the
    # caller always gets a valid model back.
    if not Path(best_ckpt).exists():
        print(f"  [Train] No best checkpoint saved (val CSI never improved). "
              f"Saving final model weights to {best_ckpt}")
        model._epoch = actual_epochs
        model.save(
            path    = best_ckpt,
            metrics = {"val_csi": best_csi, "note": "final_epoch_fallback"},
        )
    best_model = WeatherTwin.load(best_ckpt, config=config, device="auto")
    return best_model


# ═══════════════════════════════════════════════════════════════════════════════
# Post-training validation on test set
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test_set(
    model:        WeatherTwin,
    config:       dict,
    city:         str,
    processed_dir: str = "data/processed",
    raw_dir:       str = "data/raw",
) -> None:
    """
    Run final evaluation on the held-out test set (September 2024 Kolkata flood).
    Prints full metrics report with bootstrap CIs.
    """
    device = next(model.parameters()).device
    model.eval()

    print(f"\n{'='*60}")
    print(f"  Final evaluation on TEST set ({city} 2024 flood event)")
    print(f"{'='*60}")

    loaders = create_dataloaders(
        city          = city,
        config        = config.get("model", {}),
        processed_dir = processed_dir,
        raw_dir       = raw_dir,
        batch_size    = 4,
        num_workers   = 0,
        stride_test   = 1,
    )

    elevation, manning_n = load_terrain_tensors(city, processed_dir, device)
    evaluator = FloodMetrics(
        threshold   = config.get("evaluation", {}).get("flood_threshold_m", 0.20),
        bootstrap   = True,
        n_bootstrap = config.get("evaluation", {}).get("n_bootstrap", 500),
    )

    for inputs, targets, *_ in loaders["test"]:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        preds   = model.convlstm(inputs)[:, :, 0, :, :]   # [B, T_out, H, W]
        evaluator.update(preds.cpu().numpy(), targets.cpu().numpy())

    result = evaluator.compute()
    print(result.summary())

    # Save test results
    out_path = Path(config["training"].get("checkpoint_dir", "checkpoints")) / "test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\n  Test metrics saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Neural Weather Twin flood prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--city", default=None,
        help="City to train on (overrides config). Options: kolkata | chennai | mumbai",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use synthetic data only — no real data downloads needed",
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Quick 2-epoch run to verify pipeline works (CPU friendly)",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training, evaluate checkpoint on test set only",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint path for --eval_only mode",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--crop_size", type=int, default=0,
        help="Spatial crop size per sample (0=full grid). Use 112 for 4GB GPU, 224 for 8GB GPU.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── CLI overrides ────────────────────────────────────────────────────────
    city = args.city or config.get("project", {}).get("city", "kolkata")

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.crop_size > 0 and not args.smoke_test:
        config["training"]["crop_size"] = args.crop_size

    # ── Eval only mode ───────────────────────────────────────────────────────
    if args.eval_only:
        if not args.checkpoint:
            print("ERROR: --eval_only requires --checkpoint <path>")
            sys.exit(1)
        model = WeatherTwin.load(args.checkpoint, config=config)
        evaluate_test_set(
            model         = model,
            config        = config,
            city          = city,
            processed_dir = config["data"].get("processed_dir", "data/processed"),
            raw_dir       = config["data"].get("raw_dir",       "data/raw"),
        )
        return

    # ── Train ────────────────────────────────────────────────────────────────
    best_model = train(
        config     = config,
        city       = city,
        resume     = args.resume,
        demo_mode  = args.demo,
        smoke_test = args.smoke_test,
    )

    # ── Test set evaluation ───────────────────────────────────────────────────
    if not args.smoke_test:
        evaluate_test_set(
            model         = best_model,
            config        = config,
            city          = city,
            processed_dir = config["data"].get("processed_dir", "data/processed"),
            raw_dir       = config["data"].get("raw_dir",       "data/raw"),
        )


if __name__ == "__main__":
    main()