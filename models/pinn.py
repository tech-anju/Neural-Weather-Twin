"""
models/pinn.py
Physics-Informed Neural Network (PINN) wrapper for flood prediction.

The core innovation of the Neural Weather Twin:
Saint-Venant shallow water equations are embedded as differentiable
residual terms inside the PyTorch loss function. The model must satisfy
real hydraulic physics — not just fit historical data.

Saint-Venant Shallow Water Equations (1D simplified for urban grid):

  Continuity (mass conservation):
    ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = r - i
    where:
      h = water depth (m)
      u = depth-averaged x-velocity (m/s)
      v = depth-averaged y-velocity (m/s)
      r = rainfall rate (m/s)
      i = infiltration rate (m/s)

  Momentum (x-direction):
    ∂(hu)/∂t + ∂(hu²+gh²/2)/∂x + ∂(huv)/∂y =
      -gh·∂z/∂x - g·n²·u·|U|/h^(1/3)
    where:
      g = gravitational acceleration (9.81 m/s²)
      z = bed elevation (m)
      n = Manning's roughness coefficient
      |U| = sqrt(u² + v²) = flow speed

  Momentum (y-direction): symmetric to x.

Implementation notes:
  - All spatial derivatives use central finite differences on the 50m grid
  - Wetting/drying handled by masking cells with h < min_depth threshold
  - Boundary condition: zero-flux at grid edges (reflective walls)
  - Physics residuals are averaged over flooded cells only (not dry cells)
  - Gradients flow through residuals back into ConvLSTM weights

Total training loss:
  L = L_data + λ_c × L_continuity + λ_m × L_momentum + λ_b × L_boundary
  where L_data = MSE(predicted_depth, SAR_observed_depth)

References:
  Raissi et al. (2019). Physics-Informed Neural Networks.
  Journal of Computational Physics. doi:10.1016/j.jcp.2018.10.045

  Bates et al. (2010). A simple inertial formulation of the shallow
  water equations for efficient two-dimensional flood inundation modelling.
  Journal of Hydrology. doi:10.1016/j.jhydrol.2010.03.027

Usage:
  from models.pinn import PhysicsLoss, PINNWrapper, build_pinn

  physics_loss = PhysicsLoss(config["physics"])
  wrapper = build_pinn(config)

  total_loss, loss_dict = wrapper.compute_loss(
      predictions, targets, inputs, terrain, elevation_raw
  )
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Finite Difference Operators (differentiable, PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

class FiniteDiff:
    """
    Differentiable finite difference operators on a regular 2D grid.
    All operations preserve spatial dimensions via padding.
    Gradients flow through all operators into model weights.

    Grid convention:
      axis 0 (dim -2) = y direction (rows, north-south)
      axis 1 (dim -1) = x direction (cols, west-east)
    """

    @staticmethod
    def ddx(field: torch.Tensor, dx: float) -> torch.Tensor:
        """
        Central difference ∂f/∂x.
        At boundaries: one-sided forward/backward difference.

        Args:
            field: [B, H, W] or [H, W]
            dx:    cell size in x-direction (metres)

        Returns:
            dfdx: same shape as field
        """
        # Pad last dimension (x) with edge values for boundary handling
        f = F.pad(field, (1, 1), mode="replicate")  # [..., H, W+2]
        return (f[..., 2:] - f[..., :-2]) / (2.0 * dx)

    @staticmethod
    def ddy(field: torch.Tensor, dy: float) -> torch.Tensor:
        """Central difference ∂f/∂y (rows direction)."""
        f = F.pad(field, (0, 0, 1, 1), mode="replicate")  # [..., H+2, W]
        return (f[..., 2:, :] - f[..., :-2, :]) / (2.0 * dy)

    @staticmethod
    def divergence(
        fx: torch.Tensor, fy: torch.Tensor, dx: float, dy: float
    ) -> torch.Tensor:
        """
        Divergence ∂fx/∂x + ∂fy/∂y.
        Used for continuity equation: ∇·(hu) = ∂(hu)/∂x + ∂(hv)/∂y
        """
        return FiniteDiff.ddx(fx, dx) + FiniteDiff.ddy(fy, dy)

    @staticmethod
    def time_derivative(
        field_t1: torch.Tensor, field_t0: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """Forward difference ∂f/∂t = (f_{t+1} - f_t) / dt."""
        return (field_t1 - field_t0) / dt


# ═══════════════════════════════════════════════════════════════════════════════
# Physics Loss Terms
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsLoss(nn.Module):
    """
    Saint-Venant shallow water equation residuals as differentiable loss terms.

    Args:
        gravity:           g = 9.81 m/s²
        dt:                Timestep in seconds (3600 = 1 hour)
        dx:                Cell size in x-direction (metres, default 50m)
        dy:                Cell size in y-direction (metres, default 50m)
        min_depth:         Wetting/drying threshold (m) — cells below this
                           are treated as dry and excluded from physics loss
        manning_default:   Default Manning's n if not provided per-cell
        lambda_continuity: Weight of continuity residual in total loss
        lambda_momentum:   Weight of momentum residual in total loss
        lambda_boundary:   Weight of boundary condition residual
        reduction:         "mean" or "sum" over spatial dimensions
    """

    def __init__(
        self,
        gravity:           float = 9.81,
        dt:                float = 3600.0,
        dx:                float = 50.0,
        dy:                float = 50.0,
        min_depth:         float = 0.001,
        manning_default:   float = 0.035,
        lambda_continuity: float = 0.5,
        lambda_momentum:   float = 0.3,
        lambda_boundary:   float = 0.1,
        reduction:         str   = "mean",
    ):
        super().__init__()

        self.g                  = gravity
        self.dt                 = dt
        self.dx                 = dx
        self.dy                 = dy
        self.min_depth          = min_depth
        self.manning_default    = manning_default
        self.lambda_continuity  = lambda_continuity
        self.lambda_momentum    = lambda_momentum
        self.lambda_boundary    = lambda_boundary
        self.reduction          = reduction

        self.fd = FiniteDiff()

    def _wet_mask(self, h: torch.Tensor) -> torch.Tensor:
        """
        Boolean mask of wet cells (h > min_depth).
        Physics residuals only computed for wet cells.
        Avoids division by zero in Manning's friction term.
        """
        return (h > self.min_depth).float()

    def _estimate_velocity(
        self,
        h: torch.Tensor,       # [B, H, W] — water depth
        dzdx: torch.Tensor,    # [B, H, W] — bed slope x
        dzdy: torch.Tensor,    # [B, H, W] — bed slope y
        manning_n: torch.Tensor,  # [B, H, W] or scalar
        prev_u: Optional[torch.Tensor] = None,  # [B, H, W]
        prev_v: Optional[torch.Tensor] = None,  # [B, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate depth-averaged velocities using Manning's equation.

        Manning's equation for open channel flow:
          U = (1/n) × R_h^(2/3) × S^(1/2)
          where R_h ≈ h (wide channel approximation)
                S   = bed slope magnitude

        We decompose U into x,y components using slope direction:
          u = -U × (∂z/∂x) / (|∇z| + ε)
          v = -U × (∂z/∂y) / (|∇z| + ε)

        Negative sign: water flows downslope.

        When previous velocities are available (from prior timestep),
        we blend Manning's estimate with prev velocity for continuity.

        Returns:
            u: [B, H, W] x-velocity (m/s)
            v: [B, H, W] y-velocity (m/s)
        """
        eps   = 1e-6
        wet   = self._wet_mask(h)

        # Hydraulic radius ≈ depth for wide shallow flow
        h_safe = torch.clamp(h, min=self.min_depth)

        # Bed slope magnitude
        slope_mag = torch.sqrt(dzdx ** 2 + dzdy ** 2 + eps)

        # Manning's velocity magnitude: U = (1/n) × h^(2/3) × S^(1/2)
        speed = (1.0 / manning_n) * (h_safe ** (2.0 / 3.0)) * torch.sqrt(slope_mag)
        speed = speed * wet   # zero velocity in dry cells

        # Decompose into x,y components (flow in downslope direction)
        u_manning = -speed * dzdx / slope_mag
        v_manning = -speed * dzdy / slope_mag

        if prev_u is not None and prev_v is not None:
            # Blend: 60% Manning's estimate, 40% previous (inertia)
            u = 0.6 * u_manning + 0.4 * prev_u
            v = 0.6 * v_manning + 0.4 * prev_v
        else:
            u, v = u_manning, v_manning

        return u, v

    def continuity_residual(
        self,
        h_t0:    torch.Tensor,   # [B, H, W] — depth at time t
        h_t1:    torch.Tensor,   # [B, H, W] — depth at time t+1 (predicted)
        u:       torch.Tensor,   # [B, H, W] — x-velocity
        v:       torch.Tensor,   # [B, H, W] — y-velocity
        rainfall: torch.Tensor,  # [B, H, W] — rainfall rate (m/s)
        infiltration: Optional[torch.Tensor] = None,  # [B, H, W]
    ) -> torch.Tensor:
        """
        Continuity equation residual (mass conservation):

          R_c = ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y - (r - i)

        A well-trained model should drive R_c → 0.

        Returns:
            residual: scalar loss value
        """
        h_t0_safe = torch.clamp(h_t0, min=0.0)
        h_t1_safe = torch.clamp(h_t1, min=0.0)

        # Time derivative
        dhdt = self.fd.time_derivative(h_t1_safe, h_t0_safe, self.dt)

        # Flux divergence  ∇·(hu)
        flux_x  = h_t0_safe * u
        flux_y  = h_t0_safe * v
        div_flux = self.fd.divergence(flux_x, flux_y, self.dx, self.dy)

        # Source term: rainfall - infiltration
        if infiltration is None:
            # Simple Green-Ampt infiltration proxy: 10% of rainfall infiltrates
            infiltration = 0.10 * rainfall
        source = rainfall - infiltration

        # Residual
        residual = dhdt + div_flux - source

        # Only penalise wet cells
        wet = self._wet_mask(h_t0 + h_t1)
        residual = residual * wet

        if self.reduction == "mean":
            n_wet = wet.sum().clamp(min=1)
            return (residual ** 2).sum() / n_wet
        return (residual ** 2).sum()

    def momentum_residual(
        self,
        h:         torch.Tensor,  # [B, H, W] — water depth
        u:         torch.Tensor,  # [B, H, W] — x-velocity
        v:         torch.Tensor,  # [B, H, W] — y-velocity
        dzdx:      torch.Tensor,  # [B, H, W] — bed elevation slope x
        dzdy:      torch.Tensor,  # [B, H, W] — bed elevation slope y
        manning_n: torch.Tensor,  # [B, H, W] — Manning's n per cell
    ) -> torch.Tensor:
        """
        Momentum equation residuals (simplified shallow water, diffusive wave):

        The full Saint-Venant momentum equation is:
          ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂y =
            -gh·∂z/∂x - g·n²·u·√(u²+v²) / h^(1/3)

        For urban flooding at hourly timescales, the diffusive wave
        approximation is standard (neglects ∂(hu)/∂t term):
          0 = -gh·∂z/∂x - gh·∂h/∂x - g·n²·u·√(u²+v²) / h^(1/3)

        This simplification is physically valid when:
          Froude number Fr = U/√(gh) << 1 (subcritical flow)
          which holds for urban flooding (typical Fr < 0.1)

        Returns:
            residual: scalar loss value (x + y momentum combined)
        """
        eps    = 1e-6
        g      = self.g
        h_safe = torch.clamp(h, min=self.min_depth)
        wet    = self._wet_mask(h)

        # Flow speed magnitude
        speed = torch.sqrt(u ** 2 + v ** 2 + eps)

        # Manning's friction coefficient: Sf = n² |U| / h^(4/3)
        friction = (manning_n ** 2) * speed / (h_safe ** (4.0 / 3.0))

        # Water surface slope: ∂(z+h)/∂x, ∂(z+h)/∂y
        # (bed slope + free surface slope = total driving force)
        dhdx = self.fd.ddx(h_safe, self.dx)
        dhdy = self.fd.ddy(h_safe, self.dy)

        water_surface_slope_x = dzdx + dhdx
        water_surface_slope_y = dzdy + dhdy

        # Diffusive wave momentum residual:
        # R_mx = u + (g × h × ∂(z+h)/∂x) / (g × n² × |U|) → should = 0
        # Equivalently: u × friction = -g × h_slope_x  (rearranged)
        res_x = u * friction + g * water_surface_slope_x
        res_y = v * friction + g * water_surface_slope_y

        # Apply wet mask
        res_x = res_x * wet
        res_y = res_y * wet

        if self.reduction == "mean":
            n_wet = wet.sum().clamp(min=1)
            loss_x = (res_x ** 2).sum() / n_wet
            loss_y = (res_y ** 2).sum() / n_wet
        else:
            loss_x = (res_x ** 2).sum()
            loss_y = (res_y ** 2).sum()

        return 0.5 * (loss_x + loss_y)

    def boundary_residual(self, h: torch.Tensor) -> torch.Tensor:
        """
        Zero-flux (reflective) boundary condition at grid edges.

        At domain boundaries, no water should flow out:
          ∂h/∂n = 0  at edges  (n = outward normal)

        Implemented as: boundary cells should equal their interior neighbours.
        This penalises unrealistic depth gradients at the domain edge.

        Returns:
            residual: scalar loss value
        """
        # Edge cells should match adjacent interior cells (zero gradient)
        # Top/bottom rows
        res_top    = h[..., 0,  :] - h[..., 1,   :]
        res_bottom = h[..., -1, :] - h[..., -2,  :]
        # Left/right columns
        res_left   = h[..., :, 0 ] - h[..., :,  1]
        res_right  = h[..., :, -1] - h[..., :, -2]

        all_res = torch.cat([
            res_top.flatten(),
            res_bottom.flatten(),
            res_left.flatten(),
            res_right.flatten(),
        ])

        if self.reduction == "mean":
            return (all_res ** 2).mean()
        return (all_res ** 2).sum()

    def forward(
        self,
        predictions:   torch.Tensor,          # [B, T_out, 1, H, W] predicted depth
        h_t0:          torch.Tensor,           # [B, H, W] depth at last input step
        rainfall_seq:  torch.Tensor,           # [B, T_out, H, W] rainfall during forecast
        elevation:     torch.Tensor,           # [B, H, W] raw elevation (metres)
        manning_n:     Optional[torch.Tensor], # [B, H, W] Manning's n per cell
        prev_u:        Optional[torch.Tensor] = None,
        prev_v:        Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined physics loss across all forecast timesteps.

        Args:
            predictions:   Model output [B, T_out, 1, H, W]
            h_t0:          Last known depth before forecast [B, H, W]
            rainfall_seq:  Rainfall during forecast horizon [B, T_out, H, W]
            elevation:     Bed elevation in metres [B, H, W]
            manning_n:     Manning's roughness [B, H, W] or None (uses default)
            prev_u/v:      Previous velocities for momentum initialisation

        Returns:
            total_physics_loss: scalar
            loss_components: dict with "continuity", "momentum", "boundary"
        """
        B, T_out, _, H, W = predictions.shape
        device = predictions.device

        if manning_n is None:
            manning_n = torch.full((B, H, W), self.manning_default, device=device)

        # Bed elevation slopes (static — don't change with time)
        dzdx = self.fd.ddx(elevation, self.dx)
        dzdy = self.fd.ddy(elevation, self.dy)

        # Rainfall rate: convert mm/h → m/s
        # rainfall_seq is in mm, dt=3600s
        rainfall_ms = rainfall_seq / (1000.0 * self.dt)  # m/s

        loss_continuity = torch.tensor(0.0, device=device)
        loss_momentum   = torch.tensor(0.0, device=device)
        loss_boundary   = torch.tensor(0.0, device=device)

        h_prev  = h_t0
        u_prev  = prev_u
        v_prev  = prev_v

        for t in range(T_out):
            h_pred = predictions[:, t, 0, :, :]   # [B, H, W]
            rain_t = rainfall_ms[:, t, :, :]       # [B, H, W]

            # Estimate velocities from depth + terrain
            u, v = self._estimate_velocity(
                h       = h_pred,
                dzdx    = dzdx,
                dzdy    = dzdy,
                manning_n = manning_n,
                prev_u  = u_prev,
                prev_v  = v_prev,
            )

            # Continuity residual
            loss_continuity = loss_continuity + self.continuity_residual(
                h_t0     = h_prev,
                h_t1     = h_pred,
                u        = u,
                v        = v,
                rainfall = rain_t,
            )

            # Momentum residual
            loss_momentum = loss_momentum + self.momentum_residual(
                h         = h_pred,
                u         = u,
                v         = v,
                dzdx      = dzdx,
                dzdy      = dzdy,
                manning_n = manning_n,
            )

            # Boundary condition
            loss_boundary = loss_boundary + self.boundary_residual(h_pred)

            # Advance state
            h_prev = h_pred.detach()
            u_prev = u.detach()
            v_prev = v.detach()

        # Average over timesteps
        loss_continuity = loss_continuity / T_out
        loss_momentum   = loss_momentum   / T_out
        loss_boundary   = loss_boundary   / T_out

        total = (
            self.lambda_continuity * loss_continuity
            + self.lambda_momentum * loss_momentum
            + self.lambda_boundary * loss_boundary
        )

        return total, {
            "continuity": loss_continuity.detach(),
            "momentum":   loss_momentum.detach(),
            "boundary":   loss_boundary.detach(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loss
# ═══════════════════════════════════════════════════════════════════════════════

class FloodDataLoss(nn.Module):
    """
    Data-fitting loss between predicted and observed flood depths.

    Combines:
      - MSE on flooded cells (depth accuracy)
      - BCE on wet/dry classification (flood extent)
      - Depth-weighted MSE (penalises deep flood errors more)

    Args:
        flood_threshold:    Depth threshold for wet/dry classification (m)
        weight_flooded:     Extra weight on flooded cells (rare event emphasis)
        lambda_mse:         Weight of MSE term
        lambda_bce:         Weight of BCE classification term
        lambda_depth_wt:    Weight of depth-weighted MSE term
    """

    def __init__(
        self,
        flood_threshold: float = 0.20,
        weight_flooded:  float = 8.0,   # 3.8% prevalence needs ~25x weight
        lambda_mse:      float = 1.0,
        lambda_bce:      float = 0.5,   # raised: class signal must overcome imbalance
        lambda_depth_wt: float = 0.5,
    ):
        super().__init__()
        self.flood_threshold = flood_threshold
        self.weight_flooded  = weight_flooded
        self.lambda_mse      = lambda_mse
        self.lambda_bce      = lambda_bce
        self.lambda_depth_wt = lambda_depth_wt

    def forward(
        self,
        predictions: torch.Tensor,  # [B, T_out, 1, H, W]
        targets:     torch.Tensor,  # [B, T_out, H, W]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            total_data_loss: scalar
            loss_components: dict with "mse", "bce", "depth_weighted"
        """
        pred = predictions[:, :, 0, :, :]   # [B, T_out, H, W]
        tgt  = targets                        # [B, T_out, H, W]

        # ── Pixel weights: flood cells weighted more ────────────────────────
        flooded   = (tgt > self.flood_threshold).float()
        not_flood = 1.0 - flooded
        weights   = not_flood + self.weight_flooded * flooded
        # Normalise weights so total weight = number of pixels
        weights   = weights / weights.mean().clamp(min=1e-6)

        # ── MSE loss (weighted) ────────────────────────────────────────────
        mse = (weights * (pred - tgt) ** 2).mean()

        # ── BCE classification loss (FP16-safe, imbalance-corrected) ────────
        # pos_weight corrects for class imbalance: 3.8% flooded cells means
        # a missed flood should be penalised ~25x more than a false alarm.
        # Computed dynamically from the batch so it adapts to each sample.
        n_flooded   = flooded.sum().clamp(min=1)
        n_dry       = (flooded.numel() - n_flooded).clamp(min=1)
        pos_weight  = (n_dry / n_flooded).clamp(max=50.0)  # cap at 50x
        pred_logits = (pred - self.flood_threshold) * 10.0
        tgt_binary  = flooded
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, tgt_binary,
            pos_weight=pos_weight.expand_as(tgt_binary),
        )

        # ── Depth-weighted MSE ─────────────────────────────────────────────
        # Errors at greater depth matter more (deeper floods are more dangerous)
        depth_weights = 1.0 + tgt / (tgt.max().clamp(min=1e-6))
        depth_mse = (depth_weights * (pred - tgt) ** 2).mean()

        total = (
            self.lambda_mse      * mse
            + self.lambda_bce      * bce
            + self.lambda_depth_wt * depth_mse
        )

        return total, {
            "mse":           mse.detach(),
            "bce":           bce.detach(),
            "depth_weighted": depth_mse.detach(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PINN Wrapper — combines data loss + physics loss
# ═══════════════════════════════════════════════════════════════════════════════

class PINNWrapper(nn.Module):
    """
    Wraps ConvLSTM model with physics-informed training.

    Total loss:
      L_total = L_data + L_physics
             = [MSE + BCE + depth_MSE]
               + λ_c × continuity + λ_m × momentum + λ_b × boundary

    The physics weight (lambda_physics) can be scheduled during training:
      - Start low (0.1) so the model first learns from data
      - Ramp up to full weight after ~10 epochs
      This prevents physics constraints from dominating before the model
      has learned basic flood patterns.

    Args:
        convlstm_model:  FloodConvLSTM instance
        physics_config:  Dict from config["physics"]
        flood_threshold: Depth threshold for flood classification (m)
        lambda_physics:  Overall physics loss weight (scales all physics terms)
    """

    def __init__(
        self,
        convlstm_model,
        physics_config:  dict,
        flood_threshold: float = 0.20,
        lambda_physics:  float = 1.0,
    ):
        super().__init__()

        self.model          = convlstm_model
        self.lambda_physics = lambda_physics

        self.physics_loss = PhysicsLoss(
            gravity           = physics_config.get("gravity",           9.81),
            dt                = physics_config.get("dt",                3600.0),
            dx                = 50.0,   # 50m grid (from config grid.resolution_m)
            dy                = 50.0,
            min_depth         = physics_config.get("min_depth",         0.001),
            manning_default   = physics_config.get("manning_default",   0.035),
            lambda_continuity = physics_config.get("lambda_continuity", 0.5),
            lambda_momentum   = physics_config.get("lambda_momentum",   0.3),
            lambda_boundary   = physics_config.get("lambda_boundary",   0.1),
        )

        self.data_loss = FloodDataLoss(
            flood_threshold = flood_threshold,
            weight_flooded  = 8.0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Direct forward pass — returns predictions [B, T_out, 1, H, W]."""
        return self.model(inputs)

    def compute_loss(
        self,
        predictions:  torch.Tensor,           # [B, T_out, 1, H, W]
        targets:      torch.Tensor,            # [B, T_out, H, W]
        inputs:       torch.Tensor,            # [B, T_in, 8, H, W]
        elevation:    torch.Tensor,            # [B, H, W] raw elevation (m)
        manning_n:    Optional[torch.Tensor],  # [B, H, W] or None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total PINN loss.

        Args:
            predictions:  ConvLSTM output [B, T_out, 1, H, W]
            targets:      SAR flood depth ground truth [B, T_out, H, W]
            inputs:       Full input sequence [B, T_in, 8, H, W]
            elevation:    Raw (un-normalised) terrain elevation [B, H, W]
            manning_n:    Per-cell Manning's n [B, H, W] or None

        Returns:
            total_loss:   scalar — total training loss
            loss_dict:    dict with all individual loss components
        """
        # ── Data loss ─────────────────────────────────────────────────────
        l_data, data_components = self.data_loss(predictions, targets)

        # ── Extract inputs for physics ────────────────────────────────────
        # Last input timestep depth (channel 7 = prev_flood_depth)
        h_t0 = inputs[:, -1, 7, :, :]   # [B, H, W]

        # Rainfall during forecast: use channel 0 of each input step
        # (We repeat the last 3 rainfall steps as proxy for forecast rainfall)
        T_out = predictions.shape[1]
        rainfall_last = inputs[:, -T_out:, 0, :, :]  # [B, T_out, H, W]
        # Denormalise rainfall: was log1p normalised with max=150mm
        rainfall_mm = (torch.exp(rainfall_last * torch.log(
            torch.tensor(151.0, device=inputs.device)
        )) - 1.0).clamp(min=0)

        # ── Physics loss ──────────────────────────────────────────────────
        l_physics, physics_components = self.physics_loss(
            predictions   = predictions,
            h_t0          = h_t0,
            rainfall_seq  = rainfall_mm,
            elevation     = elevation,
            manning_n     = manning_n,
        )

        # ── Total loss ────────────────────────────────────────────────────
        total = l_data + self.lambda_physics * l_physics

        loss_dict = {
            "total":          total.detach(),
            "data":           l_data.detach(),
            "physics":        l_physics.detach(),
            **{f"data_{k}":   v for k, v in data_components.items()},
            **{f"phys_{k}":   v for k, v in physics_components.items()},
        }

        return total, loss_dict

    def set_physics_weight(self, weight: float):
        """Adjust physics loss weight during training (for scheduling)."""
        self.lambda_physics = weight

    @staticmethod
    def physics_weight_schedule(epoch: int, warmup_epochs: int = 10) -> float:
        """
        Linear ramp from 0.1 → 1.0 over warmup_epochs.
        Use in train.py:
          weight = PINNWrapper.physics_weight_schedule(epoch)
          wrapper.set_physics_weight(weight)
        """
        if epoch >= warmup_epochs:
            return 1.0
        return 0.1 + 0.9 * (epoch / warmup_epochs)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_pinn(config: dict, convlstm_model=None) -> PINNWrapper:
    """
    Build PINNWrapper from full config dict.

    Args:
        config:          Full config dict (from config.yaml)
        convlstm_model:  Pre-built FloodConvLSTM, or None to build from config

    Returns:
        PINNWrapper ready for training
    """
    if convlstm_model is None:
        from models.convlstm import build_convlstm
        convlstm_model = build_convlstm(config["model"])

    return PINNWrapper(
        convlstm_model  = convlstm_model,
        physics_config  = config.get("physics", {}),
        flood_threshold = config.get("evaluation", {}).get("flood_threshold_m", 0.20),
        lambda_physics  = 1.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yaml

    print("=" * 60)
    print("  PINN — smoke test")
    print("=" * 60)

    torch.manual_seed(42)
    B, T_in, T_out = 2, 6, 3
    H, W = 50, 50

    # ── Standalone PhysicsLoss ────────────────────────────────────────────
    phys = PhysicsLoss()
    preds   = torch.rand(B, T_out, 1, H, W) * 0.5
    h_t0    = torch.rand(B, H, W) * 0.3
    rain    = torch.rand(B, T_out, H, W) * 10.0
    elev    = torch.rand(B, H, W) * 5.0
    manning = torch.full((B, H, W), 0.035)

    pl, pd = phys(preds, h_t0, rain, elev, manning)
    assert pl.item() >= 0, "Negative physics loss"
    print(f"✓  PhysicsLoss forward OK")
    print(f"   continuity={pd['continuity']:.4f}  "
          f"momentum={pd['momentum']:.4f}  "
          f"boundary={pd['boundary']:.4f}")

    # ── Standalone FloodDataLoss ──────────────────────────────────────────
    data_loss = FloodDataLoss()
    tgts = torch.rand(B, T_out, H, W) * 0.5
    dl, dd = data_loss(preds, tgts)
    assert dl.item() >= 0
    print(f"✓  FloodDataLoss forward OK  "
          f"mse={dd['mse']:.4f} bce={dd['bce']:.4f}")

    # ── Full PINNWrapper ──────────────────────────────────────────────────
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        wrapper = build_pinn(cfg)
        print("  Built from config.yaml")
    except FileNotFoundError:
        from models.convlstm import FloodConvLSTM
        wrapper = PINNWrapper(
            convlstm_model = FloodConvLSTM(),
            physics_config = {},
        )
        cfg = {"model": {"output_steps": 3}}
        print("  Built with defaults")

    inputs = torch.randn(B, T_in, 8, H, W)
    with torch.no_grad():
        preds_full = wrapper(inputs)
    assert preds_full.shape == (B, T_out, 1, H, W)
    print(f"✓  PINNWrapper forward OK: {tuple(preds_full.shape)}")

    # Loss needs gradients — use autograd
    inputs_g = torch.randn(B, T_in, 8, H, W, requires_grad=False)
    preds_g  = wrapper(inputs_g)
    tgts_g   = torch.rand(B, T_out, H, W)
    elev_g   = torch.rand(B, H, W) * 5.0

    total, ld = wrapper.compute_loss(preds_g, tgts_g, inputs_g, elev_g, None)
    assert total.item() >= 0
    print(f"✓  compute_loss OK  total={total.item():.4f}")
    print(f"   data={ld['data']:.4f}  physics={ld['physics']:.4f}")
    print(f"   continuity={ld['phys_continuity']:.4f}  "
          f"momentum={ld['phys_momentum']:.4f}")

    # Physics weight schedule
    for ep in [0, 5, 10, 15]:
        w = PINNWrapper.physics_weight_schedule(ep, warmup_epochs=10)
        print(f"   epoch {ep:2d} → physics_weight={w:.2f}")

    print("\n  All checks passed ✓")