"""
models/convlstm.py
Convolutional LSTM for spatiotemporal flood prediction.

Architecture:
  ConvLSTMCell   — single cell: replaces FC gates with conv gates
  ConvLSTMLayer  — stacks T cells over a time sequence
  ConvLSTMEncoder — 3-layer encoder that compresses [B, T, C, H, W] → hidden states
  ConvLSTMDecoder — rolls out hidden states → [B, T_out, C_out, H, W]
  FloodConvLSTM   — full encoder-decoder assembled and ready for training

Key differences from standard LSTM:
  - Gates use 2D convolutions instead of matrix multiplications
  - Hidden state h and cell state c are spatial maps [B, hidden, H, W]
  - Captures both spatial patterns (convolution) and temporal dynamics (LSTM)
  - Peephole connections between cell state and gates (improves gradient flow)

References:
  Shi et al. (2015). Convolutional LSTM Network: A Machine Learning Approach
  for Precipitation Nowcasting. NeurIPS 2015.
  https://arxiv.org/abs/1506.04214

Usage:
  from models.convlstm import FloodConvLSTM, build_convlstm

  model = build_convlstm(config["model"])
  # inputs: [B, T_in=6, C=8, H, W]
  outputs = model(inputs)
  # outputs: [B, T_out=3, C_out=1, H, W]  — flood depth maps
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# ConvLSTM Cell
# ═══════════════════════════════════════════════════════════════════════════════

class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.

    Gate equations (all convolutions, not FC):
      i = sigmoid( W_xi * x_t + W_hi * h_{t-1} + W_ci ⊙ c_{t-1} + b_i )
      f = sigmoid( W_xf * x_t + W_hf * h_{t-1} + W_cf ⊙ c_{t-1} + b_f )
      g = tanh(    W_xg * x_t + W_hg * h_{t-1}                   + b_g )
      o = sigmoid( W_xo * x_t + W_ho * h_{t-1} + W_co ⊙ c_t     + b_o )
      c_t = f ⊙ c_{t-1} + i ⊙ g
      h_t = o ⊙ tanh(c_t)

    ⊙ = Hadamard product (elementwise)
    * = convolution

    Args:
        input_channels:  Number of input feature channels C
        hidden_channels: Number of hidden state channels H_c
        kernel_size:     Convolution kernel size (default 3 → 3×3)
        peephole:        Use peephole connections (W_ci, W_cf, W_co)
        dropout:         Zoneout-style dropout on hidden state
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        peephole: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.peephole = peephole
        self.dropout_p = dropout

        pad = kernel_size // 2   # same-padding: output spatial size = input size

        # Fused gate convolution: compute i, f, g, o in one pass (4× channels)
        # Input path: x_t → gates
        self.conv_x = nn.Conv2d(
            input_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=pad, bias=True
        )
        # Hidden path: h_{t-1} → gates (no bias — added by conv_x already)
        self.conv_h = nn.Conv2d(
            hidden_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=pad, bias=False
        )

        # Peephole weights: c_{t-1} → i, f gates and c_t → o gate
        # These are 1×1 pointwise (scalar per channel per spatial location)
        if peephole:
            self.W_ci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
            self.W_cf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
            self.W_co = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        # Layer normalisation on cell state (stabilises training)
        self.ln_c = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        self.ln_h = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)

        # Dropout on hidden state output
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialise weights for stable training."""
        # Orthogonal init for hidden-to-hidden convolution
        nn.init.orthogonal_(self.conv_h.weight)
        # Xavier for input convolution
        nn.init.xavier_uniform_(self.conv_x.weight)
        # Forget gate bias = 1.0 (helps remember long-term patterns)
        nn.init.constant_(self.conv_x.bias[self.hidden_channels:2*self.hidden_channels], 1.0)

    def forward(
        self,
        x: torch.Tensor,                           # [B, C_in, H, W]
        state: Tuple[torch.Tensor, torch.Tensor],  # (h, c) each [B, H_c, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_next: [B, hidden_channels, H, W]
            c_next: [B, hidden_channels, H, W]
        """
        h, c = state

        # Fused gate computation: [B, 4*H_c, H, W]
        gates = self.conv_x(x) + self.conv_h(h)

        # Split into 4 gates
        i, f, g, o = gates.chunk(4, dim=1)   # each [B, H_c, H, W]

        # Peephole connections
        if self.peephole:
            i = torch.sigmoid(i + self.W_ci * c)
            f = torch.sigmoid(f + self.W_cf * c)
        else:
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)

        g = torch.tanh(g)

        # Cell state update
        c_next = f * c + i * g
        c_next = self.ln_c(c_next)   # normalise cell state

        if self.peephole:
            o = torch.sigmoid(o + self.W_co * c_next)
        else:
            o = torch.sigmoid(o)

        # Hidden state
        h_next = o * torch.tanh(c_next)
        h_next = self.ln_h(h_next)
        h_next = self.dropout(h_next)

        return h_next, c_next

    def init_hidden(
        self, batch_size: int, H: int, W: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialise hidden and cell states to zero."""
        shape = (batch_size, self.hidden_channels, H, W)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ConvLSTM Layer (one layer across T timesteps)
# ═══════════════════════════════════════════════════════════════════════════════

class ConvLSTMLayer(nn.Module):
    """
    One layer of ConvLSTM: processes a full sequence [B, T, C, H, W].

    Args:
        input_channels:  Channels in input sequence
        hidden_channels: Hidden state channels
        kernel_size:     Conv kernel size
        return_sequence: If True return all hidden states; else only last
        peephole:        Use peephole connections
        dropout:         Dropout on hidden state
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        return_sequence: bool = True,
        peephole: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.return_sequence = return_sequence
        self.hidden_channels = hidden_channels

        self.cell = ConvLSTMCell(
            input_channels  = input_channels,
            hidden_channels = hidden_channels,
            kernel_size     = kernel_size,
            peephole        = peephole,
            dropout         = dropout,
        )

    def forward(
        self,
        x: torch.Tensor,                                        # [B, T, C, H, W]
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:             Input sequence [B, T, C, H, W]
            initial_state: Optional (h_0, c_0) — defaults to zeros

        Returns:
            output:        [B, T, H_c, H, W] if return_sequence else [B, H_c, H, W]
            final_state:   (h_T, c_T) — last hidden and cell state
        """
        B, T, C, H, W = x.shape
        device = x.device

        if initial_state is None:
            h, c = self.cell.init_hidden(B, H, W, device)
        else:
            h, c = initial_state

        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
            if self.return_sequence:
                outputs.append(h)

        if self.return_sequence:
            output = torch.stack(outputs, dim=1)   # [B, T, H_c, H, W]
        else:
            output = h                              # [B, H_c, H, W]

        return output, (h, c)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-layer ConvLSTM Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class ConvLSTMEncoder(nn.Module):
    """
    Stacked ConvLSTM encoder.

    Processes input sequence through N layers of ConvLSTM.
    Each layer takes the hidden state sequence from the previous layer as input.

    Returns the final hidden states of all layers — these are passed to
    the decoder to initialise its hidden states.

    Args:
        input_channels:   C — input channels (8 for our project)
        hidden_channels:  List of hidden channels per layer e.g. [64, 64, 32]
        kernel_size:      Kernel size (same for all layers)
        dropout:          Dropout applied to hidden states (except last layer)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_layers = len(hidden_channels)
        self.hidden_channels = hidden_channels

        layers = []
        for i, h_ch in enumerate(hidden_channels):
            in_ch  = input_channels if i == 0 else hidden_channels[i - 1]
            dp     = dropout if i < self.n_layers - 1 else 0.0  # no dropout on last
            layers.append(ConvLSTMLayer(
                input_channels  = in_ch,
                hidden_channels = h_ch,
                kernel_size     = kernel_size,
                return_sequence = True,   # encoder always returns full sequence
                peephole        = True,
                dropout         = dp,
            ))

        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: [B, T_in, C, H, W]

        Returns:
            last_seq:    Final layer's hidden sequence [B, T_in, H_last, H, W]
            all_states:  List of (h_T, c_T) per layer — used to init decoder
        """
        current = x
        all_states = []

        for layer in self.layers:
            current, state = layer(current)
            all_states.append(state)

        return current, all_states   # current = [B, T_in, H_last, H, W]


# ═══════════════════════════════════════════════════════════════════════════════
# ConvLSTM Decoder
# ═══════════════════════════════════════════════════════════════════════════════

class ConvLSTMDecoder(nn.Module):
    """
    Autoregressive ConvLSTM decoder.

    At each output step:
      1. Takes the last predicted depth map + terrain features as input
      2. Runs through mirrored ConvLSTM layers
      3. Projects hidden state → flood depth prediction

    Initialised from the encoder's final hidden states (seq2seq style).

    Args:
        hidden_channels:  Must match encoder (reversed) e.g. [32, 64, 64]
        out_channels:     Output channels — 1 (depth only) or 3 (h, u, v)
        kernel_size:      Conv kernel size
        forecast_steps:   T_out — number of steps to forecast
        terrain_channels: Static terrain channels concatenated at each step
        dropout:          MC Dropout for uncertainty quantification
    """

    def __init__(
        self,
        hidden_channels: List[int],
        out_channels: int = 1,
        kernel_size: int = 3,
        forecast_steps: int = 3,
        terrain_channels: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.forecast_steps  = forecast_steps
        self.out_channels    = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers        = len(hidden_channels)

        # Input at each decode step:
        # prev_depth(1) + terrain(terrain_channels) = 1 + 6 = 7 channels
        decode_input_ch = out_channels + terrain_channels

        layers = []
        for i, h_ch in enumerate(hidden_channels):
            in_ch = decode_input_ch if i == 0 else hidden_channels[i - 1]
            layers.append(ConvLSTMLayer(
                input_channels  = in_ch,
                hidden_channels = h_ch,
                kernel_size     = kernel_size,
                return_sequence = False,   # decoder: one step at a time
                peephole        = True,
                dropout         = dropout,  # keep dropout ON for MC Dropout inference
            ))

        self.layers = nn.ModuleList(layers)

        # Output projection: hidden → flood depth
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], hidden_channels[-1] // 2,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels[-1] // 2, out_channels,
                      kernel_size=1),
            nn.Softplus()
        )

    def forward(
        self,
        encoder_states: List[Tuple[torch.Tensor, torch.Tensor]],
        terrain: torch.Tensor,               # [B, terrain_channels, H, W]
        prev_depth: Optional[torch.Tensor],  # [B, out_channels, H, W] — last known depth
    ) -> torch.Tensor:
        """
        Args:
            encoder_states: List of (h, c) per encoder layer
            terrain:        Static terrain features [B, 6, H, W]
            prev_depth:     Last known flood depth [B, 1, H, W]

        Returns:
            predictions: [B, T_out, out_channels, H, W]
        """
        B = terrain.shape[0]
        H, W = terrain.shape[2], terrain.shape[3]
        device = terrain.device

        # Initialise decoder hidden states from encoder
        # Encoder layers: [64, 64, 32] — decoder layers: [32, 64, 64] (mirrored)
        # We reverse so the deepest encoder state feeds the first decoder layer
        states = list(reversed(encoder_states))

        # Initial depth: zeros if not provided
        if prev_depth is None:
            prev_depth = torch.zeros(B, self.out_channels, H, W, device=device)

        predictions = []

        for _ in range(self.forecast_steps):
            # Input: prev prediction + static terrain
            dec_input = torch.cat([prev_depth, terrain], dim=1)   # [B, 7, H, W]
            dec_input = dec_input.unsqueeze(1)                     # [B, 1, 7, H, W]

            # Run through each decoder layer
            new_states = []
            current = dec_input
            for i, layer in enumerate(self.layers):
                out, new_state = layer(current, initial_state=states[i])
                new_states.append(new_state)
                current = out.unsqueeze(1)   # [B, 1, H_c, H, W] for next layer

            states = new_states

            # Project hidden state → depth prediction
            h_last = current.squeeze(1)              # [B, H_last, H, W]
            depth_pred = self.output_head(h_last)    # [B, out_channels, H, W]

            predictions.append(depth_pred)
            prev_depth = depth_pred.detach()         # use prediction as next input

        # Stack: [B, T_out, out_channels, H, W]
        return torch.stack(predictions, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Encoder-Decoder Model
# ═══════════════════════════════════════════════════════════════════════════════

class FloodConvLSTM(nn.Module):
    """
    Full ConvLSTM encoder-decoder for flood depth forecasting.

    Input:  [B, T_in=6, C=8, H, W]
            C = rainfall(1) + terrain(6) + prev_depth(1)

    Output: [B, T_out=3, 1, H, W]
            Flood depth in metres at T+1h, T+2h, T+3h

    The terrain channels are extracted from the input and also passed
    directly to the decoder at each forecast step (they don't change
    over time, so there's no point running them through the LSTM encoder).

    Args:
        input_channels:      Total input channels (8)
        terrain_channels:    Static terrain channels (6)
        hidden_channels:     Per-layer hidden channels e.g. [64, 64, 32]
        kernel_size:         Conv kernel size for all cells
        out_channels:        1 (depth only) — physics head adds u,v later
        forecast_steps:      T_out = 3
        encoder_dropout:     Dropout in encoder layers
        decoder_dropout:     Dropout in decoder — kept ON during inference
                             for Monte Carlo Dropout uncertainty estimation
    """

    def __init__(
        self,
        input_channels:   int = 8,
        terrain_channels: int = 6,
        hidden_channels:  List[int] = None,
        kernel_size:      int = 3,
        out_channels:     int = 1,
        forecast_steps:   int = 3,
        encoder_dropout:  float = 0.2,
        decoder_dropout:  float = 0.2,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [64, 64, 32]

        self.terrain_channels = terrain_channels
        self.forecast_steps   = forecast_steps
        self.out_channels     = out_channels
        # Channel index where terrain starts in input tensor
        # Input layout: [rainfall(1), terrain(6), prev_depth(1)]
        self.terrain_start = 1
        self.terrain_end   = 1 + terrain_channels   # indices 1:7

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = ConvLSTMEncoder(
            input_channels  = input_channels,
            hidden_channels = hidden_channels,
            kernel_size     = kernel_size,
            dropout         = encoder_dropout,
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        # Mirror the encoder hidden channels
        decoder_hidden = list(reversed(hidden_channels))
        self.decoder = ConvLSTMDecoder(
            hidden_channels  = decoder_hidden,
            out_channels     = out_channels,
            kernel_size      = kernel_size,
            forecast_steps   = forecast_steps,
            terrain_channels = terrain_channels,
            dropout          = decoder_dropout,
        )

        # ── Input projection ─────────────────────────────────────────────────
        # Optional: project raw input to a richer representation
        # Helps the model learn feature interactions before temporal processing
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1),
            nn.GroupNorm(num_groups=4, num_channels=input_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Apply sensible weight initialisation to conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,                            # [B, T_in, C, H, W]
        prev_depth: Optional[torch.Tensor] = None,  # [B, 1, H, W]
    ) -> torch.Tensor:
        """
        Args:
            x:          Input sequence [B, T_in, C=8, H, W]
            prev_depth: Optional override for initial depth [B, 1, H, W]
                        Defaults to last timestep's prev_depth channel

        Returns:
            predictions: [B, T_out=3, 1, H, W]  — flood depth in metres
        """
        B, T, C, H, W = x.shape

        # Apply input projection to each timestep
        x_proj = x.view(B * T, C, H, W)
        x_proj = self.input_proj(x_proj)
        x_proj = x_proj.view(B, T, C, H, W)

        # Extract static terrain from last timestep (same across all T)
        # Terrain channels 1:7 don't change with time
        terrain = x[:, -1, self.terrain_start:self.terrain_end, :, :]  # [B, 6, H, W]

        # Previous depth: use last timestep's prev_depth channel (index 7)
        if prev_depth is None:
            prev_depth = x[:, -1, 7:8, :, :]   # [B, 1, H, W]

        # Encode
        _, encoder_states = self.encoder(x_proj)   # states: [(h,c) × n_layers]

        # Decode
        predictions = self.decoder(
            encoder_states = encoder_states,
            terrain        = terrain,
            prev_depth     = prev_depth,
        )   # [B, T_out, 1, H, W]

        return predictions

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_passes: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout uncertainty estimation.

        Runs N stochastic forward passes with dropout ACTIVE.
        Returns mean prediction + per-cell uncertainty (std dev).

        Args:
            x:        Input [B, T_in, C, H, W]
            n_passes: Number of MC samples (more = better estimate, slower)

        Returns:
            mean_pred:   [B, T_out, 1, H, W] — mean flood depth
            uncertainty: [B, T_out, 1, H, W] — std dev across passes
        """
        self.train()   # activate dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds.append(self.forward(x))

        self.eval()
        preds = torch.stack(preds, dim=0)   # [n_passes, B, T_out, 1, H, W]
        mean_pred   = preds.mean(dim=0)
        uncertainty = preds.std(dim=0)
        return mean_pred, uncertainty

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_convlstm(model_config: dict) -> FloodConvLSTM:
    """
    Build FloodConvLSTM from config.yaml model section.

    Args:
        model_config: dict from config["model"]

    Returns:
        FloodConvLSTM ready for training
    """
    conv_cfg    = model_config.get("convlstm", {})
    decoder_cfg = model_config.get("decoder",  {})

    return FloodConvLSTM(
        input_channels   = conv_cfg.get("input_channels",  8),
        terrain_channels = 6,
        hidden_channels  = conv_cfg.get("hidden_channels", [64, 64, 32]),
        kernel_size      = conv_cfg.get("kernel_size",     3),
        out_channels     = 1,   # depth only — physics head adds u,v
        forecast_steps   = model_config.get("output_steps", 3),
        encoder_dropout  = conv_cfg.get("dropout", 0.2),
        decoder_dropout  = conv_cfg.get("dropout", 0.2),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yaml

    print("=" * 60)
    print("  FloodConvLSTM — smoke test")
    print("=" * 60)

    # Load config if available, else use defaults
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        model = build_convlstm(cfg["model"])
        print("  Built from config.yaml")
    except FileNotFoundError:
        model = FloodConvLSTM()
        print("  Built with defaults (config.yaml not found)")

    model.eval()
    params = model.count_parameters()
    print(f"  Parameters: {params['total']:,} total | {params['trainable']:,} trainable")

    # Forward pass
    B, T_in, C, H, W = 2, 6, 8, 50, 50
    dummy = torch.randn(B, T_in, C, H, W)

    with torch.no_grad():
        out = model(dummy)
    print(f"\n  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(out.shape)}  — expected [2, 3, 1, 50, 50]")
    assert out.shape == (B, 3, 1, H, W), f"Shape mismatch: {out.shape}"
    assert (out >= 0).all(), "Negative depth predicted — output activation broken"
    print(f"  Depth range: [{out.min():.4f}, {out.max():.4f}] m  (all ≥ 0 ✓)")

    # Uncertainty estimation
    mean, unc = model.predict_with_uncertainty(dummy, n_passes=5)
    print(f"\n  MC Dropout ({5} passes):")
    print(f"  Mean depth:   {mean.mean():.4f} m")
    print(f"  Uncertainty:  {unc.mean():.4f} m (std dev)")

    # Per-timestep output check
    print(f"\n  Per-forecast-step mean depth:")
    for t in range(out.shape[1]):
        print(f"    T+{t+1}h: {out[:, t].mean():.4f} m")

    print("\n  All checks passed ✓")