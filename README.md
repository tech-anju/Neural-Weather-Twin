# Neural Weather Twin — Hyperlocal Flood Prediction

> Physics-Informed Neural Network for street-level flood prediction in Indian monsoon cities.
> **50m resolution · 3-hour advance warning · Saint-Venant equations in the loss function**

---

## The Problem

Every monsoon season, cities like Kolkata, Chennai, and Mumbai flood — with **zero advance warning at street level**. IMD provides city-wide alerts but cannot tell emergency services *which specific wards* will flood, *how deep*, or *how soon*.

**September 21, 2025 — Kolkata:** 94 mm of rain in 3 hours. 15+ wards inundated. No block-level warning.

## Our Solution

A **Physics-Informed Neural Network (PINN)** that creates a digital twin of a city's drainage and terrain system, predicting flood depth per 50m street block up to **3 hours ahead**.

The key innovation: **Saint-Venant shallow water equations** are embedded as differentiable residual terms inside the PyTorch training loss. The model must satisfy real hydraulic physics — not just fit historical data.

```
Total Loss = MSE(predicted depth, SAR observed depth)
           + λ₁ × Continuity residual   (∂h/∂t + ∇·(hu) = r)
           + λ₂ × Momentum residual     (diffusive wave)
           + λ₃ × Boundary condition
```

| | Existing tools | Neural Weather Twin |
|---|---|---|
| Spatial resolution | City-wide (10 km) | Street-block (50 m) |
| Advance warning | Reactive | 3 hours ahead |
| Physics | None / separate model | Baked into training loss |
| Inference time | Hours | < 30 seconds |
| Hardware needed | Supercomputer | Single GPU laptop |

---

## Demo — What Judges See

```bash
# 30-second terminal demo
python scripts/kolkata_demo.py

# Interactive Streamlit app
streamlit run app/app.py
```

The demo replays the September 2024 Kolkata flood hour by hour and shows:

```
⚡ Model issued FLOOD WARNING 3 hours before the actual inundation
   reached WARNING level — giving emergency services time to:
   • Pre-position rescue boats in Salt Lake
   • Issue ward-level evacuation advisories
   • Deploy emergency pumping units
```

---

## Project Structure

```
flood_twin/
│
├── config.yaml                  ← All hyperparameters and thresholds
├── requirements.txt             ← Python dependencies
├── train.py                     ← Training script (run overnight before hackathon)
├── evaluate.py                  ← Evaluation: CSI / POD / FAR / FSS metrics
│
├── data/
│   ├── ingest.py                ← Download IMD, SRTM DEM, OSM, Sentinel-1 SAR
│   ├── grid.py                  ← Build 50m spatial grid with terrain features
│   └── dataset.py               ← PyTorch Dataset: rainfall → flood depth sequences
│
├── models/
│   ├── convlstm.py              ← ConvLSTM encoder-decoder (spatiotemporal learning)
│   ├── pinn.py                  ← Physics loss: Saint-Venant equations (PyTorch)
│   └── weather_twin.py          ← Full model: ConvLSTM + PINN + alerts + checkpointing
│
├── utils/
│   ├── physics.py               ← Manning's n table, Green-Ampt infiltration, hydraulics
│   └── metrics.py               ← CSI, POD, FAR, FSS, AUC with bootstrap CIs
│
├── app/
│   ├── app.py                   ← Streamlit demo (4 tabs: forecast / replay / metrics / physics)
│   └── inference.py             ← Production inference engine + FastAPI REST endpoint
│
└── scripts/
    └── kolkata_demo.py          ← CLI replay of Sept 2024 Kolkata flood event
```

---

## Quickstart

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Generate data (synthetic, no credentials needed)

```bash
python data/ingest.py --city kolkata --demo
python data/grid.py --city kolkata
```

### Step 3 — Train the model

```bash
# Quick smoke test (2 epochs, ~10 min on CPU — verifies pipeline works)
python train.py --config config.yaml --demo --smoke_test

# Full training (60 epochs — run overnight on GPU)
python train.py --config config.yaml --demo
```

### Step 4 — Run the demo

```bash
# Terminal replay of 2024 Kolkata flood
python scripts/kolkata_demo.py

# Full interactive Streamlit app
streamlit run app/app.py
```

### Step 5 — Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --mode test
python evaluate.py --checkpoint checkpoints/best_model.pth --mode replay
python evaluate.py --checkpoint checkpoints/best_model.pth --mode benchmark
```

---

## Architecture

```
Input [B, T_in=6, C=8, H, W]
  C = rainfall(1) + elevation(1) + slope(1) + flow_acc(1)
    + drain_density(1) + impervious(1) + manning_n(1) + prev_depth(1)
        ↓
ConvLSTM Encoder  [3 layers: 64 → 64 → 32 hidden channels]
  • Convolutional gates (spatial pattern learning)
  • Peephole connections (better gradient flow)
  • GroupNorm on hidden states (training stability)
        ↓
Autoregressive Decoder  [32 → 64 → 64, reversed mirror]
  • Input per step: prev_prediction + static terrain
  • MC Dropout active during inference → uncertainty maps
        ↓
Output [B, T_out=3, 1, H, W]  — flood depth in metres at T+1h, T+2h, T+3h
        ↓
Alert Map [H, W]
  ✅ DRY       < 0.15m
  🟡 WATCH    ≥ 0.15m — monitor, review plans
  🟠 WARNING  ≥ 0.30m — prepare evacuation
  🔴 DANGER   ≥ 0.60m — immediate action, life risk
```

**Parameter count:** ~2.3 million (encoder: ~1.4M, decoder: ~0.9M)

---

## Datasets (all free / open-access)

| Source | What | Access |
|---|---|---|
| IMD Open Data | Hourly rainfall, 700+ stations | Free — imdpune.gov.in |
| NASA SRTM 30m DEM | Terrain elevation | Free — earthdata.nasa.gov |
| OpenStreetMap | Drainage network, land use | Free — via osmnx |
| Sentinel-1 SAR | Flood inundation ground truth | Free — dataspace.copernicus.eu |
| NRSC Bhuvan | India flood layers | Free — bhuvan.nrsc.gov.in |

> **For hackathon demo:** All data is generated synthetically via `--demo` flag. No credentials or downloads needed.

---

## Key Metrics (target)

| Metric | Target | Meaning |
|---|---|---|
| **CSI** | ≥ 0.65 | Fraction of flooded cells correctly predicted |
| **POD** | ≥ 0.80 | Fraction of actual floods detected |
| **FAR** | ≤ 0.25 | Fraction of predictions that were false alarms |
| **RMSE** | ≤ 0.08m | Depth prediction error in metres |
| **FSS** | ≥ 0.60 | Spatial skill score (>0.5 = skillful) |

CSI is the primary metric — accuracy is misleading because 80%+ of cells stay dry every monsoon hour. CSI focuses only on flood cells.

---

## Physics — Saint-Venant Equations

### Continuity (mass conservation)
$$\frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} + \frac{\partial (hv)}{\partial y} = r - i$$

### Momentum (diffusive wave approximation)
$$u \cdot \frac{n^2\sqrt{u^2+v^2}}{h^{4/3}} = -g\frac{\partial(z+h)}{\partial x}$$

Where:
- `h` = water depth (m)
- `u, v` = depth-averaged velocity (m/s)
- `r` = rainfall rate (m/s), `i` = infiltration (m/s)
- `n` = Manning's roughness coefficient
- `z` = bed elevation (m), `g` = 9.81 m/s²

The **diffusive wave approximation** is standard for urban flooding (Froude number Fr < 0.1) and numerically stable at hourly timesteps.

### Why PINN beats a plain CNN

A plain ConvLSTM can predict water appearing from nowhere, or flowing uphill. The physics residuals penalise these violations during training. The model learns *both* from data *and* from hydraulic laws — giving physically plausible predictions even at locations with limited training data.

---

## Alert System

Alerts are generated at **ward level** using a configurable fraction threshold:

```yaml
alerts:
  thresholds:
    watch:   0.15  # metres
    warning: 0.30  # metres
    danger:  0.60  # metres
  min_cell_fraction: 0.05  # 5% of ward cells must exceed threshold
```

A ward is elevated to WARNING only when ≥ 5% of its 50m cells exceed 0.30m — preventing single-cell noise from triggering city-wide alerts.

---

## Scalability

Kolkata is the prototype. Replicating to any Indian city requires:

1. `config.yaml` — update bounding box
2. `python data/ingest.py --city chennai --demo`
3. `python data/grid.py --city chennai`
4. `python train.py --city chennai`

Supported out of the box: **Kolkata, Chennai, Mumbai**

Minimum data needed for a new city: **2 years of monsoon rainfall + SRTM DEM** (both free).

---

## Training Details

```yaml
model:
  hidden_channels: [64, 64, 32]   # ConvLSTM layers
  input_steps:  6                  # 6-hour input window
  output_steps: 3                  # T+1h, T+2h, T+3h forecast

training:
  epochs:           60
  batch_size:       8
  learning_rate:    0.001
  scheduler:        cosine_warmup  # linear warmup → cosine decay
  warmup_epochs:    5              # physics weight ramps 0.1 → 1.0
  gradient_clip:    1.0
  mixed_precision:  true           # FP16 on CUDA
  early_stopping:   12 epochs
```

**Physics warmup:** The physics loss weight starts at 0.1 and ramps to 1.0 over the first 10 epochs. This lets the model learn basic flood patterns from data before the hydraulic constraints tighten — preventing training divergence.

---


## References

1. **Shi et al. (2015).** Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. *NeurIPS 2015.* [arxiv.org/abs/1506.04214](https://arxiv.org/abs/1506.04214)

2. **Raissi et al. (2019).** Physics-Informed Neural Networks. *Journal of Computational Physics.* [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

3. **Bates et al. (2010).** A simple inertial formulation of the shallow water equations for efficient two-dimensional flood inundation modelling. *Journal of Hydrology.* [doi:10.1016/j.jhydrol.2010.03.027](https://doi.org/10.1016/j.jhydrol.2010.03.027)

4. **Kabir et al. (2020).** A deep convolutional neural network model for rapid prediction of fluvial flood inundation. *Journal of Hydrology.* [doi:10.1016/j.jhydrol.2020.12479](https://doi.org/10.1016/j.jhydrol.2020.12479)

5. **Chow, V.T. (1959).** Open-Channel Hydraulics. McGraw-Hill. *(Manning's n values)*

---

## File Dependency Map

```
config.yaml
    ↓
data/ingest.py  ──────────────────────────────────────────────────────┐
data/grid.py    ──────────────────────────────────────────────────┐   │
    ↓                                                              ↓   ↓
data/dataset.py                                            data/raw/  data/processed/
    ↓
models/convlstm.py
models/pinn.py      ──── utils/physics.py
    ↓
models/weather_twin.py
    ↓
train.py  ───────────────────────────────→  checkpoints/best_model.pth
                                                        ↓
evaluate.py  ←──────────────────────────────────────────┤
app/app.py   ←──────────────────────────────────────────┤
app/inference.py  ←─────────────────────────────────────┤
scripts/kolkata_demo.py  ←──────────────────────────────┘

utils/metrics.py  ←── evaluate.py, train.py
utils/physics.py  ←── models/pinn.py, evaluate.py
```

---

*Built for INNOFUSION 3.0 — India's Premier Software + Hardware Hackathon*
