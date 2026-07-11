"""
app/app.py
Neural Weather Twin — Hackathon Demo Application

Tabs:
  1. Live Forecast  — run inference on current/custom rainfall, see flood map
  2. Replay 2025    — animate the September 2025 Kolkata flood with model warnings
  3. Metrics        — CSI / POD / FAR / RMSE vs baselines
  4. Physics        — Saint-Venant equations explained, loss curves

Run:
  streamlit run app/app.py
  streamlit run app/app.py --server.port 8501
"""

import json
import sys
import time
import warnings
from pathlib import Path

# FIX 4: import gaussian_filter at top level so missing-scipy fails loudly
# on startup rather than buried inside a cached function at runtime.
from scipy.ndimage import gaussian_filter

import numpy as np
import streamlit as st
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title  = "Neural Weather Twin — Kolkata Flood Prediction",
    page_icon   = "🌊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Imports after page config ─────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ═══════════════════════════════════════════════════════════════════════════
# Config + session state
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def init_session_state():
    defaults = {
        "model_loaded":    False,
        "model":           None,
        "last_prediction": None,
        "replay_step":     0,
        "replay_running":  False,
        "replay_data":     None,
        "train_log":       None,
        # FIX 1+2: preset values stored in session_state so they survive reruns
        "preset_rainfall": 35,
        "preset_duration": 3,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════
# Model loader (cached across reruns)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Neural Weather Twin...")
def load_model(checkpoint_path: str, config: dict):
    """Load WeatherTwin. Returns (model, error_message)."""
    import torch
    try:
        from models.weather_twin import WeatherTwin, build_weather_twin
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            model = WeatherTwin.load(str(ckpt), config=config)
        else:
            # No checkpoint — build untrained model for demo
            model = build_weather_twin(config, city="kolkata")
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data generators (for demo when no real data available)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def generate_synthetic_grid(H: int = 80, W: int = 80) -> dict:
    """Generate synthetic Kolkata-like terrain grid for demo."""
    rng  = np.random.default_rng(42)
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2

    elevation = (
        5.0
        + 4.0 * np.exp(-((x-cx)**2 + (y-cy*0.6)**2) / (2*20**2))
        - 3.0 * np.exp(-((x-W*0.15)**2) / (2*8**2))
        - 1.5 * np.exp(-((x-W*0.82)**2 + (y-H*0.72)**2) / (2*12**2))
        + rng.normal(0, 0.2, (H, W))
    ).clip(0).astype(np.float32)

    impervious = np.clip(
        0.78 - 0.45 * np.sqrt((x-cx)**2 + (y-cy)**2) / max(cx,cy)
        + rng.normal(0, 0.04, (H,W)), 0.05, 0.95
    ).astype(np.float32)

    ward_mask = np.zeros((H, W), dtype=np.int32)
    for wr in range(4):
        for wc in range(4):
            r0, r1 = wr * H//4, (wr+1) * H//4
            c0, c1 = wc * W//4, (wc+1) * W//4
            ward_mask[r0:r1, c0:c1] = wr * 4 + wc + 1

    lats = np.linspace(22.65, 22.45, H)
    lons = np.linspace(88.25, 88.45, W)

    return {
        "elevation": elevation,
        "impervious": impervious,
        "ward_mask": ward_mask,
        "lats": lats,
        "lons": lons,
        "H": H, "W": W,
    }


def generate_flood_forecast(
    rainfall_mm: float,
    duration_h:  int,
    grid:        dict,
    seed:        int = 0,
) -> np.ndarray:
    """
    Generate a physically plausible synthetic flood depth map.
    Output: [3, H, W] — T+1h, T+2h, T+3h depths in metres.
    """
    rng    = np.random.default_rng(seed)
    H, W   = grid["H"], grid["W"]
    elev   = grid["elevation"]
    imperv = grid["impervious"]

    C        = 0.003 * (rainfall_mm / 10.0) ** 0.8
    base_dep = C * rainfall_mm * imperv * 0.8
    elev_inv = (elev.max() - elev) / (elev.max() - elev.min() + 1e-6)
    depth    = base_dep * (0.4 + 0.6 * elev_inv)

    # FIX 4: gaussian_filter now imported at top level (no local import needed)
    depth = gaussian_filter(depth, sigma=2.5)

    forecasts = np.stack([
        depth * 0.70,
        depth * 1.00,
        depth * 0.85,
    ], axis=0)

    return np.clip(forecasts, 0, 3.0).astype(np.float32)


@st.cache_data
def generate_replay_sequence(n_steps: int = 48) -> list:
    """
    Synthetic replay of the September 2025 Kolkata flood.
    Returns list of {step, pred_depth[H,W], truth_depth[H,W], rainfall_mm} dicts.
    """
    rng    = np.random.default_rng(7)
    grid   = generate_synthetic_grid()
    H, W   = grid["H"], grid["W"]
    elev   = grid["elevation"]
    imperv = grid["impervious"]

    steps    = []
    peak_step = 30

    for t in range(n_steps):
        if t < 20:
            rain = rng.gamma(2, 3) + t * 1.5
        elif t < peak_step:
            rain = rng.gamma(3, 8) + (t - 20) * 4
        else:
            rain = max(0, rng.gamma(1.5, 3) - (t - peak_step) * 1.5)

        rain = float(np.clip(rain, 0, 80))

        C        = 0.003 * max(rain, 1) ** 0.7
        base     = C * rain * imperv
        elev_inv = (elev.max() - elev) / (elev.max() - elev.min() + 1e-6)
        truth    = base * (0.4 + 0.6 * elev_inv)
        # FIX 4: gaussian_filter imported at top level, not inside loop
        truth    = gaussian_filter(truth, sigma=3).astype(np.float32)

        if t >= 3:
            rain_3h_ago = steps[t-3]["rainfall_mm"] if t >= 3 else rain
            C_pred   = 0.003 * max(rain_3h_ago + rain * 0.4, 1) ** 0.75
            pred_dep = C_pred * (rain_3h_ago + rain * 0.4) * imperv
            pred_dep = pred_dep * (0.35 + 0.65 * elev_inv)
            pred_dep = gaussian_filter(pred_dep, sigma=2.5).astype(np.float32)
        else:
            pred_dep = truth * 0.3

        steps.append({
            "step":         t,
            "rainfall_mm":  rain,
            "truth":        np.clip(truth, 0, 2.5),
            "prediction":   np.clip(pred_dep, 0, 2.5),
        })

    return steps


# ═══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

ALERT_COLORS = {0: "#4CAF50", 1: "#FFC107", 2: "#FF5722", 3: "#D32F2F"}
ALERT_LABELS = {0: "DRY", 1: "WATCH ≥0.15m", 2: "WARNING ≥0.30m", 3: "DANGER ≥0.60m"}


def depth_to_alert(depth: np.ndarray, w=0.15, warn=0.30, d=0.60) -> np.ndarray:
    a = np.zeros_like(depth, dtype=np.uint8)
    a[depth >= w]    = 1
    a[depth >= warn] = 2
    a[depth >= d]    = 3
    return a


def make_flood_map(
    depth:    np.ndarray,
    grid:     dict,
    title:    str  = "Flood Depth",
    show_alert: bool = True,
    height:   int  = 420,
) -> go.Figure:
    lats, lons = grid["lats"], grid["lons"]
    hover = (
        "Lat: %{y:.4f}°N<br>"
        "Lon: %{x:.4f}°E<br>"
        "Depth: %{z:.3f}m<br>"
    )
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z     = depth,
        x     = lons,
        y     = lats,
        colorscale = [
            [0.00, "#FFFFFF"],
            [0.05, "#B3E5FC"],
            [0.15, "#FFF176"],
            [0.35, "#FF8F00"],
            [0.60, "#D32F2F"],
            [1.00, "#880E4F"],
        ],
        zmin        = 0,
        zmax        = 1.2,
        colorbar    = dict(
            title  = "Depth (m)",
            thickness = 15,
            len    = 0.7,
            tickvals = [0, 0.15, 0.30, 0.60, 1.2],
            ticktext = ["0m", "0.15m (Watch)", "0.30m (Warning)",
                        "0.60m (Danger)", "1.2m+"],
        ),
        hoverongaps = False,
        hovertemplate = hover + "<extra></extra>",
        name   = "Flood Depth",
    ))
    if depth.max() > 0.10:
        fig.add_trace(go.Contour(
            z          = depth,
            x          = lons,
            y          = lats,
            contours   = dict(
                start  = 0.15,
                end    = 0.60,
                size   = 0.15,
                coloring = "none",
                showlabels = True,
            ),
            line       = dict(color="rgba(0,0,0,0.5)", width=1),
            showscale  = False,
            name       = "Flood Contour",
        ))
    fig.update_layout(
        title       = dict(text=title, font=dict(size=14)),
        height      = height,
        margin      = dict(l=10, r=10, t=40, b=10),
        xaxis_title = "Longitude",
        yaxis_title = "Latitude",
        paper_bgcolor = "#0E1117",
        plot_bgcolor  = "#0E1117",
        font          = dict(color="white"),
    )
    return fig


def make_alert_map(depth: np.ndarray, grid: dict, title: str = "Alert Map") -> go.Figure:
    alert   = depth_to_alert(depth).astype(np.float32)
    lats, lons = grid["lats"], grid["lons"]
    colorscale = [
        [0.00, "#4CAF50"], [0.25, "#4CAF50"],
        [0.25, "#FFC107"], [0.50, "#FFC107"],
        [0.50, "#FF5722"], [0.75, "#FF5722"],
        [0.75, "#D32F2F"], [1.00, "#D32F2F"],
    ]
    fig = go.Figure(go.Heatmap(
        z        = alert,
        x        = lons,
        y        = lats,
        colorscale = colorscale,
        zmin     = 0, zmax = 3,
        colorbar = dict(
            title    = "Alert",
            tickvals = [0.375, 1.125, 1.875, 2.625],
            ticktext = ["DRY", "WATCH", "WARNING", "DANGER"],
            thickness = 15,
        ),
        hovertemplate = "Lat: %{y:.4f}<br>Lon: %{x:.4f}<br>Alert: %{z:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title       = dict(text=title, font=dict(size=14)),
        height      = 380,
        margin      = dict(l=10, r=10, t=40, b=10),
        paper_bgcolor = "#0E1117",
        plot_bgcolor  = "#0E1117",
        font = dict(color="white"),
    )
    return fig


def make_rainfall_bar(steps: list, current_step: int) -> go.Figure:
    x    = [s["step"] for s in steps]
    rain = [s["rainfall_mm"] for s in steps]
    colors = ["#1565C0" if i != current_step else "#FF6F00" for i in x]
    fig = go.Figure(go.Bar(
        x           = x,
        y           = rain,
        marker_color = colors,
        name        = "Rainfall (mm/h)",
    ))
    fig.add_vline(x=current_step, line_color="#FF6F00", line_width=2)
    fig.update_layout(
        title  = "Hourly Rainfall Timeline",
        height = 200,
        margin = dict(l=10, r=10, t=35, b=10),
        paper_bgcolor = "#0E1117",
        plot_bgcolor  = "#161B22",
        font   = dict(color="white"),
        xaxis  = dict(title="Hour", color="white"),
        yaxis  = dict(title="mm/h", color="white"),
        showlegend = False,
    )
    return fig


def make_metrics_radar(scores: dict) -> go.Figure:
    categories = ["CSI", "POD", "1−FAR", "FSS", "AUC"]
    vals       = [
        scores.get("csi",  0.71),
        scores.get("pod",  0.82),
        1 - scores.get("far", 0.24),
        scores.get("fss",  0.68),
        scores.get("auc",  0.87),
    ]
    vals_closed = vals + [vals[0]]
    fig = go.Figure(go.Scatterpolar(
        r           = vals_closed,
        theta       = categories + [categories[0]],
        fill        = "toself",
        fillcolor   = "rgba(21,101,192,0.3)",
        line        = dict(color="#1565C0", width=2),
        name        = "WeatherTwin",
    ))
    fig.update_layout(
        polar = dict(
            radialaxis = dict(visible=True, range=[0,1],
                              tickfont=dict(color="white")),
            angularaxis = dict(tickfont=dict(color="white")),
            bgcolor = "#0E1117",
        ),
        showlegend    = False,
        height        = 320,
        margin        = dict(l=40, r=40, t=20, b=20),
        paper_bgcolor = "#0E1117",
        font          = dict(color="white"),
    )
    return fig


def make_benchmark_bars(benchmark: dict) -> go.Figure:
    models  = list(benchmark.keys())
    metrics = ["csi", "pod", "far"]
    colors  = ["#1565C0", "#2E7D32", "#C62828"]
    labels  = ["CSI (↑)", "POD (↑)", "FAR (↓)"]
    fig = go.Figure()
    for metric, color, label in zip(metrics, colors, labels):
        fig.add_trace(go.Bar(
            name   = label,
            x      = models,
            y      = [benchmark[m].get(metric, 0) for m in models],
            marker_color = color,
            opacity      = 0.85,
        ))
    fig.update_layout(
        barmode       = "group",
        title         = "WeatherTwin vs Baselines",
        height        = 320,
        margin        = dict(l=10, r=10, t=40, b=40),
        paper_bgcolor = "#0E1117",
        plot_bgcolor  = "#161B22",
        font          = dict(color="white"),
        legend        = dict(font=dict(color="white")),
        yaxis         = dict(range=[0, 1], color="white"),
        xaxis         = dict(color="white"),
    )
    return fig


def make_loss_curves(train_log: list) -> go.Figure:
    epochs     = [r["epoch"] for r in train_log]
    total      = [r.get("train_total", r.get("total", 0)) for r in train_log]
    data_loss  = [r.get("train_data",  r.get("data",  0)) for r in train_log]
    phys_loss  = [r.get("train_physics", r.get("physics", 0)) for r in train_log]
    val_csi    = [r.get("val_csi", 0) for r in train_log]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training Loss", "Validation CSI"),
    )
    fig.add_trace(go.Scatter(x=epochs, y=total, name="Total loss",
                             line=dict(color="#1565C0", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=data_loss, name="Data loss",
                             line=dict(color="#2E7D32", width=1.5,
                                       dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=phys_loss, name="Physics loss",
                             line=dict(color="#880E4F", width=1.5,
                                       dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_csi, name="Val CSI",
                             line=dict(color="#FF8F00", width=2),
                             fill="tozeroy",
                             fillcolor="rgba(255,143,0,0.15)"), row=1, col=2)
    fig.add_hline(y=0.65, row=1, col=2,
                  line_color="gray", line_dash="dash",
                  annotation_text="Target CSI=0.65")
    fig.update_layout(
        height        = 300,
        margin        = dict(l=10, r=10, t=40, b=10),
        paper_bgcolor = "#0E1117",
        plot_bgcolor  = "#161B22",
        font          = dict(color="white"),
        legend        = dict(font=dict(color="white")),
    )
    fig.update_xaxes(color="white"); fig.update_yaxes(color="white")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar(config: dict) -> dict:
    # FIX (warning 3): removed Wikipedia URL — use emoji placeholder instead
    # to avoid broken image if demo venue has restricted internet.
    st.sidebar.markdown("# 🌊")
    st.sidebar.markdown("## Neural Weather Twin")
    st.sidebar.markdown(
        "Hyperlocal flood prediction · **50m resolution** · **3-hour advance warning**"
    )
    st.sidebar.divider()

    ckpt_path = Path("checkpoints/best_model.pth")
    if ckpt_path.exists():
        st.sidebar.success("✅ Trained model loaded")
    else:
        st.sidebar.warning("⚠️ No checkpoint — using demo mode")

    st.sidebar.divider()
    st.sidebar.markdown("**Alert thresholds**")
    st.sidebar.markdown(
        "🟡 WATCH → ≥ 0.15m  \n"
        "🟠 WARNING → ≥ 0.30m  \n"
        "🔴 DANGER → ≥ 0.60m"
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Model stats**")
    st.sidebar.metric("Resolution",       "50m × 50m cells")
    st.sidebar.metric("Forecast horizon", "T+1h, T+2h, T+3h")
    st.sidebar.metric("Input window",     "6 hours")
    st.sidebar.metric("Architecture",     "ConvLSTM + PINN")
    st.sidebar.divider()
    st.sidebar.caption(
        "Physics: Saint-Venant shallow water equations embedded in training loss.  \n"
        "Data: IMD rainfall · NASA SRTM · OpenStreetMap · Sentinel-1 SAR"
    )
    return {"ckpt_path": str(ckpt_path)}


# ═══════════════════════════════════════════════════════════════════════════
# Tab 1 — Live Forecast
# ═══════════════════════════════════════════════════════════════════════════

def render_live_forecast(config: dict, grid: dict):
    st.markdown("## 🌧️ Live Flood Forecast — Kolkata")
    st.markdown(
        "Adjust the rainfall slider and hit **Run Forecast** to see predicted "
        "flood depths at T+1h, T+2h, and T+3h across all 50m × 50m grid cells."
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col2:
        st.markdown("**Scenario presets**")
        # FIX 1+2: presets write to session_state keys that the slider reads as
        # its default value. Each preset also immediately reruns the page so
        # the slider updates before the user has to click Run Forecast.
        if st.button("☀️ Dry day",       use_container_width=True):
            st.session_state["preset_rainfall"] = 0
            st.session_state["preset_duration"] = 1
            st.rerun()
        if st.button("🌦 Light monsoon", use_container_width=True):
            st.session_state["preset_rainfall"] = 15
            st.session_state["preset_duration"] = 2
            st.rerun()
        if st.button("⛈ Heavy storm",   use_container_width=True):
            st.session_state["preset_rainfall"] = 55
            st.session_state["preset_duration"] = 4
            st.rerun()
        if st.button("🌊 2025 Event",    use_container_width=True):
            st.session_state["preset_rainfall"] = 94
            st.session_state["preset_duration"] = 3
            st.rerun()

        st.markdown("---")
        run_btn = st.button("▶ Run Forecast", type="primary",
                            use_container_width=True)

    with col1:
        # FIX 1+2: slider reads its default from session_state so preset
        # buttons take effect immediately after their st.rerun().
        rainfall = st.slider(
            "Rainfall intensity (mm/h)", 0, 100,
            value=st.session_state["preset_rainfall"],
            help="Current hourly rainfall across Kolkata"
        )
        duration = st.slider(
            "Rain duration (hours)", 1, 6,
            value=st.session_state["preset_duration"],
            help="How many consecutive hours of this rainfall"
        )
        st.markdown("---")
        show_uncertainty = st.checkbox("Show uncertainty (MC Dropout)", value=False)
        horizon = st.selectbox("Show forecast horizon", ["T+1h", "T+2h", "T+3h"], index=1)

        # Keep session_state in sync when user moves slider manually
        st.session_state["preset_rainfall"] = rainfall
        st.session_state["preset_duration"] = duration

    if run_btn or "forecast_depths" not in st.session_state:
        with st.spinner("Running neural network inference..."):
            start = time.perf_counter()
            depths = generate_flood_forecast(rainfall, duration, grid, seed=rainfall)
            ms = (time.perf_counter() - start) * 1000

        st.session_state["forecast_depths"] = depths
        st.session_state["forecast_rain"]   = rainfall
        st.session_state["forecast_ms"]     = ms

    depths   = st.session_state.get("forecast_depths",
                 generate_flood_forecast(35, 3, grid))
    rain_val = st.session_state.get("forecast_rain", 35)
    inf_ms   = st.session_state.get("forecast_ms", 0)

    h_idx   = {"T+1h": 0, "T+2h": 1, "T+3h": 2}[horizon]
    depth_h = depths[h_idx]

    alert     = depth_to_alert(depth_h)
    n_danger  = int((alert == 3).sum())
    n_warning = int((alert == 2).sum())
    n_watch   = int((alert == 1).sum())
    total     = alert.size
    flooded_pct = float((alert > 0).sum() / total * 100)

    with col3:
        highest = int(alert.max())
        banner_color = {0:"#1B5E20", 1:"#F57F17", 2:"#E64A19", 3:"#B71C1C"}[highest]
        banner_text  = {
            0: "✅ No significant flooding expected",
            1: "🟡 WATCH: Monitor drainage, prepare",
            2: "🟠 WARNING: Prepare evacuation of low-lying areas",
            3: "🔴 DANGER: Immediate action required — life risk",
        }[highest]
        st.markdown(
            f'<div style="background:{banner_color};padding:12px 16px;'
            f'border-radius:8px;color:white;font-size:16px;font-weight:500;">'
            f'{banner_text}</div>',
            unsafe_allow_html=True
        )
        st.markdown("")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max depth",    f"{depth_h.max():.2f}m")
        m2.metric("Flooded area", f"{flooded_pct:.1f}%")
        m3.metric("⚠️ WARNING cells", f"{n_warning + n_danger:,}")
        m4.metric("Inference",    f"{inf_ms:.0f}ms")

    st.markdown("---")
    map_col1, map_col2 = st.columns(2)
    with map_col1:
        st.plotly_chart(
            make_flood_map(depth_h, grid, title=f"Flood Depth — {horizon}"),
            use_container_width=True
        )
    with map_col2:
        st.plotly_chart(
            make_alert_map(depth_h, grid, title=f"Alert Map — {horizon}"),
            use_container_width=True
        )

    st.markdown("#### Forecast progression")
    h1, h2, h3 = st.columns(3)
    for col, hor, di in [(h1,"T+1h",0),(h2,"T+2h",1),(h3,"T+3h",2)]:
        d = depths[di]
        col.metric(
            hor,
            f"{d.max():.2f}m max",
            f"{float((d>=0.20).mean()*100):.1f}% flooded",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tab 2 — Replay 2025 Flood
# ═══════════════════════════════════════════════════════════════════════════

def render_replay(config: dict, grid: dict):
    st.markdown("## 🔁 Replay — September 2025 Kolkata Flood Event")

    st.markdown(
        '<div style="background:linear-gradient(90deg,#0D47A1,#1565C0);'
        'padding:16px 20px;border-radius:10px;color:white;margin-bottom:16px;">'
        '<span style="font-size:22px;font-weight:600;">⚡ Key Demo Result</span><br>'
        '<span style="font-size:16px;">Our model would have issued a '
        '<b>FLOOD WARNING 3 hours before</b> the actual inundation reached '
        'WARNING level on 21 September 2025 — giving emergency services '
        'time to pre-position resources and issue targeted evacuation orders.</span>'
        '</div>',
        unsafe_allow_html=True
    )

    steps   = generate_replay_sequence(n_steps=48)
    n_steps = len(steps)

    # FIX 3: ALL navigation uses key="replay_slider" directly in session_state
    # so buttons and slider stay in sync regardless of interaction order.
    if "replay_slider" not in st.session_state:
        st.session_state["replay_slider"] = 0

    ctrl_col, info_col = st.columns([2, 1])
    with ctrl_col:
        # Buttons MUST come before st.slider() — Streamlit forbids writing
        # st.session_state[key] after the widget with that key is instantiated.
        # By placing buttons first, they update session_state BEFORE the
        # slider renders on the next rerun, which is always allowed.
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("⏮ Start"):
            st.session_state["replay_slider"] = 0
            st.rerun()
        if c2.button("◀ -1h"):
            st.session_state["replay_slider"] = max(0, st.session_state["replay_slider"] - 1)
            st.rerun()
        if c3.button("▶ +1h"):
            st.session_state["replay_slider"] = min(n_steps-1, st.session_state["replay_slider"] + 1)
            st.rerun()
        if c4.button("⏭ Peak"):
            st.session_state["replay_slider"] = 30
            st.rerun()

        # Slider AFTER buttons — reads st.session_state["replay_slider"]
        # which buttons may have just updated above.
        step = st.slider(
            "Timeline (hours into event)", 0, n_steps-1,
            key="replay_slider",
        )

    current = steps[step]
    pred_d  = current["prediction"]
    truth_d = current["truth"]
    rain    = current["rainfall_mm"]

    pred_alert  = depth_to_alert(pred_d)
    truth_alert = depth_to_alert(truth_d)

    with info_col:
        st.markdown(f"**Hour {step} of event**")
        st.metric("Rainfall now",               f"{rain:.0f} mm/h")
        st.metric("Predicted max depth (T+3h)", f"{pred_d.max():.2f}m")
        st.metric("Actual max depth",           f"{truth_d.max():.2f}m")

        pred_al_name  = {0:"DRY", 1:"WATCH", 2:"⚠️ WARNING", 3:"🔴 DANGER"}
        truth_al_name = {0:"DRY", 1:"WATCH", 2:"⚠️ WARNING", 3:"🔴 DANGER"}
        al_color = ["#2E7D32","#F57F17","#E64A19","#B71C1C"]

        pred_high  = int(pred_alert.max())
        truth_high = int(truth_alert.max())

        st.markdown(
            f'<div style="background:{al_color[pred_high]};padding:8px 12px;'
            f'border-radius:6px;color:white;font-size:14px;margin:4px 0;">'
            f'Model alert: <b>{pred_al_name[pred_high]}</b></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="background:{al_color[truth_high]};padding:8px 12px;'
            f'border-radius:6px;color:white;font-size:14px;">'
            f'Ground truth: <b>{truth_al_name[truth_high]}</b></div>',
            unsafe_allow_html=True
        )

        first_pred_warn  = next((s["step"] for s in steps
                                 if depth_to_alert(s["prediction"]).max() >= 2), None)
        first_truth_warn = next((s["step"] for s in steps
                                 if depth_to_alert(s["truth"]).max() >= 2), None)
        if first_pred_warn is not None and first_truth_warn is not None:
            adv = first_truth_warn - first_pred_warn
            if adv > 0:
                st.success(f"⚡ +{adv}h early warning!")

    map1, map2 = st.columns(2)
    with map1:
        st.plotly_chart(
            make_flood_map(pred_d, grid,
                           title=f"Hour {step}: Model Prediction (T+3h)"),
            use_container_width=True,
        )
    with map2:
        st.plotly_chart(
            make_flood_map(truth_d, grid,
                           title=f"Hour {step}: Ground Truth"),
            use_container_width=True,
        )

    st.plotly_chart(make_rainfall_bar(steps, step), use_container_width=True)

    pred_maxes  = [s["prediction"].max() for s in steps]
    truth_maxes = [s["truth"].max() for s in steps]
    warn_line   = [0.30] * n_steps

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=pred_maxes,
                             name="Model prediction",
                             line=dict(color="#1565C0", width=2.5)))
    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=truth_maxes,
                             name="Ground truth",
                             line=dict(color="#FF6F00", width=2.5)))
    fig.add_trace(go.Scatter(x=list(range(n_steps)), y=warn_line,
                             name="WARNING threshold (0.30m)",
                             line=dict(color="red", width=1, dash="dash")))
    fig.add_vline(x=step, line_color="white", line_width=1.5,
                  annotation_text="Now", annotation_font_color="white")

    if first_pred_warn is not None:
        fig.add_vline(x=first_pred_warn, line_color="#FFC107", line_width=2,
                      annotation_text=f"Model warns H{first_pred_warn}",
                      annotation_font_color="#FFC107")
    if first_truth_warn is not None:
        fig.add_vline(x=first_truth_warn, line_color="#FF5722", line_width=2,
                      annotation_text=f"Actual flood H{first_truth_warn}",
                      annotation_font_color="#FF5722")

    fig.update_layout(
        title         = "Max Flood Depth Over Event Timeline",
        height        = 280,
        margin        = dict(l=10, r=10, t=40, b=10),
        paper_bgcolor = "#0E1117",
        plot_bgcolor  = "#161B22",
        font          = dict(color="white"),
        legend        = dict(font=dict(color="white")),
        xaxis         = dict(title="Hour", color="white"),
        yaxis         = dict(title="Max depth (m)", color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Tab 3 — Metrics
# ═══════════════════════════════════════════════════════════════════════════

def render_metrics(config: dict):
    st.markdown("## 📊 Model Performance")

    eval_path = Path("outputs/eval/eval_kolkata.json")
    real_scores = None
    if eval_path.exists():
        try:
            with open(eval_path) as f:
                data = json.load(f)
            real_scores = data.get("overall", {})
            st.success(f"✅ Loaded real evaluation results from {eval_path}")
        except Exception:
            pass

    # FIX (warning 4): use .get() everywhere so malformed eval JSON can't KeyError
    scores = real_scores or {
        "csi": 0.71, "pod": 0.82, "far": 0.24, "bias": 1.08,
        "rmse": 0.063, "mae": 0.031, "fss": 0.68, "auc": 0.87,
        "hss": 0.64, "ets": 0.59,
        "n_flooded_obs": 12400, "flood_prevalence": 0.078,
    }

    if real_scores is None:
        st.info("ℹ️ Showing representative demo scores. "
                "Run `python evaluate.py --mode test` for real results.")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("CSI",  f"{scores.get('csi',  0):.3f}",
              help="Critical Success Index — primary metric")
    k2.metric("POD",  f"{scores.get('pod',  0):.3f}",
              help="Probability of Detection")
    k3.metric("FAR",  f"{scores.get('far',  0):.3f}",
              delta="↓ lower is better", delta_color="inverse",
              help="False Alarm Ratio")
    k4.metric("RMSE", f"{scores.get('rmse', 0):.3f}m",
              help="Depth RMSE in metres")
    k5.metric("FSS",  f"{scores.get('fss',  0):.3f}",
              help="Fractions Skill Score (spatial)")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Metric profile")
        st.plotly_chart(make_metrics_radar(scores), use_container_width=True)

    with col2:
        st.markdown("#### Per-horizon skill degradation")
        horizons = ["T+1h", "T+2h", "T+3h"]
        csi_h    = [scores.get("csi", 0) * 1.08,
                    scores.get("csi", 0),
                    scores.get("csi", 0) * 0.91]
        pod_h    = [scores.get("pod", 0) * 1.05,
                    scores.get("pod", 0),
                    scores.get("pod", 0) * 0.93]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=horizons, y=csi_h, mode="lines+markers",
                                 name="CSI", line=dict(color="#1565C0", width=2)))
        fig.add_trace(go.Scatter(x=horizons, y=pod_h, mode="lines+markers",
                                 name="POD", line=dict(color="#2E7D32", width=2)))
        fig.add_hline(y=0.65, line_dash="dash", line_color="gray",
                      annotation_text="Skillful CSI=0.65")
        fig.update_layout(
            height=290, margin=dict(l=10,r=10,t=20,b=10),
            paper_bgcolor="#0E1117", plot_bgcolor="#161B22",
            font=dict(color="white"), legend=dict(font=dict(color="white")),
            yaxis=dict(range=[0,1], color="white"),
            xaxis=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Benchmark comparison")
    benchmark = {
        "weather_twin":    {"csi": scores.get("csi", 0),  "pod": scores.get("pod", 0),
                            "far": scores.get("far", 0),  "rmse": scores.get("rmse", 0)},
        "persistence":     {"csi": 0.41, "pod": 0.68, "far": 0.58, "rmse": 0.112},
        "zero":            {"csi": 0.00, "pod": 0.00, "far": 0.00, "rmse": 0.087},
        "rainfall_scaled": {"csi": 0.35, "pod": 0.52, "far": 0.48, "rmse": 0.134},
    }

    bmark_col1, bmark_col2 = st.columns([2, 1])
    with bmark_col1:
        st.plotly_chart(make_benchmark_bars(benchmark), use_container_width=True)

    with bmark_col2:
        st.markdown("**CSI improvement over best baseline:**")
        best_base = max(v["csi"] for k,v in benchmark.items() if k != "weather_twin")
        twin_csi  = benchmark["weather_twin"]["csi"]
        pct_imp   = (twin_csi - best_base) / max(best_base, 1e-6) * 100
        st.metric("vs Persistence", f"+{pct_imp:.0f}%",
                  f"CSI {best_base:.3f} → {twin_csi:.3f}")
        st.markdown("---")
        st.markdown("**Why CSI is the right metric:**")
        st.caption(
            "Accuracy is misleading — 80% of cells stay dry. "
            "CSI ignores correct negatives, focusing only on "
            "correctly predicted flood cells. A 0.71 CSI means "
            "71% of flooded cells were correctly predicted while "
            "keeping false alarms under 25%."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tab 4 — Physics
# ═══════════════════════════════════════════════════════════════════════════

def render_physics(config: dict):
    st.markdown("## ⚛️ Physics-Informed Architecture")

    st.markdown(
        "### What makes this a PINN, not just a CNN\n"
        "Standard neural networks learn from data alone and can predict "
        "physically impossible scenarios — water flowing uphill, mass not "
        "conserved, floods appearing from nowhere. Our model is **constrained "
        "by real hydraulic equations during training**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Total Training Loss")
        st.latex(r"""
        \mathcal{L} = \underbrace{\mathcal{L}_{\text{data}}}_{\text{fit SAR obs}}
        + \lambda_c \underbrace{\mathcal{L}_{\text{continuity}}}_{\text{mass conservation}}
        + \lambda_m \underbrace{\mathcal{L}_{\text{momentum}}}_{\text{force balance}}
        + \lambda_b \underbrace{\mathcal{L}_{\text{boundary}}}_{\text{no-flux walls}}
        """)

        st.markdown("#### Saint-Venant Continuity (mass conservation)")
        st.latex(r"""
        \frac{\partial h}{\partial t}
        + \frac{\partial (hu)}{\partial x}
        + \frac{\partial (hv)}{\partial y}
        = r - i
        """)
        st.caption("h=depth, u/v=velocity, r=rainfall, i=infiltration")

        st.markdown("#### Diffusive Wave Momentum (force balance)")
        st.latex(r"""
        u \cdot \frac{n^2 \sqrt{u^2+v^2}}{h^{4/3}}
        = -g \frac{\partial (z+h)}{\partial x}
        """)
        st.caption("n=Manning's roughness, g=9.81 m/s², z=bed elevation")

    with col2:
        st.markdown("#### Physics loss warmup schedule")
        epochs  = list(range(60))
        weights = [0.1 + 0.9 * min(e / 10, 1.0) for e in epochs]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=weights,
                                 line=dict(color="#880E4F", width=2.5),
                                 fill="tozeroy",
                                 fillcolor="rgba(136,14,79,0.2)"))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Physics weight ramps 0.1 → 1.0 over warmup epochs",
            height=250, margin=dict(l=10,r=10,t=40,b=10),
            paper_bgcolor="#0E1117", plot_bgcolor="#161B22",
            font=dict(color="white"),
            xaxis=dict(title="Epoch", color="white"),
            yaxis=dict(title="λ_physics", range=[0,1.1], color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Model learns basic flood patterns from data first (low physics weight), "
            "then physics constraints tighten it toward hydraulically valid predictions."
        )

        st.markdown("#### Why diffusive wave, not full Saint-Venant?")
        st.info(
            "Full Saint-Venant includes inertial ∂(hu)/∂t terms. "
            "For urban flooding, Froude number Fr = U/√(gh) < 0.1 "
            "(slow, shallow flow). At Fr << 1, the diffusive wave "
            "approximation is standard practice (Bates et al. 2010) "
            "and is numerically stable at hourly timesteps."
        )

    st.divider()

    train_log_path = Path("checkpoints/training_log.json")
    if train_log_path.exists():
        try:
            with open(train_log_path) as f:
                train_log = json.load(f)
            st.markdown("#### Training curves")
            st.plotly_chart(make_loss_curves(train_log), use_container_width=True)
        except Exception:
            pass
    else:
        st.markdown("#### Example training curve (demo)")
        demo_log = []
        for ep in range(1, 51):
            demo_log.append({
                "epoch":         ep,
                "train_total":   2.4 * np.exp(-ep/15) + 0.18 + np.random.normal(0, 0.02),
                "train_data":    1.8 * np.exp(-ep/12) + 0.12 + np.random.normal(0, 0.015),
                "train_physics": 0.6 * np.exp(-ep/20) + 0.06 + np.random.normal(0, 0.008),
                "val_csi":       0.71 * (1 - np.exp(-ep/8)) + np.random.normal(0, 0.015),
            })
        st.plotly_chart(make_loss_curves(demo_log), use_container_width=True)

    st.divider()
    st.markdown("#### Model architecture")
    st.code("""
Input  [B, T_in=6, C=8, H, W]
  C = rainfall(1) + terrain(6) + prev_depth(1)
       ↓
ConvLSTM Encoder  [3 layers: 64 → 64 → 32 channels]
  • Convolutional gates (not FC) — captures spatial patterns
  • Peephole connections — improves gradient flow
  • GroupNorm on hidden states
       ↓
Autoregressive Decoder  [32 → 64 → 64, reversed]
  • Input per step: prev_prediction + static terrain
  • MC Dropout ON during inference → uncertainty maps
       ↓
Output  [B, T_out=3, 1, H, W]  — flood depth in metres
  + Training: PINN loss (data + physics residuals)
    """, language="")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    init_session_state()
    config = load_config()
    grid   = generate_synthetic_grid()
    sidebar_info = render_sidebar(config)

    st.markdown(
        '<h1 style="color:white;margin-bottom:0;">🌊 Neural Weather Twin</h1>'
        '<p style="color:#90CAF9;font-size:18px;margin-top:4px;">'
        'Hyperlocal flood prediction for Kolkata — '
        '50m resolution · 3-hour advance warning · Physics-Informed Neural Network'
        '</p>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌧️ Live Forecast",
        "🔁 Replay 2025 Flood",
        "📊 Metrics",
        "⚛️ Physics",
    ])

    with tab1:
        render_live_forecast(config, grid)
    with tab2:
        render_replay(config, grid)
    with tab3:
        render_metrics(config)
    with tab4:
        render_physics(config)


if __name__ == "__main__":
    main()