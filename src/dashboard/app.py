"""
dashboard/app.py

Streamlit Urban Digital Twin Dashboard.

Features:
- Real OpenStreetMap road network through OSMnx
- Synthetic road network fallback
- Live traffic simulation
- Vehicles and pedestrians
- Traffic lights
- Resource monitoring
- Congestion visualization
- Weather display
- Scenario engine
- Alert feed
- Traffic/resource charts
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# PROJECT PATH
# =============================================================================

# =============================================================================
# PROJECT PATH
# =============================================================================

from pathlib import Path
import sys

# app.py is located at:
# project_root/src/dashboard/app.py
#
# Therefore:
# parents[0] = dashboard
# parents[1] = src
# parents[2] = project_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add project root to Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Urban Digital Twin AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# DARK THEME
# =============================================================================

st.markdown(
    """
    <style>

        .main {
            background-color: #0e1117;
        }

        .stMetric {
            background-color: #1e2533;
            border-radius: 8px;
            padding: 12px;
        }

        .alert-critical {
            background-color: #3d1212;
            border-left: 4px solid #ff4444;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
        }

        .alert-high {
            background-color: #3d2a12;
            border-left: 4px solid #ff8800;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
        }

        .alert-medium {
            background-color: #3d3a12;
            border-left: 4px solid #ffdd00;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
        }

        .alert-low {
            background-color: #123d1c;
            border-left: 4px solid #44ff88;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
        }

        h1, h2, h3 {
            color: #63b3ed !important;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================

if "model" not in st.session_state:
    st.session_state.model = None

if "tick_history" not in st.session_state:
    st.session_state.tick_history = []

if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

if "running" not in st.session_state:
    st.session_state.running = False

if "ticks_run" not in st.session_state:
    st.session_state.ticks_run = 0


# =============================================================================
# SYNTHETIC MAP BOUNDARIES
# =============================================================================
#
# These are ONLY used when the simulation is running on the synthetic graph.
#
# For OSM graphs:
#     x = longitude
#     y = latitude
#
# For synthetic graphs:
#     x/y = simulation coordinates from approximately 0-50
# =============================================================================

MAP_LAT_MIN = 18.88
MAP_LAT_MAX = 19.32

MAP_LON_MIN = 72.75
MAP_LON_MAX = 73.05


# =============================================================================
# COORDINATE CONVERSION
# =============================================================================

def sim_to_latlon(x, y):
    """
    Convert synthetic simulation coordinates to
    approximate Mumbai latitude/longitude.
    """

    x = float(np.clip(x, 0, 50))
    y = float(np.clip(y, 0, 50))

    lon = (
        MAP_LON_MIN
        + (x / 50.0)
        * (MAP_LON_MAX - MAP_LON_MIN)
    )

    lat = (
        MAP_LAT_MIN
        + (y / 50.0)
        * (MAP_LAT_MAX - MAP_LAT_MIN)
    )

    return lat, lon


def graph_xy_to_latlon(model, x, y):
    """
    Convert graph coordinates into latitude/longitude.

    OSM graph:
        x = longitude
        y = latitude

    Synthetic graph:
        x/y = simulation coordinates
        which are converted to Mumbai lat/lon.
    """

    if model.road_graph.is_osm_graph:

        return float(y), float(x)

    return sim_to_latlon(x, y)


# =============================================================================
# GET NODE COORDINATES
# =============================================================================

def get_node_latlon(model, node_id):
    """
    Return latitude/longitude for a graph node.
    """

    G = model.road_graph.G

    if not G.has_node(node_id):
        return None

    node = G.nodes[node_id]

    x = node.get("x")
    y = node.get("y")

    if x is None or y is None:
        return None

    return graph_xy_to_latlon(
        model,
        x,
        y,
    )


# =============================================================================
# SIMULATION INITIALIZATION
# =============================================================================

@st.cache_resource
def get_simulation(use_osm_override=None):

    import yaml

    from pathlib import Path

    from src.simulation_engine.city_model import CityModel

    from src.anomaly_detection.detectors import (
        AlertManager,
        ResourceAnomalyDetector,
    )

    from src.graph_analysis.graph_metrics import GraphAnalyzer

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    config_path = Path(
        os.path.join(
            PROJECT_ROOT,
            "config",
            "simulation.yaml",
        )
    )

    if config_path.exists():

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as f:

            cfg = yaml.safe_load(f)

    else:

        cfg = {
            "simulation": {
                "num_vehicles": 150,
                "num_pedestrians": 300,
                "num_resource_nodes": 5,

                # IMPORTANT:
                # Set this to True for OSM
                "use_osm": False,

                "osm_place": "Mumbai, India",

                "osm_network_type": "drive",

                "synthetic_nodes": 30,

                "default_green_ns": 30,
            }
        }

    # -------------------------------------------------------------------------
    # Apply dashboard-selected map mode.
    # The sidebar selection is authoritative for this dashboard instance.
    # -------------------------------------------------------------------------
    if use_osm_override is not None:
        cfg.setdefault("simulation", {})
        cfg["simulation"]["use_osm"] = bool(use_osm_override)

    # -------------------------------------------------------------------------
    # Create simulation
    # -------------------------------------------------------------------------

    model = CityModel(
        cfg,
        seed=42,
    )

    # -------------------------------------------------------------------------
    # Supporting systems
    # -------------------------------------------------------------------------

    alert_mgr = AlertManager()

    res_detector = ResourceAnomalyDetector()

    analyzer = GraphAnalyzer(
        model.road_graph
    )

    return (
        model,
        alert_mgr,
        res_detector,
        analyzer,
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        "## 🏙️ Urban Digital Twin"
    )

    st.markdown(
        "**Smart City AI Research System**"
    )

    st.divider()

    # =========================================================================
    # SIMULATION CONTROLS
    # =========================================================================

    st.markdown(
        "### ⚙️ Simulation Controls"
    )

    auto_run = st.toggle(
        "▶ Auto-run simulation",
        value=False,
    )

    ticks_per_frame = st.slider(
        "Ticks per frame",
        min_value=1,
        max_value=20,
        value=5,
    )

    show_agents = st.toggle(
        "Show agents on map",
        value=True,
    )

    show_heatmap = st.toggle(
        "Show congestion heatmap",
        value=True,
    )

    st.divider()

    # =========================================================================
    # MAP INFORMATION
    # =========================================================================

    st.markdown(
        "### 🗺️ Map"
    )

    map_mode = st.selectbox(
        "Map Mode",
        [
            "🏙️ Synthetic Road Network",
            "🌍 OpenStreetMap",
        ],
        index=0,
        key="map_mode",
    )

    use_osm_override = map_mode == "🌍 OpenStreetMap"

    try:

        current_model, _, _, _ = get_simulation(
            use_osm_override
        )

        if current_model.road_graph.is_osm_graph:

            st.success(
                "Real OpenStreetMap roads enabled"
            )

            st.caption(
                "Road network loaded from OpenStreetMap."
            )

        else:

            st.warning(
                "Synthetic road network"
            )

            st.caption(
                "Synthetic roads are being displayed "
            )

    except Exception as exc:

        st.error(
            f"Unable to initialize map: {exc}"
        )

    st.divider()

    # =========================================================================
    # SCENARIO ENGINE
    # =========================================================================

    st.markdown(
        "### 🌦️ Scenario Engine"
    )

    scenario = st.selectbox(
        "Active Scenario",
        [
            "None",
            "Light Rain",
            "Heavy Rain",
            "Road Closure",
            "Mass Event",
            "Power Outage",
        ],
    )

    if st.button(
        "Apply Scenario",
        type="primary",
    ):

        model, _, _, _ = get_simulation(use_osm_override)

        scenario_map = {

            "Light Rain": (
                "rain",
                {
                    "intensity": "light"
                },
            ),

            "Heavy Rain": (
                "rain",
                {
                    "intensity": "heavy"
                },
            ),

            "Road Closure": (
                "road_closure",
                {
                    "num_closures": 3
                },
            ),

            "Mass Event": (
                "mass_event",
                {
                    "multiplier": 3.0
                },
            ),

            "Power Outage": (
                "power_outage",
                {
                    "zone_id": 0
                },
            ),
        }

        if scenario != "None":

            stype, params = scenario_map[
                scenario
            ]

            model.apply_scenario(
                stype,
                params,
            )

            st.success(
                f"Scenario '{scenario}' applied!"
            )

        else:

            model.clear_scenario()

            st.info(
                "Scenarios cleared."
            )

    st.divider()

    # =========================================================================
    # MODEL STATUS
    # =========================================================================

    st.markdown(
        "### 📊 Model Status"
    )

    try:

        from src.ml_models.model_registry import (
            get_registry
        )

        reg = get_registry()

        loaded = reg.list_models()

        for name in [
            "traffic",
            "energy",
            "water",
            "anomaly_traffic",
        ]:

            icon = (
                "✅"
                if name in loaded
                else "⬜"
            )

            st.markdown(
                f"{icon} `{name}`"
            )

    except Exception:

        st.info(
            "Run `python scripts/train_models.py` "
            "to train models."
        )


# =============================================================================
# HEADER
# =============================================================================

col_title, col_status = st.columns(
    [3, 1]
)

with col_title:

    st.title(
        "🏙️ Urban Digital Twin AI System"
    )

with col_status:

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    status_placeholder = st.empty()


# =============================================================================
# SIMULATION FRAME
# =============================================================================

def run_frame():

    model, alert_mgr, res_detector, analyzer = (
        get_simulation(use_osm_override)
    )

    snap = None

    # -------------------------------------------------------------------------
    # Advance simulation
    # -------------------------------------------------------------------------

    for _ in range(
        ticks_per_frame
    ):

        snap = model.tick_step()

        new_alerts = alert_mgr.process_snapshot(
            snap,
            resource_detector=res_detector,
        )

        snap["alerts"] = new_alerts

        st.session_state.alert_history.extend(
            new_alerts
        )

        if len(
            st.session_state.alert_history
        ) > 100:

            st.session_state.alert_history = (
                st.session_state.alert_history[-100:]
            )

    st.session_state.ticks_run += (
        ticks_per_frame
    )

    # -------------------------------------------------------------------------
    # Metrics history
    # -------------------------------------------------------------------------

    if snap is not None:

        metrics = snap["metrics"]

        st.session_state.tick_history.append(
            metrics
        )

        if len(
            st.session_state.tick_history
        ) > 300:

            st.session_state.tick_history = (
                st.session_state.tick_history[-300:]
            )

    return (
        snap,
        model,
        analyzer,
    )


# =============================================================================
# KPI ROW
# =============================================================================

kpi_row = st.columns(6)

kpi_placeholders = [
    c.empty()
    for c in kpi_row
]


# =============================================================================
# MAIN PANELS
# =============================================================================

col_map, col_right = st.columns(
    [3, 2]
)

with col_map:

    # Keep the heading consistent with the actual road graph mode.
    try:
        _preview_model, _, _, _ = get_simulation(use_osm_override)
        _preview_is_osm = bool(
            getattr(
                _preview_model.road_graph,
                "is_osm_graph",
                False,
            )
        )
    except Exception:
        _preview_is_osm = False

    st.markdown(
        "### 🗺️ "
        + (
            "Live OpenStreetMap City Map"
            if _preview_is_osm
            else "Synthetic Digital Twin City Map"
        )
    )

    map_placeholder = st.empty()


with col_right:

    st.markdown(
        "### ⚡ Resource Monitor"
    )

    resource_placeholder = st.empty()

    st.markdown(
        "### 📈 Traffic Forecast"
    )

    forecast_placeholder = st.empty()


# =============================================================================
# ALERT FEED
# =============================================================================

st.markdown(
    "### 🚨 Alert Feed"
)

alert_placeholder = st.empty()


# =============================================================================
# CITY MAP
# =============================================================================

def build_city_map(
    snap,
    model,
    show_agents_flag,
    show_heatmap_flag,
):
    """
    Build city map.

    OSM mode:
        Uses real OpenStreetMap coordinates and basemap.

    Synthetic mode:
        Uses a clean grey grid with simulation coordinates.
        No OpenStreetMap background is displayed.
    """

    # ================================================================
    # DETECT MAP MODE FROM THE ACTUAL ROAD GRAPH
    # ================================================================
    # The road graph is the source of truth for the renderer.  The
    # previous implementation read model.config["simulation"]["use_osm"],
    # which could disagree with road_graph.is_osm_graph and cause the
    # dashboard to show an OSM basemap while the sidebar reported
    # Synthetic mode.

    use_osm = bool(
        getattr(model.road_graph, "is_osm_graph", False)
    )

    # ================================================================
    # CREATE FIGURE
    # ================================================================

    fig = go.Figure()

    G = model.road_graph.G

    # ================================================================
    # SYNTHETIC MODE
    # ================================================================

    if not use_osm:

        # ------------------------------------------------------------
        # 1. SYNTHETIC ROAD NETWORK
        # ------------------------------------------------------------

        edge_x = []
        edge_y = []

        for u, v, d in G.edges(data=True):

            if not G.has_node(u) or not G.has_node(v):
                continue

            ux = G.nodes[u].get("x")
            uy = G.nodes[u].get("y")

            vx = G.nodes[v].get("x")
            vy = G.nodes[v].get("y")

            if (
                ux is None
                or uy is None
                or vx is None
                or vy is None
            ):
                continue

            edge_x.extend([
                ux,
                vx,
                None,
            ])

            edge_y.extend([
                uy,
                vy,
                None,
            ])

        if edge_x:

            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(
                        color="#6b7280",
                        width=2,
                    ),
                    hoverinfo="none",
                    name="Synthetic Roads",
                )
            )

        # ------------------------------------------------------------
        # 2. CONGESTION
        # ------------------------------------------------------------

        if show_heatmap_flag:

            for u, v, d in G.edges(data=True):

                if (
                    not G.has_node(u)
                    or not G.has_node(v)
                ):
                    continue

                load = d.get(
                    "current_load",
                    0,
                )

                capacity = max(
                    d.get(
                        "capacity",
                        1,
                    ),
                    1,
                )

                congestion = min(
                    load / capacity,
                    1.0,
                )

                if congestion <= 0.30:
                    continue

                ux = G.nodes[u].get("x")
                uy = G.nodes[u].get("y")

                vx = G.nodes[v].get("x")
                vy = G.nodes[v].get("y")

                if (
                    ux is None
                    or uy is None
                    or vx is None
                    or vy is None
                ):
                    continue

                if congestion < 0.6:
                    congestion_color = "#ffaa00"
                else:
                    congestion_color = "#ff3333"

                fig.add_trace(
                    go.Scatter(
                        x=[
                            ux,
                            vx,
                        ],
                        y=[
                            uy,
                            vy,
                        ],
                        mode="lines",
                        line=dict(
                            color=congestion_color,
                            width=6,
                        ),
                        hovertemplate=(
                            f"Congestion: "
                            f"{congestion * 100:.0f}%"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

        # ------------------------------------------------------------
        # 3. TRAFFIC LIGHTS
        # ------------------------------------------------------------

        intersection_x = []
        intersection_y = []
        intersection_text = []
        intersection_colors = []

        for iid in model.intersection_ids[:12]:

            if not G.has_node(iid):
                continue

            node = G.nodes[iid]

            x = float(
                node.get(
                    "x",
                    0,
                )
            )

            y = float(
                node.get(
                    "y",
                    0,
                )
            )

            phase = (
                model.environment
                .traffic_light_phases
                .get(
                    iid,
                    "NS_GREEN",
                )
            )

            remaining = (
                model.environment
                .traffic_light_timers
                .get(
                    iid,
                    0,
                )
            )

            intersection_x.append(x)
            intersection_y.append(y)

            intersection_text.append(
                f"Intersection {iid}<br>"
                f"Phase: {phase}<br>"
                f"Remaining: {remaining} ticks"
            )

            if phase == "NS_GREEN":
                intersection_colors.append(
                    "#00aa55"
                )
            else:
                intersection_colors.append(
                    "#dd3333"
                )

        if intersection_x:

            fig.add_trace(
                go.Scatter(
                    x=intersection_x,
                    y=intersection_y,
                    mode="markers",
                    marker=dict(
                        size=13,
                        color=intersection_colors,
                    ),
                    text=intersection_text,
                    hovertemplate=(
                        "%{text}"
                        "<extra></extra>"
                    ),
                    name="Traffic Lights",
                )
            )

        # ------------------------------------------------------------
        # 4. RESOURCE NODES
        # ------------------------------------------------------------

        resources = snap.get(
            "resources",
            [],
        )

        resource_x = []
        resource_y = []
        resource_text = []
        resource_colors = []

        for resource in resources:

            if (
                "x" not in resource
                or "y" not in resource
            ):
                continue

            x = float(resource["x"])
            y = float(resource["y"])

            energy = resource.get(
                "energy_load_pct",
                0,
            )

            water = resource.get(
                "water_load_pct",
                0,
            )

            resource_x.append(x)
            resource_y.append(y)

            resource_text.append(
                f"Resource Node "
                f"{resource.get('node_id', 'N/A')}<br>"
                f"Energy Load: {energy:.0f}%<br>"
                f"Water Load: {water:.0f}%"
            )

            if energy > 85:
                resource_colors.append("#ff3333")

            elif energy > 70:
                resource_colors.append("#ffaa00")

            else:
                resource_colors.append("#0088cc")

        if resource_x:

            fig.add_trace(
                go.Scatter(
                    x=resource_x,
                    y=resource_y,
                    mode="markers",
                    marker=dict(
                        size=15,
                        color=resource_colors,
                    ),
                    text=resource_text,
                    hovertemplate=(
                        "%{text}"
                        "<extra></extra>"
                    ),
                    name="Resource Nodes",
                )
            )

        # ------------------------------------------------------------
        # 5. AGENTS
        # ------------------------------------------------------------

        if show_agents_flag:

            agents = snap.get(
                "agents",
                [],
            )

            # --------------------------------------------------------
            # Vehicles
            # --------------------------------------------------------

            vehicles = [
                a
                for a in agents
                if a.get("agent_type")
                == "vehicle"
            ]

            vehicle_x = []
            vehicle_y = []
            vehicle_text = []
            vehicle_colors = []

            for vehicle in vehicles[:100]:

                if (
                    "x" not in vehicle
                    or "y" not in vehicle
                ):
                    continue

                x = float(vehicle["x"])
                y = float(vehicle["y"])

                speed = vehicle.get(
                    "speed",
                    0,
                )

                vehicle_x.append(x)
                vehicle_y.append(y)

                vehicle_text.append(
                    f"Vehicle<br>"
                    f"Speed: {speed:.1f} km/h"
                )

                if speed < 10:
                    vehicle_colors.append(
                        "#ff3333"
                    )

                elif speed < 30:
                    vehicle_colors.append(
                        "#ffaa00"
                    )

                else:
                    vehicle_colors.append(
                        "#00aa55"
                    )

            if vehicle_x:

                fig.add_trace(
                    go.Scatter(
                        x=vehicle_x,
                        y=vehicle_y,
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=vehicle_colors,
                        ),
                        text=vehicle_text,
                        hovertemplate=(
                            "%{text}"
                            "<extra></extra>"
                        ),
                        name="Vehicles",
                    )
                )

            # --------------------------------------------------------
            # Pedestrians
            # --------------------------------------------------------

            pedestrians = [
                a
                for a in agents
                if a.get("agent_type")
                == "pedestrian"
            ]

            pedestrian_x = []
            pedestrian_y = []

            for pedestrian in pedestrians[:150]:

                if (
                    "x" not in pedestrian
                    or "y" not in pedestrian
                ):
                    continue

                pedestrian_x.append(
                    float(pedestrian["x"])
                )

                pedestrian_y.append(
                    float(pedestrian["y"])
                )

            if pedestrian_x:

                fig.add_trace(
                    go.Scatter(
                        x=pedestrian_x,
                        y=pedestrian_y,
                        mode="markers",
                        marker=dict(
                            size=5,
                            color="#7777cc",
                        ),
                        name="Pedestrians",
                    )
                )

        # ------------------------------------------------------------
        # 6. ALERTS
        # ------------------------------------------------------------

        for alert in (
            st.session_state.alert_history[-5:]
        ):

            if (
                "location_x" not in alert
                or "location_y" not in alert
            ):
                continue

            x = float(
                alert["location_x"]
            )

            y = float(
                alert["location_y"]
            )

            severity = alert.get(
                "severity",
                "HIGH",
            )

            alert_color = (
                "#ff0000"
                if severity == "CRITICAL"
                else "#ff8800"
            )

            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color=alert_color,
                    ),
                    text=["⚠️ ALERT"],
                    hovertemplate=(
                        f"{alert.get('description', 'Alert')}"
                        "<extra></extra>"
                    ),
                    name="Alert",
                    showlegend=False,
                )
            )

        # ------------------------------------------------------------
        # 7. WEATHER
        # ------------------------------------------------------------

        weather_data = snap.get(
            "weather",
            {},
        )

        weather_condition = weather_data.get(
            "condition",
            "clear",
        )

        temperature = weather_data.get(
            "temperature_c",
            0,
        )

        weather_icon = {

            "clear": "☀️",
            "light_rain": "🌧️",
            "heavy_rain": "⛈️",
            "hot": "🌡️",

        }.get(
            weather_condition,
            "☀️",
        )

        # ------------------------------------------------------------
        # 8. DARK SYNTHETIC DIGITAL-TWIN GRID
        # ------------------------------------------------------------

        fig.update_layout(

            height=550,

            # Dark digital-twin background
            paper_bgcolor="#070b12",

            plot_bgcolor="#070b12",

            font=dict(
                color="#dbeafe"
            ),

            showlegend=True,

            legend=dict(
                bgcolor="#111827",
                bordercolor="#26344a",
                borderwidth=1,
                font=dict(
                    size=10,
                    color="#dbeafe",
                ),
            ),

            title=dict(
                text=(
                    f"{weather_icon} "
                    f"{snap['timestamp_sim']} | "
                    f"Tick {snap['tick']} | "
                    f"Weather: "
                    f"{weather_condition} | "
                    f"Temperature: "
                    f"{temperature:.1f}°C"
                ),

                font=dict(
                    size=13,
                    color="#63b3ed",
                ),
            ),

            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            ),

            xaxis=dict(
                range=[0, 50],

                title="",

                showgrid=True,

                gridcolor="#26344a",

                gridwidth=1,

                zeroline=False,

                showticklabels=False,

                fixedrange=False,
            ),

            yaxis=dict(
                range=[0, 50],

                title="",

                showgrid=True,

                gridcolor="#26344a",

                gridwidth=1,

                zeroline=False,

                showticklabels=False,

                scaleanchor="x",

                scaleratio=1,

                fixedrange=False,
            ),
        )

        return fig

    # ========================================================================
    # OSM MODE
    # ========================================================================

    # Everything below is used ONLY when:
    #
    # use_osm == True
    #
    # Therefore synthetic mode never gets an OpenStreetMap background.
    # ========================================================================

    # ------------------------------------------------------------------------
    # Simulation roads
    # ------------------------------------------------------------------------

    edge_lat = []
    edge_lon = []

    for u, v, d in G.edges(data=True):

        if (
            not G.has_node(u)
            or not G.has_node(v)
        ):
            continue

        ux = G.nodes[u].get("x")
        uy = G.nodes[u].get("y")

        vx = G.nodes[v].get("x")
        vy = G.nodes[v].get("y")

        if (
            ux is None
            or uy is None
            or vx is None
            or vy is None
        ):
            continue

        # OSM coordinates:
        # x = longitude
        # y = latitude

        edge_lat.extend([
            uy,
            vy,
            None,
        ])

        edge_lon.extend([
            ux,
            vx,
            None,
        ])

    if edge_lat:

        fig.add_trace(
            go.Scattermap(
                lat=edge_lat,
                lon=edge_lon,
                mode="lines",
                line=dict(
                    width=2,
                    color="#4a5568",
                ),
                hoverinfo="none",
                name="OpenStreetMap Roads",
            )
        )

    # ------------------------------------------------------------------------
    # Traffic lights
    # ------------------------------------------------------------------------

    intersection_lat = []
    intersection_lon = []
    intersection_text = []
    intersection_colors = []

    for iid in model.intersection_ids[:12]:

        if not G.has_node(iid):
            continue

        node = G.nodes[iid]

        lat = float(
            node.get("y", 0)
        )

        lon = float(
            node.get("x", 0)
        )

        phase = (
            model.environment
            .traffic_light_phases
            .get(
                iid,
                "NS_GREEN",
            )
        )

        remaining = (
            model.environment
            .traffic_light_timers
            .get(
                iid,
                0,
            )
        )

        intersection_lat.append(lat)
        intersection_lon.append(lon)

        intersection_text.append(
            f"Intersection {iid}<br>"
            f"Phase: {phase}<br>"
            f"Remaining: {remaining} ticks"
        )

        intersection_colors.append(
            "#00ff66"
            if phase == "NS_GREEN"
            else "#ff3333"
        )

    if intersection_lat:

        fig.add_trace(
            go.Scattermap(
                lat=intersection_lat,
                lon=intersection_lon,
                mode="markers",
                marker=dict(
                    size=13,
                    color=intersection_colors,
                ),
                text=intersection_text,
                hovertemplate=(
                    "%{text}"
                    "<extra></extra>"
                ),
                name="Traffic Lights",
            )
        )

    # ------------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------------

    if show_agents_flag:

        agents = snap.get(
            "agents",
            [],
        )

        vehicles = [
            a
            for a in agents
            if a.get("agent_type")
            == "vehicle"
        ]

        vehicle_lat = []
        vehicle_lon = []

        for vehicle in vehicles[:100]:

            if (
                "x" not in vehicle
                or "y" not in vehicle
            ):
                continue

            vehicle_lon.append(
                float(vehicle["x"])
            )

            vehicle_lat.append(
                float(vehicle["y"])
            )

        if vehicle_lat:

            fig.add_trace(
                go.Scattermap(
                    lat=vehicle_lat,
                    lon=vehicle_lon,
                    mode="markers",
                    marker=dict(
                        size=7,
                        color="#00aa55",
                    ),
                    name="Vehicles",
                )
            )

        pedestrians = [
            a
            for a in agents
            if a.get("agent_type")
            == "pedestrian"
        ]

        pedestrian_lat = []
        pedestrian_lon = []

        for pedestrian in pedestrians[:150]:

            if (
                "x" not in pedestrian
                or "y" not in pedestrian
            ):
                continue

            pedestrian_lon.append(
                float(pedestrian["x"])
            )

            pedestrian_lat.append(
                float(pedestrian["y"])
            )

        if pedestrian_lat:

            fig.add_trace(
                go.Scattermap(
                    lat=pedestrian_lat,
                    lon=pedestrian_lon,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="#7777cc",
                    ),
                    name="Pedestrians",
                )
            )

    # ------------------------------------------------------------------------
    # Resource nodes
    # ------------------------------------------------------------------------

    resources = snap.get(
        "resources",
        [],
    )

    resource_lat = []
    resource_lon = []
    resource_text = []

    for resource in resources:

        if (
            "x" not in resource
            or "y" not in resource
        ):
            continue

        resource_lon.append(
            float(resource["x"])
        )

        resource_lat.append(
            float(resource["y"])
        )

        resource_text.append(
            f"Resource Node "
            f"{resource.get('node_id', 'N/A')}<br>"
            f"Energy: "
            f"{resource.get('energy_load_pct', 0):.0f}%<br>"
            f"Water: "
            f"{resource.get('water_load_pct', 0):.0f}%"
        )

    if resource_lat:

        fig.add_trace(
            go.Scattermap(
                lat=resource_lat,
                lon=resource_lon,
                mode="markers",
                marker=dict(
                    size=15,
                    color="#0088cc",
                ),
                text=resource_text,
                hovertemplate=(
                    "%{text}"
                    "<extra></extra>"
                ),
                name="Resource Nodes",
            )
        )

    # ------------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------------

    weather_data = snap.get(
        "weather",
        {},
    )

    weather_condition = weather_data.get(
        "condition",
        "clear",
    )

    temperature = weather_data.get(
        "temperature_c",
        0,
    )

    weather_icon = {

        "clear": "☀️",
        "light_rain": "🌧️",
        "heavy_rain": "⛈️",
        "hot": "🌡️",

    }.get(
        weather_condition,
        "☀️",
    )

    # ------------------------------------------------------------------------
    # OSM layout
    # ------------------------------------------------------------------------

    fig.update_layout(

        height=550,

        paper_bgcolor="#0e1117",

        font=dict(
            color="#e2e8f0"
        ),

        showlegend=True,

        legend=dict(
            bgcolor="#1e2533",
            font=dict(
                size=10
            ),
        ),

        map=dict(

            style="open-street-map",

            center=dict(

                lat=(
                    MAP_LAT_MIN
                    + MAP_LAT_MAX
                ) / 2,

                lon=(
                    MAP_LON_MIN
                    + MAP_LON_MAX
                ) / 2,
            ),

            zoom=10.5,
        ),

        title=dict(

            text=(
                f"{weather_icon} "
                f"{snap['timestamp_sim']} | "
                f"Tick {snap['tick']} | "
                f"Weather: "
                f"{weather_condition} | "
                f"Temperature: "
                f"{temperature:.1f}°C"
            ),

            font=dict(
                size=13,
                color="#63b3ed",
            ),
        ),

        margin=dict(
            l=10,
            r=10,
            t=55,
            b=10,
        ),
    )

    return fig

# =============================================================================
# RESOURCE CHART
# =============================================================================

def build_resource_chart(history):

    if not history:
        fig = go.Figure()

        fig.update_layout(
            height=220,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#e2e8f0"),
        )

        return fig

    recent_history = history[-120:]

    ticks = [
        h.get("tick", 0)
        for h in recent_history
    ]

    energy = [
        h.get(
            "total_energy_demand_kwh",
            0
        ) / 100
        for h in recent_history
    ]

    water = [
        h.get(
            "total_water_demand_m3h",
            0
        )
        for h in recent_history
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.12,
        subplot_titles=[
            "Energy Demand (×100 kWh)",
            "Water Demand (m³/h)",
        ],
    )

    # Energy
    fig.add_trace(
        go.Scatter(
            x=ticks,
            y=energy,
            mode="lines",
            line=dict(
                color="#63b3ed",
                width=2,
            ),
            fill="tozeroy",
            fillcolor="rgba(99,179,237,0.12)",
            name="Energy",
        ),
        row=1,
        col=1,
    )

    # Water
    fig.add_trace(
        go.Scatter(
            x=ticks,
            y=water,
            mode="lines",
            line=dict(
                color="#48bb78",
                width=2,
            ),
            fill="tozeroy",
            fillcolor="rgba(72,187,120,0.12)",
            name="Water",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=260,

        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",

        font=dict(
            color="#e2e8f0",
            size=10,
        ),

        showlegend=False,

        margin=dict(
            l=45,
            r=10,
            t=45,
            b=10,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        color="#718096",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#2d3748",
        color="#718096",
    )

    return fig

# =============================================================================
# TRAFFIC CHART
# =============================================================================

def build_traffic_chart(history):

    if not history:
        fig = go.Figure()

        fig.update_layout(
            height=180,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#e2e8f0"),
        )

        return fig

    recent_history = history[-120:]

    ticks = [
        h.get("tick", 0)
        for h in recent_history
    ]

    speeds = [
        h.get(
            "avg_vehicle_speed",
            0
        )
        for h in recent_history
    ]

    moving = [
        h.get(
            "vehicles_moving",
            0
        )
        for h in recent_history
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Avg Speed (km/h)",
            "Vehicles Moving",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=ticks,
            y=speeds,
            mode="lines",
            line=dict(
                color="#f6ad55",
                width=2,
            ),
            name="Speed",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=ticks[-30:],
            y=moving[-30:],
            marker_color="#9f7aea",
            name="Moving",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=180,

        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",

        font=dict(
            color="#e2e8f0",
            size=10,
        ),

        showlegend=False,

        margin=dict(
            l=40,
            r=10,
            t=35,
            b=10,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        color="#718096",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#2d3748",
        color="#718096",
    )

    return fig
# =============================================================================
# ALERT RENDERER
# =============================================================================

def render_alerts(alerts):

    if not alerts:

        return (
            "<p style='color:#718096; "
            "font-size:13px;'>"
            "No active alerts."
            "</p>"
        )

    severity_class = {
        "CRITICAL": "alert-critical",
        "HIGH": "alert-high",
        "MEDIUM": "alert-medium",
        "LOW": "alert-low",
    }

    severity_icon = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢",
    }

    html_parts = []

    for alert in reversed(alerts[-8:]):

        severity = str(
            alert.get(
                "severity",
                "LOW"
            )
        ).upper()

        description = str(
            alert.get(
                "description",
                "Unknown alert"
            )
        )

        tick = alert.get(
            "tick",
            0
        )

        confidence = float(
            alert.get(
                "confidence",
                0
            )
        )

        confidence_pct = int(
            confidence * 100
        )

        css_class = severity_class.get(
            severity,
            "alert-low"
        )

        icon = severity_icon.get(
            severity,
            "⚪"
        )

        html_parts.append(
            f"""
<div class="{css_class}">
    <span style="font-size:11px; color:#a0aec0;">
        Tick {tick}
    </span>

    &nbsp;

    {icon}

    <strong>{severity}</strong>

    &mdash;

    <span style="font-size:13px;">
        {description}
    </span>

    <span style="float:right; font-size:11px; color:#718096;">
        conf: {confidence_pct}%
    </span>
</div>
"""
        )

    return "".join(html_parts)

# =============================================================================
# MAIN RENDER
# =============================================================================

try:

    snap, model, analyzer = run_frame()

except Exception as exc:

    st.error(
        "Simulation failed to start."
    )

    st.exception(exc)

    st.stop()


metrics = snap["metrics"]

env = model.environment


# =============================================================================
# STATUS
# =============================================================================

weather_labels = {

    "clear":
        "☀️ Clear",

    "light_rain":
        "🌧️ Light Rain",

    "heavy_rain":
        "⛈️ Heavy Rain",

    "hot":
        "🌡️ Hot",
}

weather_condition = snap.get(
    "weather",
    {}
).get(
    "condition",
    "clear",
)

weather_label = weather_labels.get(
    weather_condition,
    "☀️ Clear",
)

status_placeholder.markdown(

    f"**{env.sim_datetime_str}** | "
    f"{weather_label}"
)


# =============================================================================
# KPI DATA
# =============================================================================

kpi_data = [

    (
        "🚗 Vehicles Moving",
        f"{metrics['vehicles_moving']}",
        "vehicles",
    ),

    (
        "⚡ Max Energy Load",
        f"{metrics['max_energy_load_pct']:.0f}%",
        (
            "⚠️"
            if metrics["max_energy_load_pct"] > 85
            else ""
        ),
    ),

    (
        "💧 Max Water Load",
        f"{metrics['max_water_load_pct']:.0f}%",
        (
            "⚠️"
            if metrics["max_water_load_pct"] > 85
            else ""
        ),
    ),

    (
        "🏎️ Avg Speed",
        f"{metrics['avg_vehicle_speed']:.1f} km/h",
        "",
    ),

    (
        "🚨 Active Alerts",
        f"{len(st.session_state.alert_history[-20:])}",
        "",
    ),

    (
        "🔁 Sim Tick",
        f"{metrics['tick']:,}",
        "",
    ),
]


for i, (
    label,
    val,
    delta,
) in enumerate(
    kpi_data
):

    with kpi_placeholders[i]:

        st.metric(
            label,
            val,
            delta or None,
        )


# =============================================================================
# MAP
# =============================================================================

with map_placeholder:

    fig_map = build_city_map(

        snap,

        model,

        show_agents,

        show_heatmap,
    )

    st.plotly_chart(

        fig_map,

        use_container_width=True,

        config={

            "displayModeBar":
                True,

            "scrollZoom":
                True,
        },
    )


# =============================================================================
# RESOURCE CHART
# =============================================================================

with resource_placeholder:

    st.plotly_chart(

        build_resource_chart(
            st.session_state.tick_history
        ),

        use_container_width=True,

        config={
            "displayModeBar": False
        },
    )


# =============================================================================
# TRAFFIC FORECAST
# =============================================================================

with forecast_placeholder:

    st.plotly_chart(

        build_traffic_chart(
            st.session_state.tick_history
        ),

        use_container_width=True,

        config={
            "displayModeBar": False
        },
    )


# =============================================================================
# ALERTS
# =============================================================================

with alert_placeholder:

    st.html(
        render_alerts(
            st.session_state.alert_history
        )
    )


# =============================================================================
# AUTO-RUN
# =============================================================================

if auto_run:

    time.sleep(
        0.15
    )

    st.rerun()
