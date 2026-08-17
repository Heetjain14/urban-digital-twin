"""
api_layer/main.py
FastAPI application: REST endpoints + WebSocket streaming for real-time dashboard.
"""
from __future__ import annotations
import asyncio
import json
import logging
import threading
import time
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Urban Digital Twin API",
    description="Smart City AI Research System — REST + WebSocket API",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# ── Global state ──────────────────────────────────────────────────────────────
_simulation = None
_sim_thread: Optional[threading.Thread] = None
_sim_running = threading.Event()
_alert_manager = None
_resource_detector = None
_registry = None
_latest_snapshot: Optional[dict] = None
_ws_clients: List[WebSocket] = []
_alerts_buffer: List[dict] = []


def _get_or_create_simulation():
    global _simulation, _alert_manager, _resource_detector
    if _simulation is None:
        config_path = Path("config/simulation.yaml")
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {"simulation": {
                "num_vehicles": 100, "num_pedestrians": 200,
                "num_resource_nodes": 5, "synthetic_nodes": 25,
                "default_green_ns": 30,
            }}

        from src.simulation_engine.city_model import CityModel
        from src.anomaly_detection.detectors import AlertManager, ResourceAnomalyDetector

        _simulation = CityModel(cfg, seed=42)
        _alert_manager = AlertManager()
        _resource_detector = ResourceAnomalyDetector()
        logger.info("Simulation initialized via API")
    return _simulation


def _get_registry():
    global _registry
    if _registry is None:
        from src.ml_models.model_registry import ModelRegistry
        _registry = ModelRegistry("data/processed")
        _registry.load_all()
    return _registry


# ── Background simulation loop ────────────────────────────────────────────────
def _simulation_loop(ticks: int = 10080, tick_delay: float = 0.05):
    """Runs simulation in background thread, pushing snapshots to WS clients."""
    global _latest_snapshot, _alerts_buffer

    sim = _get_or_create_simulation()
    _sim_running.set()
    logger.info(f"Background simulation loop started: {ticks} ticks")

    for _ in range(ticks):
        if not _sim_running.is_set():
            break

        snap = sim.tick_step()

        # Run anomaly detection
        new_alerts = _alert_manager.process_snapshot(
            snap, resource_detector=_resource_detector)
        snap["alerts"] = new_alerts
        _alerts_buffer.extend(new_alerts)
        if len(_alerts_buffer) > 200:
            _alerts_buffer = _alerts_buffer[-200:]

        _latest_snapshot = snap
        time.sleep(tick_delay)

    _sim_running.clear()
    logger.info("Simulation loop finished")


# ── REST Endpoints ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"system": "Urban Digital Twin AI",
            "version": "1.0.0",
            "status": "operational"}


@app.get("/status")
def get_status():
    sim = _simulation
    return {
        "simulation_running": _sim_running.is_set(),
        "tick": sim.tick if sim else 0,
        "sim_time": sim.environment.sim_datetime_str if sim else "—",
        "models_loaded": _get_registry().list_models() if _registry else [],
        "active_alerts": len([a for a in _alerts_buffer if not a.get("resolved")]),
    }


@app.get("/snapshot")
def get_snapshot():
    """Latest simulation snapshot."""
    if _latest_snapshot is None:
        sim = _get_or_create_simulation()
        return sim._build_snapshot()
    return _latest_snapshot


@app.get("/metrics/history")
def get_metrics_history(n: int = 60):
    """Last N simulation snapshots metrics."""
    from src.simulation_engine.event_bus import get_bus
    snaps = get_bus().recent_snapshots(n)
    return [s.get("metrics", {}) for s in snaps]


@app.get("/graph/analysis")
def get_graph_analysis():
    sim = _get_or_create_simulation()
    from src.graph_analysis.graph_metrics import GraphAnalyzer
    analyzer = GraphAnalyzer(sim.road_graph)
    return analyzer.full_report()


@app.get("/graph/heatmap")
def get_graph_heatmap():
    sim = _get_or_create_simulation()
    from src.graph_analysis.graph_metrics import GraphAnalyzer
    analyzer = GraphAnalyzer(sim.road_graph)
    return {"heatmap": analyzer.get_congestion_heatmap()}


@app.get("/alerts")
def get_alerts(n: int = 20):
    return {"alerts": _alerts_buffer[-n:], "total": len(_alerts_buffer)}


@app.get("/predict/traffic")
def predict_traffic(steps: int = 30):
    """Predict traffic for next N steps using trained LSTM."""
    reg = _get_registry()
    if not reg.is_available("traffic"):
        return {"error": "Traffic model not trained yet. Run: python scripts/train_models.py"}

    # Use recent simulation data as input window
    snap = _latest_snapshot or _get_or_create_simulation()._build_snapshot()
    tick = snap["metrics"].get("tick", 0)

    # Build a synthetic prediction (representative values)
    from src.data_ingestion.synthetic_generator import SyntheticCityGenerator
    gen = SyntheticCityGenerator(days=2, seed=tick)
    traffic_df = gen.generate_traffic(1)
    from src.ml_models.feature_engineering import build_traffic_features
    X, _ = build_traffic_features(traffic_df)
    if len(X) > 0:
        preds = reg.predict("traffic", X[-steps:])
        return {"predictions": preds.tolist(), "horizon_ticks": list(range(1, len(preds)+1))}
    return {"predictions": [], "error": "Insufficient data"}


@app.get("/predict/energy")
def predict_energy(steps: int = 30):
    reg = _get_registry()
    if not reg.is_available("energy"):
        return {"error": "Energy model not trained yet."}
    snap = _latest_snapshot or _get_or_create_simulation()._build_snapshot()
    resources = snap.get("resources", [])
    if resources:
        avg_load = sum(r["energy_load_pct"] for r in resources) / len(resources)
    else:
        avg_load = 50.0

    # Synthetic forecast: sinusoidal around current load
    t = snap["metrics"].get("hour_of_day", 12)
    forecasts = [avg_load * (1 + 0.1 * np.sin(2 * np.pi * (t + i/60) / 24))
                 for i in range(steps)]
    import numpy as np
    return {"predictions": [round(f, 1) for f in forecasts],
            "current_load_pct": round(avg_load, 1)}


# ── Scenario control ──────────────────────────────────────────────────────────
class ScenarioRequest(BaseModel):
    scenario_type: str
    duration_ticks: int = 120
    parameters: Dict[str, Any] = {}


@app.post("/scenario/apply")
def apply_scenario(req: ScenarioRequest):
    sim = _get_or_create_simulation()
    sim.apply_scenario(req.scenario_type, req.parameters)
    return {"status": "applied", "scenario": req.scenario_type,
            "params": req.parameters}


@app.post("/scenario/clear")
def clear_scenario():
    sim = _get_or_create_simulation()
    sim.clear_scenario()
    return {"status": "cleared"}


# ── Simulation control ────────────────────────────────────────────────────────
@app.post("/simulate/start")
def start_simulation(background_tasks: BackgroundTasks,
                     ticks: int = 10080, tick_delay: float = 0.02):
    global _sim_thread
    if _sim_running.is_set():
        return {"status": "already_running"}
    _sim_thread = threading.Thread(
        target=_simulation_loop, args=(ticks, tick_delay), daemon=True)
    _sim_thread.start()
    return {"status": "started", "ticks": ticks}


@app.post("/simulate/stop")
def stop_simulation():
    _sim_running.clear()
    return {"status": "stopped"}


@app.post("/simulate/step")
def step_simulation(n: int = 1):
    """Manually step simulation (for controlled testing)."""
    sim = _get_or_create_simulation()
    snaps = []
    for _ in range(n):
        snap = sim.tick_step()
        new_alerts = _alert_manager.process_snapshot(
            snap, resource_detector=_resource_detector)
        snap["alerts"] = new_alerts
        snaps.append(snap)
    global _latest_snapshot
    _latest_snapshot = snaps[-1]
    return {"ticks_advanced": n, "current_tick": sim.tick,
            "metrics": snaps[-1]["metrics"]}


# ── WebSocket streaming ───────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Stream simulation snapshots to dashboard.
    Sends latest snapshot every 0.5 seconds.
    """
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(_ws_clients)}")

    sim = _get_or_create_simulation()
    try:
        while True:
            # Advance sim by 1 tick per WS frame
            snap = sim.tick_step()
            new_alerts = _alert_manager.process_snapshot(
                snap, resource_detector=_resource_detector)
            snap["alerts"] = new_alerts
            global _latest_snapshot
            _latest_snapshot = snap
            _alerts_buffer.extend(new_alerts)

            # Send compact payload
            payload = {
                "tick": snap["tick"],
                "timestamp": snap["timestamp_sim"],
                "metrics": snap["metrics"],
                "weather": snap["weather"],
                "alerts": new_alerts,
                "resources": snap["resources"],
                "intersections": snap["intersections"][:8],
                # Thin agents for bandwidth
                "agents": snap["agents"][:100],
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.1)   # 10 fps max

    except WebSocketDisconnect:
        _ws_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(_ws_clients)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api_layer.main:app", host="0.0.0.0", port=8000,
                reload=False, log_level="info")
