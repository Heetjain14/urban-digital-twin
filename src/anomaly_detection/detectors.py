"""
anomaly_detection/traffic_anomaly.py + resource_anomaly.py + alert_manager.py
Multi-layer anomaly detection for traffic accidents, energy spikes, water leaks.
"""
from __future__ import annotations
import numpy as np
import pickle
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime
import logging

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Traffic anomaly detector ──────────────────────────────────────────────────
class TrafficAnomalyDetector:
    """
    Isolation Forest on traffic features.
    Detects accidents, unusual slowdowns, sudden density spikes.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.model = IsolationForest(
            n_estimators=cfg.get("n_estimators", 200),
            contamination=cfg.get("contamination", 0.02),
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.trained = False
        self._baseline: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray):
        """Train on normal traffic behavior."""
        X_s = self.scaler.fit_transform(features)
        self.model.fit(X_s)
        self.trained = True
        self._baseline = features.mean(axis=0)
        logger.info(f"TrafficAnomalyDetector trained on {len(features)} samples")

    def detect(self, features: np.ndarray) -> np.ndarray:
        """Returns -1 for anomaly, 1 for normal (sklearn convention)."""
        if not self.trained:
            return np.ones(len(features))
        X_s = self.scaler.transform(features)
        return self.model.predict(X_s)

    def anomaly_score(self, features: np.ndarray) -> np.ndarray:
        """Normalized anomaly score: 0=normal, 1=very anomalous."""
        if not self.trained:
            return np.zeros(len(features))
        X_s = self.scaler.transform(features)
        raw = self.model.score_samples(X_s)
        # Invert and normalize: lower score_samples → more anomalous
        scores = -raw
        return np.clip((scores - scores.min()) / (scores.max() - scores.min() + 1e-8), 0, 1)

    def save(self, path: str = "data/processed/anomaly_traffic.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "baseline": self._baseline}, f)

    def load(self, path: str = "data/processed/anomaly_traffic.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._baseline = data.get("baseline")
        self.trained = True
        return self


# ── Resource anomaly detector ─────────────────────────────────────────────────
class ResourceAnomalyDetector:
    """
    Z-score based anomaly detection on energy/water time series.
    Uses rolling statistics to adapt to diurnal patterns.
    """

    def __init__(self, window: int = 60, z_threshold: float = 3.0,
                 consecutive: int = 3):
        self.window = window
        self.z_threshold = z_threshold
        self.consecutive = consecutive
        self._buffers: Dict[str, deque] = {}
        self._consec_counts: Dict[str, int] = {}
        self.trained = True   # no training needed

    def update(self, series_id: str, value: float) -> Dict[str, Any]:
        """
        Feed a new value; returns anomaly info dict.
        Returns {"is_anomaly": bool, "z_score": float, "severity": str}
        """
        if series_id not in self._buffers:
            self._buffers[series_id] = deque(maxlen=self.window)
            self._consec_counts[series_id] = 0

        buf = self._buffers[series_id]
        buf.append(value)

        if len(buf) < 10:
            return {"is_anomaly": False, "z_score": 0.0, "severity": "LOW"}

        arr = np.array(buf)
        mu, sigma = arr.mean(), arr.std()

        if sigma < 1e-8:
            return {"is_anomaly": False, "z_score": 0.0, "severity": "LOW"}

        z = abs(value - mu) / sigma

        if z > self.z_threshold:
            self._consec_counts[series_id] += 1
        else:
            self._consec_counts[series_id] = 0

        is_anomaly = self._consec_counts[series_id] >= self.consecutive
        severity = "LOW"
        if is_anomaly:
            if z > self.z_threshold * 2:
                severity = "CRITICAL"
            elif z > self.z_threshold * 1.5:
                severity = "HIGH"
            else:
                severity = "MEDIUM"

        return {
            "is_anomaly": is_anomaly,
            "z_score": round(float(z), 2),
            "severity": severity,
            "value": value,
            "mean": round(float(mu), 2),
            "std": round(float(sigma), 2),
        }

    def save(self, path: str = "data/processed/anomaly_resource.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"window": self.window, "z_threshold": self.z_threshold,
                         "consecutive": self.consecutive}, f)

    def load(self, path: str = "data/processed/anomaly_resource.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.window = data["window"]
        self.z_threshold = data["z_threshold"]
        self.consecutive = data["consecutive"]
        return self


# ── Alert manager ─────────────────────────────────────────────────────────────
class AlertManager:
    """
    Deduplicates and manages anomaly alerts.
    Multi-sensor fusion: combines traffic + resource signals.
    Pushes categorized alerts to event bus.
    """

    COOLDOWN_TICKS = 30   # min ticks between same-location alerts

    def __init__(self):
        self._active_alerts: Dict[str, dict] = {}
        self._alert_history: List[dict] = []
        self._last_alert_tick: Dict[str, int] = {}

    def process_snapshot(self, snapshot: dict,
                          traffic_detector: Optional[TrafficAnomalyDetector] = None,
                          resource_detector: Optional[ResourceAnomalyDetector] = None) -> List[dict]:
        """
        Process a simulation snapshot, run detectors, return new alerts.
        """
        tick = snapshot.get("tick", 0)
        new_alerts = []

        # ── Resource anomalies ─────────────────────────────────────────────
        if resource_detector:
            for res in snapshot.get("resources", []):
                zone_id = res["node_id"]

                e_result = resource_detector.update(
                    f"energy_{zone_id}", res["energy_demand"])
                if e_result["is_anomaly"]:
                    alert = self._make_alert(
                        tick, "energy_spike", e_result["severity"],
                        res["x"], res["y"],
                        f"Zone {zone_id}: Energy demand {res['energy_load_pct']:.0f}% "
                        f"(z={e_result['z_score']})",
                        confidence=min(e_result["z_score"] / 6.0, 0.99)
                    )
                    if self._should_emit(alert, tick):
                        new_alerts.append(alert)

                w_result = resource_detector.update(
                    f"water_{zone_id}", res["water_demand"])
                if w_result["is_anomaly"]:
                    alert = self._make_alert(
                        tick, "water_anomaly", w_result["severity"],
                        res["x"], res["y"],
                        f"Zone {zone_id}: Water demand anomaly "
                        f"(z={w_result['z_score']})",
                        confidence=min(w_result["z_score"] / 6.0, 0.99)
                    )
                    if self._should_emit(alert, tick):
                        new_alerts.append(alert)

        # ── Vehicle-based traffic anomaly (simplified for live detection) ──
        agents = snapshot.get("agents", [])
        vehicles = [a for a in agents if a["agent_type"] == "vehicle"]
        if vehicles:
            moving = [v for v in vehicles if v["speed"] > 0]
            if moving:
                speeds = [v["speed"] for v in moving]
                mean_speed = np.mean(speeds)
                std_speed = np.std(speeds) if len(speeds) > 5 else 0

                # Flag grid cell with anomalously low speed
                slow = [v for v in moving if v["speed"] < 5]
                if len(slow) > 3 and mean_speed > 0:
                    cx = np.mean([v["x"] for v in slow])
                    cy = np.mean([v["y"] for v in slow])
                    key = f"traffic_{int(cx//5)}_{int(cy//5)}"
                    if tick - self._last_alert_tick.get(key, -1000) > self.COOLDOWN_TICKS:
                        alert = self._make_alert(
                            tick, "traffic_accident", "HIGH",
                            cx, cy,
                            f"Slow traffic cluster at ({cx:.0f},{cy:.0f}): "
                            f"{len(slow)} vehicles < 5 km/h",
                            confidence=0.78,
                        )
                        new_alerts.append(alert)
                        self._last_alert_tick[key] = tick

        # Store history
        self._alert_history.extend(new_alerts)
        if len(self._alert_history) > 500:
            self._alert_history = self._alert_history[-500:]

        return new_alerts

    def _make_alert(self, tick, alert_type, severity, x, y, description, confidence=0.8):
        return {
            "alert_id": str(uuid.uuid4())[:8],
            "tick": tick,
            "alert_type": alert_type,
            "severity": severity,
            "location_x": round(float(x), 2),
            "location_y": round(float(y), 2),
            "description": description,
            "confidence": round(confidence, 3),
            "resolved": False,
        }

    def _should_emit(self, alert: dict, tick: int) -> bool:
        key = f"{alert['alert_type']}_{int(alert['location_x'])//5}_{int(alert['location_y'])//5}"
        last = self._last_alert_tick.get(key, -9999)
        if tick - last >= self.COOLDOWN_TICKS:
            self._last_alert_tick[key] = tick
            return True
        return False

    def get_recent_alerts(self, n: int = 20) -> List[dict]:
        return list(reversed(self._alert_history[-n:]))

    def get_active_count(self) -> int:
        return len([a for a in self._alert_history[-50:] if not a["resolved"]])
