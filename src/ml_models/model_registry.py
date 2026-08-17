"""
ml_models/model_registry.py
Unified model registry — loads and caches all trained models,
exposes a single .predict() API, integrates with MLflow for versioning.
"""
from __future__ import annotations
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for all ML models in the Urban Digital Twin.
    Provides a unified predict() interface and MLflow logging hooks.
    """

    def __init__(self, model_dir: str = "data/processed"):
        self.model_dir = Path(model_dir)
        self._models: Dict[str, Any] = {}
        self._metrics: Dict[str, Dict] = {}
        self._loaded: Dict[str, bool] = {}

    # ── Model registration ────────────────────────────────────────────────────
    def register(self, name: str, model: Any, metrics: dict = None):
        """Register a trained model instance."""
        self._models[name] = model
        self._metrics[name] = metrics or {}
        self._loaded[name] = True
        logger.info(f"Registered model: {name} | metrics={metrics}")

    # ── Lazy loading ──────────────────────────────────────────────────────────
    def load_all(self):
        """Attempt to load all saved models from disk."""
        self._try_load_traffic()
        self._try_load_energy()
        self._try_load_water()
        self._try_load_anomaly_traffic()
        self._try_load_anomaly_resource()
        logger.info(f"Registry loaded: {list(self._loaded.keys())}")

    def _try_load_traffic(self):
        path = self.model_dir / "traffic_lstm.pt"
        if path.exists():
            try:
                from src.ml_models.traffic_forecaster import TrafficForecaster
                m = TrafficForecaster()
                m.load(str(path))
                self._models["traffic"] = m
                self._loaded["traffic"] = True
            except Exception as e:
                logger.warning(f"Could not load traffic model: {e}")

    def _try_load_energy(self):
        path = self.model_dir / "energy_xgb.pkl"
        if path.exists():
            try:
                from src.ml_models.energy_forecaster import EnergyForecaster
                m = EnergyForecaster()
                m.load(str(path))
                self._models["energy"] = m
                self._loaded["energy"] = True
            except Exception as e:
                logger.warning(f"Could not load energy model: {e}")

    def _try_load_water(self):
        path = self.model_dir / "water_xgb.pkl"
        if path.exists():
            try:
                from src.ml_models.energy_forecaster import WaterForecaster
                m = WaterForecaster()
                m.load(str(path))
                self._models["water"] = m
                self._loaded["water"] = True
            except Exception as e:
                logger.warning(f"Could not load water model: {e}")

    def _try_load_anomaly_traffic(self):
        path = self.model_dir / "anomaly_traffic.pkl"
        if path.exists():
            try:
                from src.anomaly_detection.detectors import TrafficAnomalyDetector
                m = TrafficAnomalyDetector()
                m.load(str(path))
                self._models["anomaly_traffic"] = m
                self._loaded["anomaly_traffic"] = True
            except Exception as e:
                logger.warning(f"Could not load traffic anomaly model: {e}")

    def _try_load_anomaly_resource(self):
        path = self.model_dir / "anomaly_resource.pkl"
        if path.exists():
            try:
                from src.anomaly_detection.detectors import ResourceAnomalyDetector
                m = ResourceAnomalyDetector()
                m.load(str(path))
                self._models["anomaly_resource"] = m
                self._loaded["anomaly_resource"] = True
            except Exception as e:
                logger.warning(f"Could not load resource anomaly model: {e}")

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, model_name: str, X: np.ndarray) -> Optional[np.ndarray]:
        if model_name not in self._models:
            logger.warning(f"Model '{model_name}' not in registry")
            return None
        try:
            return self._models[model_name].predict(X)
        except Exception as e:
            logger.error(f"Prediction error for {model_name}: {e}")
            return None

    def is_available(self, name: str) -> bool:
        return self._loaded.get(name, False)

    def get_metrics(self, name: str) -> dict:
        return self._metrics.get(name, {})

    def list_models(self) -> list:
        return list(self._loaded.keys())

    # ── MLflow logging ────────────────────────────────────────────────────────
    def log_to_mlflow(self, model_name: str, metrics: dict,
                      params: dict = None, run_name: str = None):
        try:
            import mlflow
            mlflow.set_experiment("urban_digital_twin")
            with mlflow.start_run(run_name=run_name or model_name):
                if params:
                    mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                logger.info(f"MLflow logged: {model_name} metrics={metrics}")
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-critical): {e}")

    # ── Status report ─────────────────────────────────────────────────────────
    def status_report(self) -> dict:
        return {
            "loaded_models": self.list_models(),
            "metrics": self._metrics,
            "model_dir": str(self.model_dir),
            "timestamp": datetime.now().isoformat(),
        }


# Global singleton
_registry: Optional[ModelRegistry] = None

def get_registry(model_dir: str = "data/processed") -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(model_dir)
        _registry.load_all()
    return _registry
