"""
scripts/train_models.py
End-to-end model training pipeline:
1. Generate synthetic data (if not present)
2. Engineer features
3. Train TrafficLSTM, EnergyXGB, WaterXGB
4. Train TrafficAnomalyDetector, ResourceAnomalyDetector
5. Log metrics to MLflow
6. Save all models to data/processed/
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging
import yaml
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("train_models")


def main():
    # ── Config ────────────────────────────────────────────────────────────────
    with open("config/ml_models.yaml") as f:
        ml_cfg = yaml.safe_load(f)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/synthetic").mkdir(parents=True, exist_ok=True)

    # ── Step 1: Synthetic data ────────────────────────────────────────────────
    traffic_path = Path("data/synthetic/traffic.parquet")
    if not traffic_path.exists():
        logger.info("Generating synthetic data...")
        from src.data_ingestion.synthetic_generator import SyntheticCityGenerator
        gen = SyntheticCityGenerator(days=14, seed=42)
        data = gen.generate_all("data/synthetic")
    else:
        logger.info("Loading existing synthetic data...")
        data = {
            "traffic":    pd.read_parquet("data/synthetic/traffic.parquet"),
            "energy":     pd.read_parquet("data/synthetic/energy.parquet"),
            "water":      pd.read_parquet("data/synthetic/water.parquet"),
        }

    results = {}

    # ── Step 2: Traffic LSTM ──────────────────────────────────────────────────
    logger.info("=== Training Traffic LSTM ===")
    from src.ml_models.feature_engineering import (
        build_traffic_features, build_energy_features, build_anomaly_features,
        train_test_split_temporal
    )
    from src.ml_models.traffic_forecaster import TrafficForecaster

    # Use segment 0 as representative
    seg_df = data["traffic"][data["traffic"]["segment_id"] == 0].copy()
    X_tr, y_tr = build_traffic_features(seg_df)

    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split_temporal(X_tr, y_tr)
    logger.info(f"  Traffic — train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    forecaster = TrafficForecaster(config=ml_cfg.get("traffic_lstm", {}))
    history = forecaster.train(X_train, y_train, X_val, y_val)
    metrics = forecaster.evaluate(X_test, y_test)
    forecaster.save("data/processed/traffic_lstm.pt")
    results["traffic_lstm"] = metrics
    logger.info(f"  Traffic LSTM metrics: {metrics}")

    # ── Step 3: Energy XGBoost ────────────────────────────────────────────────
    logger.info("=== Training Energy XGBoost ===")
    from src.ml_models.energy_forecaster import EnergyForecaster, WaterForecaster

    zone_df = data["energy"][data["energy"]["zone_id"] == 0].copy()
    X_e, y_e = build_energy_features(zone_df)
    X_train_e, y_train_e, X_val_e, y_val_e, X_test_e, y_test_e = train_test_split_temporal(X_e, y_e)

    energy_model = EnergyForecaster(config=ml_cfg.get("energy_xgboost", {}))
    energy_model.train(X_train_e, y_train_e, X_val_e, y_val_e)
    e_metrics = energy_model.evaluate(X_test_e, y_test_e)
    energy_model.save("data/processed/energy_xgb.pkl")
    results["energy_xgb"] = e_metrics
    logger.info(f"  Energy XGB metrics: {e_metrics}")

    # ── Step 4: Water XGBoost ─────────────────────────────────────────────────
    logger.info("=== Training Water XGBoost ===")
    water_df = data["water"][data["water"]["zone_id"] == 0].copy()
    # Rename column for feature builder
    water_df = water_df.rename(columns={"demand_m3h": "demand_kwh"})
    X_w, y_w = build_energy_features(water_df)
    X_train_w, y_train_w, _, _, X_test_w, y_test_w = train_test_split_temporal(X_w, y_w)

    water_model = WaterForecaster(config=ml_cfg.get("water_xgboost", {}))
    water_model.train(X_train_w, y_train_w)
    w_metrics = water_model.evaluate(X_test_w, y_test_w)
    water_model.save("data/processed/water_xgb.pkl")
    results["water_xgb"] = w_metrics
    logger.info(f"  Water XGB metrics: {w_metrics}")

    # ── Step 5: Anomaly detectors ─────────────────────────────────────────────
    logger.info("=== Training Anomaly Detectors ===")
    from src.anomaly_detection.detectors import TrafficAnomalyDetector, ResourceAnomalyDetector

    # Traffic anomaly — train on normal data (non-anomaly rows)
    normal_traffic = data["traffic"][~data["traffic"]["is_anomaly"]].copy()
    feats = build_anomaly_features(normal_traffic[normal_traffic["segment_id"] == 0])
    traffic_anomaly = TrafficAnomalyDetector(config=ml_cfg.get("isolation_forest", {}))
    traffic_anomaly.fit(feats)
    traffic_anomaly.save("data/processed/anomaly_traffic.pkl")

    # Resource anomaly — uses statistical approach, no training data needed
    resource_anomaly = ResourceAnomalyDetector(window=60, z_threshold=3.0, consecutive=3)
    resource_anomaly.save("data/processed/anomaly_resource.pkl")

    logger.info("  Anomaly detectors saved.")

    # ── Step 6: MLflow logging ────────────────────────────────────────────────
    from src.ml_models.model_registry import ModelRegistry
    registry = ModelRegistry("data/processed")
    registry.register("traffic", forecaster, metrics)
    registry.register("energy", energy_model, e_metrics)
    registry.register("water", water_model, w_metrics)

    for name, m in results.items():
        registry.log_to_mlflow(name, m)

    # Save summary
    summary = {"status": "success", "models_trained": list(results.keys()),
               "metrics": results}
    with open("data/processed/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Training complete ===")
    for model, m in results.items():
        logger.info(f"  {model}: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, MAPE={m['mape']:.1f}%")

    return summary


if __name__ == "__main__":
    main()
