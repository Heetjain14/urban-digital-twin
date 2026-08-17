"""
ml_models/feature_engineering.py
Builds feature matrices for all ML models from raw simulation/synthetic data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List


def cyclical_encode(values: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    """Encode periodic features as (sin, cos) pair to avoid ordinal bias."""
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def build_traffic_features(df: pd.DataFrame, target_col: str = "vehicle_count",
                            window: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build LSTM input features for a single road segment.
    Input shape: (samples, window, n_features)
    Features: vehicle_count, avg_speed, congestion, hour_sin, hour_cos, day_sin, day_cos
    """
    # Sort by tick
    df = df.sort_values("tick").reset_index(drop=True)
    ticks = df["tick"].values
    hours = (ticks % 1440) / 60.0
    days = (ticks // 1440) % 7

    hour_sin, hour_cos = cyclical_encode(hours, 24.0)
    day_sin, day_cos = cyclical_encode(days.astype(float), 7.0)

    feature_cols = np.column_stack([
        df["vehicle_count"].values / df["vehicle_count"].max(),
        df.get("avg_speed_kmh", pd.Series(np.zeros(len(df)))).values / 80.0,
        df.get("congestion_score", pd.Series(np.zeros(len(df)))).values,
        hour_sin, hour_cos, day_sin, day_cos,
    ])

    targets = df[target_col].values.astype(float)

    X, y = [], []
    for i in range(window, len(feature_cols)):
        X.append(feature_cols[i - window: i])
        y.append(targets[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_energy_features(df: pd.DataFrame, window: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Features for energy/water XGBoost: tabular lag + rolling stats."""
    df = df.sort_values("tick").reset_index(drop=True)
    ticks = df["tick"].values
    hours = (ticks % 1440) / 60.0
    days = (ticks // 1440) % 7
    is_weekend = (days >= 5).astype(float)

    hour_sin, hour_cos = cyclical_encode(hours, 24.0)
    day_sin, day_cos = cyclical_encode(days.astype(float), 7.0)

    demand = df["demand_kwh"].values if "demand_kwh" in df.columns else df["demand_m3h"].values
    demand_norm = demand / (demand.max() + 1e-8)

    rows = []
    for i in range(window, len(demand_norm)):
        window_data = demand_norm[i - window: i]
        row = {
            "lag_1": demand_norm[i - 1],
            "lag_2": demand_norm[i - 2],
            "lag_3": demand_norm[i - 3],
            "lag_6": demand_norm[i - 6],
            "lag_12": demand_norm[i - 12],
            "lag_24": demand_norm[i - 24],
            "rolling_mean_6": window_data[-6:].mean(),
            "rolling_std_6": window_data[-6:].std(),
            "rolling_mean_24": window_data.mean(),
            "rolling_max_24": window_data.max(),
            "hour_sin": hour_sin[i],
            "hour_cos": hour_cos[i],
            "day_sin": day_sin[i],
            "day_cos": day_cos[i],
            "is_weekend": is_weekend[i],
        }
        rows.append(row)

    X = pd.DataFrame(rows).values.astype(np.float32)
    y = demand[window:].astype(np.float32)
    return X, y


def build_anomaly_features(df: pd.DataFrame) -> np.ndarray:
    """
    Feature matrix for Isolation Forest anomaly detection on traffic.
    Returns: (n_samples, n_features) array
    """
    df = df.sort_values("tick").reset_index(drop=True)
    demand_col = "vehicle_count" if "vehicle_count" in df.columns else "demand_kwh"
    vals = df[demand_col].values.astype(float)

    # Rolling statistics (window=12 = 12 min)
    s = pd.Series(vals)
    roll_mean = s.rolling(12, min_periods=1).mean().values
    roll_std = s.rolling(12, min_periods=1).std().fillna(0).values
    diff1 = np.diff(vals, prepend=vals[0])
    diff2 = np.diff(diff1, prepend=diff1[0])

    extra = {}
    if "avg_speed_kmh" in df.columns:
        extra["speed"] = df["avg_speed_kmh"].values / 80.0
    if "congestion_score" in df.columns:
        extra["congestion"] = df["congestion_score"].values

    features = [vals / (vals.max() + 1e-8), roll_mean / (vals.max() + 1e-8),
                roll_std / (vals.max() + 1e-8),
                diff1 / (np.abs(diff1).max() + 1e-8)]
    for v in extra.values():
        features.append(v)

    return np.column_stack(features).astype(np.float32)


def train_test_split_temporal(X: np.ndarray, y: np.ndarray,
                               train_frac=0.8, val_frac=0.1):
    """Temporal split — never shuffle time-series data."""
    n = len(X)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return (X[:t1], y[:t1],
            X[t1:t2], y[t1:t2],
            X[t2:], y[t2:])
