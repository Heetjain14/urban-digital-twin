"""
ml_models/energy_forecaster.py  +  water_forecaster.py
XGBoost models for energy demand and water consumption prediction.
"""
from __future__ import annotations
import numpy as np
import pickle
from pathlib import Path
import logging

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class EnergyForecaster:
    """
    XGBoost-based district energy demand forecaster.
    Trains on tabular lag + rolling + cyclical features.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.model = XGBRegressor(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self._max_val: float = 1.0
        self.trained: bool = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        self._max_val = float(y_train.max()) if y_train.max() > 0 else 1.0
        X_s = self.scaler.fit_transform(X_train)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_s = self.scaler.transform(X_val)
            eval_set = [(X_val_s, y_val / self._max_val)]

        self.model.fit(
            X_s, y_train / self._max_val,
            eval_set=eval_set,
            verbose=False,
        )
        self.trained = True
        train_preds = self.model.predict(X_s) * self._max_val
        mae = mean_absolute_error(y_train, train_preds)
        logger.info(f"EnergyForecaster trained — train MAE={mae:.2f}")
        return {"train_mae": mae}

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s) * self._max_val

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        preds = self.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mape = float(np.mean(np.abs((preds - y_test) / (y_test + 1e-8))) * 100)
        logger.info(f"EnergyForecaster — MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%")
        return {"mae": mae, "rmse": rmse, "mape": mape}

    def save(self, path: str = "data/processed/energy_xgb.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "max_val": self._max_val}, f)
        logger.info(f"EnergyForecaster saved to {path}")

    def load(self, path: str = "data/processed/energy_xgb.pkl") -> "EnergyForecaster":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._max_val = data["max_val"]
        self.trained = True
        return self


class WaterForecaster:
    """
    XGBoost model for water demand forecasting.
    Same architecture as EnergyForecaster; separate training data.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.model = XGBRegressor(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", 5),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self._max_val: float = 1.0
        self.trained: bool = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        self._max_val = float(y_train.max()) if y_train.max() > 0 else 1.0
        X_s = self.scaler.fit_transform(X_train)
        self.model.fit(X_s, y_train / self._max_val, verbose=False)
        self.trained = True
        preds = self.model.predict(X_s) * self._max_val
        mae = mean_absolute_error(y_train, preds)
        logger.info(f"WaterForecaster trained — train MAE={mae:.2f}")
        return {"train_mae": mae}

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s) * self._max_val

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        preds = self.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mape = float(np.mean(np.abs((preds - y_test) / (y_test + 1e-8))) * 100)
        logger.info(f"WaterForecaster — MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%")
        return {"mae": mae, "rmse": rmse, "mape": mape}

    def save(self, path: str = "data/processed/water_xgb.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "max_val": self._max_val}, f)

    def load(self, path: str = "data/processed/water_xgb.pkl") -> "WaterForecaster":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._max_val = data["max_val"]
        self.trained = True
        return self
