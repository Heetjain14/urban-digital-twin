"""
ml_models/traffic_forecaster.py
Stacked LSTM for traffic flow prediction at 15/30/60-min horizons.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


# ── Model architecture ────────────────────────────────────────────────────────
class TrafficLSTM(nn.Module):
    """
    2-layer stacked LSTM with dropout.
    Input:  (batch, seq_len, n_features)
    Output: (batch, 1)  — next-step traffic count (normalized)
    """
    def __init__(self, input_size: int = 7, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]           # take last timestep
        return self.head(self.dropout(last)).squeeze(-1)


# ── Trainer / wrapper ─────────────────────────────────────────────────────────
class TrafficForecaster:
    """
    Wraps TrafficLSTM with train / predict / evaluate / save / load.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.input_size = cfg.get("input_size", 7)
        self.hidden_size = cfg.get("hidden_size", 128)
        self.num_layers = cfg.get("num_layers", 2)
        self.dropout = cfg.get("dropout", 0.2)
        self.batch_size = cfg.get("batch_size", 64)
        self.lr = cfg.get("learning_rate", 0.001)
        self.epochs = cfg.get("epochs", 50)
        self.patience = cfg.get("patience", 10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrafficLSTM(self.input_size, self.hidden_size,
                                 self.num_layers, self.dropout).to(self.device)
        self._max_val: float = 1.0   # for denormalization

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train LSTM. Returns dict of training history.
        X shape: (N, window, features), y shape: (N,)
        """
        self._max_val = float(y_train.max()) if y_train.max() > 0 else 1.0
        y_train_n = y_train / self._max_val
        y_val_n = y_val / self._max_val if y_val is not None else None

        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train_n).to(self.device)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs)
        criterion = nn.HuberLoss(delta=1.0)

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience_count = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for xb, yb in dl:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
            scheduler.step()

            avg_train = np.mean(train_losses)
            history["train_loss"].append(avg_train)

            if X_val is not None and y_val_n is not None:
                val_loss = self._compute_val_loss(X_val, y_val_n, criterion)
                history["val_loss"].append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    patience_count = 0
                    self._best_state = {k: v.clone()
                                        for k, v in self.model.state_dict().items()}
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            if (epoch + 1) % 10 == 0:
                vl = history["val_loss"][-1] if history["val_loss"] else 0
                logger.info(f"  Epoch {epoch+1}/{self.epochs} "
                            f"train_loss={avg_train:.4f} val_loss={vl:.4f}")

        # Restore best weights
        if hasattr(self, "_best_state") and X_val is not None:
            self.model.load_state_dict(self._best_state)

        return history

    def _compute_val_loss(self, X_val, y_val_n, criterion):
        self.model.eval()
        with torch.no_grad():
            xv = torch.FloatTensor(X_val).to(self.device)
            yv = torch.FloatTensor(y_val_n).to(self.device)
            pred = self.model(xv)
            return criterion(pred, yv).item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predictions in original scale."""
        self.model.eval()
        with torch.no_grad():
            xt = torch.FloatTensor(X).to(self.device)
            pred = self.model(xt).cpu().numpy()
        return pred * self._max_val

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        preds = self.predict(X_test)
        mae = float(np.mean(np.abs(preds - y_test)))
        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        mape = float(np.mean(np.abs((preds - y_test) / (y_test + 1e-8))) * 100)
        logger.info(f"Traffic LSTM — MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%")
        return {"mae": mae, "rmse": rmse, "mape": mape}

    def save(self, path: str = "data/processed/traffic_lstm.pt"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "max_val": self._max_val,
            "config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            }
        }, path)
        logger.info(f"TrafficForecaster saved to {path}")

    def load(self, path: str = "data/processed/traffic_lstm.pt") -> "TrafficForecaster":
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt.get("config", {})
        self.model = TrafficLSTM(
            cfg.get("input_size", self.input_size),
            cfg.get("hidden_size", self.hidden_size),
            cfg.get("num_layers", self.num_layers),
            cfg.get("dropout", self.dropout),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self._max_val = ckpt.get("max_val", 1.0)
        return self
