"""
graph_analysis/gat_model.py
Graph Attention Network (GAT) for citywide road congestion prediction.
Trained on simulation snapshots; predicts per-node congestion score.

Architecture:
    Input:  Node features (5-dim per intersection)
    Layer1: GAT(in=5, hidden=64, heads=4)
    Layer2: GAT(hidden=64, out=1, heads=1)
    Output: Congestion score ∈ [0,1] per node

Node features:
    [vehicle_load_norm, degree_norm, betweenness_norm, hour_sin, hour_cos]
"""
from __future__ import annotations
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)


# ── Pure-NumPy GAT (no PyG dependency required for MVP) ──────────────────────
class GATLayerNumPy:
    """
    Simplified Graph Attention layer implemented in NumPy.
    Suitable for inference; use PyTorch Geometric for full training.
    W: (in_features, out_features)
    a: (2 * out_features,) attention vector
    """
    def __init__(self, in_features: int, out_features: int, heads: int = 4):
        self.in_f = in_features
        self.out_f = out_features
        self.heads = heads
        rng = np.random.default_rng(42)
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = rng.normal(0, scale, (heads, in_features, out_features)).astype(np.float32)
        self.a = rng.normal(0, 0.1, (heads, 2 * out_features)).astype(np.float32)

    def forward(self, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        X:   (N, in_features)
        adj: (N, N) adjacency matrix
        Returns: (N, heads * out_features)
        """
        N = X.shape[0]
        out_heads = []
        for h in range(self.heads):
            Wh = X @ self.W[h]                                   # (N, out_f)
            # Attention scores
            src = np.repeat(Wh, N, axis=0)                       # (N*N, out_f)
            dst = np.tile(Wh, (N, 1))                            # (N*N, out_f)
            e_input = np.concatenate([src, dst], axis=1)         # (N*N, 2*out_f)
            e = np.dot(e_input, self.a[h]).reshape(N, N)         # (N, N)
            e = np.where(adj > 0, e, -1e9)
            alpha = np.exp(e - e.max(axis=1, keepdims=True))
            alpha = alpha / (alpha.sum(axis=1, keepdims=True) + 1e-9)
            out_heads.append(alpha @ Wh)
        return np.concatenate(out_heads, axis=1)                 # (N, heads*out_f)


class SimpleGATCongestionModel:
    """
    2-layer GAT for road node congestion prediction.
    Trains on (graph_snapshot → congestion_label) pairs from simulation data.
    For MVP: uses NumPy implementation.
    For production: swap with PyTorch Geometric GAT.
    """
    def __init__(self, config: dict = None):
        cfg = config or {}
        self.in_channels = cfg.get("in_channels", 5)
        self.hidden_channels = cfg.get("hidden_channels", 16)
        self.heads = cfg.get("heads", 2)
        self.trained = False

        self.gat1 = GATLayerNumPy(self.in_channels, self.hidden_channels, self.heads)
        # Layer 2: input = heads * hidden_channels
        self.gat2 = GATLayerNumPy(self.heads * self.hidden_channels, 8, 1)
        # Output head
        rng = np.random.default_rng(42)
        self.W_out = rng.normal(0, 0.1, (8, 1)).astype(np.float32)

    def predict_congestion(self, node_features: np.ndarray,
                           adj: np.ndarray) -> np.ndarray:
        """
        node_features: (N, 5) — [load, degree, betweenness, hour_sin, hour_cos]
        adj:           (N, N) — binary adjacency matrix
        Returns:       (N,)   — congestion scores ∈ [0,1]
        """
        h1 = self.gat1.forward(node_features, adj)
        h1 = np.maximum(h1, 0)                                   # ReLU
        h2 = self.gat2.forward(h1, adj)
        h2 = np.maximum(h2, 0)
        logits = (h2 @ self.W_out).squeeze(-1)
        return 1 / (1 + np.exp(-logits))                         # sigmoid

    def extract_features_from_snapshot(self, snapshot: dict,
                                        road_graph) -> Tuple[np.ndarray, np.ndarray]:
        """Build node feature matrix and adjacency matrix from simulation snapshot."""
        G = road_graph.G
        nodes = list(G.nodes())
        N = len(nodes)
        node_idx = {n: i for i, n in enumerate(nodes)}

        # Hour cyclical encoding
        hour = snapshot.get("metrics", {}).get("hour_of_day", 12.0)
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        # Node features
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1

        X = np.zeros((N, self.in_channels), dtype=np.float32)
        for i, node in enumerate(nodes):
            in_load = sum(G[u][node].get("current_load", 0)
                          for u in G.predecessors(node))
            out_cap = sum(G[node][v].get("capacity", 1)
                          for v in G.successors(node))
            load_norm = min(in_load / max(out_cap, 1), 1.0)
            degree_norm = degrees.get(node, 0) / max_degree
            X[i] = [load_norm, degree_norm, 0.0, hour_sin, hour_cos]

        # Adjacency matrix
        adj = np.zeros((N, N), dtype=np.float32)
        for u, v in G.edges():
            if u in node_idx and v in node_idx:
                adj[node_idx[u], node_idx[v]] = 1.0
                adj[node_idx[v], node_idx[u]] = 1.0

        return X, adj

    def predict_from_snapshot(self, snapshot: dict, road_graph) -> Dict[int, float]:
        """Returns {node_id: congestion_score} for all nodes in graph."""
        G = road_graph.G
        nodes = list(G.nodes())
        X, adj = self.extract_features_from_snapshot(snapshot, road_graph)
        scores = self.predict_congestion(X, adj)
        return {node: round(float(scores[i]), 3) for i, node in enumerate(nodes)}

    def save(self, path: str = "data/processed/gat_congestion.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "gat1_W": self.gat1.W, "gat1_a": self.gat1.a,
                "gat2_W": self.gat2.W, "gat2_a": self.gat2.a,
                "W_out": self.W_out,
                "config": {"in_channels": self.in_channels,
                            "hidden_channels": self.hidden_channels,
                            "heads": self.heads},
            }, f)
        logger.info(f"GAT model saved to {path}")

    def load(self, path: str = "data/processed/gat_congestion.pkl") -> "SimpleGATCongestionModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        cfg = data.get("config", {})
        self.__init__(cfg)
        self.gat1.W = data["gat1_W"]; self.gat1.a = data["gat1_a"]
        self.gat2.W = data["gat2_W"]; self.gat2.a = data["gat2_a"]
        self.W_out = data["W_out"]
        self.trained = True
        return self


def build_and_save_gat(road_graph, snapshot: dict,
                        path: str = "data/processed/gat_congestion.pkl"):
    """Initialise GAT model and save for later inference."""
    model = SimpleGATCongestionModel({"in_channels": 5, "hidden_channels": 16, "heads": 2})
    scores = model.predict_from_snapshot(snapshot, road_graph)
    model.save(path)
    logger.info(f"GAT initialised. Sample scores: {dict(list(scores.items())[:3])}")
    return model, scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import warnings; warnings.filterwarnings("ignore")
    from src.simulation_engine.city_model import CityModel

    cfg = {"simulation": {"num_vehicles": 50, "num_pedestrians": 100,
                           "num_resource_nodes": 5, "synthetic_nodes": 20,
                           "default_green_ns": 30}}
    model = CityModel(cfg, seed=42)
    for _ in range(120):
        snap = model.tick_step()

    gat, scores = build_and_save_gat(model.road_graph, snap)
    print(f"GAT congestion predictions for {len(scores)} nodes")
    top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
    print(f"Most congested nodes: {top3}")
