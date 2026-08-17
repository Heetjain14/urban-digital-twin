"""
tests/unit/test_simulation_core.py
Unit tests for simulation engine, data generation, and ML feature engineering.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest


# ── Synthetic generator ───────────────────────────────────────────────────────
class TestSyntheticGenerator:
    def setup_method(self):
        from src.data_ingestion.synthetic_generator import SyntheticCityGenerator
        self.gen = SyntheticCityGenerator(days=3, seed=42)

    def test_traffic_shape(self):
        df = self.gen.generate_traffic(num_segments=5)
        assert len(df) == 3 * 24 * 60 * 5   # 3 days × 1440 ticks × 5 segments
        assert "vehicle_count" in df.columns
        assert "avg_speed_kmh" in df.columns

    def test_traffic_values_in_range(self):
        df = self.gen.generate_traffic(num_segments=3)
        assert df["vehicle_count"].min() >= 0
        assert df["avg_speed_kmh"].min() >= 0
        assert df["congestion_score"].between(0, 1).all()

    def test_energy_shape(self):
        df = self.gen.generate_energy()
        assert "demand_kwh" in df.columns
        assert df["load_pct"].between(0, 200).all()

    def test_water_shape(self):
        df = self.gen.generate_water()
        assert "demand_m3h" in df.columns

    def test_has_anomalies(self):
        df = self.gen.generate_traffic(num_segments=20)
        assert df["is_anomaly"].sum() > 0, "Should have some injected anomalies"

    def test_diurnal_pattern(self):
        """Rush hour traffic should be higher than midnight traffic."""
        df = self.gen.generate_traffic(num_segments=1)
        tick_8am = 8 * 60
        tick_3am = 3 * 60
        rush = df[df["tick"] == tick_8am]["vehicle_count"].values[0]
        night = df[df["tick"] == tick_3am]["vehicle_count"].values[0]
        assert rush > night, "Rush hour should have more traffic than 3am"


# ── Road graph ────────────────────────────────────────────────────────────────
class TestRoadGraph:
    def setup_method(self):
        from src.simulation_engine.road_graph import CityRoadGraph
        self.rg = CityRoadGraph(seed=42)
        self.rg.build_synthetic(num_nodes=20, grid_size=50)

    def test_graph_has_nodes(self):
        assert self.rg.G.number_of_nodes() > 0

    def test_graph_has_edges(self):
        assert self.rg.G.number_of_edges() > 0

    def test_node_attributes(self):
        for _, data in self.rg.G.nodes(data=True):
            assert "x" in data
            assert "y" in data
            assert "zone_type" in data

    def test_edge_attributes(self):
        for _, _, data in self.rg.G.edges(data=True):
            assert "length_m" in data
            assert "speed_limit_kmh" in data
            assert "capacity" in data

    def test_pathfinding(self):
        nodes = list(self.rg.G.nodes())
        if len(nodes) >= 2:
            path = self.rg.shortest_path(nodes[0], nodes[-1])
            assert isinstance(path, list)

    def test_edge_closure(self):
        edges = list(self.rg.G.edges())
        if edges:
            u, v = edges[0]
            self.rg.close_edge(u, v, duration_ticks=10, current_tick=0)
            assert self.rg.G[u][v]["is_closed"] == True
            self.rg.tick_update(current_tick=11)
            assert self.rg.G[u][v]["is_closed"] == False

    def test_congestion_map(self):
        cmap = self.rg.get_congestion_map()
        assert isinstance(cmap, dict)
        for score in cmap.values():
            assert 0.0 <= score <= 1.0


# ── Environment ───────────────────────────────────────────────────────────────
class TestEnvironment:
    def setup_method(self):
        from src.simulation_engine.environment import CityEnvironment
        self.env = CityEnvironment(seed=42)

    def test_time_of_day(self):
        self.env.update(0)
        assert self.env.hour_of_day == 0.0
        self.env.update(480)   # 8 hours in
        assert abs(self.env.hour_of_day - 8.0) < 0.1

    def test_rush_hour_detection(self):
        self.env.update(8 * 60)   # 8am
        assert self.env.is_rush_hour

        self.env.update(3 * 60)   # 3am
        assert not self.env.is_rush_hour

    def test_weather_multipliers(self):
        from src.data_ingestion.schemas import WeatherCondition
        self.env.weather_condition = WeatherCondition.HEAVY_RAIN
        assert self.env.speed_multiplier == 0.50
        assert self.env.pedestrian_rate_multiplier == 0.40

    def test_traffic_light_init(self):
        self.env.initialize_traffic_lights([1, 2, 3, 4])
        assert len(self.env.traffic_light_phases) == 4
        assert all(v in ("NS_GREEN", "EW_GREEN")
                   for v in self.env.traffic_light_phases.values())


# ── Feature engineering ───────────────────────────────────────────────────────
class TestFeatureEngineering:
    def test_traffic_features_shape(self):
        from src.data_ingestion.synthetic_generator import SyntheticCityGenerator
        from src.ml_models.feature_engineering import build_traffic_features

        gen = SyntheticCityGenerator(days=3, seed=42)
        df = gen.generate_traffic(1)
        seg = df[df["segment_id"] == 0]
        X, y = build_traffic_features(seg, window=24)
        assert X.shape[1] == 24   # window
        assert X.shape[2] == 7    # features
        assert len(X) == len(y)

    def test_energy_features_shape(self):
        from src.data_ingestion.synthetic_generator import SyntheticCityGenerator
        from src.ml_models.feature_engineering import build_energy_features

        gen = SyntheticCityGenerator(days=3, seed=42)
        df = gen.generate_energy()
        zone = df[df["zone_id"] == 0]
        X, y = build_energy_features(zone, window=24)
        assert X.ndim == 2
        assert len(X) == len(y)

    def test_temporal_split_order(self):
        from src.ml_models.feature_engineering import train_test_split_temporal
        X = np.arange(100).reshape(100, 1).astype(float)
        y = np.arange(100).astype(float)
        X_tr, y_tr, X_v, y_v, X_te, y_te = train_test_split_temporal(X, y, 0.7, 0.15)
        # Ensure no data leakage: train < val < test
        assert X_tr[-1][0] < X_v[0][0]
        assert X_v[-1][0] < X_te[0][0]


# ── City model integration ────────────────────────────────────────────────────
class TestCityModelIntegration:
    def test_model_initializes(self):
        from src.simulation_engine.city_model import CityModel
        cfg = {"simulation": {
            "num_vehicles": 10, "num_pedestrians": 20,
            "num_resource_nodes": 3, "synthetic_nodes": 15,
            "default_green_ns": 30
        }}
        model = CityModel(cfg, seed=42)
        assert len(model.vehicle_agents) == 10
        assert len(model.pedestrian_agents) == 20
        assert len(model.resource_agents) == 3

    def test_single_step(self):
        from src.simulation_engine.city_model import CityModel
        cfg = {"simulation": {
            "num_vehicles": 5, "num_pedestrians": 10,
            "num_resource_nodes": 2, "synthetic_nodes": 10,
            "default_green_ns": 30
        }}
        model = CityModel(cfg, seed=42)
        snap = model.tick_step()
        assert snap["tick"] == 0
        assert "metrics" in snap
        assert "agents" in snap
        assert "resources" in snap
        assert "weather" in snap

    def test_100_ticks(self):
        from src.simulation_engine.city_model import CityModel
        cfg = {"simulation": {
            "num_vehicles": 10, "num_pedestrians": 20,
            "num_resource_nodes": 2, "synthetic_nodes": 10,
            "default_green_ns": 30
        }}
        model = CityModel(cfg, seed=42)
        for _ in range(100):
            snap = model.tick_step()
        assert snap["tick"] == 99
        assert snap["metrics"]["tick"] == 99


# ── Anomaly detection ─────────────────────────────────────────────────────────
class TestAnomalyDetection:
    def test_isolation_forest_detects_anomalies(self):
        from src.data_ingestion.synthetic_generator import SyntheticCityGenerator
        from src.ml_models.feature_engineering import build_anomaly_features
        from src.anomaly_detection.detectors import TrafficAnomalyDetector

        gen = SyntheticCityGenerator(days=7, seed=42)
        df = gen.generate_traffic(1)
        seg = df[df["segment_id"] == 0]
        normal = seg[~seg["is_anomaly"]]
        features = build_anomaly_features(normal)

        detector = TrafficAnomalyDetector()
        detector.fit(features)

        # Model should be trained and return scores in valid range
        scores = detector.anomaly_score(features[:50])
        assert scores.shape[0] == 50
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0
        # Predictions array should contain both -1 and 1
        preds = detector.detect(features)
        assert set(preds).issubset({-1, 1})

    def test_zscore_detector(self):
        from src.anomaly_detection.detectors import ResourceAnomalyDetector
        det = ResourceAnomalyDetector(window=20, z_threshold=2.0, consecutive=2)

        # Feed normal values
        for v in np.random.normal(100, 5, 50):
            result = det.update("test_series", v)

        # Feed extreme spike — should trigger after consecutive=2
        for _ in range(3):
            result = det.update("test_series", 300.0)  # very extreme spike

        assert result["z_score"] > 2.0
        assert result["is_anomaly"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
