"""
synthetic_generator.py
Generates statistically realistic city time-series data without external APIs.
Produces traffic, energy, and water demand with diurnal/weekly patterns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# ── Diurnal profile helpers ───────────────────────────────────────────────────

def _bimodal_traffic_profile(hours: np.ndarray) -> np.ndarray:
    """AM peak ~8h, PM peak ~17h, valley overnight."""
    am = 0.85 * np.exp(-0.5 * ((hours - 8.0) / 1.2) ** 2)
    pm = 1.00 * np.exp(-0.5 * ((hours - 17.0) / 1.5) ** 2)
    night = 0.08 * np.exp(-0.5 * ((hours - 0) / 2.0) ** 2)
    return np.clip(am + pm + night, 0.05, 1.0)


def _energy_profile(hours: np.ndarray, is_weekend: np.ndarray) -> np.ndarray:
    """Energy demand: residential evening peak, commercial daytime."""
    commercial = 0.6 * np.exp(-0.5 * ((hours - 12.0) / 3.0) ** 2)
    residential = 0.9 * np.exp(-0.5 * ((hours - 19.5) / 2.5) ** 2)
    base = 0.25
    weekday_factor = np.where(is_weekend, 0.7, 1.0)
    return np.clip((base + commercial + residential) * weekday_factor, 0.1, 1.0)


def _water_profile(hours: np.ndarray, is_weekend: np.ndarray) -> np.ndarray:
    """Water demand: morning and evening peaks."""
    morning = 0.8 * np.exp(-0.5 * ((hours - 7.5) / 1.0) ** 2)
    evening = 0.7 * np.exp(-0.5 * ((hours - 20.0) / 1.5) ** 2)
    base = 0.2
    return np.clip((base + morning + evening), 0.1, 1.0)


# ── Main generator ────────────────────────────────────────────────────────────

class SyntheticCityGenerator:
    """
    Generates synthetic time-series for a city district.

    Parameters
    ----------
    days : int         How many days of data to generate
    tick_minutes : int Minutes per simulation tick (default 1)
    num_zones : int    Number of city zones
    seed : int         Random seed for reproducibility
    """

    def __init__(self, days: int = 30, tick_minutes: int = 1,
                 num_zones: int = 5, seed: int = 42):
        self.days = days
        self.tick_minutes = tick_minutes
        self.num_zones = num_zones
        self.rng = np.random.default_rng(seed)
        self.ticks_per_day = 24 * 60 // tick_minutes
        self.total_ticks = days * self.ticks_per_day

    # ── Traffic ───────────────────────────────────────────────────────────────
    def generate_traffic(self, num_segments: int = 20) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        tick, segment_id, vehicle_count, avg_speed_kmh, congestion_score
        """
        rows = []
        t = np.arange(self.total_ticks)
        hours = (t * self.tick_minutes / 60) % 24
        day_of_week = (t * self.tick_minutes // (60 * 24)) % 7
        is_weekend = (day_of_week >= 5).astype(float)

        base_profile = _bimodal_traffic_profile(hours)
        # Weekend = 70% of weekday traffic
        profile = base_profile * (1 - 0.3 * is_weekend)

        for seg in range(num_segments):
            capacity = self.rng.integers(30, 80)  # vehicles/min capacity
            # Each segment has slightly different timing
            phase_shift = self.rng.uniform(-0.5, 0.5)
            seg_hours = (hours + phase_shift) % 24
            seg_profile = _bimodal_traffic_profile(seg_hours) * (1 - 0.3 * is_weekend)

            noise = self.rng.normal(0, 0.05, self.total_ticks)
            raw_count = np.clip(seg_profile + noise, 0, 1) * capacity
            vehicle_count = np.round(raw_count).astype(int)

            # Speed degrades with congestion (Greenshields model)
            load = raw_count / capacity
            free_flow_speed = self.rng.uniform(40, 70)
            avg_speed = free_flow_speed * (1 - 0.8 * load ** 2)
            avg_speed = np.clip(avg_speed, 5, free_flow_speed)

            # Inject random accidents (0.01% chance per tick)
            accident_mask = self.rng.random(self.total_ticks) < 0.0001
            vehicle_count[accident_mask] = np.minimum(
                vehicle_count[accident_mask] * 3, capacity)
            avg_speed[accident_mask] = self.rng.uniform(3, 8, accident_mask.sum())

            congestion = np.clip(vehicle_count / capacity, 0, 1)

            for i in range(self.total_ticks):
                rows.append({
                    "tick": i, "segment_id": seg,
                    "vehicle_count": int(vehicle_count[i]),
                    "avg_speed_kmh": round(float(avg_speed[i]), 1),
                    "congestion_score": round(float(congestion[i]), 3),
                    "capacity": capacity,
                    "is_anomaly": bool(accident_mask[i]),
                })

        df = pd.DataFrame(rows)
        logger.info(f"Traffic data: {len(df):,} rows for {num_segments} segments over {self.days} days")
        return df

    # ── Energy ────────────────────────────────────────────────────────────────
    def generate_energy(self) -> pd.DataFrame:
        """Returns DataFrame: tick, zone_id, demand_kwh, capacity_kwh, load_pct, is_anomaly"""
        rows = []
        t = np.arange(self.total_ticks)
        hours = (t * self.tick_minutes / 60) % 24
        day_of_week = (t * self.tick_minutes // (60 * 24)) % 7
        is_weekend = (day_of_week >= 5).astype(float)

        zone_types = ["residential", "commercial", "industrial", "campus", "transit"]
        capacities = [500, 800, 1200, 400, 300]

        for zone_id in range(self.num_zones):
            zone_type = zone_types[zone_id % len(zone_types)]
            cap = capacities[zone_id % len(capacities)]
            profile = _energy_profile(hours, is_weekend)

            # Zone-specific modifiers
            if zone_type == "industrial":
                profile = profile * 1.3 * (1 - 0.4 * is_weekend)
            elif zone_type == "residential":
                profile = profile * 0.8 + 0.1

            noise = self.rng.normal(0, 0.03, self.total_ticks)
            demand = np.clip(profile + noise, 0.05, 1.2) * cap

            # Spike anomalies (~0.005% chance)
            spike_mask = self.rng.random(self.total_ticks) < 0.00005
            demand[spike_mask] *= self.rng.uniform(1.4, 1.8, spike_mask.sum())
            demand = np.clip(demand, 0, cap * 1.5)

            for i in range(self.total_ticks):
                rows.append({
                    "tick": i, "zone_id": zone_id, "zone_type": zone_type,
                    "demand_kwh": round(float(demand[i]), 2),
                    "capacity_kwh": float(cap),
                    "load_pct": round(float(demand[i] / cap * 100), 1),
                    "is_anomaly": bool(spike_mask[i]),
                })

        return pd.DataFrame(rows)

    # ── Water ─────────────────────────────────────────────────────────────────
    def generate_water(self) -> pd.DataFrame:
        """Returns DataFrame: tick, zone_id, demand_m3h, capacity_m3h, load_pct, is_anomaly"""
        rows = []
        t = np.arange(self.total_ticks)
        hours = (t * self.tick_minutes / 60) % 24
        day_of_week = (t * self.tick_minutes // (60 * 24)) % 7
        is_weekend = (day_of_week >= 5).astype(float)

        zone_caps = [80, 120, 200, 60, 40]

        for zone_id in range(self.num_zones):
            cap = zone_caps[zone_id % len(zone_caps)]
            profile = _water_profile(hours, is_weekend)
            noise = self.rng.normal(0, 0.03, self.total_ticks)
            demand = np.clip(profile + noise, 0.05, 1.0) * cap

            # Leak anomalies: sustained high usage
            leak_starts = np.where(self.rng.random(self.total_ticks) < 0.00002)[0]
            leak_mask = np.zeros(self.total_ticks, dtype=bool)
            for ls in leak_starts:
                end = min(ls + self.rng.integers(30, 120), self.total_ticks)
                leak_mask[ls:end] = True
                demand[ls:end] *= self.rng.uniform(1.3, 1.6)

            demand = np.clip(demand, 0, cap * 1.5)

            for i in range(self.total_ticks):
                rows.append({
                    "tick": i, "zone_id": zone_id,
                    "demand_m3h": round(float(demand[i]), 2),
                    "capacity_m3h": float(cap),
                    "load_pct": round(float(demand[i] / cap * 100), 1),
                    "is_anomaly": bool(leak_mask[i]),
                })

        return pd.DataFrame(rows)

    # ── Population movement ───────────────────────────────────────────────────
    def generate_population_od(self, num_agents: int = 700) -> pd.DataFrame:
        """
        Origin-destination records for agent initialization.
        Returns home_x, home_y, work_x, work_y, zone_type, depart_hour
        """
        zone_centers = {
            "residential": (10, 40), "commercial": (25, 25),
            "industrial": (40, 15), "campus": (15, 15), "transit": (25, 40),
        }
        records = []
        for i in range(num_agents):
            home_zone = self.rng.choice(["residential", "campus"])
            work_zone = self.rng.choice(["commercial", "industrial", "campus", "transit"])
            hc = zone_centers[home_zone]
            wc = zone_centers[work_zone]
            records.append({
                "agent_id": i,
                "home_x": float(self.rng.normal(hc[0], 4)),
                "home_y": float(self.rng.normal(hc[1], 4)),
                "work_x": float(self.rng.normal(wc[0], 4)),
                "work_y": float(self.rng.normal(wc[1], 4)),
                "zone_type": home_zone,
                "depart_hour": float(self.rng.normal(8.0, 0.75)),
                "return_hour": float(self.rng.normal(17.5, 0.75)),
                "agent_type": "vehicle" if i < num_agents * 0.4 else "pedestrian",
            })
        return pd.DataFrame(records)

    # ── Save all ──────────────────────────────────────────────────────────────
    def generate_all(self, output_dir: str = "data/synthetic") -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating {self.days} days of synthetic city data...")

        traffic = self.generate_traffic(num_segments=20)
        energy = self.generate_energy()
        water = self.generate_water()
        population = self.generate_population_od()

        traffic.to_parquet(f"{output_dir}/traffic.parquet", index=False)
        energy.to_parquet(f"{output_dir}/energy.parquet", index=False)
        water.to_parquet(f"{output_dir}/water.parquet", index=False)
        population.to_parquet(f"{output_dir}/population_od.parquet", index=False)

        logger.info(f"Saved to {output_dir}/")
        return {"traffic": traffic, "energy": energy,
                "water": water, "population": population}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = SyntheticCityGenerator(days=7, seed=42)
    data = gen.generate_all("data/synthetic")
    print(f"Traffic rows: {len(data['traffic']):,}")
    print(f"Energy rows:  {len(data['energy']):,}")
    print(f"Water rows:   {len(data['water']):,}")
    print(f"Agents:       {len(data['population']):,}")
