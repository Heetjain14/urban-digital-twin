"""
agents/resource_agent.py  —  Mesa 3.x compatible
"""
from __future__ import annotations
import numpy as np
import mesa
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation_engine.city_model import CityModel


def _e(hour, zone):
    if zone == "residential":
        return 0.3 + 0.5*np.exp(-0.5*((hour-19.5)/2.5)**2) + 0.25*np.exp(-0.5*((hour-7.5)/1.5)**2)
    elif zone == "commercial":
        return 0.2 + 0.8*np.exp(-0.5*((hour-12.0)/3.5)**2)
    elif zone == "industrial":
        return 0.5 + 0.4*(1 if 6<=hour<=22 else 0)
    elif zone == "campus":
        return 0.2 + 0.6*np.exp(-0.5*((hour-13.0)/4.0)**2)
    return 0.3 + 0.4*(np.exp(-0.5*((hour-8)/1.2)**2) + np.exp(-0.5*((hour-17)/1.5)**2))


def _w(hour):
    return 0.2 + 0.6*np.exp(-0.5*((hour-7.5)/1.0)**2) + 0.5*np.exp(-0.5*((hour-20.0)/1.5)**2)


class ResourceAgent(mesa.Agent):
    ANOMALY_THRESHOLD = 0.90

    def __init__(self, model: "CityModel", zone_id: int, zone_type: str,
                 x: float, y: float, energy_capacity=500.0, water_capacity=80.0):
        super().__init__(model)
        self.zone_id = zone_id
        self.zone_type = zone_type
        self.x = x; self.y = y
        self.energy_capacity = energy_capacity
        self.water_capacity = water_capacity
        self.energy_demand = energy_capacity * 0.3
        self.water_demand = water_capacity * 0.3
        self.anomaly_flag = False
        self.anomaly_type = ""
        self._leak_active = False
        self._leak_mult = 1.0
        self._leak_end = 0

    def step(self):
        env = self.model.environment
        hour = env.hour_of_day
        wknd = 0.7 if env.is_weekend else 1.0
        ne = self.model.rng.normal(1.0, 0.03)
        nw = self.model.rng.normal(1.0, 0.03)
        self.energy_demand = _e(hour, self.zone_type) * wknd * ne * env.energy_multiplier * self.energy_capacity
        self.water_demand  = _w(hour) * wknd * nw * self.water_capacity

        if self._leak_active:
            if env.tick >= self._leak_end:
                self._leak_active = False; self._leak_mult = 1.0
            else:
                self.water_demand *= self._leak_mult

        if env.scenario_active == "power_outage" and env.scenario_params.get("zone_id", -1) == self.zone_id:
            self.energy_demand = 0.0

        el = self.energy_demand / max(self.energy_capacity, 1)
        wl = self.water_demand / max(self.water_capacity, 1)
        self.anomaly_flag = el > self.ANOMALY_THRESHOLD or wl > self.ANOMALY_THRESHOLD
        self.anomaly_type = ("energy_critical" if el > 1.1 else
                             "water_critical" if wl > 1.1 else
                             "overload_warning" if self.anomaly_flag else "")

        if not self._leak_active and self.model.rng.random() < 0.00002:
            self._leak_active = True
            self._leak_mult = self.model.rng.uniform(1.3, 1.6)
            self._leak_end = env.tick + int(self.model.rng.integers(60, 240))

    @property
    def energy_load_pct(self):
        return round(self.energy_demand / max(self.energy_capacity, 1) * 100, 1)

    @property
    def water_load_pct(self):
        return round(self.water_demand / max(self.water_capacity, 1) * 100, 1)

    def get_record(self):
        return {"node_id": self.zone_id, "zone_type": self.zone_type,
                "x": round(self.x, 2), "y": round(self.y, 2),
                "energy_demand": round(self.energy_demand, 2),
                "energy_capacity": self.energy_capacity,
                "water_demand": round(self.water_demand, 2),
                "water_capacity": self.water_capacity,
                "energy_load_pct": self.energy_load_pct,
                "water_load_pct": self.water_load_pct,
                "anomaly_flag": self.anomaly_flag}
