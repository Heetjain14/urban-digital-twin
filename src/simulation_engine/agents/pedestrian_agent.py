"""
agents/pedestrian_agent.py  —  Mesa 3.x compatible
"""
from __future__ import annotations
import numpy as np
import mesa
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation_engine.city_model import CityModel


class PedestrianAgent(mesa.Agent):
    BASE_SPEED = 1.4

    def __init__(self, model: "CityModel", home_x: float, home_y: float,
                 work_x: float, work_y: float, depart_hour: float = 8.0):
        super().__init__(model)
        self.home = (np.clip(home_x, 0, 49), np.clip(home_y, 0, 49))
        self.work = (np.clip(work_x, 0, 49), np.clip(work_y, 0, 49))
        self.depart_hour = depart_hour
        self.x, self.y = self.home
        self.destination: Optional[Tuple[float, float]] = None
        self.state: str = "idle"
        self.speed: float = 0.0
        self._trips_today = 0
        self._max_trips_today = 4
        self._next_depart_tick = int(depart_hour * 60)

    def step(self):
        env = self.model.environment
        tick = env.tick
        if tick % 1440 == 0 and tick > 0:
            self._max_trips_today = max(1, int(np.random.poisson(4 * env.pedestrian_rate_multiplier)))
            self._trips_today = 0
            self._next_depart_tick = tick + int(self.depart_hour * 60)

        if self.state == "idle":
            if tick >= self._next_depart_tick and self._trips_today < self._max_trips_today:
                r = self.model.rng.random()
                self.destination = self.work if r < 0.4 else (
                    self.home if r < 0.6 else
                    (float(self.model.rng.uniform(5, 45)), float(self.model.rng.uniform(5, 45))))
                self.state = "walking"
                self._trips_today += 1
        elif self.state == "walking":
            self._walk()

    def _walk(self):
        if not self.destination:
            self.state = "idle"; return
        dx = self.destination[0] - self.x
        dy = self.destination[1] - self.y
        dist = np.hypot(dx, dy)
        if dist < 0.5:
            self.x, self.y = self.destination
            self.state = "idle"; self.speed = 0.0
            self._next_depart_tick = self.model.environment.tick + int(self.model.rng.integers(30, 120))
            return
        speed = max(self.model.rng.normal(self.BASE_SPEED, 0.3), 0.5)
        speed *= self.model.environment.pedestrian_rate_multiplier
        self.speed = speed
        step = min(speed * 60 / 100.0, dist)
        self.x = np.clip(self.x + (dx / dist) * step, 0, 49)
        self.y = np.clip(self.y + (dy / dist) * step, 0, 49)

    def get_record(self) -> dict:
        return {"agent_id": self.unique_id, "agent_type": "pedestrian",
                "x": round(self.x, 2), "y": round(self.y, 2),
                "speed": round(self.speed, 2), "state": self.state,
                "destination_x": round(self.destination[0], 2) if self.destination else None,
                "destination_y": round(self.destination[1], 2) if self.destination else None}
