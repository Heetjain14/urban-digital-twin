"""
agents/vehicle_agent.py  —  Mesa 3.x compatible
"""
from __future__ import annotations
import numpy as np
import mesa
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation_engine.city_model import CityModel


class VehicleAgent(mesa.Agent):
    def __init__(self, model: "CityModel", home_node: int, work_node: int,
                 depart_hour: float = 8.0, return_hour: float = 17.5):
        super().__init__(model)
        self.home_node = home_node
        self.work_node = work_node
        self.depart_hour = depart_hour
        self.return_hour = return_hour

        self.current_node: int = home_node
        self.x: float = model.road_graph.G.nodes[home_node]["x"]
        self.y: float = model.road_graph.G.nodes[home_node]["y"]
        self.path: List[int] = []
        self.state: str = "parked_home"
        self.speed_kmh: float = 0.0
        self.wait_ticks: int = 0
        self.total_wait: int = 0
        self.trips_completed: int = 0

    def step(self):
        env = self.model.environment
        hour = env.hour_of_day

        if self.state == "parked_home":
            if hour >= self.depart_hour and not env.is_weekend:
                self._start_trip(self.work_node, "driving_to_work")
        elif self.state == "driving_to_work":
            self._drive_step()
            if self.current_node == self.work_node and not self.path:
                self.state = "parked_work"; self.speed_kmh = 0.0; self.trips_completed += 1
        elif self.state == "parked_work":
            if hour >= self.return_hour:
                self._start_trip(self.home_node, "driving_home")
        elif self.state == "driving_home":
            self._drive_step()
            if self.current_node == self.home_node and not self.path:
                self.state = "parked_home"; self.speed_kmh = 0.0; self.trips_completed += 1

    def _start_trip(self, destination: int, next_state: str):
        path = self.model.road_graph.shortest_path(self.current_node, destination)
        if len(path) > 1:
            self.path = path[1:]
            self.state = next_state

    def _drive_step(self):
        if not self.path:
            return
        env = self.model.environment
        next_n = self.path[0]
        G = self.model.road_graph.G

        if not G.has_edge(self.current_node, next_n) or G[self.current_node][next_n].get("is_closed"):
            self._reroute(); return

        if self._blocked_by_light(self.current_node, next_n):
            self.wait_ticks += 1; self.total_wait += 1; self.speed_kmh = 0.0; return

        d = G[self.current_node][next_n]
        load = d.get("current_load", 0); cap = max(d.get("capacity", 1), 1)
        congestion = min(load / cap, 1.0)
        speed = d.get("speed_limit_kmh", 40) * (1 - 0.8 * congestion**2) * env.speed_multiplier
        self.speed_kmh = max(speed, 5.0)

        self.model.road_graph.update_edge_load(self.current_node, next_n, -1)
        self.current_node = next_n
        self.path.pop(0)
        self.model.road_graph.update_edge_load(self.current_node, next_n, 1)

        if G.has_node(self.current_node):
            self.x = G.nodes[self.current_node]["x"]
            self.y = G.nodes[self.current_node]["y"]
        self.wait_ticks = 0

    def _blocked_by_light(self, fn: int, tn: int) -> bool:
        env = self.model.environment
        if fn not in self.model.intersection_ids:
            return False
        phase = env.traffic_light_phases.get(fn, "NS_GREEN")
        if phase == "ALL_RED":
            return True
        G = self.model.road_graph.G
        if not G.has_node(fn) or not G.has_node(tn):
            return False
        dy = abs(G.nodes[tn]["y"] - G.nodes[fn]["y"])
        dx = abs(G.nodes[tn]["x"] - G.nodes[fn]["x"])
        direction = "NS" if dy >= dx else "EW"
        return (phase == "NS_GREEN" and direction == "EW") or \
               (phase == "EW_GREEN" and direction == "NS")

    def _reroute(self):
        if not self.path:
            return
        dest = self.path[-1]
        new = self.model.road_graph.shortest_path(self.current_node, dest)
        self.path = new[1:] if len(new) > 1 else []

    def get_record(self) -> dict:
        return {"agent_id": self.unique_id, "agent_type": "vehicle",
                "x": round(self.x, 2), "y": round(self.y, 2),
                "speed": round(self.speed_kmh, 1), "state": self.state,
                "destination_x": None, "destination_y": None}
