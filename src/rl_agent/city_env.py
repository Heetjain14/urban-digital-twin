"""
rl_agent/city_env.py
Gymnasium environment wrapping CityModel for RL training.
The RL agent controls traffic light phase durations.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CityTrafficEnv(gym.Env):
    """
    Custom Gymnasium environment for Urban Digital Twin RL.

    Observation space (20-dim vector):
    - queue_ns × 4 intersections     (4 dims, normalized)
    - queue_ew × 4 intersections     (4 dims, normalized)
    - avg_speed × 4 road zones       (4 dims, normalized)
    - energy_load × 3 districts      (3 dims, normalized)
    - hour_sin, hour_cos             (2 dims)
    - weather_rain, weather_temp     (2 dims)
    - active_anomaly_flag            (1 dim)

    Action space: Discrete(16)
    Encodes green-phase duration for 4 intersections × 4 options:
    [15s, 30s, 45s, 60s] per intersection → 4^2 = 16 combinations
    """

    metadata = {"render_modes": ["human"]}
    PHASE_OPTIONS = [15, 30, 45, 60]   # possible green durations (ticks)

    def __init__(self, config: dict, seed: int = 42):
        super().__init__()
        self.config = config
        self._seed = seed
        self.ticks_per_step = config.get("ticks_per_step", 10)
        self.episode_length = config.get("episode_length_ticks", 1440)
        self.reward_weights = config.get("reward_weights", {
            "wait_time_penalty": -0.6,
            "throughput_reward": 0.3,
            "energy_efficiency": 0.1,
            "anomaly_penalty": -5.0,
        })

        # Spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Discrete(16)

        self.model = None
        self._episode_tick = 0
        self._prev_wait = 0.0
        self._prev_throughput = 0
        self._total_reward = 0.0

    # ── Gymnasium interface ───────────────────────────────────────────────────
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        import yaml
        from pathlib import Path
        from src.simulation_engine.city_model import CityModel

        config_path = Path("config/simulation.yaml")
        if config_path.exists():
            with open(config_path) as f:
                sim_config = yaml.safe_load(f)
        else:
            sim_config = self.config

        # Lightweight config for fast RL rollouts
        sim_config["simulation"]["num_vehicles"] = 80
        sim_config["simulation"]["num_pedestrians"] = 100
        sim_config["simulation"]["synthetic_nodes"] = 25

        self.model = CityModel(sim_config, seed=self._seed + self._episode_tick)
        self._episode_tick = 0
        self._prev_wait = 0.0
        self._prev_throughput = 0
        self._total_reward = 0.0

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Decode action → phase durations for 4 intersections
        self._apply_action(action)

        # Advance simulation N ticks
        for _ in range(self.ticks_per_step):
            self.model.tick_step()
        self._episode_tick += self.ticks_per_step

        obs = self._get_observation()
        reward = self._compute_reward()
        self._total_reward += reward

        terminated = self._episode_tick >= self.episode_length
        truncated = False

        info = {
            "episode_tick": self._episode_tick,
            "total_reward": self._total_reward,
            "sim_time": self.model.environment.sim_datetime_str,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        snap = self.model._build_snapshot()
        m = snap["metrics"]
        print(f"  [RL Env] tick={m['tick']} speed={m['avg_vehicle_speed']:.1f} "
              f"moving={m['vehicles_moving']} reward_so_far={self._total_reward:.2f}")

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_observation(self) -> np.ndarray:
        if self.model is None:
            return np.zeros(20, dtype=np.float32)

        env = self.model.environment
        intersections = self.model.intersection_ids[:4]

        obs = []
        # Queue lengths at 4 intersections (normalized to 0–1)
        for iid in intersections[:4]:
            obs.append(min(env.traffic_light_timers.get(iid, 0) / 60.0, 1.0))
        while len(obs) < 4:
            obs.append(0.0)

        # Again for EW (use phase timer remainder as proxy)
        for iid in intersections[:4]:
            phase = env.traffic_light_phases.get(iid, "NS_GREEN")
            obs.append(1.0 if phase == "EW_GREEN" else 0.0)
        while len(obs) < 8:
            obs.append(0.0)

        # Average speed in 4 road zones (normalized)
        vehicle_speeds = [a.speed_kmh / 80.0
                          for a in self.model.vehicle_agents
                          if a.state.startswith("driving")]
        if vehicle_speeds:
            mean_speed = np.mean(vehicle_speeds)
        else:
            mean_speed = 0.0
        obs.extend([mean_speed] * 4)  # simplified: same value for all zones

        # Energy load per district (3 districts, normalized)
        resource_loads = [r.energy_load_pct / 100.0
                          for r in self.model.resource_agents[:3]]
        while len(resource_loads) < 3:
            resource_loads.append(0.0)
        obs.extend(resource_loads)

        # Cyclical time
        h = env.hour_of_day
        obs.append(np.sin(2 * np.pi * h / 24.0))
        obs.append(np.cos(2 * np.pi * h / 24.0))

        # Weather
        obs.append(env.precipitation_mm_h / 25.0)
        obs.append((env.temperature_c - 20.0) / 20.0)

        # Anomaly flag
        any_anomaly = any(r.anomaly_flag for r in self.model.resource_agents)
        obs.append(1.0 if any_anomaly else 0.0)

        return np.clip(np.array(obs[:20], dtype=np.float32), -1.0, 1.0)

    # ── Action ────────────────────────────────────────────────────────────────
    def _apply_action(self, action: int):
        """
        Decode discrete action → 4 pairs of (intersection, duration).
        Action 0–15: binary encoding of 4 intersections × 2 duration choices.
        """
        intersections = self.model.intersection_ids[:4]
        # Decode: action is a base-2 number, each bit = long(1) or short(0) phase
        for i, iid in enumerate(intersections):
            bit = (action >> i) & 1
            duration = self.PHASE_OPTIONS[2] if bit else self.PHASE_OPTIONS[1]  # 45 or 30
            self.model.environment.set_rl_phase(iid, "NS_GREEN", duration)

    # ── Reward ────────────────────────────────────────────────────────────────
    def _compute_reward(self) -> float:
        snap = self.model._build_snapshot()
        m = snap["metrics"]
        w = self.reward_weights

        # Wait time component
        total_wait = m.get("total_wait_ticks", 0)
        wait_delta = total_wait - self._prev_wait
        self._prev_wait = total_wait
        r_wait = w["wait_time_penalty"] * (wait_delta / max(self.model.num_vehicles, 1))

        # Throughput component
        moving = m.get("vehicles_moving", 0)
        throughput = moving / max(self.model.num_vehicles, 1)
        r_throughput = w["throughput_reward"] * throughput

        # Energy efficiency
        max_load = m.get("max_energy_load_pct", 50) / 100.0
        r_energy = w["energy_efficiency"] * (1.0 - max_load)

        # Anomaly penalty
        r_anomaly = w["anomaly_penalty"] * m.get("active_anomalies", 0)

        return float(r_wait + r_throughput + r_energy + r_anomaly)
