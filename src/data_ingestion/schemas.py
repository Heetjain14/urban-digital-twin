"""
Pydantic schemas — shared data contracts across all modules.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class WeatherCondition(str, Enum):
    CLEAR = "clear"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    HOT = "hot"


class ZoneType(str, Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    CAMPUS = "campus"
    TRANSIT = "transit"


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AgentRecord(BaseModel):
    agent_id: int
    agent_type: str          # "vehicle" | "pedestrian"
    x: float
    y: float
    speed: float             # m/s or km/h equivalent
    state: str               # "moving" | "waiting" | "parked"
    destination_x: Optional[float] = None
    destination_y: Optional[float] = None


class ResourceState(BaseModel):
    node_id: int
    zone_type: ZoneType
    x: float
    y: float
    energy_demand: float     # kWh current demand
    energy_capacity: float   # kWh max capacity
    water_demand: float      # m³/h current demand
    water_capacity: float    # m³/h max capacity
    energy_load_pct: float   # 0–100
    water_load_pct: float    # 0–100
    anomaly_flag: bool = False


class IntersectionState(BaseModel):
    intersection_id: int
    x: float
    y: float
    phase: str               # "NS_GREEN" | "EW_GREEN" | "ALL_RED"
    phase_remaining_ticks: int
    queue_ns: int            # vehicles queued north-south
    queue_ew: int            # vehicles queued east-west


class WeatherState(BaseModel):
    condition: WeatherCondition
    temperature_c: float
    precipitation_mm_h: float
    speed_multiplier: float  # effect on vehicle speed
    energy_multiplier: float # effect on energy demand


class AlertEvent(BaseModel):
    alert_id: str
    tick: int
    alert_type: str          # "traffic_accident" | "energy_spike" | "water_anomaly" | "crowd_surge"
    severity: AlertSeverity
    location_x: float
    location_y: float
    description: str
    confidence: float        # 0–1
    resolved: bool = False


class SimulationSnapshot(BaseModel):
    tick: int
    timestamp_sim: str       # simulated datetime string
    agents: List[AgentRecord]
    resources: List[ResourceState]
    intersections: List[IntersectionState]
    weather: WeatherState
    alerts: List[AlertEvent]
    metrics: Dict[str, float]  # avg_speed, total_vehicles_moving, etc.


class ScenarioConfig(BaseModel):
    scenario_type: str       # "rain" | "road_closure" | "mass_event" | "power_outage"
    start_tick: int = 0
    duration_ticks: int = 120
    parameters: Dict[str, Any] = Field(default_factory=dict)
