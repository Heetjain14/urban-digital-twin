"""
city_model.py

Mesa 3.x compatible Urban Digital Twin simulation.

Supports:
- Real OpenStreetMap road network through OSMnx
- Synthetic road network fallback
- Vehicle agents
- Pedestrian agents
- Resource agents
- Traffic lights
- Weather/environment
- Dynamic road closures
- Map data for Streamlit dashboard
- Simulation snapshots and metrics
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import List

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import mesa

from src.simulation_engine.road_graph import CityRoadGraph
from src.simulation_engine.environment import CityEnvironment
from src.simulation_engine.event_bus import EventBus, get_bus

from src.simulation_engine.agents.vehicle_agent import VehicleAgent
from src.simulation_engine.agents.pedestrian_agent import PedestrianAgent
from src.simulation_engine.agents.resource_agent import ResourceAgent


logger = logging.getLogger(__name__)


class CityModel(mesa.Model):
    """
    Main Urban Digital Twin simulation model.

    The model manages:

    1. Road network
    2. Vehicles
    3. Pedestrians
    4. Resource nodes
    5. Traffic lights
    6. Weather/environment
    7. Simulation scenarios
    8. Dashboard map data
    """

    # ======================================================================
    # INITIALIZATION
    # ======================================================================

    def __init__(self, config: dict, seed: int = 42):

        # Mesa 3.x
        super().__init__(rng=seed)

        self.config = config

        # Current simulation tick
        self.tick: int = 0

        # ------------------------------------------------------------------
        # Simulation configuration
        # ------------------------------------------------------------------

        sim = config.get("simulation", {})

        self.num_vehicles = sim.get(
            "num_vehicles",
            200
        )

        self.num_pedestrians = sim.get(
            "num_pedestrians",
            500
        )

        self.num_resource_nodes = sim.get(
            "num_resource_nodes",
            10
        )

        # ------------------------------------------------------------------
        # Road graph
        # ------------------------------------------------------------------

        self.road_graph = CityRoadGraph(
            seed=seed
        )

        # Configuration:
        #
        # use_osm: True  -> OpenStreetMap
        # use_osm: False -> synthetic network
        #
        use_osm = sim.get(
            "use_osm",
            True
        )

        self.use_osm = bool(use_osm)

        if self.use_osm:

            # --------------------------------------------------------------
            # REAL OPENSTREETMAP NETWORK
            # --------------------------------------------------------------

            osm_city = sim.get(
                "osm_place",
                "Mumbai, India"
            )

            osm_radius = int(
                sim.get(
                    "osm_radius_m",
                    1500
                )
            )

            logger.info(
                "Loading OpenStreetMap road network: "
                f"{osm_city}, radius={osm_radius}m"
            )

            try:

                self.road_graph.build_from_osm(
                    city=osm_city,
                    radius_m=osm_radius
                )

                logger.info(
                    "OpenStreetMap network loaded successfully."
                )

            except Exception as exc:

                # ----------------------------------------------------------
                # Fallback to synthetic graph
                # ----------------------------------------------------------

                logger.warning(
                    "OpenStreetMap loading failed. "
                    "Falling back to synthetic road network."
                )

                logger.warning(
                    f"OSM error: {exc}"
                )

                self.use_osm = False

                self.road_graph.build_synthetic(
                    num_nodes=sim.get(
                        "synthetic_nodes",
                        40
                    ),
                    grid_size=50
                )

        else:

            # --------------------------------------------------------------
            # SYNTHETIC NETWORK
            # --------------------------------------------------------------

            logger.info(
                "Using synthetic road network."
            )

            self.road_graph.build_synthetic(
                num_nodes=sim.get(
                    "synthetic_nodes",
                    40
                ),
                grid_size=50
            )

        # ------------------------------------------------------------------
        # Environment
        # ------------------------------------------------------------------

        self.environment = CityEnvironment(
            seed=seed
        )

        # Event bus
        self.bus: EventBus = get_bus()

        # ------------------------------------------------------------------
        # Spawn agents
        # ------------------------------------------------------------------

        self._spawn_resource_nodes()

        self._spawn_vehicles()

        self._spawn_pedestrians()

        # ------------------------------------------------------------------
        # Typed agent lists
        # ------------------------------------------------------------------

        self.vehicle_agents: List[VehicleAgent] = list(
            self.agents_by_type.get(
                VehicleAgent,
                []
            )
        )

        self.pedestrian_agents: List[PedestrianAgent] = list(
            self.agents_by_type.get(
                PedestrianAgent,
                []
            )
        )

        self.resource_agents: List[ResourceAgent] = list(
            self.agents_by_type.get(
                ResourceAgent,
                []
            )
        )

        # ------------------------------------------------------------------
        # Traffic lights
        # ------------------------------------------------------------------

        self.intersection_ids = (
            self.road_graph.get_intersections()[:12]
        )

        self.environment.initialize_traffic_lights(
            self.intersection_ids,
            default_green_ticks=sim.get(
                "default_green_ns",
                30
            )
        )

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------

        logger.info(
            "CityModel ready — "
            f"{self.num_vehicles} vehicles / "
            f"{self.num_pedestrians} pedestrians / "
            f"{self.num_resource_nodes} resources | "
            f"{self.road_graph.G.number_of_nodes()} nodes, "
            f"{self.road_graph.G.number_of_edges()} edges | "
            f"OSM={self.use_osm}"
        )

    # ======================================================================
    # RESOURCE NODES
    # ======================================================================

    def _spawn_resource_nodes(self):

        """
        Create resource nodes representing city zones.

        For synthetic maps, resource positions use the
        existing 0-50 simulation coordinate system.

        For OSM maps, resource positions are placed around
        actual road-network coordinates.
        """

        zone_cfgs = [
            (
                "residential",
                10,
                40,
                500,
                80
            ),
            (
                "commercial",
                25,
                25,
                800,
                120
            ),
            (
                "industrial",
                40,
                15,
                1200,
                200
            ),
            (
                "campus",
                15,
                15,
                400,
                60
            ),
            (
                "transit",
                25,
                40,
                300,
                40
            ),
        ]

        # ------------------------------------------------------------------
        # OSM coordinates
        # ------------------------------------------------------------------

        if self.use_osm:

            nodes = list(
                self.road_graph.G.nodes()
            )

            if not nodes:
                logger.warning(
                    "OSM graph contains no nodes."
                )
                return

            for i in range(
                self.num_resource_nodes
            ):

                z, _, _, ec, wc = zone_cfgs[
                    i % len(zone_cfgs)
                ]

                node_id = nodes[
                    i % len(nodes)
                ]

                node = self.road_graph.G.nodes[
                    node_id
                ]

                x = float(
                    node.get(
                        "x",
                        0.0
                    )
                )

                y = float(
                    node.get(
                        "y",
                        0.0
                    )
                )

                ResourceAgent(
                    self,
                    i % len(zone_cfgs),
                    z,
                    x,
                    y,
                    ec,
                    wc
                )

            return

        # ------------------------------------------------------------------
        # Synthetic coordinates
        # ------------------------------------------------------------------

        for i in range(
            self.num_resource_nodes
        ):

            z, bx, by, ec, wc = zone_cfgs[
                i % len(zone_cfgs)
            ]

            x = float(
                np.clip(
                    bx + self.rng.uniform(
                        -5,
                        5
                    ),
                    1,
                    48
                )
            )

            y = float(
                np.clip(
                    by + self.rng.uniform(
                        -5,
                        5
                    ),
                    1,
                    48
                )
            )

            ResourceAgent(
                self,
                i % len(zone_cfgs),
                z,
                x,
                y,
                ec,
                wc
            )

    # ======================================================================
    # VEHICLES
    # ======================================================================

    def _spawn_vehicles(self):

        """
        Create vehicle agents.

        Vehicles select random source and destination
        nodes from the road graph.
        """

        nodes = list(
            self.road_graph.G.nodes()
        )

        if len(nodes) < 2:

            logger.warning(
                "Not enough road nodes to spawn vehicles."
            )

            return

        for _ in range(
            self.num_vehicles
        ):

            # Random home node
            home = int(
                self.rng.choice(
                    nodes
                )
            )

            # Different work node
            possible_work_nodes = [
                n
                for n in nodes
                if n != home
            ]

            work = int(
                self.rng.choice(
                    possible_work_nodes
                    if possible_work_nodes
                    else nodes
                )
            )

            VehicleAgent(
                self,
                home,
                work,
                float(
                    self.rng.normal(
                        8.0,
                        0.75
                    )
                ),
                float(
                    self.rng.normal(
                        17.5,
                        0.75
                    )
                )
            )

    # ======================================================================
    # PEDESTRIANS
    # ======================================================================

    def _spawn_pedestrians(self):

        """
        Create pedestrian agents.

        The existing pedestrian agent interface uses
        x/y coordinates directly.
        """

        # ------------------------------------------------------------------
        # Synthetic map
        # ------------------------------------------------------------------

        if not self.use_osm:

            for _ in range(
                self.num_pedestrians
            ):

                PedestrianAgent(
                    self,
                    float(
                        self.rng.uniform(
                            5,
                            45
                        )
                    ),
                    float(
                        self.rng.uniform(
                            5,
                            45
                        )
                    ),
                    float(
                        self.rng.uniform(
                            5,
                            45
                        )
                    ),
                    float(
                        self.rng.uniform(
                            5,
                            45
                        )
                    ),
                    float(
                        self.rng.normal(
                            8.0,
                            1.0
                        )
                    )
                )

            return

        # ------------------------------------------------------------------
        # OSM map
        #
        # Use actual geographic coordinates from the road graph.
        # ------------------------------------------------------------------

        nodes = list(
            self.road_graph.G.nodes()
        )

        if not nodes:

            return

        for _ in range(
            self.num_pedestrians
        ):

            start_node = self.rng.choice(
                nodes
            )

            end_node = self.rng.choice(
                nodes
            )

            start_data = (
                self.road_graph.G.nodes[
                    start_node
                ]
            )

            end_data = (
                self.road_graph.G.nodes[
                    end_node
                ]
            )

            PedestrianAgent(
                self,
                float(
                    start_data.get(
                        "x",
                        0.0
                    )
                ),
                float(
                    start_data.get(
                        "y",
                        0.0
                    )
                ),
                float(
                    end_data.get(
                        "x",
                        0.0
                    )
                ),
                float(
                    end_data.get(
                        "y",
                        0.0
                    )
                ),
                float(
                    self.rng.normal(
                        8.0,
                        1.0
                    )
                )
            )

    # ======================================================================
    # SIMULATION STEP
    # ======================================================================

    def tick_step(self):

        # --------------------------------------------------------------
        # Update environment
        # --------------------------------------------------------------

        self.environment.update(
            self.tick
        )

        # --------------------------------------------------------------
        # Update road closures
        # --------------------------------------------------------------

        self.road_graph.tick_update(
            self.tick
        )

        # --------------------------------------------------------------
        # Execute Mesa agents
        # --------------------------------------------------------------

        self.agents.shuffle_do(
            "step"
        )

        # --------------------------------------------------------------
        # Build dashboard snapshot
        # --------------------------------------------------------------

        snapshot = self._build_snapshot()

        # --------------------------------------------------------------
        # Publish simulation event
        # --------------------------------------------------------------

        self.bus.publish(
            "simulation.tick",
            snapshot
        )

        # Increment tick
        self.tick += 1

        return snapshot

    # ======================================================================
    # RUN SIMULATION
    # ======================================================================

    def run(
        self,
        ticks: int,
        verbose_every: int = 1440
    ):

        logger.info(
            f"Starting simulation: {ticks} ticks"
        )

        t0 = time.time()

        summaries = []

        for t in range(
            ticks
        ):

            snap = self.tick_step()

            if (
                t % verbose_every == 0
                and t > 0
            ):

                elapsed = (
                    time.time()
                    - t0
                )

                m = snap[
                    "metrics"
                ]

                logger.info(
                    f"Tick {t:>6} | "
                    f"{self.environment.sim_datetime_str} | "
                    f"speed="
                    f"{m.get('avg_vehicle_speed', 0):.1f} km/h | "
                    f"moving="
                    f"{m.get('vehicles_moving', 0)} | "
                    f"elapsed="
                    f"{elapsed:.1f}s"
                )

                summaries.append(
                    m
                )

        elapsed = (
            time.time()
            - t0
        )

        ticks_per_second = (
            ticks / elapsed
            if elapsed > 0
            else 0
        )

        logger.info(
            f"Done: {ticks} ticks in "
            f"{elapsed:.1f}s "
            f"({ticks_per_second:.0f} t/s)"
        )

        return summaries

    # ======================================================================
    # MAP DATA
    # ======================================================================

    def get_map_data(self) -> dict:

        """
        Return map information for the Streamlit dashboard.

        Contains:

        - GeoJSON road network
        - Vehicles
        - Intersections
        - Map type
        - Geographic bounds

        For OSM:

            x = longitude
            y = latitude

        For synthetic:

            x/y are simulation coordinates.
        """

        # ------------------------------------------------------------------
        # Roads
        # ------------------------------------------------------------------

        roads = self.road_graph.to_geojson()

        # ------------------------------------------------------------------
        # Vehicles
        # ------------------------------------------------------------------

        vehicles = []

        current_vehicles = list(
            self.agents_by_type.get(
                VehicleAgent,
                []
            )
        )

        for vehicle in current_vehicles:

            if not hasattr(
                vehicle,
                "get_record"
            ):
                continue

            try:

                record = vehicle.get_record()

                # Make sure map coordinates exist
                if (
                    "x" in record
                    and "y" in record
                ):

                    vehicles.append(
                        record
                    )

            except Exception as exc:

                logger.debug(
                    f"Could not read vehicle record: {exc}"
                )

        # ------------------------------------------------------------------
        # Pedestrians
        # ------------------------------------------------------------------

        pedestrians = []

        current_pedestrians = list(
            self.agents_by_type.get(
                PedestrianAgent,
                []
            )
        )

        for pedestrian in current_pedestrians:

            if not hasattr(
                pedestrian,
                "get_record"
            ):
                continue

            try:

                record = pedestrian.get_record()

                if (
                    "x" in record
                    and "y" in record
                ):

                    pedestrians.append(
                        record
                    )

            except Exception as exc:

                logger.debug(
                    f"Could not read pedestrian record: {exc}"
                )

        # ------------------------------------------------------------------
        # Intersections
        # ------------------------------------------------------------------

        intersections = []

        G = self.road_graph.G

        for iid in self.intersection_ids:

            if not G.has_node(
                iid
            ):
                continue

            node = G.nodes[
                iid
            ]

            intersections.append(
                {
                    "intersection_id": iid,

                    "x": float(
                        node.get(
                            "x",
                            0.0
                        )
                    ),

                    "y": float(
                        node.get(
                            "y",
                            0.0
                        )
                    ),

                    "phase":
                        self.environment
                        .traffic_light_phases
                        .get(
                            iid,
                            "NS_GREEN"
                        ),

                    "phase_remaining_ticks":
                        self.environment
                        .traffic_light_timers
                        .get(
                            iid,
                            0
                        ),

                    "is_major_intersection":
                        bool(
                            node.get(
                                "is_major_intersection",
                                False
                            )
                        )
                }
            )

        # ------------------------------------------------------------------
        # Resource nodes
        # ------------------------------------------------------------------

        resources = []

        current_resources = list(
            self.agents_by_type.get(
                ResourceAgent,
                []
            )
        )

        for resource in current_resources:

            if not hasattr(
                resource,
                "get_record"
            ):
                continue

            try:

                resources.append(
                    resource.get_record()
                )

            except Exception as exc:

                logger.debug(
                    f"Could not read resource record: {exc}"
                )

        # ------------------------------------------------------------------
        # Map bounds
        # ------------------------------------------------------------------

        map_bounds = self._get_map_bounds()

        # ------------------------------------------------------------------
        # Final map payload
        # ------------------------------------------------------------------

        return {

            "roads": roads,

            "vehicles": vehicles,

            "pedestrians": pedestrians,

            "resources": resources,

            "intersections": intersections,

            "map_type":
                "osm"
                if self.use_osm
                else "synthetic",

            "map_bounds":
                map_bounds,
        }

    # ======================================================================
    # MAP BOUNDS
    # ======================================================================

    def _get_map_bounds(self) -> dict:

        """
        Calculate map coordinate bounds.

        Used by Streamlit/Plotly/Folium maps.
        """

        G = self.road_graph.G

        if G.number_of_nodes() == 0:

            return {
                "min_x": 0,
                "max_x": 50,
                "min_y": 0,
                "max_y": 50,
            }

        xs = []

        ys = []

        for _, data in G.nodes(
            data=True
        ):

            try:

                xs.append(
                    float(
                        data.get(
                            "x",
                            0.0
                        )
                    )
                )

                ys.append(
                    float(
                        data.get(
                            "y",
                            0.0
                        )
                    )
                )

            except (
                ValueError,
                TypeError
            ):

                continue

        if not xs or not ys:

            return {
                "min_x": 0,
                "max_x": 50,
                "min_y": 0,
                "max_y": 50,
            }

        return {

            "min_x":
                min(xs),

            "max_x":
                max(xs),

            "min_y":
                min(ys),

            "max_y":
                max(ys),
        }

    # ======================================================================
    # SNAPSHOT
    # ======================================================================

    def _build_snapshot(self) -> dict:

        # ------------------------------------------------------------------
        # Re-fetch agents
        # ------------------------------------------------------------------

        vehicles = list(
            self.agents_by_type.get(
                VehicleAgent,
                []
            )
        )

        pedestrians = list(
            self.agents_by_type.get(
                PedestrianAgent,
                []
            )
        )

        resources = list(
            self.agents_by_type.get(
                ResourceAgent,
                []
            )
        )

        # ------------------------------------------------------------------
        # Agent records
        # ------------------------------------------------------------------

        all_agents = (
            [
                a.get_record()
                for a in vehicles
            ]
            +
            [
                a.get_record()
                for a in pedestrians
            ]
        )

        # Limit dashboard payload
        if len(all_agents) > 300:

            idxs = self.rng.choice(
                len(all_agents),
                300,
                replace=False
            )

            all_agents = [
                all_agents[i]
                for i in idxs
            ]

        # ------------------------------------------------------------------
        # Intersections
        # ------------------------------------------------------------------

        intersections = []

        G = self.road_graph.G

        for iid in self.intersection_ids:

            if not G.has_node(
                iid
            ):
                continue

            nd = G.nodes[
                iid
            ]

            intersections.append(
                {
                    "intersection_id":
                        iid,

                    "x":
                        float(
                            nd.get(
                                "x",
                                0.0
                            )
                        ),

                    "y":
                        float(
                            nd.get(
                                "y",
                                0.0
                            )
                        ),

                    "phase":
                        self.environment
                        .traffic_light_phases
                        .get(
                            iid,
                            "NS_GREEN"
                        ),

                    "phase_remaining_ticks":
                        self.environment
                        .traffic_light_timers
                        .get(
                            iid,
                            0
                        ),

                    "queue_ns":
                        int(
                            self.rng.integers(
                                0,
                                8
                            )
                        ),

                    "queue_ew":
                        int(
                            self.rng.integers(
                                0,
                                8
                            )
                        ),
                }
            )

        # ------------------------------------------------------------------
        # Vehicle metrics
        # ------------------------------------------------------------------

        moving = [
            v
            for v in vehicles
            if v.state.startswith(
                "driving"
            )
        ]

        speeds = [
            v.speed_kmh
            for v in moving
            if v.speed_kmh > 0
        ]

        avg_speed = (
            float(
                np.mean(
                    speeds
                )
            )
            if speeds
            else 0.0
        )

        # ------------------------------------------------------------------
        # Resource metrics
        # ------------------------------------------------------------------

        total_energy = sum(
            r.energy_demand
            for r in resources
        )

        total_water = sum(
            r.water_demand
            for r in resources
        )

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------

        metrics = {

            "tick":
                self.tick,

            "hour_of_day":
                round(
                    self.environment.hour_of_day,
                    2
                ),

            "vehicles_moving":
                len(moving),

            "vehicles_parked":
                sum(
                    1
                    for v in vehicles
                    if "parked"
                    in v.state
                ),

            "avg_vehicle_speed":
                round(
                    avg_speed,
                    2
                ),

            "total_wait_ticks":
                sum(
                    v.wait_ticks
                    for v in vehicles
                ),

            "total_energy_demand_kwh":
                round(
                    total_energy,
                    1
                ),

            "total_water_demand_m3h":
                round(
                    total_water,
                    1
                ),

            "max_energy_load_pct":
                round(
                    max(
                        (
                            r.energy_load_pct
                            for r in resources
                        ),
                        default=0
                    ),
                    1
                ),

            "max_water_load_pct":
                round(
                    max(
                        (
                            r.water_load_pct
                            for r in resources
                        ),
                        default=0
                    ),
                    1
                ),

            "active_anomalies":
                sum(
                    1
                    for r in resources
                    if r.anomaly_flag
                ),
        }

        # ------------------------------------------------------------------
        # Final snapshot
        # ------------------------------------------------------------------

        return {

            "tick":
                self.tick,

            "timestamp_sim":
                self.environment.sim_datetime_str,

            "agents":
                all_agents,

            "resources":
                [
                    r.get_record()
                    for r in resources
                ],

            "intersections":
                intersections,

            "weather":
                self.environment
                .get_weather_state()
                .model_dump(),

            "alerts":
                [],

            "metrics":
                metrics,

            # Extra information for dashboard
            "map_type":
                "osm"
                if self.use_osm
                else "synthetic",

            "map_bounds":
                self._get_map_bounds(),
        }

    # ======================================================================
    # SCENARIOS
    # ======================================================================

    def apply_scenario(
        self,
        stype: str,
        params: dict
    ):

        """
        Apply a simulation scenario.
        """

        self.environment.apply_scenario(
            stype,
            params
        )

    def clear_scenario(self):

        """
        Clear the active scenario.
        """

        self.environment.clear_scenario()