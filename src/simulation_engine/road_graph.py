"""
road_graph.py

Builds and manages the city road graph.

Supports:
1. Synthetic road generation for MVP/testing
2. Real-world OpenStreetMap road networks using OSMnx

The same CityRoadGraph interface is used by the simulation engine
regardless of which source is selected.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple, List, Dict

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class CityRoadGraph:
    """
    Wraps a NetworkX DiGraph representing the city road network.

    Nodes:
        intersections / road junctions
        x, y coordinates
        zone_type

    Edges:
        road segments
        length_m
        speed_limit_kmh
        lanes
        capacity
        current_load
        is_closed
    """

    def __init__(self, seed: int = 42):

        self.G: nx.DiGraph = nx.DiGraph()

        self.rng = np.random.default_rng(seed)

        # edge -> tick at which it should reopen
        self._closed_edges: Dict[Tuple, int] = {}

        # True when graph was loaded from OpenStreetMap
        self.is_osm_graph = False

    # ======================================================================
    # SYNTHETIC GRAPH
    # ======================================================================

    def build_synthetic(
        self,
        num_nodes: int = 40,
        grid_size: int = 50,
        num_zones: int = 5
    ) -> "CityRoadGraph":

        """
        Build a synthetic road network.

        Used when OSM is disabled or unavailable.
        """

        self.G.clear()
        self.is_osm_graph = False

        zone_names = [
            "residential",
            "commercial",
            "industrial",
            "campus",
            "transit"
        ]

        zone_centers = [
            (10, 40),
            (25, 25),
            (40, 15),
            (15, 15),
            (25, 40)
        ]

        # --------------------------------------------------------------
        # Place nodes
        # --------------------------------------------------------------

        node_positions = {}
        node_zones = {}

        grid_n = num_nodes // 2

        grid_side = max(1, int(np.sqrt(grid_n)))

        xs = np.linspace(
            3,
            grid_size - 3,
            grid_side
        )

        ys = np.linspace(
            3,
            grid_size - 3,
            grid_side
        )

        grid_pts = [
            (x, y)
            for x in xs
            for y in ys
        ][:grid_n]

        extra_pts = [
            (
                self.rng.uniform(3, grid_size - 3),
                self.rng.uniform(3, grid_size - 3)
            )
            for _ in range(num_nodes - grid_n)
        ]

        all_pts = grid_pts + extra_pts

        # --------------------------------------------------------------
        # Add nodes
        # --------------------------------------------------------------

        for i, (x, y) in enumerate(all_pts):

            dists = [
                np.hypot(
                    x - zc[0],
                    y - zc[1]
                )
                for zc in zone_centers
            ]

            zone_idx = int(np.argmin(dists))

            node_positions[i] = (x, y)

            node_zones[i] = zone_names[zone_idx]

            self.G.add_node(
                i,
                x=float(x),
                y=float(y),
                zone_type=zone_names[zone_idx],
                is_intersection=True
            )

        # --------------------------------------------------------------
        # Connect nodes
        # --------------------------------------------------------------

        from scipy.spatial import KDTree

        pts_array = np.array(
            list(node_positions.values())
        )

        tree = KDTree(pts_array)

        k_neighbors = 4

        for i, (x, y) in node_positions.items():

            dists, idxs = tree.query(
                [x, y],
                k=k_neighbors + 1
            )

            for j_idx, dist_val in zip(
                idxs[1:],
                dists[1:]
            ):

                j = int(j_idx)

                if i == j:
                    continue

                length_m = float(dist_val) * 100

                zone = node_zones[i]

                if zone == "industrial":

                    speed_limit = self.rng.choice(
                        [40, 50, 60]
                    )

                    lanes = self.rng.integers(
                        2,
                        5
                    )

                elif zone == "commercial":

                    speed_limit = self.rng.choice(
                        [30, 40, 50]
                    )

                    lanes = self.rng.integers(
                        2,
                        4
                    )

                elif zone == "transit":

                    speed_limit = self.rng.choice(
                        [50, 60, 70]
                    )

                    lanes = self.rng.integers(
                        3,
                        6
                    )

                else:

                    speed_limit = self.rng.choice(
                        [30, 40]
                    )

                    lanes = self.rng.integers(
                        1,
                        3
                    )

                capacity = max(
                    int(lanes * speed_limit / 5),
                    1
                )

                attrs = {
                    "length_m": round(
                        length_m,
                        1
                    ),
                    "speed_limit_kmh": int(
                        speed_limit
                    ),
                    "lanes": int(lanes),
                    "capacity": capacity,
                    "current_load": 0,
                    "is_closed": False,
                }

                self.G.add_edge(
                    i,
                    j,
                    **attrs
                )

                self.G.add_edge(
                    j,
                    i,
                    **attrs
                )

        # --------------------------------------------------------------
        # Identify major intersections
        # --------------------------------------------------------------

        for node in self.G.nodes():

            degree = self.G.degree(node)

            self.G.nodes[node][
                "is_major_intersection"
            ] = degree >= 6

        logger.info(
            "Synthetic road graph: "
            f"{self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges"
        )

        return self

    # ======================================================================
    # OPENSTREETMAP GRAPH
    # ======================================================================

    def build_from_osm(
        self,
        city: str = "Mumbai, India",
        radius_m: int = 1500
    ) -> "CityRoadGraph":

        """
        Download a real road network from OpenStreetMap using OSMnx.

        Parameters
        ----------
        city:
            Location used by OSMnx geocoder.

        radius_m:
            Radius around the city/location to download.

        Returns
        -------
        CityRoadGraph
            The current object containing the OSM road network.
        """

        try:

            import osmnx as ox

        except ImportError:

            raise ImportError(
                "OSMnx is not installed. "
                "Run: pip install osmnx"
            )

        logger.info(
            f"Downloading OpenStreetMap road network "
            f"for {city} within {radius_m}m."
        )

        try:

            # ----------------------------------------------------------
            # Download real driving roads
            # ----------------------------------------------------------

            osm_graph = ox.graph_from_address(
                city,
                dist=radius_m,
                network_type="drive",
                simplify=True
            )

        except Exception as exc:

            logger.exception(
                "Failed to download OpenStreetMap graph."
            )

            raise RuntimeError(
                f"Could not download OSM data for {city}: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Clear existing graph
        # --------------------------------------------------------------

        self.G.clear()

        self.is_osm_graph = True

        # --------------------------------------------------------------
        # Keep geographic coordinates
        #
        # OSM:
        # x = longitude
        # y = latitude
        # --------------------------------------------------------------

        for node, data in osm_graph.nodes(
            data=True
        ):

            longitude = float(
                data.get("x", 0.0)
            )

            latitude = float(
                data.get("y", 0.0)
            )

            self.G.add_node(
                node,
                x=longitude,
                y=latitude,

                # Geographic information
                longitude=longitude,
                latitude=latitude,

                zone_type="urban",

                is_intersection=True,

                # Degree-based classification
                is_major_intersection=False
            )

        # --------------------------------------------------------------
        # Add road edges
        # --------------------------------------------------------------

        for u, v, data in osm_graph.edges(
            data=True
        ):

            # ----------------------------------------------------------
            # Length
            # ----------------------------------------------------------

            length = float(
                data.get(
                    "length",
                    100.0
                )
            )

            # ----------------------------------------------------------
            # Speed limit
            # ----------------------------------------------------------

            speed = data.get(
                "maxspeed",
                40
            )

            if isinstance(speed, list):

                speed = speed[0]

            try:

                speed_text = (
                    str(speed)
                    .replace(
                        "km/h",
                        ""
                    )
                    .replace(
                        "kmh",
                        ""
                    )
                    .strip()
                )

                speed_limit = float(
                    speed_text
                )

            except (
                ValueError,
                TypeError
            ):

                speed_limit = 40.0

            # Keep speed within reasonable range
            speed_limit = max(
                5.0,
                min(
                    speed_limit,
                    130.0
                )
            )

            # ----------------------------------------------------------
            # Number of lanes
            # ----------------------------------------------------------

            lanes = data.get(
                "lanes",
                1
            )

            if isinstance(lanes, list):

                lanes = lanes[0]

            try:

                lanes = int(
                    float(lanes)
                )

            except (
                ValueError,
                TypeError
            ):

                lanes = 1

            lanes = max(
                lanes,
                1
            )

            # ----------------------------------------------------------
            # Capacity
            # ----------------------------------------------------------

            capacity = max(
                int(
                    lanes
                    * speed_limit
                    / 5
                ),
                1
            )

            # ----------------------------------------------------------
            # Store edge
            # ----------------------------------------------------------

            self.G.add_edge(
                u,
                v,

                length_m=length,

                speed_limit_kmh=int(
                    speed_limit
                ),

                lanes=lanes,

                capacity=capacity,

                current_load=0,

                is_closed=False
            )

        # --------------------------------------------------------------
        # Identify major intersections
        # --------------------------------------------------------------

        for node in self.G.nodes():

            degree = self.G.degree(
                node
            )

            self.G.nodes[node][
                "is_major_intersection"
            ] = degree >= 4

        logger.info(
            "OpenStreetMap road graph created: "
            f"{self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges"
        )

        return self

    # ======================================================================
    # PATHFINDING
    # ======================================================================

    def shortest_path(
        self,
        source: int,
        target: int
    ) -> List[int]:

        """
        A* shortest path by travel time.
        """

        if source == target:

            return [source]

        if (
            source not in self.G
            or target not in self.G
        ):

            return []

        try:

            def weight_fn(
                u,
                v,
                d
            ):

                if d.get(
                    "is_closed",
                    False
                ):

                    return 1e9

                speed = max(
                    float(
                        d.get(
                            "speed_limit_kmh",
                            30
                        )
                    ),
                    5.0
                )

                length = float(
                    d.get(
                        "length_m",
                        d.get(
                            "length",
                            50.0
                        )
                    )
                )

                capacity = max(
                    float(
                        d.get(
                            "capacity",
                            1
                        )
                    ),
                    1.0
                )

                load = float(
                    d.get(
                        "current_load",
                        0
                    )
                )

                congestion_penalty = (
                    1.0
                    + 2.0
                    * (
                        load
                        / capacity
                    )
                )

                return (
                    length
                    / (speed / 3.6)
                ) * congestion_penalty

            return nx.astar_path(
                self.G,
                source,
                target,
                weight=weight_fn
            )

        except (
            nx.NetworkXNoPath,
            nx.NodeNotFound
        ):

            return []

    # ======================================================================
    # TRAVEL TIME
    # ======================================================================

    def travel_time_seconds(
        self,
        u: int,
        v: int
    ) -> float:

        """
        Travel time in seconds for one road edge.
        """

        if not self.G.has_edge(
            u,
            v
        ):

            return 1e9

        d = self.G[u][v]

        speed = max(
            float(
                d.get(
                    "speed_limit_kmh",
                    30
                )
            ),
            5.0
        )

        length = float(
            d.get(
                "length_m",
                50.0
            )
        )

        capacity = max(
            float(
                d.get(
                    "capacity",
                    1
                )
            ),
            1.0
        )

        load = float(
            d.get(
                "current_load",
                0
            )
        )

        congestion_penalty = (
            1.0
            + 2.0
            * (
                load
                / capacity
            )
        )

        return (
            length
            / (speed / 3.6)
        ) * congestion_penalty

    # ======================================================================
    # DYNAMIC UPDATES
    # ======================================================================

    def update_edge_load(
        self,
        u: int,
        v: int,
        delta: int
    ):

        """
        Increment/decrement vehicle count on an edge.
        """

        if self.G.has_edge(
            u,
            v
        ):

            cur = self.G[u][v].get(
                "current_load",
                0
            )

            self.G[u][v][
                "current_load"
            ] = max(
                0,
                cur + delta
            )

    def close_edge(
        self,
        u: int,
        v: int,
        duration_ticks: int,
        current_tick: int
    ):

        """
        Temporarily close a road segment.
        """

        if self.G.has_edge(
            u,
            v
        ):

            self.G[u][v][
                "is_closed"
            ] = True

            self._closed_edges[
                (u, v)
            ] = (
                current_tick
                + duration_ticks
            )

            logger.info(
                f"Road closed: "
                f"{u}→{v} for "
                f"{duration_ticks} ticks"
            )

    def tick_update(
        self,
        current_tick: int
    ):

        """
        Re-open edges whose closure has expired.
        """

        to_reopen = [
            edge
            for edge, reopen_tick
            in self._closed_edges.items()
            if reopen_tick <= current_tick
        ]

        for u, v in to_reopen:

            if self.G.has_edge(
                u,
                v
            ):

                self.G[u][v][
                    "is_closed"
                ] = False

            del self._closed_edges[
                (u, v)
            ]

    # ======================================================================
    # GETTERS
    # ======================================================================

    def get_node_positions(
        self
    ) -> Dict[int, Tuple[float, float]]:

        """
        Return node coordinates.

        Synthetic:
            x/y are simulation coordinates.

        OSM:
            x = longitude
            y = latitude.
        """

        return {
            n: (
                d["x"],
                d["y"]
            )
            for n, d
            in self.G.nodes(
                data=True
            )
        }

    def get_intersections(
        self
    ) -> List[int]:

        """
        Return major intersections.
        """

        return [
            n
            for n, d
            in self.G.nodes(
                data=True
            )
            if d.get(
                "is_major_intersection",
                False
            )
        ]

    def get_congestion_map(
        self
    ) -> Dict[Tuple[int, int], float]:

        """
        Return:

            {(u, v): congestion_score}

        where congestion score is between 0 and 1.
        """

        return {

            (u, v):

            min(
                d.get(
                    "current_load",
                    0
                )
                /
                max(
                    d.get(
                        "capacity",
                        1
                    ),
                    1
                ),
                1.0
            )

            for u, v, d
            in self.G.edges(
                data=True
            )
        }

    def random_node(self) -> int:

        """
        Return a random node.
        """

        nodes = list(
            self.G.nodes()
        )

        if not nodes:

            raise RuntimeError(
                "Road graph contains no nodes."
            )

        return self.rng.choice(
            nodes
        )

    # ======================================================================
    # PERSISTENCE
    # ======================================================================

    def save(
        self,
        path: str = "data/processed/road_graph.pkl"
    ):

        Path(
            path
        ).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "wb"
        ) as f:

            pickle.dump(
                self.G,
                f
            )

        logger.info(
            f"Graph saved to {path}"
        )

    def load(
        self,
        path: str = "data/processed/road_graph.pkl"
    ) -> "CityRoadGraph":

        with open(
            path,
            "rb"
        ) as f:

            self.G = pickle.load(
                f
            )

        return self

    # ======================================================================
    # GEOJSON
    # ======================================================================

    def to_geojson(self) -> dict:

        """
        Export road edges as GeoJSON.

        For OSM graphs:
            coordinates are [longitude, latitude].

        For synthetic graphs:
            coordinates are synthetic x/y values.
        """

        features = []

        for u, v, d in self.G.edges(
            data=True
        ):

            ux = self.G.nodes[u]["x"]
            uy = self.G.nodes[u]["y"]

            vx = self.G.nodes[v]["x"]
            vy = self.G.nodes[v]["y"]

            features.append({

                "type": "Feature",

                "geometry": {

                    "type": "LineString",

                    "coordinates": [
                        [ux, uy],
                        [vx, vy]
                    ]
                },

                "properties": {
                    **d,
                    "source": u,
                    "target": v
                }
            })

        return {

            "type": "FeatureCollection",

            "features": features
        }