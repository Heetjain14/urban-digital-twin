"""
graph_analysis/graph_metrics.py
Road graph intelligence: bottleneck detection, centrality, flow analysis.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """
    Performs structural and dynamic analysis on the city road graph.
    Used for both static reports and live congestion intelligence.
    """

    def __init__(self, road_graph):
        self.rg = road_graph  # CityRoadGraph instance
        self.G = road_graph.G
        self._cached_centrality: Dict[int, float] = {}
        self._cached_betweenness: Dict[int, float] = {}

    # ── Static structural analysis ────────────────────────────────────────────
    def compute_betweenness_centrality(self, normalized: bool = True) -> Dict[int, float]:
        """
        Betweenness centrality: nodes on many shortest paths = critical infrastructure.
        High-betweenness nodes are bottlenecks.
        """
        logger.info("Computing betweenness centrality (may take a moment)...")
        bc = nx.betweenness_centrality(self.G, normalized=normalized, weight="length_m")
        self._cached_betweenness = bc
        logger.info(f"Top-5 bottlenecks: {sorted(bc.items(), key=lambda x: -x[1])[:5]}")
        return bc

    def get_top_bottlenecks(self, k: int = 5) -> List[Tuple[int, float]]:
        """Return k nodes with highest betweenness centrality."""
        if not self._cached_betweenness:
            self.compute_betweenness_centrality()
        return sorted(self._cached_betweenness.items(), key=lambda x: -x[1])[:k]

    def compute_degree_stats(self) -> Dict[str, float]:
        degrees = [d for _, d in self.G.degree()]
        return {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "avg_degree": float(np.mean(degrees)),
            "max_degree": int(np.max(degrees)),
            "density": nx.density(self.G),
            "is_strongly_connected": nx.is_strongly_connected(self.G),
        }

    def find_critical_edges(self, k: int = 10) -> List[Tuple]:
        """Edges whose removal most increases total travel time (edge betweenness)."""
        ebc = nx.edge_betweenness_centrality(self.G, weight="length_m")
        return sorted(ebc.items(), key=lambda x: -x[1])[:k]

    # ── Dynamic congestion analysis ───────────────────────────────────────────
    def get_congestion_heatmap(self) -> List[Dict]:
        """
        Returns per-node congestion score (0–1) based on
        sum of loads on incoming/outgoing edges.
        """
        heatmap = []
        for node, data in self.G.nodes(data=True):
            in_load = sum(self.G[u][node].get("current_load", 0)
                          for u in self.G.predecessors(node))
            out_cap = sum(self.G[node][v].get("capacity", 1)
                          for v in self.G.successors(node))
            congestion = min(in_load / max(out_cap, 1), 1.0)
            heatmap.append({
                "node": node,
                "x": data["x"],
                "y": data["y"],
                "congestion": round(congestion, 3),
                "zone_type": data.get("zone_type", "unknown"),
            })
        return heatmap

    def identify_congested_corridors(self, threshold: float = 0.7) -> List[Dict]:
        """Find road segments with load > threshold * capacity."""
        corridors = []
        for u, v, data in self.G.edges(data=True):
            load = data.get("current_load", 0)
            cap = max(data.get("capacity", 1), 1)
            if load / cap > threshold:
                corridors.append({
                    "from": u, "to": v,
                    "load": load, "capacity": cap,
                    "utilization": round(load / cap, 3),
                    "x0": self.G.nodes[u]["x"], "y0": self.G.nodes[u]["y"],
                    "x1": self.G.nodes[v]["x"], "y1": self.G.nodes[v]["y"],
                })
        return sorted(corridors, key=lambda x: -x["utilization"])

    # ── Infrastructure recommendations ────────────────────────────────────────
    def recommend_improvements(self) -> List[Dict]:
        """
        Analyze graph and suggest infrastructure improvements:
        - High-betweenness nodes → candidate for bypass routes
        - High-load edges → candidate for lane expansion
        - Poorly connected zones → candidate for new connections
        """
        recommendations = []
        bottlenecks = self.get_top_bottlenecks(k=3)
        for node, score in bottlenecks:
            data = self.G.nodes[node]
            recommendations.append({
                "type": "bypass_route",
                "priority": "HIGH",
                "location": (data["x"], data["y"]),
                "reason": f"Node {node} has betweenness centrality={score:.3f}. "
                          f"Failure here disrupts {score*100:.0f}% of shortest paths.",
                "suggestion": "Build parallel route to reduce single-point dependency."
            })

        corridors = self.identify_congested_corridors(threshold=0.8)
        for c in corridors[:3]:
            recommendations.append({
                "type": "lane_expansion",
                "priority": "MEDIUM",
                "location": ((c["x0"]+c["x1"])/2, (c["y0"]+c["y1"])/2),
                "reason": f"Edge {c['from']}→{c['to']} running at "
                          f"{c['utilization']*100:.0f}% capacity.",
                "suggestion": "Add 1–2 lanes or implement congestion pricing."
            })

        return recommendations

    def full_report(self) -> Dict[str, Any]:
        """Generate a comprehensive graph analysis report."""
        stats = self.compute_degree_stats()
        bottlenecks = self.get_top_bottlenecks()
        corridors = self.identify_congested_corridors()
        recs = self.recommend_improvements()

        return {
            "graph_stats": stats,
            "top_bottlenecks": [{"node": n, "score": s} for n, s in bottlenecks],
            "congested_corridors": corridors[:10],
            "recommendations": recs,
        }
