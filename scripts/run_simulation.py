"""
scripts/run_simulation.py
Run the Urban Digital Twin simulation:
- Generates synthetic data
- Builds road graph
- Runs simulation for N ticks
- Prints performance stats
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import yaml
import json
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("run_simulation")


def main(ticks: int = 1440, save_history: bool = True):
    """
    Run simulation for N ticks and report results.
    Default: 1440 ticks = 1 simulated day.
    """
    logger.info(f"Urban Digital Twin Simulation — {ticks} ticks")

    # Load config
    config_path = Path("config/simulation.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"simulation": {
            "num_vehicles": 200, "num_pedestrians": 500,
            "num_resource_nodes": 10, "synthetic_nodes": 40,
            "default_green_ns": 30,
        }}

    # Initialize city model
    from src.simulation_engine.city_model import CityModel
    from src.anomaly_detection.detectors import AlertManager, ResourceAnomalyDetector
    from src.graph_analysis.graph_metrics import GraphAnalyzer

    logger.info("Initializing CityModel...")
    t0 = time.time()
    model = CityModel(cfg, seed=42)
    alert_mgr = AlertManager()
    res_detector = ResourceAnomalyDetector()
    logger.info(f"Model initialized in {time.time()-t0:.2f}s")

    # Graph analysis
    logger.info("Running graph analysis...")
    analyzer = GraphAnalyzer(model.road_graph)
    graph_report = analyzer.full_report()
    logger.info(f"  Graph stats: {graph_report['graph_stats']}")
    logger.info(f"  Top bottlenecks: {graph_report['top_bottlenecks'][:3]}")

    # Run simulation
    logger.info(f"Running {ticks} ticks...")
    all_metrics = []
    all_alerts = []
    t0 = time.time()

    for tick in range(ticks):
        snap = model.tick_step()
        new_alerts = alert_mgr.process_snapshot(snap, resource_detector=res_detector)
        all_alerts.extend(new_alerts)

        if tick % 240 == 0:  # every 4 hours of sim time
            m = snap["metrics"]
            logger.info(
                f"  {snap['timestamp_sim']} | "
                f"speed={m['avg_vehicle_speed']:.1f} km/h | "
                f"moving={m['vehicles_moving']} | "
                f"energy_max={m['max_energy_load_pct']:.0f}% | "
                f"alerts={len(all_alerts)}"
            )
            all_metrics.append(m)

    elapsed = time.time() - t0
    tps = ticks / elapsed

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SIMULATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Ticks run:       {ticks:,}")
    logger.info(f"  Wall time:       {elapsed:.1f}s")
    logger.info(f"  Ticks/second:    {tps:.0f}")
    logger.info(f"  Total alerts:    {len(all_alerts)}")
    if all_alerts:
        from collections import Counter
        sev = Counter(a["severity"] for a in all_alerts)
        logger.info(f"  Alert breakdown: {dict(sev)}")

    if all_metrics:
        import numpy as np
        avg_speeds = [m["avg_vehicle_speed"] for m in all_metrics]
        logger.info(f"  Avg vehicle speed: {np.mean(avg_speeds):.1f} km/h "
                    f"(min={np.min(avg_speeds):.1f}, max={np.max(avg_speeds):.1f})")

    # Save results
    if save_history:
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        results = {
            "ticks": ticks,
            "elapsed_seconds": round(elapsed, 2),
            "ticks_per_second": round(tps, 1),
            "total_alerts": len(all_alerts),
            "metrics_samples": all_metrics,
            "graph_report": {
                "stats": graph_report["graph_stats"],
                "top_bottlenecks": graph_report["top_bottlenecks"][:5],
            }
        }
        with open("data/processed/simulation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to data/processed/simulation_results.json")

    return {"ticks": ticks, "elapsed": elapsed, "tps": tps,
            "alerts": len(all_alerts)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Urban Digital Twin simulation")
    parser.add_argument("--ticks", type=int, default=1440,
                        help="Number of ticks to run (1440 = 1 day)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    main(ticks=args.ticks, save_history=not args.no_save)
