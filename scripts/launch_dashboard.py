"""
scripts/launch_dashboard.py
One-command launcher for the Urban Digital Twin dashboard.
Checks dependencies, trains models if needed, then opens Streamlit.
"""
from __future__ import annotations
import sys
import os
import subprocess
import time
from pathlib import Path

# Make sure we're running from project root
os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, ".")


def check_or_train_models():
    """Train all models if not already present."""
    required = [
        "data/processed/traffic_lstm.pt",
        "data/processed/energy_xgb.pkl",
        "data/processed/water_xgb.pkl",
        "data/processed/anomaly_traffic.pkl",
    ]
    missing = [p for p in required if not Path(p).exists()]

    if missing:
        print(f"Missing trained models: {[Path(p).name for p in missing]}")
        print("Running training pipeline (takes ~2 minutes)...")
        result = subprocess.run(
            [sys.executable, "-W", "ignore", "scripts/train_models.py"],
            check=False
        )
        if result.returncode != 0:
            print("Warning: Training had errors — dashboard will still run with available models.")
    else:
        print("✅ All trained models found.")


def generate_data_if_needed():
    """Generate synthetic data if not present."""
    if not Path("data/synthetic/traffic.parquet").exists():
        print("Generating synthetic city data (14 days)...")
        import warnings
        warnings.filterwarnings("ignore")
        from src.data_ingestion.synthetic_generator import SyntheticCityGenerator
        gen = SyntheticCityGenerator(days=14, seed=42)
        gen.generate_all("data/synthetic")
        print("✅ Synthetic data generated.")
    else:
        print("✅ Synthetic data found.")


def run_quick_smoke_test():
    """30-second smoke test before launching dashboard."""
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from src.simulation_engine.city_model import CityModel
        cfg = {"simulation": {"num_vehicles": 10, "num_pedestrians": 20,
                               "num_resource_nodes": 3, "synthetic_nodes": 12,
                               "default_green_ns": 30}}
        m = CityModel(cfg, seed=42)
        for _ in range(5):
            m.tick_step()
        print("✅ Simulation engine OK.")
    except Exception as e:
        print(f"⚠️  Simulation warning: {e}")


def launch_dashboard():
    """Start the Streamlit dashboard."""
    print("\n" + "="*55)
    print("  🏙️  URBAN DIGITAL TWIN AI SYSTEM")
    print("  Smart City AI Research Dashboard")
    print("="*55)
    print("\n  Dashboard starting at: http://localhost:8501")
    print("  Press Ctrl+C to stop\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/app.py",
        "--server.port=8501",
        "--server.address=localhost",
        "--browser.gatherUsageStats=false",
        "--logger.level=warning",
    ])


def main():
    print("\n🏙️  Urban Digital Twin — Launch Sequence")
    print("-" * 45)

    # Step 1: Data
    generate_data_if_needed()

    # Step 2: Models
    check_or_train_models()

    # Step 3: Smoke test
    run_quick_smoke_test()

    # Step 4: Dashboard
    launch_dashboard()


if __name__ == "__main__":
    main()
