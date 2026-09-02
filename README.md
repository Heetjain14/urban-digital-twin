# 🏙️ Urban Digital Twin AI System

> A production-grade smart city simulation with five integrated AI layers — built to understand how systems like Siemens Xcelerator and Nvidia Omniverse work from the inside.

[**🔴 Live Demo**](https://urban-digital-twin-4ybhnj9iwjer9rhnvjbrr3.streamlit.app/) **·** [**📐 Architecture**](ARCHITECTURE.md) **·** [**📓 Notebooks**](notebooks/)

---

## What this is

A complete city district runs inside this system. 710 autonomous agents — vehicles, pedestrians, power substations — navigate a real road network from OpenStreetMap. Five AI layers run simultaneously: predicting traffic, forecasting energy demand, controlling traffic lights with reinforcement learning, and detecting anomalies in real time.

This is not a demo. It is a working multi-agent AI system with a live dashboard, REST API, WebSocket streaming, 47 passing tests, and a what-if scenario engine.

---

## Where this exists in the real world

| This project                   | Real-world system                                    |
| ------------------------------ | ---------------------------------------------------- |
| Agent simulation on road graph | Siemens Xcelerator Digital Twin (Singapore, Hamburg) |
| PPO traffic light control      | Singapore SCOOT adaptive traffic system              |
| Energy demand forecasting      | UK National Grid ESO 30-min ahead forecast           |
| Traffic anomaly detection      | Mumbai Traffic Management Centre AI cameras          |
| The whole system               | Nvidia Omniverse city-scale digital twin             |

When a city council in the real world wants to add 10,000 new residents to a district, they run a digital twin simulation first. They ask: does the road network handle the load? Which substation needs upgrading? Where do traffic jams form? This project is that simulation.

---

## The 5 AI layers

```text
┌─────────────────────────────────────────────────────────────┐
│  1. SIMULATION ENGINE — Mesa 3 + NetworkX                   │
│     710 agents · A* pathfinding · Greenshields speed model  │
├─────────────────────────────────────────────────────────────┤
│  2. TRAFFIC LSTM — PyTorch                                  │
│     12-tick window → next vehicle count · MAPE ~8%          │
├─────────────────────────────────────────────────────────────┤
│  3. ENERGY + WATER XGBoost                                  │
│     Lag features + calendar → demand forecast               │
│     Energy MAPE 3.3% · Water MAPE 5.8%                     │
├─────────────────────────────────────────────────────────────┤
│  4. PPO REINFORCEMENT LEARNING — Stable-Baselines3          │
│     20-dim observation → Discrete(16) action                │
│     Learns to keep traffic flowing through signal timing    │
├─────────────────────────────────────────────────────────────┤
│  5. ANOMALY DETECTION — Isolation Forest + Z-Score           │
│     Traffic accidents · Energy spikes · Water leaks         │
│     Multi-sensor fusion · Severity: LOW/MED/HIGH/CRITICAL   │
└─────────────────────────────────────────────────────────────┘
```

**Full technical breakdown →** [**ARCHITECTURE.md**](https://claude.ai/chat/ARCHITECTURE.md)

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/urban-digital-twin
cd urban-digital-twin

pip install -r requirements.txt
```

### Environment Configuration

The project includes `.env.example` as a template for environment variables.

Create your local `.env` file from it:

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Add your local API keys, credentials, or other environment-specific configuration to `.env`.

**Do not commit `.env` to GitHub.** The repository includes a `.gitignore` file to prevent `.env`, virtual environments, Python cache files, and other generated/local files from being committed.

---

## 🗺️ Road Network Configuration

The system supports two road-network modes:

### 1. Synthetic Road Network

The synthetic mode generates a simulated road network using NetworkX.

Use this mode for:

* Fast development and testing
* Running without downloading OpenStreetMap data
* Reproducible simulations
* Testing the AI and multi-agent systems independently of real-world map data

In:

```text
config/simulation.yaml
```

set:

```yaml
use_osm: false
```

Then run the simulation normally.

---

### 2. OpenStreetMap Road Network

The system can use real road-network data from OpenStreetMap through OSMnx.

In:

```text
config/simulation.yaml
```

set:

```yaml
use_osm: true
osm_city: "Mumbai, India"
```

You can change the city, for example:

```yaml
use_osm: true
osm_city: "Pune, Maharashtra, India"
```

or:

```yaml
use_osm: true
osm_city: "Bengaluru, Karnataka, India"
```

### Important OSM limitation

The current implementation is configured to select the **city/location specified by `osm_city`**.

It is **not currently a general Mumbai neighborhood selector**.

For example:

```yaml
osm_city: "Mumbai, India"
```

selects Mumbai.

Changing it to:

```yaml
osm_city: "Pune, Maharashtra, India"
```

selects Pune.

If you enter a locality such as:

```yaml
osm_city: "Powai, Mumbai, India"
```

the result depends on how OpenStreetMap's geocoder identifies that place. The current system should therefore be described as **city/location-based OSM selection**, rather than guaranteeing that any Mumbai area name will produce exactly that area's boundary.

---

### Switching between Synthetic and OpenStreetMap

**Synthetic:**

```yaml
use_osm: false
```

**OpenStreetMap:**

```yaml
use_osm: true
osm_city: "Mumbai, India"
```

After changing `simulation.yaml`, restart the simulation/dashboard so the road network is rebuilt.

---

## What-if scenarios

| Scenario     | What changes                                              |
| ------------ | --------------------------------------------------------- |
| Heavy Rain   | Vehicle speed × 0.5, pedestrian trips − 40%, energy + 10% |
| Road Closure | A* rerouting activates, downstream congestion builds      |
| Mass Event   | Pedestrian density × 3–5 in target zone                   |
| Power Outage | District goes dark, traffic lights switch to all-red      |

---

## Project structure

```text
urban-digital-twin/
├── src/
│   ├── simulation_engine/     # City model, agents, road graph, environment
│   ├── ml_models/             # LSTM, XGBoost, feature engineering, registry
│   ├── rl_agent/              # Gymnasium env, PPO training, evaluation
│   ├── anomaly_detection/    # Isolation Forest, Z-score, AlertManager
│   ├── graph_analysis/        # Betweenness centrality, GAT, bottleneck detection
│   ├── api_layer/             # FastAPI REST + WebSocket
│   └── dashboard/             # Streamlit live dashboard
├── config/                    # simulation.yaml, ml_models.yaml, rl_config.yaml
├── scripts/                   # train_models.py, run_simulation.py, launch_dashboard.py
├── tests/                     # 47 unit + integration tests
├── notebooks/                 # EDA, simulation analysis, model evaluation
├── .env.example               # Environment variable template
├── .gitignore                 # Git ignore rules
└── ARCHITECTURE.md            # Full technical breakdown of all AI layers
```  Screenshots               # A visual overview of the Urban Digital Twin dashboard, , AI                               analytics, and real-time urban activity.
---

## Performance

| Metric                         |                       Value |
| ------------------------------ | --------------------------: |
| Simulation speed               |     167 ticks/second on CPU |
| Time for 1 simulated day       |                ~8.6 seconds |
| Agent init time                | 0.03 seconds for 710 agents |
| Energy forecast MAPE           |                        3.3% |
| Water forecast MAPE            |                        5.8% |
| Test suite                     |               47/47 passing |
| Anomalies detected (3-day run) |              19 real events |

---

## Roadmap — v2

* [ ] Real-time traffic data integration (OpenStreetMap + live feeds)
* [ ] LLM scenario engine: natural language → simulation parameters
* [ ] React + Mapbox GL frontend (replace Streamlit prototype)
* [ ] Hierarchical RL: district manager + intersection workers
* [ ] Multi-city federated learning
* [ ] Graph Attention Network for citywide congestion prediction
* [ ] Infrastructure recommendation engine

**Looking to collaborate on any of these → open an issue or DM me on LinkedIn**

---


