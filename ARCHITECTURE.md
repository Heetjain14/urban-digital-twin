# 🏗️ System Architecture — Urban Digital Twin AI

> Full technical breakdown of every AI layer, model, and data flow.

---

## System Overview

```
Real World Data (OpenStreetMap, Weather)
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — SIMULATION ENGINE                                │
│  710 autonomous agents · A* pathfinding · Mesa 3            │
│  VehicleAgent · PedestrianAgent · ResourceAgent             │
└───────────────────────┬─────────────────────────────────────┘
                        │ SimulationSnapshot (every tick)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  EVENT BUS (pub/sub)                                        │
│  Decouples simulation from all AI layers                    │
└────┬──────────┬───────────────┬──────────────┬─────────────┘
     │          │               │              │
     ▼          ▼               ▼              ▼
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│ LAYER 2 │ │ LAYER 3  │ │ LAYER 4  │ │   LAYER 5    │
│  LSTM   │ │ XGBoost  │ │   PPO    │ │  Anomaly Det │
│ Traffic │ │ Energy + │ │   RL     │ │  Isolation   │
│Forecast │ │  Water   │ │  Agent   │ │Forest+Zscore │
└────┬────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘
     │           │             │               │
     └─────┬─────┘             │               │
           │              Commands to      Alert Events
           ▼              traffic lights        │
┌─────────────────────────────────────────────────────────────┐
│  LAYER 6 — DASHBOARD + API                                  │
│  Streamlit · FastAPI · WebSocket · Plotly · Folium          │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1 — Simulation Engine

**File:** `src/simulation_engine/city_model.py`

| Component | File | Role |
|---|---|---|
| City Model | `city_model.py` | Orchestrates all agents, publishes snapshots |
| Road Graph | `road_graph.py` | NetworkX DiGraph, A* pathfinding, edge capacity |
| Environment | `environment.py` | Clock, weather Markov chain, traffic light state |
| Event Bus | `event_bus.py` | In-process pub/sub, connects simulation to AI |
| Vehicle Agent | `agents/vehicle_agent.py` | State machine: parked→driving→parked |
| Pedestrian Agent | `agents/pedestrian_agent.py` | Trip generation, crowd density |
| Resource Agent | `agents/resource_agent.py` | Diurnal energy/water demand model |

**Agent state machine (vehicles):**
```
parked_home ──(depart_hour reached)──► driving_to_work
                                               │
                                    (destination reached)
                                               │
                                               ▼
                                         parked_work
                                               │
                                    (return_hour reached)
                                               │
                                               ▼
                                        driving_home
                                               │
                                    (home reached)
                                               │
                                               ▼
                                          parked_home
```

**Speed model (Greenshields):**
```
speed = speed_limit × (1 − 0.8 × congestion²) × weather_multiplier
```

---

## Layer 2 — Traffic LSTM

**File:** `src/ml_models/traffic_forecaster.py`

**Architecture:**
```
Input: (batch, 12, 7)
         │
    LSTM layer 1 (hidden=128, dropout=0.2)
         │
    LSTM layer 2 (hidden=128, dropout=0.2)
         │
    Last timestep output → (batch, 128)
         │
    Linear(128 → 64) + ReLU
         │
    Linear(64 → 1)
         │
Output: predicted vehicle count (scalar)
```

**Input features (7 per timestep):**
```
1. vehicle_count_normalized   (count / max_count)
2. avg_speed_normalized       (speed / 80 km/h)
3. congestion_score           (load / capacity, 0–1)
4. hour_sin                   sin(2π × hour / 24)
5. hour_cos                   cos(2π × hour / 24)
6. day_sin                    sin(2π × day / 7)
7. day_cos                    cos(2π × day / 7)
```

**Why cyclical encoding?**
Without it, 23:59 and 00:00 appear numerically far apart (23 vs 0) but they are only 1 minute apart. Sin/cos encoding wraps around — both values are close on the unit circle.

**Training:**
- Loss: Huber Loss (robust to traffic spike outliers)
- Optimizer: Adam, lr=0.001, cosine LR schedule
- Early stopping: patience=10 epochs
- Split: 80% train / 10% val / 10% test (temporal, never shuffled)

---

## Layer 3 — Energy + Water XGBoost

**File:** `src/ml_models/energy_forecaster.py`

**Why XGBoost instead of another LSTM?**
Energy demand is tabular — the most predictive features are explicit lag values and calendar features, not subtle sequential patterns. XGBoost handles tabular data better than neural nets at this scale and is fully interpretable (feature importance available).

**Input features (15 per sample):**
```
Lag features:    lag_1, lag_2, lag_3, lag_6, lag_12, lag_24
Rolling stats:   rolling_mean_6, rolling_std_6, rolling_mean_24, rolling_max_24
Time encoding:   hour_sin, hour_cos, day_sin, day_cos
Calendar:        is_weekend (0 or 1)
```

**Model config:**
```
n_estimators:    300 trees
max_depth:       6
learning_rate:   0.05
subsample:       0.8 (prevents overfitting)
```

**Results:**
```
Energy: MAE=12.35 kWh    MAPE=3.3%
Water:  MAE=2.10 m³/h   MAPE=5.8%
```

---

## Layer 4 — PPO Reinforcement Learning

**File:** `src/rl_agent/city_env.py`, `src/rl_agent/train_rl.py`

**Problem framing:**
Traffic light control is a sequential decision problem. The agent must decide at every step how long to keep each green phase — too short and cars don't clear the intersection, too long and cross traffic backs up.

**Observation space (20 dimensions):**
```
dims 0–3:   Traffic light timer remaining at 4 intersections (normalized)
dims 4–7:   Current phase (NS_GREEN=0, EW_GREEN=1) at 4 intersections
dims 8–11:  Average vehicle speed in 4 road zones (normalized 0–1)
dims 12–14: Energy load at 3 districts (normalized 0–1)
dim  15:    hour_sin
dim  16:    hour_cos
dim  17:    precipitation (normalized)
dim  18:    temperature (normalized)
dim  19:    active_anomaly_flag (0 or 1)
```

**Action space:**
```
Discrete(16)
Each action encodes green duration for 4 intersections
as a binary number: bit 0=intersection 0, bit 1=intersection 1, etc.
0 = all intersections get 30s green
1 = intersection 0 gets 45s, others get 30s
...
15 = all intersections get 45s green
```

**Reward function:**
```python
reward = (
    -0.6 × (wait_time_delta / num_vehicles)   # penalise waiting
  +  0.3 × (vehicles_moving / total_vehicles) # reward flow
  +  0.1 × (1 - max_energy_load_pct)          # reward efficiency
  +  -5.0 × active_anomalies                  # punish crises
)
```

**Algorithm:** PPO (Proximal Policy Optimization)
- Policy network: MLP [256, 256]
- Trained for: 6,000–200,000 timesteps depending on config
- Same family of algorithms used by OpenAI for robotics and game playing

---

## Layer 5 — Anomaly Detection

**File:** `src/anomaly_detection/detectors.py`

### Isolation Forest (traffic)

**Idea:** Normal data points require many random splits to isolate in a tree. Anomalies (accidents = sudden speed drop + density spike) are isolated in very few splits because they are far from the normal distribution.

```
Training data: normal traffic readings only
contamination: 0.02 (expect 2% anomalies)
n_estimators:  200 trees

Input features: [vehicle_count, rolling_mean_12, rolling_std_12, diff1]
Output:         +1 (normal) or -1 (anomaly)
Confidence:     normalized anomaly score 0–1
```

### Z-Score (energy + water)

**Idea:** Maintain a rolling window of the last 60 readings. If a new value is more than 3 standard deviations from the window mean, it is anomalous. Three consecutive anomalous readings confirm the alert (reduces false positives).

```
window:       60 ticks (1 hour of simulation time)
z_threshold:  3.0 standard deviations
consecutive:  3 readings before alerting

z_score = |value - window_mean| / window_std

Severity mapping:
  z > 3.0:  MEDIUM
  z > 4.5:  HIGH
  z > 6.0:  CRITICAL
```

### Alert Manager (fusion)

Combines signals from both detectors, deduplicates by location and type (30-tick cooldown), assigns final severity, publishes to dashboard.

---

## Data Flow — one complete tick

```
tick 481 (08:01am simulated)
│
├─ environment.update(481)
│    └─ hour_of_day = 8.017 → is_rush_hour = True
│    └─ weather: CLEAR → speed_multiplier = 1.0
│
├─ road_graph.tick_update(481)
│    └─ reopen any roads whose closure expired
│
├─ agents.shuffle_do("step")   ← all 710 agents act in random order
│    ├─ VehicleAgent 1021: state=parked_home, hour=8.017 ≥ depart=8.12 → wait
│    ├─ VehicleAgent 1043: state=driving_to_work → _drive_step()
│    │    ├─ check traffic light at current node → GREEN → proceed
│    │    ├─ compute speed: 50 × (1 - 0.8×0.3²) × 1.0 = 46.4 km/h
│    │    └─ move to next node, update edge loads
│    ├─ PedestrianAgent 2087: walking → move 0.84 grid units toward work
│    └─ ResourceAgent 3: update energy demand = f(hour=8, zone=commercial)
│         = 0.2 + 0.8×exp(-0.5×((8-12)/3.5)²) × 800kWh = 312 kWh
│
├─ _build_snapshot()
│    └─ serialize all 710 agent states + metrics to dict
│
├─ event_bus.publish("simulation.tick", snapshot)
│    ├─ LSTM subscriber: update feature buffer, run inference
│    ├─ XGBoost subscriber: run energy/water prediction
│    ├─ PPO subscriber: observe state, output action → set light phases
│    └─ AnomalyDetector: run Isolation Forest + Z-score checks
│
└─ tick = 482
```

---

## Performance

| Metric | Value |
|---|---|
| Simulation speed | 167 ticks/second |
| 1 simulated day | ~8.6 seconds real time |
| Init time (710 agents) | 0.03 seconds |
| Unit tests | 25 / 25 passing |
| Integration tests | 22 / 22 passing |
| Energy MAPE | 3.3% |
| Water MAPE | 5.8% |

---

## Real-world equivalents

| This project | Real-world equivalent |
|---|---|
| Simulation engine | Siemens Xcelerator Digital Twin platform |
| PPO traffic lights | Singapore SCOOT adaptive traffic system |
| Energy forecasting | UK National Grid ESO demand forecasting |
| Anomaly detection | Mumbai traffic management centre CCTV AI |
| The whole system | Nvidia Omniverse City Digital Twin |
