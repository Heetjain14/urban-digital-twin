"""
rl_agent/train_rl.py
Train the PPO traffic light control agent using Stable-Baselines3.
Compares RL performance against fixed-cycle baseline.
"""
from __future__ import annotations
import numpy as np
import yaml
import logging
import json
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_rl_config(path: str = "config/rl_config.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "total_timesteps": 50000,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 5,
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "environment": {
                "ticks_per_step": 10,
                "episode_length_ticks": 720,
            },
            "reward_weights": {
                "wait_time_penalty": -0.6,
                "throughput_reward": 0.3,
                "energy_efficiency": 0.1,
                "anomaly_penalty": -5.0,
            }
        }


def train_rl_agent(output_dir: str = "data/processed",
                   timesteps: int = None) -> Dict[str, Any]:
    """
    Train PPO agent on CityTrafficEnv.
    Returns dict with model path and training metrics.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from src.rl_agent.city_env import CityTrafficEnv
    except ImportError as e:
        logger.error(f"RL dependencies missing: {e}")
        return {"error": str(e)}

    cfg = load_rl_config()
    env_cfg = cfg.get("environment", {})
    env_cfg.update({"reward_weights": cfg.get("reward_weights", {})})

    logger.info("Initializing RL environment...")
    env = CityTrafficEnv(config=env_cfg, seed=42)

    # Quick environment check
    try:
        check_env(env, warn=True)
    except Exception as e:
        logger.warning(f"Env check warning (non-fatal): {e}")

    total_ts = timesteps or cfg.get("total_timesteps", 50000)

    logger.info(f"Training PPO for {total_ts:,} timesteps...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=cfg.get("n_steps", 512),
        batch_size=cfg.get("batch_size", 64),
        n_epochs=cfg.get("n_epochs", 5),
        gamma=cfg.get("gamma", 0.99),
        learning_rate=cfg.get("learning_rate", 0.0003),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.01),
        verbose=0,
        seed=42,
    )
    model.learn(total_timesteps=total_ts)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{output_dir}/ppo_traffic_agent"
    model.save(save_path)
    logger.info(f"RL agent saved to {save_path}.zip")

    # Quick evaluation
    eval_result = evaluate_rl_agent(model, env_cfg, n_episodes=3)

    return {
        "model_path": save_path,
        "total_timesteps": total_ts,
        "evaluation": eval_result,
    }


def evaluate_rl_agent(model=None, env_cfg: dict = None,
                      n_episodes: int = 3) -> Dict[str, Any]:
    """
    Evaluate RL agent vs fixed-cycle baseline.
    Returns comparison metrics.
    """
    from src.rl_agent.city_env import CityTrafficEnv

    env_cfg = env_cfg or {"ticks_per_step": 10, "episode_length_ticks": 720}

    def run_episode(use_rl: bool, ep_seed: int) -> dict:
        env = CityTrafficEnv(config=env_cfg, seed=ep_seed)
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            if use_rl and model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = 1   # fixed: action=1 → 30s green for all
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        final_snap = env.model._build_snapshot()
        return {
            "total_reward": round(total_reward, 3),
            "avg_speed": final_snap["metrics"].get("avg_vehicle_speed", 0),
            "total_wait": final_snap["metrics"].get("total_wait_ticks", 0),
            "steps": steps,
        }

    rl_results, baseline_results = [], []
    for ep in range(n_episodes):
        rl_results.append(run_episode(True, 100 + ep))
        baseline_results.append(run_episode(False, 100 + ep))

    rl_avg_reward = np.mean([r["total_reward"] for r in rl_results])
    bl_avg_reward = np.mean([r["total_reward"] for r in baseline_results])
    rl_avg_wait = np.mean([r["total_wait"] for r in rl_results])
    bl_avg_wait = np.mean([r["total_wait"] for r in baseline_results])

    improvement = (bl_avg_wait - rl_avg_wait) / max(bl_avg_wait, 1) * 100

    result = {
        "rl_avg_reward": round(float(rl_avg_reward), 3),
        "baseline_avg_reward": round(float(bl_avg_reward), 3),
        "rl_avg_wait": round(float(rl_avg_wait), 1),
        "baseline_avg_wait": round(float(bl_avg_wait), 1),
        "wait_reduction_pct": round(float(improvement), 1),
    }
    logger.info(f"RL vs Baseline — reward: {rl_avg_reward:.2f} vs {bl_avg_reward:.2f} | "
                f"wait reduction: {improvement:.1f}%")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    result = train_rl_agent(timesteps=30000)
    print(json.dumps(result, indent=2))
