"""
task2_medium.py
───────────────
Task 2 — Anomaly Timing Detection (Medium)
"""

from __future__ import annotations

from typing import Callable, Dict, Any, List, Optional

from cognitive_env import CogTraceEnv
from patient_simulator import PatientConfig


def _score_timing(true_anomaly_day: int, first_alert_day: Optional[int], episode_length: int) -> float:
    if first_alert_day is None:
        return 0.001
    delta = abs(true_anomaly_day - first_alert_day)
    if delta == 0:
        return 0.999
    elif delta == 1:
        return 0.75
    elif delta == 2:
        return 0.50
    elif delta == 3:
        return 0.25
    else:
        return 0.001


def grade(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.001
    score = sum(r["score"] for r in results) / len(results)
    return max(0.001, min(0.999, score))


def build_env(seed: int = 0) -> CogTraceEnv:
    stage = (seed % 3) + 1
    cfg = PatientConfig(
        true_stage=stage,
        episode_length=7,
        decline_rate=0.0,
        noise_level=1.0,
        anomaly_day=None,
        anomaly_duration=2,
        seed=seed,
        patient_id=f"task2_seed{seed}",
    )
    return CogTraceEnv(config=cfg)


def run_episode(
    env: CogTraceEnv,
    agent_fn: Callable[[Dict[str, Any], int], int],
) -> Dict[str, Any]:
    obs = env.reset()
    true_anomaly_day = env._sim.anomaly_day
    episode_length   = env.config.episode_length
    first_alert_day: Optional[int] = None
    step = 0

    while True:
        obs_dict = obs.model_dump()
        action   = agent_fn(obs_dict, step)
        action   = max(0, min(3, int(action)))
        if action > 0 and first_alert_day is None:
            first_alert_day = step
        obs, reward, done, info = env.step(action)
        step += 1
        if done:
            break

    score = _score_timing(true_anomaly_day, first_alert_day, episode_length)
    return {
        "anomaly_day":     true_anomaly_day,
        "first_alert_day": first_alert_day if first_alert_day is not None else -1,
        "score":           score,
    }