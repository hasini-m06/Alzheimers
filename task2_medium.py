"""
task2_medium.py
───────────────
Task 2 (Medium) — Anomaly Timing Detection

The agent observes a 7-day behavioral time series and must decide
on which day to raise its first alert (action >= 1).

Grading (continuous 0.0–1.0):
  Alert on exact anomaly day          → 1.00
  Alert within ±1 day                 → 0.75
  Alert within ±2 days                → 0.50
  Alert within ±3 days                → 0.25
  Alert outside anomaly window        → 0.00
  No alert raised in 7 days           → 0.00
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.cognitive_env import CogTraceEnv
from env.patient_simulator import PatientConfig, make_medium_patient


TASK_ID    = "task2_medium"
TASK_NAME  = "Anomaly Timing Detection"
DIFFICULTY = "medium"
DESCRIPTION = (
    "Observe a 7-day stream of patient behavioral signals. "
    "Raise an alert (action 1–3) on the day you detect an anomaly. "
    "Scored on how close your first alert is to the true anomaly onset."
)


def _timing_score(alert_day: Optional[int], anomaly_day: int) -> float:
    if alert_day is None:
        return 0.0
    diff = abs(alert_day - anomaly_day)
    if diff == 0: return 1.00
    if diff == 1: return 0.75
    if diff == 2: return 0.50
    if diff == 3: return 0.25
    return 0.00


def run_episode(env: CogTraceEnv, agent_fn) -> Dict[str, Any]:
    """
    Run a 7-step episode and find the first alerting step.

    agent_fn : callable(observation_dict, step: int) -> int
        Returns action 0–3 for each step.
    """
    obs = env.reset()
    first_alert_day: Optional[int] = None
    trajectory = []

    for step in range(env.config.episode_length):
        action = int(agent_fn(obs.model_dump(), step))
        action = max(0, min(3, action))

        next_obs, reward, done, info = env.step(action)

        trajectory.append({
            "step":           step,
            "action":         action,
            "anomaly_active": info.anomaly_active,
            "reward":         reward.value,
        })

        if action > 0 and first_alert_day is None:
            first_alert_day = step

        obs = next_obs
        if done:
            break

    anomaly_day = env._sim.anomaly_day
    score = _timing_score(first_alert_day, anomaly_day)

    return {
        "first_alert_day": first_alert_day,
        "anomaly_day":     anomaly_day,
        "score":           score,
        "trajectory":      trajectory,
    }


def grade(trajectory: List[Dict[str, Any]]) -> float:
    """
    Grade a list of episode result dicts.

    Each dict must contain:
      - "first_alert_day" : int or None
      - "anomaly_day"     : int

    Returns mean timing score in [0.0, 1.0].
    """
    if not trajectory:
        return 0.0
    scores = [_timing_score(r["first_alert_day"], r["anomaly_day"]) for r in trajectory]
    return round(sum(scores) / len(scores), 4)


def build_env(seed: int = 0) -> CogTraceEnv:
    sim = make_medium_patient(seed)
    return CogTraceEnv(config=sim.config)


if __name__ == "__main__":
    import random
    results = []
    for seed in range(20):
        env = build_env(seed)
        result = run_episode(
            env,
            agent_fn=lambda obs, step: random.choices([0, 0, 0, 1], k=1)[0]
        )
        results.append(result)

    final = grade(results)
    print(f"[Task 2] Random baseline score: {final:.4f}")
    print("Expected ~0.15–0.30 (random rarely hits the right day)")
