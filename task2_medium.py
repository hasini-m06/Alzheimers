"""
task2_medium.py
───────────────
Task 2 — Anomaly Timing Detection (Medium)

The agent observes a 7-day time series and must identify the first day
it believes an anomaly has occurred by raising any alert (action > 0).

Grading (based on days off from true anomaly onset):
  Exact (0 days off)  → 1.00
  ±1 day              → 0.75
  ±2 days             → 0.50
  ±3 days             → 0.25
  Otherwise           → 0.00
"""

from __future__ import annotations

from typing import Callable, Dict, Any, List, Optional

from cognitive_env import CogTraceEnv
from patient_simulator import PatientConfig


# ── Grading ───────────────────────────────────────────────────────────────────

def _score_timing(true_anomaly_day: int, first_alert_day: Optional[int], episode_length: int) -> float:
    if first_alert_day is None:
        return 0.00  # no alert raised at all
    delta = abs(true_anomaly_day - first_alert_day)
    if delta == 0:
        return 1.00
    elif delta == 1:
        return 0.75
    elif delta == 2:
        return 0.50
    elif delta == 3:
        return 0.25
    else:
        return 0.00


def grade(results: List[Dict[str, Any]]) -> float:
    """Average score across all seeds."""
    if not results:
        return 0.0
    return sum(r["score"] for r in results) / len(results)


# ── Environment builder ───────────────────────────────────────────────────────

def build_env(seed: int = 0) -> CogTraceEnv:
    """Build a Task 2 env: 7-day episode with one anomaly window."""
    stage = (seed % 3) + 1  # stages 1–3
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


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    env: CogTraceEnv,
    agent_fn: Callable[[Dict[str, Any], int], int],
) -> Dict[str, Any]:
    """
    Run one Task 2 episode.

    agent_fn(obs_dict, step) → action (0–3)
    The first step where action > 0 is recorded as the agent's alert day.
    """
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
