"""
task3_hard.py
─────────────
Task 3 — Full Triage Episode (Hard)
"""

from __future__ import annotations

from typing import Callable, Dict, Any, List, Optional, Tuple

from cognitive_env import CogTraceEnv
from patient_simulator import PatientConfig


def _compute_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


LEVEL_BONUS_MAP = {3: 1.0, 2: 0.6, 1: 0.3, 0: 0.0}


def grade(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.001
    score = sum(r["score"] for r in results) / len(results)
    return max(0.001, min(0.999, score))


def build_env(seed: int = 0) -> CogTraceEnv:
    stage = (seed % 4) + 1
    cfg = PatientConfig(
        true_stage=stage,
        episode_length=30,
        decline_rate=0.01,
        noise_level=1.2,
        anomaly_day=None,
        anomaly_duration=5,
        seed=seed,
        patient_id=f"task3_seed{seed}",
    )
    return CogTraceEnv(config=cfg)


def run_episode(
    env: CogTraceEnv,
    agent_fn: Callable[[Dict[str, Any], int, List[int]], int],
) -> Dict[str, Any]:
    obs = env.reset()
    anomaly_start = env._sim.anomaly_day
    anomaly_end   = env._sim.anomaly_end

    alert_history: List[int] = []
    step = 0
    tp = fp = fn = 0
    level_bonus_total = 0.0
    anomaly_steps = 0

    while True:
        obs_dict = obs.model_dump()
        action   = agent_fn(obs_dict, step, list(alert_history))
        action   = max(0, min(3, int(action)))

        in_anomaly = anomaly_start <= step < anomaly_end

        if in_anomaly:
            anomaly_steps += 1
            if action > 0:
                tp += 1
                level_bonus_total += LEVEL_BONUS_MAP[action]
            else:
                fn += 1
        else:
            if action > 0:
                fp += 1

        alert_history.append(action)
        obs, reward, done, info = env.step(action)
        step += 1
        if done:
            break

    f1          = _compute_f1(tp, fp, fn)
    level_bonus = (level_bonus_total / anomaly_steps) if anomaly_steps > 0 else 0.0
    score       = max(0.001, min(0.999, round(0.70 * f1 + 0.30 * level_bonus, 4)))

    return {
        "tp":            tp,
        "fp":            fp,
        "fn":            fn,
        "f1_score":      round(f1, 4),
        "level_bonus":   round(level_bonus, 4),
        "score":         score,
        "anomaly_start": anomaly_start,
        "anomaly_end":   anomaly_end,
    }