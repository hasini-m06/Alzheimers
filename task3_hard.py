"""
task3_hard.py
─────────────
Task 3 (Hard) — Full Triage Episode

The agent manages a 30-step episode. Each step it observes new sensor
readings and chooses one of 4 actions. It must balance:

  - Detecting true anomaly events (sensitivity)
  - Avoiding false alerts when the patient is stable (specificity)
  - Using proportionate alert levels (soft before escalate)

Grading uses a composite F1-style score:

  precision  = TP / (TP + FP)  — of all alerts, how many were warranted?
  recall     = TP / (TP + FN)  — of all anomaly steps, how many were caught?
  f1         = 2 * P * R / (P + R)
  level_bonus = partial credit for using appropriate alert levels

  final_score = 0.70 * f1 + 0.30 * level_bonus  ∈ [0.0, 1.0]
"""

from __future__ import annotations
from typing import List, Dict, Any
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.cognitive_env import CogTraceEnv
from env.patient_simulator import PatientConfig, make_hard_patient


TASK_ID    = "task3_hard"
TASK_NAME  = "Full Triage Episode"
DIFFICULTY = "hard"
DESCRIPTION = (
    "Manage a 30-step patient monitoring episode. "
    "Each step, observe behavioral signals and choose an action: "
    "0=do_nothing, 1=soft_alert, 2=medium_alert, 3=escalate. "
    "Scored on F1 (sensitivity + specificity) and alert proportionality."
)

# Mapping: anomaly active → ideal minimum action level
_IDEAL_LEVEL: Dict[bool, int] = {True: 2, False: 0}


def _f1_score(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _level_bonus(trajectory: List[Dict[str, Any]]) -> float:
    """
    Score how well the agent matched alert level to anomaly severity.

    During anomaly: soft alert scores 0.5, medium 0.8, escalate 1.0
    Outside anomaly: do_nothing 1.0, soft 0.5, medium 0.2, escalate 0.0
    """
    scores = []
    for step in trajectory:
        a = step["action"]
        anomaly = step["anomaly_active"]
        if anomaly:
            level_score = {0: 0.0, 1: 0.50, 2: 0.80, 3: 1.00}[a]
        else:
            level_score = {0: 1.00, 1: 0.50, 2: 0.20, 3: 0.00}[a]
        scores.append(level_score)
    return sum(scores) / len(scores) if scores else 0.0


def run_episode(env: CogTraceEnv, agent_fn) -> Dict[str, Any]:
    """
    Run a 30-step episode.

    agent_fn : callable(observation_dict, step: int, history: list) -> int
        Receives current observation, step index, and list of past actions.
        Must return action 0–3.
    """
    obs = env.reset()
    trajectory = []
    action_history = []

    tp = fp = fn = tn = 0

    for step in range(env.config.episode_length):
        action = int(agent_fn(obs.model_dump(), step, list(action_history)))
        action = max(0, min(3, action))

        next_obs, reward, done, info = env.step(action)

        alerted = (action > 0)
        anomaly = info.anomaly_active

        if alerted and anomaly:      tp += 1
        elif alerted and not anomaly: fp += 1
        elif not alerted and anomaly: fn += 1
        else:                         tn += 1

        trajectory.append({
            "step":           step,
            "action":         action,
            "anomaly_active": anomaly,
            "reward":         reward.value,
            "true_stage":     info.true_stage,
        })
        action_history.append(action)
        obs = next_obs
        if done:
            break

    f1 = _f1_score(tp, fp, fn)
    lb = _level_bonus(trajectory)
    final_score = round(0.70 * f1 + 0.30 * lb, 4)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "f1_score":    round(f1, 4),
        "level_bonus": round(lb, 4),
        "score":       final_score,
        "trajectory":  trajectory,
    }


def grade(trajectory: List[Dict[str, Any]]) -> float:
    """
    Grade a list of episode result dicts.

    Each dict must contain "score" in [0.0, 1.0].
    Returns mean score.
    """
    if not trajectory:
        return 0.0
    return round(sum(r["score"] for r in trajectory) / len(trajectory), 4)


def build_env(seed: int = 0) -> CogTraceEnv:
    sim = make_hard_patient(seed)
    return CogTraceEnv(config=sim.config)


if __name__ == "__main__":
    import random
    results = []
    for seed in range(10):
        env = build_env(seed)
        result = run_episode(
            env,
            agent_fn=lambda obs, step, hist: random.randint(0, 3)
        )
        results.append(result)

    final = grade(results)
    print(f"[Task 3] Random baseline score: {final:.4f}")
    print("Expected ~0.25–0.35 (random F1 on imbalanced classes)")
