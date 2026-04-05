"""
task1_easy.py
─────────────
Task 1 (Easy) — Cognitive Stage Classification

The agent sees a SINGLE snapshot of behavioral metrics and must output
the correct Alzheimer's stage (0–4).

Grading:
  Exact match          → 1.00
  Off by 1 stage       → 0.50
  Off by 2 stages      → 0.20
  Off by 3+ stages     → 0.00
"""

from __future__ import annotations
from typing import List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.cognitive_env import CogTraceEnv
from env.patient_simulator import PatientConfig, make_easy_patient


TASK_ID   = "task1_easy"
TASK_NAME = "Cognitive Stage Classification"
DIFFICULTY = "easy"
DESCRIPTION = (
    "Given a single snapshot of a patient's behavioral metrics, "
    "classify their Alzheimer's stage (0=healthy, 1=very mild, "
    "2=mild, 3=moderate, 4=severe)."
)


def _stage_score(predicted: int, true: int) -> float:
    diff = abs(int(predicted) - int(true))
    if diff == 0: return 1.00
    if diff == 1: return 0.50
    if diff == 2: return 0.20
    return 0.00


def run_episode(env: CogTraceEnv, agent_fn) -> Dict[str, Any]:
    """
    Run a single-step episode and collect trajectory.

    agent_fn : callable(observation_dict) -> int
        The agent function. Receives the observation as a plain dict,
        must return an integer 0–4 representing the predicted stage.
        (This is NOT the action space — Task 1 overloads the action
         to mean "stage prediction".)
    """
    obs = env.reset()
    predicted_stage = int(agent_fn(obs.model_dump()))
    predicted_stage = max(0, min(4, predicted_stage))

    true_stage = env._sim.true_stage(0)
    score = _stage_score(predicted_stage, true_stage)

    return {
        "predicted_stage": predicted_stage,
        "true_stage":      true_stage,
        "score":           score,
        "observation":     obs.model_dump(),
    }


def grade(trajectory: List[Dict[str, Any]]) -> float:
    """
    Grade a list of trajectory records (one per patient seed).

    Each record must contain:
      - "predicted_stage" : int (0–4)
      - "true_stage"      : int (0–4)

    Returns mean score across all records, in [0.0, 1.0].
    """
    if not trajectory:
        return 0.0
    scores = [_stage_score(r["predicted_stage"], r["true_stage"]) for r in trajectory]
    return round(sum(scores) / len(scores), 4)


def build_env(seed: int = 0) -> CogTraceEnv:
    """Build a Task-1 environment for a given seed."""
    sim = make_easy_patient(seed)
    cfg = sim.config
    return CogTraceEnv(config=cfg)


if __name__ == "__main__":
    # Quick smoke-test with a random-guess agent
    results = []
    for seed in range(10):
        env = build_env(seed)
        sim = make_easy_patient(seed)
        true_s = sim.true_stage(0)

        import random
        result = run_episode(env, agent_fn=lambda obs: random.randint(0, 4))
        results.append(result)

    final = grade(results)
    print(f"[Task 1] Random baseline score: {final:.4f}")
    print("Expected ~0.25 (random on 5 classes with partial credit)")
