"""
task1_easy.py
─────────────
Task 1 — Cognitive Stage Classification (Easy)

The agent receives a SINGLE observation snapshot and must predict
the patient's CDR stage (0–4).

Grading:
  Exact match  → 1.00
  Off by 1     → 0.50
  Off by 2     → 0.20
  Off by 3+    → 0.00
"""

from __future__ import annotations

from typing import Callable, Dict, Any, List

from cognitive_env import CogTraceEnv
from patient_simulator import PatientConfig, PatientSimulator
from models import Observation


# ── Grading ───────────────────────────────────────────────────────────────────

def _score_prediction(true_stage: int, predicted_stage: int) -> float:
    delta = abs(true_stage - predicted_stage)
    if delta == 0:
        return 1.00
    elif delta == 1:
        return 0.50
    elif delta == 2:
        return 0.20
    else:
        return 0.00


def grade(results: List[Dict[str, Any]]) -> float:
    """Average score across all seeds."""
    if not results:
        return 0.0
    return sum(r["score"] for r in results) / len(results)


# ── Environment builder ───────────────────────────────────────────────────────

def build_env(seed: int = 0) -> CogTraceEnv:
    """Build a Task 1 env: single snapshot, varying stage per seed."""
    true_stage = seed % 5  # cycles through stages 0–4
    cfg = PatientConfig(
        true_stage=true_stage,
        episode_length=1,
        decline_rate=0.0,
        noise_level=0.5,
        anomaly_day=None,
        anomaly_duration=0,
        seed=seed,
        patient_id=f"task1_seed{seed}",
    )
    return CogTraceEnv(config=cfg)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env: CogTraceEnv, agent_fn: Callable[[Dict[str, Any]], int]) -> Dict[str, Any]:
    """
    Run one Task 1 episode.

    agent_fn receives an observation dict and must return a predicted stage (0–4).
    Note: in Task 1 the "action" IS the predicted stage (we reuse the integer).
    """
    obs = env.reset()
    obs_dict = obs.model_dump()

    true_stage = env._sim.true_stage(0)

    # Agent predicts a stage (0–4)
    predicted = agent_fn(obs_dict)
    predicted = max(0, min(4, int(predicted)))

    score = _score_prediction(true_stage, predicted)

    return {
        "true_stage":      true_stage,
        "predicted_stage": predicted,
        "score":           score,
        "observation":     obs_dict,
    }
