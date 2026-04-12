"""
task1_easy.py
─────────────
Task 1 — Cognitive Stage Classification (Easy)
"""

from __future__ import annotations

from typing import Callable, Dict, Any, List

from cognitive_env import CogTraceEnv
from patient_simulator import PatientConfig, PatientSimulator
from models import Observation


def _score_prediction(true_stage: int, predicted_stage: int) -> float:
    delta = abs(true_stage - predicted_stage)
    if delta == 0:
        return 0.999
    elif delta == 1:
        return 0.50
    elif delta == 2:
        return 0.20
    else:
        return 0.001


def grade(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.001
    score = sum(r["score"] for r in results) / len(results)
    return max(0.001, min(0.999, score))


def build_env(seed: int = 0) -> CogTraceEnv:
    true_stage = seed % 5
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


def run_episode(env: CogTraceEnv, agent_fn: Callable[[Dict[str, Any]], int]) -> Dict[str, Any]:
    obs = env.reset()
    obs_dict = obs.model_dump()
    true_stage = env._sim.true_stage(0)
    predicted = agent_fn(obs_dict)
    predicted = max(0, min(4, int(predicted)))
    score = _score_prediction(true_stage, predicted)
    return {
        "true_stage":      true_stage,
        "predicted_stage": predicted,
        "score":           score,
        "observation":     obs_dict,
    }