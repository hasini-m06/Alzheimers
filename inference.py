"""
inference.py
────────────
CogTraceEnv — Baseline Inference Script
Meta × PyTorch OpenEnv Hackathon

Required environment variables:
  API_BASE_URL — e.g. https://api-inference.huggingface.co/v1
  MODEL_NAME   — e.g. meta-llama/Meta-Llama-3.1-8B-Instruct
  HF_TOKEN     — your Hugging Face token

STDOUT FORMAT (strictly followed):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import os
import sys
import json
import time
import re
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Env config ────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

BENCHMARK        = "CogTraceEnv"
NUM_SEEDS_EASY   = 10
NUM_SEEDS_MEDIUM = 10
NUM_SEEDS_HARD   = 5

SUCCESS_SCORE_THRESHOLD = 0.5  # score >= 0.5 counts as success

client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

# ── Structured log helpers (exact format required by grader) ──────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── LLM helpers ───────────────────────────────────────────────────────────────
def _chat(system: str, user: str, max_tokens: int = 64, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[DEBUG] LLM error: {e}", file=sys.stderr, flush=True)
                return "0"
    return "0"


def _parse_int(text: str, lo: int = 0, hi: int = 3) -> int:
    nums = re.findall(r"\d+", text)
    if not nums:
        return 0
    return max(lo, min(hi, int(nums[0])))

# ── Task agents ───────────────────────────────────────────────────────────────
TASK1_SYSTEM = """\
You are a clinical AI assistant. You will receive behavioral metrics for a patient
being monitored for Alzheimer's disease.

Your task: classify the patient's CDR (Clinical Dementia Rating) stage.
  0 = Healthy / no impairment
  1 = Very mild impairment
  2 = Mild impairment
  3 = Moderate impairment
  4 = Severe impairment

Respond with a SINGLE integer 0-4. No explanation."""

def task1_agent(obs: Dict[str, Any]) -> int:
    user = f"Patient behavioral snapshot:\n{json.dumps(obs, indent=2)}\n\nStage (0-4):"
    raw  = _chat(TASK1_SYSTEM, user, max_tokens=8)
    return _parse_int(raw, lo=0, hi=4)


TASK2_SYSTEM = """\
You are a clinical AI monitoring a patient for Alzheimer's-related deterioration.
Each day you receive behavioral sensor readings. You must choose an action:
  0 = do_nothing
  1 = soft_alert
  2 = medium_alert
  3 = escalate

Raise your FIRST alert (action >= 1) as soon as you detect an anomaly.
Respond with a SINGLE integer 0-3. No explanation."""

def task2_agent(obs: Dict[str, Any], step: int) -> int:
    user = (
        f"Day {step + 1} observation:\n{json.dumps(obs, indent=2)}\n\n"
        "Action (0=do_nothing, 1=soft_alert, 2=medium_alert, 3=escalate):"
    )
    raw = _chat(TASK2_SYSTEM, user, max_tokens=8)
    return _parse_int(raw, lo=0, hi=3)


TASK3_SYSTEM = """\
You are a clinical AI monitoring a patient over 30 days.
Each day, choose the most appropriate action:
  0 = do_nothing      (patient appears stable)
  1 = soft_alert      (minor concern, flag for review)
  2 = medium_alert    (notable deterioration, schedule checkup)
  3 = escalate        (acute event, immediate intervention)

- Most days the patient is stable, prefer do_nothing to avoid alert fatigue.
- When multiple signals deteriorate together, escalate proportionally.
- Avoid raising more than 4 alerts in any 7-day window.

Respond with a SINGLE integer 0-3. No explanation."""

def task3_agent(obs: Dict[str, Any], step: int, history: List[int]) -> int:
    recent = history[-7:] if len(history) >= 7 else history
    user   = (
        f"Day {step + 1}/30 observation:\n{json.dumps(obs, indent=2)}\n"
        f"Recent actions (last {len(recent)} days): {recent}\n\n"
        "Action (0-3):"
    )
    raw = _chat(TASK3_SYSTEM, user, max_tokens=8)
    return _parse_int(raw, lo=0, hi=3)

# ── Task runners ──────────────────────────────────────────────────────────────
def run_task1() -> float:
    from task1_easy import run_episode, grade, build_env

    log_start(task="task1_easy", env=BENCHMARK, model=MODEL_NAME)

    results     = []
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    try:
        for seed in range(NUM_SEEDS_EASY):
            env    = build_env(seed)
            result = run_episode(env, agent_fn=task1_agent)
            results.append(result)

            reward = result["score"]
            rewards.append(reward)
            steps_taken += 1

            log_step(
                step=seed + 1,
                action=f"predict_stage({result['predicted_stage']})",
                reward=reward,
                done=(seed == NUM_SEEDS_EASY - 1),
                error=None,
            )

        score   = grade(results)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        traceback.print_exc()

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def run_task2() -> float:
    from task2_medium import run_episode, grade, build_env

    log_start(task="task2_medium", env=BENCHMARK, model=MODEL_NAME)

    results     = []
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    try:
        for seed in range(NUM_SEEDS_MEDIUM):
            env    = build_env(seed)
            result = run_episode(env, agent_fn=task2_agent)
            results.append(result)

            reward = result["score"]
            rewards.append(reward)
            steps_taken += 1

            log_step(
                step=seed + 1,
                action=f"first_alert_day({result['first_alert_day']})",
                reward=reward,
                done=(seed == NUM_SEEDS_MEDIUM - 1),
                error=None,
            )

        score   = grade(results)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        traceback.print_exc()

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def run_task3() -> float:
    from task3_hard import run_episode, grade, build_env

    log_start(task="task3_hard", env=BENCHMARK, model=MODEL_NAME)

    results     = []
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    try:
        for seed in range(NUM_SEEDS_HARD):
            env = build_env(seed)

            # task3 needs history passed into agent; wrap to match run_episode signature
            history: List[int] = []
            def agent_with_history(obs_dict, step, _hist=None):
                action = task3_agent(obs_dict, step, list(history))
                history.append(action)
                return action

            result = run_episode(env, agent_fn=agent_with_history)
            results.append(result)

            reward = result["score"]
            rewards.append(reward)
            steps_taken += 1

            log_step(
                step=seed + 1,
                action=f"episode(tp={result['tp']},fp={result['fp']},fn={result['fn']})",
                reward=reward,
                done=(seed == NUM_SEEDS_HARD - 1),
                error=None,
            )

        score   = grade(results)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        traceback.print_exc()

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    task1_score = run_task1()
    task2_score = run_task2()
    task3_score = run_task3()

    mean = (task1_score + task2_score + task3_score) / 3
    print(f"[DEBUG] Mean score across all tasks: {mean:.4f}", flush=True)


if __name__ == "__main__":
    main()
