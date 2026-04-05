"""
inference.py
────────────
Baseline inference script for CogTraceEnv.

Runs an LLM agent (via OpenAI-compatible client) against all 3 tasks
and prints reproducible scores.

Required environment variables:
  API_BASE_URL  — e.g. https://api-inference.huggingface.co/v1
  MODEL_NAME    — e.g. meta-llama/Meta-Llama-3.1-8B-Instruct
  HF_TOKEN      — your Hugging Face token (used as API key)

Usage:
  python inference.py

Runtime: < 20 min on 2 vCPU / 8 GB RAM
"""

from __future__ import annotations

import os
import sys
import json
import time
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Env config ────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

NUM_SEEDS_EASY   = 10
NUM_SEEDS_MEDIUM = 10
NUM_SEEDS_HARD   = 5   # fewer because 30-step episodes take longer

client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

# ── LLM helpers ───────────────────────────────────────────────────────────────

def _chat(system: str, user: str, max_tokens: int = 64, retries: int = 3) -> str:
    """Call the LLM and return the stripped response text."""
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
                print(f"  [LLM ERROR] {e}", file=sys.stderr)
                return "0"
    return "0"


def _parse_int(text: str, lo: int = 0, hi: int = 3) -> int:
    """Extract first integer from model output, clamped to [lo, hi]."""
    import re
    nums = re.findall(r"\d+", text)
    if not nums:
        return 0
    return max(lo, min(hi, int(nums[0])))


# ── Task 1 — Stage Classification ────────────────────────────────────────────

TASK1_SYSTEM = """\
You are a clinical AI assistant. You will receive behavioral metrics for a patient
being monitored for Alzheimer's disease.

Your task: classify the patient's CDR (Clinical Dementia Rating) stage.
  0 = Healthy / no impairment
  1 = Very mild impairment
  2 = Mild impairment
  3 = Moderate impairment
  4 = Severe impairment

Respond with a SINGLE integer 0–4. No explanation."""

def task1_agent(obs: Dict[str, Any]) -> int:
    user = f"Patient behavioral snapshot:\n{json.dumps(obs, indent=2)}\n\nStage (0–4):"
    raw = _chat(TASK1_SYSTEM, user, max_tokens=8)
    return _parse_int(raw, lo=0, hi=4)


# ── Task 2 — Anomaly Timing ───────────────────────────────────────────────────

TASK2_SYSTEM = """\
You are a clinical AI monitoring a patient for Alzheimer's-related deterioration.

Each day you receive behavioral sensor readings. You must choose an action:
  0 = do_nothing
  1 = soft_alert
  2 = medium_alert
  3 = escalate

Raise your FIRST alert (action >= 1) as soon as you detect an anomaly —
a sudden worsening across multiple signals compared to the patient's baseline.
Be neither too early (normal variation) nor too late (delayed response).

Respond with a SINGLE integer 0–3. No explanation."""

def task2_agent(obs: Dict[str, Any], step: int) -> int:
    user = (
        f"Day {step + 1} observation:\n{json.dumps(obs, indent=2)}\n\n"
        "Action (0=do_nothing, 1=soft_alert, 2=medium_alert, 3=escalate):"
    )
    raw = _chat(TASK2_SYSTEM, user, max_tokens=8)
    return _parse_int(raw, lo=0, hi=3)


# ── Task 3 — Full Triage ──────────────────────────────────────────────────────

TASK3_SYSTEM = """\
You are a clinical AI monitoring a patient over 30 days.
Each day, choose the most appropriate action:
  0 = do_nothing     (patient appears stable)
  1 = soft_alert     (minor concern, flag for review)
  2 = medium_alert   (notable deterioration, schedule checkup)
  3 = escalate       (acute event, immediate intervention)

Key principles:
- Most days the patient is stable → prefer do_nothing to avoid alert fatigue
- When multiple signals deteriorate together, escalate proportionally
- Avoid raising more than 4 alerts in any 7-day window unless truly warranted

Respond with a SINGLE integer 0–3. No explanation."""

def task3_agent(obs: Dict[str, Any], step: int, history: List[int]) -> int:
    recent = history[-7:] if len(history) >= 7 else history
    user = (
        f"Day {step + 1}/30 observation:\n{json.dumps(obs, indent=2)}\n"
        f"Recent actions (last {len(recent)} days): {recent}\n\n"
        "Action (0–3):"
    )
    raw = _chat(TASK3_SYSTEM, user, max_tokens=8)
    return _parse_int(raw, lo=0, hi=3)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_task1() -> float:
    from tasks.task1_easy import run_episode, grade, build_env
    results = []
    for seed in range(NUM_SEEDS_EASY):
        env = build_env(seed)
        result = run_episode(env, agent_fn=task1_agent)
        results.append(result)
        print(f"  seed={seed:02d}  true={result['true_stage']}  pred={result['predicted_stage']}  score={result['score']:.2f}")
    return grade(results)


def run_task2() -> float:
    from tasks.task2_medium import run_episode, grade, build_env
    results = []
    for seed in range(NUM_SEEDS_MEDIUM):
        env = build_env(seed)
        result = run_episode(env, agent_fn=task2_agent)
        results.append(result)
        print(f"  seed={seed:02d}  anomaly_day={result['anomaly_day']}  first_alert={result['first_alert_day']}  score={result['score']:.2f}")
    return grade(results)


def run_task3() -> float:
    from tasks.task3_hard import run_episode, grade, build_env
    results = []
    for seed in range(NUM_SEEDS_HARD):
        env = build_env(seed)
        result = run_episode(env, agent_fn=task3_agent)
        results.append(result)
        print(f"  seed={seed:02d}  tp={result['tp']}  fp={result['fp']}  fn={result['fn']}  f1={result['f1_score']:.2f}  score={result['score']:.2f}")
    return grade(results)


def main():
    print("=" * 60)
    print("CogTraceEnv — Baseline Inference")
    print(f"Model : {MODEL_NAME}")
    print(f"API   : {API_BASE_URL}")
    print("=" * 60)

    scores: Dict[str, Optional[float]] = {}

    for task_id, runner, label in [
        ("task1_easy",   run_task1, "Task 1 (Easy)   — Stage Classification"),
        ("task2_medium", run_task2, "Task 2 (Medium) — Anomaly Timing"),
        ("task3_hard",   run_task3, "Task 3 (Hard)   — Full Triage"),
    ]:
        print(f"\n▶ {label}")
        t0 = time.time()
        try:
            score = runner()
            elapsed = time.time() - t0
            scores[task_id] = score
            print(f"  ✓ SCORE: {score:.4f}  ({elapsed:.1f}s)")
        except Exception:
            print(f"  ✗ FAILED")
            traceback.print_exc()
            scores[task_id] = None

    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    valid = [s for s in scores.values() if s is not None]
    for task_id, score in scores.items():
        val = f"{score:.4f}" if score is not None else "ERROR"
        print(f"  {task_id:<20} {val}")
    if valid:
        print(f"\n  Mean score: {sum(valid) / len(valid):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
