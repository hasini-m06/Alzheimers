# CogTraceEnv 🧠

**An OpenEnv environment for AI-driven Alzheimer's cognitive monitoring**

> A Meta × Hugging Face OpenEnv Hackathon submission.

---

## Motivation

Alzheimer's disease affects over 55 million people worldwide. One of the greatest clinical challenges is *early intervention* — detecting deterioration before it becomes irreversible. Passive behavioral monitoring (typing patterns, sleep quality, daily routine adherence, speech characteristics) offers a non-invasive window into cognitive decline.

CogTraceEnv simulates this problem as a reinforcement learning environment: an AI agent must observe daily behavioral signals from a synthetic patient and decide *when* and *at what level* to raise a clinical alert — balancing sensitivity against alert fatigue.

---

## Environment Description

CogTraceEnv generates synthetic patient trajectories grounded in published CDR (Clinical Dementia Rating) scale behavioral correlates. Each episode runs for up to 30 timesteps. The agent observes multi-modal behavioral signals and chooses from 4 discrete actions.

The environment's defining challenge is the **"do nothing" dilemma**: most days the patient is stable, and over-alerting is penalised. The agent must develop restraint — which turns out to be exactly what makes this problem hard for frontier LLMs.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `typing_delay_delta` | float | Change in typing latency vs personal baseline (z-score) |
| `sleep_hours` | float [0–24] | Hours of sleep last night |
| `routine_adherence_score` | float [0–1] | Fraction of daily routine completed on time |
| `speech_pause_freq` | float ≥ 0 | Average pauses per minute during speech |
| `memory_lapse_count` | int ≥ 0 | Observed memory lapses today |
| `days_elapsed` | int | Days since episode start |
| `trend_typing_delay` | float | 7-day slope of typing delay (positive = worsening) |
| `trend_sleep` | float | 7-day slope of sleep hours |
| `trend_routine` | float | 7-day slope of routine adherence |
| `alerts_last_7_days` | int | Number of alerts agent raised in the last 7 days |

---

## Action Space

| Action | Label | Description |
|---|---|---|
| 0 | `do_nothing` | Continue passive monitoring |
| 1 | `soft_alert` | Flag for non-urgent clinical review |
| 2 | `medium_alert` | Schedule clinical check within 48 hours |
| 3 | `escalate` | Immediate clinical intervention |

---

## Reward Function

The reward provides dense signal throughout each trajectory:

| Situation | Reward |
|---|---|
| Escalate during anomaly | +1.00 |
| Medium alert during anomaly | +0.60 |
| Soft alert during anomaly | +0.30 |
| Do nothing during anomaly | −1.00 (missed event) |
| Do nothing when stable | +0.10 (correct restraint) |
| Soft alert when stable | −0.10 |
| Medium alert when stable | −0.25 |
| Escalate when stable | −0.50 |
| Alert spam (4+ in 7 days) | −0.15 additional |

---

## Tasks

### Task 1 — Cognitive Stage Classification (Easy)

Given a **single snapshot** of behavioral metrics, classify the patient's Alzheimer's stage (0–4, mapping to CDR 0, 0.5, 1, 2, 3).

**Grading:** Exact match → 1.00 · Off by 1 → 0.50 · Off by 2 → 0.20 · Off by 3+ → 0.00

---

### Task 2 — Anomaly Timing Detection (Medium)

Observe a **7-day time series**. Raise your first alert (action ≥ 1) as close as possible to the true anomaly onset day.

**Grading:** Exact day → 1.00 · ±1 day → 0.75 · ±2 days → 0.50 · ±3 days → 0.25 · Otherwise → 0.00

---

### Task 3 — Full Triage Episode (Hard)

Manage a **30-step episode** with gradual patient decline and one acute anomaly window. Scored on F1 (sensitivity + specificity) and alert proportionality.

**Grading:** `0.70 × F1 + 0.30 × level_bonus`

---

## Baseline Scores

Scores from `inference.py` using `meta-llama/Meta-Llama-3.1-8B-Instruct` (10 seeds each):

| Task | Score |
|---|---|
| Task 1 — Stage Classification | ~0.55 |
| Task 2 — Anomaly Timing | ~0.42 |
| Task 3 — Full Triage | ~0.35 |

Random baseline for comparison: Task 1 ≈ 0.25 · Task 2 ≈ 0.20 · Task 3 ≈ 0.28

---

## Setup & Usage

### Local development

```bash
git clone <this-repo>
cd cogtraceenv
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
```

### Docker

```bash
docker build -t cogtraceenv .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="your_token_here" \
  cogtraceenv
```

### Run baseline inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py
```

### API Quick Start

```bash
# Reset environment
curl -X POST http://localhost:7860/reset

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": 0}'

# Check state
curl http://localhost:7860/state

# List tasks
curl http://localhost:7860/tasks
```

---

## Project Structure

```
cogtraceenv/
├── env/
│   ├── __init__.py
│   ├── cognitive_env.py      # OpenEnv step/reset/state API
│   ├── patient_simulator.py  # Synthetic patient data generator
│   └── models.py             # Pydantic Observation/Action/Reward models
├── tasks/
│   ├── __init__.py
│   ├── task1_easy.py         # Stage classification + grader
│   ├── task2_medium.py       # Anomaly timing + grader
│   └── task3_hard.py         # Full triage + grader
├── inference.py              # Baseline LLM agent (root level, required)
├── app.py                    # FastAPI server
├── openenv.yaml              # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## License

Apache 2.0
