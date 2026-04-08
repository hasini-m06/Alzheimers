# CogTraceEnv 🧠

**An OpenEnv-compliant reinforcement learning benchmark for AI-driven Alzheimer's cognitive monitoring**

> **Meta × PyTorch OpenEnv Hackathon submission** · Scaler School of Technology, Bangalore · April 2026

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-v1-purple.svg)](openenv.yaml)
[![Docker Ready](https://img.shields.io/badge/Docker-ready-blue.svg)](Dockerfile)
[![Domain: Healthcare AI](https://img.shields.io/badge/Domain-Healthcare_AI-red.svg)]()

---

## 🏥 Why This Problem Matters

Alzheimer's disease affects over **55 million people worldwide**, with India alone projected to have 14 million cases by 2050. The greatest clinical challenge is **early intervention** — detecting cognitive decline before irreversible damage occurs.

Traditional screening (MMSE, CDR assessments) happens infrequently, is expensive, and requires specialist access that rural patients often lack. Passive behavioral signals — typing speed, sleep duration, daily routine adherence, speech fluency — offer a non-invasive, continuous window into cognitive health that can be monitored at home.

CogTraceEnv frames this as a **reinforcement learning problem**: an AI agent monitors daily behavioral signals from a patient and must decide *when* and *how urgently* to raise a clinical alert. Getting this right saves lives. Getting it wrong causes either missed crises or alert fatigue that erodes clinical trust.

---

## 🧠 Why This Is a Hard RL Problem

CogTraceEnv is specifically designed to expose reasoning failures in frontier LLMs:

### 1. Extreme Class Imbalance
In a 30-day episode, only ~5 days contain a genuine anomaly (~17% of timesteps). A naive agent that always does `do_nothing` misses real events. An agent that always alerts scores even worse due to spam penalties. The agent must learn **selective restraint** — harder than it sounds.

### 2. Delayed and Noisy Signal
Individual readings are unreliable (σ ≈ 0.15–0.40 z-scores). Only **multi-signal correlated deviation over 7-day trends** reliably indicates deterioration. Agents must reason over time, not react to snapshots.

### 3. Proportional Response Required
Not all anomalies warrant an `escalate`. Mild disruption calls for a `soft_alert`; multi-system collapse warrants `escalate`. The reward function rewards **calibrated urgency**, not binary detection.

### 4. The "Do Nothing" Dilemma
Alert spam (≥4 alerts in 7 days) incurs an additional −0.15 penalty. The agent must track its own alert history — a form of working memory that LLMs handle inconsistently.

### 5. Gradual Drift + Acute Events
In Task 3, patients experience slow cognitive decline (0.01 CDR/day) AND a sudden anomaly window. Agents must distinguish background worsening from acute deterioration.

> **Empirical result**: Llama-3.1-8B scores ~0.35 on Task 3 (barely above the random baseline of 0.28). A rule-based oracle scores ~0.82. This 2× gap validates CogTraceEnv as a meaningful benchmark for agent reasoning under clinical uncertainty.

---

## 📊 Observation Space

Signals derived from published CDR-scale behavioral correlates (Morris 1993; Blessed 1968):

| Field | Type | Clinical Basis |
|---|---|---|
| `typing_delay_delta` | float (z-score) | Motor coordination & processing speed |
| `sleep_hours` | float [0–24] | Sleep disruption as early AD marker |
| `routine_adherence_score` | float [0–1] | Executive function & daily living |
| `speech_pause_freq` | float ≥ 0 | Word-finding difficulty (anomia) |
| `memory_lapse_count` | int ≥ 0 | Episodic memory failure |
| `days_elapsed` | int | Temporal context |
| `trend_typing_delay` | float | 7-day slope (positive = worsening) |
| `trend_sleep` | float | 7-day slope |
| `trend_routine` | float | 7-day slope |
| `alerts_last_7_days` | int | Agent's own alert history (working memory) |

---

## ⚡ Action Space

| Action | Label | Clinical Meaning |
|---|---|---|
| 0 | `do_nothing` | Continue passive monitoring |
| 1 | `soft_alert` | Flag for non-urgent review |
| 2 | `medium_alert` | Schedule checkup within 48 hours |
| 3 | `escalate` | Immediate clinical intervention |

---

## 🎯 Reward Function

Designed to reward **sensitivity without alert fatigue**:

| Situation | Reward |
|---|---|
| Escalate during anomaly | **+1.00** |
| Medium alert during anomaly | +0.60 |
| Soft alert during anomaly | +0.30 |
| Miss anomaly (`do_nothing`) | **−1.00** |
| Correct restraint (stable day) | +0.10 |
| Soft alert when stable | −0.10 |
| Medium alert when stable | −0.25 |
| False escalation | −0.50 |
| Alert spam (≥4 in 7 days) | −0.15 additional |

---

## 🗂️ Three-Tier Task Structure

### Task 1 — Cognitive Stage Classification `[Easy]`
**Input**: Single behavioral snapshot  
**Output**: CDR stage (0–4)  
**Grading**: Exact → 1.0 · Off-by-1 → 0.50 · Off-by-2 → 0.20 · Off-by-3+ → 0.00

### Task 2 — Anomaly Timing Detection `[Medium]`
**Input**: 7-day time series  
**Output**: First alert day (closest to true anomaly onset)  
**Grading**: Exact → 1.0 · ±1d → 0.75 · ±2d → 0.50 · ±3d → 0.25 · Otherwise → 0.00

### Task 3 — Full Triage Episode `[Hard]`
**Input**: 30-step episode with gradual decline + acute anomaly window  
**Output**: Per-step action sequence  
**Grading**: `0.70 × F1 + 0.30 × level_bonus`

---

## 📈 Benchmark Results

See [`BENCHMARK.md`](BENCHMARK.md) for full results and analysis.

| Agent | Task 1 | Task 2 | Task 3 | Mean |
|---|---|---|---|---|
| Random baseline | 0.25 | 0.20 | 0.28 | 0.24 |
| Always-alert baseline | 0.25 | 0.31 | 0.19 | 0.25 |
| Llama-3.1-8B (zero-shot) | 0.55 | 0.42 | 0.35 | 0.44 |
| **DQN Agent (PyTorch, ours)** | **0.62** | **0.58** | **0.625** | **0.608** |
| Rule-based oracle | **0.91** | **0.78** | **0.82** | **0.84** |

---

## 🏥 Named Patient Archetypes

Three evaluation scenarios in `configs/`:

| Config | CDR Stage | Profile |
|---|---|---|
| `early_onset.yaml` | Stage 1 (CDR 0.5) | Subtle decline, high noise — hardest to detect |
| `rapid_decliner.yaml` | Stage 2, drift=0.03 | Fast progression with acute event |
| `stable_elderly.yaml` | Stage 0 | Healthy — tests false positive rate |

---

## 🚀 Setup & Usage

### Local development
```bash
git clone https://github.com/Sparsha2708/Alzheimers
cd Alzheimers
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

### Run the rule-based oracle
```bash
python rule_based_agent.py
```

### API Quick Start
```bash
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": 0}'
curl http://localhost:7860/state
curl http://localhost:7860/tasks
```

### Interactive Demo Notebook
```bash
pip install jupyter matplotlib
jupyter notebook demo.ipynb
```

---

## 📁 Project Structure

```
cogtraceenv/
├── cognitive_env.py        # OpenEnv step/reset/state API
├── patient_simulator.py    # CDR-grounded synthetic patient generator
├── models.py               # Pydantic Observation/Action/Reward models
├── tasks/
│   ├── task1_easy.py       # Stage classification + grader
│   ├── task2_medium.py     # Anomaly timing + grader
│   └── task3_hard.py       # Full triage episode + grader
├── configs/                # Named patient archetypes (YAML)
│   ├── early_onset.yaml
│   ├── rapid_decliner.yaml
│   └── stable_elderly.yaml
├── results/                # Pre-run benchmark results (JSON + CSV)
├── demo.ipynb              # Interactive episode walkthrough with plots
├── inference.py            # LLM baseline agent
├── rule_based_agent.py     # Oracle agent for upper-bound benchmarking
├── app.py                  # FastAPI server
├── openenv.yaml            # OpenEnv spec metadata
├── BENCHMARK.md            # Full benchmark results & analysis
├── .gitignore
├── Dockerfile
└── requirements.txt
```

---

## 🔬 Clinical Grounding

| Feature | Source |
|---|---|
| CDR stage profiles | Morris (1993). *The CDR.* Neurology, 43(11) |
| Typing delay as biomarker | Zeng et al. (2021). *Digital biomarkers.* npj Digital Medicine |
| Sleep & Alzheimer's link | Ju et al. (2014). *Sleep quality.* JAMA Neurology |
| Speech pause frequency | König et al. (2015). *Automatic speech analysis.* Alzheimer's & Dementia |
| Acute anomaly events (UTI, medication errors) | Fick et al. (2002). *Comorbidity of dementia.* Arch Int Med |

---

## 📄 License

Apache 2.0

---

*Built for the Meta × PyTorch OpenEnv Hackathon · Scaler School of Technology · April 2026*
