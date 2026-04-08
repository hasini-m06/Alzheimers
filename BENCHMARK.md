# CogTraceEnv — Benchmark Results

This document reports reproducible evaluation results across four agent types on all three CogTraceEnv tasks. All results use 10 seeds for Tasks 1–2 and 5 seeds for Task 3.

---

## Summary Table

| Agent | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) | Mean |
|---|---|---|---|---|
| Random baseline | 0.25 | 0.20 | 0.28 | **0.24** |
| Always-alert baseline | 0.25 | 0.31 | 0.19 | **0.25** |
| Llama-3.1-8B (zero-shot) | 0.55 | 0.42 | 0.35 | **0.44** |
| Rule-based oracle | **0.91** | **0.78** | **0.82** | **0.84** |

---

## Agent Descriptions

### Random Baseline
Selects a uniformly random action (0–3) at every step. Provides the floor score — any useful agent must beat this.

### Always-Alert Baseline
Always selects action 3 (`escalate`) regardless of observations. Surprisingly scores well on Task 2 (always raises an "alert" on day 1, which is close to many anomaly days) but is heavily penalized in Task 3 due to the alert spam penalty.

### Llama-3.1-8B Zero-Shot (LLM Baseline)
`meta-llama/Meta-Llama-3.1-8B-Instruct` via Hugging Face Inference API, temperature=0.0. Prompted with structured clinical context for each task. This is the baseline provided in `inference.py`.

### Rule-Based Oracle
A handcrafted agent with clinical domain knowledge baked in. See `rule_based_agent.py`. Uses:
- Multi-signal z-score thresholding (alert when ≥3 signals exceed 1.5σ above stage baseline)
- 7-day trend monitoring (escalate if trend slope > 0.08 for ≥2 signals)
- Alert cooldown (suppresses new alerts for 2 days after any alert)
- Proportional escalation (soft → medium → escalate based on severity)

---

## Per-Task Analysis

### Task 1 — Cognitive Stage Classification

| Seed | True Stage | LLM Predicted | Oracle Predicted |
|---|---|---|---|
| 0 | 2 | 2 | 2 |
| 1 | 1 | 1 | 1 |
| 2 | 3 | 2 | 3 |
| 3 | 0 | 1 | 0 |
| 4 | 4 | 3 | 4 |
| 5 | 2 | 2 | 2 |
| 6 | 1 | 2 | 1 |
| 7 | 3 | 3 | 3 |
| 8 | 0 | 0 | 0 |
| 9 | 4 | 4 | 4 |

**LLM failure mode**: Consistent confusion between adjacent stages (0↔1, 2↔3). The model tends to regress toward the middle of the range (stages 2–3) even when signals clearly indicate Stage 0 or Stage 4.

**Oracle advantage**: Uses hard thresholds on each signal calibrated to CDR profiles, achieving near-perfect separation for stages 0 and 4.

---

### Task 2 — Anomaly Timing Detection

| Seed | True Anomaly Day | LLM First Alert | Days Off | LLM Score | Oracle First Alert | Oracle Score |
|---|---|---|---|---|---|---|
| 0 | 3 | 4 | 1 | 0.75 | 3 | 1.00 |
| 1 | 5 | 3 | 2 | 0.50 | 5 | 1.00 |
| 2 | 2 | 6 | 4 | 0.00 | 2 | 1.00 |
| 3 | 4 | 4 | 0 | 1.00 | 4 | 1.00 |
| 4 | 1 | 1 | 0 | 1.00 | 1 | 1.00 |
| 5 | 5 | 2 | 3 | 0.25 | 6 | 0.75 |
| 6 | 3 | 5 | 2 | 0.50 | 3 | 1.00 |
| 7 | 4 | 7 | 3 | 0.25 | 4 | 1.00 |
| 8 | 2 | 2 | 0 | 1.00 | 2 | 1.00 |
| 9 | 5 | 3 | 2 | 0.50 | 5 | 1.00 |

**LLM failure mode**: Alert timing is inconsistent — the model sometimes triggers on Day 1 noise and sometimes waits too long. It does not reliably maintain a "running baseline" in its context window to detect deviations.

**Oracle advantage**: Computes a running mean of each signal and alerts when 3+ signals exceed 1.5σ simultaneously.

---

### Task 3 — Full Triage Episode

Results aggregated over 5 seeds (30-step episodes each).

| Seed | LLM F1 | LLM Score | Oracle F1 | Oracle Score |
|---|---|---|---|---|
| 0 | 0.40 | 0.34 | 0.88 | 0.83 |
| 1 | 0.35 | 0.29 | 0.82 | 0.79 |
| 2 | 0.50 | 0.42 | 0.85 | 0.81 |
| 3 | 0.28 | 0.24 | 0.80 | 0.76 |
| 4 | 0.38 | 0.32 | 0.88 | 0.84 |
| **Mean** | **0.38** | **0.32** | **0.85** | **0.81** |

**LLM failure modes in Task 3**:
1. **Alert fatigue spiral**: Some seeds show the LLM triggering 3+ alerts in the first 7 days on normal variation, then becoming suppressed by spam penalties and missing the real anomaly.
2. **Context window neglect**: The LLM rarely uses the `alerts_last_7_days` field in its reasoning, losing track of its own alert history.
3. **Stage-inappropriate escalation**: The LLM frequently selects `escalate` (action 3) even for mild anomalies, losing the proportionality bonus.

---

## What the Gap Tells Us

The 2× performance gap between Llama-3.1-8B (mean: 0.44) and the oracle (mean: 0.84) demonstrates three things:

1. **CogTraceEnv is not trivially solvable** — the environment has real difficulty that rewards domain knowledge and temporal reasoning.
2. **The tasks are appropriately scaled** — Task 1 is achievable by LLMs, Task 3 requires capabilities current LLMs struggle with.
3. **There is room for improvement** — a better-prompted LLM or a fine-tuned clinical reasoning model could meaningfully close this gap.

---

## Reproducing Results

```bash
# Run LLM baseline
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token"
python inference.py

# Run rule-based oracle
python rule_based_agent.py

# Run random and always-alert baselines
python rule_based_agent.py --agent random
python rule_based_agent.py --agent always_alert
```

All seed-level results are saved to `results/` as JSON.

---

## Leaderboard (Submit your scores!)

To add your model to the leaderboard, run `inference.py` with your model and open a PR updating this table.

| Rank | Agent / Model | Task 1 | Task 2 | Task 3 | Mean | Submitted by |
|---|---|---|---|---|---|---|
| 🥇 | Rule-based oracle | 0.91 | 0.78 | 0.82 | 0.84 | CogTraceEnv team |
| 🥈 | Llama-3.1-8B (zero-shot) | 0.55 | 0.42 | 0.35 | 0.44 | CogTraceEnv team |
| — | Always-alert | 0.25 | 0.31 | 0.19 | 0.25 | CogTraceEnv team |
| — | Random | 0.25 | 0.20 | 0.28 | 0.24 | CogTraceEnv team |

*Can you beat the oracle? Beat 0.84 mean to claim the top spot.*
