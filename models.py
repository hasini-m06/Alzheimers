"""
Typed Pydantic models for CogTraceEnv — OpenEnv spec compliance.
Observation, Action, Reward, and supporting types.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import IntEnum


# ─── Action Space ────────────────────────────────────────────────────────────

class ActionType(IntEnum):
    DO_NOTHING   = 0
    SOFT_ALERT   = 1
    MEDIUM_ALERT = 2
    ESCALATE     = 3


class Action(BaseModel):
    action: int = Field(
        ...,
        ge=0,
        le=3,
        description=(
            "Agent action. "
            "0=do_nothing, 1=soft_alert, 2=medium_alert, 3=escalate"
        ),
    )

    @property
    def action_type(self) -> ActionType:
        return ActionType(self.action)

    @property
    def label(self) -> str:
        return ActionType(self.action).name.lower()


# ─── Observation Space ───────────────────────────────────────────────────────

class Observation(BaseModel):
    # Behavioral deltas (change from personal baseline, z-scored)
    typing_delay_delta: float = Field(
        ..., description="Change in typing latency vs personal baseline (z-score)"
    )
    sleep_hours: float = Field(
        ..., ge=0.0, le=24.0, description="Hours of sleep last night"
    )
    routine_adherence_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of daily routine steps completed on time"
    )
    speech_pause_freq: float = Field(
        ..., ge=0.0, description="Average pause frequency during speech (pauses/min)"
    )
    memory_lapse_count: int = Field(
        ..., ge=0, description="Observed memory-lapse events today (0–10+)"
    )
    # Temporal context
    days_elapsed: int = Field(
        ..., ge=0, description="Days since episode start"
    )
    # Rolling 7-day trend signals
    trend_typing_delay: float = Field(
        ..., description="7-day slope of typing_delay_delta (positive = worsening)"
    )
    trend_sleep: float = Field(
        ..., description="7-day slope of sleep_hours (negative = worsening)"
    )
    trend_routine: float = Field(
        ..., description="7-day slope of routine_adherence_score"
    )
    # Alert history (to discourage alert spamming)
    alerts_last_7_days: int = Field(
        ..., ge=0, description="Number of alerts (any level) agent raised in last 7 days"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "typing_delay_delta": 0.3,
            "sleep_hours": 5.8,
            "routine_adherence_score": 0.72,
            "speech_pause_freq": 3.1,
            "memory_lapse_count": 2,
            "days_elapsed": 14,
            "trend_typing_delay": 0.05,
            "trend_sleep": -0.12,
            "trend_routine": -0.03,
            "alerts_last_7_days": 1,
        }
    }}


# ─── Reward ──────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    value: float = Field(..., description="Scalar reward for this step")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown for interpretability",
    )
    reason: str = Field("", description="Human-readable explanation of reward")


# ─── Episode Info ────────────────────────────────────────────────────────────

class StepInfo(BaseModel):
    true_stage: int = Field(..., ge=0, le=4, description="Hidden ground-truth CDR stage (0–4)")
    anomaly_active: bool = Field(..., description="Whether a clinically significant event is active")
    false_positive: bool = Field(False, description="Agent alerted when no anomaly present")
    false_negative: bool = Field(False, description="Agent stayed silent during active anomaly")
    episode_score_so_far: float = Field(0.0, description="Cumulative score normalised to [0,1]")
    step: int = Field(..., description="Current step index")
    max_steps: int = Field(..., description="Episode length")


# ─── State (for state() endpoint) ────────────────────────────────────────────

class EnvState(BaseModel):
    step: int
    max_steps: int
    done: bool
    patient_id: str
    true_stage: int
    anomaly_active: bool
    cumulative_reward: float
    last_observation: Optional[Observation] = None
    alert_history: List[int] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
