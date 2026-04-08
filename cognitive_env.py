"""
cognitive_env.py
────────────────
CogTraceEnv — OpenEnv compliant environment for Alzheimer's monitoring.

Implements the full OpenEnv interface:
  reset()  → Observation
  step()   → (Observation, Reward, done: bool, StepInfo)
  state()  → EnvState
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any

from models import (
    Observation, Action, ActionType, Reward, StepInfo, EnvState
)
from patient_simulator import PatientSimulator, PatientConfig


# ─── Reward constants ─────────────────────────────────────────────────────────

_R = {
    # Correct actions during anomaly
    "escalate_during_anomaly":       +1.00,
    "medium_alert_during_anomaly":   +0.60,
    "soft_alert_during_anomaly":     +0.30,
    "do_nothing_during_anomaly":     -1.00,  # missed critical event

    # Actions when no anomaly (penalise over-alerting)
    "do_nothing_no_anomaly":         +0.10,  # small reward for correct restraint
    "soft_alert_no_anomaly":         -0.10,
    "medium_alert_no_anomaly":       -0.25,
    "escalate_no_anomaly":           -0.50,

    # Alert spam penalty (applied on top if alerts_last_7 >= 4)
    "spam_penalty":                  -0.15,
}


class CogTraceEnv:
    """
    OpenEnv-compliant environment for Alzheimer's cognitive monitoring.

    The agent observes daily behavioral signals for a synthetic patient
    and must decide when and at what level to raise a clinical alert.

    Parameters
    ----------
    config : PatientConfig
        Patient configuration. If None, defaults to a moderate-stage
        30-day episode.
    """

    ENV_ID   = "CogTraceEnv-v1"
    VERSION  = "1.0.0"

    def __init__(self, config: Optional[PatientConfig] = None):
        self.config = config or PatientConfig(
            true_stage=2,
            episode_length=30,
            decline_rate=0.01,
            noise_level=1.0,
            seed=None,
            patient_id="default",
        )
        self._sim: Optional[PatientSimulator] = None
        self._step_idx: int = 0
        self._done: bool = False
        self._alert_history: List[int] = []   # action per step
        self._cumulative_reward: float = 0.0
        self._last_obs: Optional[Observation] = None

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """
        Initialise (or re-initialise) the environment.
        Returns the first observation.
        """
        self._sim = PatientSimulator(self.config)
        self._step_idx = 0
        self._done = False
        self._alert_history = []
        self._cumulative_reward = 0.0
        obs = self._make_observation()
        self._last_obs = obs
        return obs

    def step(self, action: Action | int | dict) -> Tuple[Observation, Reward, bool, StepInfo]:
        """
        Advance the environment by one timestep.

        Parameters
        ----------
        action : Action | int | dict
            The agent's chosen action for this step.

        Returns
        -------
        observation : Observation  — next state
        reward      : Reward       — reward for this step
        done        : bool         — episode complete
        info        : StepInfo     — diagnostics
        """
        if self._sim is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        # Normalise action
        if isinstance(action, dict):
            action = Action(**action)
        elif isinstance(action, int):
            action = Action(action=action)

        anomaly_active = self._sim.is_anomaly_active(self._step_idx)
        true_stage     = self._sim.true_stage(self._step_idx)
        reward, fp, fn = self._compute_reward(action, anomaly_active)

        # Record
        self._alert_history.append(action.action)
        self._cumulative_reward += reward.value

        # Advance
        self._step_idx += 1
        done = self._step_idx >= self.config.episode_length
        self._done = done

        # Build next observation (or terminal zero-obs)
        if not done:
            obs = self._make_observation()
        else:
            obs = self._last_obs  # return last valid obs on terminal step

        self._last_obs = obs

        # Normalised episode score
        max_possible = self.config.episode_length * _R["escalate_during_anomaly"]
        ep_score = max(0.0, min(1.0, self._cumulative_reward / max(max_possible, 1.0)))

        info = StepInfo(
            true_stage=true_stage,
            anomaly_active=anomaly_active,
            false_positive=fp,
            false_negative=fn,
            episode_score_so_far=round(ep_score, 4),
            step=self._step_idx - 1,
            max_steps=self.config.episode_length,
        )

        return obs, reward, done, info

    def state(self) -> EnvState:
        """Return the full current state (internal, for inspection / grading)."""
        if self._sim is None:
            raise RuntimeError("Call reset() before state().")
        true_stage    = self._sim.true_stage(min(self._step_idx, self.config.episode_length - 1))
        anomaly_active = self._sim.is_anomaly_active(min(self._step_idx, self.config.episode_length - 1))
        return EnvState(
            step=self._step_idx,
            max_steps=self.config.episode_length,
            done=self._done,
            patient_id=self.config.patient_id,
            true_stage=true_stage,
            anomaly_active=anomaly_active,
            cumulative_reward=round(self._cumulative_reward, 4),
            last_observation=self._last_obs,
            alert_history=list(self._alert_history),
            config=self.config.__dict__,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        obs_dict = self._sim.observation_dict(self._step_idx, self._alert_history)
        return Observation(**obs_dict)

    def _compute_reward(
        self, action: Action, anomaly_active: bool
    ) -> Tuple[Reward, bool, bool]:
        """
        Compute reward for one step.

        Returns (Reward, false_positive, false_negative)
        """
        a = action.action_type
        breakdown: Dict[str, float] = {}
        fp = fn = False

        if anomaly_active:
            fn = (a == ActionType.DO_NOTHING)
            key_map = {
                ActionType.ESCALATE:     "escalate_during_anomaly",
                ActionType.MEDIUM_ALERT: "medium_alert_during_anomaly",
                ActionType.SOFT_ALERT:   "soft_alert_during_anomaly",
                ActionType.DO_NOTHING:   "do_nothing_during_anomaly",
            }
            base_key = key_map[a]
        else:
            fp = (a != ActionType.DO_NOTHING)
            key_map = {
                ActionType.DO_NOTHING:   "do_nothing_no_anomaly",
                ActionType.SOFT_ALERT:   "soft_alert_no_anomaly",
                ActionType.MEDIUM_ALERT: "medium_alert_no_anomaly",
                ActionType.ESCALATE:     "escalate_no_anomaly",
            }
            base_key = key_map[a]

        base_reward = _R[base_key]
        breakdown[base_key] = base_reward

        # Spam penalty
        spam_pen = 0.0
        recent_alerts = sum(1 for x in self._alert_history[-7:] if x > 0)
        if recent_alerts >= 4 and a != ActionType.DO_NOTHING:
            spam_pen = _R["spam_penalty"]
            breakdown["spam_penalty"] = spam_pen

        total = base_reward + spam_pen
        reward = Reward(
            value=round(total, 4),
            breakdown=breakdown,
            reason=(
                f"Step {self._step_idx}: action={action.label}, "
                f"anomaly={'yes' if anomaly_active else 'no'}, "
                f"base={base_reward:.2f}, spam_pen={spam_pen:.2f}"
            ),
        )
        return reward, fp, fn

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def action_space_description(self) -> Dict[int, str]:
        return {a.value: a.name.lower() for a in ActionType}

    @property
    def observation_space_description(self) -> Dict[str, str]:
        return {k: v.description for k, v in Observation.model_fields.items()}

    def __repr__(self) -> str:
        return (
            f"CogTraceEnv(stage={self.config.true_stage}, "
            f"steps={self.config.episode_length}, "
            f"step={self._step_idx}/{self.config.episode_length})"
        )
