"""
patient_simulator.py
────────────────────
Rule-based synthetic patient generator for CogTraceEnv.

Generates a 30-day stream of behavioral observations for a simulated
Alzheimer's patient at a configurable CDR stage, with realistic noise,
clinically motivated decline curves, and injected anomaly events.

No real patient data is used. All statistics are derived from published
CDR-scale behavioral correlates (Blessed 1968, Morris 1993, Auer 1997).
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


# ─── CDR Stage Profiles ──────────────────────────────────────────────────────
# Each stage defines (mean, std) for each behavioral signal at baseline.
# CDR 0 = healthy; CDR 0.5 = very mild; CDR 1 = mild; CDR 2 = moderate; CDR 3 = severe

CDR_PROFILES: Dict[int, Dict[str, Tuple[float, float]]] = {
    0: {   # Healthy
        "typing_delay_delta":      (0.00, 0.15),
        "sleep_hours":             (7.20, 0.60),
        "routine_adherence_score": (0.92, 0.05),
        "speech_pause_freq":       (1.20, 0.30),
        "memory_lapse_count":      (0.10, 0.30),
    },
    1: {   # Very mild (CDR 0.5 mapped to stage 1)
        "typing_delay_delta":      (0.25, 0.20),
        "sleep_hours":             (6.80, 0.70),
        "routine_adherence_score": (0.83, 0.07),
        "speech_pause_freq":       (1.80, 0.40),
        "memory_lapse_count":      (0.80, 0.60),
    },
    2: {   # Mild (CDR 1)
        "typing_delay_delta":      (0.60, 0.25),
        "sleep_hours":             (6.20, 0.80),
        "routine_adherence_score": (0.68, 0.10),
        "speech_pause_freq":       (2.80, 0.60),
        "memory_lapse_count":      (2.50, 1.00),
    },
    3: {   # Moderate (CDR 2)
        "typing_delay_delta":      (1.10, 0.35),
        "sleep_hours":             (5.50, 1.00),
        "routine_adherence_score": (0.48, 0.12),
        "speech_pause_freq":       (4.20, 0.80),
        "memory_lapse_count":      (5.00, 1.50),
    },
    4: {   # Severe (CDR 3)
        "typing_delay_delta":      (1.80, 0.40),
        "sleep_hours":             (4.80, 1.20),
        "routine_adherence_score": (0.25, 0.12),
        "speech_pause_freq":       (6.50, 1.00),
        "memory_lapse_count":      (8.00, 2.00),
    },
}

# Anomaly = acute deterioration event (e.g., UTI, medication error, sleep disruption)
# Multipliers applied to the stage baseline during anomaly window
ANOMALY_MULTIPLIERS: Dict[str, float] = {
    "typing_delay_delta":      2.2,
    "sleep_hours":             0.65,   # sleep drops
    "routine_adherence_score": 0.55,   # routine breaks down
    "speech_pause_freq":       2.0,
    "memory_lapse_count":      2.5,
}


@dataclass
class PatientConfig:
    """Configures a synthetic patient episode."""
    true_stage: int = 1                    # CDR stage (0–4)
    episode_length: int = 30               # total timesteps
    decline_rate: float = 0.0              # per-step stage drift (0 = stable)
    noise_level: float = 1.0               # multiplier on all std devs
    anomaly_day: Optional[int] = None      # day index when anomaly begins (None = random)
    anomaly_duration: int = 5              # how many steps the anomaly lasts
    seed: Optional[int] = None             # for reproducibility
    patient_id: str = "patient_001"


@dataclass
class DayRecord:
    """Internal record for one simulated day."""
    day: int
    stage_at_day: float                    # continuous stage value
    typing_delay_delta: float
    sleep_hours: float
    routine_adherence_score: float
    speech_pause_freq: float
    memory_lapse_count: int
    anomaly_active: bool
    true_stage_int: int                    # discretised


class PatientSimulator:
    """
    Generates a complete synthetic episode for one patient.

    Usage
    -----
    sim = PatientSimulator(PatientConfig(true_stage=2, seed=42))
    for day_record in sim.records:
        obs = sim.observation_at(day_record.day)
    """

    def __init__(self, config: PatientConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._random = random.Random(config.seed)

        # Resolve anomaly day
        if config.anomaly_duration == 0:
            # No anomaly requested (e.g. Task 1 single-snapshot)
            self.anomaly_day = config.episode_length  # beyond episode → never active
        elif config.anomaly_day is not None:
            self.anomaly_day = config.anomaly_day
        else:
            # Place anomaly in the middle third of the episode (not too early, not too late)
            earliest = max(1, config.episode_length // 3)
            latest   = max(earliest + 1, (2 * config.episode_length) // 3)
            self.anomaly_day = int(self.rng.integers(earliest, latest))

        self.anomaly_end = self.anomaly_day + config.anomaly_duration

        # Pre-generate the full episode
        self.records: List[DayRecord] = self._generate_episode()

        # Pre-compute 7-day rolling trends for each day
        self._trend_cache: Dict[int, Dict[str, float]] = {}
        self._compute_trends()

    # ── Generation ────────────────────────────────────────────────────────────

    def _stage_at_day(self, day: int) -> float:
        """Continuous stage value accounting for decline drift."""
        return min(4.0, self.config.true_stage + day * self.config.decline_rate)

    def _interpolate_profile(self, stage_float: float, key: str) -> Tuple[float, float]:
        """Linearly interpolate between CDR stage profiles."""
        lo = int(math.floor(stage_float))
        hi = min(lo + 1, 4)
        alpha = stage_float - lo
        mu_lo, sd_lo = CDR_PROFILES[lo][key]
        mu_hi, sd_hi = CDR_PROFILES[hi][key]
        return (1 - alpha) * mu_lo + alpha * mu_hi, (1 - alpha) * sd_lo + alpha * sd_hi

    def _sample(self, mu: float, sd: float, lo: float = -np.inf, hi: float = np.inf) -> float:
        """Truncated-ish normal sample via clipping."""
        val = self.rng.normal(mu, sd * self.config.noise_level)
        return float(np.clip(val, lo, hi))

    def _generate_day(self, day: int) -> DayRecord:
        stage_f = self._stage_at_day(day)
        anomaly = self.anomaly_day <= day < self.anomaly_end

        signals = {}
        for key in CDR_PROFILES[0]:
            mu, sd = self._interpolate_profile(stage_f, key)
            if anomaly:
                mu *= ANOMALY_MULTIPLIERS.get(key, 1.0)
            signals[key] = mu, sd

        # Sample each signal with appropriate bounds
        typing_delay = self._sample(*signals["typing_delay_delta"])
        sleep_h = self._sample(*signals["sleep_hours"], lo=0.0, hi=14.0)
        routine = self._sample(*signals["routine_adherence_score"], lo=0.0, hi=1.0)
        speech_p = self._sample(*signals["speech_pause_freq"], lo=0.0)
        memory_mu, memory_sd = signals["memory_lapse_count"]
        memory = max(0, int(round(self._sample(memory_mu, memory_sd, lo=0.0))))

        return DayRecord(
            day=day,
            stage_at_day=stage_f,
            typing_delay_delta=typing_delay,
            sleep_hours=sleep_h,
            routine_adherence_score=routine,
            speech_pause_freq=speech_p,
            memory_lapse_count=memory,
            anomaly_active=anomaly,
            true_stage_int=int(round(stage_f)),
        )

    def _generate_episode(self) -> List[DayRecord]:
        return [self._generate_day(d) for d in range(self.config.episode_length)]

    # ── Trend Computation ─────────────────────────────────────────────────────

    def _slope(self, values: List[float]) -> float:
        """Linear regression slope over a short window."""
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        y = np.array(values, dtype=float)
        x -= x.mean()
        denom = (x ** 2).sum()
        return float((x * (y - y.mean())).sum() / denom) if denom > 0 else 0.0

    def _compute_trends(self):
        window = 7
        keys = ["typing_delay_delta", "sleep_hours", "routine_adherence_score"]
        for day in range(len(self.records)):
            start = max(0, day - window + 1)
            window_records = self.records[start: day + 1]
            self._trend_cache[day] = {
                k: self._slope([getattr(r, k) for r in window_records])
                for k in keys
            }

    # ── Public API ────────────────────────────────────────────────────────────

    def observation_dict(self, day: int, alert_history: List[int]) -> Dict[str, Any]:
        """Return raw dict matching the Observation model fields."""
        r = self.records[day]
        trends = self._trend_cache[day]
        alerts_last_7 = sum(1 for d, a in enumerate(alert_history) if d >= day - 6 and a > 0)
        return {
            "typing_delay_delta":      round(r.typing_delay_delta, 4),
            "sleep_hours":             round(r.sleep_hours, 2),
            "routine_adherence_score": round(r.routine_adherence_score, 4),
            "speech_pause_freq":       round(r.speech_pause_freq, 4),
            "memory_lapse_count":      r.memory_lapse_count,
            "days_elapsed":            day,
            "trend_typing_delay":      round(trends["typing_delay_delta"], 5),
            "trend_sleep":             round(trends["sleep_hours"], 5),
            "trend_routine":           round(trends["routine_adherence_score"], 5),
            "alerts_last_7_days":      alerts_last_7,
        }

    def is_anomaly_active(self, day: int) -> bool:
        return self.records[day].anomaly_active

    def true_stage(self, day: int) -> int:
        return self.records[day].true_stage_int

    @property
    def anomaly_window(self) -> Tuple[int, int]:
        """(start, end) inclusive day indices of the anomaly."""
        return self.anomaly_day, self.anomaly_end - 1


# ─── Factory Helpers ─────────────────────────────────────────────────────────

def make_easy_patient(seed: int = 0) -> PatientSimulator:
    """Task 1: Single snapshot, deterministic stage 2."""
    cfg = PatientConfig(
        true_stage=2,
        episode_length=1,
        decline_rate=0.0,
        noise_level=0.5,
        anomaly_day=None,
        anomaly_duration=0,
        seed=seed,
        patient_id=f"easy_{seed}",
    )
    return PatientSimulator(cfg)


def make_medium_patient(seed: int = 0) -> PatientSimulator:
    """Task 2: 7-day series, random stage 1–3, agent must find anomaly day."""
    stage = (seed % 3) + 1
    cfg = PatientConfig(
        true_stage=stage,
        episode_length=7,
        decline_rate=0.0,
        noise_level=1.0,
        anomaly_day=None,
        anomaly_duration=2,
        seed=seed,
        patient_id=f"medium_{seed}",
    )
    return PatientSimulator(cfg)


def make_hard_patient(seed: int = 0) -> PatientSimulator:
    """Task 3: Full 30-step episode with drift and noise."""
    stage = (seed % 4) + 1
    cfg = PatientConfig(
        true_stage=stage,
        episode_length=30,
        decline_rate=0.01,
        noise_level=1.2,
        anomaly_day=None,
        anomaly_duration=5,
        seed=seed,
        patient_id=f"hard_{seed}",
    )
    return PatientSimulator(cfg)
