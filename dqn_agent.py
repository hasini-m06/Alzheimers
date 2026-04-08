"""
dqn_agent.py — PyTorch DQN Agent for CogTraceEnv
Meta × PyTorch OpenEnv Hackathon · Scaler School of Technology

Trains a Dueling Double DQN on CogTraceEnv-v1 to issue calibrated clinical
alerts (do_nothing / soft_alert / medium_alert / escalate) from a 10-dim
behavioral observation of a simulated Alzheimer's patient.

Compatible with the real CogTraceEnv from this repo. Falls back to a
faithful simulation if the env can't be imported (e.g. standalone testing).

Usage:
    python dqn_agent.py                   # train + evaluate, saves model + results
    python dqn_agent.py --eval-only       # load saved model and evaluate
    python dqn_agent.py --episodes 2000   # train longer
    python dqn_agent.py --config early_onset
"""

from __future__ import annotations
import argparse
import json
import os
import random
import time
from collections import deque, namedtuple
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =============================================================================
# Environment wrapper — real CogTraceEnv or faithful fallback
# =============================================================================

def _try_import_real_env():
    """Try importing the actual CogTraceEnv from this repo."""
    try:
        from cognitive_env import CogTraceEnv
        from patient_simulator import PatientConfig, make_hard_patient
        return CogTraceEnv, PatientConfig, make_hard_patient, True
    except ImportError:
        return None, None, None, False

_CogTraceEnv, _PatientConfig, _make_hard_patient, ENV_AVAILABLE = _try_import_real_env()


class RealEnvWrapper:
    """
    Thin wrapper around the real CogTraceEnv to give a standard
    obs-as-numpy-array interface matching the DQN's input expectations.

    Observation order (10 floats) matches the README and patient_simulator.py:
        typing_delay_delta, sleep_hours, routine_adherence_score,
        speech_pause_freq, memory_lapse_count, days_elapsed,
        trend_typing_delay, trend_sleep, trend_routine, alerts_last_7_days
    """
    OBS_DIM = 10
    ACT_DIM = 4

    _OBS_KEYS = [
        "typing_delay_delta", "sleep_hours", "routine_adherence_score",
        "speech_pause_freq", "memory_lapse_count", "days_elapsed",
        "trend_typing_delay", "trend_sleep", "trend_routine", "alerts_last_7_days",
    ]

    def __init__(self, stage: int = 2, seed: Optional[int] = None):
        cfg = _PatientConfig(
            true_stage=stage,
            episode_length=30,
            decline_rate=0.01,
            noise_level=1.2,
            anomaly_day=None,
            anomaly_duration=5,
            seed=seed,
            patient_id=f"dqn_train_s{stage}",
        )
        self._env = _CogTraceEnv(config=cfg)

    def reset(self) -> np.ndarray:
        obs = self._env.reset()
        return self._obs_to_array(obs)

    def step(self, action: int):
        obs, reward, done, info = self._env.step(action)
        r = reward.value if hasattr(reward, "value") else float(reward)
        anomaly = info.anomaly_active if hasattr(info, "anomaly_active") else info.get("anomaly_active", False)
        return self._obs_to_array(obs), r, done, {"in_anomaly": anomaly}

    def _obs_to_array(self, obs) -> np.ndarray:
        if hasattr(obs, "model_dump"):          # Pydantic v2
            d = obs.model_dump()
        elif hasattr(obs, "dict"):              # Pydantic v1
            d = obs.dict()
        else:
            d = obs
        return np.array([float(d[k]) for k in self._OBS_KEYS], dtype=np.float32)


class SimulatedEnv:
    """
    Faithful simulation of CogTraceEnv Task 3 used when the real env
    can't be imported. Matches CDR profiles, reward table, and observation
    space from patient_simulator.py and cognitive_env.py exactly.
    """
    OBS_DIM = 10
    ACT_DIM = 4

    # CDR stage profiles from patient_simulator.py
    _CDR = {
        0: dict(typing=(0.00,0.15), sleep=(7.20,0.60), routine=(0.92,0.05), speech=(1.20,0.30), memory=(0.10,0.30)),
        1: dict(typing=(0.25,0.20), sleep=(6.80,0.70), routine=(0.83,0.07), speech=(1.80,0.40), memory=(0.80,0.60)),
        2: dict(typing=(0.60,0.25), sleep=(6.20,0.80), routine=(0.68,0.10), speech=(2.80,0.60), memory=(2.50,1.00)),
        3: dict(typing=(1.10,0.35), sleep=(5.50,1.00), routine=(0.48,0.12), speech=(4.20,0.80), memory=(5.00,1.50)),
        4: dict(typing=(1.80,0.40), sleep=(4.80,1.20), routine=(0.25,0.12), speech=(6.50,1.00), memory=(8.00,2.00)),
    }
    _ANOMALY_MUL = dict(typing=2.2, sleep=0.65, routine=0.55, speech=2.0, memory=2.5)

    # Reward table from cognitive_env.py
    _R = {
        "escalate_during_anomaly":   +1.00,
        "medium_alert_during_anomaly": +0.60,
        "soft_alert_during_anomaly":   +0.30,
        "do_nothing_during_anomaly":   -1.00,
        "do_nothing_no_anomaly":       +0.10,
        "soft_alert_no_anomaly":       -0.10,
        "medium_alert_no_anomaly":     -0.25,
        "escalate_no_anomaly":         -0.50,
        "spam_penalty":                -0.15,
    }
    _ACTION_NAMES = ["do_nothing", "soft_alert", "medium_alert", "escalate"]

    def __init__(self, stage: int = 2):
        self.stage = stage
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self) -> np.ndarray:
        self.day = 0
        onset = int(self.rng.integers(10, 21))
        self.anomaly_days = set(range(onset, min(onset + 5, 30)))
        self._typing_hist  = deque([0.0]*7, maxlen=7)
        self._sleep_hist   = deque([7.2]*7, maxlen=7)
        self._routine_hist = deque([0.92]*7, maxlen=7)
        self._alert_hist   = deque([0]*7, maxlen=7)
        return self._obs()

    def step(self, action: int):
        in_anomaly = self.day in self.anomaly_days
        alerts_7d  = sum(self._alert_hist)

        if in_anomaly:
            keys = ["escalate_during_anomaly","medium_alert_during_anomaly",
                    "soft_alert_during_anomaly","do_nothing_during_anomaly"]
        else:
            keys = ["escalate_no_anomaly","medium_alert_no_anomaly",
                    "soft_alert_no_anomaly","do_nothing_no_anomaly"]
        # action 3=escalate,2=medium,1=soft,0=nothing → index into keys list
        reward_key = keys[3 - action]
        r = self._R[reward_key]

        self._alert_hist.append(1 if action > 0 else 0)
        if sum(self._alert_hist) >= 4 and action > 0:
            r += self._R["spam_penalty"]

        self.day += 1
        done = self.day >= 30
        obs  = self._obs() if not done else np.zeros(self.OBS_DIM, dtype=np.float32)
        return obs, r, done, {"in_anomaly": in_anomaly}

    def _obs(self) -> np.ndarray:
        stage_f = min(4.0, self.stage + self.day * 0.01)
        lo, hi  = int(stage_f), min(int(stage_f)+1, 4)
        alpha   = stage_f - lo
        in_anom = self.day in self.anomaly_days

        def interp(key):
            mu_lo,sd_lo = self._CDR[lo][key]
            mu_hi,sd_hi = self._CDR[hi][key]
            return (1-alpha)*mu_lo+alpha*mu_hi, (1-alpha)*sd_lo+alpha*sd_hi

        def sample(mu, sd, lo_c=-np.inf, hi_c=np.inf):
            return float(np.clip(self.rng.normal(mu, sd*1.2), lo_c, hi_c))

        mu_t,sd_t = interp("typing");   typing  = sample(mu_t*(self._ANOMALY_MUL["typing"] if in_anom else 1), sd_t)
        mu_s,sd_s = interp("sleep");    sleep   = sample(mu_s*(self._ANOMALY_MUL["sleep"]  if in_anom else 1), sd_s, 0, 14)
        mu_r,sd_r = interp("routine");  routine = sample(mu_r*(self._ANOMALY_MUL["routine"]if in_anom else 1), sd_r, 0, 1)
        mu_sp,sd_sp = interp("speech"); speech  = sample(mu_sp*(self._ANOMALY_MUL["speech"]if in_anom else 1), sd_sp, 0)
        mu_m,sd_m = interp("memory");   memory  = max(0, round(sample(mu_m*(self._ANOMALY_MUL["memory"] if in_anom else 1), sd_m, 0)))

        self._typing_hist.append(typing)
        self._sleep_hist.append(sleep)
        self._routine_hist.append(routine)

        def slope(h):
            arr = np.array(list(h), dtype=float); x = np.arange(len(arr), dtype=float) - np.mean(np.arange(len(arr)))
            d = (x**2).sum(); return float((x*(arr-arr.mean())).sum()/d) if d > 0 else 0.0

        return np.array([
            typing, sleep, routine, speech, float(memory), float(self.day),
            slope(self._typing_hist), slope(self._sleep_hist), slope(self._routine_hist),
            float(sum(self._alert_hist)),
        ], dtype=np.float32)


def make_env(stage: int = 2, seed: Optional[int] = None):
    if ENV_AVAILABLE:
        return RealEnvWrapper(stage=stage, seed=seed)
    return SimulatedEnv(stage=stage)


# =============================================================================
# Dueling DQN Network
# =============================================================================

class CogNet(nn.Module):
    """
    Dueling DQN for CogTraceEnv.

    Separates the value stream V(s) from the advantage stream A(s,a).
    This is crucial here because `do_nothing` is the correct action for
    ~83% of timesteps, so standard DQN tends to overfit to passivity.
    The dueling decomposition lets the value head learn episode-level
    context independently of per-action advantage estimates.

        Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]
    """

    def __init__(self, obs_dim: int = 10, act_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, act_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        v    = self.value_head(feat)                        # (B, 1)
        a    = self.adv_head(feat)                          # (B, A)
        return v + (a - a.mean(dim=1, keepdim=True))        # dueling combine


# =============================================================================
# Replay Buffer
# =============================================================================

Transition = namedtuple("Transition", ["obs","action","reward","next_obs","done"])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs),  dtype=torch.float32),
            torch.tensor(act,            dtype=torch.long),
            torch.tensor(rew,            dtype=torch.float32),
            torch.tensor(np.array(nobs), dtype=torch.float32),
            torch.tensor(done,           dtype=torch.float32),
        )

    def __len__(self): return len(self.buf)


# =============================================================================
# Hyperparameters
# =============================================================================

@dataclass
class HParams:
    obs_dim:        int   = 10
    act_dim:        int   = 4
    # Training
    total_episodes: int   = 1_200
    batch_size:     int   = 64
    gamma:          float = 0.97       # high discount — future anomalies matter
    lr:             float = 3e-4
    hidden:         int   = 128
    # Exploration
    eps_start:      float = 1.0
    eps_end:        float = 0.05
    eps_decay:      int   = 600
    # Memory
    buffer_size:    int   = 50_000
    warmup_steps:   int   = 1_000
    target_update:  int   = 20
    # Eval
    eval_episodes:  int   = 100
    seed:           int   = 42
    # Paths
    model_path:     str   = "dqn_cogtraceenv.pt"
    results_path:   str   = "results/dqn_results.json"


# =============================================================================
# Agent
# =============================================================================

class DQNAgent:
    def __init__(self, hp: HParams, device: torch.device):
        self.hp     = hp
        self.device = device
        self.policy_net = CogNet(hp.obs_dim, hp.act_dim, hp.hidden).to(device)
        self.target_net = CogNet(hp.obs_dim, hp.act_dim, hp.hidden).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=hp.lr)
        self.buffer    = ReplayBuffer(hp.buffer_size)
        self.steps     = 0

    def select_action(self, obs: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.hp.act_dim)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy_net(t).argmax(dim=1).item())

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.hp.warmup_steps:
            return None
        obs, act, rew, nobs, done = [t.to(self.device) for t in
                                      self.buffer.sample(self.hp.batch_size)]
        q_cur  = self.policy_net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: policy net selects action, target net evaluates it
            best_a  = self.policy_net(nobs).argmax(dim=1, keepdim=True)
            q_next  = self.target_net(nobs).gather(1, best_a).squeeze(1)
            q_tgt   = rew + self.hp.gamma * q_next * (1 - done)
        loss = F.smooth_l1_loss(q_cur, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.steps += 1
        return float(loss.item())

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"policy_state_dict": self.policy_net.state_dict(),
                    "hp": asdict(self.hp)}, path)
        print(f"  ✓ Model saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt["policy_state_dict"])
        print(f"  ✓ Model loaded ← {path}")


# =============================================================================
# Training
# =============================================================================

def train(hp: HParams, device: torch.device) -> DQNAgent:
    random.seed(hp.seed); np.random.seed(hp.seed); torch.manual_seed(hp.seed)

    # Train across all CDR stages so the agent generalises
    stages = [1, 2, 2, 3]   # oversample stage 2 (most clinically relevant)

    print(f"\n{'='*60}")
    print(f"  CogTraceEnv Dueling Double DQN — Training")
    print(f"  Device  : {device}")
    print(f"  Env     : {'CogTraceEnv (real)' if ENV_AVAILABLE else 'SimulatedEnv (fallback)'}")
    print(f"  Stages  : {stages} (cycling)")
    print(f"  Episodes: {hp.total_episodes}")
    print(f"{'='*60}\n")

    agent  = DQNAgent(hp, device)
    ep_rewards, losses = [], []
    t0 = time.time()

    for ep in range(1, hp.total_episodes + 1):
        frac = min(ep / hp.eps_decay, 1.0)
        eps  = hp.eps_start + frac * (hp.eps_end - hp.eps_start)

        stage = stages[(ep - 1) % len(stages)]
        env   = make_env(stage=stage, seed=ep)
        obs   = env.reset()
        ep_r  = 0.0

        while True:
            action           = agent.select_action(obs, eps)
            nobs, r, done, _ = env.step(action)
            agent.buffer.push(obs, action, r, nobs, float(done))
            loss = agent.learn()
            if loss is not None: losses.append(loss)
            obs   = nobs
            ep_r += r
            if done: break

        ep_rewards.append(ep_r)
        if ep % hp.target_update == 0:
            agent.sync_target()

        if ep % 100 == 0:
            avg_r    = np.mean(ep_rewards[-100:])
            avg_loss = np.mean(losses[-200:]) if losses else float("nan")
            print(f"  Ep {ep:>5d} | ε={eps:.3f} | "
                  f"Avg reward (100)={avg_r:+.3f} | "
                  f"Loss={avg_loss:.4f} | "
                  f"Elapsed={time.time()-t0:.0f}s")

    print(f"\n  Training complete. Final 100-ep avg: {np.mean(ep_rewards[-100:]):+.3f}")
    agent.save(hp.model_path)
    return agent


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(agent: DQNAgent, hp: HParams) -> dict:
    """Greedy evaluation across all CDR stages. Reports F1 + Task 3 score."""
    stages = {
        "stage_1_early_onset":    1,
        "stage_2_moderate":       2,
        "stage_3_rapid_decliner": 3,
    }
    results = {}

    print(f"\n{'='*60}")
    print(f"  Evaluation ({hp.eval_episodes} episodes × {len(stages)} configs)")
    print(f"{'='*60}")

    for name, stage in stages.items():
        ep_rewards = []
        tp = fp = fn = 0

        for seed in range(hp.eval_episodes):
            env = make_env(stage=stage, seed=seed + 10_000)
            obs = env.reset()
            ep_r = 0.0
            while True:
                with torch.no_grad():
                    t      = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    action = int(agent.policy_net(t).argmax(dim=1).item())
                obs, r, done, info = env.step(action)
                ep_r += r
                in_anom = info.get("in_anomaly", False)
                alerted = action > 0
                if in_anom and alerted:      tp += 1
                if not in_anom and alerted:  fp += 1
                if in_anom and not alerted:  fn += 1
                if done: break
            ep_rewards.append(ep_r)

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        mean_r    = float(np.mean(ep_rewards))
        level_bon = float(np.clip(mean_r / 30, 0, 1))
        task3     = round(0.70 * f1 + 0.30 * level_bon, 4)

        results[name] = {
            "cdr_stage": stage,
            "mean_reward": round(mean_r, 4),
            "f1":          round(f1, 4),
            "precision":   round(precision, 4),
            "recall":      round(recall, 4),
            "task3_score": task3,
        }

        print(f"\n  {name}")
        print(f"    Reward   : {mean_r:+.3f}")
        print(f"    F1       : {f1:.3f}  (P={precision:.3f}  R={recall:.3f})")
        print(f"    Task3    : {task3:.3f}")

    mean_t3 = float(np.mean([v["task3_score"] for v in results.values()]))
    results["mean_task3_score"] = round(mean_t3, 4)

    print(f"\n  ── Benchmark ────────────────────────────────────────")
    print(f"  Random baseline          : 0.28")
    print(f"  Always-alert baseline    : 0.19")
    print(f"  Llama-3.1-8B (zero-shot) : 0.35")
    print(f"  DQN agent (ours)         : {mean_t3:.3f}  ◄")
    print(f"  Rule-based oracle        : 0.82")
    print(f"  {'='*48}")
    return results


# =============================================================================
# Save results
# =============================================================================

def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "agent": "DQN (Dueling + Double Q-Learning)",
        "architecture": "CogNet — LayerNorm MLP + dueling V/A heads",
        "framework": "PyTorch",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env_version": "CogTraceEnv-v1",
        "results": results,
        "baseline_comparison": {
            "random_baseline":          0.28,
            "always_alert_baseline":    0.19,
            "llama_3_1_8b_zeroshot":    0.35,
            "dqn_agent":                results.get("mean_task3_score"),
            "rule_based_oracle":        0.82,
        }
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  ✓ Results saved → {path}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CogTraceEnv DQN Agent")
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--episodes",   type=int,  default=None)
    parser.add_argument("--config",     type=str,  default="rapid_decliner",
                        choices=["rapid_decliner","early_onset","stable_elderly"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = HParams()
    if args.episodes:
        hp.total_episodes = args.episodes

    agent = DQNAgent(hp, device)

    if args.eval_only:
        if not os.path.exists(hp.model_path):
            print(f"  No saved model at {hp.model_path}. Run training first.")
            return
        agent.load(hp.model_path)
    else:
        agent = train(hp, device)

    results = evaluate(agent, hp)
    save_results(results, hp.results_path)


if __name__ == "__main__":
    main()
    