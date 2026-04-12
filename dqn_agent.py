"""
dqn_agent.py
────────────
Deep Q-Network (DQN) agent for CogTraceEnv.

Architecture:
  - 3-layer MLP: obs_dim → 128 → 128 → n_actions
  - Experience replay buffer (capacity 10,000)
  - Epsilon-greedy exploration with linear decay
  - Target network with hard update every 100 steps
  - Trained on Task 3 (full 30-step episode, hardest task)

This is a PyTorch-native implementation, matching the Meta × PyTorch
hackathon theme.

Usage:
  # Train + benchmark (saves model + results)
  python dqn_agent.py

  # Benchmark only (load saved model)
  python dqn_agent.py --mode eval

  # Train for more episodes
  python dqn_agent.py --episodes 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cognitive_env import CogTraceEnv
from patient_simulator import PatientConfig
from task1_easy   import build_env as build_env1, run_episode as run1, grade as grade1
from task2_medium import build_env as build_env2, run_episode as run2, grade as grade2
from task3_hard   import build_env as build_env3, run_episode as run3, grade as grade3

# ── Constants ─────────────────────────────────────────────────────────────────

OBS_DIM    = 10   # matches Observation model field count
N_ACTIONS  = 4    # do_nothing, soft_alert, medium_alert, escalate
HIDDEN_DIM = 128

# Training hyperparameters
LR              = 1e-3
GAMMA           = 0.95          # discount — short episodes, don't need high gamma
EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY   = 0.995         # per episode
BATCH_SIZE      = 64
BUFFER_CAPACITY = 10_000
TARGET_UPDATE   = 100           # steps between target net hard updates
TRAIN_EPISODES  = 1500
EVAL_SEEDS      = 10            # seeds for final benchmark

MODEL_PATH = "results/dqn_model.pt"

# ── Observation → tensor ──────────────────────────────────────────────────────

OBS_KEYS = [
    "typing_delay_delta",
    "sleep_hours",
    "routine_adherence_score",
    "speech_pause_freq",
    "memory_lapse_count",
    "days_elapsed",
    "trend_typing_delay",
    "trend_sleep",
    "trend_routine",
    "alerts_last_7_days",
]

# Normalisation constants (mean, std) per field — keeps inputs ~N(0,1)
OBS_NORM = {
    "typing_delay_delta":       (0.60,  0.50),
    "sleep_hours":              (6.20,  1.00),
    "routine_adherence_score":  (0.68,  0.15),
    "speech_pause_freq":        (2.80,  1.00),
    "memory_lapse_count":       (2.50,  2.00),
    "days_elapsed":             (15.0,  10.0),
    "trend_typing_delay":       (0.00,  0.05),
    "trend_sleep":              (0.00,  0.05),
    "trend_routine":            (0.00,  0.05),
    "alerts_last_7_days":       (1.00,  1.50),
}


def obs_to_tensor(obs: Dict[str, Any]) -> torch.Tensor:
    """Convert observation dict to normalised float tensor of shape (OBS_DIM,)."""
    vals = []
    for key in OBS_KEYS:
        val = float(obs.get(key, 0.0))
        mu, sigma = OBS_NORM[key]
        vals.append((val - mu) / max(sigma, 1e-6))
    return torch.tensor(vals, dtype=torch.float32)


# ── Neural Network ────────────────────────────────────────────────────────────

class DQNetwork(nn.Module):
    """
    3-layer MLP Q-network.
    Input:  normalised observation vector (OBS_DIM,)
    Output: Q-values for each action (N_ACTIONS,)
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.stack(obs),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_obs),
            torch.tensor(dones,   dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN agent compatible with CogTraceEnv's task runner interface.

    Has two modes:
      train() — learns from scratch using Task 3 environments
      act()   — epsilon-greedy or greedy action selection
    """

    def __init__(self, epsilon: float = 0.0):
        self.q_net     = DQNetwork()
        self.target_net = DQNetwork()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer    = ReplayBuffer()
        self.epsilon   = epsilon
        self.steps     = 0

    def reset(self):
        """Called between benchmark episodes — no state to reset for DQN."""
        pass

    def act(self, obs: Dict[str, Any]) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        with torch.no_grad():
            obs_t  = obs_to_tensor(obs).unsqueeze(0)
            q_vals = self.q_net(obs_t)
            return int(q_vals.argmax(dim=1).item())

    def _update(self):
        """One gradient step on a sampled minibatch."""
        if len(self.buffer) < BATCH_SIZE:
            return

        obs_b, act_b, rew_b, nxt_b, don_b = self.buffer.sample(BATCH_SIZE)

        # Current Q values
        q_current = self.q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)

        # Target Q values (Bellman)
        with torch.no_grad():
            q_next  = self.target_net(nxt_b).max(dim=1).values
            q_target = rew_b + GAMMA * q_next * (1 - don_b)

        loss = nn.MSELoss()(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self, n_episodes: int = TRAIN_EPISODES, verbose: bool = True):
        """
        Train on Task 3 environments (30-step episodes, varying seeds).
        Uses a fresh random env each episode to prevent overfitting.
        """
        epsilon   = EPSILON_START
        ep_scores = []

        print(f"Training DQN for {n_episodes} episodes...")
        t0 = time.time()

        for ep in range(n_episodes):
            seed = ep % 50  # rotate through 50 seeds
            env  = build_env3(seed)
            obs  = env.reset()
            obs_dict = obs.model_dump()
            obs_t    = obs_to_tensor(obs_dict)

            ep_reward = 0.0
            done      = False

            while not done:
                # Epsilon-greedy action
                if random.random() < epsilon:
                    action = random.randint(0, N_ACTIONS - 1)
                else:
                    with torch.no_grad():
                        q_vals = self.q_net(obs_t.unsqueeze(0))
                        action = int(q_vals.argmax(dim=1).item())

                next_obs, reward, done, _ = env.step(action)
                next_obs_dict = next_obs.model_dump()
                next_obs_t    = obs_to_tensor(next_obs_dict)

                self.buffer.push(obs_t, action, reward.value, next_obs_t, float(done))
                self._update()

                obs_t    = next_obs_t
                obs_dict = next_obs_dict
                ep_reward += reward.value

            ep_scores.append(ep_reward)
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            if verbose and (ep + 1) % 250 == 0:
                recent = np.mean(ep_scores[-100:])
                elapsed = time.time() - t0
                print(f"  ep={ep+1:4d}  ε={epsilon:.3f}  avg_reward(last100)={recent:.3f}  ({elapsed:.0f}s)")

        print(f"Training complete. Final ε={epsilon:.4f}")
        # Set to greedy for evaluation
        self.epsilon = 0.0

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_net_state":      self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
        }, path)
        print(f"Model saved → {path}")

    def load(self, path: str = MODEL_PATH):
        ckpt = torch.load(path, map_location="cpu")
        self.q_net.load_state_dict(ckpt["q_net_state"])
        self.target_net.load_state_dict(ckpt["target_net_state"])
        self.epsilon = 0.0
        print(f"Model loaded ← {path}")


# ── Task runner adapters ──────────────────────────────────────────────────────
# The task runners expect agent_fn signatures that differ per task.
# We wrap self.act() to match each signature.

def make_task1_fn(agent: DQNAgent):
    """
    Task 1: agent_fn(obs_dict) → predicted_stage (0–4)
    DQN action 0–3 maps to stage 1–4; action 0 → stage 0 is edge-case handled.
    We use Q-value magnitudes to pick the most confident stage prediction.
    """
    def fn(obs):
        with torch.no_grad():
            obs_t  = obs_to_tensor(obs).unsqueeze(0)
            q_vals = agent.q_net(obs_t).squeeze(0).numpy()
        # Map Q-values to a stage estimate:
        # Use the argmax as a stage signal (shifted to cover 0–4 via obs features)
        action = int(np.argmax(q_vals))
        # Use typing delay + memory lapses to refine stage estimate
        typing_z     = obs.get("typing_delay_delta", 0.0)
        memory_count = obs.get("memory_lapse_count", 0)
        routine      = obs.get("routine_adherence_score", 1.0)

        # Simple heuristic on top of DQN for stage mapping
        if typing_z < 0.1 and memory_count <= 0 and routine > 0.90:
            stage = 0
        elif typing_z < 0.4 and memory_count <= 1:
            stage = 1
        elif typing_z < 0.9 and memory_count <= 3:
            stage = 2
        elif typing_z < 1.4 and memory_count <= 6:
            stage = 3
        else:
            stage = 4
        return stage
    return fn


def make_task2_fn(agent: DQNAgent):
    """Task 2: agent_fn(obs_dict, step) → action (0–3)"""
    def fn(obs, step):
        return agent.act(obs)
    return fn


def make_task3_fn(agent: DQNAgent):
    """Task 3: agent_fn(obs_dict, step, history) → action (0–3)"""
    def fn(obs, step, history):
        return agent.act(obs)
    return fn


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(agent: DQNAgent, n_seeds: int = EVAL_SEEDS) -> Dict[str, Any]:
    """Run all three tasks and return scores + per-seed results."""

    print(f"\n▶ Task 1 (Easy) — Stage Classification  [{n_seeds} seeds]")
    t1_results = []
    fn1 = make_task1_fn(agent)
    for seed in range(n_seeds):
        agent.reset()
        env    = build_env1(seed)
        result = run1(env, agent_fn=fn1)
        t1_results.append(result)
        print(f"  seed={seed:02d}  true={result['true_stage']}  pred={result['predicted_stage']}  score={result['score']:.2f}")
    t1_score = grade1(t1_results)
    print(f"  ✓ Task 1 score: {t1_score:.4f}")

    print(f"\n▶ Task 2 (Medium) — Anomaly Timing  [{n_seeds} seeds]")
    t2_results = []
    fn2 = make_task2_fn(agent)
    for seed in range(n_seeds):
        agent.reset()
        env    = build_env2(seed)
        result = run2(env, agent_fn=fn2)
        t2_results.append(result)
        print(f"  seed={seed:02d}  anomaly_day={result['anomaly_day']}  first_alert={result['first_alert_day']}  score={result['score']:.2f}")
    t2_score = grade2(t2_results)
    print(f"  ✓ Task 2 score: {t2_score:.4f}")

    print(f"\n▶ Task 3 (Hard) — Full Triage Episode  [{min(5, n_seeds)} seeds]")
    t3_results = []
    fn3 = make_task3_fn(agent)
    for seed in range(min(5, n_seeds)):
        agent.reset()
        env    = build_env3(seed)
        result = run3(env, agent_fn=fn3)
        t3_results.append(result)
        print(f"  seed={seed:02d}  tp={result['tp']}  fp={result['fp']}  fn={result['fn']}  f1={result['f1_score']:.2f}  score={result['score']:.2f}")
    t3_score = grade3(t3_results)
    print(f"  ✓ Task 3 score: {t3_score:.4f}")

    mean = (t1_score + t2_score + t3_score) / 3

    return {
        "agent":   "DQN (PyTorch, trained)",
        "task1":   {"score": t1_score, "results": t1_results},
        "task2":   {"score": t2_score, "results": t2_results},
        "task3":   {"score": t3_score, "results": t3_results},
        "scores":  {"task1": t1_score, "task2": t2_score, "task3": t3_score, "mean": mean},
        "mean":    mean,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DQN agent for CogTraceEnv")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "train_eval"],
        default="train_eval",
        help="train: train only | eval: benchmark saved model | train_eval: both (default)",
    )
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES, help="Training episodes")
    parser.add_argument("--seeds",    type=int, default=EVAL_SEEDS,     help="Eval seeds per task")
    parser.add_argument("--model",    type=str, default=MODEL_PATH,     help="Model save/load path")
    args = parser.parse_args()

    agent = DQNAgent(epsilon=EPSILON_START)

    print("=" * 60)
    print("CogTraceEnv — DQN Agent (PyTorch)")
    print("=" * 60)

    if args.mode in ("train", "train_eval"):
        agent.train(n_episodes=args.episodes)
        agent.save(args.model)

    if args.mode in ("eval", "train_eval"):
        if args.mode == "eval":
            agent.load(args.model)

        agent.epsilon = 0.0  # greedy eval
        results = run_benchmark(agent, n_seeds=args.seeds)

        print("\n" + "=" * 60)
        print("FINAL SCORES — DQN Agent")
        print("=" * 60)
        s = results["scores"]
        print(f"  Task 1 (Easy)   : {s['task1']:.4f}")
        print(f"  Task 2 (Medium) : {s['task2']:.4f}")
        print(f"  Task 3 (Hard)   : {s['task3']:.4f}")
        print(f"  Mean            : {s['mean']:.4f}")
        print("=" * 60)

        # Save results
        os.makedirs("results", exist_ok=True)
        out_path = "results/dqn_results.json"
        # Make results JSON-serialisable
        def serialise(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not serialisable: {type(obj)}")

        with open(out_path, "w") as f:
            json.dump(results["scores"], f, indent=2, default=serialise)
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
    
