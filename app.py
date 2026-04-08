"""
app.py
──────
FastAPI application serving CogTraceEnv as an OpenEnv HTTP API.

Endpoints:
  GET  /             → serves demo.html (interactive UI)
  POST /reset        → Observation
  POST /step         → {observation, reward, done, info}
  GET  /state        → EnvState
  GET  /tasks        → list of available tasks
  GET  /health       → {"status": "ok"}
  GET  /openenv.yaml → serve the spec file
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from cognitive_env import CogTraceEnv
from patient_simulator import PatientConfig
from models import Action

app = FastAPI(
    title="CogTraceEnv",
    description="OpenEnv environment for Alzheimer's cognitive monitoring",
    version="1.0.0",
)

# Global env instance (single-session server)
_env: Optional[CogTraceEnv] = None

# ─── Request/Response models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    true_stage: Optional[int] = None   # 0–4; None = random
    episode_length: int = 30
    decline_rate: float = 0.01
    noise_level: float = 1.0
    seed: Optional[int] = None
    patient_id: str = "patient_001"
    anomaly_day: Optional[int] = None
    anomaly_duration: int = 5

class StepRequest(BaseModel):
    action: int  # 0–3

# ─── UI route ─────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    """Serve the interactive monitoring demo UI."""
    html_path = os.path.join(os.path.dirname(__file__), "demo.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="demo.html not found")
    return FileResponse(html_path, media_type="text/html")

# ─── API routes ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "CogTraceEnv-v1"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env

    stage = req.true_stage if req.true_stage is not None else random.randint(1, 3)

    cfg = PatientConfig(
        true_stage=stage,
        episode_length=req.episode_length,
        decline_rate=req.decline_rate,
        noise_level=req.noise_level,
        seed=req.seed,
        patient_id=req.patient_id,
        anomaly_day=req.anomaly_day,
        anomaly_duration=req.anomaly_duration,
    )

    _env = CogTraceEnv(config=cfg)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        action = Action(action=req.action)
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward":      reward.model_dump(),
            "done":        done,
            "info":        info.model_dump(),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env.state().model_dump()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task1_easy",
                "name": "Cognitive Stage Classification",
                "difficulty": "easy",
                "description": (
                    "Given one snapshot of behavioral metrics, "
                    "predict the patient's Alzheimer's stage (0–4)."
                ),
            },
            {
                "id": "task2_medium",
                "name": "Anomaly Timing Detection",
                "difficulty": "medium",
                "description": (
                    "Observe 7 days of signals. Raise an alert on "
                    "the day you detect an anomaly."
                ),
            },
            {
                "id": "task3_hard",
                "name": "Full Triage Episode",
                "difficulty": "hard",
                "description": (
                    "Manage a 30-step episode, balancing sensitivity "
                    "and specificity across declining patient trajectories."
                ),
            },
        ]
    }


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def serve_yaml():
    yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    if not os.path.exists(yaml_path):
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    with open(yaml_path) as f:
        return f.read()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)