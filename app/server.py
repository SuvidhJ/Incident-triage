from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from app.env import IncidentTriageEnv
from app.loaders import list_tasks
from app.models import Action, EnvState, Observation, StepResponse

app = FastAPI(title="Incident Triage Orchestrator", version="0.1.0")
env = IncidentTriageEnv()

ENV_NAME = "incident-triage-orchestrator"
ENV_DESCRIPTION = (
    "Real-world AI incident triage environment where an agent gathers evidence, "
    "sets structured triage decisions, executes response actions, and optimizes for correctness and MTTR."
)


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 0


@app.get("/")
def root():
    return {
        "status": "ok",
        "env": ENV_NAME,
        "tasks": list_tasks(),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/tasks")
def tasks():
    return {"tasks": list_tasks()}


@app.get("/metadata")
def metadata():
    return {
        "name": ENV_NAME,
        "description": ENV_DESCRIPTION,
        "version": app.version,
        "mode": "simulation",
        "tasks": list_tasks(),
    }


@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": EnvState.model_json_schema(),
    }


@app.post("/mcp")
def mcp(payload: dict = Body(default_factory=dict)):
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "error": {
            "code": -32601,
            "message": "MCP methods are not implemented for this environment.",
        },
    }


@app.get("/openenv.yaml")
def openenv_manifest():
    return FileResponse(Path(__file__).resolve().parent.parent / "openenv.yaml")


@app.get("/reset", response_model=Observation)
def reset_get(task_id: str = Query("easy"), seed: int = Query(0)):
    return env.reset(task_id=task_id, seed=seed)


@app.post("/reset", response_model=Observation)
def reset_post(req: Optional[ResetRequest] = Body(default=None)):
    req = req or ResetRequest()
    return env.reset(task_id=req.task_id, seed=req.seed)


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    observation, reward, done, info = env.step(action)
    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=EnvState)
def state():
    return env.state()


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app.server:app", host=host, port=port)


if __name__ == "__main__":
    main()