from __future__ import annotations

import json
from pathlib import Path

from app.models import ScenarioConfig, TaskConfig

BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIO_DIR = BASE_DIR / "data" / "scenarios"
TASK_DIR = BASE_DIR / "data" / "tasks"


def load_scenario(scenario_id: str) -> ScenarioConfig:
    path = SCENARIO_DIR / f"{scenario_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ScenarioConfig(**data)


def load_task(task_id: str) -> TaskConfig:
    path = TASK_DIR / f"{task_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TaskConfig(**data)


def list_tasks() -> list[str]:
    preferred = ["easy", "medium", "hard"]
    existing = {p.stem for p in TASK_DIR.glob("*.json")}
    ordered = [x for x in preferred if x in existing]
    extras = sorted(existing - set(preferred))
    return ordered + extras
