from __future__ import annotations

import json
from pathlib import Path

from app.models import ScenarioConfig, TaskConfig

BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIO_DIR = BASE_DIR / "data" / "scenarios"
TASK_DIR = BASE_DIR / "data" / "tasks"


def _validate_scenario_integrity(config: ScenarioConfig) -> None:
    inspectable_keys = set(config.inspectables.keys())
    for key in config.ground_truth.key_evidence:
        if key.startswith("system:"):
            continue
        if key not in inspectable_keys:
            raise ValueError(f"Scenario '{config.scenario_id}' has unknown key_evidence '{key}'")

    gt = config.ground_truth
    for field_name, value in {
        "severity": gt.severity,
        "owner_team": gt.owner_team,
        "root_cause_service": gt.root_cause_service,
        "decision_type": gt.decision_type,
        "decision_target": gt.decision_target,
    }.items():
        allowed = config.allowed_values.get(field_name, [])
        if allowed and value not in allowed:
            raise ValueError(
                f"Scenario '{config.scenario_id}' has ground truth value '{value}' outside allowed_values['{field_name}']"
            )


def load_scenario(scenario_id: str) -> ScenarioConfig:
    path = SCENARIO_DIR / f"{scenario_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = ScenarioConfig(**data)
    _validate_scenario_integrity(config)
    return config


def load_task(task_id: str) -> TaskConfig:
    path = TASK_DIR / f"{task_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    task = TaskConfig(**data)
    if not task.scenario_ids:
        raise ValueError(f"Task '{task.task_id}' must define at least one scenario_id")
    return task


def list_tasks() -> list[str]:
    preferred = ["easy", "medium", "hard"]
    existing = {p.stem for p in TASK_DIR.glob("*.json")}
    ordered = [x for x in preferred if x in existing]
    extras = sorted(existing - set(preferred))
    return ordered + extras
