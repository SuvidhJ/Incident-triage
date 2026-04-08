from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

ActionType = Literal["inspect", "set_field", "execute_response", "submit"]
TargetType = Literal["alert", "service", "runbook", "incident", "change"]
DecisionType = Literal["run_runbook", "escalate", "ignore", "false_positive", "monitor"]
FieldName = Literal["severity", "owner_team", "root_cause_service", "decision_type", "decision_target"]


class Alert(BaseModel):
    alert_id: str
    service: str
    metric: str
    severity_hint: str
    current_value: str
    threshold: str
    duration_minutes: int
    summary: str


class Runbook(BaseModel):
    runbook_id: str
    title: str
    summary: str


class WorkingDecision(BaseModel):
    severity: Optional[str] = None
    owner_team: Optional[str] = None
    root_cause_service: Optional[str] = None
    decision_type: Optional[str] = None
    decision_target: Optional[str] = None


class Action(BaseModel):
    action_type: ActionType
    target_type: Optional[TargetType] = None
    target_id: Optional[str] = None
    field_name: Optional[FieldName] = None
    value: Optional[str] = None
    note: Optional[str] = None


class InspectableItem(BaseModel):
    kind: str
    content: Dict[str, Any]


class ScenarioGroundTruth(BaseModel):
    severity: str
    owner_team: str
    root_cause_service: str
    decision_type: str
    decision_target: str
    expected_response_status: str
    key_evidence: List[str] = Field(default_factory=list)
    ideal_mttr_minutes: int = 10
    false_positive: bool = False


class ScenarioConfig(BaseModel):
    scenario_id: str
    title: str
    objective: str
    active_alerts: List[Alert]
    available_runbooks: List[Runbook]
    visible_context: Dict[str, Any]
    inspectables: Dict[str, InspectableItem]
    allowed_values: Dict[str, List[str]]
    ground_truth: ScenarioGroundTruth


class TaskConfig(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    scenario_ids: List[str]
    max_steps: int
    success_threshold: float = Field(ge=0.0, le=1.0)
    reward_weights: Dict[str, float]

    @model_validator(mode="after")
    def validate_reward_weights(self):
        expected_keys = {
            "evidence",
            "severity",
            "owner_team",
            "root_cause_service",
            "decision_type",
            "decision_target",
            "response_status",
        }
        actual_keys = set(self.reward_weights.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"reward_weights keys must be exactly {sorted(expected_keys)}; got {sorted(actual_keys)}"
            )
        total = sum(self.reward_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"reward_weights must sum to 1.0; got {total}")
        return self


class Observation(BaseModel):
    task_id: str
    scenario_id: str
    objective: str
    active_alerts: List[Alert]
    visible_context: Dict[str, Any]
    available_inspections: Dict[str, List[str]]
    discovered_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    available_runbooks: List[Runbook]
    current_decision: WorkingDecision
    response_status: str = "not_attempted"
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    allowed_values: Dict[str, List[str]]
    action_schema_help: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    minutes_elapsed: int = 0
    step_count: int = 0
    max_steps: int
    remaining_steps: int
    last_action_error: Optional[str] = None


class RewardModel(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    reason: str
    progress_score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)


class EnvState(BaseModel):
    task_id: str
    scenario_id: str
    done: bool
    step_count: int
    minutes_elapsed: int
    invalid_actions: int
    response_status: str
    hidden_ground_truth: Dict[str, Any]
    working_decision: Dict[str, Any]
    discovered_keys: List[str]
    score: float = Field(ge=0.0, le=1.0)
    score_breakdown: Dict[str, float]
    last_action_error: Optional[str] = None


class StepResponse(BaseModel):
    observation: Observation
    reward: RewardModel
    done: bool
    info: Dict[str, Any]
