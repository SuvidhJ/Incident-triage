from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.graders import score_progress
from app.loaders import load_scenario, load_task
from app.models import Action, EnvState, Observation, RewardModel, WorkingDecision

ACTION_COSTS = {
    "inspect": 2,
    "set_field": 0,
    "execute_response": 4,
    "submit": 0,
}


class IncidentTriageEnv:
    def __init__(self, task_id: str = "easy", seed: int = 0):
        self.task_id = task_id
        self.seed = seed
        self.task = None
        self.scenario = None
        self.done = False
        self.step_count = 0
        self.minutes_elapsed = 0
        self.invalid_actions = 0
        self.last_action_error = None
        self.discovered_keys: list[str] = []
        self.discovered_evidence: list[dict[str, Any]] = []
        self.action_history: list[dict[str, Any]] = []
        self.response_status = "not_attempted"
        self.working_decision: dict[str, Any] = {}
        self.runtime_visible_context = {}
        self.reset(task_id=task_id, seed=seed)

    def reset(self, task_id: str | None = None, seed: int | None = None) -> Observation:
        if task_id is not None:
            self.task_id = task_id
        if seed is not None:
            self.seed = seed

        self.task = load_task(self.task_id)
        scenario_id = self.task.scenario_ids[self.seed % len(self.task.scenario_ids)]
        self.scenario = load_scenario(scenario_id)
        self.runtime_visible_context = deepcopy(self.scenario.visible_context)

        self.done = False
        self.step_count = 0
        self.minutes_elapsed = 0
        self.invalid_actions = 0
        self.last_action_error = None
        self.discovered_keys = []
        self.discovered_evidence = []
        self.action_history = []
        self.response_status = "not_attempted"
        self.working_decision = {
            "severity": None,
            "owner_team": None,
            "root_cause_service": None,
            "decision_type": None,
            "decision_target": None,
        }

        return self._observation()

    def close(self):
        return None

    def state(self) -> EnvState:
        score, breakdown = self._current_score()
        return EnvState(
            task_id=self.task.task_id,
            scenario_id=self.scenario.scenario_id,
            done=self.done,
            step_count=self.step_count,
            minutes_elapsed=self.minutes_elapsed,
            invalid_actions=self.invalid_actions,
            response_status=self.response_status,
            hidden_ground_truth=self.scenario.ground_truth.model_dump(),
            working_decision=deepcopy(self.working_decision),
            discovered_keys=deepcopy(self.discovered_keys),
            score=score,
            score_breakdown=breakdown,
            last_action_error=self.last_action_error,
        )

    def step(self, action: Action | dict):
        if isinstance(action, dict):
            action = Action(**action)

        if self.done:
            score, breakdown = self._current_score()
            reward = RewardModel(
                value=0.0,
                reason="episode_already_done",
                progress_score=score,
                breakdown=breakdown,
            )
            return self._observation(), reward, True, {
                "score": score,
                "score_breakdown": breakdown,
                "last_action_error": self.last_action_error,
                "success": score >= self.task.success_threshold,
            }

        prev_score, _ = self._current_score()
        self.last_action_error = None

        self.action_history.append(action.model_dump(exclude_none=True))
        self._apply_action(action)

        self.step_count += 1
        self.minutes_elapsed += ACTION_COSTS.get(action.action_type, 0)

        if action.action_type == "submit" or self.step_count >= self.task.max_steps:
            self.done = True

        curr_score, breakdown = self._current_score()

        reward_value = max(0.0, round(curr_score - prev_score, 4))
        reason = "invalid_action" if self.last_action_error else ("progress" if curr_score > prev_score else "no_progress")

        if self.done:
            reward_value = max(reward_value, curr_score)
            reason = "episode_complete"

        reward = RewardModel(
            value=min(1.0, reward_value),
            reason=reason,
            progress_score=curr_score,
            breakdown=breakdown,
        )

        info = {
            "score": curr_score,
            "score_breakdown": breakdown,
            "last_action_error": self.last_action_error,
            "success": curr_score >= self.task.success_threshold,
        }

        return self._observation(), reward, self.done, info

    def _current_score(self):
        return score_progress(
            task=self.task,
            scenario=self.scenario,
            working_decision=self.working_decision,
            discovered_keys=self.discovered_keys,
            response_status=self.response_status,
            invalid_actions=self.invalid_actions,
            minutes_elapsed=self.minutes_elapsed,
        )

    def _apply_action(self, action: Action):
        if action.action_type == "inspect":
            self._inspect(action.target_type, action.target_id)
            return

        if action.action_type == "set_field":
            self._set_field(action.field_name, action.value)
            return

        if action.action_type == "execute_response":
            self._execute_response()
            return

        if action.action_type == "submit":
            return

        self._invalid(f"unsupported action_type: {action.action_type}")

    def _inspect(self, target_type: str | None, target_id: str | None):
        if not target_type or not target_id:
            self._invalid("inspect requires both target_type and target_id")
            return

        key = f"{target_type}:{target_id}"
        item = self.scenario.inspectables.get(key)
        if item is None:
            self._invalid(f"unknown inspect target: {key}")
            return

        if key in self.discovered_keys:
            self._invalid(f"evidence already discovered: {key}")
            return

        self.discovered_keys.append(key)
        self.discovered_evidence.append(
            {
                "key": key,
                "kind": item.kind,
                "content": deepcopy(item.content),
            }
        )

    def _set_field(self, field_name: str | None, value: str | None):
        if field_name not in self.working_decision:
            self._invalid(f"invalid field_name: {field_name}")
            return

        if value is None:
            self._invalid(f"value is required for field '{field_name}'")
            return

        value = str(value)
        allowed = self.scenario.allowed_values.get(field_name, [])
        if allowed and value not in allowed:
            self._invalid(f"invalid value '{value}' for field '{field_name}'")
            return

        self.working_decision[field_name] = value

    def _execute_response(self):
        if self.response_status != "not_attempted":
            self._invalid("response has already been executed for this episode")
            return

        gt = self.scenario.ground_truth
        decision_type = self.working_decision.get("decision_type")
        decision_target = self.working_decision.get("decision_target") or "NONE"

        if decision_type is None:
            self._invalid("decision_type must be set before execute_response")
            return

        if decision_type in {"run_runbook", "escalate"} and decision_target == "NONE":
            self._invalid("decision_target must be set before execute_response")
            return

        if decision_type == "run_runbook":
            if gt.decision_type == "run_runbook" and decision_target == gt.decision_target:
                self.response_status = "mitigation_applied"
                message = "Correct mitigation runbook executed; system recovery should begin."
            else:
                self.response_status = "wrong_runbook"
                message = "Selected runbook does not address the real incident."
        elif decision_type == "escalate":
            if gt.decision_type == "escalate" and decision_target == gt.decision_target:
                self.response_status = "escalated_correctly"
                message = "Incident escalated to the correct owner team."
            else:
                self.response_status = "escalated_wrong_team"
                message = "Incident escalated, but to the wrong team."
        elif decision_type == "false_positive":
            if gt.decision_type == "false_positive":
                self.response_status = "closed_as_false_positive"
                message = "Alert closed as false positive."
            else:
                self.response_status = "closed_incorrectly"
                message = "Real incident was incorrectly closed as false positive."
        elif decision_type == "monitor":
            if gt.decision_type == "monitor":
                self.response_status = "monitoring_only"
                message = "Incident placed under monitoring."
            else:
                self.response_status = "underreacted"
                message = "Monitoring was insufficient for this incident."
        elif decision_type == "ignore":
            if gt.decision_type == "ignore":
                self.response_status = "ignored_without_action"
                message = "Alert ignored."
            else:
                self.response_status = "underreacted"
                message = "Alert was ignored even though action was required."
        else:
            self._invalid(f"unsupported decision_type: {decision_type}")
            return

        self._set_or_replace_response_effect(message)
        self._apply_runtime_outcome()

    def _apply_runtime_outcome(self):
        if self.response_status == "mitigation_applied":
            self.runtime_visible_context["system_status"] = "recovering"
            self.runtime_visible_context["post_action_summary"] = (
                "Correct mitigation applied. Latency and error rate should begin improving."
            )
        elif self.response_status == "wrong_runbook":
            self.runtime_visible_context["system_status"] = "still_degraded"
            self.runtime_visible_context["post_action_summary"] = (
                "Runbook executed, but no meaningful improvement is expected."
            )
        elif self.response_status == "escalated_correctly":
            self.runtime_visible_context["system_status"] = "handoff_in_progress"
            self.runtime_visible_context["post_action_summary"] = (
                "Incident escalated to the correct team for faster resolution."
            )
        elif self.response_status == "escalated_wrong_team":
            self.runtime_visible_context["system_status"] = "handoff_delayed"
            self.runtime_visible_context["post_action_summary"] = (
                "Incident was escalated, but to the wrong team."
            )
        elif self.response_status == "closed_as_false_positive":
            self.runtime_visible_context["system_status"] = "alert_suppressed"
            self.runtime_visible_context["post_action_summary"] = (
                "Alert closed as false positive; customer impact remains absent."
            )
        elif self.response_status == "closed_incorrectly":
            self.runtime_visible_context["system_status"] = "real_incident_missed"
            self.runtime_visible_context["post_action_summary"] = (
                "A real incident was incorrectly dismissed."
            )
        elif self.response_status == "monitoring_only":
            self.runtime_visible_context["system_status"] = "under_observation"
            self.runtime_visible_context["post_action_summary"] = (
                "Incident placed under monitoring."
            )
        elif self.response_status == "underreacted":
            self.runtime_visible_context["system_status"] = "still_degraded"
            self.runtime_visible_context["post_action_summary"] = (
                "Chosen response was insufficient for the actual incident."
            )
        elif self.response_status == "ignored_without_action":
            self.runtime_visible_context["system_status"] = "ignored"
            self.runtime_visible_context["post_action_summary"] = "Alert ignored."

    def _set_or_replace_response_effect(self, message: str):
        effect_key = "system:response_effect"
        effect_payload = {
            "key": effect_key,
            "kind": "response_effect",
            "content": {
                "response_status": self.response_status,
                "message": message,
            },
        }

        self.discovered_evidence = [x for x in self.discovered_evidence if x.get("key") != effect_key]
        self.discovered_evidence.append(effect_payload)

        if effect_key not in self.discovered_keys:
            self.discovered_keys.append(effect_key)

    def _invalid(self, msg: str):
        self.invalid_actions += 1
        self.last_action_error = msg

    def _available_inspections(self) -> dict[str, list[str]]:
        out = {
            "alert": [],
            "service": [],
            "runbook": [],
            "incident": [],
            "change": [],
        }
        for key in self.scenario.inspectables.keys():
            if ":" not in key:
                continue
            prefix, identifier = key.split(":", 1)
            if prefix in out:
                out[prefix].append(identifier)
        for k in out:
            out[k] = sorted(out[k])
        return out

    def _observation(self) -> Observation:
        return Observation(
            task_id=self.task.task_id,
            scenario_id=self.scenario.scenario_id,
            objective=self.scenario.objective,
            active_alerts=deepcopy(self.scenario.active_alerts),
            visible_context=deepcopy(self.runtime_visible_context),
            available_inspections=self._available_inspections(),
            discovered_evidence=deepcopy(self.discovered_evidence),
            available_runbooks=deepcopy(self.scenario.available_runbooks),
            current_decision=WorkingDecision(**self.working_decision),
            response_status=self.response_status,
            action_history=deepcopy(self.action_history),
            allowed_values=deepcopy(self.scenario.allowed_values),
            action_schema_help=[
                "{\"action_type\":\"inspect\",\"target_type\":\"service\",\"target_id\":\"payments-api\"}",
                "{\"action_type\":\"set_field\",\"field_name\":\"severity\",\"value\":\"sev2\"}",
                "{\"action_type\":\"execute_response\"}",
                "{\"action_type\":\"submit\"}",
            ],
            success_criteria=[
                "Discover relevant evidence before deciding.",
                "Set the correct severity, owner team, and root cause service.",
                "Choose the correct decision type and decision target.",
                "Execute a response when action is required.",
                "Minimize minutes elapsed and avoid invalid actions.",
            ],
            minutes_elapsed=self.minutes_elapsed,
            step_count=self.step_count,
            max_steps=self.task.max_steps,
            remaining_steps=max(0, self.task.max_steps - self.step_count),
            last_action_error=self.last_action_error,
        )
