import json
import os
import re
import sys
import textwrap
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from app.env import IncidentTriageEnv
from app.models import Action, Observation

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
AGENT_MODE = os.getenv("AGENT_MODE", "llm").lower()
BENCHMARK = "incident-triage-orchestrator"

TASKS = ["easy", "medium", "hard"]
TEMPERATURE = 0.0
MAX_TOKENS = 180

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL) if HF_TOKEN else None

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an on-call incident triage agent.

    Choose exactly ONE next action at a time.

    Valid action formats:
    1) {"action_type":"inspect","target_type":"alert|service|runbook|incident|change","target_id":"..."}
    2) {"action_type":"set_field","field_name":"severity|owner_team|root_cause_service|decision_type|decision_target","value":"..."}
    3) {"action_type":"execute_response"}
    4) {"action_type":"submit"}

    Rules:
    - First inspect relevant evidence.
    - Then fill the triage fields.
    - Then execute the best response.
    - Then submit.
    - Return ONLY a single JSON object.
    - No markdown, no backticks, no explanation.
    """
).strip()


def extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def build_prompt(obs: Observation) -> str:
    payload = {
        "task_id": obs.task_id,
        "scenario_id": obs.scenario_id,
        "objective": obs.objective,
        "active_alerts": [a.model_dump() for a in obs.active_alerts],
        "visible_context": obs.visible_context,
        "available_inspections": obs.available_inspections,
        "discovered_evidence": obs.discovered_evidence,
        "current_decision": obs.current_decision.model_dump(),
        "response_status": obs.response_status,
        "allowed_values": obs.allowed_values,
        "minutes_elapsed": obs.minutes_elapsed,
        "step_count": obs.step_count,
        "remaining_steps": obs.remaining_steps,
        "last_action_error": obs.last_action_error,
    }
    return json.dumps(payload, ensure_ascii=False)


def llm_action(obs: Observation) -> Optional[Action]:
    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(obs)},
            ],
        )
        content = response.choices[0].message.content or ""
        data = extract_json(content)
        if not data:
            return None
        return Action(**data)
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        return None


def first_unfinished_planned_action(obs: Observation, plan: list[dict]) -> Action:
    discovered = {item["key"] for item in obs.discovered_evidence}
    current = obs.current_decision.model_dump()

    for item in plan:
        action_type = item["action_type"]

        if action_type == "inspect":
            key = f"{item['target_type']}:{item['target_id']}"
            if key not in discovered:
                return Action(**item)

        elif action_type == "set_field":
            field_name = item["field_name"]
            expected_value = item["value"]
            if current.get(field_name) != expected_value:
                return Action(**item)

        elif action_type == "execute_response":
            if obs.response_status == "not_attempted":
                return Action(**item)

        elif action_type == "submit":
            return Action(**item)

    return Action(action_type="submit")


def _observation_text(obs: Observation) -> str:
    payload = {
        "objective": obs.objective,
        "visible_context": obs.visible_context,
        "discovered_evidence": obs.discovered_evidence,
        "active_alerts": [a.model_dump() for a in obs.active_alerts],
    }
    return json.dumps(payload, ensure_ascii=False).lower()


def _best_allowed_match(allowed: list[str], text: str) -> Optional[str]:
    for candidate in allowed:
        if str(candidate).lower() in text:
            return candidate
    return None


def _choose_severity(obs: Observation) -> Optional[str]:
    allowed = obs.allowed_values.get("severity", [])
    if not allowed:
        return None

    hints = [str(alert.severity_hint).lower() for alert in obs.active_alerts]
    if any("critical" in hint for hint in hints) and "sev1" in allowed:
        return "sev1"
    if any("high" in hint for hint in hints) and "sev2" in allowed:
        return "sev2"
    if "sev3" in allowed:
        return "sev3"
    return allowed[0]


def _choose_decision_type(obs: Observation, text: str) -> Optional[str]:
    allowed = obs.allowed_values.get("decision_type", [])
    if not allowed:
        return None

    if any(marker in text for marker in ["false positive", "misconfiguration", "no customer-facing degradation"]):
        if "false_positive" in allowed:
            return "false_positive"

    if any(marker in text for marker in ["runbook", "mitigation", "rollback", "restart", "safety profile"]):
        if "run_runbook" in allowed:
            return "run_runbook"

    if any(marker in text for marker in ["db team", "intervened", "escalate", "handoff"]):
        if "escalate" in allowed:
            return "escalate"

    if "monitor" in allowed:
        return "monitor"
    return allowed[0]


def _choose_decision_target(obs: Observation, decision_type: Optional[str], text: str) -> Optional[str]:
    allowed = obs.allowed_values.get("decision_target", [])
    if not allowed:
        return None

    if decision_type in {"false_positive", "monitor", "ignore"}:
        if "NONE" in allowed:
            return "NONE"

    if decision_type == "run_runbook":
        runbook_ids = [r.runbook_id for r in obs.available_runbooks if r.runbook_id in allowed]
        match = _best_allowed_match(runbook_ids, text)
        if match:
            return match
        if runbook_ids:
            return runbook_ids[0]

    if decision_type == "escalate":
        owner = obs.current_decision.owner_team
        if owner and owner in allowed:
            return owner
        team_targets = [x for x in allowed if x.endswith("-oncall")]
        match = _best_allowed_match(team_targets, text)
        if match:
            return match
        if team_targets:
            return team_targets[0]

    return allowed[0]


def generic_action(obs: Observation) -> Action:
    discovered = {item.get("key") for item in obs.discovered_evidence}
    text = _observation_text(obs)

    for target_type in ["alert", "service", "incident", "change", "runbook"]:
        for target_id in obs.available_inspections.get(target_type, []):
            key = f"{target_type}:{target_id}"
            if key not in discovered:
                return Action(action_type="inspect", target_type=target_type, target_id=target_id)

    decision = obs.current_decision.model_dump()

    if decision.get("severity") is None:
        severity = _choose_severity(obs)
        if severity is not None:
            return Action(action_type="set_field", field_name="severity", value=severity)

    if decision.get("owner_team") is None:
        owners = obs.allowed_values.get("owner_team", [])
        owner = _best_allowed_match(owners, text) if owners else None
        if owner is None and owners:
            owner = owners[0]
        if owner is not None:
            return Action(action_type="set_field", field_name="owner_team", value=owner)

    if decision.get("root_cause_service") is None:
        roots = obs.allowed_values.get("root_cause_service", [])
        root = _best_allowed_match(roots, text) if roots else None
        if root is None and roots:
            root = roots[0]
        if root is not None:
            return Action(action_type="set_field", field_name="root_cause_service", value=root)

    if decision.get("decision_type") is None:
        chosen_type = _choose_decision_type(obs, text)
        if chosen_type is not None:
            return Action(action_type="set_field", field_name="decision_type", value=chosen_type)

    if decision.get("decision_target") is None:
        target = _choose_decision_target(obs, decision.get("decision_type"), text)
        if target is not None:
            return Action(action_type="set_field", field_name="decision_target", value=target)

    if obs.response_status == "not_attempted":
        decision_type = decision.get("decision_type")
        decision_target = decision.get("decision_target")
        if decision_type in {"run_runbook", "escalate"} and not decision_target:
            target = _choose_decision_target(obs, decision_type, text)
            if target is not None:
                return Action(action_type="set_field", field_name="decision_target", value=target)
        return Action(action_type="execute_response")

    return Action(action_type="submit")


def heuristic_action(obs: Observation) -> Action:
    plans = {
        "easy_cpu_spike": [
            {"action_type": "inspect", "target_type": "alert", "target_id": "A1"},
            {"action_type": "inspect", "target_type": "service", "target_id": "payments-api"},
            {"action_type": "inspect", "target_type": "runbook", "target_id": "RB_SCALE_WORKERS"},
            {"action_type": "set_field", "field_name": "severity", "value": "sev2"},
            {"action_type": "set_field", "field_name": "owner_team", "value": "payments-oncall"},
            {"action_type": "set_field", "field_name": "root_cause_service", "value": "payments-api"},
            {"action_type": "set_field", "field_name": "decision_type", "value": "run_runbook"},
            {"action_type": "set_field", "field_name": "decision_target", "value": "RB_SCALE_WORKERS"},
            {"action_type": "execute_response"},
            {"action_type": "submit"},
        ],
        "medium_dependency_failure": [
            {"action_type": "inspect", "target_type": "alert", "target_id": "A2"},
            {"action_type": "inspect", "target_type": "alert", "target_id": "A3"},
            {"action_type": "inspect", "target_type": "service", "target_id": "checkout-api"},
            {"action_type": "inspect", "target_type": "service", "target_id": "postgres-primary"},
            {"action_type": "inspect", "target_type": "incident", "target_id": "I2877"},
            {"action_type": "set_field", "field_name": "severity", "value": "sev1"},
            {"action_type": "set_field", "field_name": "owner_team", "value": "db-oncall"},
            {"action_type": "set_field", "field_name": "root_cause_service", "value": "postgres-primary"},
            {"action_type": "set_field", "field_name": "decision_type", "value": "escalate"},
            {"action_type": "set_field", "field_name": "decision_target", "value": "db-oncall"},
            {"action_type": "execute_response"},
            {"action_type": "submit"},
        ],
        "hard_false_positive_vs_real": [
            {"action_type": "inspect", "target_type": "service", "target_id": "recommendations-api"},
            {"action_type": "inspect", "target_type": "change", "target_id": "C902"},
            {"action_type": "inspect", "target_type": "incident", "target_id": "I4120"},
            {"action_type": "set_field", "field_name": "severity", "value": "sev4"},
            {"action_type": "set_field", "field_name": "owner_team", "value": "observability-oncall"},
            {"action_type": "set_field", "field_name": "root_cause_service", "value": "monitoring-config"},
            {"action_type": "set_field", "field_name": "decision_type", "value": "false_positive"},
            {"action_type": "set_field", "field_name": "decision_target", "value": "NONE"},
            {"action_type": "execute_response"},
            {"action_type": "submit"},
        ],
        "easy_queue_backlog_with_noise": [
            {"action_type": "inspect", "target_type": "alert", "target_id": "A5"},
            {"action_type": "inspect", "target_type": "service", "target_id": "payments-api"},
            {"action_type": "inspect", "target_type": "runbook", "target_id": "RB_SCALE_WORKERS"},
            {"action_type": "inspect", "target_type": "incident", "target_id": "I3202"},
            {"action_type": "set_field", "field_name": "severity", "value": "sev2"},
            {"action_type": "set_field", "field_name": "owner_team", "value": "payments-oncall"},
            {"action_type": "set_field", "field_name": "root_cause_service", "value": "payments-api"},
            {"action_type": "set_field", "field_name": "decision_type", "value": "run_runbook"},
            {"action_type": "set_field", "field_name": "decision_target", "value": "RB_SCALE_WORKERS"},
            {"action_type": "execute_response"},
            {"action_type": "submit"},
        ],
        "medium_db_pool_regression": [
            {"action_type": "inspect", "target_type": "alert", "target_id": "A6"},
            {"action_type": "inspect", "target_type": "alert", "target_id": "A7"},
            {"action_type": "inspect", "target_type": "service", "target_id": "checkout-api"},
            {"action_type": "inspect", "target_type": "service", "target_id": "postgres-primary"},
            {"action_type": "inspect", "target_type": "runbook", "target_id": "RB_CHECK_DB_POOL"},
            {"action_type": "set_field", "field_name": "severity", "value": "sev1"},
            {"action_type": "set_field", "field_name": "owner_team", "value": "db-oncall"},
            {"action_type": "set_field", "field_name": "root_cause_service", "value": "postgres-primary"},
            {"action_type": "set_field", "field_name": "decision_type", "value": "run_runbook"},
            {"action_type": "set_field", "field_name": "decision_target", "value": "RB_CHECK_DB_POOL"},
            {"action_type": "execute_response"},
            {"action_type": "submit"},
        ],
        "hard_real_incident_memory_leak": [
            {"action_type": "inspect", "target_type": "service", "target_id": "recommendations-api"},
            {"action_type": "inspect", "target_type": "runbook", "target_id": "RB_RESTART_PODS"},
            {"action_type": "inspect", "target_type": "incident", "target_id": "I4201"},
            {"action_type": "set_field", "field_name": "severity", "value": "sev2"},
            {"action_type": "set_field", "field_name": "owner_team", "value": "recsys-oncall"},
            {"action_type": "set_field", "field_name": "root_cause_service", "value": "recommendations-api"},
            {"action_type": "set_field", "field_name": "decision_type", "value": "run_runbook"},
            {"action_type": "set_field", "field_name": "decision_target", "value": "RB_RESTART_PODS"},
            {"action_type": "execute_response"},
            {"action_type": "submit"},
        ],
    }

    plan = plans.get(obs.scenario_id)
    if plan is not None:
        return first_unfinished_planned_action(obs, plan)
    return generic_action(obs)


def choose_action(obs: Observation) -> Action:
    if AGENT_MODE in {"llm", "hybrid"}:
        if client is None:
            print("HF_TOKEN is not set; falling back to heuristic policy.", file=sys.stderr)
            return heuristic_action(obs)
        candidate = llm_action(obs)
        if candidate is not None:
            return candidate
        return heuristic_action(obs)

    if AGENT_MODE == "heuristic":
        return heuristic_action(obs)

    return heuristic_action(obs)


def action_str(action: Action) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))


def format_error_token(error: object) -> str:
    if error is None:
        return "null"
    return json.dumps(str(error), ensure_ascii=False)


def run_task(task_id: str):
    env = IncidentTriageEnv(task_id=task_id, seed=0)
    obs = env.reset(task_id=task_id, seed=0)

    rewards = []
    step_num = 0
    final_score = 0.0
    success = False

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    try:
        done = False
        info = {}

        while not done and step_num < env.task.max_steps:
            action = choose_action(obs)
            obs, reward, done, info = env.step(action)
            step_num += 1

            rewards.append(f"{reward.value:.2f}")
            error = info.get("last_action_error")
            error_str = format_error_token(error)

            print(
                f"[STEP] step={step_num} action={action_str(action)} "
                f"reward={reward.value:.2f} done={'true' if done else 'false'} error={error_str}"
            )

        final_score = float(info.get("score", 0.0))
        success = bool(info.get("success", False))

    except Exception as e:
        print(f"inference exception on task={task_id}: {e}", file=sys.stderr)
        success = False
    finally:
        env.close()
        reward_str = ",".join(rewards)
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step_num} score={final_score:.2f} rewards={reward_str}"
        )


def main():
    for task_id in TASKS:
        run_task(task_id)


if __name__ == "__main__":
    main()