from app.env import IncidentTriageEnv
from app.models import Action


def test_reset_returns_valid_observation():
    env = IncidentTriageEnv(task_id="easy", seed=0)
    obs = env.reset(task_id="easy", seed=0)
    assert obs.task_id == "easy"
    assert obs.scenario_id == "easy_cpu_spike"
    assert obs.step_count == 0
    assert obs.minutes_elapsed == 0


def test_invalid_inspect_sets_error():
    env = IncidentTriageEnv(task_id="easy", seed=0)
    env.reset(task_id="easy", seed=0)
    obs, reward, done, info = env.step(
        Action(action_type="inspect", target_type="service", target_id="does-not-exist")
    )
    assert done is False
    assert info["last_action_error"] is not None
    assert reward.value == 0.0


def test_easy_perfect_run_reaches_full_score():
    env = IncidentTriageEnv(task_id="easy", seed=0)
    env.reset(task_id="easy", seed=0)

    actions = [
        {"action_type": "inspect", "target_type": "alert", "target_id": "A1"},
        {"action_type": "inspect", "target_type": "service", "target_id": "payments-api"},
        {"action_type": "inspect", "target_type": "runbook", "target_id": "RB_SCALE_WORKERS"},
        {"action_type": "set_field", "field_name": "severity", "value": "sev2"},
        {"action_type": "set_field", "field_name": "owner_team", "value": "payments-oncall"},
        {"action_type": "set_field", "field_name": "root_cause_service", "value": "payments-api"},
        {"action_type": "set_field", "field_name": "decision_type", "value": "run_runbook"},
        {"action_type": "set_field", "field_name": "decision_target", "value": "RB_SCALE_WORKERS"},
        {"action_type": "execute_response"},
        {"action_type": "submit"}
    ]

    done = False
    info = {}
    for action in actions:
        _, _, done, info = env.step(Action(**action))

    assert done is True
    assert info["score"] == 1.0
    assert info["success"] is True


def test_duplicate_inspect_is_penalized():
    env = IncidentTriageEnv(task_id="easy", seed=0)
    env.reset(task_id="easy", seed=0)

    env.step(Action(action_type="inspect", target_type="alert", target_id="A1"))
    _, reward, _, info = env.step(Action(action_type="inspect", target_type="alert", target_id="A1"))

    assert reward.value == 0.0
    assert info["last_action_error"] is not None


def test_execute_response_twice_is_invalid():
    env = IncidentTriageEnv(task_id="easy", seed=0)
    env.reset(task_id="easy", seed=0)

    env.step(Action(action_type="set_field", field_name="decision_type", value="run_runbook"))
    env.step(Action(action_type="set_field", field_name="decision_target", value="RB_SCALE_WORKERS"))
    env.step(Action(action_type="execute_response"))
    _, reward, _, info = env.step(Action(action_type="execute_response"))

    assert reward.value == 0.0
    assert info["last_action_error"] is not None


def test_medium_perfect_run_reaches_full_score():
    env = IncidentTriageEnv(task_id="medium", seed=0)
    env.reset(task_id="medium", seed=0)

    actions = [
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
    ]

    done = False
    info = {}
    for action in actions:
        _, _, done, info = env.step(Action(**action))

    assert done is True
    assert info["score"] == 1.0
    assert info["success"] is True


def test_hard_perfect_run_reaches_full_score():
    env = IncidentTriageEnv(task_id="hard", seed=0)
    env.reset(task_id="hard", seed=0)

    actions = [
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
    ]

    done = False
    info = {}
    for action in actions:
        _, _, done, info = env.step(Action(**action))

    assert done is True
    assert info["score"] == 1.0
    assert info["success"] is True


def test_reset_is_deterministic_for_same_seed():
    env = IncidentTriageEnv(task_id="hard", seed=0)
    obs1 = env.reset(task_id="hard", seed=0)
    obs2 = env.reset(task_id="hard", seed=0)

    assert obs1.scenario_id == obs2.scenario_id
    assert obs1.objective == obs2.objective


def test_seed_one_selects_variant_scenarios():
    easy_env = IncidentTriageEnv(task_id="easy", seed=1)
    medium_env = IncidentTriageEnv(task_id="medium", seed=1)
    hard_env = IncidentTriageEnv(task_id="hard", seed=1)

    easy_obs = easy_env.reset(task_id="easy", seed=1)
    medium_obs = medium_env.reset(task_id="medium", seed=1)
    hard_obs = hard_env.reset(task_id="hard", seed=1)

    assert easy_obs.scenario_id == "easy_queue_backlog_with_noise"
    assert medium_obs.scenario_id == "medium_db_pool_regression"
    assert hard_obs.scenario_id == "hard_real_incident_memory_leak"


def test_hard_variant_real_incident_perfect_run_reaches_full_score():
    env = IncidentTriageEnv(task_id="hard", seed=1)
    env.reset(task_id="hard", seed=1)

    actions = [
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
    ]

    done = False
    info = {}
    for action in actions:
        _, _, done, info = env.step(Action(**action))

    assert done is True
    assert info["score"] == 1.0
    assert info["success"] is True