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