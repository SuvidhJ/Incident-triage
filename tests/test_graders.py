from app.graders import score_progress
from app.loaders import load_scenario, load_task


def test_perfect_easy_score_is_one():
    task = load_task("easy")
    scenario = load_scenario("easy_cpu_spike")

    working_decision = {
        "severity": "sev2",
        "owner_team": "payments-oncall",
        "root_cause_service": "payments-api",
        "decision_type": "run_runbook",
        "decision_target": "RB_SCALE_WORKERS",
    }
    discovered_keys = ["alert:A1", "service:payments-api", "runbook:RB_SCALE_WORKERS"]
    score, breakdown = score_progress(
        task=task,
        scenario=scenario,
        working_decision=working_decision,
        discovered_keys=discovered_keys,
        response_status="mitigation_applied",
        invalid_actions=0,
        minutes_elapsed=10,
    )
    assert score == 1.0
    assert breakdown["final_score"] == 1.0


def test_partial_score_in_range():
    task = load_task("hard")
    scenario = load_scenario("hard_false_positive_vs_real")

    working_decision = {
        "severity": "sev4",
        "owner_team": None,
        "root_cause_service": None,
        "decision_type": None,
        "decision_target": None,
    }
    discovered_keys = ["service:recommendations-api"]

    score, _ = score_progress(
        task=task,
        scenario=scenario,
        working_decision=working_decision,
        discovered_keys=discovered_keys,
        response_status="not_attempted",
        invalid_actions=0,
        minutes_elapsed=2,
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.0