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


def test_easy_mttr_overrun_penalizes_score():
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
        minutes_elapsed=20,
    )

    assert score == 0.9
    assert breakdown["efficiency_multiplier"] == 0.9


def test_efficiency_multiplier_has_floor():
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
        invalid_actions=20,
        minutes_elapsed=200,
    )

    assert score == 0.65
    assert breakdown["efficiency_multiplier"] == 0.65


def test_false_positive_escalation_gets_extra_penalty():
    task = load_task("hard")
    scenario = load_scenario("hard_false_positive_vs_real")
    discovered_keys = ["service:recommendations-api", "change:C902", "incident:I4120"]

    monitor_score, _ = score_progress(
        task=task,
        scenario=scenario,
        working_decision={
            "severity": "sev4",
            "owner_team": "observability-oncall",
            "root_cause_service": "monitoring-config",
            "decision_type": "monitor",
            "decision_target": "observability-oncall",
        },
        discovered_keys=discovered_keys,
        response_status="monitoring_only",
        invalid_actions=0,
        minutes_elapsed=10,
    )

    escalated_score, _ = score_progress(
        task=task,
        scenario=scenario,
        working_decision={
            "severity": "sev4",
            "owner_team": "observability-oncall",
            "root_cause_service": "monitoring-config",
            "decision_type": "escalate",
            "decision_target": "observability-oncall",
        },
        discovered_keys=discovered_keys,
        response_status="escalated_wrong_team",
        invalid_actions=0,
        minutes_elapsed=10,
    )

    assert monitor_score == 0.65
    assert escalated_score == 0.52


def test_real_incident_underreaction_penalty_applies():
    task = load_task("easy")
    scenario = load_scenario("easy_cpu_spike")
    discovered_keys = ["alert:A1", "service:payments-api", "runbook:RB_SCALE_WORKERS"]

    monitor_score, _ = score_progress(
        task=task,
        scenario=scenario,
        working_decision={
            "severity": "sev2",
            "owner_team": "payments-oncall",
            "root_cause_service": "payments-api",
            "decision_type": "monitor",
            "decision_target": "NONE",
        },
        discovered_keys=discovered_keys,
        response_status="underreacted",
        invalid_actions=0,
        minutes_elapsed=10,
    )

    underreacted_score, _ = score_progress(
        task=task,
        scenario=scenario,
        working_decision={
            "severity": "sev2",
            "owner_team": "payments-oncall",
            "root_cause_service": "payments-api",
            "decision_type": "false_positive",
            "decision_target": "NONE",
        },
        discovered_keys=discovered_keys,
        response_status="closed_incorrectly",
        invalid_actions=0,
        minutes_elapsed=10,
    )

    assert monitor_score == 0.55
    assert underreacted_score == 0.4125


def test_not_attempted_response_cap_is_enforced():
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

    score, _ = score_progress(
        task=task,
        scenario=scenario,
        working_decision=working_decision,
        discovered_keys=discovered_keys,
        response_status="not_attempted",
        invalid_actions=0,
        minutes_elapsed=10,
    )

    assert score == 0.79


def test_irrelevant_evidence_reduces_evidence_component():
    task = load_task("hard")
    scenario = load_scenario("hard_false_positive_vs_real")

    working_decision = {
        "severity": "sev4",
        "owner_team": "observability-oncall",
        "root_cause_service": "monitoring-config",
        "decision_type": "false_positive",
        "decision_target": "NONE",
    }

    clean_score, clean_breakdown = score_progress(
        task=task,
        scenario=scenario,
        working_decision=working_decision,
        discovered_keys=["service:recommendations-api", "change:C902", "incident:I4120"],
        response_status="closed_as_false_positive",
        invalid_actions=0,
        minutes_elapsed=10,
    )

    noisy_score, noisy_breakdown = score_progress(
        task=task,
        scenario=scenario,
        working_decision=working_decision,
        discovered_keys=[
            "service:recommendations-api",
            "change:C902",
            "incident:I4120",
            "alert:A4",
            "runbook:RB_RESTART_PODS",
            "runbook:RB_ROLLBACK_DEPLOY",
        ],
        response_status="closed_as_false_positive",
        invalid_actions=0,
        minutes_elapsed=10,
    )

    assert clean_score == 1.0
    assert noisy_breakdown["evidence"] < clean_breakdown["evidence"]
    assert noisy_score < clean_score


def test_escalation_target_mismatch_gets_coherence_penalty():
    task = load_task("medium")
    scenario = load_scenario("medium_dependency_failure")
    discovered_keys = [
        "alert:A2",
        "alert:A3",
        "service:checkout-api",
        "service:postgres-primary",
        "incident:I2877",
    ]

    aligned_score, aligned_breakdown = score_progress(
        task=task,
        scenario=scenario,
        working_decision={
            "severity": "sev1",
            "owner_team": "db-oncall",
            "root_cause_service": "postgres-primary",
            "decision_type": "escalate",
            "decision_target": "db-oncall",
        },
        discovered_keys=discovered_keys,
        response_status="escalated_correctly",
        invalid_actions=0,
        minutes_elapsed=14,
    )

    mismatched_score, mismatched_breakdown = score_progress(
        task=task,
        scenario=scenario,
        working_decision={
            "severity": "sev1",
            "owner_team": "db-oncall",
            "root_cause_service": "postgres-primary",
            "decision_type": "escalate",
            "decision_target": "infra-oncall",
        },
        discovered_keys=discovered_keys,
        response_status="escalated_wrong_team",
        invalid_actions=0,
        minutes_elapsed=14,
    )

    assert aligned_breakdown["coherence_multiplier"] == 1.0
    assert mismatched_breakdown["coherence_multiplier"] == 0.9
    assert mismatched_score < aligned_score