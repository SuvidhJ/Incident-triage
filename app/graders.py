from __future__ import annotations

from typing import Dict, Tuple

from app.models import ScenarioConfig, TaskConfig

SEVERITY_ORDER = {
    "sev1": 1,
    "sev2": 2,
    "sev3": 3,
    "sev4": 4,
}


def _exact_match(a, b) -> float:
    return 1.0 if a is not None and a == b else 0.0


def _severity_score(predicted: str | None, truth: str) -> float:
    if predicted is None:
        return 0.0
    if predicted == truth:
        return 1.0
    if predicted in SEVERITY_ORDER and truth in SEVERITY_ORDER:
        distance = abs(SEVERITY_ORDER[predicted] - SEVERITY_ORDER[truth])
        if distance == 1:
            return 0.5
    return 0.0


def score_progress(
    task: TaskConfig,
    scenario: ScenarioConfig,
    working_decision: dict,
    discovered_keys: list[str],
    response_status: str,
    invalid_actions: int,
    minutes_elapsed: int,
) -> Tuple[float, Dict[str, float]]:
    gt = scenario.ground_truth
    weights = task.reward_weights

    key_evidence = gt.key_evidence
    if key_evidence:
        evidence_score = len(set(discovered_keys).intersection(set(key_evidence))) / len(key_evidence)
    else:
        evidence_score = 1.0

    severity_score = _severity_score(working_decision.get("severity"), gt.severity)
    owner_score = _exact_match(working_decision.get("owner_team"), gt.owner_team)
    root_cause_score = _exact_match(working_decision.get("root_cause_service"), gt.root_cause_service)
    decision_type_score = _exact_match(working_decision.get("decision_type"), gt.decision_type)
    decision_target_score = _exact_match(working_decision.get("decision_target"), gt.decision_target)
    response_score = _exact_match(response_status, gt.expected_response_status)

    raw_score = (
        weights["evidence"] * evidence_score
        + weights["severity"] * severity_score
        + weights["owner_team"] * owner_score
        + weights["root_cause_service"] * root_cause_score
        + weights["decision_type"] * decision_type_score
        + weights["decision_target"] * decision_target_score
        + weights["response_status"] * response_score
    )

    time_overrun = max(0, minutes_elapsed - gt.ideal_mttr_minutes)
    efficiency_multiplier = max(0.65, 1.0 - 0.05 * invalid_actions - 0.01 * time_overrun)

    final_score = raw_score * efficiency_multiplier

    # If a real action was required but never executed, cap the score.
    if gt.decision_type not in {"ignore"} and response_status == "not_attempted":
        final_score = min(final_score, 0.79)

    # Strongly penalize escalating / mitigating a false positive.
    chosen_decision = working_decision.get("decision_type")
    if gt.false_positive and chosen_decision in {"run_runbook", "escalate"}:
        final_score *= 0.8

    # Penalize underreaction on real incidents.
    if not gt.false_positive and chosen_decision in {"ignore", "false_positive"}:
        final_score *= 0.75

    final_score = max(0.0, min(1.0, final_score))

    breakdown = {
        "evidence": round(evidence_score, 4),
        "severity": round(severity_score, 4),
        "owner_team": round(owner_score, 4),
        "root_cause_service": round(root_cause_score, 4),
        "decision_type": round(decision_type_score, 4),
        "decision_target": round(decision_target_score, 4),
        "response_status": round(response_score, 4),
        "raw_score": round(raw_score, 4),
        "efficiency_multiplier": round(efficiency_multiplier, 4),
        "final_score": round(final_score, 4),
    }

    return round(final_score, 4), breakdown
