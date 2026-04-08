#!/usr/bin/env bash
set -euo pipefail

echo "==> Running tests"
pytest -q

echo "==> Running deterministic local inference smoke test"
AGENT_MODE=heuristic python inference.py > /tmp/inference.log
tail -n 20 /tmp/inference.log

echo "==> Verifying strict inference stdout format"
if grep -vE '^\[(START|STEP|END)\] ' /tmp/inference.log >/dev/null; then
    echo "Inference log format check failed: found non-bracketed stdout lines"
    exit 1
fi

echo "==> Building docker image"
docker build -t incident-triage-orchestrator .

echo "==> Local smoke test"
python - <<'PY'
from app.env import IncidentTriageEnv
from app.models import Action

env = IncidentTriageEnv(task_id="easy", seed=0)
obs = env.reset(task_id="easy", seed=0)
assert obs.task_id == "easy"

actions = [
    {"action_type":"inspect","target_type":"alert","target_id":"A1"},
    {"action_type":"inspect","target_type":"service","target_id":"payments-api"},
    {"action_type":"inspect","target_type":"runbook","target_id":"RB_SCALE_WORKERS"},
    {"action_type":"set_field","field_name":"severity","value":"sev2"},
    {"action_type":"set_field","field_name":"owner_team","value":"payments-oncall"},
    {"action_type":"set_field","field_name":"root_cause_service","value":"payments-api"},
    {"action_type":"set_field","field_name":"decision_type","value":"run_runbook"},
    {"action_type":"set_field","field_name":"decision_target","value":"RB_SCALE_WORKERS"},
    {"action_type":"execute_response"},
    {"action_type":"submit"}
]

done = False
info = {}
for a in actions:
    obs, reward, done, info = env.step(Action(**a))

assert done is True
assert 0.0 <= info["score"] <= 1.0
print("Local env smoke test passed. Final score:", info["score"])
PY

echo "==> All prechecks passed"
