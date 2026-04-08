$ErrorActionPreference = "Stop"

Write-Host "==> Running tests"
pytest -q

Write-Host "==> Running deterministic local inference smoke test"
$previousAgentMode = $env:AGENT_MODE
$env:AGENT_MODE = "heuristic"
$logPath = Join-Path $env:TEMP "incident-triage-inference.log"
python inference.py 1> $logPath
Get-Content $logPath | Select-Object -Last 20

Write-Host "==> Verifying strict inference stdout format"
$badLines = Get-Content $logPath | Where-Object {
    $_.Trim() -ne "" -and ($_ -notmatch '^\[(START|STEP|END)\] ')
}
if ($badLines) {
    Write-Host "Inference log format check failed: found non-bracketed stdout lines"
    $badLines | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" }
    throw "Inference log format validation failed"
}

Write-Host "==> Building docker image"
docker build -t incident-triage-orchestrator .

Write-Host "==> Local smoke test"
@'
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
    _, _, done, info = env.step(Action(**a))

assert done is True
assert 0.0 <= info["score"] <= 1.0
print("Local env smoke test passed. Final score:", info["score"])
'@ | python

if (Get-Command openenv -ErrorAction SilentlyContinue) {
    Write-Host "==> Running openenv validate"
    openenv validate
}
else {
    Write-Host "==> Skipping openenv validate (openenv CLI not found)"
}

if ($null -eq $previousAgentMode) {
    Remove-Item Env:AGENT_MODE -ErrorAction SilentlyContinue
}
else {
    $env:AGENT_MODE = $previousAgentMode
}

Write-Host "==> All prechecks passed"
