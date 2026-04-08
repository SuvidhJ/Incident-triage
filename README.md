---
title: Incident Triage Orchestrator
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - incident-triage
  - sre
  - rl
---

# Incident Triage Orchestrator

A real-world OpenEnv environment where an AI agent acts as an on-call engineer:
- reads alerts
- inspects system evidence
- selects severity, owner, and root cause
- executes a response
- submits a final triage decision

This models a genuine human workflow: production incident triage during on-call operations.

## Observation space

The observation model is app.models:Observation.

Key fields:
- task_id
- scenario_id
- objective
- active_alerts
- visible_context
- available_inspections
- discovered_evidence
- available_runbooks
- current_decision
- response_status
- allowed_values
- minutes_elapsed
- step_count
- remaining_steps
- last_action_error

## Action space

The action model is app.models:Action.

Supported actions:

1. Inspect evidence

```json
{"action_type":"inspect","target_type":"alert|service|runbook|incident|change","target_id":"..."}
```

2. Set a decision field

```json
{"action_type":"set_field","field_name":"severity|owner_team|root_cause_service|decision_type|decision_target","value":"..."}
```

3. Execute selected response

```json
{"action_type":"execute_response"}
```

4. Submit final decision

```json
{"action_type":"submit"}
```

## Reward design

Reward is normalized to [0, 1].

Partial progress credit is given for:
- discovering key evidence
- setting correct severity
- setting correct owner_team
- setting correct root_cause_service
- setting correct decision_type
- setting correct decision_target
- executing the correct response

MTTR awareness:
- inspect: 2 minutes
- set_field: 0 minutes
- execute_response: 4 minutes
- submit: 0 minutes

If the agent exceeds the scenario's ideal MTTR, the final score is reduced.

Efficiency penalties apply for:
- invalid actions
- wasted exploration
- unnecessary MTTR overrun

Additional anti-gaming guardrails:
- irrelevant evidence discovery is mildly penalized (precision-aware evidence score)
- incoherent decisions (for example, escalate decision_target mismatching owner_team) are penalized
- severe underreaction and false escalation penalties are applied by task context

State-leak prevention:
- `/state` hides `hidden_ground_truth` while the episode is in progress
- ground truth is exposed only after `done=true` for auditability/post-episode analysis

## Tasks

Easy:
- scenarios: `easy_cpu_spike`, `easy_queue_backlog_with_noise`
- single-service overload where noisy secondary signals can distract triage.

Medium:
- scenarios: `medium_dependency_failure`, `medium_db_pool_regression`
- symptom/root-cause mismatch requiring either correct escalation or targeted runbook execution.

Hard:
- scenarios: `hard_false_positive_vs_real`, `hard_real_incident_memory_leak`
- ambiguity stress: one variant is a false positive, another is a real customer-impacting incident.

Scenario selection is deterministic by seed:
- `seed=0` selects the first scenario id in each task
- `seed=1` selects the second scenario id in each task

## Project structure

```text
app/                Environment code
data/scenarios/     Incident scenarios
data/tasks/         Task configs
tests/              Unit + API tests
inference.py        Required baseline script
openenv.yaml        OpenEnv manifest
Dockerfile          Container deployment
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Run the API server:

```bash
uvicorn app.server:app --host 0.0.0.0 --port 7860 --reload
```

Test endpoints:

```bash
curl http://localhost:7860/
curl http://localhost:7860/health
curl "http://localhost:7860/reset?task_id=easy&seed=0"
```

## Baseline inference

Required env vars:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

Optional:
- AGENT_MODE=hybrid|llm|heuristic
- USE_HF_DATASET=1|0
- HF_DATASET_NAME (default: cais/mmlu)
- HF_DATASET_CONFIG (default: computer_security)
- HF_DATASET_SPLIT (default: test)

Run:

```bash
python inference.py
```

Logging format:
- [START]
- [STEP]
- [END]

Deterministic local smoke run:

```bash
AGENT_MODE=heuristic python inference.py
```

Local precheck scripts:
- Linux/macOS (bash): `./scripts/precheck.sh`
- Windows (PowerShell): `./scripts/precheck.ps1`

Variant-scenario spot check (seed 1):

```bash
curl -X POST "http://localhost:7860/reset" -H "Content-Type: application/json" -d '{"task_id":"hard","seed":1}'
```

Easy environment data source:
- By default, `incident_rl/envs/easy_security_env.py` attempts to load a Hugging Face dataset
  and derive alert features from dataset rows.
- If dataset loading fails (offline, missing package, or dataset unavailable), it automatically
  falls back to the synthetic alert simulator so training/inference remains runnable.

Expected bundled-scenario heuristic scores:
- easy: 1.00
- medium: 1.00
- hard: 1.00

## Docker

Build:

```bash
docker build -t incident-triage-orchestrator .
```

Notes:
- Docker image uses `requirements-runtime.txt` to keep build time low for validator time limits.
- Full local development dependencies remain in `requirements.txt`.

Run:

```bash
docker run -p 7860:7860 incident-triage-orchestrator
```

## Hugging Face Space deployment

Use a Docker Space.

Set secrets:
- HF_TOKEN
- API_BASE_URL
- MODEL_NAME

Optional:
- AGENT_MODE=hybrid

After deploy, verify:
- /
- /health
- /reset
- /step
- /state

## OpenEnv notes

The environment implements:
- typed Action / Observation / Reward / State models
- reset()
- step()
- state()
- deterministic task configs
- task-specific grading in [0, 1]

If the official validator expects slightly different openenv.yaml keys, adjust only that file to match the starter template.

## Final no-fail checklist

Core env:
- reset() works
- step() works
- state() works
- reward always in [0,1]
- final score in [0,1]
- easy / medium / hard all exist
- all tasks deterministic

Infra:
- Dockerfile builds
- server starts
- root returns 200
- /reset works
- /step works
- /state works

Inference:
- inference.py at repo root
- uses OpenAI client
- reads API_BASE_URL, MODEL_NAME, HF_TOKEN
- stdout only uses [START], [STEP], [END]

Docs:
- README present
- action space documented
- observation space documented
- reward documented
- setup documented

Validation:
- pytest -q
- docker build
- python inference.py
- openenv validate if available in your setup