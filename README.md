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

## Tasks

Easy:
- single-service overload with an obvious mitigation runbook.

Medium:
- application symptoms are caused by a dependency issue, requiring correct team escalation.

Hard:
- noisy alert under ambiguity; the agent must distinguish a false positive from a real incident.

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

Expected bundled-scenario heuristic scores:
- easy: 1.00
- medium: 1.00
- hard: 1.00

## Docker

Build:

```bash
docker build -t incident-triage-orchestrator .
```

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