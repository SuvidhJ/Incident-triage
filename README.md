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

A production-style OpenEnv benchmark where an agent performs realistic incident triage:

- ingest active alerts
- inspect evidence across alerts, services, runbooks, incidents, and changes
- form a structured decision (severity, owner team, root cause)
- execute response action
- submit final triage outcome

This environment is designed for practical agent evaluation in SRE/on-call workflows, not toy gameplay.

## Why This Environment Matters

Modern incident response is high-stakes, ambiguous, and time-sensitive. Strong agents must:

- identify signal in noisy telemetry
- avoid false escalation and underreaction
- optimize for correctness and MTTR
- remain deterministic and auditable under evaluation

This benchmark captures those constraints with structured actions, transparent grading, and difficulty progression.

## Evaluator Quick Check

Run these and you should be submission-ready:

```bash
pytest -q
openenv validate
AGENT_MODE=heuristic python inference.py
bash ./scripts/validate-submission.sh https://<your-space>.hf.space .
```

## OpenEnv Compliance

Implemented end-to-end:

- typed models: Action, Observation, RewardModel, EnvState
- reset(task_id, seed) -> Observation
- step(action) -> (Observation, RewardModel, done, info)
- state() -> EnvState
- openenv.yaml manifest with tasks/models/entrypoint

Manifest: openenv.yaml
Server entrypoint: app.server:app

## Environment Interface

### API Endpoints

- GET / : basic status + tasks
- GET /health : liveness
- GET /tasks : available tasks
- GET /metadata : benchmark metadata
- GET /schema : JSON schema for action/observation/state
- GET /openenv.yaml : manifest
- POST /reset : starts episode
- POST /step : applies action
- GET /state : internal state snapshot (with active redaction)

### Action Space

```json
{"action_type":"inspect","target_type":"alert|service|runbook|incident|change","target_id":"..."}
```

```json
{"action_type":"set_field","field_name":"severity|owner_team|root_cause_service|decision_type|decision_target","value":"..."}
```

```json
{"action_type":"execute_response"}
```

```json
{"action_type":"submit"}
```

### Observation Highlights

Each step includes:

- objective + active alerts
- visible_context
- available_inspections
- discovered_evidence
- available_runbooks
- current_decision
- response_status
- allowed_values
- step_count, remaining_steps, minutes_elapsed
- last_action_error

## Task Suite

Three tasks with deterministic scenario selection by seed and increasing ambiguity:

| Task | Difficulty | Scenarios | Challenge |
|---|---|---|---|
| easy | easy | easy_cpu_spike, easy_queue_backlog_with_noise | clear overload with mild distractors |
| medium | medium | medium_dependency_failure, medium_db_pool_regression | symptom vs root-cause mismatch |
| hard | hard | hard_false_positive_vs_real, hard_real_incident_memory_leak | false positive discrimination under uncertainty |

Deterministic selection:

- seed=0 picks first scenario id
- seed=1 picks second scenario id

## Reward and Grading Design

All scores are clamped to [0.0, 1.0].

### Partial Progress Signals

Weighted components include:

- evidence quality
- severity correctness
- owner correctness
- root cause correctness
- decision type correctness
- decision target correctness
- response outcome correctness

### Time and Behavior Penalties

Action time costs:

- inspect: +2 min
- set_field: +0 min
- execute_response: +4 min
- submit: +0 min

Penalties include:

- invalid actions
- MTTR overrun
- low-precision evidence farming
- incoherent decision combinations
- underreaction on real incidents
- overreaction on false positives

### Anti-Gaming and Safety

- /state redacts hidden_ground_truth while done=false
- hidden answer keys are exposed only after episode completion
- scenario/task loaders validate data consistency and allowed values

## Baseline Inference

File location: inference.py (repo root, as required)

### LLM Client and Auth

- uses OpenAI Python client for all LLM calls
- environment variables:
  - API_BASE_URL (default: https://router.huggingface.co/v1)
  - MODEL_NAME (default: Qwen/Qwen2.5-72B-Instruct)
  - HF_TOKEN
  - OPENAI_API_KEY (alias fallback; HF_TOKEN takes precedence)

### Agent Modes

- AGENT_MODE=heuristic (deterministic baseline)
- AGENT_MODE=llm
- AGENT_MODE=hybrid

### Required Structured Stdout Contract

The script emits only:

- [START]
- [STEP]
- [END]

Example:

```text
[START] task=easy env=incident-triage-orchestrator model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"inspect","target_type":"alert","target_id":"A1"} reward=0.05 done=false error=null
[END] success=true steps=10 score=1.00 rewards=0.05,0.05,...
```

Expected deterministic heuristic baseline (seed=0):

- easy: 1.00
- medium: 1.00
- hard: 1.00

## Local Setup

### Python

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Run API

```bash
uvicorn app.server:app --host 0.0.0.0 --port 7860 --reload
```

### Smoke Test

```bash
curl http://localhost:7860/health
curl -X POST "http://localhost:7860/reset" -H "Content-Type: application/json" -d '{}'
```

## Validation Scripts

Linux/macOS:

```bash
bash ./scripts/precheck.sh
bash ./scripts/validate-submission.sh https://<your-space>.hf.space .
```

Windows PowerShell:

```powershell
./scripts/precheck.ps1
```

## Docker

Build:

```bash
docker build -t incident-triage-orchestrator .
```

Run:

```bash
docker run -p 7860:7860 incident-triage-orchestrator
```

The runtime image uses requirements-runtime.txt to keep build and startup fast under validator constraints.

## Hugging Face Space Deployment

Use a Docker Space and set:

- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

Recommended post-deploy checks:

- GET /health
- POST /reset with {}
- POST /step with a valid action
- GET /state

## Repository Structure

```text
app/                    FastAPI OpenEnv server + environment logic
data/scenarios/         Scenario definitions
data/tasks/             Task definitions (easy/medium/hard)
incident_rl/            RL-oriented auxiliary envs/utilities
scripts/                Precheck and submission validator scripts
tests/                  API/env/grader/loader/inference tests
inference.py            Required baseline inference script
openenv.yaml            OpenEnv manifest
Dockerfile              Container entry
README.md               This document
```

## What Makes This Submission Strong

- real-world operational domain with practical utility
- clear difficulty progression and deterministic reproducibility
- rich, non-sparse reward shaping with explicit anti-gaming controls
- strict structured logging for reliable automated scoring
- validator-aligned deployment, docs, and scripts

## License and Usage

Intended for benchmark and research workflows around agentic incident triage and policy evaluation.
