from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_and_openenv_runtime_endpoints():
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "healthy"

    m = client.get("/metadata")
    assert m.status_code == 200
    assert m.json()["name"] == "incident-triage-orchestrator"
    assert "description" in m.json()

    s = client.get("/schema")
    assert s.status_code == 200
    body = s.json()
    assert "action" in body
    assert "observation" in body
    assert "state" in body

    rpc = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert rpc.status_code == 200
    assert rpc.json()["jsonrpc"] == "2.0"


def test_reset_step_state_flow():
    r = client.post("/reset", json={"task_id": "easy", "seed": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "easy"

    s = client.post("/step", json={"action_type": "inspect", "target_type": "alert", "target_id": "A1"})
    assert s.status_code == 200
    step_body = s.json()
    assert "reward" in step_body
    assert "observation" in step_body

    st = client.get("/state")
    assert st.status_code == 200
    state_body = st.json()
    assert state_body["task_id"] == "easy"
    assert state_body["hidden_ground_truth"] == {}


def test_state_reveals_ground_truth_only_after_done():
    client.post("/reset", json={"task_id": "easy", "seed": 0})

    in_progress = client.get("/state")
    assert in_progress.status_code == 200
    assert in_progress.json()["done"] is False
    assert in_progress.json()["hidden_ground_truth"] == {}

    completed = client.post("/step", json={"action_type": "submit"})
    assert completed.status_code == 200
    assert completed.json()["done"] is True

    final_state = client.get("/state")
    assert final_state.status_code == 200
    body = final_state.json()
    assert body["done"] is True
    assert body["hidden_ground_truth"]["severity"] == "sev2"