from app.loaders import list_tasks, load_scenario, load_task


def test_list_tasks():
    assert list_tasks() == ["easy", "medium", "hard"]


def test_load_easy_task():
    task = load_task("easy")
    assert task.task_id == "easy"
    assert task.max_steps == 12


def test_load_easy_scenario():
    scenario = load_scenario("easy_cpu_spike")
    assert scenario.scenario_id == "easy_cpu_spike"
    assert scenario.ground_truth.decision_type == "run_runbook"


def test_tasks_define_multiple_scenarios():
    assert len(load_task("easy").scenario_ids) >= 2
    assert len(load_task("medium").scenario_ids) >= 2
    assert len(load_task("hard").scenario_ids) >= 2


def test_variant_scenarios_load_with_consistent_ground_truth():
    for scenario_id in [
        "easy_queue_backlog_with_noise",
        "medium_db_pool_regression",
        "hard_real_incident_memory_leak",
    ]:
        scenario = load_scenario(scenario_id)
        assert scenario.ground_truth.severity in scenario.allowed_values["severity"]
        assert scenario.ground_truth.owner_team in scenario.allowed_values["owner_team"]
        assert scenario.ground_truth.root_cause_service in scenario.allowed_values["root_cause_service"]
        assert scenario.ground_truth.decision_type in scenario.allowed_values["decision_type"]
        assert scenario.ground_truth.decision_target in scenario.allowed_values["decision_target"]
        assert set(scenario.ground_truth.key_evidence).issubset(set(scenario.inspectables.keys()))