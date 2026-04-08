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