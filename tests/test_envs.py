from incident_rl.envs.unified_env import UnifiedIncidentTriageEnv


def run_one_episode(env: UnifiedIncidentTriageEnv, max_steps: int = 200):
    obs, info = env.reset(seed=123)
    assert obs is not None
    assert isinstance(info, dict)

    total_reward = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(info, dict)
        total_reward += float(reward)
        done = bool(terminated or truncated)
        steps += 1
    return steps, total_reward


def test_easy_env_runs():
    env = UnifiedIncidentTriageEnv(difficulty="easy", rng_seed=1)
    steps, _ = run_one_episode(env, max_steps=5)
    assert steps == 1  # easy mode is single-step


def test_medium_env_runs():
    env = UnifiedIncidentTriageEnv(difficulty="medium", rng_seed=2)
    steps, _ = run_one_episode(env, max_steps=80)
    assert steps > 1


def test_hard_env_runs():
    env = UnifiedIncidentTriageEnv(difficulty="hard", rng_seed=3)
    steps, _ = run_one_episode(env, max_steps=120)
    assert steps > 1
