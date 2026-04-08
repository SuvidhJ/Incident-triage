from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from incident_rl.envs.unified_env import UnifiedIncidentTriageEnv
from incident_rl.eval import run_policy


def make_env(difficulty: str, seed: int) -> UnifiedIncidentTriageEnv:
    return UnifiedIncidentTriageEnv(difficulty=difficulty, rng_seed=seed)


def heuristic_policy_easy(obs: dict[str, Any]) -> int:
    severity = int(obs["alert_severity"])
    reputation = float(obs["source_reputation"][0])
    user_risk = float(obs["user_risk_score"][0])
    fp_rate = float(obs["historical_fp_rate"][0])

    if severity >= 4 or user_risk > 0.8:
        return 2  # escalate
    if severity >= 3 and reputation < 0.35:
        return 1  # remediate
    if fp_rate > 0.75 and severity <= 1:
        return 0  # dismiss
    return 3  # request context


def random_policy(action_space_n: int):
    rng = np.random.default_rng(123)
    return lambda _: int(rng.integers(0, action_space_n))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate RL agents for incident triage.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="models/agent.zip")
    parser.add_argument("--model-path", default="models/agent.zip")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()

    env = make_env(args.difficulty, seed=args.seed)

    if args.eval_only:
        if args.difficulty == "easy":
            baseline = run_policy(env, heuristic_policy_easy, episodes=args.eval_episodes)
        else:
            baseline = run_policy(env, random_policy(env.action_space.n), episodes=args.eval_episodes)
        print(f"[baseline] episodes={baseline.episodes} avg_reward={baseline.avg_reward:.2f} std={baseline.std_reward:.2f}")

    try:
        from stable_baselines3 import DQN, PPO
    except Exception as exc:
        print("stable-baselines3 is not available, skipping RL training/loading.")
        print(f"import error: {exc}")
        return

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    if args.eval_only:
        if args.algo == "ppo":
            model = PPO.load(args.model_path, env=env)
        else:
            model = DQN.load(args.model_path, env=env)

        rl = run_policy(env, lambda obs: int(model.predict(obs, deterministic=True)[0]), episodes=args.eval_episodes)
        print(f"[rl] episodes={rl.episodes} avg_reward={rl.avg_reward:.2f} std={rl.std_reward:.2f}")
        return

    if args.algo == "ppo":
        policy = "MultiInputPolicy"
        model = PPO(policy, env, verbose=1, seed=args.seed)
    else:
        policy = "MultiInputPolicy"
        model = DQN(policy, env, verbose=1, seed=args.seed)

    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()
