from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EvalResult:
    episodes: int
    avg_reward: float
    std_reward: float


def run_policy(env, policy_fn: Callable, episodes: int = 100, seed: int = 7) -> EvalResult:
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total += reward
        rewards.append(total)

    arr = np.asarray(rewards, dtype=np.float32)
    return EvalResult(episodes=episodes, avg_reward=float(arr.mean()), std_reward=float(arr.std()))
