from __future__ import annotations

from typing import Any

import gymnasium as gym

from .easy_security_env import SecurityTriageEnv
from .hard_sre_env import SREIncidentEnv
from .medium_fraud_env import FraudQueueEnv


class UnifiedIncidentTriageEnv(gym.Env):
    """Single wrapper to switch between easy/medium/hard incident triage tasks."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, difficulty: str = "easy", rng_seed: int | None = None):
        super().__init__()
        self.difficulty = difficulty
        if difficulty == "easy":
            self.env = SecurityTriageEnv(rng_seed=rng_seed)
        elif difficulty == "medium":
            self.env = FraudQueueEnv(rng_seed=rng_seed)
        elif difficulty == "hard":
            self.env = SREIncidentEnv(rng_seed=rng_seed)
        else:
            raise ValueError("difficulty must be one of: easy, medium, hard")

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any):
        return self.env.step(action)

    def render(self):
        return self.env.render() if hasattr(self.env, "render") else None

    def close(self):
        return self.env.close()
