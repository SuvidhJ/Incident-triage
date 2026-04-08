from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class FraudAlert:
    alert_id: int
    fraud_likelihood: float
    transaction_amount: float
    time_in_queue: int
    deadline: int
    merchant_id: int
    user_id: int
    device_fingerprint: int
    priority_score: float
    is_fraud: bool
    cluster_id: int


class FraudQueueEnv(gym.Env):
    """Queue management env: ordering + batching + deadlines under capacity."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, top_k: int = 20, max_steps: int = 50, rng_seed: int | None = None) -> None:
        super().__init__()
        self.top_k = top_k
        self.max_steps = max_steps
        self.rng = np.random.default_rng(rng_seed)
        self.step_count = 0
        self.current_time = 0
        self.next_alert_id = 0
        self.queue: list[FraudAlert] = []
        self.analyst_capacity = 3

        self.observation_space = gym.spaces.Dict(
            {
                "queue_features": gym.spaces.Box(low=0.0, high=1.0, shape=(self.top_k, 8), dtype=np.float32),
                "queue_length": gym.spaces.Discrete(500),
                "current_time": gym.spaces.Discrete(24),
                "analyst_capacity": gym.spaces.Discrete(10),
            }
        )
        self.action_space = gym.spaces.Discrete(self.top_k + 3)

    def _sample_alert(self, force_cluster: bool = False, cluster_id: int | None = None) -> FraudAlert:
        is_fraud = bool(self.rng.random() < 0.12)
        amount = float(np.clip(self.rng.lognormal(mean=4.5, sigma=1.0), 10, 20000))

        if is_fraud:
            likelihood = float(np.clip(self.rng.normal(0.8, 0.12), 0.0, 1.0))
            priority = float(np.clip(self.rng.normal(0.75, 0.18), 0.0, 1.0))
            deadline = int(self.rng.integers(12, 49))
        else:
            likelihood = float(np.clip(self.rng.normal(0.2, 0.16), 0.0, 1.0))
            priority = float(np.clip(self.rng.normal(0.35, 0.2), 0.0, 1.0))
            deadline = int(self.rng.integers(18, 73))

        if force_cluster and cluster_id is not None:
            merchant_id = 1000 + cluster_id
            device_fp = 2000 + cluster_id
            user_id = int(self.rng.integers(10000, 99999))
        else:
            merchant_id = int(self.rng.integers(1, 3000))
            device_fp = int(self.rng.integers(1, 6000))
            user_id = int(self.rng.integers(1, 1000000))
            cluster_id = int(self.rng.integers(0, 200))

        alert = FraudAlert(
            alert_id=self.next_alert_id,
            fraud_likelihood=likelihood,
            transaction_amount=amount,
            time_in_queue=0,
            deadline=deadline,
            merchant_id=merchant_id,
            user_id=user_id,
            device_fingerprint=device_fp,
            priority_score=priority,
            is_fraud=is_fraud,
            cluster_id=cluster_id,
        )
        self.next_alert_id += 1
        return alert

    def _top_queue(self) -> list[FraudAlert]:
        return self.queue[: self.top_k]

    def _normalize(self, alert: FraudAlert) -> np.ndarray:
        return np.array(
            [
                alert.fraud_likelihood,
                min(alert.transaction_amount / 20000.0, 1.0),
                min(alert.time_in_queue / 72.0, 1.0),
                min(alert.deadline / 72.0, 1.0),
                min(alert.merchant_id / 3000.0, 1.0),
                min(alert.user_id / 1_000_000.0, 1.0),
                min(alert.device_fingerprint / 6000.0, 1.0),
                alert.priority_score,
            ],
            dtype=np.float32,
        )

    def _get_obs(self) -> dict[str, Any]:
        matrix = np.zeros((self.top_k, 8), dtype=np.float32)
        for i, alert in enumerate(self._top_queue()):
            matrix[i] = self._normalize(alert)

        return {
            "queue_features": matrix,
            "queue_length": len(self.queue),
            "current_time": self.current_time,
            "analyst_capacity": self.analyst_capacity,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.current_time = int(self.rng.integers(0, 24))
        self.next_alert_id = 0
        self.analyst_capacity = int(self.rng.integers(2, 6))
        self.queue = [self._sample_alert() for _ in range(int(self.rng.integers(8, 21)))]
        self.queue.sort(key=lambda a: a.priority_score, reverse=True)
        return self._get_obs(), {}

    def _investigate(self, idx: int) -> tuple[float, dict[str, Any]]:
        if idx >= len(self.queue):
            return -10.0, {"invalid_action": True}

        alert = self.queue.pop(idx)
        reward = 0.0
        outcome: dict[str, Any] = {"invalid_action": False}

        if alert.is_fraud:
            saved = alert.transaction_amount
            reward += saved / 100.0
            if (alert.deadline - alert.time_in_queue) > 24:
                reward *= 1.5

            related = [a for a in self.queue if a.cluster_id == alert.cluster_id]
            if len(related) > 2:
                reward += 100.0 * len(related)
                self.queue = [a for a in self.queue if a.cluster_id != alert.cluster_id]
                outcome["related_alerts_found"] = len(related)
            else:
                outcome["related_alerts_found"] = 0
        else:
            reward -= 20.0

        return reward, outcome

    def _batch_investigate_cluster(self) -> float:
        if not self.queue:
            return -5.0

        counts: dict[int, int] = {}
        for a in self._top_queue():
            counts[a.cluster_id] = counts.get(a.cluster_id, 0) + 1

        best_cluster = max(counts, key=counts.get)
        cluster_alerts = [a for a in self.queue if a.cluster_id == best_cluster]
        self.queue = [a for a in self.queue if a.cluster_id != best_cluster]

        fraud_count = sum(1 for a in cluster_alerts if a.is_fraud)
        benign_count = len(cluster_alerts) - fraud_count
        return fraud_count * 25.0 - benign_count * 8.0

    def _auto_approve_low_risk(self) -> float:
        approved = [a for a in self.queue if a.fraud_likelihood < 0.2 and a.transaction_amount < 100]
        self.queue = [a for a in self.queue if a not in approved]
        fraud_missed = sum(1 for a in approved if a.is_fraud)
        return len(approved) * 3.0 - fraud_missed * 200.0

    def _age_and_arrivals(self) -> float:
        penalty = 0.0
        expired = []
        for a in self.queue:
            a.time_in_queue += 1
            if a.time_in_queue > a.deadline:
                expired.append(a)
        if expired:
            penalty -= 500.0 * len(expired)
            self.queue = [a for a in self.queue if a not in expired]

        new_alerts = int(self.rng.poisson(3))
        for _ in range(new_alerts):
            if self.rng.random() < 0.25 and self.queue:
                cluster_hint = int(self.rng.choice([a.cluster_id for a in self._top_queue()]))
                self.queue.append(self._sample_alert(force_cluster=True, cluster_id=cluster_hint))
            else:
                self.queue.append(self._sample_alert())

        self.queue.sort(key=lambda a: (a.priority_score + a.fraud_likelihood), reverse=True)
        self.current_time = (self.current_time + 1) % 24
        return penalty

    def step(self, action: int):
        reward = 0.0
        info: dict[str, Any] = {}

        if action < self.top_k:
            action_reward, outcome = self._investigate(int(action))
            reward += action_reward
            info.update(outcome)
        elif action == self.top_k:
            reward += self._batch_investigate_cluster()
        elif action == self.top_k + 1:
            reward -= 2.0  # defer cost
        elif action == self.top_k + 2:
            reward += self._auto_approve_low_risk()

        reward += self._age_and_arrivals()
        self.step_count += 1

        done = self.step_count >= self.max_steps
        truncated = False
        info["queue_length"] = len(self.queue)
        return self._get_obs(), reward, done, truncated, info
