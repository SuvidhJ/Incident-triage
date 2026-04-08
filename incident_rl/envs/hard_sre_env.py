from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class ServiceState:
    name: str
    error_rate: float
    latency_p99: float
    cpu_usage: float
    memory_usage: float
    request_rate: int
    health_status: int  # 0 healthy, 1 degraded, 2 down
    root_cause: bool = False
    internal_queue_depth: int = 0
    db_connection_pool: float = 1.0


class SREIncidentEnv(gym.Env):
    """Simplified multi-service incident response with partial observability."""

    metadata = {"render_modes": ["human"]}

    INVESTIGATION_ACTIONS = set(range(0, 20))
    MITIGATION_ACTIONS = set(range(20, 70))

    def __init__(self, n_services: int = 5, max_steps: int = 100, rng_seed: int | None = None) -> None:
        super().__init__()
        self.n_services = n_services
        self.max_steps = max_steps
        self.rng = np.random.default_rng(rng_seed)

        self.action_space = gym.spaces.Discrete(73)
        self.observation_space = gym.spaces.Dict(
            {
                "service_metrics": gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_services, 5), dtype=np.float32),
                "health_status": gym.spaces.Box(low=0, high=2, shape=(self.n_services,), dtype=np.int32),
                "dependency_graph": gym.spaces.Box(low=0, high=1, shape=(self.n_services, self.n_services), dtype=np.int32),
                "time_since_incident": gym.spaces.Discrete(500),
                "cost_spent": gym.spaces.Box(low=0.0, high=1_000_000.0, shape=(1,), dtype=np.float32),
                "customer_impact": gym.spaces.Box(low=0.0, high=1_000_000.0, shape=(1,), dtype=np.float32),
            }
        )

        self.services: list[ServiceState] = []
        self.dependency_graph = np.zeros((self.n_services, self.n_services), dtype=np.int32)
        self.time_since_incident = 0
        self.cost_spent = 0.0
        self.customer_impact = 0.0
        self.step_count = 0
        self.root_service_idx = 0
        self.root_cause_identified_at_step: int | None = None
        self.investigation_history: list[dict[str, Any]] = []

    def _init_topology(self) -> None:
        # Simple directed dependency DAG
        self.dependency_graph.fill(0)
        for i in range(self.n_services - 1):
            self.dependency_graph[i, i + 1] = 1
        if self.n_services >= 5:
            self.dependency_graph[0, 2] = 1
            self.dependency_graph[1, 3] = 1

    def _init_services(self) -> None:
        self.services = []
        for i in range(self.n_services):
            self.services.append(
                ServiceState(
                    name=f"service_{i}",
                    error_rate=float(np.clip(self.rng.normal(0.03, 0.01), 0.0, 1.0)),
                    latency_p99=float(np.clip(self.rng.normal(0.15, 0.03), 0.0, 1.0)),
                    cpu_usage=float(np.clip(self.rng.normal(0.45, 0.15), 0.0, 1.0)),
                    memory_usage=float(np.clip(self.rng.normal(0.5, 0.12), 0.0, 1.0)),
                    request_rate=int(self.rng.integers(80, 1200)),
                    health_status=0,
                )
            )

    def _inject_incident(self) -> None:
        self.root_service_idx = int(self.rng.integers(0, self.n_services))
        root = self.services[self.root_service_idx]
        root.root_cause = True
        root.internal_queue_depth = int(self.rng.integers(200, 1000))
        root.db_connection_pool = float(np.clip(self.rng.normal(0.1, 0.07), 0.0, 1.0))
        root.error_rate = float(np.clip(self.rng.normal(0.7, 0.15), 0.0, 1.0))
        root.latency_p99 = float(np.clip(self.rng.normal(0.8, 0.12), 0.0, 1.0))
        root.health_status = 2

        # Warm start cascade
        for _ in range(3):
            self._propagate_failures()

    def _propagate_failures(self) -> None:
        degraded_indices = [i for i, s in enumerate(self.services) if s.health_status > 0]
        for i in degraded_indices:
            for down in np.where(self.dependency_graph[i] == 1)[0]:
                target = self.services[int(down)]
                if self.rng.random() < 0.3:
                    target.error_rate = float(np.clip(target.error_rate + 0.12, 0.0, 1.0))
                    target.latency_p99 = float(np.clip(target.latency_p99 + 0.1, 0.0, 1.0))
                    target.health_status = min(2, target.health_status + 1)

        # Natural slight drift toward recovery if low error
        for s in self.services:
            if s.error_rate < 0.08:
                s.health_status = max(0, s.health_status - 1)

    def _decode_action(self, action: int) -> tuple[str, int | None]:
        if action < 10:
            return "check_logs", int(action % self.n_services)
        if action < 20:
            return "run_diagnostics", int((action - 10) % self.n_services)
        if action < 30:
            return "rollback", int((action - 20) % self.n_services)
        if action < 40:
            return "scale_up", int((action - 30) % self.n_services)
        if action < 50:
            return "restart", int((action - 40) % self.n_services)
        if action < 60:
            return "isolate", int((action - 50) % self.n_services)
        if action < 70:
            return "failover", int((action - 60) % self.n_services)
        if action == 70:
            return "global_circuit_breaker", None
        if action == 71:
            return "gradual_traffic_shift", None
        return "declare_major_incident", None

    def _investigate(self, kind: str, sid: int) -> dict[str, Any]:
        s = self.services[sid]
        found = bool(s.root_cause and (kind == "run_diagnostics" or self.rng.random() < 0.6))
        if found and self.root_cause_identified_at_step is None:
            self.root_cause_identified_at_step = self.step_count
        result = {
            "revealed_root_cause": found,
            "error_rate_delta": 0.0,
            "cost": 0.0,
            "users_affected": 0.0,
            "incident_resolved": False,
            "data_loss": False,
            "prevented_cascade": False,
        }
        self.investigation_history.append({"service": sid, "action": kind, "found": found})
        return result

    def _mitigate(self, kind: str, sid: int) -> dict[str, Any]:
        s = self.services[sid]
        before = s.error_rate
        cost = 0.0
        data_loss = False
        prevented_cascade = False

        if kind == "rollback":
            cost = 120.0
            if sid == self.root_service_idx:
                s.error_rate = max(0.02, s.error_rate - 0.45)
                s.health_status = max(0, s.health_status - 1)
            else:
                s.error_rate = min(1.0, s.error_rate + 0.08)
        elif kind == "scale_up":
            cost = 250.0
            s.error_rate = max(0.0, s.error_rate - 0.2)
            s.latency_p99 = max(0.02, s.latency_p99 - 0.15)
        elif kind == "restart":
            cost = 80.0
            s.error_rate = max(0.0, s.error_rate - 0.35)
            s.latency_p99 = max(0.02, s.latency_p99 - 0.2)
            s.health_status = max(0, s.health_status - 1)
            if sid != self.root_service_idx and self.rng.random() < 0.08:
                data_loss = True
        elif kind == "isolate":
            cost = 60.0
            prevented_cascade = True
            self.dependency_graph[sid, :] = 0
            s.error_rate = max(0.0, s.error_rate - 0.15)
        elif kind == "failover":
            cost = 300.0
            s.error_rate = max(0.0, s.error_rate - 0.5)
            s.latency_p99 = max(0.02, s.latency_p99 - 0.25)

        error_rate_delta = before - s.error_rate
        return {
            "revealed_root_cause": False,
            "error_rate_delta": error_rate_delta,
            "cost": cost,
            "users_affected": 0.0,
            "incident_resolved": False,
            "data_loss": data_loss,
            "prevented_cascade": prevented_cascade,
        }

    def _coordination(self, kind: str) -> dict[str, Any]:
        cost = 1000.0
        if kind == "global_circuit_breaker":
            for s in self.services:
                s.request_rate = max(10, int(s.request_rate * 0.25))
        elif kind == "gradual_traffic_shift":
            for s in self.services:
                s.error_rate = max(0.0, s.error_rate - 0.1)
        return {
            "revealed_root_cause": False,
            "error_rate_delta": 0.05,
            "cost": cost,
            "users_affected": 0.0,
            "incident_resolved": False,
            "data_loss": False,
            "prevented_cascade": kind == "global_circuit_breaker",
        }

    def _incident_resolved(self) -> bool:
        return all(s.error_rate < 0.08 and s.health_status == 0 for s in self.services)

    def _compute_reward(self, action: int, outcome: dict[str, Any]) -> float:
        reward = 0.0
        if outcome["incident_resolved"]:
            reward += 1000.0 / max(1, self.time_since_incident)
            if self.root_cause_identified_at_step is not None and self.root_cause_identified_at_step < 5:
                reward += 200.0

        reward -= outcome["cost"] * 0.1
        reward -= (outcome["users_affected"] / 1000.0) ** 2
        if outcome["data_loss"]:
            reward -= 5000.0

        if action in self.INVESTIGATION_ACTIONS:
            reward += 50.0 if outcome["revealed_root_cause"] else -5.0
        if action in self.MITIGATION_ACTIONS:
            reward += outcome["error_rate_delta"] * 100.0
            if outcome["error_rate_delta"] < 0:
                reward -= 200.0
        if outcome["prevented_cascade"]:
            reward += 300.0
        return reward

    def _get_obs(self) -> dict[str, Any]:
        metrics = np.zeros((self.n_services, 5), dtype=np.float32)
        health = np.zeros((self.n_services,), dtype=np.int32)
        for i, s in enumerate(self.services):
            metrics[i] = np.array([s.error_rate, s.latency_p99, s.cpu_usage, s.memory_usage, min(s.request_rate / 2000.0, 1.0)], dtype=np.float32)
            health[i] = s.health_status

        return {
            "service_metrics": metrics,
            "health_status": health,
            "dependency_graph": self.dependency_graph.copy(),
            "time_since_incident": self.time_since_incident,
            "cost_spent": np.array([self.cost_spent], dtype=np.float32),
            "customer_impact": np.array([self.customer_impact], dtype=np.float32),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._init_topology()
        self._init_services()
        self._inject_incident()
        self.time_since_incident = 0
        self.cost_spent = 0.0
        self.customer_impact = 0.0
        self.step_count = 0
        self.root_cause_identified_at_step = None
        self.investigation_history = []
        return self._get_obs(), {}

    def step(self, action: int):
        kind, sid = self._decode_action(int(action))
        if kind in {"check_logs", "run_diagnostics"} and sid is not None:
            outcome = self._investigate(kind, sid)
        elif kind in {"rollback", "scale_up", "restart", "isolate", "failover"} and sid is not None:
            outcome = self._mitigate(kind, sid)
        else:
            outcome = self._coordination(kind)

        self._propagate_failures()
        self.time_since_incident += 1
        self.step_count += 1

        impacted = sum(s.error_rate * s.request_rate for s in self.services)
        self.customer_impact += impacted
        self.cost_spent += outcome["cost"]

        resolved = self._incident_resolved()
        outcome["incident_resolved"] = resolved
        outcome["users_affected"] = impacted

        reward = self._compute_reward(int(action), outcome)
        done = resolved or self.step_count >= self.max_steps
        truncated = False

        info = {
            "cost": outcome["cost"],
            "customer_impact": self.customer_impact,
            "root_cause_found": bool(self.root_cause_identified_at_step is not None),
            "action_kind": kind,
            "action_service": sid,
        }
        return self._get_obs(), reward, done, truncated, info
