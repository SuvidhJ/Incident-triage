from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

import gymnasium as gym
import numpy as np


@dataclass
class AlertSample:
    features: dict[str, Any]
    ground_truth_threat: bool


class HFAlertDatasetAdapter:
    """Optional adapter to derive alert samples from Hugging Face datasets rows."""

    def __init__(
        self,
        dataset_name: str = "cais/mmlu",
        dataset_config: str | None = "computer_security",
        split: str = "test",
        rng_seed: int | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.rng = np.random.default_rng(rng_seed)
        self.rows: list[dict[str, Any]] = []
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            from datasets import load_dataset  # type: ignore[reportMissingImports]

            kwargs: dict[str, Any] = {"split": self.split}
            if self.dataset_config:
                ds = load_dataset(self.dataset_name, self.dataset_config, **kwargs)
            else:
                ds = load_dataset(self.dataset_name, **kwargs)

            if len(ds) == 0:
                self.error = "dataset split is empty"
                return
            self.rows = [dict(x) for x in ds]
        except Exception as exc:  # pragma: no cover - depends on optional external dataset/network
            self.error = str(exc)
            self.rows = []

    @property
    def is_available(self) -> bool:
        return len(self.rows) > 0

    def _risk_from_text(self, text: str) -> float:
        lowered = text.lower()
        high_risk_terms = [
            "malware",
            "phishing",
            "intrusion",
            "exfiltration",
            "credential",
            "ransomware",
            "privilege escalation",
            "c2",
            "attack",
            "exploit",
        ]
        medium_terms = ["anomaly", "suspicious", "failed login", "unauthorized", "firewall"]

        high_hits = sum(term in lowered for term in high_risk_terms)
        med_hits = sum(term in lowered for term in medium_terms)
        base = 0.18 + (0.16 * high_hits) + (0.08 * med_hits)
        noise = float(self.rng.normal(0.0, 0.08))
        return float(np.clip(base + noise, 0.02, 0.98))

    def generate_alert(self, recent_alert_count: int) -> AlertSample:
        row = self.rows[int(self.rng.integers(0, len(self.rows)))]
        row_text = " ".join(str(v) for v in row.values())
        risk = self._risk_from_text(row_text)

        is_threat = bool(risk >= 0.56)
        severity = int(np.clip(np.floor(risk * 5), 0, 4))
        alert_type = abs(hash(row_text)) % 10

        # Threat-like rows skew to lower source reputation and lower historical FP rate.
        source_rep = float(np.clip(1.0 - (risk * 0.85) + self.rng.normal(0.0, 0.06), 0.0, 1.0))
        historical_fp_rate = float(np.clip(0.75 - (risk * 0.65) + self.rng.normal(0.0, 0.07), 0.0, 1.0))
        user_risk = float(np.clip(risk + self.rng.normal(0.0, 0.07), 0.0, 1.0))

        features = {
            "alert_severity": severity,
            "alert_type": int(alert_type),
            "source_reputation": np.array([source_rep], dtype=np.float32),
            "time_of_day": int(self.rng.integers(0, 24)),
            "recent_alert_count": int(np.clip(recent_alert_count, 0, 100)),
            "user_risk_score": np.array([user_risk], dtype=np.float32),
            "asset_criticality": int(np.clip(round(risk * 2), 0, 2)),
            "historical_fp_rate": np.array([historical_fp_rate], dtype=np.float32),
        }
        return AlertSample(features=features, ground_truth_threat=is_threat)


class AlertSimulator:
    """Synthetic SOC alert stream with correlated threat/benign feature patterns."""

    def __init__(self, benign_rate: float = 0.85, rng_seed: int | None = None) -> None:
        self.benign_rate = benign_rate
        self.rng = np.random.default_rng(rng_seed)

    def generate_alert(self, recent_alert_count: int) -> AlertSample:
        is_threat = bool(self.rng.random() > self.benign_rate)

        if is_threat:
            severity = int(self.rng.choice([2, 3, 4], p=[0.15, 0.35, 0.5]))
            source_rep = float(self.rng.beta(2, 5))
            historical_fp_rate = float(self.rng.beta(2, 7))
            user_risk = float(self.rng.beta(5, 2))
        else:
            severity = int(self.rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))
            source_rep = float(self.rng.beta(5, 2))
            historical_fp_rate = float(self.rng.beta(6, 2))
            user_risk = float(self.rng.beta(2, 5))

        features = {
            "alert_severity": severity,
            "alert_type": int(self.rng.integers(0, 10)),
            "source_reputation": np.array([source_rep], dtype=np.float32),
            "time_of_day": int(self.rng.integers(0, 24)),
            "recent_alert_count": int(np.clip(recent_alert_count, 0, 100)),
            "user_risk_score": np.array([user_risk], dtype=np.float32),
            "asset_criticality": int(self.rng.integers(0, 3)),
            "historical_fp_rate": np.array([historical_fp_rate], dtype=np.float32),
        }
        return AlertSample(features=features, ground_truth_threat=is_threat)


class SecurityTriageEnv(gym.Env):
    """Single-step incident triage environment for SOC alert handling."""

    metadata = {"render_modes": ["human"]}

    ACTIONS = {
        0: "auto_dismiss",
        1: "auto_remediate",
        2: "escalate_to_L2",
        3: "request_context",
    }

    REWARD_MAP = {
        True: {"auto_dismiss": -10, "auto_remediate": 5, "escalate_to_L2": 8, "request_context": 3},
        False: {"auto_dismiss": 2, "auto_remediate": -3, "escalate_to_L2": -5, "request_context": -1},
    }

    def __init__(
        self,
        benign_rate: float = 0.85,
        rng_seed: int | None = None,
        use_hf_dataset: bool | None = None,
        hf_dataset_name: str | None = None,
        hf_dataset_config: str | None = None,
        hf_dataset_split: str | None = None,
    ) -> None:
        super().__init__()
        self.simulator = AlertSimulator(benign_rate=benign_rate, rng_seed=rng_seed)

        if use_hf_dataset is None:
            use_hf_dataset = os.getenv("USE_HF_DATASET", "0").lower() in {"1", "true", "yes"}
        self.use_hf_dataset = use_hf_dataset

        dataset_name = hf_dataset_name or os.getenv("HF_DATASET_NAME", "cais/mmlu")
        dataset_config = hf_dataset_config if hf_dataset_config is not None else os.getenv("HF_DATASET_CONFIG", "computer_security")
        dataset_split = hf_dataset_split or os.getenv("HF_DATASET_SPLIT", "test")

        self.dataset_adapter: HFAlertDatasetAdapter | None = None
        self.data_source = "synthetic"
        if self.use_hf_dataset:
            self.dataset_adapter = HFAlertDatasetAdapter(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=dataset_split,
                rng_seed=rng_seed,
            )
            if self.dataset_adapter.is_available:
                self.data_source = "huggingface"
        self.observation_space = gym.spaces.Dict(
            {
                "alert_severity": gym.spaces.Discrete(5),
                "alert_type": gym.spaces.Discrete(10),
                "source_reputation": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "time_of_day": gym.spaces.Discrete(24),
                "recent_alert_count": gym.spaces.Discrete(101),
                "user_risk_score": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "asset_criticality": gym.spaces.Discrete(3),
                "historical_fp_rate": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Discrete(4)
        self._recent_alert_count = 0
        self._current_alert: AlertSample | None = None

    def _generate_alert(self, recent_alert_count: int) -> AlertSample:
        if self.dataset_adapter is not None and self.dataset_adapter.is_available:
            return self.dataset_adapter.generate_alert(recent_alert_count)
        return self.simulator.generate_alert(recent_alert_count)

    def compute_reward(self, action: int, is_threat: bool) -> float:
        action_name = self.ACTIONS[int(action)]
        return float(self.REWARD_MAP[is_threat][action_name])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.simulator.rng = np.random.default_rng(seed)
            if self.dataset_adapter is not None:
                self.dataset_adapter.rng = np.random.default_rng(seed)
        self._recent_alert_count = int(self.np_random.integers(0, 101))
        self._current_alert = self._generate_alert(self._recent_alert_count)
        return self._current_alert.features, {}

    def step(self, action: int):
        assert self._current_alert is not None, "Call reset() before step()."
        is_threat = self._current_alert.ground_truth_threat
        reward = self.compute_reward(action, is_threat)

        info = {
            "is_threat": is_threat,
            "action_name": self.ACTIONS[int(action)],
            "correct": (is_threat and action in {1, 2, 3}) or (not is_threat and action == 0),
            "data_source": self.data_source,
        }

        terminated = True  # single-step triage
        truncated = False

        self._recent_alert_count = int(np.clip(self._recent_alert_count + self.np_random.integers(-3, 6), 0, 100))
        self._current_alert = self._generate_alert(self._recent_alert_count)

        return self._current_alert.features, reward, terminated, truncated, info
