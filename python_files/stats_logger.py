"""
StatsLogger: collects training statistics in memory and saves them to a CSV on shutdown.

Hierarchy:
    StatsLogger              — base class with common fields (timestamp, step, games, reward)
    └── QLearningStatsLogger — adds Q-table specific fields (q_states, coverage, std_q, etc.)

To add a new AI method later, subclass StatsLogger, define EXTRA_FIELDS, and override
_build_extra_row() to extract the method-specific data from its stats dict.
"""

import csv
import gzip
import io
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import yaml


class StatsLogger(ABC):
    """
    Base class: thread-safe in-memory buffer of training statistics.

    Subclasses extend EXTRA_FIELDS and override _build_extra_row() to add
    algorithm-specific columns. Do not instantiate this class directly.
    """

    BASE_FIELDS = [
        "timestamp",
        "worker_id",
        "step",
        "total_games",
        "avg_reward",
    ]

    EXTRA_FIELDS: list[str] = []  # Overridden by subclasses

    @property
    def FIELDS(self) -> list[str]:
        return self.BASE_FIELDS + self.EXTRA_FIELDS

    def __init__(self) -> None:
        self._buffer: list[dict] = []
        self._episode_rewards: list[float] = []
        self._lock = threading.Lock()

    def add_episode_reward(self, reward: float, window: int) -> None:
        """Append a completed episode's reward to the shared pool (thread-safe)."""
        with self._lock:
            self._episode_rewards.append(reward)
            if len(self._episode_rewards) > window:
                self._episode_rewards.pop(0)

    def avg_episode_reward(self) -> float:
        """Return the mean reward over the current pool (thread-safe)."""
        with self._lock:
            pool = list(self._episode_rewards)
        return sum(pool) / len(pool) if pool else 0.0

    @abstractmethod
    def _build_extra_row(self, stats: dict) -> dict:
        """Extract algorithm-specific fields from a stats dict.

        Subclasses must override this to populate EXTRA_FIELDS.
        """
        raise NotImplementedError

    def record(self, worker_id: int, step: int, total_games: int, avg_reward: float, stats: dict) -> None:
        """
        Append one row to the in-memory buffer.

        Args:
            worker_id:   ID of the worker recording this row.
            step:        Current learning step.
            total_games: Episodes completed.
            avg_reward:  Average reward over recent episodes.
            stats:       Algorithm-specific stats dict (e.g. from agent.get_stats()).
                         Pass {} for non-primary workers — Q-table fields will be blank.
        """
        row = {
            "timestamp":   time.time(),
            "worker_id":   worker_id,
            "step":        step,
            "total_games": total_games,
            "avg_reward":  round(avg_reward, 4),
        }
        row.update(self._build_extra_row(stats))

        with self._lock:
            self._buffer.append(row)

    def save_log(self, base_dir: Path, config: dict) -> None:
        """
        Write the buffered records to a timestamped folder inside base_dir.

        Creates:
            base_dir/log_<YYYYMMDD_HHMMSS>/
                training.csv.gz   — gzip-compressed CSV of all recorded rows
                config.yaml       — copy of the training configuration

        Args:
            base_dir: Parent directory under which the log folder is created.
            config:   Raw config dict to snapshot alongside the CSV.
        """
        with self._lock:
            snapshot = list(self._buffer)

        if not snapshot:
            print("StatsLogger: no data to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(base_dir) / f"log_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Write gzip-compressed CSV
        csv_path = log_dir / "training.csv.gz"
        with gzip.open(csv_path, "wt", newline="", encoding="utf-8") as gz:
            writer = csv.DictWriter(gz, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(snapshot)

        # Write config snapshot
        config_path = log_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Training log saved to {log_dir} ({len(snapshot)} records)")


class QLearningStatsLogger(StatsLogger):
    """
    StatsLogger for Q-learning (value iteration).

    Extra columns logged per record:
        - q_states:         Number of (state, action) pairs in the Q-table
        - q_coverage:       Percentage of the total state-action space visited
        - avg_q:            Mean Q-value across all Q-table entries
        - max_q:            Highest Q-value in the Q-table
        - std_q:            Standard deviation of Q-values
        - exploration_rate: Percentage of actions that were random explorations
        - updates:          Total Q-value updates applied
    """

    EXTRA_FIELDS = [
        "q_states",
        "q_coverage",
        "avg_q",
        "max_q",
        "std_q",
        "exploration_rate",
        "updates",
    ]

    def _build_extra_row(self, stats: dict) -> dict:
        # Return empty strings when stats is empty (non-primary worker)
        # so Q-table columns are blank in the CSV rather than filled with zeros
        if not stats:
            return {field: "" for field in self.EXTRA_FIELDS}
        return {
            "q_states":         stats.get("num_states", 0),
            "q_coverage":       round(stats.get("q_coverage", 0.0), 4),
            "avg_q":            round(stats.get("avg_q", 0.0), 6),
            "max_q":            round(stats.get("max_q", 0.0), 6),
            "std_q":            round(stats.get("std_q", 0.0), 6),
            "exploration_rate": round(stats.get("exploration_rate", 0.0), 2),
            "updates":          stats.get("updates", 0),
        }
