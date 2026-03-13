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
    AGENT_NAME:   str       = "agent"  # Overridden by subclasses — used in log folder name

    @property
    def FIELDS(self) -> list[str]:
        return self.BASE_FIELDS + self.EXTRA_FIELDS

    def __init__(self) -> None:
        self._buffer: list[dict] = []
        self._episode_rewards: list[float] = []
        self._episode_losses: list[float] = []
        self._episode_entropies: list[float] = []
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

    def add_episode_loss(self, loss: float, window: int) -> None:
        """Append a completed episode's policy loss to the shared pool (thread-safe)."""
        with self._lock:
            self._episode_losses.append(loss)
            if len(self._episode_losses) > window:
                self._episode_losses.pop(0)

    def avg_episode_loss(self) -> float:
        """Return the mean loss over the current pool (thread-safe)."""
        with self._lock:
            pool = list(self._episode_losses)
        return sum(pool) / len(pool) if pool else 0.0

    def add_episode_entropy(self, entropy: float, window: int) -> None:
        """Append a completed episode's entropy to the shared pool (thread-safe)."""
        with self._lock:
            self._episode_entropies.append(entropy)
            if len(self._episode_entropies) > window:
                self._episode_entropies.pop(0)

    def avg_episode_entropy(self) -> float:
        """Return the mean entropy over the current pool (thread-safe)."""
        with self._lock:
            pool = list(self._episode_entropies)
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
        log_dir = Path(base_dir) / f"log_{self.AGENT_NAME}_{timestamp}"
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


class PolicyGradientStatsLogger(StatsLogger):
    """
    StatsLogger for the REINFORCE policy gradient agent.

    Extra columns logged per record:
        - pg_episodes:   Total episodes that triggered a gradient update
        - pg_loss:       Policy loss from the most recent episode
        - avg_w:         Mean absolute weight value of W
        - max_w:         Max absolute weight value of W
        - std_w:         Std of W weights
        - avg_entropy:   Mean action entropy at the zero state (proxy for
                         policy confidence; drops as policy sharpens)
        - b_0 .. b_N:    Individual bias value per action. Reveals which action
                         the policy favours in a neutral state independently of W.
        - updates:       Total timesteps processed
    """

    AGENT_NAME   = "PolicyGradient"

    EXTRA_FIELDS = [
        "pg_episodes",      # total episodes that triggered a gradient update
        "pg_loss",          # last policy loss
        "episode_return",   # return of the last finished episode
        "episode_length",   # length (timesteps) of the last finished episode
        "avg_w",            # mean |W|
        "max_w",            # max |W|
        "std_w",            # std of W
        "avg_entropy",      # policy entropy at zero state
        "grad_norm_W",      # L2 norm of last dW
        "grad_norm_b",      # L2 norm of last db
        "b_0",              # per-action bias terms (3 actions)
        "b_1",
        "b_2",
        "updates",          # total timesteps processed
    ]

    def _build_extra_row(self, stats: dict) -> dict:
        # Return empty strings for all extra fields when no stats dict is
        # provided (e.g. non-primary workers). This keeps CSV columns blank
        # rather than filled with misleading zeros.
        if not stats:
            return {field: "" for field in self.EXTRA_FIELDS}

        row = {
            "pg_episodes":    stats.get("episodes", 0),
            "pg_loss":        round(stats.get("last_loss", 0.0), 6),
            "episode_return": round(stats.get("episode_return", 0.0), 6),
            "episode_length": stats.get("episode_length", 0),
            "avg_w":          round(stats.get("avg_w", 0.0), 6),
            "max_w":          round(stats.get("max_w", 0.0), 6),
            "std_w":          round(stats.get("std_w", 0.0), 6),
            "avg_entropy":    round(stats.get("avg_entropy", 0.0), 6),
            "grad_norm_W":    round(stats.get("grad_norm_W", 0.0), 6),
            "grad_norm_b":    round(stats.get("grad_norm_b", 0.0), 6),
            "updates":        stats.get("updates", 0),
        }

        # Individual bias values — for now this logger assumes three actions
        # and therefore three bias terms b_0, b_1, b_2.
        for i in range(3):
            row[f"b_{i}"] = round(stats.get(f"b_{i}", 0.0), 6)

        return row


class PolicyGradientDNNStatsLogger(StatsLogger):
    """
    StatsLogger for the REINFORCE policy gradient agent — DNN (JAX MLP) version.

    Compared to PolicyGradientStatsLogger (linear agent):
      - Individual bias terms (b_0/b_1/b_2) are dropped: for a multi-layer
        network the output biases are far less interpretable because hidden-layer
        representations dominate. Logging them would be misleading.
      - Separate grad_norm_W / grad_norm_b are merged into a single grad_norm
        that spans all layers, matching what the DNN agent computes.

    Extra columns logged per record:
        pg_episodes     — total episodes that triggered a gradient update
        avg_loss        — windowed mean policy loss (same window as avg_reward)
        episode_return  — sum of rewards in the last finished episode
        episode_length  — number of steps in the last finished episode
        avg_w           — mean |weight| across all W matrices
        max_w           — max |weight| across all W matrices
        std_w           — std of all weights across all W matrices
        avg_entropy     — windowed mean H(π) at zero state (same window as avg_reward)
        grad_norm       — L2 norm of the full gradient across all layers
        updates         — total timesteps processed
    """

    AGENT_NAME = "PolicyGradientDNN"

    _BASE_EXTRA_FIELDS = [
        "pg_episodes",
        "avg_loss",
        "episode_return",
        "episode_length",
        "avg_w",
        "max_w",
        "std_w",
        "avg_entropy",
        "grad_norm",
        "updates",
    ]

    # EXTRA_FIELDS is set as an instance attribute in __init__ so that
    # per-layer avg_w_N columns can be appended once the architecture is known.
    EXTRA_FIELDS: list[str] = _BASE_EXTRA_FIELDS

    def __init__(self) -> None:
        super().__init__()
        # Own copy so extending it doesn't affect the class-level list or
        # other instances.
        self.EXTRA_FIELDS = list(self._BASE_EXTRA_FIELDS)
        self._layer_fields_added = False

    def _build_extra_row(self, stats: dict) -> dict:
        # Return empty strings for all extra fields when no stats dict is
        # provided (e.g. non-primary workers). This keeps CSV columns blank
        # rather than filled with misleading zeros.
        if not stats:
            return {field: "" for field in self.EXTRA_FIELDS}

        # On the first real record, discover per-layer keys and register them
        # so they appear as CSV columns.
        if not self._layer_fields_added:
            for key in sorted(k for k in stats if k.startswith("avg_w_")):
                if key not in self.EXTRA_FIELDS:
                    self.EXTRA_FIELDS.append(key)
            self._layer_fields_added = True

        row = {
            "pg_episodes":    stats.get("episodes", 0),
            "avg_loss":        round(self.avg_episode_loss(), 6),
            "episode_return": round(stats.get("episode_return", 0.0), 6),
            "episode_length": stats.get("episode_length", 0),
            "avg_w":          round(stats.get("avg_w", 0.0), 6),
            "max_w":          round(stats.get("max_w", 0.0), 6),
            "std_w":          round(stats.get("std_w", 0.0), 6),
            "avg_entropy":    round(self.avg_episode_entropy(), 6),
            "grad_norm":      round(stats.get("grad_norm", 0.0), 6),
            "updates":        stats.get("updates", 0),
        }
        for key in (k for k in self.EXTRA_FIELDS if k.startswith("avg_w_")):
            row[key] = round(stats.get(key, 0.0), 6)
        return row


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

    AGENT_NAME   = "QLearning"

    EXTRA_FIELDS = [
        "q_states",
        "q_coverage",
        "avg_q",
        "max_q",
        "min_q",
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
            "min_q":            round(stats.get("min_q", 0.0), 6),
            "std_q":            round(stats.get("std_q", 0.0), 6),
            "exploration_rate": round(stats.get("exploration_rate", 0.0), 2),
            "updates":          stats.get("updates", 0),
        }


class A2CStatsLogger(StatsLogger):
    """
    StatsLogger for the Advantage Actor-Critic (A2C) agent.

    Extra columns logged per record:
        a2c_episodes    — total episodes that triggered a gradient update
        avg_actor_loss  — windowed mean actor loss (same window as avg_reward)
        avg_critic_loss — windowed mean critic loss (same window as avg_reward)
        episode_return  — sum of rewards in the last finished episode
        episode_length  — number of steps in the last finished episode
        mean_advantage  — mean |A_t| at the last update
        avg_entropy     — windowed mean H(π) at zero state
        actor_grad_norm — L2 norm of the actor gradient at last update
        critic_grad_norm— L2 norm of the critic gradient at last update
        actor_avg_w     — mean |weight| across all actor W matrices
        actor_max_w     — max  |weight| across all actor W matrices
        actor_std_w     — std of all actor weights
        critic_avg_w    — mean |weight| across all critic W matrices
        critic_max_w    — max  |weight| across all critic W matrices
        critic_std_w    — std of all critic weights
        updates         — total timesteps processed
        avg_w_actor_N   — mean |weight| of actor layer N (one column per layer)
    """

    AGENT_NAME = "A2C"

    _BASE_EXTRA_FIELDS = [
        "a2c_episodes",
        "avg_actor_loss",
        "avg_critic_loss",
        "episode_return",
        "episode_length",
        "mean_advantage",
        "avg_entropy",
        "actor_grad_norm",
        "critic_grad_norm",
        "actor_avg_w",
        "actor_max_w",
        "actor_std_w",
        "critic_avg_w",
        "critic_max_w",
        "critic_std_w",
        "updates",
    ]

    EXTRA_FIELDS: list[str] = _BASE_EXTRA_FIELDS

    def __init__(self) -> None:
        super().__init__()
        self.EXTRA_FIELDS = list(self._BASE_EXTRA_FIELDS)
        self._layer_fields_added = False
        # Second loss window — tracks critic loss separately from the base
        # episode-loss pool (which is used for actor loss via add_episode_loss).
        self._episode_critic_losses: list[float] = []

    def add_episode_critic_loss(self, loss: float, window: int) -> None:
        """Append a completed episode's critic loss to the shared pool (thread-safe)."""
        with self._lock:
            self._episode_critic_losses.append(loss)
            if len(self._episode_critic_losses) > window:
                self._episode_critic_losses.pop(0)

    def avg_episode_critic_loss(self) -> float:
        """Return the mean critic loss over the current pool (thread-safe)."""
        with self._lock:
            pool = list(self._episode_critic_losses)
        return sum(pool) / len(pool) if pool else 0.0

    def _build_extra_row(self, stats: dict) -> dict:
        if not stats:
            return {field: "" for field in self.EXTRA_FIELDS}

        # Discover per-layer actor weight keys on first real record.
        if not self._layer_fields_added:
            for key in sorted(k for k in stats if k.startswith("avg_w_actor_")):
                if key not in self.EXTRA_FIELDS:
                    self.EXTRA_FIELDS.append(key)
            self._layer_fields_added = True

        row = {
            "a2c_episodes":    stats.get("episodes", 0),
            "avg_actor_loss":  round(self.avg_episode_loss(),        6),
            "avg_critic_loss": round(self.avg_episode_critic_loss(), 6),
            "episode_return":  round(stats.get("episode_return",    0.0), 6),
            "episode_length":       stats.get("episode_length",    0),
            "mean_advantage":  round(stats.get("mean_advantage",    0.0), 6),
            "avg_entropy":     round(self.avg_episode_entropy(),     6),
            "actor_grad_norm": round(stats.get("actor_grad_norm",   0.0), 6),
            "critic_grad_norm":round(stats.get("critic_grad_norm",  0.0), 6),
            "actor_avg_w":     round(stats.get("actor_avg_w",       0.0), 6),
            "actor_max_w":     round(stats.get("actor_max_w",       0.0), 6),
            "actor_std_w":     round(stats.get("actor_std_w",       0.0), 6),
            "critic_avg_w":    round(stats.get("critic_avg_w",      0.0), 6),
            "critic_max_w":    round(stats.get("critic_max_w",      0.0), 6),
            "critic_std_w":    round(stats.get("critic_std_w",      0.0), 6),
            "updates":              stats.get("updates",            0),
        }
        for key in (k for k in self.EXTRA_FIELDS if k.startswith("avg_w_actor_")):
            row[key] = round(stats.get(key, 0.0), 6)
        return row


class PPOStatsLogger(StatsLogger):
    """
    StatsLogger for the Proximal Policy Optimization (PPO-Clip) agent.

    Extends A2CStatsLogger with two PPO-specific diagnostics:
        clip_fraction   — fraction of timesteps where |r_t − 1| > ε (after K epochs)
                          Healthy range: 0.05 – 0.20.
                          > 0.30 → policy moving too aggressively.
        avg_approx_kl   — windowed mean approximate KL divergence between old and new policy
                          = mean(log π_old − log π_new) after K epochs.
                          Healthy range: 0.01 – 0.05.

    All other columns mirror A2CStatsLogger:
        ppo_episodes     — total episodes that triggered a gradient update
        avg_actor_loss   — windowed mean actor loss (averaged over K epochs)
        avg_critic_loss  — windowed mean critic loss (averaged over K epochs)
        episode_return   — sum of rewards in the last finished episode
        episode_length   — number of steps in the last finished episode
        mean_advantage   — mean |Â_t| (unnormalised) at the last update
        avg_entropy      — windowed mean H(π) at zero state
        actor_grad_norm  — L2 norm of actor gradient (last of K epochs)
        critic_grad_norm — L2 norm of critic gradient (last of K epochs)
        actor_avg_w      — mean |weight| across all actor W matrices
        actor_max_w      — max  |weight| across all actor W matrices
        actor_std_w      — std of all actor weights
        critic_avg_w     — mean |weight| across all critic W matrices
        critic_max_w     — max  |weight| across all critic W matrices
        critic_std_w     — std of all critic weights
        updates          — total timesteps processed
        avg_w_actor_N    — mean |weight| of actor layer N (one column per layer)
    """

    AGENT_NAME = "PPO"

    _BASE_EXTRA_FIELDS = [
        "ppo_episodes",
        "avg_actor_loss",
        "avg_critic_loss",
        "episode_return",
        "episode_length",
        "mean_advantage",
        "avg_entropy",
        "clip_fraction",       # PPO-specific: fraction of steps clipped
        "avg_approx_kl",       # PPO-specific: windowed mean approx KL divergence
        "actor_grad_norm",
        "critic_grad_norm",
        "actor_avg_w",
        "actor_max_w",
        "actor_std_w",
        "critic_avg_w",
        "critic_max_w",
        "critic_std_w",
        "updates",
    ]

    EXTRA_FIELDS: list[str] = _BASE_EXTRA_FIELDS

    def __init__(self) -> None:
        super().__init__()
        self.EXTRA_FIELDS = list(self._BASE_EXTRA_FIELDS)
        self._layer_fields_added = False
        # Separate windows for critic loss and approx KL (PPO-specific)
        self._episode_critic_losses: list[float] = []
        self._episode_approx_kls:    list[float] = []

    def add_episode_critic_loss(self, loss: float, window: int) -> None:
        """Append a completed episode's critic loss (thread-safe)."""
        with self._lock:
            self._episode_critic_losses.append(loss)
            if len(self._episode_critic_losses) > window:
                self._episode_critic_losses.pop(0)

    def avg_episode_critic_loss(self) -> float:
        """Return the windowed mean critic loss (thread-safe)."""
        with self._lock:
            pool = list(self._episode_critic_losses)
        return sum(pool) / len(pool) if pool else 0.0

    def add_episode_approx_kl(self, kl: float, window: int) -> None:
        """Append a completed episode's approx KL divergence (thread-safe)."""
        with self._lock:
            self._episode_approx_kls.append(kl)
            if len(self._episode_approx_kls) > window:
                self._episode_approx_kls.pop(0)

    def avg_episode_approx_kl(self) -> float:
        """Return the windowed mean approx KL divergence (thread-safe)."""
        with self._lock:
            pool = list(self._episode_approx_kls)
        return sum(pool) / len(pool) if pool else 0.0

    def _build_extra_row(self, stats: dict) -> dict:
        if not stats:
            return {field: "" for field in self.EXTRA_FIELDS}

        # Discover per-layer actor weight keys on first real record.
        if not self._layer_fields_added:
            for key in sorted(k for k in stats if k.startswith("avg_w_actor_")):
                if key not in self.EXTRA_FIELDS:
                    self.EXTRA_FIELDS.append(key)
            self._layer_fields_added = True

        row = {
            "ppo_episodes":    stats.get("episodes",          0),
            "avg_actor_loss":  round(self.avg_episode_loss(),         6),
            "avg_critic_loss": round(self.avg_episode_critic_loss(),  6),
            "episode_return":  round(stats.get("episode_return", 0.0), 6),
            "episode_length":       stats.get("episode_length", 0),
            "mean_advantage":  round(stats.get("mean_advantage", 0.0), 6),
            "avg_entropy":     round(self.avg_episode_entropy(),       6),
            "clip_fraction":   round(stats.get("clip_fraction",  0.0), 6),
            "avg_approx_kl":   round(self.avg_episode_approx_kl(),    6),
            "actor_grad_norm": round(stats.get("actor_grad_norm",  0.0), 6),
            "critic_grad_norm":round(stats.get("critic_grad_norm", 0.0), 6),
            "actor_avg_w":     round(stats.get("actor_avg_w",      0.0), 6),
            "actor_max_w":     round(stats.get("actor_max_w",      0.0), 6),
            "actor_std_w":     round(stats.get("actor_std_w",      0.0), 6),
            "critic_avg_w":    round(stats.get("critic_avg_w",     0.0), 6),
            "critic_max_w":    round(stats.get("critic_max_w",     0.0), 6),
            "critic_std_w":    round(stats.get("critic_std_w",     0.0), 6),
            "updates":              stats.get("updates",           0),
        }
        for key in (k for k in self.EXTRA_FIELDS if k.startswith("avg_w_actor_")):
            row[key] = round(stats.get(key, 0.0), 6)
        return row
