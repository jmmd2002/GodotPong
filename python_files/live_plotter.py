"""
LivePlotter: renders a live matplotlib window showing Q-learning training progress.

Runs in its own daemon thread. Reads from a QLearningStatsLogger buffer every
`refresh_interval` seconds and redraws 4 subplots:
    1. Average reward (last 20 episodes) over steps
    2. Q-table size and coverage % over steps
    3. Mean Q and std_q over steps
    4. Exploration rate % over steps

Usage:
    plotter = QLearningLivePlotter(logger, refresh_interval=5.0)
    plotter.start(shutdown_event)
    # plotter thread runs until shutdown_event is set
"""

import threading
from abc import ABC, abstractmethod
import matplotlib
matplotlib.use("TkAgg")   # must be set before importing pyplot
import matplotlib.pyplot as plt

from stats_logger import QLearningStatsLogger, PolicyGradientStatsLogger


class Plotter(ABC):
    """Abstract base class for live training plotters."""

    def __init__(self, refresh_interval: float = 5.0) -> None:
        self._refresh_interval = refresh_interval

    @abstractmethod
    def _run(self, shutdown_event: threading.Event) -> None:
        """Blocking plot loop that runs until ``shutdown_event`` is set."""


class QLearningLivePlotter(Plotter):
    """
    Live plot window for Q-learning training statistics.

    Args:
        logger:           A QLearningStatsLogger instance to read from.
        refresh_interval: Seconds between plot refreshes. Default: 5.0
    """

    def __init__(self, logger: QLearningStatsLogger, refresh_interval: float = 5.0,
                 reward_window: int = 20) -> None:
        super().__init__(refresh_interval=refresh_interval)
        self._logger = logger
        self._reward_window = reward_window
        self._thread: threading.Thread | None = None

    def start(self, shutdown_event: threading.Event) -> None:
        """Start the background plotting thread."""
        self._thread = threading.Thread(
            target=self._run,
            args=(shutdown_event,),
            daemon=True,
            name="live-plotter",
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, shutdown_event: threading.Event) -> None:
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle("Q-Learning Training Progress", fontsize=13)
        fig.tight_layout(pad=3.0)

        ax_reward, ax_qtable, ax_qval, ax_explore = axes.flatten()

        ax_reward.set_title("Avg Reward (last 20 eps)")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")

        ax_qtable.set_title("Q-Table Coverage")
        ax_qtable.set_xlabel("Step")
        ax_qtable.set_ylabel("Coverage %")
        ax_qtable_r = ax_qtable.twinx()
        ax_qtable_r.set_visible(False)

        ax_qval.set_title("Q-Values")
        ax_qval.set_xlabel("Step")
        ax_qval.set_ylabel("Q")

        ax_explore.set_title("Exploration Rate")
        ax_explore.set_xlabel("Step")
        ax_explore.set_ylabel("%")

        plt.show(block=False)

        while not shutdown_event.is_set():
            try:
                self._redraw(fig, ax_reward, ax_qtable, ax_qtable_r, ax_qval, ax_explore)
            except Exception as e:
                print(f"[LivePlotter] Draw error: {e}")

            # Wait for next refresh, checking shutdown every 0.5s
            for _ in range(int(self._refresh_interval / 0.5)):
                if shutdown_event.is_set():
                    break
                plt.pause(0.5)

        plt.close(fig)

    def _redraw(self, fig: plt.Figure, ax_reward: plt.Axes, ax_qtable: plt.Axes, 
                ax_qtable_r: plt.Axes, ax_qval: plt.Axes, ax_explore: plt.Axes) -> None:
        """Pull latest data from the logger and redraw all subplots."""
        with self._logger._lock:
            snapshot = list(self._logger._buffer)

        if len(snapshot) < 2:
            return  # not enough data yet

        # Group records by worker_id. Only worker 0 rows are used for all
        # plots: avg_reward is already computed from the shared episode reward
        # pool (all workers feed into it), and agent internals (Q-table, W/b)
        # are shared so every worker would log identical snapshots anyway.
        by_worker: dict[int, list[dict]] = {}
        for r in snapshot:
            by_worker.setdefault(r["worker_id"], []).append(r)

        w0_rows = by_worker.get(0, [])

        # --- Avg reward (from shared pool, recorded by worker 0) ---
        ax_reward.cla()
        ax_reward.set_title(f"Avg Reward (pooled, window={self._reward_window} eps)")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")
        if w0_rows:
            steps = [r["step"] for r in w0_rows]
            rewards = [r["avg_reward"] for r in w0_rows]
            ax_reward.plot(steps, rewards, color="tab:blue")

        if len(w0_rows) < 2:
            fig.tight_layout(pad=3.0)
            fig.canvas.draw_idle()
            return

        steps0       = [r["step"]             for r in w0_rows]
        q_states     = [r["q_states"]         for r in w0_rows]
        q_coverage   = [r["q_coverage"]       for r in w0_rows]
        avg_q        = [r["avg_q"]            for r in w0_rows]
        std_q        = [r["std_q"]            for r in w0_rows]
        explore_rate = [r["exploration_rate"] for r in w0_rows]

        # --- Q-table coverage (worker 0 only) ---
        ax_qtable.cla()
        ax_qtable_r.cla()
        ax_qtable_r.set_visible(False)
        ax_qtable.set_title("Q-Table Coverage (W0)")
        ax_qtable.set_xlabel("Step")
        ax_qtable.set_ylabel("Coverage %")
        ax_qtable.set_ylim(0, 100)
        ax_qtable.plot(steps0, q_coverage, color="tab:orange", label="Coverage %")

        # --- Q-values (worker 0 only) ---
        ax_qval.cla()
        ax_qval.set_title("Q-Values (W0)")
        ax_qval.set_xlabel("Step")
        ax_qval.set_ylabel("Q")
        ax_qval.plot(steps0, avg_q, color="tab:blue", label="avg Q")
        ax_qval.fill_between(
            steps0,
            [a - s for a, s in zip(avg_q, std_q)],
            [a + s for a, s in zip(avg_q, std_q)],
            alpha=0.2, color="tab:blue", label="±std"
        )
        ax_qval.legend(fontsize=8)

        # --- Exploration rate (global across all workers) ---
        ax_explore.cla()
        ax_explore.set_title("Exploration Rate (global)")
        ax_explore.set_xlabel("Step")
        ax_explore.set_ylabel("%")
        ax_explore.plot(steps0, explore_rate, color="tab:red")

        fig.tight_layout(pad=3.0)
        fig.canvas.draw_idle()


class PolicyGradientLivePlotter(Plotter):
    """Live plot window for policy-gradient (REINFORCE) training statistics.

    Shows:
        1) Average reward (last N episodes) over steps
        2) Policy loss over steps
        3) Episode return and entropy over steps
        4) Bias terms b_0, b_1, b_2 over steps

    Args:
        logger:           A PolicyGradientStatsLogger instance to read from.
        refresh_interval: Seconds between plot refreshes. Default: 5.0
    """

    def __init__(self, logger: PolicyGradientStatsLogger, refresh_interval: float = 5.0,
                 reward_window: int = 20) -> None:
        super().__init__(refresh_interval=refresh_interval)
        self._logger = logger
        self._reward_window = reward_window
        self._thread: threading.Thread | None = None

    def start(self, shutdown_event: threading.Event) -> None:
        """Start the background plotting thread."""
        self._thread = threading.Thread(
            target=self._run,
            args=(shutdown_event,),
            daemon=True,
            name="pg-live-plotter",
        )
        self._thread.start()

    def _run(self, shutdown_event: threading.Event) -> None:
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle("Policy Gradient Training Progress", fontsize=13)
        fig.tight_layout(pad=3.0)

        ax_reward, ax_loss, ax_return, ax_bias = axes.flatten()

        ax_reward.set_title("Avg Reward (pooled)")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")

        ax_loss.set_title("Policy Loss")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")

        ax_return.set_title("Episode Return / Entropy")
        ax_return.set_xlabel("Step")
        ax_return.set_ylabel("Return")
        ax_return_r = ax_return.twinx()
        ax_return_r.set_ylabel("Entropy")

        ax_bias.set_title("Bias Terms (b_i)")
        ax_bias.set_xlabel("Step")
        ax_bias.set_ylabel("b")

        plt.show(block=False)

        while not shutdown_event.is_set():
            try:
                self._redraw(fig, ax_reward, ax_loss, ax_return, ax_return_r, ax_bias)
            except Exception as e:
                print(f"[PolicyGradientLivePlotter] Draw error: {e}")

            for _ in range(int(self._refresh_interval / 0.5)):
                if shutdown_event.is_set():
                    break
                plt.pause(0.5)

        plt.close(fig)

    def _redraw(self, fig: plt.Figure, ax_reward: plt.Axes, ax_loss: plt.Axes,
                ax_return: plt.Axes, ax_return_r: plt.Axes, ax_bias: plt.Axes) -> None:
        """Pull latest data from the logger and redraw all subplots."""
        with self._logger._lock:
            snapshot = list(self._logger._buffer)

        if len(snapshot) < 2:
            return

        # Only worker 0 rows are used. avg_reward is computed from the shared
        # episode reward pool that all workers feed into, so it already reflects
        # all workers. Agent weights (W, b) are shared, so every worker would
        # log identical snapshots — using worker 0 avoids redundant data.
        by_worker: dict[int, list[dict]] = {}
        for r in snapshot:
            by_worker.setdefault(r["worker_id"], []).append(r)

        w0_rows = by_worker.get(0, [])
        if len(w0_rows) < 2:
            return

        steps = [r["step"] for r in w0_rows]

        # --- Avg reward (from shared pool, recorded by worker 0) ---
        ax_reward.cla()
        ax_reward.set_title(f"Avg Reward (pooled, window={self._reward_window} eps)")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")
        ax_reward.plot(steps, [r["avg_reward"] for r in w0_rows], color="tab:blue")

        # --- Policy loss ---
        ax_loss.cla()
        ax_loss.set_title("Policy Loss")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.plot(steps, [r.get("pg_loss", 0.0) for r in w0_rows], color="tab:red")

        # --- Episode return + entropy ---
        ax_return.cla()
        ax_return_r.cla()
        ax_return.set_title("Episode Return / Entropy")
        ax_return.set_xlabel("Step")
        ax_return.set_ylabel("Return")
        ax_return.plot(steps, [r.get("episode_return", 0.0) for r in w0_rows],
                       color="tab:green", label="Episode return")

        ax_return_r.set_ylabel("Entropy")
        ax_return_r.plot(steps, [r.get("avg_entropy", 0.0) for r in w0_rows],
                         color="tab:orange", label="Entropy")

        # Merge handles from both y-axes into a single legend.
        # Each twin axis keeps its own handle list, so we must collect both.
        handles_l, labels_l = ax_return.get_legend_handles_labels()
        handles_r, labels_r = ax_return_r.get_legend_handles_labels()
        ax_return.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8)

        # --- Bias terms ---
        ax_bias.cla()
        ax_bias.set_title("Bias Terms (b_0, b_1, b_2)")
        ax_bias.set_xlabel("Step")
        ax_bias.set_ylabel("b")
        ax_bias.plot(steps, [r.get("b_0", 0.0) for r in w0_rows], color="tab:blue",   label="b_0")
        ax_bias.plot(steps, [r.get("b_1", 0.0) for r in w0_rows], color="tab:orange", label="b_1")
        ax_bias.plot(steps, [r.get("b_2", 0.0) for r in w0_rows], color="tab:green",  label="b_2")
        ax_bias.legend(fontsize=8)

        fig.tight_layout(pad=3.0)
        fig.canvas.draw_idle()
