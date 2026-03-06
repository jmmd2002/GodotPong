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
import matplotlib
matplotlib.use("TkAgg")   # must be set before importing pyplot
import matplotlib.pyplot as plt

from stats_logger import QLearningStatsLogger


class QLearningLivePlotter:
    """
    Live plot window for Q-learning training statistics.

    Args:
        logger:           A QLearningStatsLogger instance to read from.
        refresh_interval: Seconds between plot refreshes. Default: 5.0
    """

    def __init__(self, logger: QLearningStatsLogger, refresh_interval: float = 5.0,
                 reward_window: int = 20) -> None:
        self._logger = logger
        self._refresh_interval = refresh_interval
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

        ax_qtable.set_title("Q-Table Size & Coverage")
        ax_qtable.set_xlabel("Step")
        ax_qtable.set_ylabel("States", color="tab:blue")
        ax_qtable_r = ax_qtable.twinx()
        ax_qtable_r.set_ylabel("Coverage %", color="tab:orange")

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

    def _redraw(self, fig, ax_reward, ax_qtable, ax_qtable_r, ax_qval, ax_explore) -> None:
        """Pull latest data from the logger and redraw all subplots."""
        with self._logger._lock:
            snapshot = list(self._logger._buffer)

        if len(snapshot) < 2:
            return  # not enough data yet

        # Group records by worker_id
        by_worker: dict[int, list[dict]] = {}
        for r in snapshot:
            by_worker.setdefault(r["worker_id"], []).append(r)

        # Worker 0 rows only — used for Q-table plots
        w0_rows = by_worker.get(0, [])

        # Color cycle: one colour per worker
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # --- Avg reward: one line per worker ---
        ax_reward.cla()
        ax_reward.set_title(f"Avg Reward (window={self._reward_window} eps)")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")
        for wid, rows in sorted(by_worker.items()):
            steps = [r["step"] for r in rows]
            rewards = [r["avg_reward"] for r in rows]
            ax_reward.plot(steps, rewards, color=colors[wid % len(colors)], label=f"W{wid}")
        if len(by_worker) > 1:
            ax_reward.legend(fontsize=8)

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

        # --- Q-table size & coverage (worker 0 only) ---
        ax_qtable.cla()
        ax_qtable_r.cla()
        ax_qtable.set_title("Q-Table Size & Coverage (W0)")
        ax_qtable.set_xlabel("Step")
        ax_qtable.set_ylabel("States", color="tab:blue")
        ax_qtable_r.set_ylabel("Coverage %", color="tab:orange")
        ax_qtable.plot(steps0, q_states, color="tab:blue", label="Q-states")
        ax_qtable_r.plot(steps0, q_coverage, color="tab:orange", linestyle="--", label="Coverage %")

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

        # --- Exploration rate (worker 0 only) ---
        ax_explore.cla()
        ax_explore.set_title("Exploration Rate (W0)")
        ax_explore.set_xlabel("Step")
        ax_explore.set_ylabel("%")
        ax_explore.plot(steps0, explore_rate, color="tab:red")

        fig.tight_layout(pad=3.0)
        fig.canvas.draw_idle()
