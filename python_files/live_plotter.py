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

from stats_logger import QLearningStatsLogger, PolicyGradientStatsLogger, PolicyGradientDNNStatsLogger, A2CStatsLogger


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
        min_q        = [r.get("min_q", 0.0)   for r in w0_rows]
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
        # avg_q ± std_q shows the central tendency and spread.
        # min_q (the floor of all Q-values) is a useful extra signal: a rising
        # min_q indicates that even the least-visited state-action pairs are
        # accumulating positive value, while a persistently negative or
        # diverging min_q can flag instability early.
        ax_qval.cla()
        ax_qval.set_title("Q-Values (W0)")
        ax_qval.set_xlabel("Step")
        ax_qval.set_ylabel("Q")
        ax_qval.plot(steps0, avg_q, color="tab:blue", label="avg Q")
        ax_qval.fill_between(
            steps0,
            [a - s for a, s in zip(avg_q, std_q)],
            [a + s for a, s in zip(avg_q, std_q)],
            alpha=0.2, color="tab:blue", label="±std",
        )
        ax_qval.plot(steps0, min_q, color="tab:red", linewidth=0.8,
                     linestyle="--", label="min Q")
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
        3) Policy entropy over steps
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

        ax_loss.set_title("Policy Loss / Grad Norm")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss_r = ax_loss.twinx()
        ax_loss_r.set_ylabel("Grad Norm")

        ax_return.set_title("Entropy")
        ax_return.set_xlabel("Step")
        ax_return.set_ylabel("Entropy")

        ax_bias.set_title("Bias Terms (b_i)")
        ax_bias.set_xlabel("Step")
        ax_bias.set_ylabel("b")

        plt.show(block=False)

        while not shutdown_event.is_set():
            try:
                self._redraw(fig, ax_reward, ax_loss, ax_loss_r, ax_return, ax_bias)
            except Exception as e:
                print(f"[PolicyGradientLivePlotter] Draw error: {e}")

            for _ in range(int(self._refresh_interval / 0.5)):
                if shutdown_event.is_set():
                    break
                plt.pause(0.5)

        plt.close(fig)

    def _redraw(self, fig: plt.Figure, ax_reward: plt.Axes, ax_loss: plt.Axes,
                ax_loss_r: plt.Axes, ax_return: plt.Axes,
                ax_bias: plt.Axes) -> None:
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

        # --- Policy loss + gradient norms (twin axis) ---
        # Loss and gradient magnitude are naturally co-located diagnostics:
        # a loss drop accompanied by a falling grad_norm confirms stable
        # convergence; a loss plateau with large grad_norm suggests the
        # updates are oscillating rather than converging.
        ax_loss.cla()
        ax_loss_r.cla()
        ax_loss.set_title("Policy Loss / Grad Norm")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.plot(steps, [r.get("pg_loss", 0.0) for r in w0_rows],
                     color="tab:red", label="Loss")
        ax_loss_r.set_ylabel("Grad Norm")
        ax_loss_r.plot(steps, [r.get("grad_norm_W", 0.0) for r in w0_rows],
                       color="tab:purple", linewidth=0.8, label="‖∇W‖")
        ax_loss_r.plot(steps, [r.get("grad_norm_b", 0.0) for r in w0_rows],
                       color="tab:brown", linewidth=0.8, linestyle="--", label="‖∇b‖")
        handles_l, labels_l = ax_loss.get_legend_handles_labels()
        handles_r, labels_r = ax_loss_r.get_legend_handles_labels()
        ax_loss.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8)

        # --- Entropy ---
        ax_return.cla()
        ax_return.set_title("Entropy")
        ax_return.set_xlabel("Step")
        ax_return.set_ylabel("Entropy")
        ax_return.plot(steps, [r.get("avg_entropy", 0.0) for r in w0_rows],
                       color="tab:orange", label="Entropy")

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


class PolicyGradientDNNLivePlotter(Plotter):
    """
    Live plot window for the REINFORCE DNN (JAX MLP) agent's training statistics.

    Four subplots:
        1. Avg reward (pooled, window=N episodes) over steps
        2. Policy loss over steps
        3. Policy entropy over steps
        4. Per-layer mean |W| (one line per weight matrix, left axis)
           + gradient norm across all layers (right twin axis)

    Subplot 4 replaces the bias-terms chart used by the linear plotter:
    for a multi-layer network, per-weight statistics and the unified gradient
    norm are far more informative diagnostics than individual output biases.
    A sharply rising grad_norm signals gradient explosion; a collapsed avg_w
    can indicate dead neurons or too-aggressive weight decay.

    Args:
        logger:           A PolicyGradientDNNStatsLogger instance to read from.
        refresh_interval: Seconds between plot refreshes. Default: 5.0
        reward_window:    Episode window size used in the avg reward title label.
    """

    def __init__(self, logger: PolicyGradientDNNStatsLogger,
                 refresh_interval: float = 5.0, reward_window: int = 20) -> None:
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
            name="pg-dnn-live-plotter",
        )
        self._thread.start()

    def _run(self, shutdown_event: threading.Event) -> None:
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle("Policy Gradient DNN Training Progress", fontsize=13)
        fig.tight_layout(pad=3.0)

        ax_reward, ax_loss, ax_return, ax_weights = axes.flatten()

        ax_reward.set_title("Avg Reward (pooled)")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")

        ax_loss.set_title("Avg Policy Loss (pooled)")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")

        ax_return.set_title("Avg Entropy (pooled)")
        ax_return.set_xlabel("Step")
        ax_return.set_ylabel("Entropy")

        ax_weights.set_title("Per-Layer |w| / Grad Norm")
        ax_weights.set_xlabel("Step")
        ax_weights.set_ylabel("|w|")
        ax_weights_r = ax_weights.twinx()
        ax_weights_r.set_ylabel("Grad Norm")

        plt.show(block=False)

        while not shutdown_event.is_set():
            try:
                self._redraw(
                    fig,
                    ax_reward, ax_loss,
                    ax_return,
                    ax_weights, ax_weights_r,
                )
            except Exception as e:
                print(f"[PolicyGradientDNNLivePlotter] Draw error: {e}")

            for _ in range(int(self._refresh_interval / 0.5)):
                if shutdown_event.is_set():
                    break
                plt.pause(0.5)

        plt.close(fig)

    def _redraw(
        self,
        fig:          plt.Figure,
        ax_reward:    plt.Axes,
        ax_loss:      plt.Axes,
        ax_return:    plt.Axes,
        ax_weights:   plt.Axes,
        ax_weights_r: plt.Axes,
    ) -> None:
        """Pull latest data from the logger and redraw all subplots."""
        with self._logger._lock:
            snapshot = list(self._logger._buffer)

        if len(snapshot) < 2:
            return

        # Only worker 0 rows carry agent-internal stats (weights, grad_norm,
        # entropy). avg_reward is computed from the shared episode pool so it
        # already reflects all workers — no per-worker splitting needed.
        by_worker: dict[int, list[dict]] = {}
        for r in snapshot:
            by_worker.setdefault(r["worker_id"], []).append(r)

        w0_rows = by_worker.get(0, [])
        if len(w0_rows) < 2:
            return

        steps = [r["step"] for r in w0_rows]

        # --- 1. Avg reward ---
        ax_reward.cla()
        ax_reward.set_title(f"Avg Reward (pooled, window={self._reward_window} eps)")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")
        ax_reward.plot(steps, [r["avg_reward"] for r in w0_rows], color="tab:blue")

        # --- 2. Policy loss ---
        ax_loss.cla()
        ax_loss.set_title(f"Avg Policy Loss (pooled, window={self._reward_window} eps)")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.plot(steps, [r.get("avg_loss", 0.0) for r in w0_rows], color="tab:red")

        # --- 3. Entropy ---
        ax_return.cla()
        ax_return.set_title(f"Avg Entropy (pooled, window={self._reward_window} eps)")
        ax_return.set_xlabel("Step")
        ax_return.set_ylabel("Entropy")
        ax_return.plot(
            steps, [r.get("avg_entropy", 0.0) for r in w0_rows],
            color="tab:orange", label="Entropy",
        )

        # --- 4. Per-layer mean |W| + grad norm (twin axis) ---
        # One line per weight matrix (avg_w_0 = input→hidden1, …, avg_w_N = hidden→output).
        # grad_norm (right axis) spans all layers: a sudden spike indicates
        # gradient explosion; a value staying at 0 suggests vanishing gradients.
        layer_keys = sorted(k for k in w0_rows[0] if k.startswith("avg_w_"))
        gnorm = [r.get("grad_norm", 0.0) for r in w0_rows]

        ax_weights.cla()
        ax_weights_r.cla()
        ax_weights.set_title("Per-Layer |w| / Grad Norm")
        ax_weights.set_xlabel("Step")
        ax_weights.set_ylabel("|w|")
        colors = plt.cm.tab10.colors
        num_layers = len(layer_keys)
        for i, key in enumerate(layer_keys):
            label = f"W{i} (out)" if i == num_layers - 1 else f"W{i}"
            ax_weights.plot(steps, [r.get(key, 0.0) for r in w0_rows],
                            color=colors[i % len(colors)], label=label)
        ax_weights_r.set_ylabel("Grad Norm")
        ax_weights_r.plot(steps, gnorm, color="tab:red", label="grad norm", linewidth=0.8)

        handles_l, labels_l = ax_weights.get_legend_handles_labels()
        handles_r, labels_r = ax_weights_r.get_legend_handles_labels()
        ax_weights.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8)

        fig.tight_layout(pad=3.0)
        fig.canvas.draw_idle()


class A2CLivePlotter(Plotter):
    """
    Live plot window for the Advantage Actor-Critic (A2C) agent.

    Four subplots:
        1. Avg reward (pooled, window=N episodes) over steps
        2. Actor loss (left axis) + Critic loss (right twin axis) over steps
        3. Policy entropy (left axis) + Mean |advantage| (right twin axis) over steps
        4. Per-layer actor mean |W| (left axis) + Actor & critic grad norms
           (right twin axis) over steps

    Args:
        logger:           An A2CStatsLogger instance to read from.
        refresh_interval: Seconds between plot refreshes. Default: 5.0
        reward_window:    Episode window size used in the avg reward title label.
    """

    def __init__(self, logger: A2CStatsLogger,
                 refresh_interval: float = 5.0, reward_window: int = 20) -> None:
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
            name="a2c-live-plotter",
        )
        self._thread.start()

    def _run(self, shutdown_event: threading.Event) -> None:
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle("A2C Training Progress", fontsize=13)
        fig.tight_layout(pad=3.0)

        ax_reward, ax_loss, ax_entropy, ax_weights = axes.flatten()

        ax_reward.set_title("Avg Reward (pooled)")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")

        ax_loss.set_title("Actor Loss / Critic Loss")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Actor Loss")
        ax_loss_r = ax_loss.twinx()
        ax_loss_r.set_ylabel("Critic Loss")

        ax_entropy.set_title("Entropy / Mean |Advantage|")
        ax_entropy.set_xlabel("Step")
        ax_entropy.set_ylabel("Entropy")
        ax_entropy_r = ax_entropy.twinx()
        ax_entropy_r.set_ylabel("Mean |Advantage|")

        ax_weights.set_title("Per-Layer Actor |w| / Grad Norms")
        ax_weights.set_xlabel("Step")
        ax_weights.set_ylabel("|w|")
        ax_weights_r = ax_weights.twinx()
        ax_weights_r.set_ylabel("Grad Norm")

        plt.show(block=False)

        while not shutdown_event.is_set():
            try:
                self._redraw(
                    fig,
                    ax_reward,
                    ax_loss,    ax_loss_r,
                    ax_entropy, ax_entropy_r,
                    ax_weights, ax_weights_r,
                )
            except Exception as e:
                print(f"[A2CLivePlotter] Draw error: {e}")

            for _ in range(int(self._refresh_interval / 0.5)):
                if shutdown_event.is_set():
                    break
                plt.pause(0.5)

        plt.close(fig)

    def _redraw(
        self,
        fig:          plt.Figure,
        ax_reward:    plt.Axes,
        ax_loss:      plt.Axes,
        ax_loss_r:    plt.Axes,
        ax_entropy:   plt.Axes,
        ax_entropy_r: plt.Axes,
        ax_weights:   plt.Axes,
        ax_weights_r: plt.Axes,
    ) -> None:
        """Pull latest data from the logger and redraw all subplots."""
        with self._logger._lock:
            snapshot = list(self._logger._buffer)

        if len(snapshot) < 2:
            return

        by_worker: dict[int, list[dict]] = {}
        for r in snapshot:
            by_worker.setdefault(r["worker_id"], []).append(r)

        w0_rows = by_worker.get(0, [])
        if len(w0_rows) < 2:
            return

        steps = [r["step"] for r in w0_rows]

        # --- 1. Avg reward ---
        ax_reward.cla()
        ax_reward.set_title(f"Avg Reward (pooled, window={self._reward_window} eps)")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")
        ax_reward.plot(steps, [r["avg_reward"] for r in w0_rows], color="tab:blue")

        # --- 2. Actor loss (left) + Critic loss (right) ---
        # Critic loss is typically much larger early in training (V(s) is far
        # from G_t), so twin axes let both signals remain visible.
        ax_loss.cla()
        ax_loss_r.cla()
        ax_loss.set_title(f"Avg Actor Loss / Avg Critic Loss (window={self._reward_window})")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Actor Loss")
        ax_loss.plot(
            steps, [r.get("avg_actor_loss", 0.0) for r in w0_rows],
            color="tab:red", label="actor loss",
        )
        ax_loss_r.set_ylabel("Critic Loss")
        ax_loss_r.plot(
            steps, [r.get("avg_critic_loss", 0.0) for r in w0_rows],
            color="tab:purple", linewidth=0.8, label="critic loss",
        )
        handles_l, labels_l = ax_loss.get_legend_handles_labels()
        handles_r, labels_r = ax_loss_r.get_legend_handles_labels()
        ax_loss.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8)

        # --- 3. Entropy (left) + Mean |advantage| (right) ---
        # A shrinking mean |A_t| alongside falling entropy confirms V(s) is
        # converging and the policy is specialising correctly.
        ax_entropy.cla()
        ax_entropy_r.cla()
        ax_entropy.set_title(f"Avg Entropy / Mean |Advantage| (window={self._reward_window})")
        ax_entropy.set_xlabel("Step")
        ax_entropy.set_ylabel("Entropy")
        ax_entropy.plot(
            steps, [r.get("avg_entropy", 0.0) for r in w0_rows],
            color="tab:orange", label="entropy",
        )
        ax_entropy_r.set_ylabel("Mean |Advantage|")
        ax_entropy_r.plot(
            steps, [r.get("mean_advantage", 0.0) for r in w0_rows],
            color="tab:green", linewidth=0.8, label="mean |A|",
        )
        handles_l, labels_l = ax_entropy.get_legend_handles_labels()
        handles_r, labels_r = ax_entropy_r.get_legend_handles_labels()
        ax_entropy.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8)

        # --- 4. Per-layer actor |W| (left) + actor & critic grad norms (right) ---
        layer_keys   = sorted(k for k in w0_rows[0] if k.startswith("avg_w_actor_"))
        actor_gnorm  = [r.get("actor_grad_norm",  0.0) for r in w0_rows]
        critic_gnorm = [r.get("critic_grad_norm", 0.0) for r in w0_rows]

        ax_weights.cla()
        ax_weights_r.cla()
        ax_weights.set_title("Per-Layer Actor |w| / Grad Norms")
        ax_weights.set_xlabel("Step")
        ax_weights.set_ylabel("|w|")
        colors = plt.cm.tab10.colors
        num_layers = len(layer_keys)
        for i, key in enumerate(layer_keys):
            label = f"W{i} (out)" if i == num_layers - 1 else f"W{i}"
            ax_weights.plot(
                steps, [r.get(key, 0.0) for r in w0_rows],
                color=colors[i % len(colors)], label=label,
            )
        ax_weights_r.set_ylabel("Grad Norm")
        ax_weights_r.plot(steps, actor_gnorm,  color="tab:red",    linewidth=0.8, label="actor ‖∇‖")
        ax_weights_r.plot(steps, critic_gnorm, color="tab:purple", linewidth=0.8,
                          linestyle="--", label="critic ‖∇‖")

        handles_l, labels_l = ax_weights.get_legend_handles_labels()
        handles_r, labels_r = ax_weights_r.get_legend_handles_labels()
        ax_weights.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8)

        fig.tight_layout(pad=3.0)
        fig.canvas.draw_idle()
