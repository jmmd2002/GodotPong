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
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle("Policy Gradient DNN — Training Progress", fontsize=18, fontweight="bold")
        fig.tight_layout(pad=4.5)

        (ax_reward, ax_loss,
         ax_entropy, ax_grad,
         ax_ratio,  ax_dead) = axes.flatten()

        # Create twin axes once here — creating them inside _redraw on every
        # refresh would stack new axes on top of old ones, causing the
        # "two layers overlapping" artefact.
        ax_kl      = ax_entropy.twinx()
        ax_ratio_r = ax_ratio.twinx()

        plt.show(block=False)

        while not shutdown_event.is_set():
            try:
                self._redraw(fig, ax_reward, ax_loss, ax_entropy, ax_kl, ax_grad, ax_ratio, ax_ratio_r, ax_dead)
            except Exception as e:
                print(f"[PolicyGradientDNNLivePlotter] Draw error: {e}")

            for _ in range(int(self._refresh_interval / 0.5)):
                if shutdown_event.is_set():
                    break
                plt.pause(0.5)

        plt.close(fig)

    def _redraw(
        self,
        fig:        plt.Figure,
        ax_reward:  plt.Axes,
        ax_loss:    plt.Axes,
        ax_entropy: plt.Axes,
        ax_kl:      plt.Axes,
        ax_grad:    plt.Axes,
        ax_ratio:   plt.Axes,
        ax_ratio_r: plt.Axes,
        ax_dead:    plt.Axes,
    ) -> None:
        """Pull latest data from the logger and redraw all six subplots."""
        TITLE_FS  = 14
        LABEL_FS  = 12
        LEGEND_FS = 12
        TICK_FS   = 11
        LW        = 1.8

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

        steps  = [r["step"] for r in w0_rows]
        colors = plt.cm.tab10.colors

        def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
            ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=LABEL_FS)
            ax.set_ylabel(ylabel, fontsize=LABEL_FS)
            ax.tick_params(labelsize=TICK_FS)

        # ----------------------------------------------------------------
        # 1. Avg reward
        # ----------------------------------------------------------------
        ax_reward.cla()
        _style(ax_reward,
               f"Avg Reward  (window = {self._reward_window} eps)",
               "Step", "Reward")
        ax_reward.plot(steps, [r["avg_reward"] for r in w0_rows],
                       color="tab:blue", linewidth=LW)

        # ----------------------------------------------------------------
        # 2. Policy loss (windowed avg)
        # ----------------------------------------------------------------
        ax_loss.cla()
        _style(ax_loss,
               f"Avg Policy Loss  (window = {self._reward_window} eps)",
               "Step", "Loss")
        ax_loss.plot(steps, [r.get("avg_loss", 0.0) for r in w0_rows],
                     color="tab:red", linewidth=LW)

        # ----------------------------------------------------------------
        # 3. Entropy (left) + KL divergence (right twin)
        #    Entropy falling = policy sharpening (good).
        #    KL ≈ 0 every episode = updates too small to matter.
        # ----------------------------------------------------------------
        ax_entropy.cla()
        ax_kl.cla()
        _style(ax_entropy,
               f"Entropy  +  KL Divergence (per episode)",
               "Step", "Entropy  H(π)")
        ax_entropy.plot(steps, [r.get("avg_entropy", 0.0) for r in w0_rows],
                        color="tab:orange", linewidth=LW, label="Entropy H(π)")
        ax_kl.set_ylabel("KL divergence", fontsize=LABEL_FS)
        ax_kl.tick_params(labelsize=TICK_FS)
        ax_kl.plot(steps, [r.get("kl_div", 0.0) for r in w0_rows],
                   color="tab:purple", linewidth=LW * 0.85, linestyle="--", label="KL div")
        h_l, lb_l = ax_entropy.get_legend_handles_labels()
        h_r, lb_r = ax_kl.get_legend_handles_labels()
        ax_entropy.legend(h_l + h_r, lb_l + lb_r, fontsize=LEGEND_FS, loc="upper right")

        # ----------------------------------------------------------------
        # 4. Per-layer gradient norms  ‖∇W_i‖
        #    Vanishing: early-layer norms near zero while last layer is large.
        #    Exploding: any layer norm spikes.
        # ----------------------------------------------------------------
        grad_keys = sorted(k for k in w0_rows[0] if k.startswith("grad_norm_"))
        ax_grad.cla()
        _style(ax_grad, "Per-Layer Gradient Norms  ‖∇W_i‖", "Step", "‖∇W_i‖")
        num_gl = len(grad_keys)
        for i, key in enumerate(grad_keys):
            suffix = key.replace("grad_norm_", "")
            label  = f"W{suffix} (output)" if i == num_gl - 1 else f"W{suffix}"
            ax_grad.plot(steps, [r.get(key, 0.0) for r in w0_rows],
                         color=colors[i % len(colors)], linewidth=LW, label=label)
        if grad_keys:
            ax_grad.legend(fontsize=LEGEND_FS)

        # ----------------------------------------------------------------
        # 5. Update ratios  α‖∇W_i‖ / ‖W_i‖  (Karpathy rule-of-thumb ≈ 1e-3)
        #    Left axis: update ratios — healthy ≈ 1e-3.
        #    Right axis: per-layer mean |W| — tracks weight magnitude growth.
        # ----------------------------------------------------------------
        ratio_keys     = sorted(k for k in w0_rows[0] if k.startswith("update_ratio_"))
        layer_w_keys   = sorted(k for k in w0_rows[0] if k.startswith("avg_w_"))
        ax_ratio.cla()
        ax_ratio_r.cla()
        _style(ax_ratio,
               "Update Ratios  α‖∇W_i‖ / ‖W_i‖  (≈ 1e-3)  /  Per-Layer |W|",
               "Step", "α‖∇W‖ / ‖W‖")
        num_rl = len(ratio_keys)
        for i, key in enumerate(ratio_keys):
            suffix = key.replace("update_ratio_", "")
            label  = f"W{suffix} (output)" if i == num_rl - 1 else f"W{suffix}"
            ax_ratio.plot(steps, [r.get(key, 0.0) for r in w0_rows],
                          color=colors[i % len(colors)], linewidth=LW, label=label)
        if ratio_keys:
            ax_ratio.axhline(y=1e-3, color="dimgray", linestyle=":", linewidth=1.2,
                             label="target  1e-3")
        ax_ratio_r.set_ylabel("|W|", fontsize=LABEL_FS)
        ax_ratio_r.tick_params(labelsize=TICK_FS)
        num_wl = len(layer_w_keys)
        for i, key in enumerate(layer_w_keys):
            suffix = key.replace("avg_w_", "")
            label  = f"|W{suffix}| (out)" if i == num_wl - 1 else f"|W{suffix}|"
            ax_ratio_r.plot(steps, [r.get(key, 0.0) for r in w0_rows],
                            color=colors[i % len(colors)],
                            linewidth=LW * 0.8, linestyle="--", label=label)
        h_l, lb_l = ax_ratio.get_legend_handles_labels()
        h_r, lb_r = ax_ratio_r.get_legend_handles_labels()
        ax_ratio.legend(h_l + h_r, lb_l + lb_r, fontsize=LEGEND_FS)

        # ----------------------------------------------------------------
        # 6. Dead ReLU %
        #    > 30 %: significant capacity loss — consider lower learning rate
        #            or weight re-initialisation.
        # ----------------------------------------------------------------
        ax_dead.cla()
        _style(ax_dead,
               "Dead ReLU %  (random 100-sample batch, hidden layers)",
               "Step", "Dead activations %")
        ax_dead.plot(steps, [r.get("dead_relu_pct", 0.0) for r in w0_rows],
                     color="tab:brown", linewidth=LW, label="Dead ReLU %")
        ax_dead.set_ylim(0, 100)
        ax_dead.axhline(y=30, color="tab:red", linestyle="--", linewidth=1.2,
                        label="30 % warning")
        ax_dead.legend(fontsize=LEGEND_FS)

        fig.tight_layout(pad=4.5)
        fig.canvas.draw_idle()


class A2CLivePlotter(Plotter):
    """
    Live plot window for the Advantage Actor-Critic (A2C) agent.

    Six subplots (3 × 2):
        1. Avg reward (pooled, window=N episodes) over steps
        2. Actor loss (left axis) + Critic loss (right twin axis) over steps
        3. Policy entropy (left axis) + Mean |advantage| (right twin axis) over steps
        4. Per-layer actor gradient norms  ‖∇W_i‖ over steps
        5. Per-layer actor update ratios   α‖∇W_i‖/‖W_i‖  (healthy ≈ 1e-3)
        6. Dead ReLU % in actor hidden layers  (warning threshold at 30 %)

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
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle("A2C Training Progress", fontsize=18, fontweight="bold")
        fig.tight_layout(pad=4.5)

        (ax_reward, ax_loss,
         ax_entropy, ax_grad,
         ax_ratio,  ax_dead) = axes.flatten()

        # Create twin axes once — creating them inside _redraw on every
        # refresh stacks new axes on top of old ones, causing overlap.
        ax_loss_r    = ax_loss.twinx()
        ax_entropy_r = ax_entropy.twinx()
        ax_ratio_r   = ax_ratio.twinx()

        plt.show(block=False)

        while not shutdown_event.is_set():
            try:
                self._redraw(fig, ax_reward,
                             ax_loss, ax_loss_r,
                             ax_entropy, ax_entropy_r,
                             ax_grad,
                             ax_ratio, ax_ratio_r,
                             ax_dead)
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
        ax_grad:      plt.Axes,
        ax_ratio:     plt.Axes,
        ax_ratio_r:   plt.Axes,
        ax_dead:      plt.Axes,
    ) -> None:
        """Pull latest data from the logger and redraw all six subplots."""
        TITLE_FS  = 14
        LABEL_FS  = 12
        LEGEND_FS = 12
        TICK_FS   = 11
        LW        = 1.8

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

        steps  = [r["step"] for r in w0_rows]
        colors = plt.cm.tab10.colors

        def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
            ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=LABEL_FS)
            ax.set_ylabel(ylabel, fontsize=LABEL_FS)
            ax.tick_params(labelsize=TICK_FS)

        # ----------------------------------------------------------------
        # 1. Avg reward
        # ----------------------------------------------------------------
        ax_reward.cla()
        _style(ax_reward,
               f"Avg Reward  (window = {self._reward_window} eps)",
               "Step", "Reward")
        ax_reward.plot(steps, [r["avg_reward"] for r in w0_rows],
                       color="tab:blue", linewidth=LW)

        # ----------------------------------------------------------------
        # 2. Actor loss (left) + Critic loss (right)
        #    Critic loss is typically much larger early in training (V(s) is far
        #    from G_t), so twin axes keep both signals visible.
        # ----------------------------------------------------------------
        ax_loss.cla()
        ax_loss_r.cla()
        _style(ax_loss,
               f"Avg Actor Loss / Avg Critic Loss  (window = {self._reward_window})",
               "Step", "Actor Loss")
        ax_loss.plot(steps, [r.get("avg_actor_loss",  0.0) for r in w0_rows],
                     color="tab:red", linewidth=LW, label="actor loss")
        ax_loss_r.set_ylabel("Critic Loss", fontsize=LABEL_FS)
        ax_loss_r.tick_params(labelsize=TICK_FS)
        ax_loss_r.plot(steps, [r.get("avg_critic_loss", 0.0) for r in w0_rows],
                       color="tab:purple", linewidth=LW * 0.8, label="critic loss")
        h_l, lb_l = ax_loss.get_legend_handles_labels()
        h_r, lb_r = ax_loss_r.get_legend_handles_labels()
        ax_loss.legend(h_l + h_r, lb_l + lb_r, fontsize=LEGEND_FS)

        # ----------------------------------------------------------------
        # 3. Entropy (left) + Mean |advantage| (right)
        #    A shrinking mean |A_t| alongside falling entropy confirms
        #    V(s) is converging and the policy is specialising correctly.
        # ----------------------------------------------------------------
        ax_entropy.cla()
        ax_entropy_r.cla()
        _style(ax_entropy,
               f"Avg Entropy / Mean |Advantage|  (window = {self._reward_window})",
               "Step", "Entropy  H(π)")
        ax_entropy.plot(steps, [r.get("avg_entropy",    0.0) for r in w0_rows],
                        color="tab:orange", linewidth=LW, label="entropy H(π)")
        ax_entropy_r.set_ylabel("Mean |Advantage|", fontsize=LABEL_FS)
        ax_entropy_r.tick_params(labelsize=TICK_FS)
        ax_entropy_r.plot(steps, [r.get("mean_advantage", 0.0) for r in w0_rows],
                          color="tab:green", linewidth=LW * 0.8, label="mean |A|")
        h_l, lb_l = ax_entropy.get_legend_handles_labels()
        h_r, lb_r = ax_entropy_r.get_legend_handles_labels()
        ax_entropy.legend(h_l + h_r, lb_l + lb_r, fontsize=LEGEND_FS)

        # ----------------------------------------------------------------
        # 4. Per-layer actor gradient norms  ‖∇W_i‖
        #    Vanishing: early-layer norms near zero while last layer is large.
        #    Exploding: any layer norm spikes.
        # ----------------------------------------------------------------
        grad_keys = sorted(k for k in w0_rows[0] if k.startswith("grad_norm_actor_"))
        ax_grad.cla()
        _style(ax_grad, "Per-Layer Actor Gradient Norms  ‖∇W_i‖", "Step", "‖∇W_i‖")
        num_gl = len(grad_keys)
        for i, key in enumerate(grad_keys):
            suffix = key.replace("grad_norm_actor_", "")
            label  = f"W{suffix} (output)" if i == num_gl - 1 else f"W{suffix}"
            ax_grad.plot(steps, [r.get(key, 0.0) for r in w0_rows],
                         color=colors[i % len(colors)], linewidth=LW, label=label)
        # Also overlay the combined critic grad norm as a reference.
        ax_grad.plot(steps, [r.get("critic_grad_norm", 0.0) for r in w0_rows],
                     color="dimgray", linewidth=1.0, linestyle=":", label="critic ‖∇‖")
        if grad_keys or True:
            ax_grad.legend(fontsize=LEGEND_FS)

        # ----------------------------------------------------------------
        # 5. Actor + Critic update ratios α‖∇W_i‖/‖W_i‖ (left, same axis)
        #    + Critic |W| (right twin axis)
        #    Both update ratios share the left axis so they can be directly
        #    compared against each other and against the 1e-3 reference.
        #    Critic |W| on the right axis tracks weight magnitude growth.
        # ----------------------------------------------------------------
        actor_ratio_keys  = sorted(k for k in w0_rows[0] if k.startswith("update_ratio_actor_"))
        critic_ratio_keys = sorted(k for k in w0_rows[0] if k.startswith("update_ratio_critic_"))
        critic_layer_keys = sorted(k for k in w0_rows[0] if k.startswith("avg_w_critic_"))
        ax_ratio.cla()
        ax_ratio_r.cla()
        _style(ax_ratio,
               "Actor & Critic Update Ratios  α‖∇W_i‖/‖W_i‖  (≈ 1e-3)  /  Critic |W|",
               "Step", "α‖∇W‖ / ‖W‖")
        num_arl = len(actor_ratio_keys)
        for i, key in enumerate(actor_ratio_keys):
            suffix = key.replace("update_ratio_actor_", "")
            label  = f"actor W{suffix} (out)" if i == num_arl - 1 else f"actor W{suffix}"
            ax_ratio.plot(steps, [r.get(key, 0.0) for r in w0_rows],
                          color=colors[i % len(colors)], linewidth=LW, label=label)
        num_crl = len(critic_ratio_keys)
        for i, key in enumerate(critic_ratio_keys):
            suffix = key.replace("update_ratio_critic_", "")
            label  = f"critic W{suffix} (out)" if i == num_crl - 1 else f"critic W{suffix}"
            ax_ratio.plot(steps, [r.get(key, 0.0) for r in w0_rows],
                          color=colors[(i + 5) % len(colors)],
                          linewidth=LW, linestyle="--", label=label)
        ax_ratio.axhline(y=1e-3, color="dimgray", linestyle=":", linewidth=1.2,
                         label="target  1e-3")
        ax_ratio_r.set_ylabel("Critic  |W|", fontsize=LABEL_FS)
        ax_ratio_r.tick_params(labelsize=TICK_FS)
        num_cl = len(critic_layer_keys)
        for i, key in enumerate(critic_layer_keys):
            suffix = key.replace("avg_w_critic_", "")
            label  = f"|critic W{suffix}| (out)" if i == num_cl - 1 else f"|critic W{suffix}|"
            ax_ratio_r.plot(steps, [r.get(key, 0.0) for r in w0_rows],
                            color=colors[(i + 5) % len(colors)],
                            linewidth=LW * 0.5, linestyle=":", label=label)
        h_l, lb_l = ax_ratio.get_legend_handles_labels()
        h_r, lb_r = ax_ratio_r.get_legend_handles_labels()
        ax_ratio.legend(h_l + h_r, lb_l + lb_r, fontsize=LEGEND_FS)

        # ----------------------------------------------------------------
        # 6. Dead ReLU % (actor hidden layers)
        #    > 30 %: significant capacity loss — consider lower learning rate
        #            or weight re-initialisation.
        # ----------------------------------------------------------------
        ax_dead.cla()
        _style(ax_dead,
               "Dead ReLU %  (random 100-sample batch, actor hidden layers)",
               "Step", "Dead activations %")
        ax_dead.plot(steps, [r.get("dead_relu_pct", 0.0) for r in w0_rows],
                     color="tab:brown", linewidth=LW, label="Dead ReLU %")
        ax_dead.set_ylim(0, 100)
        ax_dead.axhline(y=30, color="tab:red", linestyle="--", linewidth=1.2,
                        label="30 % warning")
        ax_dead.legend(fontsize=LEGEND_FS)

        fig.tight_layout(pad=4.5)
        fig.canvas.draw_idle()


class PPOLivePlotter(Plotter):
    """
    Live plot window for the PPO-Clip agent.

    Four subplots:
        1. Avg reward (pooled, window=N episodes) over steps
        2. Actor loss (left axis) + Critic loss (right twin axis) over steps
        3. Policy entropy (left axis) + Clip fraction and Approx KL (right twin axis)
        4. Per-layer actor mean |W| (left axis) + Actor & critic grad norms
           (right twin axis) over steps

    The third subplot is the PPO-specific addition: clip_fraction and approx_kl
    are the key diagnostics for monitoring whether the policy is updating at a
    healthy rate.  clip_fraction > 0.3 or approx_kl > 0.1 both signal that the
    policy is moving too aggressively and hyperparameters need adjustment.

    Args:
        logger:           A PPOStatsLogger instance to read from.
        refresh_interval: Seconds between plot refreshes. Default: 5.0
        reward_window:    Episode window size shown in the avg reward title.
    """

    def __init__(self, logger, refresh_interval: float = 5.0, reward_window: int = 20) -> None:
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
            name="ppo-live-plotter",
        )
        self._thread.start()

    def _run(self, shutdown_event: threading.Event) -> None:
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle("PPO Training Progress", fontsize=13)
        fig.tight_layout(pad=3.0)

        ax_reward, ax_loss, ax_ppo, ax_weights = axes.flatten()

        ax_reward.set_title("Avg Reward (pooled)")
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Reward")

        ax_loss.set_title("Actor Loss / Critic Loss")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Actor Loss")
        ax_loss_r = ax_loss.twinx()
        ax_loss_r.set_ylabel("Critic Loss")

        ax_ppo.set_title("Entropy / Clip Fraction / Approx KL")
        ax_ppo.set_xlabel("Step")
        ax_ppo.set_ylabel("Entropy")
        ax_ppo_r = ax_ppo.twinx()
        ax_ppo_r.set_ylabel("Clip Fraction / Approx KL")

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
                    ax_ppo,     ax_ppo_r,
                    ax_weights, ax_weights_r,
                )
            except Exception as e:
                print(f"[PPOLivePlotter] Draw error: {e}")

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
        ax_ppo:       plt.Axes,
        ax_ppo_r:     plt.Axes,
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

        # --- 2. Actor loss (left) / Critic loss (right twin) ---
        ax_loss.cla()
        ax_loss_r.cla()
        ax_loss.set_title(f"Actor Loss / Critic Loss (avg over {self._reward_window} eps)")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Actor Loss")
        ax_loss_r.set_ylabel("Critic Loss")
        ax_loss.plot(steps, [r.get("avg_actor_loss", 0.0) for r in w0_rows],
                     color="tab:blue", label="actor loss")
        ax_loss_r.plot(steps, [r.get("avg_critic_loss", 0.0) for r in w0_rows],
                       color="tab:orange", linewidth=0.9, linestyle="--", label="critic loss")
        handles_l, labels_l = ax_loss.get_legend_handles_labels()
        handles_r, labels_r = ax_loss_r.get_legend_handles_labels()
        ax_loss.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8)

        # --- 3. PPO-specific: Entropy + Clip Fraction + Approx KL ---
        # Entropy (left axis): falling entropy means the policy is converging to
        # confident action choices.  It should drop gradually, not instantly.
        #
        # Clip fraction (right axis, solid): fraction of timesteps where
        # |r_t − 1| > ε.  Healthy: 0.05 – 0.20.  Red dashed reference at 0.2.
        #
        # Approx KL (right axis, dashed): total policy shift across K epochs.
        # Healthy: 0.01 – 0.05.  Orange dashed reference at 0.05.
        ax_ppo.cla()
        ax_ppo_r.cla()
        ax_ppo.set_title("Entropy / Clip Fraction / Approx KL")
        ax_ppo.set_xlabel("Step")
        ax_ppo.set_ylabel("Entropy")
        ax_ppo_r.set_ylabel("Clip Frac / Approx KL")

        ax_ppo.plot(steps, [r.get("avg_entropy", 0.0) for r in w0_rows],
                    color="tab:green", label="entropy")
        ax_ppo_r.plot(steps, [r.get("clip_fraction", 0.0) for r in w0_rows],
                      color="tab:red", linewidth=0.9, label="clip frac")
        ax_ppo_r.plot(steps, [r.get("avg_approx_kl", 0.0) for r in w0_rows],
                      color="tab:orange", linewidth=0.9, linestyle="--", label="approx KL")

        # Reference lines for healthy ranges
        ax_ppo_r.axhline(0.2,  color="tab:red",    linewidth=0.5, linestyle=":")
        ax_ppo_r.axhline(0.05, color="tab:orange", linewidth=0.5, linestyle=":")

        handles_l, labels_l = ax_ppo.get_legend_handles_labels()
        handles_r, labels_r = ax_ppo_r.get_legend_handles_labels()
        ax_ppo.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8)

        # --- 4. Per-layer actor |W| (solid) + critic |W| (dashed) + grad norms ---
        actor_layer_keys  = sorted(k for k in w0_rows[0] if k.startswith("avg_w_actor_"))
        critic_layer_keys = sorted(k for k in w0_rows[0] if k.startswith("avg_w_critic_"))
        actor_gnorm  = [r.get("actor_grad_norm",  0.0) for r in w0_rows]
        critic_gnorm = [r.get("critic_grad_norm", 0.0) for r in w0_rows]

        ax_weights.cla()
        ax_weights_r.cla()
        ax_weights.set_title("Per-Layer Actor |w| (solid) / Critic |w| (dashed) / Grad Norms")
        ax_weights.set_xlabel("Step")
        ax_weights.set_ylabel("|w|")
        colors = plt.cm.tab10.colors
        num_actor_layers = len(actor_layer_keys)
        for i, key in enumerate(actor_layer_keys):
            label = f"actor W{i} (out)" if i == num_actor_layers - 1 else f"actor W{i}"
            ax_weights.plot(
                steps, [r.get(key, 0.0) for r in w0_rows],
                color=colors[i % len(colors)], label=label,
            )
        num_critic_layers = len(critic_layer_keys)
        for i, key in enumerate(critic_layer_keys):
            label = f"critic W{i} (out)" if i == num_critic_layers - 1 else f"critic W{i}"
            ax_weights.plot(
                steps, [r.get(key, 0.0) for r in w0_rows],
                color=colors[i % len(colors)], linewidth=0.9, linestyle="--", label=label,
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
