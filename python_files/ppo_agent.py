"""
Proximal Policy Optimization (PPO-Clip) Agent — JAX DNN version

=== Why PPO? The problem with A2C ===

A2C uses a fixed learning rate α to update the policy each episode:

    θ ← θ + α · ∇θ Σ_t Â_t · log π(a_t|s_t;θ)

The gradient can be arbitrarily large — a high-variance advantage Â_t or
a very small log π (action nearly forbidden) can produce a massive step.
One bad episode → policy collapse → training never recovers.

A2C mitigates this with advantage normalisation, but the step size is
still unconstrained relative to how much the policy actually changes.


=== The Probability Ratio ===

The key PPO quantity is the probability ratio:

    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

Interpretation:
    r_t = 1   →  new policy == old policy for this (s,a) pair
    r_t > 1   →  new policy increased the probability of this action
    r_t < 1   →  new policy decreased the probability of this action

The unclipped importance-sampled policy gradient objective is:
    L^PG(θ) = E_t[ r_t(θ) · Â_t ]

This is equivalent to the log-prob gradient but allows re-using old data.


=== PPO-Clip Objective ===

PPO replaces L^PG with a CLIPPED version that stops gradients once the
policy has moved "far enough" from the old policy:

    L^CLIP(θ) = E_t[ min(r_t · Â_t,  clip(r_t, 1-ε, 1+ε) · Â_t) ]

where ε ≈ 0.2 is the clipping range.

The min makes this a PESSIMISTIC bound:

    Â_t > 0 (good action, want more):
        · If r_t ≤ 1+ε:  gradient normally pushes r_t higher
        · If r_t > 1+ε:  clip takes over, gradient = 0  ← stop increasing
          "We've already made this action probable enough; no more pushing."

    Â_t < 0 (bad action, want less):
        · If r_t ≥ 1-ε:  gradient normally pushes r_t lower
        · If r_t < 1-ε:  clip takes over, gradient = 0  ← stop decreasing
          "We've already suppressed this action enough; no more pushing."

This creates a SOFT TRUST REGION without any KL constraint or second-order
optimisation (which TRPO needed).


=== Multiple Epochs (K-Epoch Trick) ===

A2C updated each episode's data exactly once.  PPO does K gradient steps
on the SAME episode data, which gives K× more sample efficiency:

    Collect episode with π_old  (store old_log_probs and values)
    For epoch k = 1 … K:
        Compute r_t = exp(log π_new − log π_old)   (drifts away from 1 over epochs)
        Update with L^CLIP  (clipping naturally zeros gradient when r_t drifts too far)

After K epochs the policy has moved as much as the clipping allows, and
gradients are essentially zero.  Further epochs would be wasteful.


=== GAE: Generalized Advantage Estimation ===

A2C computed A_t = G_t − V(s_t)  (Monte Carlo return minus baseline).
PPO pairs with GAE-λ, which interpolates between:
    · 1-step TD  (λ=0): low variance, biased by imperfect V
    · Monte Carlo (λ=1): unbiased, high variance

Recurrence (single backward pass):
    δ_t  = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)    ← TD residual
    Â_t  = δ_t + γ · λ · (1 − done_t) · Â_{t+1}

For episodic rollouts (done only True at the last step):
    δ_{T-1} = r_{T-1} + γ · 0 − V(s_{T-1})
    δ_t     = r_t     + γ · V(s_{t+1}) − V(s_t)    for t < T-1

The VALUE TARGET the critic is trained to predict:
    V_target_t = Â_t + V(s_t)   (GAE return — less noisy than raw G_t)

λ = 0.95 is nearly universal in practice.


=== Diagnostics: Clip Fraction and Approx KL ===

clip_fraction  = (1/T) Σ_t 1[ |r_t − 1| > ε ]
    · Shows what fraction of timesteps had their gradient clipped.
    · Healthy range: 0.05 – 0.20.  > 0.30 → policy moving too fast.

approx_kl  = (1/T) Σ_t (log π_old − log π_new)  = mean(old − new)
    · Approximate KL divergence between old and new policy.
    · Healthy range: 0.01 – 0.05.  > 0.10 → policy updating too aggressively.

Both are computed AFTER all K epochs, reflecting the total policy change
relative to the rollout collection point.


=== Architecture ===

Same actor/critic pair as A2C:
    Actor  π(a|s;θ):  state_dim → hidden → … → num_actions → softmax
    Critic V(s;w):    state_dim → hidden → … → 1  (no activation on output)

Networks are SEPARATE, updated with independent learning rates.


=== Update per Episode ===

    1. Collect trajectory: (s_t, a_t, r_t, done_t, V(s_t), log π_old(a_t|s_t))
    2. Compute GAE advantages: Â_t  (backward pass, O(T))
    3. Compute value targets:  V_target_t = Â_t + V_old(s_t)
    4. Normalise Â_t  →  Â̂_t  (zero mean, unit std)
    5. For k in 1 … n_epochs:
           a. r_t = exp(log π_θ(a_t|s_t) − log π_old(a_t|s_t))
           b. L_actor = −mean(min(r_t·Â̂_t,  clip(r_t,1-ε,1+ε)·Â̂_t))
                        − c_H · H(π)
           c. L_critic = mean((V_target_t − V(s_t;w))²)
           d. θ ← θ − α_actor  · ∇θ L_actor
           e. w ← w − α_critic · ∇w L_critic
    6. Compute diagnostics: clip_fraction, approx_kl
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rl_agent import RLAgent


# ---------------------------------------------------------------------------
# Module-level JAX configuration
# ---------------------------------------------------------------------------

def _log_device_info() -> None:
    """Print which compute device JAX is using."""
    backend  = jax.default_backend()
    devices  = jax.devices()
    dev_list = ", ".join(str(d) for d in devices)
    if backend == "cpu":
        print(f"[JAX/PPO] Backend: CPU ({dev_list})")
    else:
        print(f"[JAX/PPO] Backend: {backend.upper()} — devices: {dev_list}")


_log_device_info()


class PPOAgent(RLAgent):
    """Proximal Policy Optimization (PPO-Clip) agent backed by two JAX MLPs.

    Actor  π(a|s;θ):  softmax MLP — outputs action probabilities.
    Critic V(s;w):    scalar MLP  — outputs value estimate V^π(s).

    The two networks are SEPARATE (no weight-sharing) so actor and critic
    learning rates can be tuned independently.

    Learning algorithm: Episodic PPO-Clip with GAE-λ and K-epoch updates.

    Per episode:
        Collect (s_t, a_t, r_t, done_t, V_old(s_t), log π_old(a_t|s_t))
        Compute GAE advantages  Â_t
        Compute value targets   V_target_t = Â_t + V_old(s_t)
        Normalise Â_t
        Repeat n_epochs times:
            θ ← θ − α_actor  · ∇θ L_actor (PPO-clip + entropy)
            w ← w − α_critic · ∇w L_critic (MSE on V_target)
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        state_vars:          list[str],
        actions:             list[str],
        alpha_actor:         float,
        alpha_critic:        float,
        gamma:               float,
        gae_lambda:          float,
        clip_epsilon:        float,
        n_epochs:            int,
        actor_hidden_sizes:  list[int],
        critic_hidden_sizes: list[int],
        entropy_coef:        float,
        critic_coef:         float,
    ):
        """
        Initialise the PPO agent.

        Args:
            state_vars:   Ordered list of state variable names matching the
                          keys sent by Godot every frame.
                          Example: ["paddle_y", "ball_x", "ball_y", "ball_vx", "ball_vy"]

            actions:      List of action strings.
                          Example: ["UP", "DOWN", "STAY"]

            alpha_actor:  Learning rate for the ACTOR (gradient ASCENT on J).
                          Typical range: 1e-4 – 5e-4.
                          Smaller than A2C is often fine since K epochs already
                          extract more signal per episode.

            alpha_critic: Learning rate for the CRITIC (gradient DESCENT on MSE).
                          Typical range: 5e-4 – 3e-3.

            gamma:        Discount factor γ ∈ [0, 1].

            gae_lambda:   GAE smoothing parameter λ ∈ [0, 1].
                          λ = 0 → pure 1-step TD (low variance, high bias).
                          λ = 1 → Monte Carlo (high variance, unbiased).
                          Typical: 0.95.

            clip_epsilon: PPO clipping range ε.
                          The ratio r_t is clipped to [1−ε, 1+ε].
                          Typical: 0.2.  Smaller → more conservative updates.

            n_epochs:     Number of gradient update passes on each episode's data.
                          Typical: 4–10.  More epochs → more sample efficiency
                          but also more risk of overfitting to one episode's noise.

            actor_hidden_sizes:  Hidden layer widths for the ACTOR MLP.
                          Example: [64, 64]

            critic_hidden_sizes: Hidden layer widths for the CRITIC MLP.
                          Example: [128, 64]
                          Larger than actor recommended: V(s) regression is harder
                          (target shifts as policy improves).

            entropy_coef: Weight c_H of the entropy bonus in the actor loss.
                          Typical: 0.001 – 0.02.

            critic_coef:  Weight c_v of the critic loss (logging only — actor and
                          critic are updated via separate gradient passes, so this
                          does NOT affect actual updates).
        """

        # ── configuration ──────────────────────────────────────────────
        self.state_vars          = state_vars
        self.actions             = actions
        self.alpha_actor         = alpha_actor
        self.alpha_critic        = alpha_critic
        self.gamma               = gamma
        self.gae_lambda          = gae_lambda
        self.clip_epsilon        = clip_epsilon
        self.n_epochs            = n_epochs
        self.actor_hidden_sizes  = actor_hidden_sizes
        self.critic_hidden_sizes = critic_hidden_sizes
        self.entropy_coef        = entropy_coef
        self.critic_coef         = critic_coef

        # ── validate everything before deriving any dimensions ─────────
        self._validate_state()
        self._validate_actions()
        self._validate_hyperparameters()

        # ── derived dimensions ─────────────────────────────────────────
        self.state_dim   = len(self.state_vars)
        self.num_actions = len(self.actions)

        # ── network parameters ─────────────────────────────────────────
        seed_actor  = random.randint(0, 2**32 - 1)
        seed_critic = random.randint(0, 2**32 - 1)
        self.actor_params:  dict = self._init_params(seed_actor,  self.actor_hidden_sizes,  output_size=self.num_actions)
        self.critic_params: dict = self._init_params(seed_critic, self.critic_hidden_sizes, output_size=1)

        # ── JIT-compiled forward passes ────────────────────────────────
        self._actor_forward_jit  = jax.jit(PPOAgent._forward_actor)
        self._critic_forward_jit = jax.jit(PPOAgent._forward_critic)

        # ── JIT-compiled loss + gradient functions ─────────────────────
        #
        # actor_loss differentiates wrt actor_params (arg 0).
        # critic_loss differentiates wrt critic_params (arg 0).
        self._actor_loss_and_grad_jit = jax.jit(
            jax.value_and_grad(PPOAgent._actor_loss)
        )
        self._critic_loss_and_grad_jit = jax.jit(
            jax.value_and_grad(PPOAgent._critic_loss)
        )

        # ── JIT-compiled critic batch forward pass (for value collection) ─
        #
        # At episode end we need V(s_t; w) for EVERY timestep t to compute
        # GAE. We also collect V(s_t) during rollout (in process_state) so
        # we can use the SAME critic snapshot for both GAE and value targets.
        # _critic_batch_jit runs all T critic evaluations in one XLA kernel.
        self._critic_batch_jit = jax.jit(
            jax.vmap(PPOAgent._forward_critic, in_axes=(None, 0))
        )

        # ── per-thread trajectory buffer ───────────────────────────────
        #
        # Trajectory entry format:
        #   (state_jnp_array, action_idx_int, reward_float, log_prob_old_float, v_old_float)
        #
        # Compared to A2C, PPO adds two extra fields:
        #   log_prob_old: log π_old(a_t|s_t) — needed to compute r_t = π_new/π_old
        #   v_old:        V(s_t; w) at collection time — needed for GAE δ_t
        self._local = threading.local()

        # ── shared parameter update lock ───────────────────────────────
        self._lock = threading.Lock()

        # ── statistics ─────────────────────────────────────────────────
        self._episodes_completed    = 0
        self._updates_count         = 0
        self._last_actor_loss       = 0.0   # actor loss averaged over last K epochs
        self._last_critic_loss      = 0.0   # critic loss averaged over last K epochs
        self._last_episode_return   = 0.0
        self._last_episode_length   = 0
        self._last_actor_grad_norm  = 0.0   # grad norm from the last of K actor steps
        self._last_critic_grad_norm = 0.0   # grad norm from the last of K critic steps
        self._last_entropy          = 0.0
        self._last_mean_advantage   = 0.0   # mean |Â_t| (unnormalised) at last update
        self._last_clip_fraction    = 0.0   # fraction of steps clipped (after K epochs)
        self._last_approx_kl        = 0.0   # approx KL divergence (after K epochs)

    # ------------------------------------------------------------------
    # Parameter initialisation  (identical to A2CAgent._init_params)
    # ------------------------------------------------------------------

    def _init_params(self, seed: int, hidden_sizes: list[int], output_size: int) -> dict:
        """
        Build and initialise an MLP parameter dict.

        Layer architecture:
            [state_dim] + hidden_sizes + [output_size]

        Initialisation:
            Hidden layers: He (Kaiming), std = √(2/n_in)   — corrects for ReLU
            Output layer:  Xavier,        std = √(1/n_in)   — no ReLU follows

        Returns:
            dict with keys "W0","b0","W1","b1",…  (one pair per layer).
        """
        key = jax.random.PRNGKey(seed)
        layer_sizes = [self.state_dim] + hidden_sizes + [output_size]
        params = {}
        num_layers = len(layer_sizes) - 1

        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            key, subkey = jax.random.split(key)
            is_output = (i == num_layers - 1)
            std = jnp.sqrt(1.0 / n_in) if is_output else jnp.sqrt(2.0 / n_in)
            params[f"W{i}"] = jax.random.normal(subkey, (n_out, n_in)) * std
            params[f"b{i}"] = jnp.zeros(n_out)

        return params

    # ------------------------------------------------------------------
    # Validation helpers  (identical to A2CAgent)
    # ------------------------------------------------------------------

    def _validate_state(self) -> None:
        if not isinstance(self.state_vars, list):
            raise ValueError("state_vars must be a list.")
        if not self.state_vars:
            raise ValueError("state_vars must not be empty.")
        if not all(isinstance(v, str) for v in self.state_vars):
            raise ValueError("All entries in state_vars must be strings.")

    def _validate_actions(self) -> None:
        if not isinstance(self.actions, list):
            raise ValueError("actions must be a list.")
        if not self.actions:
            raise ValueError("actions must not be empty.")
        if not all(isinstance(a, str) for a in self.actions):
            raise ValueError("All entries in actions must be strings.")

    def _validate_hyperparameters(self) -> None:
        """Validate all PPO hyperparameters."""
        if not isinstance(self.alpha_actor, (int, float)) or not (self.alpha_actor > 0):
            raise ValueError(f"alpha_actor={self.alpha_actor!r} must be a strictly positive number.")
        if not isinstance(self.alpha_critic, (int, float)) or not (self.alpha_critic > 0):
            raise ValueError(f"alpha_critic={self.alpha_critic!r} must be a strictly positive number.")
        if not isinstance(self.gamma, (int, float)) or not (0 <= self.gamma <= 1):
            raise ValueError(f"gamma={self.gamma!r} must be a number in [0, 1].")
        if not isinstance(self.gae_lambda, (int, float)) or not (0 <= self.gae_lambda <= 1):
            raise ValueError(f"gae_lambda={self.gae_lambda!r} must be a number in [0, 1].")
        if not isinstance(self.clip_epsilon, (int, float)) or not (0 < self.clip_epsilon < 1):
            raise ValueError(f"clip_epsilon={self.clip_epsilon!r} must be a number in (0, 1).")
        if not isinstance(self.n_epochs, int) or self.n_epochs < 1:
            raise ValueError(f"n_epochs={self.n_epochs!r} must be a positive integer.")

        def _check_hidden(sizes, name: str) -> list[int]:
            if not isinstance(sizes, list):
                raise ValueError(f"{name} must be a list, got {type(sizes).__name__}.")
            if not sizes:
                raise ValueError(f"{name} must not be empty.")
            converted = []
            for i, size in enumerate(sizes):
                try:
                    size_int = int(size)
                    if size_int != size:
                        raise ValueError()
                except (ValueError, TypeError):
                    raise ValueError(f"{name}[{i}]={size!r} cannot be converted to an integer.")
                if size_int <= 0:
                    raise ValueError(f"{name}[{i}]={size!r} is not strictly positive.")
                converted.append(size_int)
            return converted

        self.actor_hidden_sizes  = _check_hidden(self.actor_hidden_sizes,  "actor_hidden_sizes")
        self.critic_hidden_sizes = _check_hidden(self.critic_hidden_sizes, "critic_hidden_sizes")

        if not isinstance(self.entropy_coef, (int, float)) or self.entropy_coef < 0:
            raise ValueError(f"entropy_coef={self.entropy_coef!r} must be a non-negative number.")
        if not isinstance(self.critic_coef, (int, float)) or self.critic_coef < 0:
            raise ValueError(f"critic_coef={self.critic_coef!r} must be a non-negative number.")

    # ------------------------------------------------------------------
    # Class-method constructor (from YAML config dict)
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PPOAgent":
        """
        Create a PPOAgent from a YAML/dict config.

        Expected YAML structure:

            state:   [paddle_y, ball_x, ball_y, ball_vx, ball_vy]
            actions: [UP, DOWN, STAY]
            hyperparameters:
                alpha_actor:         0.0003
                alpha_critic:        0.001
                gamma:               0.99
                gae_lambda:          0.95
                clip_epsilon:        0.2
                n_epochs:            4
                actor_hidden_sizes:  [64, 64]
                critic_hidden_sizes: [128, 64]
                entropy_coef:        0.01
                critic_coef:         0.5
        """
        state_vars: list[str] = config_dict.get("state")
        actions:    list[str] = config_dict.get("actions")

        hp: dict = config_dict.get("hyperparameters")
        if hp is None:
            raise ValueError(
                "Config is missing the 'hyperparameters' block. "
                "Expected keys: alpha_actor, alpha_critic, gamma, gae_lambda, "
                "clip_epsilon, n_epochs, actor_hidden_sizes, critic_hidden_sizes, "
                "entropy_coef, critic_coef."
            )

        return cls(
            state_vars=state_vars,
            actions=actions,
            alpha_actor=hp.get("alpha_actor"),
            alpha_critic=hp.get("alpha_critic"),
            gamma=hp.get("gamma"),
            gae_lambda=hp.get("gae_lambda"),
            clip_epsilon=hp.get("clip_epsilon"),
            n_epochs=hp.get("n_epochs"),
            actor_hidden_sizes=hp.get("actor_hidden_sizes"),
            critic_hidden_sizes=hp.get("critic_hidden_sizes"),
            entropy_coef=hp.get("entropy_coef"),
            critic_coef=hp.get("critic_coef"),
        )

    # ------------------------------------------------------------------
    # Forward passes  (pure static — identical to A2CAgent)
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_actor(params: dict, s: jnp.ndarray) -> jnp.ndarray:
        """Actor forward pass: state → action probabilities.

        x = s
        x = ReLU(W_i x + b_i)   for i in 0 … L-2   (hidden layers)
        z = W_{L-1} x + b_{L-1}                      (output logits)
        π = softmax(z)                                (valid probability distribution)

        Returns:
            π — shape (num_actions,), all values in (0,1), sum = 1.
        """
        num_layers = len(params) // 2
        x = s
        for i in range(num_layers - 1):
            x = jax.nn.relu(params[f"W{i}"] @ x + params[f"b{i}"])
        logits = params[f"W{num_layers - 1}"] @ x + params[f"b{num_layers - 1}"]
        return jax.nn.softmax(logits)

    @staticmethod
    def _forward_critic(params: dict, s: jnp.ndarray) -> jnp.ndarray:
        """Critic forward pass: state → scalar value V(s;w).

        Same MLP architecture as actor, but output is a plain scalar (no
        activation) so V can represent any real number.

        Returns:
            V — shape () (0-d JAX array).
        """
        num_layers = len(params) // 2
        x = s
        for i in range(num_layers - 1):
            x = jax.nn.relu(params[f"W{i}"] @ x + params[f"b{i}"])
        v = params[f"W{num_layers - 1}"] @ x + params[f"b{num_layers - 1}"]
        return v[0]

    # ------------------------------------------------------------------
    # State validation and conversion  (identical to A2CAgent)
    # ------------------------------------------------------------------

    def _validate_state_dict(self, state_dict: dict) -> None:
        incoming_keys = set(state_dict.keys())
        expected_keys = set(self.state_vars)
        if incoming_keys != expected_keys:
            missing = expected_keys - incoming_keys
            extra   = incoming_keys - expected_keys
            msg = "State dictionary keys do not match expected state variables."
            if missing:
                msg += f" Missing: {missing}."
            if extra:
                msg += f" Unexpected: {extra}."
            raise ValueError(msg)
        for key in self.state_vars:
            val = state_dict[key]
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"State value for '{key}' must be numeric, "
                    f"got {type(val).__name__}."
                )
            if val < -1.0 or val > 1.0:
                print(f"Warning: state['{key}'] = {val:.4f} is outside [-1,1]. Clamping.")

    def _state_to_vector(self, state_dict: dict) -> jnp.ndarray:
        return jnp.array(
            [float(state_dict[v]) for v in self.state_vars],
            dtype=jnp.float32,
        ).clip(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Per-thread pending transition and trajectory buffer
    # ------------------------------------------------------------------

    @property
    def _pending(self) -> tuple | None:
        """Per-thread (state_vec, action_idx, log_prob_old, v_old) waiting for reward.

        PPO stores two extra fields compared to A2C:
            log_prob_old: log π(a_t|s_t;θ_old) — used to compute r_t next episode
            v_old:        V(s_t;w_old)          — used for GAE δ_t computation
        """
        return getattr(self._local, "pending", None)

    @_pending.setter
    def _pending(self, value: tuple | None) -> None:
        self._local.pending = value

    @property
    def _trajectory(self) -> list:
        """Per-thread list of (state_vec, action_idx, reward, log_prob_old, v_old)."""
        if not hasattr(self._local, "trajectory"):
            self._local.trajectory = []
        return self._local.trajectory

    @_trajectory.setter
    def _trajectory(self, value: list) -> None:
        self._local.trajectory = value

    # ------------------------------------------------------------------
    # process_state — observe state → actor forward pass → return action
    # ------------------------------------------------------------------

    def process_state(self, state_dict: dict[str, float]) -> str:
        """Observe the current game state and return the action to take.

        PPO adds two computations at inference time (compared to A2C):
            1. log_prob_old = log π(a_t|s_t;θ) — stored for later ratio r_t
            2. v_old = V(s_t;w)                 — stored for GAE computation

        Both are computed NOW using the current (pre-update) networks, which
        are exactly the "old" policy/value function that the K-epoch update
        will compare against.

        Args:
            state_dict: Normalised game state from Godot.

        Returns:
            Action string, e.g. "UP", "DOWN", or "STAY".
        """
        self._validate_state_dict(state_dict)
        s = self._state_to_vector(state_dict)

        # Actor forward pass → probability distribution
        pi = self._actor_forward_jit(self.actor_params, s)   # (num_actions,)

        # Sample action stochastically
        action_idx = int(np.random.choice(self.num_actions, p=np.array(pi)))

        # log π(a_t|s_t;θ_old) — stored as a Python float
        log_prob_old = float(jnp.log(pi[action_idx] + 1e-8))

        # V(s_t; w_old) — scalar, stored for GAE δ_t = r_t + γV(s_{t+1}) − V(s_t)
        v_old = float(self._critic_forward_jit(self.critic_params, s))

        self._pending = (s, action_idx, log_prob_old, v_old)
        return self.actions[action_idx]

    # ------------------------------------------------------------------
    # _compute_gae — Generalized Advantage Estimation
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards:    list[float],
        values_old: list[float],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute GAE-λ advantages and value targets for one episode.

        Since this is an EPISODIC rollout (done is only True at the last
        step), V(s_T) = 0 (no future return after terminal state).

        Algorithm (single backward pass, O(T)):
            δ_t     = r_t + γ · V(s_{t+1}) − V(s_t)    TD residual
            Â_{T-1} = δ_{T-1}                            terminal
            Â_t     = δ_t + γ · λ · Â_{t+1}             recurrence

        Value target (what the critic is trained to predict):
            V_target_t = Â_t + V_old(s_t)
            This is the GAE return — strictly between 1-step TD and Monte
            Carlo return, inheriting the variance/bias tradeoff of λ.

        Args:
            rewards:    Per-step rewards [r_0, …, r_{T-1}].
            values_old: Per-step V(s_t;w_old) collected during the episode.

        Returns:
            advantages   — shape (T,) Â_t  (unnormalised)
            value_targets— shape (T,) V_target_t = Â_t + V_old(s_t)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            # V(s_{t+1}): zero at the terminal step (t = T-1)
            next_value = values_old[t + 1] if t < T - 1 else 0.0
            # 1-step TD residual
            delta = rewards[t] + self.gamma * next_value - values_old[t]
            # Backward accumulation: Â_t = δ_t + (γλ)·Â_{t+1}
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        advantages_jnp     = jnp.array(advantages)
        value_targets_jnp  = advantages_jnp + jnp.array(values_old, dtype=jnp.float32)
        return advantages_jnp, value_targets_jnp

    # ------------------------------------------------------------------
    # Loss functions  (pure static — differentiable by jax.grad)
    # ------------------------------------------------------------------

    @staticmethod
    def _actor_loss(
        actor_params:  dict,
        states:        jnp.ndarray,    # (T, state_dim)
        action_ids:    jnp.ndarray,    # (T,)  int indices
        advantages:    jnp.ndarray,    # (T,)  normalised Â_t
        old_log_probs: jnp.ndarray,    # (T,)  log π_old(a_t|s_t)  ← constant
        clip_epsilon:  float,
        entropy_coef:  float,
    ) -> jnp.ndarray:
        """PPO-Clip actor loss for one full episode.

        Algorithm:
            1. New log probs: log π_θ(a_t|s_t) for the SAME actions as collected
            2. Ratio: r_t = exp(new_log_prob − old_log_prob)
               (equivalent to π_new/π_old but numerically more stable)
            3. Clipped surrogate:
               L_CLIP = mean( min(r_t·Â_t,  clip(r_t,1-ε,1+ε)·Â_t) )
            4. Entropy bonus:
               H = −mean( Σ_a π(a|s) log π(a|s) )

        Why exp(new − old) for the ratio?
            r_t = π_new(a) / π_old(a)
                = exp(log π_new(a)) / exp(log π_old(a))
                = exp(log π_new(a) − log π_old(a))
            This avoids potential division-by-small-number issues.

        Why old_log_probs as a constant input, not as a parameter?
            jax.grad(f)(actor_params, ...) differentiates ONLY wrt actor_params.
            old_log_probs is a static array (collected before the update loop)
            — its gradient is zero by construction. Passing it as an arg
            rather than a captured closure keeps the function pure (required by jit).

        Args:
            actor_params:  Actor weight dict {W0,b0,…}.
            states:        Stacked state vectors, shape (T, state_dim).
            action_ids:    Indices of chosen actions, shape (T,).
            advantages:    Normalised advantages Â_t, shape (T,).
            old_log_probs: log π_old(a_t|s_t) collected during the episode, shape (T,).
                           Treated as a CONSTANT — not differentiated.
            clip_epsilon:  ε for clipping r_t to [1-ε, 1+ε].
            entropy_coef:  c_H — entropy bonus weight.

        Returns:
            Scalar loss (negate of L_CLIP minus entropy bonus).
        """
        # --- Forward pass on all T states simultaneously ---------------
        all_pi = jax.vmap(
            lambda s: PPOAgent._forward_actor(actor_params, s)
        )(states)                                                # (T, num_actions)

        # --- New log probs for the SAME actions that were taken ---------
        t_idx         = jnp.arange(states.shape[0])
        new_log_probs = jnp.log(all_pi[t_idx, action_ids] + 1e-8)   # (T,)

        # --- Probability ratio r_t = π_new / π_old ----------------------
        #
        # old_log_probs is treated as a constant (not differentiated).
        # The gradient flows ONLY through new_log_probs → actor_params.
        ratios = jnp.exp(new_log_probs - old_log_probs)              # (T,)

        # --- Clipped surrogate objective --------------------------------
        #
        # surr1: unclipped term  r_t · Â_t
        # surr2: clipped term    clip(r_t, 1-ε, 1+ε) · Â_t
        # min: pessimistic bound — only act on gradient when r_t hasn't
        #      moved too far from 1 in the direction of the advantage.
        surr1 = ratios * advantages                                   # (T,)
        surr2 = jnp.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages  # (T,)
        pg_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # --- Entropy bonus: - c_H · H(π) in the loss = + c_H · H in J --
        entropy = -jnp.mean(
            jnp.sum(all_pi * jnp.log(all_pi + 1e-8), axis=-1)
        )

        return pg_loss - entropy_coef * entropy

    @staticmethod
    def _critic_loss(
        critic_params: dict,
        states:        jnp.ndarray,   # (T, state_dim)
        value_targets: jnp.ndarray,   # (T,)  V_target_t = Â_t + V_old(s_t)
    ) -> jnp.ndarray:
        """Critic loss: MSE between predicted values and GAE value targets.

        L_critic(w) = (1/T) Σ_t (V_target_t − V(s_t;w))²

        The value target V_target_t = Â_t + V_old(s_t) is the GAE return:
        it leverages both the observed rewards (via Â_t) and the critic's
        own previous estimate (V_old), giving a lower-variance target than
        raw Monte Carlo G_t while retaining the GAE bias-variance trade-off.

        Unlike some PPO implementations, we do NOT clip the value function
        loss. Value clipping (clip(V, V_old−ε, V_old+ε)) adds complexity
        without consistently improving results.

        Args:
            critic_params:  Critic weight dict {W0,b0,…}.
            states:         Stacked state vectors, shape (T, state_dim).
            value_targets:  GAE returns V_target_t, shape (T,).
                            Treated as constants — not differentiated.

        Returns:
            Scalar MSE loss.
        """
        values = jax.vmap(
            lambda s: PPOAgent._forward_critic(critic_params, s)
        )(states)                                               # (T,)
        return jnp.mean((value_targets - values) ** 2)

    # ------------------------------------------------------------------
    # update — accumulate trajectory, run K-epoch PPO update at end
    # ------------------------------------------------------------------

    def update(self, state: dict, reward: float, done: bool) -> None:
        """Receive feedback for the last action and (on episode end) learn.

        Called every frame BEFORE process_state:
            frame N:  update(state_N, prev_reward, done)   ← reward for action N-1
            frame N:  process_state(state_N)               ← choose action N

        On a non-terminal frame: append (s, a_idx, r, log_p_old, v_old) to
        the per-thread trajectory buffer and return.

        On a terminal frame (done=True), run the full PPO update:

            1. Compute GAE advantages  Â_t  (backward pass)
            2. Compute value targets   V_target_t = Â_t + V_old(s_t)
            3. Normalise Â_t  →  Â̂_t
            4. For k in 1 … n_epochs:
               a. Actor:  θ ← θ − α_actor  · ∇θ L_actor(PPO-clip + entropy)
               b. Critic: w ← w − α_critic · ∇w L_critic(MSE on V_target)
            5. Compute diagnostics: clip_fraction, approx_kl

        Why collect V_old and log_prob_old at experience time (process_state)?
            At episode end, we want to keep the OLD policy values fixed for all
            K epochs — using the same π_old throughout the K-epoch loop is what
            gives PPO its trust-region property. If we re-computed V(s_t) with
            the updated critic at each epoch, the value targets would shift
            mid-update, destabilising training.

        Args:
            state:  Current game state (keys only used for first-frame guard).
            reward: Reward received for the action chosen last frame.
            done:   True if the episode just ended.
        """
        # ── guard: no pending transition on the very first frame ─────────
        if self._pending is None:
            return

        # ── complete the pending tuple with its reward ────────────────────
        s, action_id, log_prob_old, v_old = self._pending
        self._trajectory.append((s, action_id, reward, log_prob_old, v_old))
        self._pending = None

        if not done:
            return

        # ── snapshot + clear trajectory immediately ───────────────────────
        traj = self._trajectory
        self._trajectory = []

        states        = [e[0] for e in traj]
        action_ids    = [e[1] for e in traj]
        rewards       = [e[2] for e in traj]
        log_probs_old = [e[3] for e in traj]
        values_old    = [e[4] for e in traj]

        # ── stack into JAX arrays ─────────────────────────────────────────
        states_arr        = jnp.stack(states)                                  # (T, state_dim)
        action_ids_arr    = jnp.array(action_ids,    dtype=jnp.int32)          # (T,)
        old_log_probs_arr = jnp.array(log_probs_old, dtype=jnp.float32)        # (T,)

        # ── step 1-2: GAE advantages + value targets ──────────────────────
        #
        # advantages: Â_t  (unnormalised) — used for actor loss and diagnostics
        # value_targets: V_target_t = Â_t + V_old — used for critic loss
        advantages_arr, value_targets_arr = self._compute_gae(rewards, values_old)

        # ── step 3: normalise advantages ──────────────────────────────────
        #
        # Â̂_t = (Â_t − mean(Â)) / (std(Â) + ε)
        # Keeps actor gradient magnitude stable across episodes with different
        # reward scales. The critic is NOT affected — it uses V_target directly.
        adv_mean = advantages_arr.mean()
        adv_std  = advantages_arr.std()
        norm_advantages_arr = (advantages_arr - adv_mean) / (adv_std + 1e-8)

        # ── steps 4a-4b: K-epoch update loop ─────────────────────────────
        #
        # old_log_probs_arr is FIXED across all K epochs — it was collected
        # using the policy BEFORE this update call. The clipping ensures the
        # policy can only move by at most ε from the old policy per episode,
        # even after K gradient steps.
        actor_losses  = []
        critic_losses = []
        last_actor_grad_norm  = 0.0
        last_critic_grad_norm = 0.0

        # Take a local snapshot of params before the update loop.
        # Both networks start from these same weights for all K epochs
        # (we do NOT reset between epochs; we accumulate the K updates).
        actor_params_epoch  = self.actor_params
        critic_params_epoch = self.critic_params

        for _ in range(self.n_epochs):
            actor_loss, actor_grads = self._actor_loss_and_grad_jit(
                actor_params_epoch,
                states_arr,
                action_ids_arr,
                norm_advantages_arr,
                old_log_probs_arr,
                self.clip_epsilon,
                self.entropy_coef,
            )
            actor_params_epoch = jax.tree.map(
                lambda p, g: p - self.alpha_actor * g,
                actor_params_epoch, actor_grads,
            )
            actor_losses.append(float(actor_loss))
            last_actor_grad_norm = float(
                jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(actor_grads)))
            )

            critic_loss, critic_grads = self._critic_loss_and_grad_jit(
                critic_params_epoch, states_arr, value_targets_arr
            )
            critic_params_epoch = jax.tree.map(
                lambda p, g: p - self.alpha_critic * g,
                critic_params_epoch, critic_grads,
            )
            critic_losses.append(float(critic_loss))
            last_critic_grad_norm = float(
                jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(critic_grads)))
            )

        # ── step 5: diagnostics (computed AFTER all K epochs) ────────────
        #
        # We compare the FINAL updated actor against the original π_old to
        # measure total policy shift across the K epochs.
        all_pi_final = jax.vmap(
            lambda s: PPOAgent._forward_actor(actor_params_epoch, s)
        )(states_arr)                                                   # (T, num_actions)
        t_idx = jnp.arange(states_arr.shape[0])
        new_log_probs_final = jnp.log(all_pi_final[t_idx, action_ids_arr] + 1e-8)

        # r_t = π_new / π_old = exp(new − old)
        ratios_final   = jnp.exp(new_log_probs_final - old_log_probs_arr)
        clip_fraction  = float(jnp.mean(jnp.abs(ratios_final - 1.0) > self.clip_epsilon))
        approx_kl      = float(jnp.mean(old_log_probs_arr - new_log_probs_final))

        # ── write to shared state under lock ─────────────────────────────
        with self._lock:
            self.actor_params  = actor_params_epoch
            self.critic_params = critic_params_epoch

            self._episodes_completed    += 1
            self._updates_count         += len(rewards)
            self._last_actor_loss        = float(np.mean(actor_losses))
            self._last_critic_loss       = float(np.mean(critic_losses))
            self._last_episode_return    = float(sum(rewards))
            self._last_episode_length    = len(rewards)
            self._last_actor_grad_norm   = last_actor_grad_norm
            self._last_critic_grad_norm  = last_critic_grad_norm
            self._last_mean_advantage    = float(jnp.mean(jnp.abs(advantages_arr)))
            self._last_clip_fraction     = clip_fraction
            self._last_approx_kl         = approx_kl

            # Entropy at zero state — cheap after JIT warmup.
            s_zero  = jnp.zeros(self.state_dim)
            pi_zero = np.array(self._actor_forward_jit(self.actor_params, s_zero))
            self._last_entropy = float(
                -np.sum(pi_zero * np.log(np.clip(pi_zero, 1e-8, 1.0)))
            )

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Persist both actor and critic parameters to a JSON file.

        Uses atomic write (temp file + os.replace) to prevent corruption.

        File structure:
            state_vars, actions, all hyperparameters, actor_params, critic_params
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            actor_params_list  = {k: np.array(v).tolist() for k, v in self.actor_params.items()}
            critic_params_list = {k: np.array(v).tolist() for k, v in self.critic_params.items()}

        data = {
            "state_vars":           self.state_vars,
            "actions":              self.actions,
            "alpha_actor":          self.alpha_actor,
            "alpha_critic":         self.alpha_critic,
            "gamma":                self.gamma,
            "gae_lambda":           self.gae_lambda,
            "clip_epsilon":         self.clip_epsilon,
            "n_epochs":             self.n_epochs,
            "actor_hidden_sizes":   self.actor_hidden_sizes,
            "critic_hidden_sizes":  self.critic_hidden_sizes,
            "entropy_coef":         self.entropy_coef,
            "critic_coef":          self.critic_coef,
            "actor_params":         actor_params_list,
            "critic_params":        critic_params_list,
        }

        tmp_fd, tmp_path = tempfile.mkstemp(dir=save_path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, save_path)
        except Exception:
            os.unlink(tmp_path)
            raise

        actor_sizes  = [self.state_dim] + self.actor_hidden_sizes  + [self.num_actions]
        critic_sizes = [self.state_dim] + self.critic_hidden_sizes + [1]
        num_actor_p  = sum(np.array(v).size for v in actor_params_list.values())
        num_critic_p = sum(np.array(v).size for v in critic_params_list.values())
        print(
            f"PPO saved to {save_path}  "
            f"(actor: {actor_sizes}, {num_actor_p} params | "
            f"critic: {critic_sizes}, {num_critic_p} params)"
        )

    def load(self, filepath: str) -> None:
        """Load actor and critic parameters from a previously saved JSON file.

        Validates every layer shape against the current config before overwriting.
        """
        with open(filepath, 'r') as f:
            data: dict = json.load(f)

        loaded_actor  = {k: jnp.array(v, dtype=jnp.float32) for k, v in data["actor_params"].items()}
        loaded_critic = {k: jnp.array(v, dtype=jnp.float32) for k, v in data["critic_params"].items()}

        actor_sizes = [self.state_dim] + self.actor_hidden_sizes + [self.num_actions]
        for i, (n_in, n_out) in enumerate(zip(actor_sizes[:-1], actor_sizes[1:])):
            expected_W, expected_b = (n_out, n_in), (n_out,)
            if loaded_actor[f"W{i}"].shape != expected_W:
                raise ValueError(
                    f"Loaded actor W{i} shape {loaded_actor[f'W{i}'].shape} != expected {expected_W}."
                )
            if loaded_actor[f"b{i}"].shape != expected_b:
                raise ValueError(
                    f"Loaded actor b{i} shape {loaded_actor[f'b{i}'].shape} != expected {expected_b}."
                )

        critic_sizes = [self.state_dim] + self.critic_hidden_sizes + [1]
        for i, (n_in, n_out) in enumerate(zip(critic_sizes[:-1], critic_sizes[1:])):
            expected_W, expected_b = (n_out, n_in), (n_out,)
            if loaded_critic[f"W{i}"].shape != expected_W:
                raise ValueError(
                    f"Loaded critic W{i} shape {loaded_critic[f'W{i}'].shape} != expected {expected_W}."
                )
            if loaded_critic[f"b{i}"].shape != expected_b:
                raise ValueError(
                    f"Loaded critic b{i} shape {loaded_critic[f'b{i}'].shape} != expected {expected_b}."
                )

        with self._lock:
            self.actor_params  = loaded_actor
            self.critic_params = loaded_critic

        actor_sizes  = [self.state_dim] + self.actor_hidden_sizes  + [self.num_actions]
        critic_sizes = [self.state_dim] + self.critic_hidden_sizes + [1]
        num_actor_p  = sum(v.size for v in loaded_actor.values())
        num_critic_p = sum(v.size for v in loaded_critic.values())
        print(
            f"PPO loaded from {filepath}  "
            f"(actor: {actor_sizes}, {num_actor_p} params | "
            f"critic: {critic_sizes}, {num_critic_p} params)"
        )

    def export_for_godot(self, output_path: str) -> None:
        """Export the ACTOR network to a JSON file that Godot can consume directly.

        Only the actor is exported — Godot runs inference only (no learning),
        so the critic is not needed. The format matches A2CAgent / PolicyGradientDNNAgent
        so the same GDScript forward-pass code works unchanged.

        File structure:
            state_vars, actions, hidden_sizes (actor only), params (actor only)
        """
        with self._lock:
            actor_params_list = {k: np.array(v).tolist() for k, v in self.actor_params.items()}

        export_data = {
            "state_vars":   self.state_vars,
            "actions":      self.actions,
            "hidden_sizes": self.actor_hidden_sizes,
            "params":       actor_params_list,
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(export_data, f, indent=2)

        actor_sizes = [self.state_dim] + self.actor_hidden_sizes + [self.num_actions]
        print(f"Exported PPO actor policy (Godot format) to {out}  (layers: {actor_sizes})")

    # ------------------------------------------------------------------
    # get_stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return diagnostics for the stats logger and live plotter.

        Returns:
            episodes          — total episodes that triggered an update
            updates           — total timesteps processed
            actor_loss        — actor loss averaged over last K epochs
            critic_loss       — critic loss averaged over last K epochs
            episode_return    — sum of rewards in the last episode
            episode_length    — number of steps in the last episode
            mean_advantage    — mean |Â_t| (unnormalised) at last update
            avg_entropy       — H(π) at zero state (cached after last update)
            actor_grad_norm   — grad norm from the last of K actor steps
            critic_grad_norm  — grad norm from the last of K critic steps
            clip_fraction     — fraction of steps where r_t was clipped
            approx_kl         — approx KL divergence (total across K epochs)
            actor_avg_w       — mean |weight| across all actor W matrices
            actor_max_w       — max  |weight| across all actor W matrices
            actor_std_w       — std of all actor weights
            critic_avg_w      — mean |weight| across all critic W matrices
            critic_max_w      — max  |weight| across all critic W matrices
            critic_std_w      — std of all critic weights
            avg_w_actor_N     — mean |weight| of actor layer N (per layer)
        """
        with self._lock:
            actor_snap              = {k: np.array(v) for k, v in self.actor_params.items()}
            critic_snap             = {k: np.array(v) for k, v in self.critic_params.items()}
            episodes_completed      = self._episodes_completed
            updates_count           = self._updates_count
            last_actor_loss         = self._last_actor_loss
            last_critic_loss        = self._last_critic_loss
            last_episode_return     = self._last_episode_return
            last_episode_length     = self._last_episode_length
            last_actor_grad_norm    = self._last_actor_grad_norm
            last_critic_grad_norm   = self._last_critic_grad_norm
            last_entropy            = self._last_entropy
            last_mean_advantage     = self._last_mean_advantage
            last_clip_fraction      = self._last_clip_fraction
            last_approx_kl          = self._last_approx_kl

        # Actor weight stats (W matrices only)
        actor_w_keys  = sorted(k for k in actor_snap  if k.startswith("W"))
        actor_weights = np.concatenate([actor_snap[k].ravel()  for k in actor_w_keys])
        actor_avg_w   = float(np.abs(actor_weights).mean())
        actor_max_w   = float(np.abs(actor_weights).max())
        actor_std_w   = float(actor_weights.std())

        # Critic weight stats (W matrices only)
        critic_w_keys  = sorted(k for k in critic_snap if k.startswith("W"))
        critic_weights = np.concatenate([critic_snap[k].ravel() for k in critic_w_keys])
        critic_avg_w   = float(np.abs(critic_weights).mean())
        critic_max_w   = float(np.abs(critic_weights).max())
        critic_std_w   = float(critic_weights.std())

        # Per-layer actor mean |W| — keyed as avg_w_actor_0, avg_w_actor_1, …
        per_layer_actor = {
            f"avg_w_actor_{i}": round(float(np.abs(actor_snap[k]).mean()), 6)
            for i, k in enumerate(actor_w_keys)
        }

        # Per-layer critic mean |W| — keyed as avg_w_critic_0, avg_w_critic_1, …
        per_layer_critic = {
            f"avg_w_critic_{i}": round(float(np.abs(critic_snap[k]).mean()), 6)
            for i, k in enumerate(critic_w_keys)
        }

        return {
            "episodes":          episodes_completed,
            "updates":           updates_count,
            "actor_loss":        round(last_actor_loss,        6),
            "critic_loss":       round(last_critic_loss,       6),
            "episode_return":    round(last_episode_return,    6),
            "episode_length":    last_episode_length,
            "mean_advantage":    round(last_mean_advantage,    6),
            "avg_entropy":       round(last_entropy,           6),
            "actor_grad_norm":   round(last_actor_grad_norm,   6),
            "critic_grad_norm":  round(last_critic_grad_norm,  6),
            "clip_fraction":     round(last_clip_fraction,     6),
            "approx_kl":         round(last_approx_kl,         6),
            "actor_avg_w":       round(actor_avg_w,            6),
            "actor_max_w":       round(actor_max_w,            6),
            "actor_std_w":       round(actor_std_w,            6),
            "critic_avg_w":      round(critic_avg_w,           6),
            "critic_max_w":      round(critic_max_w,           6),
            "critic_std_w":      round(critic_std_w,           6),
            **per_layer_actor,
            **per_layer_critic,
        }

    # ------------------------------------------------------------------
    # print_config
    # ------------------------------------------------------------------

    def print_config(self) -> None:
        """Print PPO-specific configuration and architecture summary."""
        actor_sizes   = [self.state_dim] + self.actor_hidden_sizes  + [self.num_actions]
        critic_sizes  = [self.state_dim] + self.critic_hidden_sizes + [1]
        actor_arrows  = " → ".join(str(n) for n in actor_sizes)
        critic_arrows = " → ".join(str(n) for n in critic_sizes)
        num_actor_p   = sum(np.array(v).size for v in self.actor_params.values())
        num_critic_p  = sum(np.array(v).size for v in self.critic_params.values())

        print("Algorithm            : Proximal Policy Optimization (PPO-Clip) — JAX DNN")
        print(f"  State variables      : {self.state_vars}")
        print(f"  Actions              : {self.actions}")
        print(f"  Actor arch           : {actor_arrows}  ({num_actor_p} params)")
        print(f"  Critic arch          : {critic_arrows}  ({num_critic_p} params)")
        print(f"  alpha_actor          : {self.alpha_actor}")
        print(f"  alpha_critic         : {self.alpha_critic}")
        print(f"  gamma                : {self.gamma}")
        print(f"  gae_lambda           : {self.gae_lambda}")
        print(f"  clip_epsilon         : {self.clip_epsilon}")
        print(f"  n_epochs             : {self.n_epochs}")
        print(f"  entropy_coef         : {self.entropy_coef}")
        print(f"  critic_coef          : {self.critic_coef}")

    # ------------------------------------------------------------------
    # Convenience properties (used by main.py worker loop)
    # ------------------------------------------------------------------

    @property
    def last_loss(self) -> float:
        """Actor loss (averaged over K epochs) from the most recent update."""
        return self._last_actor_loss

    @property
    def last_critic_loss(self) -> float:
        """Critic loss (averaged over K epochs) from the most recent update."""
        return self._last_critic_loss
