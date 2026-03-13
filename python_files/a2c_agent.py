"""
Advantage Actor-Critic (A2C) Agent — Deep Neural Network version (JAX)

=== Why A2C? The problem with REINFORCE ===

REINFORCE updates the actor using the raw discounted return G_t:

    ∇θ J = E[ ∇θ log π(a_t|s_t;θ) · G_t ]

G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ... is the sum of ALL future rewards.
Because every future reward is random, G_t has VERY HIGH VARIANCE.
High-variance gradients point in slightly different directions each episode,
making learning slow and unstable — you take many steps that partially cancel.


=== The Baseline Trick (zero-bias variance reduction) ===

Any function b(s_t) that does NOT depend on the action can be subtracted
from G_t without changing the expected gradient direction (it is "bias-free"):

    ∇θ J = E[ ∇θ log π(a_t|s_t;θ) · (G_t − b(s_t)) ]

Proof that subtracting b is safe:
    E_a[ ∇θ log π(a|s) · b(s) ]
    = b(s) · ∇θ Σ_a π(a|s)   (score function identity)
    = b(s) · ∇θ 1             (probabilities sum to 1)
    = 0

So for any choice of b, the gradient expectation is unchanged.
The variance of (G_t − b(s_t)) is minimised when b(s_t) = V^π(s_t).


=== The Advantage Function ===

The optimal baseline is the state-value function V^π(s_t) — the expected
total return when following policy π from state s_t:

    V^π(s) = E_π[ G_t | s_t = s ]

Using it as a baseline gives the ADVANTAGE:

    A_t  =  G_t − V^π(s_t)

Intuition:
    A_t > 0  →  this action delivered MORE return than average from s_t
                → reinforce this action (increase its probability)
    A_t < 0  →  this action delivered LESS return than average from s_t
                → suppress this action (decrease its probability)
    A_t ≈ 0  →  this action was average — no strong update needed

Compared to G_t alone, A_t has much lower variance because the baseline
V^π(s_t) absorbs the "average quality" of the state, leaving only the
signal that truly distinguishes one action from another.


=== The Two Networks ===

Actor  π(a|s;θ)  —  policy network (same MLP as DNN REINFORCE agent)
    Architecture: state_dim → hidden → … → num_actions → softmax
    Loss: L_actor = −(1/T) Σ_t  A_t · log π(a_t|s_t;θ)
    Interpretation: increase log-prob of actions with positive advantage,
                    decrease log-prob of actions with negative advantage.

Critic  V(s;w)  —  value network
    Architecture: state_dim → hidden → … → 1   (scalar output, NO activation)
    Loss: L_critic = (1/T) Σ_t  (G_t − V(s_t;w))²
    Interpretation: standard mean-squared regression loss.
    The critic is trained to accurately predict G_t so the advantage
    estimate A_t = G_t − V(s_t;w) is as informative as possible.


=== Entropy Bonus (exploration regularisation) ===

Without any extra pressure, the policy will quickly peak on one action and
stop exploring. Adding the entropy H(π) = −Σ_a π(a|s) log π(a|s) to the
objective keeps the policy from collapsing too early:

    L_total = L_actor + c_v · L_critic − c_H · H(π)

c_v  (critic_coef)  — how strongly the critic loss pulls on the shared
                       representation (here actor/critic are SEPARATE networks,
                       so c_v is a weight you can tune without fear of coupling).
c_H  (entropy_coef) — entropy bonus weight; higher → more exploration.
                       Typical range: 0.001 – 0.05.


=== Update Schedule ===

Like the DNN REINFORCE agent, this agent is EPISODIC (Monte Carlo A2C):
    · Collect a full episode trajectory (s_t, a_t, r_t)
    · Compute discounted returns G_t  (exact, no bootstrapping)
    · Compute advantages A_t = G_t − V(s_t;w)
    · Update actor: gradient ascent on E[A_t · log π(a_t|s_t;θ)]
    · Update critic: gradient descent on E[(G_t − V(s_t;w))²]

Alternatively one could use 1-step TD (online update, lower variance at the
cost of bias). We stick with Monte Carlo returns for consistency with the
other agents in this project.


=== Architecture ===

    Actor  params:   actor_W0, actor_b0, actor_W1, actor_b1, ...  (→ num_actions)
    Critic params:  critic_W0, critic_b0, critic_W1, critic_b1, ...  (→ 1)

The actor and critic have SEPARATE hidden_sizes lists so each network can
be sized independently. Giving the critic more capacity (wider/deeper) is
standard practice: the critic approximates V^π(s), a harder regression
target that shifts as the policy improves, so it benefits from extra width.
The actor policy is comparatively easier to represent — it just needs to
learn which action is better, not the exact value of each state.
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
    """Print which compute device JAX is using (same helper as DNN agent)."""
    backend  = jax.default_backend()
    devices  = jax.devices()
    dev_list = ", ".join(str(d) for d in devices)
    if backend == "cpu":
        print(f"[JAX/A2C] Backend: CPU ({dev_list})")
    else:
        print(f"[JAX/A2C] Backend: {backend.upper()} — devices: {dev_list}")


_log_device_info()


class A2CAgent(RLAgent):
    """Advantage Actor-Critic (A2C) agent backed by two separate JAX MLPs.

    Actor  π(a|s;θ):  linear softmax policy, same architecture as the DNN
                       REINFORCE agent.
    Critic V(s;w):    scalar value estimator; trained to predict G_t via MSE.

    The two networks are deliberately SEPARATE (not weight-sharing) so that
    the actor and critic learning rates can be tuned independently and the
    critic loss gradient never corrupts the actor's representation.

    Learning algorithm: Monte Carlo A2C (episodic, full return G_t).

    Update per episode end:
        A_t   = G_t − V(s_t; w)                              advantage
        θ ← θ + α_actor  · ∇θ  Σ_t A_t · log π(a_t|s_t;θ)  actor ascent
        w ← w − α_critic · ∇w  Σ_t (G_t−V(s_t;w))²          critic descent
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
        actor_hidden_sizes:  list[int],
        critic_hidden_sizes: list[int],
        entropy_coef:        float,
        critic_coef:         float,
    ):
        """
        Initialise the A2C agent.

        Args:
            state_vars:   Ordered list of state variable names matching the
                          keys sent by Godot every frame.
                          Example: ["paddle_y", "ball_x", "ball_y", "ball_vx", "ball_vy"]

            actions:      List of action strings.
                          Example: ["UP", "DOWN", "STAY"]

            alpha_actor:  Learning rate for the ACTOR (gradient ASCENT on J).
                          Typical range: 1e-4 – 1e-3.
                          Smaller than REINFORCE because the advantage signal
                          A_t is already lower-variance — you don't need as
                          large a step to make progress.

            alpha_critic: Learning rate for the CRITIC (gradient DESCENT on MSE).
                          Typical range: 5e-4 – 5e-3.
                          Usually 2–10× larger than alpha_actor: the critic is
                          a pure regression problem (MSE is smoother than policy
                          loss) and it needs to track G_t fast enough that the
                          advantage signal fed to the actor stays accurate.

            gamma:        Discount factor γ ∈ [0, 1].
                          γ = 0 → purely myopic (only current reward matters)
                          γ = 1 → full undiscounted return (all future rewards
                                  matter equally — good for short episodes)
                          Typical value: 0.99

            actor_hidden_sizes:  Hidden layer widths for the ACTOR MLP.
                          Example: [64, 64]
                          Architecture: state_dim → 64 → 64 → num_actions

            critic_hidden_sizes: Hidden layer widths for the CRITIC MLP.
                          Example: [128, 64]
                          Architecture: state_dim → 128 → 64 → 1
                          The critic approximates V^π(s) — a regression target
                          that shifts as the policy evolves.  Giving it more
                          capacity than the actor helps it stay accurate,
                          which keeps the advantage signal A_t low-variance.

            entropy_coef: Weight c_H of the entropy bonus in the actor loss.
                          The combined actor objective is:
                              J_actor = E[A_t · log π] + c_H · H(π)
                          Higher c_H → more exploration pressure, slower
                          convergence to any single action.
                          Typical range: 0.001 – 0.05. Start at 0.01.

            critic_coef:  Weight c_v of the critic loss when computing combined
                          diagnostics (e.g. for logging).
                          Does NOT affect the actual update because actor and
                          critic are updated with SEPARATE gradient passes using
                          their own learning rates (alpha_actor / alpha_critic).
                          Stored here for completeness and logging only.
                          Typical value: 0.5
        """

        # ── configuration ──────────────────────────────────────────────
        self.state_vars          = state_vars
        self.actions             = actions
        self.alpha_actor         = alpha_actor
        self.alpha_critic        = alpha_critic
        self.gamma               = gamma
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
        #
        # actor_params:  { "W0": ..., "b0": ..., "W1": ..., "b1": ..., ... }
        #   Final layer shape: (num_actions, prev_hidden) → logits → softmax
        #
        # critic_params: { "W0": ..., "b0": ..., "W1": ..., "b1": ..., ... }
        #   Final layer shape: (1, prev_hidden) → scalar value estimate
        #
        # Each network gets a DIFFERENT random seed so they start with
        # independently varied weights. Using the same seed would make both
        # networks begin with identical representations — not a bug, but
        # wasteful: they'd differentiate only after the first gradient step.
        seed_actor  = random.randint(0, 2**32 - 1)
        seed_critic = random.randint(0, 2**32 - 1)
        self.actor_params:  dict = self._init_params(seed_actor,  self.actor_hidden_sizes,  output_size=self.num_actions)
        self.critic_params: dict = self._init_params(seed_critic, self.critic_hidden_sizes, output_size=1)

        # ── JIT-compiled forward passes ────────────────────────────────
        #
        # Both forward passes are pure static functions (see _forward_actor,
        # _forward_critic below). JIT-compiling them once now means XLA
        # compiles the full computation graph on the FIRST call and reuses
        # the compiled kernel for every subsequent episode — no Python
        # interpreter overhead after warmup.
        self._actor_forward_jit  = jax.jit(A2CAgent._forward_actor)
        self._critic_forward_jit = jax.jit(A2CAgent._forward_critic)

        # ── JIT-compiled loss + gradient functions ─────────────────────
        #
        # jax.value_and_grad returns (loss_scalar, gradient_dict) in one shot.
        # Wrapping with jax.jit means the full backward pass (actor graph or
        # critic graph) is XLA-compiled on the first episode and reused forever.
        #
        # actor_loss differentiates with respect to actor_params (arg index 0).
        # critic_loss differentiates with respect to critic_params (arg index 0).
        self._actor_loss_and_grad_jit = jax.jit(
            jax.value_and_grad(A2CAgent._actor_loss)
        )
        self._critic_loss_and_grad_jit = jax.jit(
            jax.value_and_grad(A2CAgent._critic_loss)
        )

        # ── JIT-compiled critic batch forward pass ─────────────────────
        #
        # At episode end we need V(s_t; w) for EVERY timestep t to compute
        # the advantage A_t = G_t − V(s_t; w).  Calling _forward_critic in a
        # Python loop would be slow.  jax.vmap maps the function over the
        # leading axis of states (axis 0 = time) while broadcasting params
        # (in_axes=None → shared across all mapped calls).  The result is a
        # single fused XLA kernel that runs all T critic evaluations in parallel.
        self._critic_batch_jit = jax.jit(
            jax.vmap(A2CAgent._forward_critic, in_axes=(None, 0))
        )

        # ── per-thread trajectory buffer ───────────────────────────────
        #
        # Each Godot worker runs in its own thread. Using threading.local()
        # gives every thread its own independent trajectory list with zero
        # synchronisation overhead (reads/writes never contend on the lock).
        #
        # Trajectory entry format: (state_jnp_array, action_idx_int, reward_float)
        # One entry is appended per frame. At episode end the full list is
        # processed and then cleared for the next episode.
        self._local = threading.local()

        # ── shared parameter update lock ───────────────────────────────
        #
        # actor_params and critic_params are shared across all worker threads.
        # The lock ensures that save() (called from the autosave timer in
        # main.py) never reads a half-written params dict, and that two
        # concurrent gradient updates don't interleave writes.
        self._lock = threading.Lock()

        # ── statistics (all updated under _lock) ───────────────────────
        self._episodes_completed   = 0    # total episodes that triggered updates
        self._updates_count        = 0    # total timesteps processed
        self._last_actor_loss      = 0.0  # actor loss from the most recent episode
        self._last_critic_loss     = 0.0  # critic loss from the most recent episode
        self._last_episode_return  = 0.0  # Σ r_t of the most recent episode
        self._last_episode_length  = 0    # T of the most recent episode
        self._last_actor_grad_norm = 0.0  # ‖∇θ L_actor‖ at last update
        self._last_critic_grad_norm= 0.0  # ‖∇w L_critic‖ at last update
        self._last_entropy         = 0.0  # H(π) at zero state, cached after update
        self._last_mean_advantage  = 0.0  # mean |A_t| at last update (quality signal)

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def _init_params(self, seed: int, hidden_sizes: list[int], output_size: int) -> dict:
        """
        Build and initialise an MLP parameter dict.

        Called separately for actor (actor_hidden_sizes, output=num_actions)
        and critic (critic_hidden_sizes, output=1).

        Layer architecture:
            [state_dim] + hidden_sizes + [output_size]
            e.g. [5, 64, 64, 3]    for the actor  (state_dim=5, hidden=[64,64],  actions=3)
            e.g. [5, 128, 64, 1]   for the critic (state_dim=5, hidden=[128,64], value=1)

        Initialisation strategy:
            Hidden layers  → He (Kaiming): std = √(2 / n_in)
                The factor 2 compensates for ReLU zeroing ~half the neurons,
                keeping activation variance ≈ 1 across all hidden layers.

            Output layer   → Xavier (Glorot): std = √(1 / n_in)
                No ReLU follows the output so the He factor-2 is not needed.

            Biases → all zeros (standard; biases are learned from there).

        Args:
            seed:        Integer seed for the JAX PRNG key.
            output_size: Width of the output layer (num_actions for actor, 1 for critic).

        Returns:
            dict with keys "W0","b0","W1","b1",...  One pair per layer.
            Wi has shape (n_out, n_in), bi has shape (n_out,).
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
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_state(self) -> None:
        """
        Validate that state_vars is a non-empty list of strings.

        Raises:
            ValueError: If state_vars is not a list, is empty, or contains
                        non-string entries.
        """
        if not isinstance(self.state_vars, list):
            raise ValueError("state_vars must be a list.")
        if not self.state_vars:
            raise ValueError("state_vars must not be empty.")
        if not all(isinstance(v, str) for v in self.state_vars):
            raise ValueError("All entries in state_vars must be strings.")

    def _validate_actions(self) -> None:
        """
        Validate that actions is a non-empty list of strings.

        Raises:
            ValueError: If actions is not a list, is empty, or contains
                        non-string entries.
        """
        if not isinstance(self.actions, list):
            raise ValueError("actions must be a list.")
        if not self.actions:
            raise ValueError("actions must not be empty.")
        if not all(isinstance(a, str) for a in self.actions):
            raise ValueError("All entries in actions must be strings.")

    def _validate_hyperparameters(self) -> None:
        """
        Validate all seven hyperparameters: alpha_actor, alpha_critic, gamma,
        actor_hidden_sizes, critic_hidden_sizes, entropy_coef, and critic_coef.

        Validation rules:
            alpha_actor         : strictly positive float
            alpha_critic        : strictly positive float
            gamma               : float in [0, 1]
            actor_hidden_sizes  : non-empty list of strictly positive integers
            critic_hidden_sizes : non-empty list of strictly positive integers
            entropy_coef        : non-negative float  (0 = no entropy bonus, that's fine)
            critic_coef         : non-negative float  (0 = critic loss not weighted, that's fine)

        Raises:
            ValueError: If any value is missing, non-numeric, or out of range.
                        No silent default fallback — the config must be correct.
        """
        if not isinstance(self.alpha_actor, (int, float)) or not (self.alpha_actor > 0):
            raise ValueError(
                f"alpha_actor={self.alpha_actor!r} is invalid. "
                "Must be a strictly positive number."
            )
        if not isinstance(self.alpha_critic, (int, float)) or not (self.alpha_critic > 0):
            raise ValueError(
                f"alpha_critic={self.alpha_critic!r} is invalid. "
                "Must be a strictly positive number."
            )
        if not isinstance(self.gamma, (int, float)) or not (0 <= self.gamma <= 1):
            raise ValueError(
                f"gamma={self.gamma!r} is invalid. "
                "Must be a number in the range [0, 1]."
            )
        def _check_hidden(sizes, name: str) -> list[int]:
            """Validate and coerce a hidden_sizes list; returns the coerced list."""
            if not isinstance(sizes, list):
                raise ValueError(
                    f"{name} must be a list, got {type(sizes).__name__}."
                )
            if not sizes:
                raise ValueError(
                    f"{name} must not be empty. "
                    "Provide at least one hidden layer width, e.g. [128, 64]."
                )
            converted = []
            for i, size in enumerate(sizes):
                try:
                    size_int = int(size)
                    if size_int != size:
                        raise ValueError()
                except (ValueError, TypeError):
                    raise ValueError(
                        f"{name}[{i}]={size!r} cannot be converted to an integer. "
                        "Each layer width must be a whole positive number (e.g. 128)."
                    )
                if size_int <= 0:
                    raise ValueError(
                        f"{name}[{i}]={size!r} is not strictly positive."
                    )
                converted.append(size_int)
            return converted

        self.actor_hidden_sizes  = _check_hidden(self.actor_hidden_sizes,  "actor_hidden_sizes")
        self.critic_hidden_sizes = _check_hidden(self.critic_hidden_sizes, "critic_hidden_sizes")

        if not isinstance(self.entropy_coef, (int, float)) or self.entropy_coef < 0:
            raise ValueError(
                f"entropy_coef={self.entropy_coef!r} is invalid. "
                "Must be a non-negative number."
            )
        if not isinstance(self.critic_coef, (int, float)) or self.critic_coef < 0:
            raise ValueError(
                f"critic_coef={self.critic_coef!r} is invalid. "
                "Must be a non-negative number."
            )

    # ------------------------------------------------------------------
    # Class-method constructor (mirrors PolicyGradientDNNAgent.from_dict)
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, config_dict: dict) -> "A2CAgent":
        """
        Create an A2CAgent from a YAML/dict config.

        Expected YAML structure:

            state: [paddle_y, ball_x, ball_y, ball_vx, ball_vy]
            actions: [UP, DOWN, STAY]
            hyperparameters:
                alpha_actor:         0.0003
                alpha_critic:        0.001
                gamma:               0.99
                actor_hidden_sizes:  [64, 64]
                critic_hidden_sizes: [128, 64]
                entropy_coef:        0.01
                critic_coef:         0.5

        The 'hyperparameters' block is checked for existence here.
        Individual value validation is delegated to __init__ → _validate_*.

        Raises:
            ValueError: If the 'hyperparameters' block is missing entirely,
                        or if any value is invalid (delegated).
        """
        state_vars: list[str] = config_dict.get("state")
        actions:    list[str] = config_dict.get("actions")

        hp: dict = config_dict.get("hyperparameters")
        if hp is None:
            raise ValueError(
                "Config is missing the 'hyperparameters' block. "
                "Expected keys: alpha_actor, alpha_critic, gamma, "
                "actor_hidden_sizes, critic_hidden_sizes, entropy_coef, critic_coef."
            )

        alpha_actor:         float     = hp.get("alpha_actor")
        alpha_critic:        float     = hp.get("alpha_critic")
        gamma:               float     = hp.get("gamma")
        actor_hidden_sizes:  list[int] = hp.get("actor_hidden_sizes")
        critic_hidden_sizes: list[int] = hp.get("critic_hidden_sizes")
        entropy_coef:        float     = hp.get("entropy_coef")
        critic_coef:         float     = hp.get("critic_coef")

        return cls(
            state_vars=state_vars,
            actions=actions,
            alpha_actor=alpha_actor,
            alpha_critic=alpha_critic,
            gamma=gamma,
            actor_hidden_sizes=actor_hidden_sizes,
            critic_hidden_sizes=critic_hidden_sizes,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
        )

    # ------------------------------------------------------------------
    # Forward passes (pure static — required by JAX jit / grad)
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_actor(params: dict, s: jnp.ndarray) -> jnp.ndarray:
        """Actor forward pass: state vector → action probabilities.

        Computation (L = total number of layers):
            x     = s                              input
            x     = ReLU(W_i @ x + b_i)    for i in 0 … L-2   hidden layers
            z     = W_{L-1} @ x + b_{L-1}         output logits
            π     = softmax(z)                     probability distribution

        Why ReLU on hidden layers?
            ReLU(x) = max(0, x) introduces non-linearity so the network can
            represent curved decision boundaries in state space.  Linear
            layers stacked without activation collapse to a single linear map.

        Why softmax on the output?
            We need π to be a valid probability distribution:
                π_a ∈ (0, 1)   for all actions a
                Σ_a π_a = 1
            softmax(z)_a = exp(z_a) / Σ_{a'} exp(z_{a'}) satisfies both.
            JAX's jax.nn.softmax subtracts max(z) before exponentiating
            (the standard numerical-stability trick) so we never overflow.

        Why static + pure?
            JAX's jit and grad require pure functions — no hidden reads from
            self.  Everything is passed explicitly as arguments so XLA can
            trace and compile the entire computation graph.

        Args:
            params: Actor parameter dict with keys "W0","b0","W1","b1",...
                    Built by _init_params(seed, actor_hidden_sizes, num_actions).
            s:      State vector, shape (state_dim,). Values in [-1, 1].

        Returns:
            π — probability vector over actions, shape (num_actions,).
                All values in (0, 1), sum = 1.
        """
        num_layers = len(params) // 2   # each layer has one W and one b
        x = s
        for i in range(num_layers - 1):
            # Hidden layer: linear projection followed by ReLU non-linearity.
            x = jax.nn.relu(params[f"W{i}"] @ x + params[f"b{i}"])
        # Output layer: logits only — softmax converts them to probabilities.
        logits = params[f"W{num_layers - 1}"] @ x + params[f"b{num_layers - 1}"]
        return jax.nn.softmax(logits)

    @staticmethod
    def _forward_critic(params: dict, s: jnp.ndarray) -> jnp.ndarray:
        """Critic forward pass: state vector → scalar value estimate V(s;w).

        Computation (L = total number of layers):
            x = s
            x = ReLU(W_i @ x + b_i)    for i in 0 … L-2   hidden layers
            V = (W_{L-1} @ x + b_{L-1})[0]     scalar output

        Why NO activation on the output?
            V^π(s) is the expected discounted return from state s:
                V^π(s) = E_π[ r_t + γ r_{t+1} + γ² r_{t+2} + ... | s_t = s ]
            This can be any real number — positive, negative, large or small
            depending on reward scale and γ.  Applying an activation would
            artificially restrict the range:
                ReLU   → V ≥ 0    (can't represent negative expected returns)
                sigmoid → V ∈ (0,1)   (can't represent returns outside [0,1])
            No activation = the output layer is a plain linear projection,
            which can represent any scalar in ℝ.

        The output layer of _init_params uses Xavier initialisation (not He)
        precisely because there is no ReLU following it — He's factor-of-2
        correction is only needed when the next operation is ReLU.

        Args:
            params: Critic parameter dict with keys "W0","b0","W1","b1",...
                    Built by _init_params(seed, critic_hidden_sizes, 1).
                    The final layer has shape (1, prev_hidden) → scalar.
            s:      State vector, shape (state_dim,). Values in [-1, 1].

        Returns:
            V — scalar value estimate, shape () (0-d JAX array).
                Represents V^π(s_t) ≈ expected future return from s_t.
        """
        num_layers = len(params) // 2
        x = s
        for i in range(num_layers - 1):
            x = jax.nn.relu(params[f"W{i}"] @ x + params[f"b{i}"])
        # Output: shape (1,) — squeeze to scalar with [0].
        # The critic's final layer was initialised with output_size=1 in
        # _init_params, so W_{L-1} has shape (1, prev_hidden).
        v = params[f"W{num_layers - 1}"] @ x + params[f"b{num_layers - 1}"]
        return v[0]   # shape () — a plain scalar

    # ------------------------------------------------------------------
    # State validation and conversion
    # ------------------------------------------------------------------

    def _validate_state_dict(self, state_dict: dict) -> None:
        """Validate that state_dict has exactly the expected keys and numeric values.

        Warns (but does not raise) if any value is outside [-1, 1] — clamping
        is done in _state_to_vector to avoid mutating the caller's dict.

        Raises:
            ValueError: If keys don't match state_vars or a value is non-numeric.
        """
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
                print(
                    f"Warning: state value for '{key}' = {val:.4f} "
                    "is outside [-1, 1]. Clamping."
                )

    def _state_to_vector(self, state_dict: dict) -> jnp.ndarray:
        """Convert a validated state dict to a JAX array in canonical order.

        The order of self.state_vars defines which input neuron each variable
        maps to.  Mixing up this order at inference vs. training time would
        completely corrupt the forward pass — the weights were learned for a
        specific neuron-to-variable assignment.

        Values are clamped to [-1, 1] here (not in the validator) so we
        never mutate the caller's dict.

        Returns:
            jnp.ndarray of shape (state_dim,), dtype float32.
        """
        return jnp.array(
            [float(state_dict[v]) for v in self.state_vars],
            dtype=jnp.float32,
        ).clip(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Per-thread pending transition and trajectory buffer
    # ------------------------------------------------------------------

    @property
    def _pending(self) -> tuple | None:
        """Per-thread (state_vec, action_idx) waiting for its reward.

        process_state() stores the (s, a_idx) it chose here.
        update() picks it up on the very next frame, attaches the reward,
        and appends the complete (s, a_idx, r) tuple to _trajectory.

        Using threading.local() means every worker thread has its OWN
        pending slot — no locking needed, no cross-thread contamination.
        """
        return getattr(self._local, "pending", None)

    @_pending.setter
    def _pending(self, value: tuple | None) -> None:
        self._local.pending = value

    @property
    def _trajectory(self) -> list:
        """Per-thread list of (state_vec, action_idx, reward) tuples.

        Grows by one entry per frame.  Cleared at episode end after the
        gradient update has been applied.
        """
        if not hasattr(self._local, "trajectory"):
            self._local.trajectory = []
        return self._local.trajectory

    @_trajectory.setter
    def _trajectory(self, value: list) -> None:
        self._local.trajectory = value

    # ------------------------------------------------------------------
    # process_state — observe state → run actor → return action
    # ------------------------------------------------------------------

    def process_state(self, state_dict: dict[str, float]) -> str:
        """Observe the current game state and return the action to take.

        Only the ACTOR runs here.  The critic is NOT called at inference
        time — it is only needed during update() to compute the advantage
        A_t = G_t − V(s_t; w) at episode end.

        Steps:
            1. Validate state_dict keys and value types.
            2. Convert to a JAX vector s of shape (state_dim,).
            3. Actor forward pass: s → π(a|s;θ), shape (num_actions,).
            4. Sample action stochastically from π.
            5. Store (s, action_idx) as a pending transition.

        Why sample stochastically (not argmax)?
            argmax always picks the currently highest-probability action.
            This is pure exploitation — the agent stops exploring as soon
            as one action edges ahead, which prevents it from discovering
            better strategies.

            Stochastic sampling: action a is chosen with probability π(a|s).
            Actions the policy likes are chosen often; others are chosen
            occasionally, keeping exploration alive.  No separate ε needed —
            exploration is built into the policy distribution itself.

        Args:
            state_dict: Normalised game state from Godot.
                        Keys must match self.state_vars exactly.

        Returns:
            Action string, e.g. "UP", "DOWN", or "STAY".
        """
        # 1. Validate keys + types; warn on out-of-range values.
        self._validate_state_dict(state_dict)

        # 2. Convert to JAX vector in canonical variable order.
        s = self._state_to_vector(state_dict)           # shape (state_dim,)

        # 3. Actor forward pass → probability distribution over actions.
        # Reading self.actor_params without the lock is safe in CPython:
        # reference assignment is atomic (GIL), so we never see a
        # half-written dict. We may see params from the episode before the
        # latest gradient step — acceptable for an on-policy approximation.
        pi = self._actor_forward_jit(self.actor_params, s)   # shape (num_actions,)

        # 4. Sample stochastically — convert JAX array to NumPy for np.random.choice.
        action_idx = int(np.random.choice(self.num_actions, p=np.array(pi)))

        # 5. Remember this transition; reward arrives in the next update() call.
        self._pending = (s, action_idx)

        return self.actions[action_idx]

    # ------------------------------------------------------------------
    # _compute_returns
    # ------------------------------------------------------------------

    def _compute_returns(self, rewards: list[float]) -> jnp.ndarray:
        """Compute raw discounted returns G_t for each timestep t.

        Algorithm (single backward pass, O(T)):
            G_T     = r_T
            G_t     = r_t + γ · G_{t+1}        for t = T-1 … 0

        Why raw (not normalised)?
        ─────────────────────────
        In the REINFORCE agent, returns were normalised to zero mean / unit
        variance because the raw G_t value was used DIRECTLY as the gradient
        weight — normalisation centred the signal so some actions were pushed
        up and others down.  Without it, every action would be reinforced when
        all returns are positive (high bias).

        In A2C the centring is handled by the CRITIC:
            A_t = G_t − V(s_t; w)
        V(s_t) absorbs the average return from that state, so A_t is already
        roughly zero-mean.  We normalise the ADVANTAGES (in update()), not
        the returns — this keeps the critic's regression target on its natural
        scale (raw G_t), while still stabilising the actor's gradient.

        If we normalised G_t here:
            · the critic would be trained to predict normalised values
            · A_t = G_t_norm − V_norm(s_t) would be on a [-ish 1, 1] scale
            · that's not wrong, but it conflates two normalisations and makes
              the critic's output hard to interpret.

        Args:
            rewards: Per-step reward list [r_0, ..., r_{T-1}] for one episode.

        Returns:
            jnp.ndarray of shape (T,), dtype float32 — raw discounted returns.
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return jnp.array(returns)

    # ------------------------------------------------------------------
    # Loss functions (pure static — differentiable by jax.grad)
    # ------------------------------------------------------------------

    @staticmethod
    def _actor_loss(
        actor_params: dict,
        states:       jnp.ndarray,   # shape (T, state_dim)
        action_ids:   jnp.ndarray,   # shape (T,)  int indices
        advantages:   jnp.ndarray,   # shape (T,)  normalised A_hat_t
        entropy_coef: float,
    ) -> jnp.ndarray:
        """Actor loss for one full episode.

        Definition:
            L_actor(θ) = −(1/T) Σ_t  Â_t · log π(a_t | s_t ; θ)
                         − c_H · (1/T) Σ_t  H(π(·|s_t ; θ))

        Where:
            Â_t = normalised advantage  (mean 0, std 1 within the episode)
            H(π) = −Σ_a π(a|s) log π(a|s)   (Shannon entropy)

        Why negative?
            jax.grad performs gradient DESCENT (subtracts gradient).
            We want gradient ASCENT on J(θ) = E[Σ_t r_t].
            Minimising −J  ==  Maximising J.
            So L_actor = −J_actor and we call jax.grad(L_actor).

        Why the entropy bonus?
            − c_H · H(π) in the LOSS means we *minimise* (−c_H · H),
            which is the same as *maximising* c_H · H.
            Higher entropy = more spread-out policy = more exploration.
            Without this term the policy collapses to always picking one
            action before it has properly explored the state space.

        Vectorisation:
            jax.vmap maps _forward_actor over the T-axis of states so all
            T forward passes run in a single fused XLA kernel — no Python
            loop over timesteps.

        Args:
            actor_params: Actor weight dict {W0,b0,W1,b1,...}.
            states:       Stacked state vectors, shape (T, state_dim).
            action_ids:   Indices of chosen actions, shape (T,).
            advantages:   Normalised advantages Â_t, shape (T,).
            entropy_coef: c_H — weight of the entropy bonus.

        Returns:
            Scalar loss value (mean over T).
        """
        # Forward pass on all T states simultaneously.
        # vmap broadcasts actor_params (not mapped) over each row of states.
        all_pi = jax.vmap(
            lambda s: A2CAgent._forward_actor(actor_params, s)
        )(states)                                              # (T, num_actions)

        # Log-probability of the action actually taken at each timestep.
        # all_pi[t, action_ids[t]] selects the right probability per row.
        # +1e-8 prevents log(0) if softmax ever returns an exact 0 (rare).
        t_idx     = jnp.arange(states.shape[0])
        log_probs = jnp.log(all_pi[t_idx, action_ids] + 1e-8)  # (T,)

        # Policy-gradient term: weight each log-prob by its advantage.
        # Negate because jax.grad descends and we want ascent.
        pg_loss = -jnp.mean(advantages * log_probs)

        # Entropy term: H(π) = −Σ_a π_a log π_a, averaged over timesteps.
        # Summing over axis=-1 (actions), then averaging over axis=0 (time).
        entropy = -jnp.mean(
            jnp.sum(all_pi * jnp.log(all_pi + 1e-8), axis=-1)
        )

        # Subtract: minimising (−c_H · H) = maximising c_H · H.
        return pg_loss - entropy_coef * entropy

    @staticmethod
    def _critic_loss(
        critic_params: dict,
        states:        jnp.ndarray,   # shape (T, state_dim)
        returns:       jnp.ndarray,   # shape (T,)  raw G_t
    ) -> jnp.ndarray:
        """Critic loss for one full episode.

        Definition:
            L_critic(w) = (1/T) Σ_t  (G_t − V(s_t ; w))²

        This is standard Mean Squared Error (MSE) regression.
        The critic is trained to predict the discounted return G_t
        from each visited state s_t.

        Why raw G_t (not normalised)?
            See _compute_returns for the full argument.  Short version:
            normalising G_t would change the scale the critic learns to
            predict, corrupting the advantage A_t = G_t − V(s_t) unless
            you also normalise V's output — adding unnecessary complexity.

        Why MSE and not e.g. Huber loss?
            MSE is differentiable everywhere and standard for value regression.
            Huber loss (less sensitive to outliers) is a valid alternative if
            reward spikes cause instability, but MSE is the right starting point.

        Args:
            critic_params: Critic weight dict {W0,b0,W1,b1,...}.
            states:        Stacked state vectors, shape (T, state_dim).
            returns:       Raw discounted returns G_t, shape (T,).

        Returns:
            Scalar MSE loss (mean over T).
        """
        # Critic forward pass on all T states simultaneously.
        values = jax.vmap(
            lambda s: A2CAgent._forward_critic(critic_params, s)
        )(states)                                              # (T,)

        return jnp.mean((returns - values) ** 2)

    # ------------------------------------------------------------------
    # update — accumulate trajectory, apply both gradient updates at end
    # ------------------------------------------------------------------

    def update(self, state: dict, reward: float, done: bool) -> None:
        """Receive feedback for the last action and (on episode end) learn.

        Called every frame BEFORE process_state, following the same contract
        as the DNN REINFORCE agent:
            frame N:  update(state_N, prev_reward, done)   ← reward for action N-1
            frame N:  process_state(state_N)               ← choose action N

        On a non-terminal frame: append (s, a_idx, r) to the per-thread
        trajectory buffer and return immediately.

        On a terminal frame (done=True), run the full A2C update:

            1. Compute returns:   G_t  (backward pass over rewards)
            2. Compute baseline:  V(s_t; w)  for all t  (critic batch forward)
            3. Compute advantage: A_t = G_t − V(s_t; w)
            4. Normalise A_t:     Â_t = (A_t − mean) / (std + ε)
            5. Actor update:      θ ← θ − α_actor  · ∇θ L_actor(Â_t)
            6. Critic update:     w ← w − α_critic · ∇w L_critic(G_t)

        Steps 5 and 6 are SEPARATE gradient passes with SEPARATE learning
        rates.  This is the key difference from shared-trunk A2C variants:
        the actor gradient never touches the critic weights and vice versa.

        Why normalise advantages (step 4)?
            A_t = G_t − V(s_t) is already centred roughly around 0 because
            V(s_t) approximates E[G_t].  But the scale of A_t varies across
            episodes (a high-reward episode has larger |A_t| than a low one).
            Normalising to unit std keeps the actor's gradient magnitude
            stable regardless of reward scale — same motivation as learning-
            rate normalisation but applied to the signal, not the step size.

        Args:
            state:  Current game state (used only to detect first frame
                    via _pending; the state itself was stored by process_state).
            reward: Reward received for the action chosen last frame.
            done:   True if the episode just ended.
        """
        # ── guard: no pending transition on the very first frame ─────────
        if self._pending is None:
            return

        # ── complete the pending (s, a_idx) tuple with its reward ────────
        s, action_id = self._pending
        self._trajectory.append((s, action_id, reward))
        self._pending = None

        if not done:
            return

        # ── snapshot + clear trajectory immediately ───────────────────────
        # Clear before any computation so a new episode can start accumulating
        # even if the gradient step below is slow.
        traj = self._trajectory
        self._trajectory = []

        states     = [entry[0] for entry in traj]   # list of jnp arrays
        action_ids = [entry[1] for entry in traj]   # list of ints
        rewards    = [entry[2] for entry in traj]   # list of floats

        # ── stack into JAX arrays ─────────────────────────────────────────
        states_arr     = jnp.stack(states)                        # (T, state_dim)
        action_ids_arr = jnp.array(action_ids, dtype=jnp.int32)   # (T,)
        returns_arr    = self._compute_returns(rewards)            # (T,) raw G_t

        # ── step 2: critic baseline V(s_t; w) for all t ──────────────────
        #
        # We use the critic params BEFORE the critic update as the baseline.
        # This is correct: we want the baseline to reflect what the critic
        # currently believes, compute advantages from that, and only THEN
        # improve the critic.  Using post-update values would introduce a
        # subtle circular dependency.
        #
        # _critic_batch_jit = jit(vmap(_forward_critic, in_axes=(None, 0)))
        # — params are broadcast (not mapped), states are mapped over axis 0.
        critic_vals: jnp.ndarray = self._critic_batch_jit(self.critic_params, states_arr)  # (T,)

        # ── step 3: raw advantages A_t = G_t − V(s_t) ───────────────────
        advantages = returns_arr - critic_vals                     # (T,)

        # ── step 4: normalise advantages ─────────────────────────────────
        #
        # Â_t = (A_t − mean(A)) / (std(A) + ε)
        #
        # The +ε prevents division by zero in the degenerate case where
        # all advantages are identical (e.g. all-zero rewards in early
        # training).  In that case we still subtract the mean so the actor
        # receives a zero gradient rather than a meaningless large one.
        adv_mean = advantages.mean()
        adv_std  = advantages.std()
        
        norm_advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # ── step 5: actor update ──────────────────────────────────────────
        #
        # L_actor = −mean(Â_t · log π(a_t|s_t)) − c_H · H(π)
        # θ ← θ − α_actor · ∇θ L_actor
        actor_loss, actor_grads = self._actor_loss_and_grad_jit(
            self.actor_params, states_arr, action_ids_arr,
            norm_advantages, self.entropy_coef
        )
        actor_grad_norm = float(
            jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(actor_grads)))
        )

        # ── step 6: critic update ─────────────────────────────────────────
        #
        # L_critic = mean((G_t − V(s_t;w))²)
        # w ← w − α_critic · ∇w L_critic
        critic_loss, critic_grads = self._critic_loss_and_grad_jit(
            self.critic_params, states_arr, returns_arr
        )
        critic_grad_norm = float(
            jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(critic_grads)))
        )
        # ── apply updates + record stats under lock ───────────────────────
        #
        # Both networks are written atomically under the same lock so save()
        # (called from the autosave timer in main.py) never reads a state
        # where one network has been updated but not the other.
        with self._lock:
            self.actor_params = jax.tree.map(
                lambda p, g: p - self.alpha_actor * g,
                self.actor_params, actor_grads,
            )
            self.critic_params = jax.tree.map(
                lambda p, g: p - self.alpha_critic * g,
                self.critic_params, critic_grads,
            )
            self._episodes_completed    += 1
            self._updates_count         += len(rewards)
            self._last_actor_loss        = float(actor_loss)
            self._last_critic_loss       = float(critic_loss)
            self._last_episode_return    = float(sum(rewards))
            self._last_episode_length    = len(rewards)
            self._last_actor_grad_norm   = actor_grad_norm
            self._last_critic_grad_norm  = critic_grad_norm
            self._last_mean_advantage    = float(jnp.mean(jnp.abs(advantages)))

            # Cache entropy at zero state — cheap after JIT warmup.
            s_zero  = jnp.zeros(self.state_dim)
            pi_zero = np.array(self._actor_forward_jit(self.actor_params, s_zero))
            self._last_entropy = float(
                -np.sum(pi_zero * np.log(np.clip(pi_zero, 1e-8, 1.0)))
            )

    def save(self, filepath: str) -> None:
        """Persist both actor and critic parameters to a self-describing JSON file.

        File structure:
            {
                "state_vars":          [...],
                "actions":             [...],
                "alpha_actor":         0.0003,
                "alpha_critic":        0.001,
                "gamma":               0.99,
                "actor_hidden_sizes":  [64, 64],
                "critic_hidden_sizes": [128, 64],
                "entropy_coef":        0.01,
                "critic_coef":         0.5,
                "actor_params":  {"W0": [[...]], "b0": [...], ...},
                "critic_params": {"W0": [[...]], "b0": [...], ...}
            }

        Uses an atomic write (temp file + os.replace) so a crash mid-save
        never leaves a corrupted file on disk.

        Args:
            filepath: Destination path (JSON). Parent dirs are created if needed.
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            actor_params_list  = {k: np.array(v).tolist() for k, v in self.actor_params.items()}
            critic_params_list = {k: np.array(v).tolist() for k, v in self.critic_params.items()}

        data = {
            "state_vars":          self.state_vars,
            "actions":             self.actions,
            "alpha_actor":         self.alpha_actor,
            "alpha_critic":        self.alpha_critic,
            "gamma":               self.gamma,
            "actor_hidden_sizes":  self.actor_hidden_sizes,
            "critic_hidden_sizes": self.critic_hidden_sizes,
            "entropy_coef":        self.entropy_coef,
            "critic_coef":         self.critic_coef,
            "actor_params":        actor_params_list,
            "critic_params":       critic_params_list,
        }

        tmp_fd, tmp_path = tempfile.mkstemp(dir=save_path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, save_path)
        except Exception:
            os.unlink(tmp_path)
            raise

        num_actor_params  = sum(np.array(v).size for v in actor_params_list.values())
        num_critic_params = sum(np.array(v).size for v in critic_params_list.values())
        actor_sizes  = [self.state_dim] + self.actor_hidden_sizes + [self.num_actions]
        critic_sizes = [self.state_dim] + self.critic_hidden_sizes + [1]
        print(
            f"A2C saved to {save_path}  "
            f"(actor: {actor_sizes}, {num_actor_params} params | "
            f"critic: {critic_sizes}, {num_critic_params} params)"
        )

    def load(self, filepath: str) -> None:
        """Load actor and critic parameters from a previously saved JSON file.

        Validates every layer's shape against the current agent config before
        overwriting anything — avoids silently loading a mismatched model.

        Args:
            filepath: Path to a JSON file written by save().

        Raises:
            FileNotFoundError: If filepath does not exist.
            ValueError: If any layer shape doesn't match the current architecture.
        """
        with open(filepath, 'r') as f:
            data: dict = json.load(f)

        loaded_actor  = {k: jnp.array(v, dtype=jnp.float32) for k, v in data["actor_params"].items()}
        loaded_critic = {k: jnp.array(v, dtype=jnp.float32) for k, v in data["critic_params"].items()}

        # Validate actor layer shapes
        actor_sizes = [self.state_dim] + self.actor_hidden_sizes + [self.num_actions]
        for i, (n_in, n_out) in enumerate(zip(actor_sizes[:-1], actor_sizes[1:])):
            expected_W, expected_b = (n_out, n_in), (n_out,)
            if loaded_actor[f"W{i}"].shape != expected_W:
                raise ValueError(
                    f"Loaded actor W{i} has shape {loaded_actor[f'W{i}'].shape}, "
                    f"expected {expected_W}. Does the saved file match this agent's config?"
                )
            if loaded_actor[f"b{i}"].shape != expected_b:
                raise ValueError(
                    f"Loaded actor b{i} has shape {loaded_actor[f'b{i}'].shape}, "
                    f"expected {expected_b}."
                )

        # Validate critic layer shapes
        critic_sizes = [self.state_dim] + self.critic_hidden_sizes + [1]
        for i, (n_in, n_out) in enumerate(zip(critic_sizes[:-1], critic_sizes[1:])):
            expected_W, expected_b = (n_out, n_in), (n_out,)
            if loaded_critic[f"W{i}"].shape != expected_W:
                raise ValueError(
                    f"Loaded critic W{i} has shape {loaded_critic[f'W{i}'].shape}, "
                    f"expected {expected_W}. Does the saved file match this agent's config?"
                )
            if loaded_critic[f"b{i}"].shape != expected_b:
                raise ValueError(
                    f"Loaded critic b{i} has shape {loaded_critic[f'b{i}'].shape}, "
                    f"expected {expected_b}."
                )

        with self._lock:
            self.actor_params  = loaded_actor
            self.critic_params = loaded_critic

        num_actor_params  = sum(v.size for v in loaded_actor.values())
        num_critic_params = sum(v.size for v in loaded_critic.values())
        print(
            f"A2C loaded from {filepath}  "
            f"(actor: {actor_sizes}, {num_actor_params} params | "
            f"critic: {critic_sizes}, {num_critic_params} params)"
        )

    def export_for_godot(self, output_path: str) -> None:
        """Export the ACTOR network to a JSON file that Godot can consume directly.

        Only the actor is exported — Godot runs inference only (no learning),
        so the critic is not needed. The exported format intentionally matches
        the PolicyGradientDNNAgent export so the same GDScript forward-pass
        code can be reused unchanged.

        File structure:
            {
                "state_vars":   [...],
                "actions":      [...],
                "hidden_sizes": [64, 64],
                "params": {
                    "W0": [[...]], "b0": [...],
                    "W1": [[...]], "b1": [...],
                    "W2": [[...]], "b2": [...]
                }
            }

        Training-only fields (alpha_*, gamma, critic) are intentionally omitted.

        Args:
            output_path: Destination path (JSON). Parent dirs are created if needed.
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
        print(f"Exported actor policy (Godot format) to {out}  (layers: {actor_sizes})")

    def get_stats(self) -> dict:
        """Return diagnostics for the stats logger and live plotter.

        Weight statistics are computed separately for actor and critic,
        aggregated across all W matrices (biases excluded).

        Per-layer actor weight means are included as avg_w_actor_0,
        avg_w_actor_1, … so the stats logger can log them as extra columns.

        Returns:
            episodes          — total episodes that triggered a gradient update
            updates           — total timesteps processed across all episodes
            actor_loss        — actor loss from the most recent episode
            critic_loss       — critic loss from the most recent episode
            episode_return    — sum of rewards in the last finished episode
            episode_length    — number of steps in the last finished episode
            mean_advantage    — mean |A_t| at the last update
            avg_entropy       — H(π) at zero state (cached from last gradient step)
            actor_grad_norm   — L2 norm of the actor gradient at last update
            critic_grad_norm  — L2 norm of the critic gradient at last update
            actor_avg_w       — mean |weight| across all actor W matrices
            actor_max_w       — max |weight| across all actor W matrices
            actor_std_w       — std of all actor weights
            critic_avg_w      — mean |weight| across all critic W matrices
            critic_max_w      — max |weight| across all critic W matrices
            critic_std_w      — std of all critic weights
            avg_w_actor_N     — mean |weight| of actor layer N (for each layer)
        """
        with self._lock:
            actor_snap             = {k: np.array(v) for k, v in self.actor_params.items()}
            critic_snap            = {k: np.array(v) for k, v in self.critic_params.items()}
            episodes_completed     = self._episodes_completed
            updates_count          = self._updates_count
            last_actor_loss        = self._last_actor_loss
            last_critic_loss       = self._last_critic_loss
            last_episode_return    = self._last_episode_return
            last_episode_length    = self._last_episode_length
            last_actor_grad_norm   = self._last_actor_grad_norm
            last_critic_grad_norm  = self._last_critic_grad_norm
            last_entropy           = self._last_entropy
            last_mean_advantage    = self._last_mean_advantage

        # Actor weight stats (W matrices only)
        actor_w_keys   = sorted(k for k in actor_snap  if k.startswith("W"))
        actor_weights  = np.concatenate([actor_snap[k].ravel()  for k in actor_w_keys])
        actor_avg_w    = float(np.abs(actor_weights).mean())
        actor_max_w    = float(np.abs(actor_weights).max())
        actor_std_w    = float(actor_weights.std())

        # Critic weight stats (W matrices only)
        critic_w_keys  = sorted(k for k in critic_snap if k.startswith("W"))
        critic_weights = np.concatenate([critic_snap[k].ravel() for k in critic_w_keys])
        critic_avg_w   = float(np.abs(critic_weights).mean())
        critic_max_w   = float(np.abs(critic_weights).max())
        critic_std_w   = float(critic_weights.std())

        # Per-layer actor mean |W| — keyed as avg_w_actor_0, avg_w_actor_1, …
        per_layer_actor_avg_w = {
            f"avg_w_actor_{i}": round(float(np.abs(actor_snap[k]).mean()), 6)
            for i, k in enumerate(actor_w_keys)
        }

        return {
            "episodes":          episodes_completed,
            "updates":           updates_count,
            "actor_loss":        round(last_actor_loss,       6),
            "critic_loss":       round(last_critic_loss,      6),
            "episode_return":    round(last_episode_return,   6),
            "episode_length":    last_episode_length,
            "mean_advantage":    round(last_mean_advantage,   6),
            "avg_entropy":       round(last_entropy,          6),
            "actor_grad_norm":   round(last_actor_grad_norm,  6),
            "critic_grad_norm":  round(last_critic_grad_norm, 6),
            "actor_avg_w":       round(actor_avg_w,           6),
            "actor_max_w":       round(actor_max_w,           6),
            "actor_std_w":       round(actor_std_w,           6),
            "critic_avg_w":      round(critic_avg_w,          6),
            "critic_max_w":      round(critic_max_w,          6),
            "critic_std_w":      round(critic_std_w,          6),
            **per_layer_actor_avg_w,
        }

    def print_config(self) -> None:
        """Print A2C-specific configuration and architecture summary."""
        actor_sizes   = [self.state_dim] + self.actor_hidden_sizes  + [self.num_actions]
        critic_sizes  = [self.state_dim] + self.critic_hidden_sizes + [1]
        actor_arrows  = " → ".join(str(n) for n in actor_sizes)
        critic_arrows = " → ".join(str(n) for n in critic_sizes)
        num_actor_params  = sum(np.array(v).size for v in self.actor_params.values())
        num_critic_params = sum(np.array(v).size for v in self.critic_params.values())

        print("Algorithm            : Advantage Actor-Critic (A2C) — JAX DNN")
        print(f"  State variables      : {self.state_vars}")
        print(f"  Actions              : {self.actions}")
        print(f"  Actor arch           : {actor_arrows}  ({num_actor_params} params)")
        print(f"  Critic arch          : {critic_arrows}  ({num_critic_params} params)")
        print(f"  alpha_actor          : {self.alpha_actor}")
        print(f"  alpha_critic         : {self.alpha_critic}")
        print(f"  gamma                : {self.gamma}")
        print(f"  entropy_coef         : {self.entropy_coef}")
        print(f"  critic_coef          : {self.critic_coef}")

    # ------------------------------------------------------------------
    # Convenience properties (used by main.py worker loop)
    # ------------------------------------------------------------------

    @property
    def last_loss(self) -> float:
        """Actor loss from the most recent gradient update."""
        return self._last_actor_loss

    @property
    def last_critic_loss(self) -> float:
        """Critic loss from the most recent gradient update."""
        return self._last_critic_loss

    @property
    def last_entropy(self) -> float:
        """H(π) at zero state from the most recent update."""
        return self._last_entropy
