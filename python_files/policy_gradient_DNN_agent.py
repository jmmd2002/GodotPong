"""
REINFORCE Policy Gradient Agent — Deep Neural Network version (JAX)

Replaces the single linear layer  z = W @ s + b  with a multi-layer
perceptron (MLP):

    h1  = ReLU( W1 @ s  + b1 )      hidden layer 1   (state_dim  → 128)
    h2  = ReLU( W2 @ h1 + b2 )      hidden layer 2   (128        →  64)
    z   =       W3 @ h2 + b3        output logits    ( 64        →  num_actions)
    π   = softmax( z )               action probabilities

Why a DNN over the linear policy?
----------------------------------
The linear policy can only draw straight-line decision boundaries in state
space.  A DNN with non-linear activations (ReLU) can express arbitrary
curved boundaries — capturing interactions like "ball approaching from above
AND fast → reinforce UP strongly".

"""

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

# JAX defaults to 32-bit floats.  We keep that here — it is faster and
# sufficient for a small Pong MLP.  Uncomment the line below if you ever
# need 64-bit precision (e.g. for numerical gradient checks).
# jax.config.update("jax_enable_x64", True)


def _log_device_info() -> None:
    """Print which compute device JAX is using and warn when CPU-only.

    Called once at module import.  The goal is to make it immediately
    visible whether a GPU is being used, rather than silently falling
    back to CPU training.

    --- How to enable GPU support on Intel hardware ---
    JAX Intel GPU support requires THREE things to align:
      1. intel-extension-for-openxla installed (not on PyPI; .whl from
         https://github.com/intel/intel-extension-for-openxla/releases).
      2. A supported JAX version (latest plugin v0.7.0 supports JAX ≤ 0.5.0).
      3. A supported GPU: Intel Data Center Max Series or Arc B-Series.
         Iris Xe integrated graphics (this machine) is NOT supported.

    As of JAX 0.9.1 + Python 3.14 + Iris Xe, CPU is the only option.
    When a compatible GPU + plugin become available, JAX will automatically
    detect it and this function will print the GPU name instead.
    """
    backend  = jax.default_backend()
    devices  = jax.devices()
    dev_list = ", ".join(str(d) for d in devices)

    if backend == "cpu":
        print(f"[JAX] Backend: CPU ({dev_list})")
        print("[JAX] WARNING: No GPU detected. Training will run on CPU.")
        print("[JAX]   Intel GPU support requires intel-extension-for-openxla")
        print("[JAX]   and a compatible JAX version / GPU type.")
    else:
        print(f"[JAX] Backend: {backend.upper()} — devices: {dev_list}")


_log_device_info()


class PolicyGradientDNNAgent(RLAgent):
    """REINFORCE policy gradient agent backed by a JAX MLP.

    The policy π(a | s ; θ) is parameterised by an MLP:

        h1 = ReLU( W1 s  + b1 )
        h2 = ReLU( W2 h1 + b2 )
        z  =       W3 h2 + b3
        π  = softmax( z )

    All weights θ = {W1,b1,W2,b2,W3,b3} live in self.params, protected by
    self._lock for safe concurrent updates from multiple worker threads.

    Learning algorithm: REINFORCE (Monte Carlo Policy Gradient).
    Updates are applied at the END of each episode using the log-derivative
    trick and discounted normalised returns — identical to the linear agent.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        state_vars:   list[str],
        actions:      list[str],
        alpha:        float,
        gamma:        float,
        hidden_sizes: list[int],
    ):
        """
        Initialise the DNN REINFORCE agent.

        Args:
            state_vars:   Ordered list of state variable names that must match
                          the keys sent by Godot every frame.
                          Example: ["paddle_y", "ball_x", "ball_y", "ball_vx", "ball_vy"]
            actions:      List of action strings.
                          Example: ["UP", "DOWN", "STAY"]
            alpha:        Learning rate for gradient ascent. Must be in (0, 1].
            gamma:        Discount factor in [0, 1].
            hidden_sizes: List of hidden layer widths, e.g. [128, 64].
                          Determines the MLP architecture:
                              state_dim → hidden_sizes[0] → ... → num_actions
                          At least one element required.
        """
        # --- configuration -----------------------------------------------
        self.state_vars   = state_vars
        self.actions      = actions
        self.alpha        = alpha
        self.gamma        = gamma
        self.hidden_sizes = hidden_sizes

        # --- validate all inputs before using them -----------------------
        self._validate_state()
        self._validate_actions()
        self._validate_hyperparameters()

        # --- derived dimensions ------------------------------------------
        self.state_dim   = len(self.state_vars)
        self.num_actions = len(self.actions)

        # --- policy parameters (the "brain") -----------------------------
        # All weights live in a single dict so jax.grad can differentiate
        # through the entire network in one shot.
        # See _init_params() for initialisation details.
        # A random seed means each agent instance starts with different weights,
        # which is useful when running multiple workers in parallel.
        seed = random.randint(0, 2**32 - 1)
        self.params: dict = self._init_params(seed)

        # JIT-compile the forward pass once at construction time.
        # Every subsequent call to self._forward_jit(params, s) runs
        # XLA-compiled code instead of the Python interpreter.
        # _forward is a static method (pure function) so jit can trace it.
        self._forward_jit = jax.jit(PolicyGradientDNNAgent._forward)

        # JIT-compile the gradient computation once at construction time.
        # jax.value_and_grad returns both the loss scalar and the gradient dict.
        # Wrapping with jax.jit means XLA compiles the full backward pass once
        # (on the first call) and reuses the compiled kernel for every subsequent
        # episode — instead of retracing the computation graph each episode.
        self._loss_and_grad_jit = jax.jit(
            jax.value_and_grad(PolicyGradientDNNAgent._episode_loss)
        )

        # --- per-thread trajectory buffer --------------------------------
        # Each worker thread accumulates its own episode independently.
        # At episode end the gradient is computed and the shared params
        # dict is updated under _lock.
        #
        # Trajectory format — one entry per frame:
        #   (state_vector, action_index, reward)
        self._local = threading.local()

        # --- shared parameter update lock --------------------------------
        # Protects self.params during save() which may be called from a
        # different thread (e.g. the autosave timer in main.py).
        self._lock = threading.Lock()

        # --- statistics --------------------------------------------------
        self._episodes_completed  = 0     # total episodes that triggered an update
        self._updates_count       = 0     # total gradient steps applied
        self._last_loss           = 0.0   # last policy loss (negative mean log-prob · G)
        self._last_episode_return = 0.0   # sum of rewards in the last finished episode
        self._last_episode_length = 0     # number of steps in the last finished episode
        self._last_grad_norm      = 0.0   # L2 norm of the full gradient at last update

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def _init_params(self, seed: int = 0) -> dict:
        """
        Build and initialise the MLP parameter dict using JAX.

        Layer widths are determined by:
            layer_sizes = [state_dim] + hidden_sizes + [num_actions]
            e.g.  [5, 128, 64, 3]  for coach with two hidden layers.

        Initialisation strategy:
            Hidden layers  → He (Kaiming) initialisation:
                std = sqrt(2 / n_in)
                The factor 2 compensates for ReLU zeroing ~half the neurons,
                keeping activation variance stable across layers.

            Output layer   → Xavier (Glorot) initialisation:
                std = sqrt(1 / n_in)
                No ReLU follows the output, so the factor is 1 instead of 2.

            Biases → all zeros. He/Xavier applies only to weights; biases
                     start at zero and are learned from there.

        JAX RNG note:
            JAX has no global random state — randomness requires an explicit
            key. jax.random.split produces a fresh subkey for each layer,
            keeping initialisation reproducible and thread-safe.

        Args:
            seed: Integer seed for the JAX PRNG key. Default 0.

        Returns:
            dict with keys "W0","b0","W1","b1",... one pair per layer.
            Wi has shape (n_out, n_in), bi has shape (n_out,).
        """
        key = jax.random.PRNGKey(seed)

        # Full list of layer widths from input to output.
        # e.g. hidden_sizes=[128, 64] → layer_sizes=[state_dim, 128, 64, num_actions]
        layer_sizes = [self.state_dim] + self.hidden_sizes + [self.num_actions]

        params = {}
        num_layers = len(layer_sizes) - 1   # number of weight matrices

        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            key, subkey = jax.random.split(key)

            is_output_layer = (i == num_layers - 1)

            if is_output_layer:
                # Xavier: std = sqrt(1 / n_in) — no ReLU after this layer
                std = jnp.sqrt(1.0 / n_in)
            else:
                # He: std = sqrt(2 / n_in) — ReLU follows this layer
                std = jnp.sqrt(2.0 / n_in)

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
        Validate all hyperparameters: alpha, gamma, and hidden_sizes.

        hidden_sizes is an architectural hyperparameter — it controls model
        capacity and is set before training, just like alpha and gamma.  All
        three therefore live under the same 'hyperparameters:' YAML key and
        are validated together here.

        Valid ranges / rules:
            alpha        : (0, 1]   — learning rate, must be strictly positive
            gamma        : [0, 1]   — discount factor, 0 = myopic, 1 = full return
            hidden_sizes : non-empty list of strictly positive integers
                           e.g. [128, 64] → two hidden layers of width 128 and 64.
                           At least one hidden layer is required (otherwise this
                           reduces to the linear policy).

        Raises:
            ValueError: If any value is missing, non-numeric, or out of range.
                        No default fallback — the caller must supply valid values.
        """
        if not isinstance(self.alpha, (int, float)) or not (0 < self.alpha <= 1):
            raise ValueError(
                f"alpha={self.alpha!r} is invalid. "
                "Must be a number in the range (0, 1]."
            )
        if not isinstance(self.gamma, (int, float)) or not (0 <= self.gamma <= 1):
            raise ValueError(
                f"gamma={self.gamma!r} is invalid. "
                "Must be a number in the range [0, 1]."
            )
        if not isinstance(self.hidden_sizes, list):
            raise ValueError(
                f"hidden_sizes must be a list, got {type(self.hidden_sizes).__name__}."
            )
        if not self.hidden_sizes:
            raise ValueError(
                "hidden_sizes must not be empty. "
                "Provide at least one hidden layer width, e.g. [128, 64]."
            )
        converted = []
        for i, size in enumerate(self.hidden_sizes):
            try:
                size_int = int(size)
                # int(3.7) → 3  silently truncates, which would be surprising.
                # Only accept values that are exactly representable as int
                # (e.g. 128, 128.0 are fine; 128.7 is not).
                if size_int != size:
                    raise ValueError()
            except (ValueError, TypeError):
                raise ValueError(
                    f"hidden_sizes[{i}]={size!r} cannot be converted to an integer. "
                    "Each layer width must be a whole positive number (e.g. 128)."
                )
            if size_int <= 0:
                raise ValueError(
                    f"hidden_sizes[{i}]={size!r} is invalid. "
                    "Each layer width must be strictly positive."
                )
            converted.append(size_int)
        # Write back the coerced list so downstream code always sees plain ints
        self.hidden_sizes = converted

    # ------------------------------------------------------------------
    # Class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PolicyGradientDNNAgent':
        """
        Create a PolicyGradientDNNAgent from a YAML/dict config.

        Expected structure (matches PolicyGradientDNN_coach/student.yaml):

            state: [paddle_y, ball_x, ball_y, ball_vx, ball_vy]
            actions: [UP, DOWN, STAY]
            hyperparameters:
                alpha: 0.001
                gamma: 0.99
                hidden_sizes: [128, 64]

        All values are passed directly to __init__, where the _validate_*
        methods raise ValueError if anything is missing or out of range.

        The 'hyperparameters' block itself is checked here explicitly:
        if it is absent, config_dict.get('hyperparameters') returns None
        and calling .get() on None would raise an unhelpful AttributeError
        rather than a clear ValueError — so we catch that case early.

        Args:
            config_dict: Dictionary parsed from a YAML config file.

        Returns:
            A fully initialised PolicyGradientDNNAgent.

        Raises:
            ValueError: If the 'hyperparameters' block is missing entirely,
                        or if any individual value is invalid (delegated to
                        __init__ → _validate_*).
        """
        state_vars: list[str] = config_dict.get('state')
        actions: list[str] = config_dict.get('actions')

        hp = config_dict.get('hyperparameters')
        if hp is None:
            raise ValueError(
                "Config is missing the 'hyperparameters' block. "
                "Expected keys: alpha, gamma, hidden_sizes."
            )

        alpha: float = hp.get('alpha')
        gamma: float = hp.get('gamma')
        hidden_sizes: list[int] = hp.get('hidden_sizes')

        return cls(
            state_vars=state_vars,
            actions=actions,
            alpha=alpha,
            gamma=gamma,
            hidden_sizes=hidden_sizes,
        )

    # ------------------------------------------------------------------
    # Forward pass  (params, state vector → action probabilities)
    # ------------------------------------------------------------------

    @staticmethod
    def _forward(params: dict, s: jnp.ndarray) -> jnp.ndarray:
        """
        Run the MLP forward pass.

        This is a *pure static function* — it takes params and s explicitly
        and reads nothing from self.  This is required by JAX:
          - jax.grad needs to differentiate through params → must be explicit
          - jax.jit needs to trace the function → no hidden Python state

        Computation:
            x = s                                          (input)
            for each hidden layer i:
                x = ReLU( W_i @ x + b_i )                 (hidden layer)
            z = W_{L-1} @ x + b_{L-1}                     (output logits)
            π = softmax( z )                               (probabilities)

        ReLU is applied after every layer except the last.  The output layer
        feeds into softmax, not ReLU — so no activation is applied there.

        softmax numerical stability:
            jax.nn.softmax subtracts max(z) before exponentiating internally,
            the same trick used in the linear agent's hand-written _softmax.
            We get this for free here.

        Args:
            params: dict with keys "W0","b0","W1","b1",...  Built by _init_params.
                    Wi shape: (n_out, n_in),  bi shape: (n_out,).
            s:      State vector, shape (state_dim,).  Values in [-1, 1].

        Returns:
            π — probability vector over actions, shape (num_actions,).
                All values in (0, 1), sums to 1.
        """
        num_layers = len(params) // 2   # each layer contributes one W and one b

        x = s
        for i in range(num_layers - 1):
            # Hidden layer: linear projection followed by ReLU
            x = jax.nn.relu(params[f"W{i}"] @ x + params[f"b{i}"])

        # Output layer: linear projection only — softmax follows
        logits = params[f"W{num_layers - 1}"] @ x + params[f"b{num_layers - 1}"]

        return jax.nn.softmax(logits)

    # ------------------------------------------------------------------
    # State validation and conversion
    # ------------------------------------------------------------------

    def _validate_state_dict(self, state_dict: dict) -> None:
        """
        Validate that state_dict has exactly the expected keys and numeric values.

        Warns (but does not raise) if any value is outside [-1, 1] — clamping
        is done later in _state_to_vector to avoid mutating the caller's dict.

        Args:
            state_dict: Raw dict from Godot, e.g. {"ball_x": 0.3, ...}

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
                    f"is outside [-1, 1]. Clamping."
                )

    def _state_to_vector(self, state_dict: dict) -> jnp.ndarray:
        """
        Convert a state dict into a JAX array in the canonical variable order.

        The order of self.state_vars defines which input neuron each state
        variable corresponds to.  Using a different order at inference vs.
        training time would corrupt W0 @ s entirely.

        Returns a jnp.array (not np.array) so it feeds directly into
        _forward_jit without any conversion step.

        Values are clamped to [-1, 1] here — not in the validator — so we
        never mutate the caller's dict.

        Args:
            state_dict: Validated dict with keys matching self.state_vars.

        Returns:
            jnp.ndarray of shape (state_dim,), dtype float32.
        """
        return jnp.array(
            [float(state_dict[v]) for v in self.state_vars],
            dtype=jnp.float32,
        ).clip(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Pending transition helpers (per-thread)
    # ------------------------------------------------------------------
    # process_state records the (state_vec, action_idx) it chose.
    # update() picks that up next frame, attaches the reward, and pushes
    # a complete (s, a_idx, r) tuple onto _trajectory.

    @property
    def _pending(self) -> tuple | None:
        """Per-thread (state_vec, action_idx) waiting for its reward."""
        return getattr(self._local, 'pending', None)

    @_pending.setter
    def _pending(self, value: tuple | None):
        self._local.pending = value

    # ------------------------------------------------------------------
    # Trajectory buffer helpers (per-thread)
    # ------------------------------------------------------------------

    @property
    def _trajectory(self) -> list:
        """Per-thread list of (state_vec, action_idx, reward) tuples."""
        if not hasattr(self._local, 'trajectory'):
            self._local.trajectory = []
        return self._local.trajectory

    @_trajectory.setter
    def _trajectory(self, value: list):
        self._local.trajectory = value

    # ------------------------------------------------------------------
    # process_state — observe state, run forward pass, return action
    # ------------------------------------------------------------------

    def process_state(self, state_dict: dict[str, float]) -> str:
        """
        Observe the current game state and return the action to take.

        Steps:
            1. Validate state_dict keys and value types
            2. Convert to a JAX vector s of shape (state_dim,)
            3. Forward pass: s → MLP → π  (probability distribution)
            4. Sample action stochastically from π
            5. Store (s, action_idx) as a pending transition so update()
               can complete the (s, a, r) tuple when the reward arrives

        Stochastic sampling (not argmax) is what drives exploration —
        actions with higher probability are chosen more often, but every
        action has a non-zero chance.  No separate epsilon needed.

        Args:
            state_dict: Normalised game state from Godot.
                        Keys must match self.state_vars exactly.

        Returns:
            Action string, e.g. "UP", "DOWN", or "STAY".
        """
        # 1. Validate keys + types; warn on out-of-range values
        self._validate_state_dict(state_dict)

        # 2. Convert to JAX vector in canonical variable order
        s = self._state_to_vector(state_dict)       # shape (state_dim,)

        # 3. Forward pass → probability distribution over actions
        # self.params is read directly without holding the lock.  In CPython
        # a reference assignment is atomic (GIL), so this never sees a
        # half-written dict.  It may however see the params from the episode
        # *before* the latest gradient has been applied — see the note in
        # update() about the on-policy approximation.
        # The moment the gradient worker executes self.params = new_params,
        # the very next call to process_state picks up the updated policy
        # with no extra signalling required.
        pi = self._forward_jit(self.params, s)      # shape (num_actions,)

        # 4. Sample — convert JAX array to NumPy for np.random.choice
        # This is a cheap one-time copy; pi has only num_actions elements.
        action_idx = int(np.random.choice(self.num_actions, p=np.array(pi)))

        # 5. Remember this transition; reward arrives in the next update() call
        self._pending = (s, action_idx)

        return self.actions[action_idx]

    # ------------------------------------------------------------------
    # Compute discounted returns
    # ------------------------------------------------------------------

    def _compute_returns(self, rewards: list[float]) -> jnp.ndarray:
        """
        Convert a sequence of per-step rewards into normalised discounted returns.

        Algorithm (backward pass):
            G_T     = r_T
            G_t     = r_t + γ · G_{t+1}

        Then normalise to zero mean, unit variance:
            G_hat_t = (G_t - mean(G)) / (std(G) + ε)

        Why normalise?
            Without a baseline, if all rewards are positive every action gets
            reinforced — even poor ones.  Centering around 0 means actions
            in above-average timesteps are reinforced and below-average ones
            are suppressed, giving a cleaner gradient signal.

        Args:
            rewards: List of scalar rewards [r_0, ..., r_T] for one episode.

        Returns:
            jnp.ndarray of shape (T,) — normalised returns, ready to feed
            directly into the JAX loss function without further conversion.
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)

        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        mean = returns.mean()
        std  = returns.std()
        if std > 0:
            returns = (returns - mean) / std
        else:
            returns = returns - mean
            print(f"Warning: zero std in returns normalisation; returns: {returns}")

        return jnp.array(returns)   # move to JAX device once, here

    # ------------------------------------------------------------------
    # Episode loss (pure static — differentiable by jax.grad)
    # ------------------------------------------------------------------

    @staticmethod
    def _episode_loss(
        params:      dict,
        states:      jnp.ndarray,   # shape (T, state_dim)
        action_ids:  jnp.ndarray,   # shape (T,)   int indices
        returns:     jnp.ndarray,   # shape (T,)   normalised G_hat_t
    ) -> jnp.ndarray:
        """
        REINFORCE loss for one full episode.

        Definition:
            L(θ) = - (1/T) Σ_t  G_hat_t · log π(a_t | s_t ; θ)

        This is the negative expected return — minimising L is equivalent to
        maximising J(θ).  jax.grad(L)(params) therefore gives us the direction
        to subtract from params (gradient descent on L = ascent on J).

        Why static + pure?
            jax.grad requires a pure function — no hidden reads from self.
            Everything the function needs (params, states, actions, returns)
            is passed explicitly so JAX can trace and differentiate it.

        Vectorisation via jax.vmap:
            Instead of a Python loop over T timesteps, we use vmap to run
            _forward on all states simultaneously:
                vmap(_forward)(states)  →  all_pi  shape (T, num_actions)
            This processes the entire trajectory in one fused XLA kernel.

        Args:
            params:     MLP parameter dict.
            states:     Stacked state vectors, shape (T, state_dim).
            action_ids: Indices of chosen actions, shape (T,).
            returns:    Normalised discounted returns G_hat_t, shape (T,).

        Returns:
            Scalar loss value (mean over T timesteps).
        """
        # Vectorise the forward pass over all T timesteps at once.
        # vmap maps _forward over the leading axis of states (axis 0 = time).
        # params is shared across all timesteps (not mapped).
        all_pi = jax.vmap(
            lambda s: PolicyGradientDNNAgent._forward(params, s)
        )(states)
        # all_pi shape: (T, num_actions)

        # Log probability of the action actually taken at each timestep.
        # all_pi[t, action_ids[t]] selects the probability of the chosen action.
        # +1e-8 avoids log(0) if softmax ever produces an exact zero (rare but safe).
        t_indices = jnp.arange(states.shape[0])
        log_probs = jnp.log(all_pi[t_indices, action_ids] + 1e-8)
        # log_probs shape: (T,)

        # REINFORCE loss: negate because we want gradient ASCENT on J(θ),
        # but jax.grad performs gradient DESCENT (subtracts gradient).
        # Minimising -J == maximising J.
        return -jnp.mean(returns * log_probs)

    # ------------------------------------------------------------------
    # update — accumulate trajectory, apply gradient at episode end
    # ------------------------------------------------------------------

    def update(self, state: dict[str, float], reward: float, done: bool) -> None:
        """
        Receive feedback for the last action and (on episode end) learn from it.

        Called every frame BEFORE process_state, following the same contract:
            frame N:  update(state_N, prev_reward, done)   ← reward for action N-1
            frame N:  process_state(state_N)               ← choose action N

        Behaviour:
          - If no pending transition (first frame of episode): do nothing.
          - Otherwise: complete (s, a_idx) → (s, a_idx, reward) and append
            to the per-thread trajectory buffer.
          - If done=True: compute discounted returns, call jax.grad on the
            episode loss, apply gradient descent on Loss (= ascent on J),
            then clear the trajectory for the next episode.

        Gradient update:
            grads = ∇_θ L(θ)  via jax.grad   (automatic, no manual derivation)
            θ ← θ - α · grads                 (descent on L = ascent on J)

            jax.tree.map applies this update to every leaf of the params dict
            (every W_i and b_i) in one line.

        Args:
            state:  Current game state (used only to detect first frame via
                    _pending — the state itself was already stored by process_state).
            reward: Reward received for the action chosen last frame.
            done:   True if the episode just ended.
        """
        # --- guard: no pending transition on the very first frame --------
        if self._pending is None:
            return

        # --- complete the pending (s, a_idx) tuple with its reward -------
        s, action_id = self._pending
        self._trajectory.append((s, action_id, reward))
        self._pending = None

        # --- on episode end: compute returns and apply gradient ----------
        if not done:
            return

        # Snapshot and clear the trajectory immediately.
        states     = [entry[0] for entry in self._trajectory]   # list of jnp arrays
        action_ids = [entry[1] for entry in self._trajectory]   # list of ints
        rewards    = [entry[2] for entry in self._trajectory]   # list of floats
        self._trajectory = []

        # Stack into JAX arrays for vectorised processing.
        states_arr     = jnp.stack(states)                       # (T, state_dim)
        action_ids_arr = jnp.array(action_ids, dtype=jnp.int32)  # (T,)
        returns_arr    = self._compute_returns(rewards)           # (T,)

        # Compute loss and gradients via the JIT-compiled function.
        # XLA reuses the compiled kernel from the first episode onward,
        # so this is fast enough to run inline without blocking Godot.
        loss, grads = self._loss_and_grad_jit(
            self.params, states_arr, action_ids_arr, returns_arr
        )

        grad_norm = float(
            jnp.sqrt(sum(
                jnp.sum(g ** 2) for g in jax.tree.leaves(grads)
            ))
        )

        # Apply the gradient update and record stats under the lock so
        # save() (called from a separate thread) never sees a half-written state.
        with self._lock:
            self.params = jax.tree.map(
                lambda p, g: p - self.alpha * g,
                self.params, grads,
            )
            self._episodes_completed  += 1
            self._updates_count       += len(rewards)
            self._last_loss            = float(loss)
            self._last_episode_return  = float(sum(rewards))
            self._last_episode_length  = len(rewards)
            self._last_grad_norm       = grad_norm

    # ------------------------------------------------------------------
    # save and load
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Persist all MLP parameters to a JSON file.

        The file is self-describing: it stores state_vars, actions,
        hyperparameters AND the full params dict so the model can be
        reloaded or inspected without the original config file.

        File structure:
            {
                "state_vars":   [...],
                "actions":      [...],
                "alpha":        0.001,
                "gamma":        0.99,
                "hidden_sizes": [128, 64],
                "params": {
                    "W0": [[...]],   "b0": [...],
                    "W1": [[...]],   "b1": [...],
                    "W2": [[...]],   "b2": [...]
                }
            }

        Uses an atomic write (temp file + os.replace) so a crash mid-save
        never leaves a corrupted model on disk.

        Args:
            filepath: Destination path (JSON). Parent dirs are created if needed.
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Snapshot params under the lock — never save a half-updated state
        with self._lock:
            params_list = {k: np.array(v).tolist() for k, v in self.params.items()}

        data = {
            "state_vars":   self.state_vars,
            "actions":      self.actions,
            "alpha":        self.alpha,
            "gamma":        self.gamma,
            "hidden_sizes": self.hidden_sizes,
            "params":       params_list,
        }

        tmp_fd, tmp_path = tempfile.mkstemp(dir=save_path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, save_path)
        except Exception:
            os.unlink(tmp_path)
            raise

        num_params = sum(np.array(v).size for v in params_list.values())
        print(f"Policy saved to {save_path}  ({num_params} parameters, "
              f"layers: {[self.state_dim] + self.hidden_sizes + [self.num_actions]})")

    def load(self, filepath: str) -> None:
        """
        Load MLP parameters from a previously saved JSON file.

        Validates that every layer's shape matches the current agent config
        before overwriting anything — avoids silently loading a mismatched model.

        Args:
            filepath: Path to a JSON file previously written by save().

        Raises:
            ValueError: If any layer shape doesn't match the current config.
            FileNotFoundError: If filepath doesn't exist.
        """
        with open(filepath, 'r') as f:
            data: dict[str, dict] = json.load(f)
        
        loaded_params = {k: jnp.array(v, dtype=jnp.float32)
                         for k, v in data["params"].items()}

        # Validate every layer shape before touching self.params
        layer_sizes = [self.state_dim] + self.hidden_sizes + [self.num_actions]
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            expected_W = (n_out, n_in)
            expected_b = (n_out,)
            if loaded_params[f"W{i}"].shape != expected_W:
                raise ValueError(
                    f"Loaded W{i} has shape {loaded_params[f'W{i}'].shape}, "
                    f"expected {expected_W}. "
                    f"Does the saved file match this agent's config?"
                )
            if loaded_params[f"b{i}"].shape != expected_b:
                raise ValueError(
                    f"Loaded b{i} has shape {loaded_params[f'b{i}'].shape}, "
                    f"expected {expected_b}."
                )

        with self._lock:
            self.params = loaded_params

        num_params = sum(v.size for v in loaded_params.values())
        print(f"Policy loaded from {filepath}  ({num_params} parameters, "
              f"layers: {layer_sizes})")

    # ------------------------------------------------------------------
    # export_for_godot
    # ------------------------------------------------------------------

    def export_for_godot(self, output_path: str) -> None:
        """
        Export the MLP to a JSON file that Godot can consume directly.

        The exported file contains the full architecture description and all
        layer weights so Godot can reconstruct and run the forward pass:

            {
                "state_vars":   [...],
                "actions":      [...],
                "hidden_sizes": [128, 64],
                "params": {
                    "W0": [[...]], "b0": [...],
                    "W1": [[...]], "b1": [...],
                    "W2": [[...]], "b2": [...]
                }
            }

        Training-only fields (alpha, gamma) are intentionally omitted —
        Godot only needs to run inference, not learn.

        Godot GDScript runs the same forward pass:
            x = s
            for each hidden layer i:  x = relu(Wi @ x + bi)
            logits = W_last @ x + b_last
            action = argmax(softmax(logits))
        """
        with self._lock:
            params_list = {k: np.array(v).tolist() for k, v in self.params.items()}

        export_data = {
            "state_vars":   self.state_vars,
            "actions":      self.actions,
            "hidden_sizes": self.hidden_sizes,
            "params":       params_list,
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(export_data, f, indent=2)

        layer_sizes = [self.state_dim] + self.hidden_sizes + [self.num_actions]
        print(f"Exported policy (Godot format) to {out}  "
              f"(layers: {layer_sizes})")

    # ------------------------------------------------------------------
    # get_stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return diagnostics for the stats logger and live plotter.

        Weight statistics are aggregated across ALL weight matrices (W0, W1, ...)
        — biases are excluded as they have a different scale.

        Entropy is evaluated at the zero state vector — a proxy for policy
        confidence.  At initialisation all weights ≈ 0 so π ≈ uniform and
        entropy ≈ log(num_actions).  As the policy sharpens, entropy drops.

        Returns:
            episodes       — total episodes that triggered a gradient update
            updates        — total timesteps processed across all episodes
            last_loss      — policy loss from the most recent episode
            episode_return — sum of rewards in the last finished episode
            episode_length — number of steps in the last finished episode
            avg_w          — mean |weight| across all layers
            max_w          — max |weight| across all layers
            std_w          — std of all weights across all layers
            avg_entropy    — H(π) at zero state = -Σ π log π
            grad_norm      — L2 norm of full gradient at last update
        """
        with self._lock:
            params_snap           = {k: np.array(v) for k, v in self.params.items()}
            episodes_completed    = self._episodes_completed
            updates_count         = self._updates_count
            last_loss             = self._last_loss
            last_episode_return   = self._last_episode_return
            last_episode_length   = self._last_episode_length
            last_grad_norm        = self._last_grad_norm

        # Aggregate weight stats across all W matrices (skip biases)
        all_weights = np.concatenate([
            params_snap[k].ravel()
            for k in params_snap if k.startswith("W")
        ])
        avg_w = float(np.abs(all_weights).mean())
        max_w = float(np.abs(all_weights).max())
        std_w = float(all_weights.std())

        # Entropy at zero state — runs through JIT forward pass
        s_zero = jnp.zeros(self.state_dim)
        pi_zero = np.array(self._forward_jit(self.params, s_zero))
        avg_entropy = float(-np.sum(pi_zero * np.log(np.clip(pi_zero, 1e-8, 1.0))))

        return {
            "episodes":        episodes_completed,
            "updates":         updates_count,
            "last_loss":       round(last_loss, 6),
            "episode_return":  round(last_episode_return, 6),
            "episode_length":  last_episode_length,
            "avg_w":           round(avg_w, 6),
            "max_w":           round(max_w, 6),
            "std_w":           round(std_w, 6),
            "avg_entropy":     round(avg_entropy, 6),
            "grad_norm":       round(last_grad_norm, 6),
        }

    # ------------------------------------------------------------------
    # print_config
    # ------------------------------------------------------------------

    def print_config(self) -> None:
        """Print DNN-specific configuration and architecture summary."""
        layer_sizes = [self.state_dim] + self.hidden_sizes + [self.num_actions]
        arrows = " → ".join(str(n) for n in layer_sizes)
        num_params = sum(np.array(v).size for v in self.params.values())

        print("Algorithm : REINFORCE (Policy Gradient) — DNN version (JAX)")
        print(f"  State variables : {self.state_vars}")
        print(f"  Actions         : {self.actions}")
        print(f"  Architecture    : {arrows}  ({num_params} parameters)")
        print(f"  Hidden sizes    : {self.hidden_sizes}")
        print(f"  Alpha (lr)      : {self.alpha}")
        print(f"  Gamma (discount): {self.gamma}")
