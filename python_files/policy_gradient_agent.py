"""
REINFORCE Policy Gradient Agent for Pong Game

Implements the REINFORCE (Monte Carlo Policy Gradient) algorithm.
The policy is a linear softmax: π(a|s) = softmax(W @ s + b)

Unlike the Q-learning agent, this agent:
  - Works directly on the continuous normalized state — no binning
  - Learns a *policy* (probability distribution over actions) rather than Q-values
  - Updates only at the END of each episode using discounted returns G_t
  - Uses the log-derivative trick: ∇θ J = E[∇θ log π(a|s) · G_t]
"""

import json
import math
import random
import tempfile
import os
import threading
from pathlib import Path

import numpy as np

from rl_agent import RLAgent


class PolicyGradientAgent(RLAgent):
    """REINFORCE policy gradient agent that learns to play Pong.

    The policy is parameterised as a linear softmax over the raw normalised
    state vector (no discretisation needed):

        logits = W @ s + b          shape: (num_actions,)
        π(a|s) = softmax(logits)

    W has shape (num_actions, state_dim) and b has shape (num_actions,).
    Both are updated by gradient ascent on the expected return J(θ).
    """

    DEFAULT_ALPHA   = 0.1   # learning rate
    DEFAULT_GAMMA   = 0.99   # discount factor
    def __init__(
        self,
        state_vars: list[str],
        actions:    list[str],
        alpha:      float = None,
        gamma:      float = None,
    ):
        """
        Initialise the REINFORCE agent.

        Args:
            state_vars: Ordered list of state variable names.
                        Must match the keys sent by Godot every frame.
                        Example: ["paddleA_y", "paddleB_y", "ball_x", "ball_y",
                                  "ball_vx", "ball_vy"]
            actions:    List of action strings.
                        Example: ["UP", "DOWN", "STAY"]
            alpha:      Learning rate for gradient ascent. Default: 3e-3
            gamma:      Discount factor in [0, 1]. Default: 0.99
        """
        # --- configuration -----------------------------------------------
        self.state_vars = state_vars
        self.actions    = actions
        self.alpha      = alpha
        self.gamma      = gamma

        # --- validate inputs before using them ---------------------------
        self._validate_state()
        self._validate_actions()
        self._validate_hyperparameters()

        self.state_dim   = len(self.state_vars)
        self.num_actions = len(self.actions)

        # --- policy parameters (the "brain") ------------------------------
        # W: weight matrix  shape (num_actions, state_dim)
        # b: bias vector    shape (num_actions,)
        #
        # Initialised with small random values so the softmax outputs are
        # slightly non-uniform from the start, helping early exploration.
        rng = np.random.default_rng()
        self.W: np.ndarray = rng.normal(0.0, 0.01, (self.num_actions, self.state_dim))
        self.b: np.ndarray = np.zeros(self.num_actions)

        # --- per-thread trajectory buffer ---------------------------------
        # Each worker thread accumulates its own episode independently.
        # At the end of an episode (done=True) the gradient is computed and
        # the *shared* W and b are updated under _lock.
        #
        # Trajectory format — one entry per frame:
        #   (state_vector, action_index, reward)
        self._local = threading.local()

        # --- shared weight update lock ------------------------------------
        # Protects W and b when multiple worker threads update them
        # simultaneously (same pattern as QLearningAgent._lock for q_table).
        self._lock = threading.Lock()

        # --- statistics ---------------------------------------------------
        self._episodes_completed = 0   # total episodes that triggered an update
        self._updates_count      = 0   # total gradient steps applied
        self._last_loss          = 0.0 # last policy loss (negative mean log-prob · G)
        self._last_episode_return  = 0.0  # return of the last finished episode
        self._last_episode_length  = 0    # number of steps in the last finished episode
        self._last_grad_norm_W     = 0.0  # L2 norm of last dW
        self._last_grad_norm_b     = 0.0  # L2 norm of last db

        #self.print_config()

    # ------------------------------------------------------------------
    # RLAgent interface — configuration display
    # ------------------------------------------------------------------

    def print_config(self) -> None:
        """Print REINFORCE-specific configuration."""
        print("Algorithm: REINFORCE (Policy Gradient)")
        print(f"  State variables : {self.state_vars}")
        print(f"  Actions         : {self.actions}")
        print(f"  Policy shape    : W{list(self.W.shape)}  b{list(self.b.shape)}")
        print(f"  Alpha (lr)      : {self.alpha}")
        print(f"  Gamma (discount): {self.gamma}")

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
    # Class-method constructor (mirrors QLearningAgent.from_dict)
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PolicyGradientAgent':
        """
        Create a PolicyGradientAgent from a YAML/dict config.

        Expected keys:
            state        — list of state variable names
            actions      — list of action strings
            hyperparameters:
                alpha    — learning rate
                gamma    — discount factor

        Example config (YAML):
            state: [paddleA_y, paddleB_y, ball_x, ball_y, ball_vx, ball_vy]
            actions: [UP, DOWN, STAY]
            hyperparameters:
                alpha: 0.003
                gamma: 0.99
        """
        state_vars = config_dict.get('state')
        actions    = config_dict.get('actions')
        hp: dict   = config_dict.get('hyperparameters')
        alpha      = hp.get('alpha')
        gamma      = hp.get('gamma')
        return cls(state_vars=state_vars, actions=actions, alpha=alpha, gamma=gamma)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_state(self) -> None:
        """
        Validate that state_vars is a non-empty list of strings.
        Raises ValueError if the type is wrong.
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
        Raises ValueError if the type is wrong.
        """
        if not isinstance(self.actions, list):
            raise ValueError("actions must be a list.")
        if not self.actions:
            raise ValueError("actions must not be empty.")
        if not all(isinstance(a, str) for a in self.actions):
            raise ValueError("All entries in actions must be strings.")

    def _validate_hyperparameters(self) -> None:
        """
        Validate alpha and gamma; reset to defaults with a warning if invalid.
        Valid ranges: alpha in (0, 1], gamma in [0, 1].
        """
        if not isinstance(self.alpha, (int, float)) or not (0 < self.alpha <= 1):
            print(f"Warning: alpha={self.alpha!r} is invalid. "
                  f"Resetting to default {self.DEFAULT_ALPHA}.")
            self.alpha = self.DEFAULT_ALPHA

        if not isinstance(self.gamma, (int, float)) or not (0 <= self.gamma <= 1):
            print(f"Warning: gamma={self.gamma!r} is invalid. "
                  f"Resetting to default {self.DEFAULT_GAMMA}.")
            self.gamma = self.DEFAULT_GAMMA

    # ------------------------------------------------------------------
    # Forward pass  (state vector → action probabilities)
    # ------------------------------------------------------------------

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Convert a vector of raw scores (logits) into a probability distribution.

        Formula:
            softmax(z)_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))

        Subtracting max(z) before exponentiating is the standard numerical
        stability trick: it keeps all exponents ≤ 0, avoiding overflow while
        leaving the output mathematically unchanged (the constant cancels).

        Args:
            z: Raw logit vector of shape (num_actions,).

        Returns:
            Probability vector of shape (num_actions,).
            All values in (0, 1) and sum exactly to 1.
        """
        # Shift logits so the largest value becomes 0.
        # e^0 = 1, and all others are e^(negative) ∈ (0, 1) — no overflow.
        z_stable = z - np.max(z)

        exp_z = np.exp(z_stable)          # shape (num_actions,)
        return exp_z / np.sum(exp_z)      # normalise → probabilities

    def _forward(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the policy network forward pass.

        Computes:
            z = W @ s + b        (linear projection, shape: num_actions)
            π = softmax(z)       (probability distribution over actions)

        Both z and π are returned because the gradient update needs z
        to avoid recomputing it.

        Args:
            s: State vector of shape (state_dim,). Values normalised to [-1, 1].

        Returns:
            (pi, z) where:
                pi — probability vector (num_actions,)  — use to sample action
                z  — raw logits        (num_actions,)   — use in gradient update
        """
        z  = self.W @ s + self.b    # shape (num_actions,)
        pi = self._softmax(z)       # shape (num_actions,)
        return pi, z

    # ------------------------------------------------------------------
    # Step 3 — state validation + action selection (process_state)
    # ------------------------------------------------------------------

    def _validate_state_dict(self, state_dict: dict) -> None:
        """
        Validate that state_dict has exactly the expected keys and numeric values.

        Warns (but does not raise) if any value is outside [-1, 1] — the
        clamping is done later in _state_to_vector to avoid mutating the
        caller's dict.

        Args:
            state_dict: Raw dict from Godot, e.g. {"ball_x": 0.3, ...}

        Raises:
            ValueError: If keys don't match state_vars or a value is non-numeric.
        """
        incoming_keys  = set(state_dict.keys())
        expected_keys  = set(self.state_vars)

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

    def _state_to_vector(self, state_dict: dict) -> np.ndarray:
        """
        Convert a state dict into a numpy vector in the canonical variable order.

        The order of self.state_vars defines which column of W each state
        variable corresponds to.  Using a different order at inference vs.
        training time would corrupt the dot product W @ s.

        Values are clamped to [-1, 1] here (not in the validator) so we
        never mutate the caller's dict.

        Args:
            state_dict: Validated dict with keys matching self.state_vars.

        Returns:
            np.ndarray of shape (state_dim,), dtype float64.
        """
        return np.array(
            [np.clip(state_dict[v], -1.0, 1.0) for v in self.state_vars],
            dtype=np.float64,
        )

    # --- pending-transition helpers (per-thread) ----------------------
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

    def process_state(self, state_dict: dict[str, float]) -> str:
        """
        Observe the current game state and return the action to take.

        Steps:
            1. Validate and convert state_dict → numpy vector s
            2. Forward pass: z = W @ s + b,  π = softmax(z)
            3. Sample action index from π  (stochastic during training)
            4. Store (s, action_idx) as a pending transition so update()
               can complete the (s, a, r) tuple when the reward arrives.

        Args:
            state_dict: Normalised game state from Godot.
                        Keys must match self.state_vars exactly.

        Returns:
            Action string, e.g. "UP", "DOWN", or "STAY".
        """
        # 1. Validate keys + types; warn on out-of-range values
        self._validate_state_dict(state_dict)

        # 2. Convert to numpy vector in canonical variable order
        s = self._state_to_vector(state_dict)   # shape (state_dim,)

        # 3. Forward pass → probability distribution over actions
        pi, _ = self._forward(s)               # pi shape (num_actions,)

        # 4. Sample — NOT argmax.  This is the stochastic policy:
        #    actions with higher probability are chosen more often, but
        #    every action has a non-zero chance.  This drives exploration
        #    without a separate epsilon parameter.
        action_id = int(np.random.choice(self.num_actions, p=pi))

        # 5. Remember this transition; reward arrives in the next update() call
        self._pending = (s, action_id)

        return self.actions[action_id]

    # ------------------------------------------------------------------
    # Step 4 — compute discounted returns for a finished episode
    # ------------------------------------------------------------------

    def _compute_returns(self, rewards: list[float]) -> np.ndarray:
        """
        Convert a sequence of per-step rewards into discounted returns G_t.

        Definition:
            G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^(T-t)·r_T

        Computed efficiently in a single backward pass:
            G_T     = r_T
            G_{t}   = r_t + γ · G_{t+1}

        The result is then normalised (zero mean, unit variance) to act as a
        simple baseline that reduces gradient variance:
            G_hat_t = (G_t - mean(G)) / (std(G) + ε)

        Why normalise?
            Without a baseline, if all rewards are positive every action gets
            reinforced — even poor ones.  Centring G around 0 means actions
            in above-average timesteps are reinforced and below-average ones
            are suppressed, giving a much cleaner learning signal.

        Args:
            rewards: List of scalar rewards [r_0, r_1, ..., r_T] for one episode.

        Returns:
            np.ndarray of shape (T+1,) containing the normalised returns G_hat_t.
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float64)

        # Backward pass: start from the last timestep and accumulate
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G   # G_{t} = r_t + γ · G_{t+1}
            returns[t] = G

        # Normalise: subtract mean, divide by std
        mean = returns.mean()
        std  = returns.std()
        if std == 0:
            returns = 0
        else:
            returns = (returns - mean) / std

        return returns

    # ------------------------------------------------------------------
    # Step 5 — update: accumulate trajectory, apply gradient at episode end
    # ------------------------------------------------------------------

    def update(self, state: dict[str, float], reward: float, done: bool) -> None:
        """
        Receive feedback for the last action and (on episode end) learn from it.

        Called every frame BEFORE process_state, following the same contract
        as QLearningAgent.update:
            frame N:  update(state_N, prev_reward, done)   ← reward for action N-1
            frame N:  process_state(state_N)               ← choose action N

        Behaviour:
          - If no pending transition (first frame): do nothing.
          - Otherwise: complete (s, a_idx) → (s, a_idx, reward) and append
            to the per-thread trajectory buffer.
          - If done=True: compute discounted returns for the full episode,
            apply the REINFORCE gradient update to W and b, then clear the
            trajectory for the next episode.

        Gradient derivation:
            ∇_W log π(a_i | s)  =  (e_i - π) ⊗ s     shape: (num_actions, state_dim)
            ∇_b log π(a_i | s)  =  (e_i - π)           shape: (num_actions,)

            Where e_i is a one-hot vector for the chosen action and π = softmax(Ws+b).
            (e_i - π) is the "error": +ve for the chosen action, -ve for the others.

            Full episode update:
                ΔW = α · Σ_t  G_hat_t · (e_{a_t} - π_t) ⊗ s_t
                Δb = α · Σ_t  G_hat_t · (e_{a_t} - π_t)

        Args:
            reward: Reward received for the action chosen last frame.
            done:   True if the episode just ended.
        """
        # --- guard: no pending transition on the very first frame --------
        if self._pending is None:
            return

        # --- complete the pending (s, a_idx) tuple with its reward -------
        s, action_id = self._pending
        self._trajectory.append((s, action_id, reward))

        # --- on episode end: compute returns and apply gradient ----------
        if done:
            # Extract the three lists from the trajectory buffer
            states      = [entry[0] for entry in self._trajectory]  # list of np arrays
            action_ids = [entry[1] for entry in self._trajectory]  # list of ints
            rewards     = [entry[2] for entry in self._trajectory]  # list of floats

            # Compute normalised discounted returns G_hat_t for every timestep
            returns = self._compute_returns(rewards)   # shape (T,)

            # Accumulate gradients for W and b across all timesteps
            # We accumulate first, then do a single write under the lock —
            # this minimises lock contention across worker threads.
            dW = np.zeros_like(self.W)   # shape (num_actions, state_dim)
            db = np.zeros_like(self.b)   # shape (num_actions,)

            total_loss = 0.0

            for s_t, a_t, G_hat_t in zip(states, action_ids, returns):

                # Forward pass to get π_t = softmax(W s_t + b) at this timestep.
                # We recompute rather than storing π in the trajectory to keep
                # memory usage proportional to episode length, not O(T × A).
                pi_t, _ = self._forward(s_t)    # shape (num_actions,)

                # One-hot vector e_{a_t}: 1 at the chosen action, 0 elsewhere
                e_t = np.zeros(self.num_actions)
                e_t[a_t] = 1.0

                # Policy gradient error signal: shape (num_actions,)
                # +ve for chosen action (however uncertain we were)
                # -ve for unchosen actions (proportional to their current probability)
                error = e_t - pi_t

                # Gradient of log π(a_t | s_t) w.r.t. W and b:
                #   ∇W = error ⊗ s_t   (outer product → shape num_actions × state_dim)
                #   ∇b = error          (shape num_actions)
                #
                # Multiply by G_hat_t: reinforce (or suppress) based on return quality.
                # This is GRADIENT ASCENT — we add, not subtract.
                dW += G_hat_t * np.outer(error, s_t)
                db += G_hat_t * error

                # Track loss: negative expected log-prob (lower = better)
                if pi_t[a_t] > 0:
                    total_loss -= np.log(pi_t[a_t]) * G_hat_t
                else:
                    print(f"Warning: π(a_t) is zero at timestep with G_hat_t={G_hat_t:.4f}. "
                          f"Skipping log(0) in loss calculation.")
                    total_loss += 0  # log(0) is undefined; treat as zero contribution to loss

            # Gradient norms — useful diagnostics for monitoring stability
            grad_norm_W = float(np.linalg.norm(dW))
            grad_norm_b = float(np.linalg.norm(db))

            # Apply the accumulated gradient and update statistics under the
            # same lock so multiple worker threads don't corrupt shared state.
            with self._lock:
                self.W += self.alpha * dW
                self.b += self.alpha * db

                self._episodes_completed   += 1
                self._updates_count        += len(rewards)
                self._last_loss             = total_loss / max(len(rewards), 1)
                self._last_episode_return   = float(sum(rewards))
                self._last_episode_length   = len(rewards)
                self._last_grad_norm_W      = grad_norm_W
                self._last_grad_norm_b      = grad_norm_b

            # Clear per-thread buffers for the next episode
            self._trajectory = []
            self._pending    = None

    # ------------------------------------------------------------------
    # Step 6 — save and load
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Persist the policy parameters (W and b) to a JSON file.

        The file is self-describing: it stores the state_vars, actions,
        and hyperparameters alongside the weights so you can inspect or
        reload them without the original config.

        File structure:
            {
                "state_vars": ["paddleA_y", ...],
                "actions":    ["UP", "DOWN", "STAY"],
                "alpha":      0.003,
                "gamma":      0.99,
                "W":          [[...], [...], [...]],   # shape (num_actions, state_dim)
                "b":          [...]                    # shape (num_actions,)
            }

        Uses an atomic write (temp file + os.replace) so a crash mid-save
        never leaves a corrupted model on disk.

        Args:
            filepath: Destination path (JSON). Parent dirs are created if needed.
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Snapshot W and b under the lock so we don't save a half-updated state
        with self._lock:
            W_list = self.W.tolist()   # numpy array → plain Python list of lists
            b_list = self.b.tolist()   # numpy array → plain Python list

        data = {
            "state_vars": self.state_vars,
            "actions":    self.actions,
            "alpha":      self.alpha,
            "gamma":      self.gamma,
            "W":          W_list,
            "b":          b_list,
        }

        # Atomic write: write to a temp file first, then rename.
        # os.replace is atomic on POSIX — if we crash mid-write the old
        # file is untouched.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=save_path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, save_path)
        except Exception:
            os.unlink(tmp_path)
            raise

        print(f"Policy saved to {save_path}  (W shape {self.W.shape}, b shape {self.b.shape})")

    def load(self, filepath: str) -> None:
        """
        Load policy parameters (W and b) from a previously saved JSON file.

        Validates that the loaded shapes match the current agent configuration
        before overwriting W and b.

        Args:
            filepath: Path to a JSON file previously written by save().

        Raises:
            ValueError: If the loaded W/b shapes don't match the agent config.
            FileNotFoundError: If filepath doesn't exist.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        W_loaded = np.array(data["W"], dtype=np.float64)
        b_loaded = np.array(data["b"], dtype=np.float64)

        # Sanity-check shapes before overwriting anything
        expected_W_shape = (self.num_actions, self.state_dim)
        expected_b_shape = (self.num_actions,)

        if W_loaded.shape != expected_W_shape:
            raise ValueError(
                f"Loaded W has shape {W_loaded.shape}, "
                f"expected {expected_W_shape}. "
                f"Does the saved file match this agent's config?"
            )
        if b_loaded.shape != expected_b_shape:
            raise ValueError(
                f"Loaded b has shape {b_loaded.shape}, "
                f"expected {expected_b_shape}."
            )

        with self._lock:
            self.W = W_loaded
            self.b = b_loaded

        print(f"Policy loaded from {filepath}  (W shape {self.W.shape}, b shape {self.b.shape})")

    # ------------------------------------------------------------------
    # Step 7 — get_stats (mirrors QLearningAgent interface for logging)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return a dictionary of diagnostics for the stats logger and live plotter.

        Mirrors the key names used by QLearningAgent.get_stats() where
        possible so the existing logging infrastructure works without changes.

        Returns:
            Dictionary containing:
            - episodes:     Total episodes that triggered a gradient update
            - updates:      Total timesteps processed across all episodes
            - last_loss:    Policy loss from the most recent episode
                            (negative mean log-prob weighted by G_hat)
            - avg_w:        Mean absolute value of W (how much the policy
                            has moved from its zero initialisation)
            - max_w:        Max absolute value in W
            - std_w:        Std of W values (spread of learned weights)
            - avg_entropy:  Mean action entropy H(π) = -Σ π log π evaluated
                            at the zero state vector. Starts at log(num_actions)
                            (uniform) and drops as the policy sharpens.
                            Range: [0, log(num_actions)]
            - b_<i>:        Individual bias value for action i (one key per action).
                            e.g. b_0, b_1, b_2 for 3 actions.
                            Tells you which action the policy favours in a neutral
                            state (all features = 0) independently of W.
        """
        with self._lock:
            W_snapshot            = self.W.copy()
            b_snapshot            = self.b.copy()
            episodes_completed    = self._episodes_completed
            updates_count         = self._updates_count
            last_loss             = self._last_loss
            last_episode_return   = self._last_episode_return
            last_episode_length   = self._last_episode_length
            last_grad_norm_W      = self._last_grad_norm_W
            last_grad_norm_b      = self._last_grad_norm_b

        # Weight statistics — track how much the policy has learned
        w_abs  = np.abs(W_snapshot)
        avg_w  = float(w_abs.mean())
        max_w  = float(w_abs.max())
        std_w  = float(W_snapshot.std())

        # Policy entropy at the zero state — a proxy for how "confident"
        # the current policy is. At initialisation W≈0 so π≈uniform and
        # entropy ≈ log(num_actions). As W trains, entropy drops.
        s_zero = np.zeros(self.state_dim)
        pi_zero, _ = self._forward(s_zero)
        # Clip to avoid log(0); softmax output is always > 0 in practice
        avg_entropy = float(-np.sum(pi_zero * np.log(np.clip(pi_zero, 1e-8, 1.0))))

        stats = {
            "episodes":        episodes_completed,
            "updates":         updates_count,
            "last_loss":       round(last_loss, 6),
            "episode_return":  round(last_episode_return, 6),
            "episode_length":  last_episode_length,
            "avg_w":           round(avg_w, 6),
            "max_w":           round(max_w, 6),
            "std_w":           round(std_w, 6),
            "avg_entropy":     round(avg_entropy, 6),
            "grad_norm_W":     round(last_grad_norm_W, 6),
            "grad_norm_b":     round(last_grad_norm_b, 6),
        }

        # Individual bias values — b has only num_actions entries so we log
        # each one separately rather than aggregating. They reveal which action
        # the policy prefers in a neutral state regardless of W.
        for i, val in enumerate(b_snapshot):
            stats[f"b_{i}"] = round(float(val), 6)

        return stats
