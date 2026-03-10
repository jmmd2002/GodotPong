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


class PolicyGradientAgent:
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

        print("REINFORCE Agent initialised")
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
        Create a ReinforceAgent from a YAML/dict config.

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
