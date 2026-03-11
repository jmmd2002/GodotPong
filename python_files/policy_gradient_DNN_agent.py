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

Why JAX?
---------
* jax.numpy  is nearly a drop-in for numpy — familiar API.
* jax.grad   differentiates any pure Python/JAX function automatically,
             so we never have to derive or code gradients by hand.
* jax.jit    compiles the forward pass and gradient computation via XLA,
             giving significant speed-ups on every episode.

Key design difference from policy_gradient_agent.py
-----------------------------------------------------
JAX requires *pure functions* — functions whose output depends only on
their explicit inputs, with no hidden side-effects.  Because of this, we
cannot store weights silently inside `self` and read them implicitly.
Instead, all learnable parameters live in a single `params` dict:

    params = {
        "W1": ndarray(128, state_dim),   "b1": ndarray(128,),
        "W2": ndarray(64,  128),         "b2": ndarray(64,),
        "W3": ndarray(num_actions, 64),  "b3": ndarray(num_actions,),
    }

This dict is passed *explicitly* to every function that needs weights, and
jax.grad differentiates through it in one shot.  self._lock still protects
it during multi-threaded updates — nothing changes there.

Everything else — REINFORCE algorithm, trajectory buffer, threading model,
process_state / update interface, config loading — is identical to the
linear agent.
"""

import json
import os
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

    DEFAULT_ALPHA   = 1e-3   # learning rate
    DEFAULT_GAMMA   = 0.99   # discount factor
    HIDDEN_SIZES    = (128, 64)   # MLP hidden layer widths — easy to change here
