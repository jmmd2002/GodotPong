"""Common abstract interface for reinforcement-learning agents.

This base class defines the minimal set of methods used by the
training loop in main.py so different algorithms (Q-learning,
policy gradient, etc.) can be swapped without changing the loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class RLAgent(ABC):
    """Abstract base class for all RL agents.

    Concrete agents must implement the following operations:

    - ``from_dict``: construct an agent from a config dict (typically
      parsed from YAML).
    - ``process_state``: choose an action given the current environment
      state.
    - ``update``: perform a learning step given the previous reward and
      whether the episode terminated.
    - ``save`` / ``load``: serialize and restore learned parameters.
    - ``get_stats``: return a dictionary of diagnostics used by the
      stats logger.
    - ``print_config``: pretty-print algorithm-specific config details.
    """

    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RLAgent":
        """Create an agent instance from a configuration dictionary."""

    @abstractmethod
    def process_state(self, state_dict: Dict[str, float]) -> str:
        """Return the action to take given the current state."""

    @abstractmethod
    def update(state: dict[str, float],  reward: float, done: bool) -> None:
        """Update the agent using the previous transition's feedback."""
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist the agent's learned parameters to ``filepath``."""

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load previously saved parameters from ``filepath``."""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return algorithm-specific diagnostics for logging/monitoring."""

    @abstractmethod
    def print_config(self) -> None:
      """Print algorithm-specific configuration (hyperparameters, spaces, etc.)."""
