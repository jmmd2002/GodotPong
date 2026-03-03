"""
Q-Learning Agent for Pong Game

This agent learns to play Pong using Q-learning with value iteration.
"""

import random
import ast
import json
from pathlib import Path


class QLearningAgent:
    """Q-Learning agent that learns to play Pong."""

    DEFAULT_STATE = ["paddleA_y", "paddleB_y", "ball_x", "ball_y", "ball_vx", "ball_vy"]
    DEFAULT_ACTIONS = ["UP", "DOWN", "STAY"]
    DEFAULT_BINS_CONFIG = {
        "paddleA_y": 6,
        "paddleB_y": 6,
        "ball_x": 8,
        "ball_y": 6,
        "ball_vx": 3,
        "ball_vy": 3
    }
    DEFAULT_ALPHA = 0.1
    DEFAULT_GAMMA = 0.95
    DEFAULT_EPSILON = 0.2

    def __init__(self, state=None, bins_config=None, actions=None, 
                 alpha=None, gamma=None, epsilon=None):
        """
        Initialize the Q-learning agent.
        
        Args:
            state: List of state variable names in order.
                  Example: ["paddleA_y", "paddleB_y", "ball_x", "ball_y", "ball_vx", "ball_vy"]
                  If None, uses default configuration.
            bins_config: Dictionary mapping state variable names to number of bins.
                        Example: {"paddleA_y": 6, "paddleB_y": 6, "ball_x": 8, "ball_y": 6, 
                                 "ball_vx": 3, "ball_vy": 3}
                        If None, uses default configuration.
            actions: List of available action strings. If None, uses default.
            alpha: Learning rate (0 to 1). Default: 0.1
            gamma: Discount factor (0 to 1). Default: 0.95
            epsilon: Exploration rate (0 to 1). Default: 0.2
        """
        # Set state variables
        if state is None:
            self.state = self.DEFAULT_STATE
        else:
            self.state = state
        
        # Set available actions
        if actions is None:
            self.actions = self.DEFAULT_ACTIONS
        else:
            self.actions = actions
        
        # Default bins configuration if none provided
        if bins_config is None:
            self.bins_config = self.DEFAULT_BINS_CONFIG
        else:
            self.bins_config = bins_config

        # Learning hyperparameters
        self.alpha = alpha if alpha is not None else self.DEFAULT_ALPHA      # Learning rate
        self.gamma = gamma if gamma is not None else self.DEFAULT_GAMMA     # Discount factor
        self.epsilon = epsilon if epsilon is not None else self.DEFAULT_EPSILON  # Exploration rate
        
        # Validate configuration
        self._validate_state()
        self._validate_actions()
        self._validate_bins()
        self._validate_hyperparameters()
        
        
        # Initialize Q-table: stores Q-values for (state, action) pairs
        # Q(s, a) = estimated value of taking action a in state s
        self.q_table = {}
        
        # For tracking the last state and action for learning
        self._last_state = None
        self._last_action = None
        
        # For tracking learning progress
        self._updates_count = 0
        self._exploration_count = 0
        self._exploitation_count = 0
        self._max_q_value = 0.0
        
        print("Q-Learning Agent initialized")
        print(f"State variables: {self.state}")
        print(f"Available actions: {self.actions}")
        print(f"State bins configuration: {self.bins_config}")
    
    def _validate_state(self):
        """
        Validate that state is a list of strings.
        Raises ValueError if validation fails.
        """
        if not isinstance(self.state, list):
            raise ValueError("State should be a list.")
        
        if not all(isinstance(s, str) for s in self.state):
            raise ValueError("All state variables should be strings.")
    
    def _validate_actions(self):
        """
        Validate that actions is a list of strings.
        Raises ValueError if validation fails.
        """
        if not isinstance(self.actions, list):
            raise ValueError("Actions should be a list.")
        
        if not all(isinstance(a, str) for a in self.actions):
            raise ValueError("All actions should be strings.")
    
    def _validate_bins(self):
        """
        Validate that bins_config keys match state variables and values are int or float.
        Raises ValueError if validation fails.
        """
        # Get keys from bins_config and state
        bins_keys = set(self.bins_config.keys())
        state_keys = set(self.state)
        
        # Check if keys match
        if bins_keys != state_keys:
            raise ValueError("Invalid keys in bins_config: keys must match state variables.")
        
        # Check if all values are int or float
        if not all(isinstance(v, (int, float)) for v in self.bins_config.values()):
            raise ValueError("Bin values should be int or float.")
        
    def _validate_hyperparameters(self):
        """
        Validate that alpha, gamma, and epsilon are in the range [0, 1].
        Raises ValueError if validation fails.
        """
        if not isinstance(self.alpha, (int, float)):
            raise ValueError("Alpha (learning rate) should be a number.")
        if not (0 <= self.alpha <= 1):
            raise ValueError("Alpha (learning rate) should be in the range [0, 1].")
        
        if not isinstance(self.gamma, (int, float)):
            raise ValueError("Gamma (discount factor) should be a number.")
        if not (0 <= self.gamma <= 1):
            raise ValueError("Gamma (discount factor) should be in the range [0, 1].")
        
        if not isinstance(self.epsilon, (int, float)):
            raise ValueError("Epsilon (exploration rate) should be a number.")
        if not (0 <= self.epsilon <= 1):
            raise ValueError("Epsilon (exploration rate) should be in the range [0, 1].")
    
    @classmethod
    def from_dict(cls: 'QLearningAgent', config_dict: dict):
        """
        Create a QLearningAgent instance from a configuration dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
                        Expected keys: 'state', 'bins_config', 'actions', 'hyperparameters'
                        hyperparameters should contain: 'alpha', 'gamma', 'epsilon'
        
        Returns:
            QLearningAgent instance configured with the provided parameters.
        """
        state = config_dict.get('state', None)
        bins_config = config_dict.get('bins_config', None)
        actions = config_dict.get('actions', None)
        
        # Extract hyperparameters from config
        hyperparams: dict = config_dict.get('hyperparameters', {})
        alpha = hyperparams.get('alpha', 0.1)
        gamma = hyperparams.get('gamma', 0.95)
        epsilon = hyperparams.get('epsilon', 0.2)
        
        return cls(state=state, bins_config=bins_config, actions=actions,
                   alpha=alpha, gamma=gamma, epsilon=epsilon)
    
    def _calculate_bin_index(self, normalized_value: float, num_bins: int) -> int:
        """
        Calculate which bin a normalized value falls into.
        
        Args:
            normalized_value: Value in [0, 1] range
            num_bins: Number of bins to divide the range into
            
        Returns:
            Bin index in range [0, num_bins-1]
        """
        # Calculate which bin this value falls into
        bin_index = int(normalized_value * num_bins)
        
        # Clamp to valid range [0, num_bins-1]
        bin_index = min(max(bin_index, 0), num_bins - 1)
        
        return bin_index
    
    def _validate_state_dict(self, state_dict: dict) -> None:
        """
        Validate that state_dict keys match expected state variables.
        Also validates each value is numeric. Values outside [-1, 1] are clamped.
        
        Args:
            state_dict: Dictionary containing state values
            
        Raises:
            ValueError: If keys don't match expected state variables or values are non-numeric
        """
        state_dict_keys = set(state_dict.keys())
        expected_keys = set(self.state)
        
        if state_dict_keys != expected_keys:
            missing = expected_keys - state_dict_keys
            extra = state_dict_keys - expected_keys
            error_msg = "State dictionary keys do not match expected state variables."
            if missing:
                error_msg += f" Missing keys: {missing}."
            if extra:
                error_msg += f" Extra keys: {extra}."
            raise ValueError(error_msg)

        for key in self.state:
            value = state_dict[key]
            if not isinstance(value, (int, float)):
                raise ValueError(f"State value for '{key}' must be numeric, got {type(value).__name__}.")
            if value < -1.0 or value > 1.0:
                print(
                    f"Warning: State value for '{key}' out of range: {value}. Clamping to [-1, 1]."
                )
                state_dict[key] = max(min(value, 1.0), -1.0)
    
    def _discretize_state(self, state: dict) -> tuple:
        """
        Convert normalized continuous state into discrete bins based on configuration.
        
        This is necessary because Q-tables can't handle infinite continuous states.
        We group similar values into "bins" to create a finite state space.
        
        Args:
            state: Dictionary with NORMALIZED values (-1.0 to 1.0 range)
                   Should contain keys matching self.state variables
        
        Returns:
            Tuple representing the discretized state.
            Order matches the order of self.state list.
            Returns None if state is incomplete.
        """
        discretized = []
        
        # Process each state variable in the order defined by self.state
        for state_var in self.state:
            
            value = state[state_var]
            
            # Get number of bins for this state variable
            num_bins = self.bins_config[state_var]
            
            # Convert from [-1, 1] range to [0, 1] range for binning
            normalized_value = (value + 1) / 2
            
            # Calculate which bin this value falls into
            bin_index = self._calculate_bin_index(normalized_value, num_bins)
            
            discretized.append(bin_index)
        
        # Return as a tuple - this becomes our "state" identifier
        return tuple(discretized)
    
    def _choose_action(self, state):
        """
        Choose an action using epsilon-greedy strategy.
        
        With probability epsilon: explore (pick random action)
        With probability (1-epsilon): exploit (pick best known action)
        
        Args:
            state: Discretized state tuple, e.g., (3, 4, 2, 1, 0, 1)
            
        Returns:
            Action string: "UP", "DOWN", or "STAY"
        """
        # Generate random number between 0 and 1
        if random.random() < self.epsilon:
            # EXPLORE: Pick a random action
            action = random.choice(self.actions)
            self._exploration_count += 1
        else:
            # EXPLOIT: Pick the action with the highest Q-value in this state
            q_values = [self.q_table.get((state, a), 0.0) for a in self.actions]
            max_q = max(q_values)
            # If multiple actions have same max value, pick one randomly
            best_actions = [self.actions[i] for i in range(len(self.actions)) 
                           if q_values[i] == max_q]
            action = random.choice(best_actions)
            self._exploitation_count += 1
        
        return action
    
    def process_state(self, state_dict: dict[str, float]) -> str:
        """
        Main method: receive game state and decide what action to take.
        
        This is the method that gets called every frame by the server.
        Also remembers this state and action internally for learning later.
        
        Args:
            state_dict: Dictionary from JSON containing raw game state
                       Example: {"paddleB_y": 300, "ball_x": 450, ...}
            
        Returns:
            Action string to take: "UP", "DOWN", or "STAY"
        """
        # Validate state_dict keys match expected state variables
        self._validate_state_dict(state_dict)
        
        # Step 1: Convert continuous state to discrete bins
        discrete_state = self._discretize_state(state_dict)
        
        # Step 2: Choose an action based on that state
        action = self._choose_action(discrete_state)
        
        # Step 3: Remember this state and action for learning when we get feedback
        self._last_state = discrete_state
        self._last_action = action
        
        # Step 4: Print for debugging (so we can see what it's doing)
        #print(f"State: {discrete_state} -> Action: {action}")
        
        return action
    
    def _update_q_value(self, prev_state: tuple, action: str, reward: float, 
                       state: tuple, done: bool) -> None:
        """
        Update a single Q-value using the Bellman equation.
        
        This is called after the agent takes an action and receives feedback.
        It's the core learning algorithm.
        
        Args:
            prev_state: Previous discretized state (tuple of bin indices)
            action: The action taken at prev_state (string)
            reward: The reward received from the environment (float)
            state: The new discretized state after the action (tuple)
            done: Whether the episode ended (bool)
        """
        # Get the current Q-value for this state-action pair
        # If we haven't seen this pair before, default to 0.0
        current_q = self.q_table.get((prev_state, action), 0.0)
        
        # If the episode is over, there's no future value to consider
        if done:
            max_next_q = 0.0
        else:
            # Otherwise, find the best Q-value for any action in the next state
            next_q_values = [self.q_table.get((state, a), 0.0) for a in self.actions]
            max_next_q = max(next_q_values)
        
        # Apply the Bellman equation to update the Q-value
        # New estimate = old estimate + learning_rate * (immediate_reward + future_value - old_estimate)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Store the updated Q-value
        self.q_table[(prev_state, action)] = new_q
        
        # Track learning progress
        self._updates_count += 1
        if new_q > self._max_q_value:
            self._max_q_value = new_q
    
    def update(self, reward: float, state: dict[str, float], done: bool) -> None:
        """
        Simpler learning method that uses internally tracked state/action.
        
        Call this every frame with feedback from the previous action.
        The agent remembers its last state/action from the previous process_state() call.
        
        Args:
            reward: The reward received from the previous action (float)
            state: Raw game state (the result of where the previous action led)
            done: Whether the episode ended (bool)
        """
        # If this is the first frame, we have no previous action to learn from
        if self._last_state is None or self._last_action is None:
            return
        
        # Convert next state from continuous to discrete bins
        state = self._discretize_state(state)
        
        # Update Q-value using the action we took last frame
        self._update_q_value(self._last_state, self._last_action, reward, state, done)
    
    def save(self, filepath: str) -> None:
        """
        Save learned Q-table to file for later use.
        
        Args:
            filepath: Path where to save the Q-table (JSON format)
        """
        # Convert Q-table keys (tuples) to strings for JSON serialization
        q_table_str_keys = {str(k): v for k, v in self.q_table.items()}

        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(q_table_str_keys, f, indent=2)
        
        print(f"Q-table saved to {save_path}")
    
    def load(self, filepath: str) -> None:
        """
        Load a previously saved Q-table from file.
        
        Args:
            filepath: Path to the saved Q-table file (JSON format)
        """
        with open(filepath, 'r') as f:
            q_table_str_keys = json.load(f)
        
        # Convert string keys back to tuples
        self.q_table = {}
        for str_key, q_value in q_table_str_keys.items():
            parsed_key = ast.literal_eval(str_key)
            self.q_table[parsed_key] = q_value
        
        print(f"Q-table loaded from {filepath}")
    
    def get_stats(self) -> dict:
        """
        Get learning statistics about the current Q-table.
        
        Useful for monitoring training progress.
        
        Returns:
            Dictionary containing:
            - num_entries: Number of (state, action) pairs learned
            - avg_q: Average Q-value
            - max_q: Highest Q-value
            - min_q: Lowest Q-value
            - updates: Total number of Q-value updates
            - exploration_rate: Percentage of actions that were explorations
        """
        if not self.q_table:
            stats = {
                "num_entries": 0,
                "avg_q": 0.0,
                "max_q": 0.0,
                "min_q": 0.0
            }
        else:
            q_values = list(self.q_table.values())
            stats = {
                "num_entries": len(self.q_table),
                "avg_q": sum(q_values) / len(q_values),
                "max_q": max(q_values),
                "min_q": min(q_values)
            }
        
        # Add learning progress stats
        total_actions = self._exploration_count + self._exploitation_count
        exploration_pct = (self._exploration_count / total_actions * 100) if total_actions > 0 else 0.0
        
        stats["updates"] = self._updates_count
        stats["exploration_rate"] = exploration_pct
        
        return stats
