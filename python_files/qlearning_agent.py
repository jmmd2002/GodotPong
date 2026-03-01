"""
Q-Learning Agent for Pong Game

This agent learns to play Pong using Q-learning with value iteration.
"""

import random


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

    def __init__(self, state=None, bins_config=None, actions=None):
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
        
        # Validate configuration
        self._validate_state()
        self._validate_actions()
        self._validate_bins()
        
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
    
    @classmethod
    def from_dict(cls: 'QLearningAgent', config_dict: dict):
        """
        Create a QLearningAgent instance from a configuration dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
                        Expected keys: 'state', 'bins_config', 'actions'
        
        Returns:
            QLearningAgent instance configured with the provided parameters.
        """
        state = config_dict.get('state', None)
        bins_config = config_dict.get('bins_config', None)
        actions = config_dict.get('actions', None)
        
        return cls(state=state, bins_config=bins_config, actions=actions)
    
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
        Choose an action based on the current state.
        
        For now, this just chooses randomly (pure exploration).
        Later we'll add:
        - Q-table lookup to find best action (exploitation)
        - Epsilon-greedy strategy (balance explore vs exploit)
        
        Args:
            state: Discretized state tuple, e.g., (3, 4, 2, 2, 0)
            
        Returns:
            Action string: "UP", "DOWN", or "STAY"
        """
        # Right now: completely random action (100% exploration)
        action = random.choice(self.actions)
        return action
    
    def process_state(self, state_dict: dict[str, float]) -> str:
        """
        Main method: receive game state and decide what action to take.
        
        This is the method that gets called every frame by the server.
        
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
        
        # Step 3: Print for debugging (so we can see what it's doing)
        #print(f"State: {discrete_state} -> Action: {action}")
        
        return action
