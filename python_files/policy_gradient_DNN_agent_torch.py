import json
import os
import random
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_agent import RLAgent


class _MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out)
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return F.softmax(self.layers[-1](x), dim=-1)


class PolicyGradientDNNAgent(RLAgent):

    def __init__(self, state_vars, actions, alpha, gamma, hidden_sizes):
        self.state_vars   = state_vars
        self.actions      = actions
        self.alpha        = alpha
        self.gamma        = gamma
        self.hidden_sizes = hidden_sizes

        self._validate_state()
        self._validate_actions()
        self._validate_hyperparameters()

        self.state_dim   = len(self.state_vars)
        self.num_actions = len(self.actions)

        layer_sizes = [self.state_dim] + self.hidden_sizes + [self.num_actions]
        self.net = _MLP(layer_sizes)
        self._init_weights(layer_sizes)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.alpha)

        self._local = threading.local()
        self._lock  = threading.Lock()

        self._episodes_completed  = 0
        self._updates_count       = 0
        self._last_loss           = 0.0
        self._last_episode_return = 0.0
        self._last_episode_length = 0
        self._last_grad_norm      = 0.0
        self._last_entropy        = 0.0

    def _init_weights(self, layer_sizes):
        num_layers = len(layer_sizes) - 1
        for i, layer in enumerate(self.net.layers):
            n_in = layer_sizes[i]
            std  = float(np.sqrt(1.0 / n_in) if i == num_layers - 1 else np.sqrt(2.0 / n_in))
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            nn.init.zeros_(layer.bias)

    @classmethod
    def from_dict(cls, config_dict):
        hp = config_dict.get('hyperparameters')
        if hp is None:
            raise ValueError("Config is missing the 'hyperparameters' block.")
        return cls(
            state_vars   = config_dict.get('state'),
            actions      = config_dict.get('actions'),
            alpha        = hp.get('alpha'),
            gamma        = hp.get('gamma'),
            hidden_sizes = hp.get('hidden_sizes'),
        )

    def _validate_state(self):
        if not isinstance(self.state_vars, list):
            raise ValueError("state_vars must be a list.")
        if not self.state_vars:
            raise ValueError("state_vars must not be empty.")
        if not all(isinstance(v, str) for v in self.state_vars):
            raise ValueError("All entries in state_vars must be strings.")

    def _validate_actions(self):
        if not isinstance(self.actions, list):
            raise ValueError("actions must be a list.")
        if not self.actions:
            raise ValueError("actions must not be empty.")
        if not all(isinstance(a, str) for a in self.actions):
            raise ValueError("All entries in actions must be strings.")

    def _validate_hyperparameters(self):
        if not isinstance(self.alpha, (int, float)) or not (self.alpha > 0):
            raise ValueError(f"alpha={self.alpha!r} is invalid. Must be a strictly positive number.")
        if not isinstance(self.gamma, (int, float)) or not (0 <= self.gamma <= 1):
            raise ValueError(f"gamma={self.gamma!r} is invalid. Must be a number in the range [0, 1].")
        if not isinstance(self.hidden_sizes, list):
            raise ValueError(f"hidden_sizes must be a list, got {type(self.hidden_sizes).__name__}.")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty.")
        converted = []
        for i, size in enumerate(self.hidden_sizes):
            try:
                size_int = int(size)
                if size_int != size:
                    raise ValueError()
            except (ValueError, TypeError):
                raise ValueError(f"hidden_sizes[{i}]={size!r} cannot be converted to an integer.")
            if size_int <= 0:
                raise ValueError(f"hidden_sizes[{i}]={size!r} is invalid.")
            converted.append(size_int)
        self.hidden_sizes = converted

    def _validate_state_dict(self, state_dict):
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
                raise ValueError(f"State value for '{key}' must be numeric, got {type(val).__name__}.")
            if val < -1.0 or val > 1.0:
                print(f"Warning: state value for '{key}' = {val:.4f} is outside [-1, 1]. Clamping.")

    def _state_to_tensor(self, state_dict):
        return torch.tensor(
            [float(np.clip(state_dict[v], -1.0, 1.0)) for v in self.state_vars],
            dtype=torch.float32,
        )

    @property
    def _pending(self):
        return getattr(self._local, 'pending', None)

    @_pending.setter
    def _pending(self, value):
        self._local.pending = value

    @property
    def _trajectory(self):
        if not hasattr(self._local, 'trajectory'):
            self._local.trajectory = []
        return self._local.trajectory

    @_trajectory.setter
    def _trajectory(self, value):
        self._local.trajectory = value

    def process_state(self, state_dict):
        self._validate_state_dict(state_dict)
        s = self._state_to_tensor(state_dict)
        with torch.no_grad():
            pi = self.net(s).numpy()
        action_idx = int(np.random.choice(self.num_actions, p=pi))
        self._pending = (s, action_idx)
        return self.actions[action_idx]

    def _compute_returns(self, rewards):
        T       = len(rewards)
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
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, state, reward, done):
        if self._pending is None:
            return

        s, action_id = self._pending
        self._trajectory.append((s, action_id, reward))
        self._pending = None

        if not done:
            return

        states     = [entry[0] for entry in self._trajectory]
        action_ids = [entry[1] for entry in self._trajectory]
        rewards    = [entry[2] for entry in self._trajectory]
        self._trajectory = []

        states_t     = torch.stack(states)
        action_ids_t = torch.tensor(action_ids, dtype=torch.long)
        returns_t    = self._compute_returns(rewards)

        self.optimizer.zero_grad()
        all_pi    = self.net(states_t)
        log_probs = torch.log(all_pi[torch.arange(len(action_ids_t)), action_ids_t] + 1e-8)
        loss      = -torch.sum(returns_t * log_probs)
        loss.backward()

        grad_norm = float(torch.sqrt(sum(
            p.grad.pow(2).sum()
            for p in self.net.parameters()
            if p.grad is not None
        )))

        self.optimizer.step()

        with self._lock:
            self._episodes_completed  += 1
            self._updates_count       += len(rewards)
            self._last_loss            = float(loss.item())
            self._last_episode_return  = float(sum(rewards))
            self._last_episode_length  = len(rewards)
            self._last_grad_norm       = grad_norm
            with torch.no_grad():
                s_zero  = torch.zeros(self.state_dim)
                pi_zero = self.net(s_zero).numpy()
            self._last_entropy = float(-np.sum(pi_zero * np.log(np.clip(pi_zero, 1e-8, 1.0))))

    @property
    def last_loss(self):
        return self._last_loss

    @property
    def last_entropy(self):
        return self._last_entropy

    def _params_as_wb_dict(self):
        return {
            key: layer.detach().numpy().tolist()
            for i, lin in enumerate(self.net.layers)
            for key, layer in [(f"W{i}", lin.weight), (f"b{i}", lin.bias)]
        }

    def save(self, filepath):
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            params_list = self._params_as_wb_dict()

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

        num_params  = sum(p.numel() for p in self.net.parameters())
        layer_sizes = [self.state_dim] + self.hidden_sizes + [self.num_actions]
        print(f"Policy saved to {save_path}  ({num_params} parameters, layers: {layer_sizes})")

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)

        layer_sizes = [self.state_dim] + self.hidden_sizes + [self.num_actions]
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            loaded_W = np.array(data["params"][f"W{i}"])
            loaded_b = np.array(data["params"][f"b{i}"])
            if loaded_W.shape != (n_out, n_in):
                raise ValueError(f"Loaded W{i} has shape {loaded_W.shape}, expected {(n_out, n_in)}.")
            if loaded_b.shape != (n_out,):
                raise ValueError(f"Loaded b{i} has shape {loaded_b.shape}, expected {(n_out,)}.")

        with self._lock:
            for i, lin in enumerate(self.net.layers):
                lin.weight.data = torch.tensor(data["params"][f"W{i}"], dtype=torch.float32)
                lin.bias.data   = torch.tensor(data["params"][f"b{i}"], dtype=torch.float32)

        num_params = sum(p.numel() for p in self.net.parameters())
        print(f"Policy loaded from {filepath}  ({num_params} parameters, layers: {layer_sizes})")

    def export_for_godot(self, output_path):
        with self._lock:
            params_list = self._params_as_wb_dict()

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
        print(f"Exported policy (Godot format) to {out}  (layers: {layer_sizes})")

    def get_stats(self):
        with self._lock:
            weight_arrays     = [lin.weight.detach().numpy().ravel() for lin in self.net.layers]
            per_layer_means   = [float(np.abs(lin.weight.detach().numpy()).mean()) for lin in self.net.layers]
            episodes_completed  = self._episodes_completed
            updates_count       = self._updates_count
            last_loss           = self._last_loss
            last_episode_return = self._last_episode_return
            last_episode_length = self._last_episode_length
            last_grad_norm      = self._last_grad_norm
            last_entropy        = self._last_entropy

        all_weights = np.concatenate(weight_arrays)
        avg_w = float(np.abs(all_weights).mean())
        max_w = float(np.abs(all_weights).max())
        std_w = float(all_weights.std())

        return {
            "episodes":       episodes_completed,
            "updates":        updates_count,
            "last_loss":      round(last_loss, 6),
            "episode_return": round(last_episode_return, 6),
            "episode_length": last_episode_length,
            "avg_w":          round(avg_w, 6),
            "max_w":          round(max_w, 6),
            "std_w":          round(std_w, 6),
            "avg_entropy":    round(last_entropy, 6),
            "grad_norm":      round(last_grad_norm, 6),
            **{f"avg_w_{i}": round(m, 6) for i, m in enumerate(per_layer_means)},
        }

    def print_config(self):
        layer_sizes = [self.state_dim] + self.hidden_sizes + [self.num_actions]
        arrows      = " → ".join(str(n) for n in layer_sizes)
        num_params  = sum(p.numel() for p in self.net.parameters())
        print("Algorithm : REINFORCE (Policy Gradient) — DNN version (PyTorch)")
        print(f"  State variables : {self.state_vars}")
        print(f"  Actions         : {self.actions}")
        print(f"  Architecture    : {arrows}  ({num_params} parameters)")
        print(f"  Hidden sizes    : {self.hidden_sizes}")
        print(f"  Alpha (lr)      : {self.alpha}")
        print(f"  Gamma (discount): {self.gamma}")
