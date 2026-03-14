# Pong Game with RL Trained AI

A Pong game that serves as an entrypoint for reinforcement learning. It employs multiple RL models so users can get a feel for how they behave, the tuning they require, and the challenges that come with training them.

---

## Philosophy

A **coach** agent first learns to bounce a ball against a wall — a simpler, isolated task. Once trained, a **student** agent is then trained by playing against the coach.

---

## Getting Started

| Platform | Command |
|----------|---------|
| Linux    | `./start.sh` |
| Windows  | `./start.ps1` |

These scripts launch both `python_files/main.py` and the Godot game together.

---

## Main Menu

The user starts at the main menu and can navigate to:

### Local Play
Play locally against another human or against your own trained AI models.

> ⚠️ Character selection screen — to be completed.

### Host / Join *(Online Multiplayer)*

> ⚠️ To be determined.

### AI Training
Space where the user can configure and run model training. During training a JSON file with the model is continuously saved to `python_files/models/`. It can then be exported to Godot using the appropriate script inside `python_files/scripts/`. Godot will automatically assign the exported model to the correct character.

---

## Configuration

Training is configured via YAML files inside `python_files/config/`. One file exists for each combination of method and mode:

| Method | Coach config | Student config |
|---|---|---|
| `qvalue` | `QAgent_coach.yaml` | `QAgent_student.yaml` |
| `policy_gradient` | `PolicyGradient_coach.yaml` | `PolicyGradient_student.yaml` |
| `policy_gradient_dnn` | `PolicyGradientDNN_coach.yaml` | `PolicyGradientDNN_student.yaml` |
| `a2c` | `A2C_coach.yaml` | `A2C_student.yaml` |
| `ppo` | `PPO_coach.yaml` | `PPO_student.yaml` |

Each YAML file contains sections for model settings, hyperparameters, and logging.

---

## Headless Mode *(Advanced)*

Godot supports headless mode, which is useful for training multiple parallel workers and reducing overhead. To enable or disable it, edit the appropriate launch script (`start.sh` or `start.ps1`) and add/remove the `--headless` flag:

```bash
# Example lines in start.sh
$godotExecutable --headless --path "godot_files/" --import
godot --headless --path godot_files/ -- --port $PORT &
```

When running headless the GUI is unavailable, so training must be configured manually:

**1. Set the training scene as the main scene in Godot:**
- Coach training → `coach_training.tscn`
- Student training → `ai_training.tscn`

**2. In `python_files/main.py`, overwrite the `training_method` and `training_mode` variables inside `main()`:**

```python
def main():
    training_method, training_mode = receive_handshake()
    training_method = "policy_gradient_dnn"  # override for headless
    training_mode   = "coach"                # override for headless
    agent, config = load_agent(training_method, training_mode)
```

**Available `training_method` values:**
`qvalue` · `policy_gradient` · `policy_gradient_dnn` · `a2c` · `ppo`

**Available `training_mode` values:**
`coach` · `vs_static` · `vs_homing` · `vs_coach`
