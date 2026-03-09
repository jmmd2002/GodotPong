"""
Export a trained Q-table to the float-key format expected by Godot.

Edit the constants below, then run:
    python export_table.py
"""

import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from qlearning_agent import QLearningAgent

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / "config" / "QAgent_coach.yaml"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "godot_files" / "models" / "q_table_coach.json"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    agent: QLearningAgent = QLearningAgent.from_dict(config)

    model_path = Path(__file__).parent.parent / config["model"]["path"]
    agent.load(str(model_path))

    agent.export_for_godot(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
