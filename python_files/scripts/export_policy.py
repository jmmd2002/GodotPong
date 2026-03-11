"""
Export a trained Policy Gradient model to the format expected by Godot.

Edit the constants below to select the agent, then run:
    python export_policy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from policy_gradient_agent import PolicyGradientAgent

import yaml

# ── Configuration ────────────────────────────────────────────────────────────
AGENT        = "coach"   # "coach" or "student"
CONFIG_PATH  = Path(__file__).parent.parent / "config" / f"PolicyGradient_{AGENT}.yaml"
OUTPUT_PATH  = Path(__file__).parent.parent.parent / "godot_files" / "models" / f"policy_{AGENT}.json"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    agent: PolicyGradientAgent = PolicyGradientAgent.from_dict(config)

    model_path = Path(__file__).parent.parent / config["model"]["path"]
    agent.load(str(model_path))

    agent.export_for_godot(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
