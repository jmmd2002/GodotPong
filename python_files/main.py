import socket
import json
import yaml
from pathlib import Path
from qlearning_agent import QLearningAgent


# Config files
QAGENT_CONFIG_PATH = Path(__file__).parent / "config" / "QAgent_config.yaml"

HOST = "127.0.0.1"
PORT = 5000


def load_agent() -> tuple[QLearningAgent, dict]:
    """Load Q-learning agent and raw config from config file."""
    config_path = QAGENT_CONFIG_PATH
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {config_path}")
        print(f"Error: {e}")
        raise

    agent = QLearningAgent.from_dict(config)
    return agent, config


def handle_learning_step(agent: QLearningAgent, prev_reward: float, current_state: dict, 
                        done: bool, learning_steps: int, total_games: int,
                        current_episode_reward: float, episode_rewards: list,
                        model_path: Path, autosave_every_n_steps: int) -> tuple[int, int, float, float]:
    """
    Handle a single learning step and update tracking variables.
    
    Returns:
        (learning_steps, total_games, current_episode_reward, updated_step_count)
    """
    # LEARNING: Update Q-values from previous action
    agent.update(prev_reward, current_state, done)
    learning_steps += 1
    current_episode_reward += prev_reward
    
    # Track end of episode
    if done:
        total_games += 1
        episode_rewards.append(current_episode_reward)
        current_episode_reward = 0.0

    # Autosave model
    if autosave_every_n_steps > 0 and learning_steps % autosave_every_n_steps == 0:
        try:
            agent.save(str(model_path))
            print(f"[Step {learning_steps}] ✓ Model autosaved")
        except Exception as e:
            print(f"Autosave failed at step {learning_steps}: {e}")
    
    return learning_steps, total_games, current_episode_reward, learning_steps


def print_training_progress(learning_steps: int, total_games: int, agent: QLearningAgent,
                           episode_rewards: list) -> None:
    """Print training progress and learning statistics."""
    stats = agent.get_stats()
    avg_reward = sum(episode_rewards[-20:]) / len(episode_rewards[-20:]) if episode_rewards else 0.0
    q_table_size = stats["num_entries"]
    avg_q = stats["avg_q"]
    max_q = stats["max_q"]
    exploration_pct = stats["exploration_rate"]
    
    print(f"[Step {learning_steps:7d}] Games: {total_games:4d} | "
          f"Q-States: {q_table_size:5d} ({stats['updates']:6d} updates) | "
          f"Q-Values: avg={avg_q:7.3f} max={max_q:7.3f} | "
          f"Explore: {exploration_pct:5.1f}% | "
          f"Reward: {avg_reward:7.2f}")


def print_config(agent: QLearningAgent, autosave_every_n_steps: int) -> None:
    """Print learning configuration at startup."""
    print(f"\n=== Learning Configuration ===")
    print(f"Learning Rate (alpha): {agent.alpha}")
    print(f"Discount Factor (gamma): {agent.gamma}")
    print(f"Exploration Rate (epsilon): {agent.epsilon}")
    print(f"Action space: {agent.actions}")
    print(f"State variables: {agent.state}")
    print(f"Autosave every {autosave_every_n_steps} steps")
    print(f"=== Starting Training ===")
    print()


def main():
    """Start the server and handle incoming connections."""
    # Load the Q-learning agent
    agent, config = load_agent()

    model_config: dict = config.get("model", {})
    model_path = Path(__file__).parent / model_config.get("path", "models/q_table.json")
    load_on_start: bool = model_config.get("load_on_start", True)
    save_on_exit: bool = model_config.get("save_on_exit", True)
    autosave_every_n_steps: int = int(model_config.get("autosave_every_n_steps", 1000))
    learning_steps: int = 0
    
    # Learning tracking variables
    total_games: int = 0
    current_episode_reward: float = 0.0
    episode_rewards: list = []
    debug_print_interval: int = 100  # Print stats every N steps

    if load_on_start and model_path.exists():
        try:
            agent.load(str(model_path))
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
    elif load_on_start:
        print(f"No existing model found at {model_path}. Starting fresh.")
    
    print_config(agent, autosave_every_n_steps)
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    try:
        print("Waiting for connection...")
        conn, addr = server.accept()
        print(f"Connected by {addr}")

        buffer = ""
        
        while True:
            data = conn.recv(1024)
            if not data:
                break

            buffer += data.decode()

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                
                try:
                    # Parse incoming message from Godot
                    message: dict = json.loads(line)
                    
                    frame_id = message.get("frame_id")
                    if frame_id is None:
                        print("Message validation error: missing 'frame_id'")
                        continue

                    # Extract feedback from previous action
                    prev_reward = message.get("prev_reward", 0.0)
                    done = message.get("done", False)
                    
                    # Extract current state (which is the next state from previous action's perspective)
                    current_state = {k: v for k, v in message.items() 
                                    if k not in ["frame_id", "prev_reward", "done"]}
                    
                    # Handle learning step and tracking
                    learning_steps, total_games, current_episode_reward, _ = handle_learning_step(
                        agent, prev_reward, current_state, done, learning_steps, total_games,
                        current_episode_reward, episode_rewards, model_path, autosave_every_n_steps
                    )
                    
                    # Print learning progress periodically
                    if learning_steps % debug_print_interval == 0:
                        print_training_progress(learning_steps, total_games, agent, episode_rewards)
                    
                    # DECISION: Get next action from current state
                    action = agent.process_state(current_state)
                    
                    # Send action back to Godot
                    response = json.dumps({"frame_id": frame_id, "action": action}) + "\n"
                    conn.sendall(response.encode())
                    #print(f"Sent frame {frame_id} action: {action}\n")
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {line}")
                    print(f"Error: {e}")
                except ValueError as e:
                    print(f"State validation error: {e}")

    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally: #close server
        if save_on_exit:
            try:
                agent.save(str(model_path))
            except Exception as e:
                print(f"Failed to save model to {model_path}: {e}")
        if conn is not None:
            conn.close()
        server.close()
        print("Server closed.")


if __name__ == "__main__":
    main()