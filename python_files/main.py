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

    if load_on_start and model_path.exists():
        try:
            agent.load(str(model_path))
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
    elif load_on_start:
        print(f"No existing model found at {model_path}. Starting fresh.")
    
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
                    
                    # LEARNING: Update Q-values from previous action
                    # The agent remembers its last state/action, and current_state is where it ended up
                    agent.update(prev_reward, current_state, done)
                    learning_steps += 1

                    if autosave_every_n_steps > 0 and learning_steps % autosave_every_n_steps == 0:
                        try:
                            agent.save(str(model_path))
                        except Exception as e:
                            print(f"Autosave failed at step {learning_steps}: {e}")
                    
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