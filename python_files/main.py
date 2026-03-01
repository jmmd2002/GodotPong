import socket
import json
import yaml
from pathlib import Path
from qlearning_agent import QLearningAgent


# Config files
QAGENT_CONFIG_PATH = Path(__file__).parent / "config" / "QAgent_config.yaml"

HOST = "127.0.0.1"
PORT = 5000


def load_agent() -> QLearningAgent:
    """Load Q-learning agent from config file."""
    config_path = QAGENT_CONFIG_PATH
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {config_path}")
        print(f"Error: {e}")
        raise

    agent = QLearningAgent.from_dict(config)
    return agent


def main():
    """Start the server and handle incoming connections."""
    # Load the Q-learning agent
    agent = load_agent()
    
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

            while "\n" in buffer: # split messages on new lines - ensure program advances only with valid message
                line, buffer = buffer.split("\n", 1)
                
                try:
                    # Parse incoming state
                    state_dict: dict[str, float] = json.loads(line)
                    #print(f"Received state: {state_dict}")

                    frame_id = state_dict.get("frame_id")
                    if frame_id is None:
                        print("State validation error: missing 'frame_id'")
                        continue

                    agent_state = dict(state_dict)
                    agent_state.pop("frame_id", None)
                    
                    # Get action from agent
                    action = agent.process_state(agent_state)
                    
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
        conn.close()
        server.close()
        print("Server closed.")


if __name__ == "__main__":
    main()