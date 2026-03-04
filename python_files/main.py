import signal
import socket
import json
import yaml
import threading
from pathlib import Path
from qlearning_agent import QLearningAgent


# Config files
QAGENT_CONFIG_PATH = Path(__file__).parent / "config" / "QAgent_config.yaml"

HOST = "127.0.0.1"
BASE_PORT = 5000       # First worker listens here
NUM_WORKERS = 1        # Set to e.g. 3 to run 3 Godot instances in parallel


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


def print_training_progress(worker_id: int, learning_steps: int, total_games: int,
                            agent: QLearningAgent, episode_rewards: list) -> None:
    """Print training progress and learning statistics."""
    stats = agent.get_stats()
    avg_reward = sum(episode_rewards[-20:]) / len(episode_rewards[-20:]) if episode_rewards else 0.0
    
    print(f"[W{worker_id}|Step {learning_steps:7d}] Games: {total_games:4d} | "
          f"Q-States: {stats['num_entries']:5d} ({stats['updates']:6d} updates) | "
          f"Q-Values: avg={stats['avg_q']:7.3f} max={stats['max_q']:7.3f} | "
          f"Explore: {stats['exploration_rate']:5.1f}% | "
          f"Reward: {avg_reward:7.2f}")


def print_config(agent: QLearningAgent, num_workers: int, autosave_every_n_steps: int) -> None:
    """Print learning configuration at startup."""
    print(f"\n=== Learning Configuration ===")
    print(f"Workers (parallel Godot instances): {num_workers}")
    print(f"Ports: {BASE_PORT} to {BASE_PORT + num_workers - 1}")
    print(f"Learning Rate (alpha): {agent.alpha}")
    print(f"Discount Factor (gamma): {agent.gamma}")
    print(f"Exploration Rate (epsilon): {agent.epsilon}")
    print(f"Action space: {agent.actions}")
    print(f"State variables: {agent.state}")
    print(f"Autosave every {autosave_every_n_steps} steps (counted per worker)")
    print(f"=== Starting Training ===\n")


def worker(worker_id: int, agent: QLearningAgent, model_path: Path,
           autosave_every_n_steps: int, save_lock: threading.Lock,
           shutdown_event: threading.Event) -> None:
    """
    Handles one Godot connection on its own port.
    
    Each worker runs in its own thread. It has its own:
      - TCP server socket on BASE_PORT + worker_id
      - learning_steps / total_games / episode_rewards counters
      - current_episode_reward tracker
    
    It shares with other workers:
      - agent (Q-table is protected by agent._lock)
      - model_path / save_lock (so two workers don't save simultaneously)
      - shutdown_event (Ctrl+C from any thread signals all workers to stop)
    """
    port = BASE_PORT + worker_id
    
    # Per-worker tracking (no sharing needed — each thread owns these)
    learning_steps: int = 0
    total_games: int = 0
    current_episode_reward: float = 0.0
    episode_rewards: list = []
    debug_print_interval: int = 100

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    server.settimeout(1.0)  # allows checking shutdown_event periodically
    server.bind((HOST, port))
    server.listen(1)
    print(f"[W{worker_id}] Listening on port {port}...")

    conn = None
    try:
        # Wait for a Godot instance to connect (check shutdown_event every second)
        while not shutdown_event.is_set():
            try:
                conn, addr = server.accept()
                break
            except socket.timeout:
                continue
        
        if shutdown_event.is_set():
            return

        print(f"[W{worker_id}] Connected by {addr}")
        conn.settimeout(1.0)
        buffer = ""

        while not shutdown_event.is_set():
            try:
                data = conn.recv(1024)
            except socket.timeout:
                continue  # no data yet, loop and check shutdown_event

            if not data:
                print(f"[W{worker_id}] Godot disconnected.")
                break

            buffer += data.decode()

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)

                try:
                    message: dict = json.loads(line)

                    frame_id = message.get("frame_id")
                    if frame_id is None:
                        print(f"[W{worker_id}] Missing frame_id")
                        continue

                    prev_reward = message.get("prev_reward", 0.0)
                    done = message.get("done", False)
                    current_state = {k: v for k, v in message.items()
                                     if k not in ["frame_id", "prev_reward", "done"]}

                    # LEARNING: update Q-values (thread-safe inside agent)
                    agent.update(prev_reward, current_state, done)
                    learning_steps += 1
                    current_episode_reward += prev_reward

                    if done:
                        total_games += 1
                        episode_rewards.append(current_episode_reward)
                        current_episode_reward = 0.0

                    if learning_steps % debug_print_interval == 0:
                        print_training_progress(worker_id, learning_steps, total_games,
                                                agent, episode_rewards)

                    # Autosave: use save_lock so two workers don't write simultaneously
                    if autosave_every_n_steps > 0 and learning_steps % autosave_every_n_steps == 0:
                        with save_lock:
                            try:
                                agent.save(str(model_path))
                                print(f"[W{worker_id}|Step {learning_steps}] ✓ Model autosaved")
                            except Exception as e:
                                print(f"[W{worker_id}] Autosave failed: {e}")

                    # DECISION: pick next action and send back to Godot
                    action = agent.process_state(current_state)
                    response = json.dumps({"frame_id": frame_id, "action": action}) + "\n"
                    conn.sendall(response.encode())

                except json.JSONDecodeError as e:
                    print(f"[W{worker_id}] JSON error: {e}")
                except UnicodeDecodeError as e:
                    print(f"[W{worker_id}] Decode error: {e}")
                except ValueError as e:
                    print(f"[W{worker_id}] State validation error: {e}")

    finally:
        if conn:
            conn.close()
        server.close()
        print(f"[W{worker_id}] Shut down.")


def main():
    """Start all workers and wait for Ctrl+C."""
    agent, config = load_agent()

    model_config: dict = config.get("model", {})
    model_path = Path(__file__).parent / model_config.get("path", "models/q_table.json")
    load_on_start: bool = model_config.get("load_on_start", True)
    save_on_exit: bool = model_config.get("save_on_exit", True)
    autosave_every_n_steps: int = int(model_config.get("autosave_every_n_steps", 1000))
    num_workers: int = int(model_config.get("num_workers", NUM_WORKERS))

    if load_on_start and model_path.exists():
        try:
            agent.load(str(model_path))
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    elif load_on_start:
        print(f"No existing model at {model_path}. Starting fresh.")

    print_config(agent, num_workers, autosave_every_n_steps)

    # shutdown_event: when set, all worker threads will exit cleanly
    shutdown_event = threading.Event()
    # save_lock: prevents two workers from writing the model file at the same time
    save_lock = threading.Lock()

    threads = []
    for i in range(num_workers):
        t = threading.Thread(
            target=worker,
            args=(i, agent, model_path, autosave_every_n_steps, save_lock, shutdown_event),
            daemon=True,
            name=f"worker-{i}"
        )
        t.start()
        threads.append(t)

    try:
        # Main thread just waits; workers do all the work
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nShutting down all workers...")
        shutdown_event.set()
        for t in threads:
            t.join()
    finally:
        if save_on_exit:
            with save_lock:
                try:
                    agent.save(str(model_path))
                    print("Final model saved.")
                except Exception as e:
                    print(f"Failed to save model: {e}")
        print("Done.")


if __name__ == "__main__":
    main()