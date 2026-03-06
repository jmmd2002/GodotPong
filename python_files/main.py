import signal
import socket
import json
import time
import yaml
import threading
from pathlib import Path
from qlearning_agent import QLearningAgent
from stats_logger import QLearningStatsLogger
from live_plotter import QLearningLivePlotter


# Config files
QAGENT_CONFIG_PATH = Path(__file__).parent / "config" / "QAgent_config.yaml"
COACH_CONFIG_PATH = Path(__file__).parent / "config" / "Coach_config.yaml"

HOST = "127.0.0.1"
BASE_PORT = 5000       # First worker listens here
NUM_WORKERS = 1        # Set to e.g. 3 to run 3 Godot instances in parallel

# Default model config values (used when config file is missing or invalid)
DEFAULT_MODEL_PATH = "models/new_q_table.json"
DEFAULT_LOAD_ON_START = True
DEFAULT_SAVE_ON_EXIT = True
DEFAULT_AUTOSAVE_EVERY_N_STEPS = 1000
DEFAULT_NUM_WORKERS = 1
DEFAULT_LOG_EVERY_N_STEPS = 100   # 0 disables stats logging
DEFAULT_LOG_PATH = "logs/training_log.csv"
DEFAULT_REWARD_WINDOW = 20        # Episodes used to compute avg reward


def load_agent() -> tuple[QLearningAgent, dict]:
    """Load Q-learning agent and raw config from config file."""
    config_path = QAGENT_CONFIG_PATH #TODO: somehow make this change according to godot
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {config_path}")
        print(f"Error: {e}")
        raise

    agent = QLearningAgent.from_dict(config)
    return agent, config

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


def validate_model_config(config) -> dict:
    """Validate the model config section from the YAML config file.
    
    Each field is checked individually. If missing or the wrong type, a warning
    is printed and the default value is used. Extra fields are ignored.
    Returns a dict with only the validated fields.
    """
    if not isinstance(config, dict):
        print("[WARNING] 'model' section is missing or invalid in config file. Using all defaults.")
        config = {
            "path":                   DEFAULT_MODEL_PATH,
            "load_on_start":          DEFAULT_LOAD_ON_START,
            "save_on_exit":           DEFAULT_SAVE_ON_EXIT,
            "autosave_every_n_steps": DEFAULT_AUTOSAVE_EVERY_N_STEPS,
            "num_workers":            DEFAULT_NUM_WORKERS,
        }
        return config

    def get(key, expected_type, default):
        value = config.get(key)
        if not isinstance(value, expected_type):
            if value is not None:
                print(f"[WARNING] model.{key} must be {expected_type.__name__}, "
                      f"got {type(value).__name__}. Using default: {default!r}")
            else:
                print(f"[WARNING] model.{key} not set. Using default: {default!r}")
            return default
        return value

    path                   = get("path",                   str,  DEFAULT_MODEL_PATH)
    load_on_start          = get("load_on_start",          bool, DEFAULT_LOAD_ON_START)
    save_on_exit           = get("save_on_exit",           bool, DEFAULT_SAVE_ON_EXIT)
    autosave_every_n_steps = get("autosave_every_n_steps", int,  DEFAULT_AUTOSAVE_EVERY_N_STEPS)
    num_workers            = get("num_workers",            int,  DEFAULT_NUM_WORKERS)

    if autosave_every_n_steps < 0:
        print(f"[WARNING] model.autosave_every_n_steps must be >= 0. Using default: {DEFAULT_AUTOSAVE_EVERY_N_STEPS}")
        autosave_every_n_steps = DEFAULT_AUTOSAVE_EVERY_N_STEPS
    if num_workers < 1:
        print(f"[WARNING] model.num_workers must be >= 1. Using default: {DEFAULT_NUM_WORKERS}")
        num_workers = DEFAULT_NUM_WORKERS

    return {
        "path":                   path,
        "load_on_start":          load_on_start,
        "save_on_exit":           save_on_exit,
        "autosave_every_n_steps": autosave_every_n_steps,
        "num_workers":            num_workers,
    }


def validate_log_config(config) -> dict:
    """Validate the logs config section from the YAML config file.

    Each field is checked individually. If missing or the wrong type, a warning
    is printed and the default value is used. Extra fields are ignored.
    Returns a dict with only the validated fields.
    """
    if not isinstance(config, dict):
        print("[WARNING] 'logs' section is missing or invalid in config file. Using all defaults.")
        config = {}

    def get(key, expected_type, default):
        value = config.get(key)
        if not isinstance(value, expected_type):
            if value is not None:
                print(f"[WARNING] logs.{key} must be {expected_type.__name__}, "
                      f"got {type(value).__name__}. Using default: {default!r}")
            else:
                print(f"[WARNING] logs.{key} not set. Using default: {default!r}")
            return default
        return value

    log_every_n_steps = get("log_every_n_steps", int, DEFAULT_LOG_EVERY_N_STEPS)
    log_path          = get("log_path",          str, DEFAULT_LOG_PATH)
    reward_window     = get("reward_window",     int, DEFAULT_REWARD_WINDOW)

    if log_every_n_steps < 0:
        print(f"[WARNING] logs.log_every_n_steps must be >= 0. Using default: {DEFAULT_LOG_EVERY_N_STEPS}")
        log_every_n_steps = DEFAULT_LOG_EVERY_N_STEPS
    if reward_window < 1:
        print(f"[WARNING] logs.reward_window must be >= 1. Using default: {DEFAULT_REWARD_WINDOW}")
        reward_window = DEFAULT_REWARD_WINDOW

    return {
        "log_every_n_steps": log_every_n_steps,
        "log_path":          log_path,
        "reward_window":     reward_window,
    }


def worker(worker_id: int, agent: QLearningAgent, model_path: Path,
           autosave_every_n_steps: int, save_lock: threading.Lock,
           shutdown_event: threading.Event,
           stats_logger: QLearningStatsLogger | None = None,
           log_every_n_steps: int = 0,
           reward_window: int = 20) -> None:
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
    episode_rewards: list[float] = []

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
        connection_start_time = time.time()
        while not shutdown_event.is_set():
            not_connected_timer = time.time() - connection_start_time
            try:
                conn, addr = server.accept()
                break
            except socket.timeout:
                if int(not_connected_timer) % 5 == 0:  # print every 5 seconds while waiting
                    print(f"[W{worker_id}] Waiting for Godot to connect on port {port}...")
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

            if not data: #standard way to detect closed connection
                print(f"[W{worker_id}] Godot disconnected.")
                break

            buffer += data.decode()

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1) #split off one line at a time

                try:
                    message: dict = json.loads(line)
                    
                    frame_id = message.get("frame_id")
                    prev_reward = message.get("prev_reward")
                    done = message.get("done")
                    current_state = {k: v for k, v in message.items()
                                     if k not in ["frame_id", "prev_reward", "done"]}
                    #verify_message(frame_id, current_state, prev_reward, done) #TODO #verify the message didnt get corrupted

                    # LEARNING: update Q-values (thread-safe inside agent)
                    agent.update(current_state, prev_reward, done)
                    learning_steps += 1

                    # Stats for logging: update episode reward and total games
                    if prev_reward is not None:
                        current_episode_reward += prev_reward
                    if done:
                        total_games += 1
                        episode_rewards.append(current_episode_reward)
                        if len(episode_rewards) > reward_window:
                            episode_rewards.pop(0)
                        current_episode_reward = 0.0

                    # Autosave: only worker 0 saves — all workers share the same Q-table
                    # so saving once is enough; no need for every worker to write the file
                    if worker_id == 0 and autosave_every_n_steps > 0 and learning_steps % autosave_every_n_steps == 0:
                        with save_lock:
                            try:
                                agent.save(str(model_path))
                                print(f"[W{worker_id}|Step {learning_steps}] Model autosaved")
                            except Exception as e:
                                print(f"[W{worker_id}] Autosave failed: {e}")

                    # Stats logging: all workers record episode rewards;
                    # only worker 0 supplies Q-table stats (others pass {})
                    if (stats_logger is not None
                            and log_every_n_steps > 0
                            and learning_steps % log_every_n_steps == 0):
                        avg_reward = (sum(episode_rewards) / len(episode_rewards)
                                      if episode_rewards else 0.0)
                        agent_stats = agent.get_stats() if worker_id == 0 else {}
                        stats_logger.record(worker_id, learning_steps, total_games, avg_reward, agent_stats)

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
                except Exception as e:
                    print(f"[W{worker_id}] Unknown error, skipping step: {e}")
    finally:
        if conn:
            conn.close()
        server.close()
        print(f"[W{worker_id}] Shut down.")


def main():
    """Start all workers and wait for Ctrl+C."""
    agent, config = load_agent()

    model_config: dict = validate_model_config(config.get("model"))
    model_path: Path = Path(__file__).parent / model_config.get("path")
    load_on_start: bool = model_config.get("load_on_start")
    save_on_exit: bool = model_config.get("save_on_exit")
    autosave_every_n_steps: int = int(model_config.get("autosave_every_n_steps"))
    num_workers: int = int(model_config.get("num_workers"))

    log_config: dict = validate_log_config(config.get("logs"))
    log_every_n_steps: int = log_config.get("log_every_n_steps")
    log_path: Path = Path(__file__).parent / log_config.get("log_path")
    reward_window: int = log_config.get("reward_window")

    if load_on_start and model_path.exists():
        try:
            agent.load(str(model_path))
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    elif load_on_start:
        print(f"No existing model at {model_path}. Starting fresh at {model_path}.")

    print_config(agent, num_workers, autosave_every_n_steps)

    # shutdown_event: when set, all worker threads will exit cleanly
    shutdown_event = threading.Event()
    # save_lock: prevents two workers from writing the model file at the same time
    save_lock = threading.Lock()

    # Stats logger (worker 0 records into it; main saves CSV on exit)
    stats_logger = QLearningStatsLogger()

    # Live plot thread (reads from stats_logger, redraws every 5s)
    plotter = QLearningLivePlotter(stats_logger, refresh_interval=5.0, reward_window=reward_window)
    plotter.start(shutdown_event)

    threads: list[threading.Thread] = []
    for i in range(num_workers):
        t = threading.Thread(
            target=worker,
            args=(i, agent, model_path, autosave_every_n_steps, save_lock, shutdown_event,
                  stats_logger,
                  log_every_n_steps,
                  reward_window),
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
        stats_logger.save_csv(log_path)
        print("Done.")


if __name__ == "__main__":
    main()