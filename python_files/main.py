import signal
import socket
import json
import time
import yaml
import threading
from pathlib import Path

from a2c_agent import A2CAgent
from ppo_agent import PPOAgent
from rl_agent import RLAgent
from qlearning_agent import QLearningAgent
from policy_gradient_agent import PolicyGradientAgent
from policy_gradient_DNN_agent import PolicyGradientDNNAgent
from stats_logger import StatsLogger, QLearningStatsLogger, PolicyGradientStatsLogger, PolicyGradientDNNStatsLogger, A2CStatsLogger, PPOStatsLogger
from live_plotter import Plotter, QLearningLivePlotter, PolicyGradientLivePlotter, PolicyGradientDNNLivePlotter, A2CLivePlotter, PPOLivePlotter


# Maps training_mode (from Godot handshake) to the config file to load
QVALUE_CONFIG_MAP: dict[str, Path] = {
    "vs_static": Path(__file__).parent / "config" / "QAgent_config.yaml",
    "vs_homing": Path(__file__).parent / "config" / "QAgent_config.yaml",
    "vs_coach":  Path(__file__).parent / "config" / "QAgent_config.yaml",
    "coach":     Path(__file__).parent / "config" / "QAgent_coach.yaml",
}
POLICY_GRADIENT_CONFIG_MAP: dict[str, Path] = {
    "vs_static": Path(__file__).parent / "config" / "PolicyGradient_student.yaml",
    "vs_homing": Path(__file__).parent / "config" / "PolicyGradient_student.yaml",
    "vs_coach":  Path(__file__).parent / "config" / "PolicyGradient_student.yaml",
    "coach":     Path(__file__).parent / "config" / "PolicyGradient_coach.yaml",
}
POLICY_GRADIENT_DNN_CONFIG_MAP: dict[str, Path] = {
    "vs_static": Path(__file__).parent / "config" / "PolicyGradientDNN_student.yaml",
    "vs_homing": Path(__file__).parent / "config" / "PolicyGradientDNN_student.yaml",
    "vs_coach":  Path(__file__).parent / "config" / "PolicyGradientDNN_student.yaml",
    "coach":     Path(__file__).parent / "config" / "PolicyGradientDNN_coach.yaml",
}

A2C_CONFIG_MAP: dict[str, Path] = {
    "vs_static": Path(__file__).parent / "config" / "A2C_student.yaml",
    "vs_homing": Path(__file__).parent / "config" / "A2C_student.yaml",
    "vs_coach":  Path(__file__).parent / "config" / "A2C_student.yaml",
    "coach":     Path(__file__).parent / "config" / "A2C_coach.yaml",
}

PPO_CONFIG_MAP: dict[str, Path] = {
    "vs_static": Path(__file__).parent / "config" / "PPO_student.yaml",
    "vs_homing": Path(__file__).parent / "config" / "PPO_student.yaml",
    "vs_coach":  Path(__file__).parent / "config" / "PPO_student.yaml",
    "coach":     Path(__file__).parent / "config" / "PPO_coach.yaml",
}

METHODS_MAP: dict[str, dict] = {
    "qvalue":               QVALUE_CONFIG_MAP,
    "policy_gradient":      POLICY_GRADIENT_CONFIG_MAP,
    "policy_gradient_dnn":  POLICY_GRADIENT_DNN_CONFIG_MAP,
    "a2c":                  A2C_CONFIG_MAP,
    "ppo":                  PPO_CONFIG_MAP,
}

AGENTS_MAP: dict[str, type] = {
    "qvalue":              QLearningAgent,
    "policy_gradient":     PolicyGradientAgent,
    "policy_gradient_dnn": PolicyGradientDNNAgent,
    "a2c":                 A2CAgent,
    "ppo":                 PPOAgent,
}
LOGGERS_MAP: dict[str, type] = {
    "qvalue":              QLearningStatsLogger,
    "policy_gradient":     PolicyGradientStatsLogger,
    "policy_gradient_dnn": PolicyGradientDNNStatsLogger,
    "a2c":                 A2CStatsLogger,
    "ppo":                 PPOStatsLogger,
}

PLOTTERS_MAP: dict[str, type] = {
    "qvalue":              QLearningLivePlotter,
    "policy_gradient":     PolicyGradientLivePlotter,
    "policy_gradient_dnn": PolicyGradientDNNLivePlotter,
    "a2c":                 A2CLivePlotter,
    "ppo":                 PPOLivePlotter,
}

HOST = "127.0.0.1"
BASE_PORT = 5000       # First worker listens here
NUM_WORKERS = 1        # Set to e.g. 3 to run 3 Godot instances in parallel
HEARTBEAT_TIMEOUT = 60.0  # seconds; warn if no message received from a worker


def receive_handshake() -> tuple[str, str]:
    """Listen on BASE_PORT for the first Godot connection, read the handshake JSON,
    then close. Returns the training_method and training_mode strings. Workers will re-accept on the same port."""
    print(f"Waiting for handshake on port {BASE_PORT}...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, 'SO_REUSEPORT'):
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    server.bind((HOST, BASE_PORT))
    server.listen(1)
    conn, addr = server.accept()
    print(f"Handshake connection from {addr}")
    try:
        buffer = ""
        while True:
            data = conn.recv(4096)
            if not data:
                raise ConnectionError("Connection closed before handshake.")
            buffer += data.decode()
            if "\n" in buffer:
                line, _ = buffer.split("\n", 1)
                msg: dict = json.loads(line.strip())
                if msg.get("type") == "handshake":
                    training_method = msg.get("training_method")
                    training_mode = msg.get("training_mode")
                    print(f"Handshake received: training_mode='{training_mode}', training_method='{training_method}'")
                    validate_handshake(training_method, training_mode)
                    return training_method, training_mode
                raise ValueError(f"Expected handshake, got: {msg}")
    finally:
        conn.close()
        server.close()

def validate_handshake(training_method: str, training_mode: str) -> None:
    """Validate the handshake message from Godot. Raises ValueError if invalid."""
    if training_method is None or training_mode is None:
        raise TypeError("Handshake corrupted. Could not find training mode or training method")
    if training_method not in METHODS_MAP:
        raise ValueError(f"Unknown training_method '{training_method}' in handshake.")
    if training_mode not in METHODS_MAP[training_method]:
        raise ValueError(f"Unknown training_mode '{training_mode}' for method '{training_method}' in handshake.")

def load_agent(training_method: str, training_mode: str) -> tuple[RLAgent, dict]:
    """Load Q-learning agent and raw config for the given training_mode."""
    
    config_path = METHODS_MAP.get(training_method).get(training_mode)
    if config_path is None:
        raise ValueError(f"Unknown training_mode '{training_mode}' for method '{training_method}'.")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {config_path}")
        print(f"Error: {e}")
        raise

    agent: RLAgent = AGENTS_MAP.get(training_method).from_dict(config)
    print(f"Agent initialized with config from {config_path}:")
    return agent, config

def print_config(agent: RLAgent, num_workers: int, autosave_every_n_steps: int) -> None:
    """Print learning configuration at startup.

    This prints the generic training setup, then defers to the agent
    for algorithm-specific hyperparameters and spaces.
    """
    print(f"\n=== Learning Configuration ===")
    print(f"Workers (parallel Godot instances): {num_workers}")
    print(f"Ports: {BASE_PORT} to {BASE_PORT + num_workers - 1}")
    print(f"Autosave every {autosave_every_n_steps} steps (counted per worker)")
    agent.print_config()
    print(f"=== Starting Training ===\n")


def validate_model_config(config) -> dict:
    """Validate the model config section from the YAML config file.

    Behaviour:
      - No built-in defaults are used.
      - The 'model' section must be present and a mapping.
      - All required fields must be provided with valid types/values.
      - Any missing or invalid field raises ValueError.
    """
    if not isinstance(config, dict):
        raise ValueError("'model' section must be a mapping (YAML object) with all required fields.")

    required_types = {
        "path": str,
        "load_on_start": bool,
        "save_on_exit": bool,
        "autosave_every_n_steps": int,
        "num_workers": int,
    }

    validated: dict = {}

    for key, expected_type in required_types.items():
        if key not in config:
            raise ValueError(f"model.{key} is required but missing.")
        value = config[key]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"model.{key} must be {expected_type.__name__}, "
                f"got {type(value).__name__}."
            )
        validated[key] = value

    if validated["path"] == "":
        raise ValueError("model.path must be a non-empty string.")
    if validated["autosave_every_n_steps"] < 0:
        raise ValueError("model.autosave_every_n_steps must be >= 0.")
    if validated["num_workers"] < 1:
        raise ValueError("model.num_workers must be >= 1.")

    return validated


def validate_log_config(config) -> dict:
    """Validate the logs config section from the YAML config file.

    Behaviour:
      - No built-in defaults are used.
      - The 'logs' section must be present and a mapping.
      - All required fields must be provided with valid types/values.
      - Any missing or invalid field raises ValueError.
    """
    if not isinstance(config, dict):
        raise ValueError("'logs' section must be a mapping (YAML object) with all required fields.")

    # Required types; plot_refresh_interval accepts both int and float
    required_types = {
        "log_enabled": bool,
        "log_every_n_steps": int,
        "log_dir": str,
        "reward_window": int,
        "plot_enabled": bool,
    }

    validated: dict = {}

    for key, expected_type in required_types.items():
        if key not in config:
            raise ValueError(f"logs.{key} is required but missing.")
        value = config[key]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"logs.{key} must be {expected_type.__name__}, "
                f"got {type(value).__name__}."
            )
        validated[key] = value

    if "plot_refresh_interval" not in config:
        raise ValueError("logs.plot_refresh_interval is required but missing.")
    pri = config["plot_refresh_interval"]
    if not isinstance(pri, (int, float)):
        raise ValueError("logs.plot_refresh_interval must be a number.")
    pri = float(pri)
    if pri <= 0:
        raise ValueError("logs.plot_refresh_interval must be > 0.")
    validated["plot_refresh_interval"] = pri

    if validated["log_every_n_steps"] < 0:
        raise ValueError("logs.log_every_n_steps must be >= 0.")
    if validated["log_dir"] == "":
        raise ValueError("logs.log_dir must be a non-empty string.")
    if validated["reward_window"] < 1:
        raise ValueError("logs.reward_window must be >= 1.")

    return validated


def worker(worker_id: int, agent: RLAgent, model_path: Path,
           autosave_every_n_steps: int, save_lock: threading.Lock,
           shutdown_event: threading.Event,
           stats_logger: StatsLogger = None,
           log_every_n_steps: int = 0,
           reward_window: int = 20) -> None:
    """
    Handles one Godot connection on its own port.
    
    Each worker runs in its own thread. It has its own:
      - TCP server socket on BASE_PORT + worker_id
      - learning_steps / total_games counters
      - current_episode_reward tracker
    
    It shares with other workers:
      - agent (Q-table is protected by agent._lock)
      - model_path / save_lock (so two workers don't save simultaneously)
      - shutdown_event (Ctrl+C from any thread signals all workers to stop)
      - stats_logger (holds the shared episode reward pool + log buffer)
    """
    port = BASE_PORT + worker_id
    
    # Per-worker tracking (no sharing needed — each thread owns these)
    learning_steps: int = 0
    total_games: int = 0
    current_episode_reward: float = 0.0

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, 'SO_REUSEPORT'):
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    server.settimeout(1.0)  # allows checking shutdown_event periodically
    try:
        server.bind((HOST, port))
    except OSError as e:
        print(f"[W{worker_id}] FATAL: Could not bind to port {port}: {e}")
        server.close()
        return
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
        last_seen: float = time.time()

        while not shutdown_event.is_set():
            try:
                data = conn.recv(1024)
            except socket.timeout:
                if time.time() - last_seen > HEARTBEAT_TIMEOUT:
                    print(f"[W{worker_id}] WARNING: No message received for {HEARTBEAT_TIMEOUT:.0f}s. Worker may be stalled.")
                    last_seen = time.time()  # reset to avoid spamming
                continue  # no data yet, loop and check shutdown_event

            if not data: #standard way to detect closed connection
                print(f"[W{worker_id}] Godot disconnected.")
                break

            buffer += data.decode()

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1) #split off one line at a time

                try:
                    message: dict = json.loads(line)

                    # Skip handshake messages (sent by each Godot instance on connect)
                    if message.get("type") == "handshake":
                        print(f"[W{worker_id}] Handshake received: training_mode='{message.get('training_mode')}'")
                        last_seen = time.time()
                        continue

                    # Heartbeat: just update last_seen and skip learning
                    if message.get("type") == "heartbeat":
                        last_seen = time.time()
                        continue

                    last_seen = time.time()
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
                        if stats_logger is not None:
                            stats_logger.add_episode_reward(current_episode_reward, reward_window)
                            stats_logger.add_episode_loss(getattr(agent, 'last_loss', 0.0), reward_window)
                            stats_logger.add_episode_entropy(getattr(agent, 'last_entropy', 0.0), reward_window)
                            if hasattr(stats_logger, 'add_episode_critic_loss'):
                                stats_logger.add_episode_critic_loss(getattr(agent, 'last_critic_loss', 0.0), reward_window)
                            if hasattr(stats_logger, 'add_episode_approx_kl'):
                                stats_logger.add_episode_approx_kl(getattr(agent, '_last_approx_kl', 0.0), reward_window)
                        current_episode_reward = 0.0

                    # Autosave: only worker 0 saves — all workers share the same agent,
                    # so saving once is enough; no need for every worker to write the file.
                    if worker_id == 0 and autosave_every_n_steps > 0 and learning_steps % autosave_every_n_steps == 0:
                        with save_lock:
                            try:
                                agent.save(str(model_path))
                                #print(f"[W{worker_id}|Step {learning_steps}] Model autosaved")
                            except Exception as e:
                                print(f"[W{worker_id}] Autosave failed: {e}")

                    # Stats logging: only worker 0 logs agent stats.
                    #
                    # All workers already contribute to the shared episode reward pool
                    # via add_episode_reward(), so avg_reward already reflects all workers.
                    # Agent internals (W/b for PG, Q-table for QL) are shared and produce
                    # identical get_stats() snapshots across workers — having multiple
                    # workers log them would only create redundant rows with the same data.
                    if (worker_id == 0
                            and stats_logger is not None
                            and log_every_n_steps > 0
                            and learning_steps % log_every_n_steps == 0):
                        avg_reward = stats_logger.avg_episode_reward()
                        stats_logger.record(worker_id, learning_steps, total_games, avg_reward, agent.get_stats())

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
    training_method, training_mode = receive_handshake()
    #training_method = "policy_gradient_dnn"  #for headless training
    #training_mode = "coach" #for headless training
    agent, config = load_agent(training_method, training_mode)

    model_config: dict = validate_model_config(config.get("model"))
    model_path: Path = Path(__file__).parent / model_config.get("path")
    load_on_start: bool = model_config.get("load_on_start")
    save_on_exit: bool = model_config.get("save_on_exit")
    autosave_every_n_steps: int = int(model_config.get("autosave_every_n_steps"))
    num_workers: int = int(model_config.get("num_workers"))

    log_config: dict = validate_log_config(config.get("logs"))
    log_enabled: bool = log_config.get("log_enabled")
    log_every_n_steps: int = log_config.get("log_every_n_steps")
    log_base_dir: Path = Path(__file__).parent / log_config.get("log_dir")
    reward_window: int = log_config.get("reward_window")
    plot_enabled: bool = log_config.get("plot_enabled")
    plot_refresh_interval: float = log_config.get("plot_refresh_interval")

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

    # Install SIGINT handler before starting workers so Ctrl+C always sets
    # shutdown_event reliably. On Linux with TkAgg, plt.pause() can swallow
    # KeyboardInterrupt, preventing the except clause below from ever running.
    def _sigint_handler(signum, frame):
        print("\nShutting down all workers...")
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore further Ctrl+C
        shutdown_event.set()
    signal.signal(signal.SIGINT, _sigint_handler)

    # Stats logger (worker 0 records into it; main saves CSV on exit).
    # Also holds the shared episode reward pool across all workers.
    # Disabled entirely when log_enabled is false — no recording, no CSV output.
    logger_cls = LOGGERS_MAP[training_method]
    stats_logger: StatsLogger = logger_cls() if log_enabled else None

    # Start all workers FIRST so every port is bound before matplotlib
    # touches the Tk event loop (plt.pause on a background thread blocks on Linux).
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

    # Watch-dog thread: sets shutdown_event when every worker exits naturally
    # (e.g. all Godot windows are closed), which unblocks the plotter below.
    def _watch_workers():
        for t in threads:
            t.join()
        shutdown_event.set()
    threading.Thread(target=_watch_workers, daemon=True, name="worker-watchdog").start()

    # Run the plot loop on the main thread — TkAgg requires GUI calls on the
    # main thread on Linux; running it as a daemon thread caused plt.pause()
    # to stall the process and delay worker socket binding.
    # If plotting is disabled, the main thread simply waits for shutdown_event.
    plotter_cls = PLOTTERS_MAP[training_method]
    _plot_active = plot_enabled and log_enabled  # plotting requires logging to be enabled
    plotter: Plotter = plotter_cls(stats_logger, refresh_interval=plot_refresh_interval, reward_window=reward_window) if _plot_active else None
    try:
        if _plot_active:
            plotter._run(shutdown_event)   # blocks until shutdown_event is set
        else:
            shutdown_event.wait()          # no plot; just wait for workers to finish
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nShutting down all workers...")
    finally:
        shutdown_event.set()
        for t in threads:
            t.join(timeout=2.0)
        if save_on_exit:
            with save_lock:
                try:
                    agent.save(str(model_path))
                    print("Final model saved.")
                except Exception as e:
                    print(f"Failed to save model: {e}")
        if stats_logger:
            stats_logger.save_log(log_base_dir, config)
        print("Done.")


if __name__ == "__main__":
    main()