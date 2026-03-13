extends Node

var client: StreamPeerTCP = StreamPeerTCP.new()
var connected: bool = false
var reconnect_timer: float = 0.0
var incoming_buffer: String = ""
var last_action: String = "STAY"
var next_frame_id: int = 0
const RECONNECT_INTERVAL: float = 1.0  # seconds
const HEARTBEAT_INTERVAL: float = 3.0  # seconds between heartbeat pings
var heartbeat_timer: float = 0.0

# Set a different port for each Godot instance in the Inspector (5000, 5001, 5002...)
@export var port: int = 5000

# sparse reward tracking
var score_left_prev: int = 0
var score_right_prev: int = 0
var prev_ball_vx: float = 0.0

# episode length limit
const MAX_SCORE: int = 10

#get game objects
var ball: Node2D
var paddleA: Node2D
var paddleB: Node2D
var game_manager: Node

#normalization values
var max_x: float = Global.VIEWPORT_SIZE.x
var max_y: float = Global.VIEWPORT_SIZE.y
var max_speed: float = Global.MAX_SPEED

func _ready() -> void:
	# Allow port to be overridden via command-line: godot -- --port 5001
	var args: PackedStringArray = OS.get_cmdline_user_args()
	for i in range(args.size() - 1):
		if args[i] == "--port":
			port = int(args[i + 1])
			break

	print("Connecting to server on port ", port, "...")
	game_manager = GameManager

func _process(delta: float):
	if not connected:
		reconnect_timer +=  delta
		if reconnect_timer >= RECONNECT_INTERVAL: #try to reconnect after a certain amount of time
			reconnect_timer = 0
			try_connect()
	else:
		heartbeat_timer += delta
		var sent_frame_id: int = send_state()
		if sent_frame_id != -1:
			heartbeat_timer = 0.0
			var action: String = receive_action(sent_frame_id)
			if not action.is_empty():
				last_action = action
				apply_action(last_action)
			else:
				print("Warning: No action received. Keeping last action: ", last_action)
		elif heartbeat_timer >= HEARTBEAT_INTERVAL:
			heartbeat_timer = 0.0
			var hb: String = JSON.stringify({"type": "heartbeat"}) + "\n"
			client.put_data(hb.to_utf8_buffer())

func try_connect():
	var err: Error = client.connect_to_host("127.0.0.1", port)
	if err == OK:
		client.poll() #poll every step for server status
		if client.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			connected = true
			incoming_buffer = ""
			next_frame_id = 0
			score_left_prev = game_manager.score_left
			score_right_prev = game_manager.score_right
			var handshake: String = JSON.stringify({"type": "handshake", "training_method": Global.training_method, 
													"training_mode": Global.training_mode}) + "\n"
			client.put_data(handshake.to_utf8_buffer())
			print("Connected to Python server!")
		else:
			client = StreamPeerTCP.new()
			print("Connecting...")
	else:
		client = StreamPeerTCP.new()
		print("Connection error with value ", err, ". Retrying...")

func send_state() -> int:
	for b in get_tree().get_nodes_in_group("ball"):
		ball = b
		break
	paddleA = get_tree().root.find_child("PaddleA", true, false)
	paddleB = get_tree().root.find_child("PaddleB", true, false)

	if ball == null:
		print("Waiting for ball...")
		return -1

	var frame_id: int = next_frame_id
	next_frame_id += 1

	var prev_reward: float = 0.0
	var done: bool = false
	var score_left_current: int = game_manager.score_left
	var score_right_current: int = game_manager.score_right

	if score_left_current > score_left_prev:
		prev_reward = 5.0
		done = true
	elif score_right_current > score_right_prev:
		prev_reward = -1.0
		done = true
	elif prev_ball_vx < 0.0 and ball.velocity.x > 0.0:
		prev_reward = 0.1  # PaddleA hit the ball

	prev_ball_vx = ball.velocity.x
	score_left_prev = score_left_current
	score_right_prev = score_right_current

	var state: Dictionary = {
		"frame_id": frame_id,
		"paddleA_y": paddleA.position.y,
		"paddleB_y": paddleB.position.y,
		"ball_x": ball.position.x,
		"ball_y": ball.position.y,
		"ball_vx": ball.velocity.x,
		"ball_vy": ball.velocity.y,
		"prev_reward": prev_reward,
		"done": done
	}

	state = normalize_state(state)
	var json: String = JSON.stringify(state) + "\n"
	client.put_data(json.to_utf8_buffer())
	return frame_id

func normalize_state(state: Dictionary) -> Dictionary:
	'''Normalize state values between -1 and 1'''

	var paddleA_y: float = (state.get("paddleA_y") - max_y/2) / (max_y/2)
	var paddleB_y: float = (state.get("paddleB_y") - max_y/2) / (max_y/2)
	var ball_x: float = (state.get("ball_x") - max_x/2) / (max_x/2)
	var ball_y: float = (state.get("ball_y") - max_y/2) / (max_y/2)
	var ball_vx: float = state.get("ball_vx") / max_speed
	var ball_vy: float = state.get("ball_vy") / max_speed

	state = {
		"frame_id": state.get("frame_id"),
		"paddleA_y": paddleA_y,
		"paddleB_y": paddleB_y,
		"ball_x": ball_x,
		"ball_y": ball_y,
		"ball_vx": ball_vx,
		"ball_vy": ball_vy,
		"prev_reward": state.get("prev_reward", 0.0),
		"done": state.get("done", false)
	}
	return state
func receive_action(expected_frame_id: int) -> String:
	"""Block until a newline-terminated action for expected_frame_id is fully received."""
	while connected:
		client.poll()
		if client.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			connected = false
			print("Disconnected from server. Attempting to reconnect...")
			client = StreamPeerTCP.new()
			incoming_buffer = ""
			return ""

		if client.get_available_bytes() <= 0:
			continue

		var response_bytes: int = client.get_available_bytes()
		incoming_buffer += client.get_utf8_string(response_bytes)

		while true:
			var newline_idx: int = incoming_buffer.find("\n")
			if newline_idx == -1:
				break

			var message: String = incoming_buffer.substr(0, newline_idx).strip_edges()
			incoming_buffer = incoming_buffer.substr(newline_idx + 1)

			if message.is_empty():
				continue

			var json_parser: JSON = JSON.new()
			var error: int = json_parser.parse(message)

			if error == OK:
				var action_data: Dictionary = json_parser.data
				if action_data.has("frame_id") and action_data.has("action"):
					var received_frame_id: int = int(action_data["frame_id"])
					if received_frame_id == expected_frame_id: #validate server data mismatch
						return str(action_data["action"])
					print("Warning: Frame mismatch. Expected ", expected_frame_id, ", got ", received_frame_id)
				else:
					print("Warning: Received JSON missing 'frame_id' or 'action': ", action_data)
			else:
				print("JSON Parse Error: ", json_parser.get_error_message())
	return ""

func apply_action(action: String) -> void:
	"""Apply the received action to PaddleA (always the trained paddle)."""
	paddleA.set_ai_action(action)
