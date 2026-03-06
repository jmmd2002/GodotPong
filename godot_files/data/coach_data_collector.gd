extends Node

var client: StreamPeerTCP = StreamPeerTCP.new()
var connected: bool = false
var reconnect_timer: float = 0.0
var incoming_buffer: String = ""
var last_action: String = "STAY"
var next_frame_id: int = 0
const RECONNECT_INTERVAL: float = 1.0

@export var port: int = 5000

# sparse reward tracking
var score_right_prev: int = 0
var prev_ball_vx: float = 0.0

var ball: Node2D
var paddleA: Node2D
var game_manager: Node

#normalization values
@onready var viewport_size: Vector2 = get_viewport().size
@onready var max_x: float = viewport_size.x
@onready var max_y: float = viewport_size.y
@onready var max_speed: float = 1600

func _ready() -> void:
	var args: PackedStringArray = OS.get_cmdline_user_args()
	for i in range(args.size() - 1):
		if args[i] == "--port":
			port = int(args[i + 1])
			break

	print("CoachCollector: Connecting to server on port ", port, "...")
	game_manager = GameManager

func _process(delta: float):
	if not connected:
		reconnect_timer += delta
		if reconnect_timer >= RECONNECT_INTERVAL:
			reconnect_timer = 0
			try_connect()
	else:
		var sent_frame_id: int = send_state()
		if sent_frame_id != -1:
			var action: String = receive_action(sent_frame_id)
			if not action.is_empty():
				last_action = action
				apply_action(last_action)
			else:
				print("Warning: No action received. Keeping last action: ", last_action)

func try_connect():
	var err: Error = client.connect_to_host("127.0.0.1", port)
	if err == OK:
		client.poll()
		if client.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			connected = true
			incoming_buffer = ""
			next_frame_id = 0
			score_right_prev = game_manager.score_right
			var handshake: String = JSON.stringify({"type": "handshake", "training_mode": Global.training_mode}) + "\n"
			client.put_data(handshake.to_utf8_buffer())
			print("CoachCollector: Connected to Python server!")
		else:
			client = StreamPeerTCP.new()
			print("CoachCollector: Connecting...")
	else:
		client = StreamPeerTCP.new()
		print("CoachCollector: Connection error ", err, ". Retrying...")

func send_state() -> int:
	for b in get_tree().get_nodes_in_group("ball"):
		ball = b
		break
	paddleA = get_tree().root.find_child("PaddleA", true, false)

	if ball == null:
		print("Waiting for ball...")
		return -1

	var frame_id: int = next_frame_id
	next_frame_id += 1

	var prev_reward: float = 0.0
	var done: bool = false
	var score_right_current: int = game_manager.score_right

	# KillZone side="left" → score_right increments when ball escapes past PaddleA
	if score_right_current > score_right_prev:
		prev_reward = -1.0  # missed the ball
		done = true
	elif prev_ball_vx < 0.0 and ball.velocity.x > 0.0:
		prev_reward = 0.1   # paddle hit the ball back toward the wall

	prev_ball_vx = ball.velocity.x
	score_right_prev = score_right_current

	var state: Dictionary = {
		"frame_id": frame_id,
		"paddle_y": paddleA.position.y,
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
	state = {
		"frame_id": state.get("frame_id"),
		"paddle_y": (state.get("paddle_y") - max_y / 2) / (max_y / 2),
		"ball_x":   (state.get("ball_x")   - max_x / 2) / (max_x / 2),
		"ball_y":    (state.get("ball_y")    - max_y / 2) / (max_y / 2),
		"ball_vx":   state.get("ball_vx") / max_speed,
		"ball_vy":   state.get("ball_vy") / max_speed,
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
			print("CoachCollector: Disconnected. Attempting to reconnect...")
			client = StreamPeerTCP.new()
			incoming_buffer = ""
			return ""

		if client.get_available_bytes() <= 0:
			continue

		incoming_buffer += client.get_utf8_string(client.get_available_bytes())

		while true:
			var newline_idx: int = incoming_buffer.find("\n")
			if newline_idx == -1:
				break

			var message: String = incoming_buffer.substr(0, newline_idx).strip_edges()
			incoming_buffer = incoming_buffer.substr(newline_idx + 1)

			if message.is_empty():
				continue

			var json_parser: JSON = JSON.new()
			if json_parser.parse(message) == OK:
				var action_data: Dictionary = json_parser.data
				if action_data.has("frame_id") and action_data.has("action"):
					var received_frame_id: int = int(action_data["frame_id"])
					if received_frame_id == expected_frame_id:
						return str(action_data["action"])
					print("Warning: Frame mismatch. Expected ", expected_frame_id, ", got ", received_frame_id)
				else:
					print("Warning: Received JSON missing 'frame_id' or 'action': ", action_data)
			else:
				print("JSON Parse Error: ", json_parser.get_error_message())
	return ""

func apply_action(action: String) -> void:
	if paddleA != null and paddleA.has_method("set_ai_action"):
		paddleA.set_ai_action(action)
