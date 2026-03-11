extends "res://game/paddle/paddle_base.gd"

# Path to the exported policy JSON file (produced by export_policy.py).
@export var policy_path: String = "res://models/policy_coach.json"

# Populated from the JSON exported by Python.
var state_vars: Array = []   # ordered list of variable names
var actions:    Array = []   # ["UP", "DOWN", "STAY"]
var W:          Array = []   # shape: (num_actions, state_dim) — list of lists
var b:          Array = []   # shape: (num_actions,)


func _ready() -> void:
	super._ready()
	_load_policy()


func _load_policy() -> void:
	assert(FileAccess.file_exists(policy_path), "PGCoach: policy not found at: " + policy_path)

	var file: FileAccess = FileAccess.open(policy_path, FileAccess.READ)
	var json: JSON = JSON.new()
	var parse_result: int = json.parse(file.get_as_text())
	assert(parse_result == OK, "PGCoach: failed to parse policy JSON: " + json.get_error_message())
	file.close()

	var data: Dictionary = json.data
	state_vars = data["state_vars"]
	actions    = data["actions"]
	W          = data["W"]
	b          = data["b"]

	print("PGCoach: loaded policy  state_vars=%s  actions=%s  W shape=(%d, %d)" \
		% [state_vars, actions, W.size(), (W[0] as Array).size()])


func get_direction() -> int:
	if ball == null or ball.is_queued_for_deletion():
		var balls: Array[Node] = get_tree().get_nodes_in_group("ball")
		if balls.size() > 0:
			ball = balls[0]
		else:
			return 0

	if W.is_empty():
		return 0

	var max_x: float = Global.VIEWPORT_SIZE.x
	var max_y: float = Global.VIEWPORT_SIZE.y

	# The coach was trained from the left side (PaddleA perspective).
	# Mirror ball_x and ball_vx when playing on the right so the state matches.
	var on_right_side: bool = position.x > max_x / 2.0
	var mirror: float       = -1.0 if on_right_side else 1.0

	# Normalised [-1, 1] values for all variables the coach model may reference.
	var norm_values: Dictionary = {
		"paddle_y": (position.y           - max_y / 2.0) / (max_y / 2.0),
		"ball_x":   mirror * (ball.position.x - max_x / 2.0) / (max_x / 2.0),
		"ball_y":   (ball.position.y       - max_y / 2.0) / (max_y / 2.0),
		"ball_vx":  mirror * ball.velocity.x / Global.MAX_SPEED,
		"ball_vy":  ball.velocity.y / Global.MAX_SPEED,
	}

	# Build the state vector in the order state_vars demands.
	var s: PackedFloat64Array = PackedFloat64Array()
	for var_name: String in state_vars:
		s.append(norm_values[var_name])

	# Compute logits = W @ s + b and pick argmax (no softmax needed for inference).
	var best_action: String = "STAY"
	var best_logit: float   = -INF
	for i: int in range(actions.size()):
		var row: Array = W[i]
		var logit: float = float(b[i])
		for j: int in range(s.size()):
			logit += float(row[j]) * s[j]
		if logit > best_logit:
			best_logit  = logit
			best_action = actions[i]

	if best_action == "UP":
		return -1
	elif best_action == "DOWN":
		return 1
	return 0
