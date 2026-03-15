extends "res://game/paddle/paddle_base.gd"

# Path to the exported DNN policy JSON file (produced by export_policy_dnn.py).
@export var policy_path: String = "res://models/policy_ppo_coach.json"

# Populated from the JSON exported by Python.
var state_vars:   Array = []   # ordered list of state variable names
var actions:      Array = []   # e.g. ["UP", "DOWN", "STAY"]
var hidden_sizes: Array = []   # e.g. [128, 64]

# MLP weights: layers[i] = {"W": Array[Array[float]], "b": Array[float]}
# layers[0..n-2] are ReLU hidden layers; layers[n-1] is the linear output layer.
var layers: Array = []

var _opponent: Node2D = null


func _ready() -> void:
	super._ready()
	_load_policy()


func _load_policy() -> void:
	assert(FileAccess.file_exists(policy_path), \
		"PGDNNStudent: policy not found at: " + policy_path)

	var file: FileAccess = FileAccess.open(policy_path, FileAccess.READ)
	var json: JSON = JSON.new()
	var parse_result: int = json.parse(file.get_as_text())
	assert(parse_result == OK, \
		"PGDNNStudent: failed to parse policy JSON: " + json.get_error_message())
	file.close()

	var data: Dictionary = json.data
	state_vars   = data["state_vars"]
	actions      = data["actions"]
	hidden_sizes = data["hidden_sizes"]

	# Reconstruct ordered layer list from params dict.
	# Python exports keys W0/b0, W1/b1, ... in order.
	var params: Dictionary = data["params"]
	var num_layers: int = hidden_sizes.size() + 1   # hidden layers + output layer
	layers.clear()
	for i: int in range(num_layers):
		layers.append({
			"W": params["W" + str(i)],
			"b": params["b" + str(i)],
		})

	print("PGDNNStudent: loaded MLP  state_vars=%s  actions=%s  hidden_sizes=%s" \
		% [state_vars, actions, hidden_sizes])


# Return the other paddle in the scene (lazy-cached).
func _get_opponent() -> Node2D:
	if _opponent != null and not _opponent.is_queued_for_deletion():
		return _opponent
	for p: Node in get_tree().get_nodes_in_group("paddle"):
		if p != self:
			_opponent = p
			return _opponent
	return null


# Compute relu(x) element-wise on a PackedFloat64Array.
func _relu(x: PackedFloat64Array) -> PackedFloat64Array:
	var out: PackedFloat64Array = PackedFloat64Array()
	out.resize(x.size())
	for i: int in range(x.size()):
		out[i] = max(0.0, x[i])
	return out


# Multiply weight matrix W (Array of rows) by vector x, add bias b.
# Returns a PackedFloat64Array of length W.size().
func _linear(W: Array, b: Array, x: PackedFloat64Array) -> PackedFloat64Array:
	var out: PackedFloat64Array = PackedFloat64Array()
	out.resize(W.size())
	for i: int in range(W.size()):
		var row: Array = W[i]
		var val: float = float(b[i])
		for j: int in range(x.size()):
			val += float(row[j]) * x[j]
		out[i] = val
	return out


func get_direction() -> int:
	if ball == null or ball.is_queued_for_deletion():
		var balls: Array[Node] = get_tree().get_nodes_in_group("ball")
		if balls.size() > 0:
			ball = balls[0]
		else:
			return 0

	if layers.is_empty():
		return 0

	var max_x: float = Global.VIEWPORT_SIZE.x
	var max_y: float = Global.VIEWPORT_SIZE.y

	# The student was trained from paddleA's (left-side) perspective.
	# Mirror ball_x and ball_vx when playing on the right.
	var on_right_side: bool = position.x > max_x / 2.0
	var mirror: float       = -1.0 if on_right_side else 1.0

	var opponent: Node2D       = _get_opponent()
	var opponent_norm_y: float = 0.0
	if opponent != null:
		opponent_norm_y = (opponent.position.y - max_y / 2.0) / (max_y / 2.0)

	var norm_values: Dictionary = {
		"paddleA_y": (position.y           - max_y / 2.0) / (max_y / 2.0),
		"paddleB_y": opponent_norm_y,
		"ball_x":    mirror * (ball.position.x - max_x / 2.0) / (max_x / 2.0),
		"ball_y":    (ball.position.y       - max_y / 2.0) / (max_y / 2.0),
		"ball_vx":   mirror * ball.velocity.x / Global.MAX_SPEED,
		"ball_vy":   ball.velocity.y / Global.MAX_SPEED,
	}

	# Build the state vector in the order state_vars demands.
	var x: PackedFloat64Array = PackedFloat64Array()
	for var_name: String in state_vars:
		x.append(norm_values[var_name])

	# MLP forward pass:
	#   hidden layers:  x = relu(Wi @ x + bi)
	#   output layer:   logits = W_last @ x + b_last
	for i: int in range(layers.size()):
		var layer: Dictionary = layers[i]
		x = _linear(layer["W"], layer["b"], x)
		if i < layers.size() - 1:   # apply ReLU to all but the last layer
			x = _relu(x)

	# argmax over logits — no softmax needed for inference.
	var best_action: String = "STAY"
	var best_logit: float   = -INF
	for i: int in range(actions.size()):
		if x[i] > best_logit:
			best_logit  = x[i]
			best_action = actions[i]

	if best_action == "UP":
		return -1
	elif best_action == "DOWN":
		return 1
	return 0
