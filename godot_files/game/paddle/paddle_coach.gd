extends "res://game/paddle/paddle_base.gd"

# Path to the exported coach Q-table JSON file (produced by export_Qtable.py).
@export var q_table_path: String = "res://models/q_table_coach.json"

const ACTIONS: Array   = ["UP", "DOWN", "STAY"]
const MAX_SPEED: float = 1600.0

# Populated from the JSON metadata exported by Python.
var state_vars: Array      = []   # ordered list of variable names
var bins: Dictionary       = {}   # var_name -> number of bins
var q_table: Dictionary    = {}   # "(bin_tuple, 'ACTION')" -> Q-value


func _ready() -> void:
	super._ready()
	_load_q_table()


func _load_q_table() -> void:
	assert(FileAccess.file_exists(q_table_path), "PaddleCoach: Q-table not found at: " + q_table_path)

	var file: FileAccess = FileAccess.open(q_table_path, FileAccess.READ)
	var json: JSON = JSON.new()
	var parse_result: int = json.parse(file.get_as_text())
	assert(parse_result == OK, "PaddleCoach: Failed to parse Q-table JSON: " + json.get_error_message())

	var data: Dictionary = json.data
	state_vars = data["state_vars"]
	bins       = data["bins"]
	q_table    = data["qtable"]

	print("PaddleCoach: loaded %d entries, state_vars=%s, bins=%s" % [q_table.size(), state_vars, bins])
	file.close()


# Map a normalized [-1, 1] value to the discrete bin index used by the Q-table.
# Mirrors Python's _discretize_state logic exactly.
func _discretize(value: float, n_bins: int) -> int:
	var norm: float = (clamp(value, -1.0, 1.0) + 1.0) / 2.0   # [-1,1] -> [0,1]
	return min(int(norm * n_bins), n_bins - 1)


func get_direction() -> int:
	if ball == null or ball.is_queued_for_deletion():
		var balls: Array[Node] = get_tree().get_nodes_in_group("ball")
		if balls.size() > 0:
			ball = balls[0]
		else:
			return 0

	if q_table.is_empty():
		return 0

	var viewport_size: Vector2 = get_viewport().get_visible_rect().size
	var max_x: float = viewport_size.x
	var max_y: float = viewport_size.y

	# The coach was trained from the left side (PaddleA perspective).
	# Mirror ball_x and ball_vx when playing on the right so the state matches.
	var on_right_side: bool = position.x > max_x / 2.0
	var mirror: float       = -1.0 if on_right_side else 1.0

	# Normalized [-1, 1] values for every state variable the model knows about.
	var norm_values: Dictionary = {
		"paddle_y": (position.y           - max_y / 2.0) / (max_y / 2.0),
		"ball_x":   mirror * (ball.position.x - max_x / 2.0) / (max_x / 2.0),
		"ball_y":   (ball.position.y       - max_y / 2.0) / (max_y / 2.0),
		"ball_vx":  mirror * ball.velocity.x / MAX_SPEED,
		"ball_vy":  ball.velocity.y / MAX_SPEED,
	}

	# Discretize each variable to its bin index, in the order state_vars specifies.
	var parts := PackedStringArray()
	for var_name: String in state_vars:
		var idx: int = _discretize(norm_values[var_name], int(bins[var_name]))
		parts.append(str(idx))

	# Reproduce Python's str(tuple) format: "(0, 3, 5, 1, 2)"
	var state_str: String = "(" + ", ".join(parts) + ")"

	var best_action: String = "STAY"
	var best_q: float       = -INF
	for action: String in ACTIONS:
		var key: String = "(%s, '%s')" % [state_str, action]
		var q: float    = float(q_table.get(key, 0.0))
		if q > best_q:
			best_q = q
			best_action  = action

	if best_q == -INF:
		return 0  # No valid actions found in Q-table for this state, default to STAY   
	if best_action == "UP":
		return -1
	elif best_action == "DOWN":
		return 1
	return 0
