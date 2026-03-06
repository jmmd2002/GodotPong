extends "res://game/paddle/paddle_base.gd"

func get_direction() -> int:
	# Lazily grab and refresh the ball reference.
	if ball == null or ball.is_queued_for_deletion():
		var balls = get_tree().get_nodes_in_group("ball")
		if balls.size() > 0:
			ball = balls[0]
		else:
			return 0

	var y_diff: float = ball.position.y - position.y
	if y_diff > 0:
		return 1
	elif y_diff < 0:
		return -1
	return 0
