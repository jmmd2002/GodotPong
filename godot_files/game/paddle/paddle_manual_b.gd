extends "res://game/paddle/paddle_base.gd"

# Human-controlled paddle using the B-side input actions.

func get_direction() -> int:
	var direction: int = 0
	if Input.is_action_pressed("move_up_B"):
		direction -= 1
	if Input.is_action_pressed("move_down_B"):
		direction += 1
	return direction
