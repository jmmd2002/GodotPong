extends "res://game/paddle/paddle_base.gd"

# Driven externally via set_ai_action().
# Covers both the ai_qlearn and coach agent roles.

func get_direction() -> int:
	if ai_action == "UP":
		return -1
	elif ai_action == "DOWN":
		return 1
	return 0  # STAY
