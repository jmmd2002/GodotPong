extends Node

signal score_changed(score_left: int, score_right: int)

var score_left: int = 0
var score_right: int = 0

func _ready() -> void:
	Dispatcher.scored.connect(_on_scored)
	return

func _on_scored(side: String) -> void:
	if side == "left":
		score_left += 1
	elif side == "right":
		score_right += 1
	else:
		print("Error: scored on unknown side.")
		
	emit_signal("score_changed", score_left, score_right)
	return
