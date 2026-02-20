extends Node

var score_left: float = 0
var score_right: float = 0

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	Dispatcher.scored.connect(_on_scored)
	return


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _on_scored(side: String) -> void:
	if side == "left":
		score_left += 1
	if side == "right":
		score_right += 1
	return
