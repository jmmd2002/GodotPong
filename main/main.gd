extends Node2D

func _ready() -> void:
	var viewport_size: Vector2 = get_viewport().get_visible_rect().size
	$Paddle.position = Vector2(40, viewport_size.y / 2)
