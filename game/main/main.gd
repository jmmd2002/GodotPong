extends Node2D

func _ready() -> void:
	var viewport_size: Vector2 = get_viewport().get_visible_rect().size
	$PaddleA.position = Vector2(40.0, viewport_size.y / 2)
	$PaddleB.position = Vector2(viewport_size.x -40.0, viewport_size.y /2)
	$Ceiling.position = Vector2(0.0, 0.0)
	$Floor.position = Vector2(0.0, viewport_size.y-60)
