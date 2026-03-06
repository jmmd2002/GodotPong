extends Node2D

@export var ball_scene: PackedScene

func _ready() -> void:
	var viewport_size: Vector2 = get_viewport().get_visible_rect().size

	$PaddleA.position = Vector2(40.0, viewport_size.y / 2)
	$PaddleB.position = Vector2(viewport_size.x - 40.0, viewport_size.y / 2)

	$Ball.position = Vector2(viewport_size.x / 2, viewport_size.y / 2)
	$Ceiling.position = Vector2(0.0, 0.0)
	$Floor.position = Vector2(0.0, viewport_size.y - $Floor.thickness)
	$KillZoneA.position = Vector2(20.0 - $KillZoneA.thickness / 2, viewport_size.y / 2)
	$KillZoneB.position = Vector2(viewport_size.x - 20.0 + $KillZoneB.thickness / 2, viewport_size.y / 2)

	Dispatcher.ball_destroyed.connect(_on_ball_destroyed)


func _on_ball_destroyed() -> void:
	spawn_ball()


func spawn_ball() -> void:
	var active_balls = get_tree().get_nodes_in_group("ball").filter(
		func(b): return not b.is_queued_for_deletion()
	)
	if active_balls.size() > 0:
		return
	var ball = ball_scene.instantiate()
	add_child(ball)
