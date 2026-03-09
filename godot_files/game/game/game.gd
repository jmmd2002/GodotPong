extends Node2D

@export var ball_scene: PackedScene

func _ready() -> void:
	var viewport_size: Vector2 = Vector2(
		ProjectSettings.get_setting("display/window/size/viewport_width", 1280),
		ProjectSettings.get_setting("display/window/size/viewport_height", 720)
	)

	# Spawn paddles dynamically based on character selection
	var paddle_a: Node = _spawn_paddle(Global.paddle_a_mode, "PaddleA")
	var paddle_b: Node = _spawn_paddle(Global.paddle_b_mode, "PaddleB")

	paddle_a.position = Vector2(40.0, viewport_size.y / 2)
	paddle_b.position = Vector2(viewport_size.x - 40.0, viewport_size.y / 2)

	$Ball.position = Vector2(viewport_size.x / 2, viewport_size.y / 2)
	$Ceiling.position = Vector2(0.0, 0.0)
	$Floor.position = Vector2(0.0, viewport_size.y - $Floor.thickness)
	$KillZoneA.position = Vector2(20.0 - $KillZoneA.thickness / 2, viewport_size.y / 2)
	$KillZoneB.position = Vector2(viewport_size.x - 20.0 + $KillZoneB.thickness / 2, viewport_size.y / 2)

	Dispatcher.ball_destroyed.connect(_on_ball_destroyed)


func _spawn_paddle(mode: String, node_name: String) -> Node:
	var scene_path: String = Global.PADDLE_SCENES.get(mode, Global.PADDLE_SCENES["manual_a"])
	var paddle: Node = load(scene_path).instantiate()
	paddle.name = node_name
	add_child(paddle)
	return paddle


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
