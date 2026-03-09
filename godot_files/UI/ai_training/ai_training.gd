extends Node2D

@export var ball_scene: PackedScene

var paddle_a: Node2D
var paddle_b: Node2D

func _ready() -> void:
	var viewport_size: Vector2 = Global.VIEWPORT_SIZE

	paddle_a = $PaddleA
	paddle_a.position = Vector2(40.0, viewport_size.y / 2)

	paddle_b = _spawn_paddle($PaddleB, Global.paddle_b_mode, "b", "PaddleB")
	paddle_b.position = Vector2(viewport_size.x - 40.0, viewport_size.y / 2)

	$Ball.position = Vector2(viewport_size.x / 2, viewport_size.y / 2)
	$Ceiling.position = Vector2(0.0, 0.0)
	$Floor.position = Vector2(0.0, viewport_size.y - $Floor.thickness)
	$KillZoneA.position = Vector2(20.0 - $KillZoneA.thickness / 2, viewport_size.y / 2)
	$KillZoneB.position = Vector2(viewport_size.x - 20.0 + $KillZoneB.thickness / 2, viewport_size.y / 2)

	Dispatcher.ball_destroyed.connect(_on_ball_destroyed)


func _spawn_paddle(placeholder: Node2D, mode: String, side: String, node_name: String) -> Node2D:
	placeholder.free()
	var scene_key: String = "manual_" + side if mode == "manual" else mode
	var scene_path: String = Global.PADDLE_SCENES.get(scene_key, Global.PADDLE_SCENES["static"])
	var paddle: Node2D = (load(scene_path) as PackedScene).instantiate()
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
