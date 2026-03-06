extends Node2D

@export var ball_scene: PackedScene

const PADDLE_SCENES: Dictionary = {
	"ai_qlearn": "res://game/paddle/paddle_ai.tscn",
	"coach":     "res://game/paddle/paddle_ai.tscn",
	"homing":    "res://game/paddle/paddle_homing.tscn",
	"static":    "res://game/paddle/paddle_static.tscn",
	"off":       "res://game/paddle/paddle_static.tscn",
	"manual_a":  "res://game/paddle/paddle_manual_a.tscn",
}

var paddle_a: Node2D

func _ready() -> void:
	var viewport_size: Vector2 = get_viewport().get_visible_rect().size

	paddle_a = _spawn_paddle($PaddleA, Global.paddle_a_mode)
	paddle_a.position = Vector2(40.0, viewport_size.y / 2)

	$BounceWall.position = Vector2(viewport_size.x - $BounceWall.thickness, 0.0)
	$Ceiling.position = Vector2(0.0, 0.0)
	$Floor.position = Vector2(0.0, viewport_size.y - $Floor.thickness)
	$KillZone.position = Vector2(0.0, viewport_size.y / 2)

	Dispatcher.ball_destroyed.connect(_on_ball_destroyed)
	spawn_ball()


func _spawn_paddle(placeholder: Node2D, mode: String) -> Node2D:
	placeholder.free()
	var scene_path: String = PADDLE_SCENES.get(mode, PADDLE_SCENES["static"])
	var paddle: Node2D = (load(scene_path) as PackedScene).instantiate()
	paddle.name = "PaddleA"
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
