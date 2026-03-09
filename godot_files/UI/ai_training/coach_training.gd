extends Node2D

@export var ball_scene: PackedScene

var paddle_a: Node2D

func _ready() -> void:
	var viewport_size: Vector2 = Global.VIEWPORT_SIZE

	paddle_a = $PaddleA
	paddle_a.position = Vector2(40.0, viewport_size.y / 2)

	$BounceWall.position = Vector2(viewport_size.x - $BounceWall.thickness, 0.0)
	$Ceiling.position = Vector2(0.0, 0.0)
	$Floor.position = Vector2(0.0, viewport_size.y - $Floor.thickness)
	$KillZone.position = Vector2(20.0 - $KillZone.thickness / 2, viewport_size.y / 2)

	Dispatcher.ball_destroyed.connect(_on_ball_destroyed)
	spawn_ball()



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
