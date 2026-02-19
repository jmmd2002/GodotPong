extends Node2D

@export var ball_scene: PackedScene

func _ready() -> void:
	var viewport_size: Vector2 = get_viewport().get_visible_rect().size
	$PaddleA.position = Vector2(40.0, viewport_size.y / 2)
	$PaddleB.position = Vector2(viewport_size.x -40.0, viewport_size.y /2)
	$Ball.position = Vector2(viewport_size.x/2, viewport_size.y/2)
	$Ceiling.position = Vector2(0.0, 0.0)
	$Floor.position = Vector2(0.0, viewport_size.y-60)
	$KillZoneA.position = Vector2(0.0, viewport_size.y/2)
	$KillZoneB.position = Vector2(viewport_size.x, viewport_size.y/2)
	
	Dispatcher.ball_destroyed.connect(_on_ball_destroyed)
	

func _on_ball_destroyed():
	spawn_ball()
	
func spawn_ball():
	var ball = ball_scene.instantiate()  # create a new Ball
	add_child(ball)
