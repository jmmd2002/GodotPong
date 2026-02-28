extends Area2D

@export var side: String
@export var thickness: float = 40.0
@onready var viewport_size: Vector2 = get_viewport_rect().size

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	_initialize()

func _initialize() -> void:
	var mask: Shape2D = RectangleShape2D.new()
	mask.size = Vector2(thickness, viewport_size.x)
	
	$CollisionShape2D.shape = mask
	return
	
func _process(_delta: float) -> void:
	_handle_collisions()
	return
	
func _handle_collisions() -> void:
	var ball: Node = _check_ball_collision()
	if ball:
		ball.queue_free()
		Dispatcher.emit_ball_destroyed()
		Dispatcher.emit_scored(side)
	
	
func _check_ball_collision() -> Node:
	for ball in get_tree().get_nodes_in_group("ball"):
		var ball_rect: Rect2 = Utils.get_global_rect(ball.get_node("CollisionShape2D"))
		var kill_zone_rect: Rect2 = Utils.get_global_rect($CollisionShape2D)
		
		if kill_zone_rect.intersects(ball_rect):
			return ball
	return null
		
	
	
