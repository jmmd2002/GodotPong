extends Node2D

@export var ball_radius: float = 16.0 #ball sprite is 32x32
@export var speed: float = 400.0
@export var max_angle: float = PI/3

var velocity: Vector2 = Vector2(0.0,0.0)

func _ready():
	
	_reset_ball() # start in the center of the screen
	await get_tree().create_timer(3.0).timeout # start the 3-second countdown
	launch_ball() # release ball

func _physics_process(delta):
	# Handle collisions
	handle_collisions()
	# Move the ball manually
	position += velocity * delta
	return
	
func _reset_ball():
	# Set size and collision mask
	var ball_size: Vector2 = Vector2(ball_radius*2, ball_radius*2)
	var ball_mask: Shape2D = CircleShape2D.new()
	ball_mask.radius = ball_radius
	
	# Set size and collision mask
	$Sprite2D.scale = ball_size / $Sprite2D.texture.get_size()
	$CollisionShape2D.shape = ball_mask
	global_position = get_viewport_rect().size / 2
	velocity = Vector2.ZERO
	
func launch_ball() -> void:
	# Random angle, but not too vertical
	var angle: float = randf_range(-PI / 4, PI / 4)
	
	# Randomly choose left or right
	if randi() % 2 == 0:
		angle += PI
	
	velocity = Vector2.RIGHT.rotated(angle) * speed
		
		
		
# --------------------- Collision handlers --------------------------

func handle_collisions() -> void:
	var wall: Node = check_wall_collision() 
	if wall:
		velocity.y = -velocity.y 
		
	var paddle: Node = check_paddle_collision()
	if paddle:
		var centre: float = paddle.global_position.y
		var offset: float = centre - global_position.y
		var offset_normalized: float = offset / (Utils.get_global_rect(paddle.get_node("CollisionShape2D")).size.y/2)
		var angle: float = max_angle * offset_normalized
		var dir: int = 1 if velocity.x < 0 else -1
		
		velocity = dir * Vector2(1.0,0).rotated(-dir * angle) * speed
		
	return
	
func check_paddle_collision() -> Node:
	for paddle in get_tree().get_nodes_in_group("paddle"):
		var ball_rect: Rect2 = Utils.get_global_rect($CollisionShape2D)
		var paddle_rect: Rect2 = Utils.get_global_rect(paddle.get_node("CollisionShape2D"))
		
		if ball_rect.intersects(paddle_rect):
			return paddle
	return null

func check_wall_collision() -> Node:
	for wall in get_tree().get_nodes_in_group("wall"):
		var ball_rect: Rect2 = Utils.get_global_rect($CollisionShape2D)
		var wall_rect: Rect2 = Utils.get_global_rect(wall.get_node("CollisionShape2D")) 
		
		if ball_rect.intersects(wall_rect):
			return wall
	return null
		
	
	
