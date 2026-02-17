extends CharacterBody2D

@export var ball_radius: float = 16.0 #ball sprite is 32x32
@export var speed: float = 400.0
@export var max_angle: float = PI/3

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
	
func _ready():
	
	collision_priority = 1 #avoid nudging paddles away
	_reset_ball() # start in the center of the screen
	await get_tree().create_timer(3.0).timeout # start the 3-second countdown
	launch_ball() # release ball
	
func launch_ball():
	# Random angle, but not too vertical
	var angle: float = randf_range(-PI / 4, PI / 4)
	
	# Randomly choose left or right
	if randi() % 2 == 0:
		angle += PI
	
	velocity = Vector2.RIGHT.rotated(angle) * speed

func _physics_process(delta):
	var collision: KinematicCollision2D = move_and_collide(velocity*delta)
	
	if collision:
		var collider: Object = collision.get_collider()
		
		if collider.is_in_group("paddle"):
			var centre: float = collider.global_position.y
			var offset: float = global_position.y - centre
			var offset_normalized: float = offset / (collider.height/2)
			var bounce_angle: float = max_angle * offset_normalized
			var dir = 1 if velocity.x < 0 else -1
			
			velocity = Vector2(dir, 0).rotated(bounce_angle*dir) * speed
			
		else:
			velocity = velocity.bounce(collision.get_normal()).normalized() * speed
		
