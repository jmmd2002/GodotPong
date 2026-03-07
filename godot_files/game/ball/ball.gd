extends Node2D

@export var ball_radius: float = 16.0 #ball sprite is 32x32
@export var speed: float = 400.0
@export var max_speed: float = 1600.0
@export var max_angle: float = PI/3

var velocity: Vector2 = Vector2(0.0,0.0)
var respawn_timer: float = 0

func _ready():
	add_to_group("ball")
	
	_reset_ball() # start in the center of the screen
	await get_tree().create_timer(respawn_timer).timeout # start the 3-second countdown
	launch_ball() # release ball

func _process(delta):
	# Sub-step: split movement so the ball never travels more than its radius in one step
	var num_steps: int = max(1, int(ceil(velocity.length() * delta / ball_radius)))
	var sub_delta: float = delta / num_steps
	for _i in range(num_steps):
		position += velocity * sub_delta
		if handle_collisions():
			return  # ball was destroyed by a kill zone, stop processing

	# Guard: respawn ball if it has escaped the viewport (physics edge case)
	var _vp_w: float = ProjectSettings.get_setting("display/window/size/viewport_width", 1280)
	var _vp_h: float = ProjectSettings.get_setting("display/window/size/viewport_height", 720)
	var vp: Rect2 = Rect2(0.0, 0.0, _vp_w, _vp_h)
	if position.x < vp.position.x or position.x > vp.end.x or \
	   position.y < vp.position.y or position.y > vp.end.y:
		print("Warning: ball escaped viewport at ", position, ". Respawning.")
		queue_free()
		Dispatcher.emit_ball_destroyed()
	return
	
func _reset_ball():
	# Set size and collision mask
	var ball_size: Vector2 = Vector2(ball_radius*2, ball_radius*2)/2 #make ball smaller
	var ball_mask: Shape2D = RectangleShape2D.new()
	ball_mask.size = ball_size
	
	# Set size and collision mask
	$Sprite2D.scale = ball_size / $Sprite2D.texture.get_size()
	$CollisionShape2D.shape = ball_mask
	var _vp_size: Vector2 = Vector2(
		ProjectSettings.get_setting("display/window/size/viewport_width", 1280),
		ProjectSettings.get_setting("display/window/size/viewport_height", 720)
	)
	global_position = _vp_size / 2
	velocity = Vector2.ZERO
	
func launch_ball() -> void:
	# Random angle, but not too vertical
	var angle: float = randf_range(-PI / 4, PI / 4)
	
	# Randomly choose left or right
	if randi() % 2 == 0:
		angle += PI
	
	velocity = Vector2.RIGHT.rotated(angle) * speed
		
		
		
# --------------------- Collision handlers --------------------------

func handle_collisions() -> bool:
	var kill_zone: Node = check_kill_zone_collision()
	if kill_zone:
		queue_free()
		Dispatcher.emit_ball_destroyed()
		Dispatcher.emit_scored(kill_zone.side)
		return true

	var wall: Node = check_wall_collision()
	if wall:
		var wall_rect: Rect2 = Utils.get_global_rect(wall.get_node("CollisionShape2D"))
		var ball_rect: Rect2 = Utils.get_global_rect($CollisionShape2D)
		var wall_center_y: float = wall_rect.get_center().y
		# Only reflect if moving toward the wall, not away from it
		if (global_position.y < wall_center_y and velocity.y > 0) or \
		   (global_position.y > wall_center_y and velocity.y < 0):
			velocity.y = -velocity.y
			# Position correction: eject ball to just outside the wall so it
			# cannot sink deeper through the wall over subsequent sub-steps
			if global_position.y <= wall_center_y:
				position.y = wall_rect.position.y - ball_rect.size.y / 2.0
			else:
				position.y = wall_rect.end.y + ball_rect.size.y / 2.0

	var side_wall: Node = check_side_wall_collision()
	if side_wall:
		var sw_rect: Rect2 = Utils.get_global_rect(side_wall.get_node("CollisionShape2D"))
		var ball_rect2: Rect2 = Utils.get_global_rect($CollisionShape2D)
		var wall_center_x: float = sw_rect.get_center().x
		# Only reflect if moving toward the wall
		if (wall_center_x < position.x and velocity.x < 0) or \
		   (wall_center_x > position.x and velocity.x > 0):
			# Bounce back in the opposite x direction
			var angle: float = randf_range(-max_angle, max_angle)
			var dir: float = -sign(velocity.x)
			velocity = Vector2(dir, 0.0).rotated(angle) * speed
			# Position correction: eject ball to just outside the side wall
			if position.x >= wall_center_x:
				position.x = sw_rect.end.x + ball_rect2.size.x / 2.0
			else:
				position.x = sw_rect.position.x - ball_rect2.size.x / 2.0
		
	var paddle: Node = check_paddle_collision()
	if paddle:
		#avoid colliding more than once with each paddle in a row
		if (paddle.position.x < position.x and velocity.x < 0) or (paddle.position.x > position.x and velocity.x > 0):
			var centre: float = paddle.global_position.y
			var offset: float = centre - global_position.y
			var offset_normalized: float = offset / (Utils.get_global_rect(paddle.get_node("CollisionShape2D")).size.y/2)
			var angle: float = max_angle * offset_normalized
			var dir: int = 1 if velocity.x < 0 else -1
			
			speed *= 1.1 #accelerate ball after collision
			speed = min(speed, max_speed) #cap speed
			velocity = dir * Vector2(1.0,0).rotated(-dir * angle) * speed
	return false
	
func check_paddle_collision() -> Node:
	for paddle in get_tree().get_nodes_in_group("paddle"):
		var ball_rect: Rect2 = Utils.get_global_rect($CollisionShape2D)
		var paddle_rect: Rect2 = Utils.get_global_rect(paddle.get_node("CollisionShape2D"))
		
		if ball_rect.intersects(paddle_rect):
			return paddle
	return null

func check_kill_zone_collision() -> Node:
	for kill_zone in get_tree().get_nodes_in_group("kill_zone"):
		var ball_rect: Rect2 = Utils.get_global_rect($CollisionShape2D)
		var kz_rect: Rect2 = Utils.get_global_rect(kill_zone.get_node("CollisionShape2D"))
		if ball_rect.intersects(kz_rect):
			return kill_zone
	return null

func check_side_wall_collision() -> Node:
	for wall in get_tree().get_nodes_in_group("side_wall"):
		var ball_rect: Rect2 = Utils.get_global_rect($CollisionShape2D)
		var wall_rect: Rect2 = Utils.get_global_rect(wall.get_node("CollisionShape2D"))
		if ball_rect.intersects(wall_rect):
			return wall
	return null

func check_wall_collision() -> Node:
	for wall in get_tree().get_nodes_in_group("wall"):
		var ball_rect: Rect2 = Utils.get_global_rect($CollisionShape2D)
		var wall_rect: Rect2 = Utils.get_global_rect(wall.get_node("CollisionShape2D")) 
		
		if ball_rect.intersects(wall_rect):
			return wall
	return null
		
	
	
