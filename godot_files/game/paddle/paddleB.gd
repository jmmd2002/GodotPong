extends Node2D

#sprite is 12x112
@export var height: float = 112.0
@export var width: float = 12.0
@export var speed : float = 400.0
@export var ai_mode: bool = false  # Enable AI control

var velocity: Vector2  = Vector2(0.0, 0.0)
var ai_action: String = "STAY"  # Current AI action

func _initialize():
	var size: Vector2 = Vector2(width, height)
	var mask: Shape2D = RectangleShape2D.new()
	mask.size = size
	
	$Sprite2D.scale = size / $Sprite2D.texture.get_size()
	$CollisionShape2D.shape = mask
	
func _ready():
	add_to_group("paddle")
	_initialize()
	
func _physics_process(delta):
	var dir: int = get_direction()
	velocity.y = dir * speed
	
	handle_collisions()
	
	position += velocity * delta
	
func get_direction() -> int:
	var direction: int = 0
	
	if ai_mode:
		# AI control
		if ai_action == "UP":
			direction = -1
		elif ai_action == "DOWN":
			direction = 1
		elif ai_action == "STAY":
			direction = 0
	else:
		# Human control
		if Input.is_action_pressed("move_up_B"):
			direction -= 1
		if Input.is_action_pressed("move_down_B"):
			direction += 1
	
	return direction

func set_ai_action(action: String) -> void:
	"""Set the AI action for this paddle."""
	ai_action = action
	
	
#----------------- Collisions --------------------
func handle_collisions() -> void:
	var wall: Node = check_wall_collision()
	if wall:
		if wall.position.y > position.y and velocity.y > 0:
			velocity.y = 0
			return
		if wall.position.y < position.y and velocity.y < 0:
			velocity.y = 0
			return
	
func check_wall_collision() -> Node:
	for wall in get_tree().get_nodes_in_group("wall"):
		var paddle_rect: Rect2 = Utils.get_global_rect($CollisionShape2D)
		var wall_rect: Rect2 = Utils.get_global_rect(wall.get_node("CollisionShape2D")) 
		
		if paddle_rect.intersects(wall_rect):
			return wall
	return null
