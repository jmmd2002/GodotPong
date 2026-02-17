extends CharacterBody2D

#sprite is 12x112
@export var height: float = 112.0
@export var width: float = 12.0
@export var speed : float = 400.0


func _initialize():
	var size: Vector2 = Vector2(width, height)
	var mask: Shape2D = RectangleShape2D.new()
	mask.size = size
	
	$Sprite2D.scale = size / $Sprite2D.texture.get_size()
	$CollisionShape2D.shape = mask
	
func _ready():
	collision_priority = 10
	add_to_group("paddle")
	_initialize()
	
func _physics_process(delta):
	var direction: float = 0.0
	
	if Input.is_action_pressed("move_up_A"):
		direction -= 1
	if Input.is_action_pressed("move_down_A"):
		direction += 1
	
	velocity.y = direction * speed
	move_and_slide()
