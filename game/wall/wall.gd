extends StaticBody2D

@export var thickness: float = 60.0
@onready var viewport_size: Vector2 = get_viewport_rect().size

func _initialize():
	# Resize the CollisionShape2D
	var shape: Shape2D = RectangleShape2D.new()
	var size: Vector2 = Vector2(viewport_size.x, thickness)
	shape.size = size
	$CollisionShape2D.shape = shape
	$CollisionShape2D.position = size / 2
	
	# Resize the ColorRect
	$ColorRect.position = Vector2(0.0, 0.0)
	$ColorRect.size = size

func _ready():
	add_to_group("wall")
	_initialize()
