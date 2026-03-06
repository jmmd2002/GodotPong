extends StaticBody2D

@export var thickness: float = 20.0
@onready var viewport_size: Vector2 = get_viewport_rect().size

func _initialize():
	var shape: Shape2D = RectangleShape2D.new()
	var size: Vector2 = Vector2(thickness, viewport_size.y)
	shape.size = size
	$CollisionShape2D.shape = shape
	$CollisionShape2D.position = size / 2

	$ColorRect.position = Vector2(0.0, 0.0)
	$ColorRect.size = size

func _ready():
	add_to_group("side_wall")
	_initialize()
