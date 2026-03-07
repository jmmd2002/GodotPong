extends StaticBody2D

@export var thickness: float = 20.0
var viewport_size: Vector2 = Vector2(
	ProjectSettings.get_setting("display/window/size/viewport_width", 1280),
	ProjectSettings.get_setting("display/window/size/viewport_height", 720)
)

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
