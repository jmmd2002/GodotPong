extends Area2D

@export var side: String
@export var thickness: float = 200.0
var viewport_size: Vector2 = Vector2(
	ProjectSettings.get_setting("display/window/size/viewport_width", 1280),
	ProjectSettings.get_setting("display/window/size/viewport_height", 720)
)

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	add_to_group("kill_zone")
	_initialize()

func _initialize() -> void:
	var mask: Shape2D = RectangleShape2D.new()
	mask.size = Vector2(thickness, viewport_size.y)
	
	$CollisionShape2D.shape = mask
	return
		
	
	
