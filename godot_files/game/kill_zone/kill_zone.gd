extends Area2D

@export var side: String
@export var thickness: float = 200.0
@onready var viewport_size: Vector2 = get_viewport_rect().size

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	add_to_group("kill_zone")
	_initialize()

func _initialize() -> void:
	var mask: Shape2D = RectangleShape2D.new()
	mask.size = Vector2(thickness, viewport_size.x)
	
	$CollisionShape2D.shape = mask
	return
		
	
	
