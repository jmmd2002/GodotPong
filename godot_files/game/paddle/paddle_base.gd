extends Node2D

#sprite is 12x112
@export var height: float = 112.0
@export var width: float = 12.0
@export var speed: float = 400.0

var velocity: Vector2 = Vector2(0.0, 0.0)
var ai_action: String = "STAY"
var ball: Node2D = null
var net_id: int = 1  # peer ID of the player who owns this paddle (set by game.gd)
var net_target_y: float = 0.0   # smoothing target for remote paddle
var net_velocity_y: float = 0.0 # last known velocity for extrapolation between packets
const NET_SNAP_THRESHOLD: float = 60.0  # snap if gap is too large to catch up smoothly

func _initialize() -> void:
	var size: Vector2 = Vector2(width, height)
	var mask: Shape2D = RectangleShape2D.new()
	mask.size = size

	$Sprite2D.scale = size / $Sprite2D.texture.get_size()
	$CollisionShape2D.shape = mask

func _ready() -> void:
	add_to_group("paddle")
	_initialize()
	net_target_y = position.y

func _process(delta: float) -> void:
	# Remote paddle: extrapolate target, snap if too far, otherwise move_toward smoothly
	if Global.is_online and multiplayer.get_unique_id() != net_id:
		net_target_y += net_velocity_y * delta
		if abs(position.y - net_target_y) > NET_SNAP_THRESHOLD:
			position.y = net_target_y
		else:
			position.y = move_toward(position.y, net_target_y, speed * delta)
		return

	var dir: int = get_direction()
	velocity.y = dir * speed

	position += velocity * delta
	handle_collisions()


# Override this in mode-specific subclasses.
# Base behaviour is static (returns 0).
func get_direction() -> int:
	return 0

func set_ai_action(action: String) -> void:
	ai_action = action


#----------------- Collisions --------------------
func handle_collisions() -> void:
	var wall: Node = check_wall_collision()
	if wall:
		var paddle_rect: Rect2 = Utils.get_global_rect($CollisionShape2D)
		var wall_rect: Rect2 = Utils.get_global_rect(wall.get_node("CollisionShape2D"))
		# Use rect edges (not wall node origin) so correction fires even when
		# the paddle overshoots past the wall's node position on a slow frame.
		if paddle_rect.end.y > wall_rect.position.y and velocity.y > 0:
			velocity.y = 0
			position.y -= paddle_rect.end.y - wall_rect.position.y
			return
		if paddle_rect.position.y < wall_rect.end.y and velocity.y < 0:
			velocity.y = 0
			position.y += wall_rect.end.y - paddle_rect.position.y
			return

func check_wall_collision() -> Node:
	for wall in get_tree().get_nodes_in_group("wall"):
		var paddle_rect: Rect2 = Utils.get_global_rect($CollisionShape2D)
		var wall_rect: Rect2 = Utils.get_global_rect(wall.get_node("CollisionShape2D"))

		if paddle_rect.intersects(wall_rect):
			return wall
	return null
