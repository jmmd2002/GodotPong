extends Node2D

@export var ball_scene: PackedScene

func _ready() -> void:
	var viewport_size: Vector2 = Vector2(
		ProjectSettings.get_setting("display/window/size/viewport_width", 1280),
		ProjectSettings.get_setting("display/window/size/viewport_height", 720)
	)

	# Spawn paddles dynamically based on character selection
	var paddle_a: Node = _spawn_paddle(Global.paddle_a_mode, "PaddleA")
	var paddle_b: Node = _spawn_paddle(Global.paddle_b_mode, "PaddleB")

	# In online mode assign ownership so each paddle only runs physics on its owner's machine
	if Global.is_online:
		paddle_a.net_id = 1  # host always has peer ID 1
		# joiner_id is only populated on the host — on the joiner use their own ID
		paddle_b.net_id = NetworkManager.joiner_id if multiplayer.is_server() else multiplayer.get_unique_id()

	paddle_a.position = Vector2(40.0, viewport_size.y / 2)
	paddle_b.position = Vector2(viewport_size.x - 40.0, viewport_size.y / 2)

	$Ball.position = Vector2(viewport_size.x / 2, viewport_size.y / 2)
	$Ceiling.position = Vector2(0.0, 0.0)
	$Floor.position = Vector2(0.0, viewport_size.y - $Floor.thickness)
	$KillZoneA.position = Vector2(20.0 - $KillZoneA.thickness / 2, viewport_size.y / 2)
	$KillZoneB.position = Vector2(viewport_size.x - 20.0 + $KillZoneB.thickness / 2, viewport_size.y / 2)

	Dispatcher.ball_destroyed.connect(_on_ball_destroyed)
	if Global.is_online and multiplayer.is_server(): #sync scores
		GameManager.score_changed.connect(_on_score_changed_host)


# Centralised network sync: routes to host or joiner update each frame
func _process(_delta: float) -> void:
	if not Global.is_online:
		return
	if multiplayer.is_server():
		_update_with_host()
	else:
		_update_with_joiner()


func _update_with_host() -> void:
	# Ball state → joiner
	var balls: Array = get_tree().get_nodes_in_group("ball")
	if balls.size() > 0:
		var b: Node2D = balls[0]
		_rpc_receive_ball_state.rpc_id(NetworkManager.joiner_id, b.position, b.velocity)
	# Paddle A position → joiner
	var pa: Node = get_node_or_null("PaddleA")
	if pa:
		_rpc_receive_paddle_y.rpc_id(NetworkManager.joiner_id, "a", pa.position.y)


func _update_with_joiner() -> void:
	# Paddle B position → host
	var pb: Node = get_node_or_null("PaddleB")
	if pb:
		_rpc_receive_paddle_y.rpc_id(1, "b", pb.position.y)


# Runs on joiner: applies host-authoritative ball state to whatever ball exists
@rpc("authority", "unreliable")
func _rpc_receive_ball_state(pos: Vector2, vel: Vector2) -> void:
	var balls: Array = get_tree().get_nodes_in_group("ball")
	if balls.size() > 0:
		balls[0].position = pos
		balls[0].velocity = vel


# Runs on recipient: updates the opponent's paddle Y
@rpc("any_peer", "unreliable")
func _rpc_receive_paddle_y(side: String, y: float) -> void:
	var paddle: Node = get_node_or_null("PaddleA" if side == "a" else "PaddleB")
	if paddle:
		paddle.position.y = y


func _spawn_paddle(mode: String, node_name: String) -> Node:
	var scene_path: String = Global.PADDLE_SCENES.get(mode, Global.PADDLE_SCENES["manual_a"])
	var paddle: Node = load(scene_path).instantiate()
	paddle.name = node_name
	add_child(paddle)
	return paddle


# Host: fired by GameManager.score_changed, forwards new score to joiner
func _on_score_changed_host(left: int, right: int) -> void:
	_rpc_sync_score.rpc_id(NetworkManager.joiner_id, left, right)


# Runs on joiner: sets GameManager scores directly and refreshes the display
@rpc("authority", "reliable")
func _rpc_sync_score(left: int, right: int) -> void:
	GameManager.score_left = left
	GameManager.score_right = right
	GameManager.score_changed.emit(left, right)


func _on_ball_destroyed() -> void:
	# Tell joiner to also destroy their ball and spawn a fresh one
	if Global.is_online and multiplayer.is_server():
		_rpc_sync_ball_lifecycle.rpc_id(NetworkManager.joiner_id)
	spawn_ball()


# Runs on joiner only: mirrors what the host just did locally
@rpc("authority", "reliable")
func _rpc_sync_ball_lifecycle() -> void:
	for b in get_tree().get_nodes_in_group("ball"):
		b.queue_free()
	spawn_ball()


func spawn_ball() -> void:
	var active_balls = get_tree().get_nodes_in_group("ball").filter(
		func(b): return not b.is_queued_for_deletion()
	)
	if active_balls.size() > 0:
		return
	var ball = ball_scene.instantiate()
	add_child(ball)
