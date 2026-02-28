extends Node

var client: StreamPeerTCP = StreamPeerTCP.new()
var connected: bool = false
var reconnect_timer: float = 0.0
const RECONNECT_INTERVAL: float = 1.0  # seconds

#get game objects
var ball: Node2D
var paddleA: Node2D
var paddleB: Node2D

func _ready() -> void:
	print("Connecting to server...")

func _process(delta: float):
	if not connected:
		reconnect_timer +=  delta
		if reconnect_timer >= RECONNECT_INTERVAL: #try to reconnect after a certain amount of time
			reconnect_timer = 0
			try_connect()
	else:
		send_state()

func try_connect():
	var err: Error = client.connect_to_host("127.0.0.1", 5000)
	if err == OK:
		client.poll() #poll every step for server status
		if client.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			connected = true
			print("Connected to Python server!")
		else:
			print("Connecting... still not ready")
	else:
		client = StreamPeerTCP.new()
		print("Connection error with value ", err, ". Retrying...")

func send_state():
	
	for b in get_tree().get_nodes_in_group("ball"):
		ball = b #currently gets only first ball; needs fixing later
		break
	paddleA = get_tree().get_root().get_node_or_null("Game/PaddleA")
	paddleB = get_tree().get_root().get_node_or_null("Game/PaddleB")
	
	#skip if any of the instances does not exist
	if ball == null or paddleA == null or paddleB == null:
		return
	else:	
		var state: Dictionary = {
			"paddleA_x": paddleA.position.x,
			"paddleA_y": paddleA.position.y,
			"paddleB_x": paddleB.position.x,
			"paddleB_y": paddleB.position.y,
			"ball_x": ball.position.x,
			"ball_y": ball.position.y,
			"ball_vx": ball.velocity.x,
			"ball_vy": ball.velocity.y
		}
		var json: String = JSON.stringify(state) + "\n"
		client.put_data(json.to_utf8_buffer())
		print("Sent:", json)
		return
