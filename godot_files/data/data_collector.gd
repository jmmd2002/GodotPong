extends Node

var client : StreamPeerTCP = StreamPeerTCP.new()

func _ready():
	var err: Error = client.connect_to_host("127.0.0.1", 5000)
	if err == OK:
		print("Connected to Python server")
	else:
		print("Connection failed")

func _process(delta):
	if client.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		send_state()

func send_state():
	var state: Dictionary = {
		"ball_y": randf(),
		"paddle_y": randf()
	}

	var json: String = JSON.stringify(state) + "\n"
	client.put_data(json.to_utf8_buffer())
