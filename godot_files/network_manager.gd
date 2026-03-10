extends Node

const PORT: int = 7777

signal player_connected
signal connection_failed
signal player_disconnected

var joiner_id: int = 0

func host() -> void:
	var peer: ENetMultiplayerPeer = ENetMultiplayerPeer.new()
	var err: int = peer.create_server(PORT, 2)
	if err != OK:
		print("NetworkManager: Failed to create server: ", err)
		return
	multiplayer.multiplayer_peer = peer
	multiplayer.peer_connected.connect(_on_peer_connected)
	multiplayer.peer_disconnected.connect(_on_peer_disconnected)
	Global.is_online = true
	print("NetworkManager: Hosting on port ", PORT)

func join(ip: String) -> void:
	var peer: ENetMultiplayerPeer = ENetMultiplayerPeer.new()
	var err: int = peer.create_client(ip, PORT)
	if err != OK:
		print("NetworkManager: Failed to create client: ", err)
		connection_failed.emit()
		return
	multiplayer.multiplayer_peer = peer
	multiplayer.connected_to_server.connect(_on_connected_to_server)
	multiplayer.connection_failed.connect(_on_connection_failed)
	Global.is_online = true
	print("NetworkManager: Connecting to ", ip, ":", PORT)

func stop() -> void:
	if multiplayer.multiplayer_peer:
		multiplayer.multiplayer_peer.close()
	multiplayer.multiplayer_peer = null
	joiner_id = 0
	Global.is_online = false
	Global.is_host = false

func _on_peer_connected(id: int) -> void:
	print("NetworkManager: Peer connected: ", id)
	joiner_id = id
	player_connected.emit()

func _on_peer_disconnected(id: int) -> void:
	print("NetworkManager: Peer disconnected: ", id)
	player_disconnected.emit()

func _on_connected_to_server() -> void:
	print("NetworkManager: Connected to server!")
	player_connected.emit()

func _on_connection_failed() -> void:
	print("NetworkManager: Connection failed!")
	connection_failed.emit()
