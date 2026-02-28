extends Control

@onready var label: Label = $Pong
@onready var local_button: Button = $LocalPlay
@onready var host_button: Button = $Host
@onready var join_button: Button = $Join
@onready var ip_space: LineEdit = $forIP
var game_path: String = "res://game/game/game.tscn"


func _ready():
	var viewport_size: Vector2 = get_viewport_rect().size
	var big_spacing: int = 50
	var small_spacing: int = 20
	
	label.text = "PONG"
	label.position = Vector2(viewport_size.x/2-label.size.x/2, viewport_size.y/6-label.size.y/2)
	
	var style: StyleBoxFlat = StyleBoxFlat.new()
	style.bg_color = Color(1.0, 0.6, 0.1)
	local_button.add_theme_stylebox_override("normal", style)
	host_button.add_theme_stylebox_override("normal", style)
	join_button.add_theme_stylebox_override("normal", style)
	local_button.text = "Local Play"
	host_button.text = "Host"
	join_button.text = "Join"
	
	var button_size: Vector2 = Vector2(150, small_spacing)
	local_button.size = button_size
	host_button.size = button_size
	join_button.size = Vector2(button_size.x/3 - small_spacing/2, button_size.y)
	ip_space.size = Vector2(button_size.x * 2/3 - small_spacing/2, button_size.y)
	
	local_button.position = Vector2(viewport_size.x/2-local_button.size.x/2, label.position.y+label.size.y+big_spacing+local_button.size.y/2)
	host_button.position = Vector2(viewport_size.x/2-host_button.size.x/2, local_button.position.y+local_button.size.y+small_spacing)
	join_button.position = Vector2(host_button.position.x+host_button.size.x-join_button.size.x, host_button.position.y+host_button.size.y+small_spacing)
	ip_space.position = Vector2(host_button.position.x, join_button.position.y)
	
	local_button.pressed.connect(_on_local_pressed)
	
func _on_local_pressed():
	get_tree().change_scene_to_file(game_path)

func _on_host_pressed():
	Global.is_host = true
	get_tree().change_scene_to_file("game_path")

func _on_join_pressed():
	Global.join_ip = $forIP.text
	get_tree().change_scene_to_file("game_path")
