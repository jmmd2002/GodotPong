extends Control

@onready var local_button: Button = $CenterContainer/VBoxContainer/LocalPlay
@onready var label: Label = $CenterContainer/VBoxContainer/Pong
var game_path: String = "res://game/game/game.tscn"


func _ready():
	var viewport_size: Vector2 = get_viewport_rect().size
	$CenterContainer.global_position = Vector2(viewport_size.x/2,viewport_size.y/2)
	
	label.text = "PONG"
	
	var style: StyleBoxFlat = StyleBoxFlat.new()
	style.bg_color = Color(1.0, 0.85, 0.2)
	local_button.add_theme_stylebox_override("normal", style)
	local_button.text = "Local Play"
	
	local_button.pressed.connect(_on_local_pressed)
	
func _on_local_pressed():
	get_tree().change_scene_to_file(game_path)

func _on_host_pressed():
	Global.is_host = true
	get_tree().change_scene_to_file("game_path")

func _on_join_pressed():
	Global.join_ip = $VBoxContainer/LineEdit.text
	get_tree().change_scene_to_file("game_path")
