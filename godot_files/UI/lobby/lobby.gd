extends Control

var _is_host: bool = false
var main_menu_path: String = "res://UI/main_menu/main_menu.tscn"
var game_path: String = "res://game/game/game.tscn"

@onready var status_label: Label = $StatusLabel
@onready var back_button: Button = $BackButton

func setup(is_host: bool) -> void:
	_is_host = is_host

func _ready() -> void:
	var viewport_size: Vector2 = Global.VIEWPORT_SIZE

	# --- Background ---
	var bg: ColorRect = ColorRect.new()
	bg.color = Color(0.08, 0.08, 0.08)
	bg.size = viewport_size
	bg.position = Vector2.ZERO
	add_child(bg)
	move_child(bg, 0)

	# --- Status label ---
	var lbl: Label = $StatusLabel
	lbl.text = "Waiting for player to join" if _is_host else "Connecting to host"
	lbl.add_theme_font_size_override("font_size", 28)
	lbl.add_theme_color_override("font_color", Color(1.0, 1.0, 1.0))
	lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	lbl.size = Vector2(viewport_size.x * 0.8, 60)
	lbl.position = Vector2(viewport_size.x * 0.1, viewport_size.y / 2 - 60)

	# --- Dots animation timer ---
	var timer: Timer = Timer.new()
	timer.wait_time = 0.5
	timer.autostart = true
	timer.timeout.connect(_on_dot_tick)
	add_child(timer)

	# --- Back button ---
	var btn: Button = $BackButton
	var style: StyleBoxFlat = StyleBoxFlat.new()
	style.bg_color = Color(0.6, 0.1, 0.1)
	var style_hover: StyleBoxFlat = StyleBoxFlat.new()
	style_hover.bg_color = Color(0.8, 0.2, 0.2)
	btn.add_theme_stylebox_override("normal", style)
	btn.add_theme_stylebox_override("hover", style_hover)
	btn.add_theme_font_size_override("font_size", 16)
	btn.text = "Cancel"
	btn.size = Vector2(120, 40)
	btn.position = Vector2(viewport_size.x / 2 - 60, viewport_size.y * 0.65)
	btn.pressed.connect(_on_back_pressed)

	# Connect network signals
	NetworkManager.player_connected.connect(_on_player_connected)
	NetworkManager.connection_failed.connect(_on_connection_failed)

var _dot_count: int = 0
var _base_text: String = ""

func _on_dot_tick() -> void:
	if _base_text.is_empty():
		_base_text = "Waiting for player to join" if _is_host else "Connecting to host"
	_dot_count = (_dot_count + 1) % 4
	$StatusLabel.text = _base_text + ".".repeat(_dot_count)

func _on_player_connected() -> void:
	$StatusLabel.text = "Connected! Starting game..."
	await get_tree().create_timer(2.0).timeout
	get_tree().change_scene_to_file(game_path)

func _on_connection_failed() -> void:
	$StatusLabel.text = "Connection failed. Returning to menu..."
	NetworkManager.stop()
	await get_tree().create_timer(2.0).timeout
	get_tree().change_scene_to_file(main_menu_path)

func _on_back_pressed() -> void:
	NetworkManager.stop()
	get_tree().change_scene_to_file(main_menu_path)
