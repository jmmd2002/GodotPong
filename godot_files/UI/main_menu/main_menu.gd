extends Control

@onready var label: Label = $Pong
@onready var local_button: Button = $LocalPlay
@onready var host_button: Button = $Host
@onready var join_button: Button = $Join
@onready var ip_space: LineEdit = $forIP
@onready var ai_train_button: Button = $AITraining



func _ready():
	var viewport_size: Vector2 = get_viewport_rect().size
	var margin: float = 20.0
	var big_spacing: float = 60.0
	var small_spacing: float = 16.0
	var button_size: Vector2 = Vector2(160.0, 40.0)
	var join_w: float = 50.0
	var ip_w: float = button_size.x - join_w - 4.0

	# --- Title ---
	label.text = "PONG"
	label.position = Vector2(viewport_size.x / 2 - label.size.x / 2, viewport_size.y / 6 - label.size.y / 2)

	# --- Orange style for main buttons ---
	var style: StyleBoxFlat = StyleBoxFlat.new()
	style.bg_color = Color(1.0, 0.6, 0.1)
	var style_hover: StyleBoxFlat = StyleBoxFlat.new()
	style_hover.bg_color = Color(1.0, 0.75, 0.3)

	for btn in [local_button, host_button, join_button]:
		btn.add_theme_stylebox_override("normal", style)
		btn.add_theme_stylebox_override("hover", style_hover)
		btn.add_theme_font_size_override("font_size", 16)

	local_button.text = "Local Play"
	host_button.text = "Host"
	join_button.text = "Join"

	local_button.size = button_size
	host_button.size = button_size
	join_button.size = Vector2(join_w, button_size.y)
	ip_space.size = Vector2(ip_w, button_size.y)

	var center_x: float = viewport_size.x / 2 - button_size.x / 2
	var start_y: float = label.position.y + label.size.y + big_spacing

	local_button.position = Vector2(center_x, start_y)
	host_button.position = Vector2(center_x, start_y + button_size.y + small_spacing)
	ip_space.position = Vector2(center_x, host_button.position.y + button_size.y + small_spacing)
	join_button.position = Vector2(center_x + ip_w + 4.0, ip_space.position.y)

	# --- AI Training button (top-right) ---
	var ai_train_style: StyleBoxFlat = StyleBoxFlat.new()
	ai_train_style.bg_color = Color(0.2, 0.5, 0.9)
	var ai_train_hover: StyleBoxFlat = StyleBoxFlat.new()
	ai_train_hover.bg_color = Color(0.35, 0.65, 1.0)

	ai_train_button.text = "AI Training"
	ai_train_button.size = Vector2(130.0, 32.0)
	ai_train_button.add_theme_stylebox_override("normal", ai_train_style)
	ai_train_button.add_theme_stylebox_override("hover", ai_train_hover)
	ai_train_button.add_theme_font_size_override("font_size", 14)
	ai_train_button.position = Vector2(viewport_size.x - ai_train_button.size.x - margin, margin)

	local_button.pressed.connect(_on_local_pressed)
	host_button.pressed.connect(_on_host_pressed)
	join_button.pressed.connect(_on_join_pressed)
	ai_train_button.pressed.connect(_on_ai_train_pressed)

func _on_local_pressed():
	get_tree().change_scene_to_file(Paths.CHAR_SELECT)

func _on_ai_train_pressed():
	get_tree().change_scene_to_file(Paths.AI_TRAINING_METHODS)

func _on_host_pressed():
	Global.is_host = true
	NetworkManager.host()
	var lobby = load(Paths.LOBBY).instantiate()
	lobby.setup(true)
	get_tree().root.add_child(lobby)
	get_tree().current_scene.queue_free()
	get_tree().current_scene = lobby

func _on_join_pressed():
	var ip: String = ip_space.text.strip_edges()
	if ip.is_empty():
		ip = "127.0.0.1"
	Global.is_host = false
	Global.join_ip = ip
	NetworkManager.join(ip)
	var lobby = load(Paths.LOBBY).instantiate()
	lobby.setup(false)
	get_tree().root.add_child(lobby)
	get_tree().current_scene.queue_free()
	get_tree().current_scene = lobby
