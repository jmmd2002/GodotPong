extends Control

@onready var label: Label = $Title
@onready var vs_static_button: Button = $TrainVsStaticButton
@onready var vs_homing_button: Button = $TrainVsHomingButton
@onready var vs_coach_button: Button = $TrainVsCoachButton
@onready var coach_button: Button = $TrainCoachButton
@onready var back_button: Button = $BackButton




func _ready() -> void:
	var viewport_size: Vector2 = get_viewport_rect().size
	var big_spacing: float = 50.0
	var small_spacing: float = 16.0
	var button_size: Vector2 = Vector2(220.0, 30.0)

	label.text = "AI TRAINING"
	label.position = Vector2(viewport_size.x / 2 - label.size.x / 2, viewport_size.y / 6 - label.size.y / 2)

	var style: StyleBoxFlat = StyleBoxFlat.new()
	style.bg_color = Color(1.0, 0.6, 0.1)

	var style_hover: StyleBoxFlat = StyleBoxFlat.new()
	style_hover.bg_color = Color(1.0, 0.75, 0.3)

	var style_back: StyleBoxFlat = StyleBoxFlat.new()
	style_back.bg_color = Color(0.4, 0.4, 0.4)

	var training_buttons: Array = [vs_static_button, vs_homing_button, vs_coach_button, coach_button]
	var training_labels: Array = ["Train vs Static", "Train vs Homing", "Train vs Coach", "Train Coach"]

	for i in training_buttons.size():
		var btn: Button = training_buttons[i]
		btn.text = training_labels[i]
		btn.size = button_size
		btn.add_theme_stylebox_override("normal", style)
		btn.add_theme_stylebox_override("hover", style_hover)

	back_button.size = Vector2(80.0, 24.0)
	back_button.text = "< Back"
	back_button.add_theme_stylebox_override("normal", style_back)

	var center_x: float = viewport_size.x / 2 - button_size.x / 2
	var start_y: float = label.position.y + label.size.y + big_spacing

	for i in training_buttons.size():
		training_buttons[i].position = Vector2(center_x, start_y + i * (button_size.y + small_spacing))

	back_button.position = Vector2(small_spacing, viewport_size.y - back_button.size.y - small_spacing)

	vs_static_button.pressed.connect(_on_vs_static_pressed)
	vs_homing_button.pressed.connect(_on_vs_homing_pressed)
	vs_coach_button.pressed.connect(_on_vs_coach_pressed)
	coach_button.pressed.connect(_on_coach_pressed)
	back_button.pressed.connect(_on_back_pressed)


func _on_vs_static_pressed() -> void:
	Global.training_mode = "vs_static"
	Global.paddle_a_mode = "ai_training"
	Global.paddle_b_mode = "static"
	get_tree().change_scene_to_file(Paths.AI_TRAINING_RUN)


func _on_vs_homing_pressed() -> void:
	Global.training_mode = "vs_homing"
	Global.paddle_a_mode = "ai_training"
	Global.paddle_b_mode = "homing"
	get_tree().change_scene_to_file(Paths.AI_TRAINING_RUN)


func _on_vs_coach_pressed() -> void:
	# PaddleA trains against the pre-trained coach model
	Global.training_mode = "vs_coach"
	Global.paddle_a_mode = "ai_training"
	Global.paddle_b_mode = "coach"
	get_tree().change_scene_to_file(Paths.AI_TRAINING_RUN)


func _on_coach_pressed() -> void:
	# Trains coach to train Qagent later
	Global.training_mode = "coach"
	Global.paddle_a_mode = "ai_training"
	Global.paddle_b_mode = "off"
	get_tree().change_scene_to_file(Paths.COACH_TRAINING)


func _on_back_pressed() -> void:
	get_tree().change_scene_to_file(Paths.MAIN_MENU)
