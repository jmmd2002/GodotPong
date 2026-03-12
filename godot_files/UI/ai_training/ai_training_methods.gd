extends Control

@onready var label: Label = $Title
@onready var q_learning_button: Button = $QLearningButton
@onready var policy_gradient_button: Button = $PolicyIterationButton
@onready var back_button: Button = $BackButton
@onready var policy_gradient_DNN_button: Button = $PolicyIterationDNNButton


func _ready() -> void:
	var viewport_size: Vector2 = get_viewport_rect().size
	var big_spacing: float = 50.0
	var small_spacing: float = 16.0
	var button_size: Vector2 = Vector2(260.0, 30.0)

	label.text = "SELECT TRAINING METHOD"
	label.position = Vector2(
		viewport_size.x / 2 - label.size.x / 2,
		viewport_size.y / 6 - label.size.y / 2
	)

	var style: StyleBoxFlat = StyleBoxFlat.new()
	style.bg_color = Color(1.0, 0.6, 0.1)
	var style_hover: StyleBoxFlat = StyleBoxFlat.new()
	style_hover.bg_color = Color(1.0, 0.75, 0.3)

	var style_back: StyleBoxFlat = StyleBoxFlat.new()
	style_back.bg_color = Color(0.4, 0.4, 0.4)

	q_learning_button.text = "Q-Learning"
	q_learning_button.size = button_size
	q_learning_button.add_theme_stylebox_override("normal", style)
	q_learning_button.add_theme_stylebox_override("hover", style_hover)

	policy_gradient_button.text = "Policy Gradient"
	policy_gradient_button.size = button_size
	policy_gradient_button.add_theme_stylebox_override("normal", style)
	policy_gradient_button.add_theme_stylebox_override("hover", style_hover)
	
	policy_gradient_DNN_button.text = "Policy Gradient DNN"
	policy_gradient_DNN_button.size = button_size
	policy_gradient_DNN_button.add_theme_stylebox_override("normal", style)
	policy_gradient_DNN_button.add_theme_stylebox_override("hover", style_hover)
	
	back_button.size = Vector2(80.0, 24.0)
	back_button.text = "< Back"
	back_button.add_theme_stylebox_override("normal", style_back)

	var center_x: float = viewport_size.x / 2 - button_size.x / 2
	var start_y: float = label.position.y + label.size.y + big_spacing

	q_learning_button.position = Vector2(center_x, start_y)
	policy_gradient_button.position = Vector2(center_x, start_y + button_size.y + small_spacing)
	policy_gradient_DNN_button.position = Vector2(center_x, start_y + button_size.y + 2 * small_spacing)
	
	back_button.position = Vector2(small_spacing, viewport_size.y - back_button.size.y - small_spacing)

	q_learning_button.pressed.connect(_on_q_learning_pressed)
	policy_gradient_button.pressed.connect(_on_policy_gradient_pressed)
	policy_gradient_DNN_button.pressed.connect(_on_policy_gradient_DNN_pressed)
	back_button.pressed.connect(_on_back_pressed)


func _on_q_learning_pressed() -> void:
	# Default / existing behaviour: use Q-learning
	Global.training_method = "qvalue"
	get_tree().change_scene_to_file(Paths.AI_TRAINING_MENU)


func _on_policy_gradient_pressed() -> void:
	Global.training_method = "policy_gradient"
	get_tree().change_scene_to_file(Paths.AI_TRAINING_MENU)

func _on_policy_gradient_DNN_pressed() -> void:
	Global.training_method = "policy_gradient_dnn"
	get_tree().change_scene_to_file(Paths.AI_TRAINING_MENU)

func _on_back_pressed() -> void:
	get_tree().change_scene_to_file(Paths.MAIN_MENU)
