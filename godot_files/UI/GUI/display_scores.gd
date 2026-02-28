extends Control

@onready var label_left: Label = $LabelLeft
@onready var label_right: Label = $LabelRight
@onready var label_middle: Label = $LabelMiddle

func _ready():
	var viewport_size: Vector2 = get_viewport().get_visible_rect().size
	var px_size: float = 200
	var px_offset: float = px_size/20
	label_left.size = Vector2(px_size, px_size)
	label_right.size = Vector2(px_size, px_size)
	label_middle.size = Vector2(px_size, px_size)
	label_left.position = Vector2(viewport_size.x/4-px_size/2, viewport_size.y/2-label_left.get_size().y/2-px_offset)
	label_right.position = Vector2(3*viewport_size.x/4-px_size/2, viewport_size.y/2 - label_right.get_size().y/2-px_offset)
	label_middle.position = Vector2(viewport_size.x/2-px_size/2, viewport_size.y/2 - label_middle.get_size().y/2-px_offset)
	label_middle.text = "-"
	
	# initialize UI with current score and make it change whenever score updates
	GameManager.score_changed.connect(_on_score_changed)
	_on_score_changed(GameManager.score_left, GameManager.score_right)

func _on_score_changed(left, right):
	label_left.text = str(right)
	label_right.text = str(left)
