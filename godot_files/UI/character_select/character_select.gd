extends Control

const CARD_W: float = 65.0
const CARD_H: float = 55.0
const CARD_GAP: float = 12.0
const CARD_Y: float = 460.0
const CARDS_PER_ROW: int = 5
const PREVIEW_H: float = 180.0
const PREVIEW_CARD_GAP: float = 40.0

# Each entry: display name, keys for each side, and sprite path
const CHARACTERS: Array = [
	{"name": "Human",   "key_a": "manual_a", "key_b": "manual_b", "texture": "res://assets/sprites/spr_paddle.png"},
	{"name": "Coach",   "key_a": "coach",    "key_b": "coach",    "texture": "res://assets/sprites/spr_paddle.png"},
	{"name": "Student", "key_a": "student",  "key_b": "student",  "texture": "res://assets/sprites/spr_paddle.png"},
]

var selected_a: int = 0
var selected_b: int = 0
var cards_a: Array = []
var cards_b: Array = []
var name_label_a: Label = null
var name_label_b: Label = null
var preview_a: TextureRect = null
var preview_b: TextureRect = null

@onready var back_btn: Button = $BackButton
@onready var play_btn: Button = $PlayButton

var font: Font = preload("res://assets/fonts/joystix monospace.otf")


func _ready() -> void:
	var vp: Vector2 = Global.VIEWPORT_SIZE

	# Title (already in tscn, just reposition after layout)
	var title: Label = $Title
	title.position = Vector2(vp.x / 2.0 - 200.0, 30.0)

	# Column centers
	var center_a: float = vp.x * 0.25
	var center_b: float = vp.x * 0.75

	# Row width of the first row (used to center the name label)
	var first_row_count: int = min(CARDS_PER_ROW, CHARACTERS.size())
	var row_w: float = first_row_count * CARD_W + (first_row_count - 1) * CARD_GAP

	# Preview sits just above cards with a gap
	var preview_y: float = CARD_Y - PREVIEW_H - PREVIEW_CARD_GAP

	# Name label centered over the card row, above the preview
	var name_y: float = preview_y - 70.0
	name_label_a = _add_player_label(CHARACTERS[selected_a]["name"], Vector2(center_a - row_w / 2.0, name_y), row_w)
	name_label_b = _add_player_label(CHARACTERS[selected_b]["name"], Vector2(center_b - row_w / 2.0, name_y), row_w)

	# Large sprite previews
	preview_a = _make_preview(CHARACTERS[selected_a]["texture"], center_a, preview_y)
	preview_b = _make_preview(CHARACTERS[selected_b]["texture"], center_b, preview_y)

	# Build cards for both sides (max CARDS_PER_ROW per row)
	for i in CHARACTERS.size():
		var data: Dictionary = CHARACTERS[i]
		var row: int = int(float(i) / float(CARDS_PER_ROW))
		var col: int = i % CARDS_PER_ROW
		var cards_in_row: int = min(CARDS_PER_ROW, CHARACTERS.size() - row * CARDS_PER_ROW)
		var row_w2: float = cards_in_row * CARD_W + (cards_in_row - 1) * CARD_GAP
		var x_a: float = center_a - row_w2 / 2.0 + col * (CARD_W + CARD_GAP)
		var x_b: float = center_b - row_w2 / 2.0 + col * (CARD_W + CARD_GAP)
		var y: float = CARD_Y + row * (CARD_H + CARD_GAP)
		cards_a.append(_make_card(data, Vector2(x_a, y), i, "a"))
		cards_b.append(_make_card(data, Vector2(x_b, y), i, "b"))

	# Default selections
	Global.paddle_a_mode = CHARACTERS[selected_a]["key_a"]
	Global.paddle_b_mode = CHARACTERS[selected_b]["key_b"]
	_refresh_highlights()

	# Back button
	_style_btn(back_btn, Color(0.25, 0.25, 0.25), Color(0.38, 0.38, 0.38))
	back_btn.size = Vector2(100.0, 36.0)
	back_btn.position = Vector2(20.0, vp.y - 56.0)
	back_btn.pressed.connect(_on_back_pressed)

	# Play button
	_style_btn(play_btn, Color(1.0, 0.6, 0.1), Color(1.0, 0.75, 0.3))
	play_btn.size = Vector2(140.0, 44.0)
	play_btn.position = Vector2(vp.x / 2.0 - 70.0, vp.y - 110.0)
	play_btn.pressed.connect(_on_play_pressed)


func _make_card(data: Dictionary, pos: Vector2, index: int, side: String) -> Panel:
	var panel := Panel.new()
	panel.size = Vector2(CARD_W, CARD_H)
	panel.position = pos
	panel.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(panel)

	var sprite_tex: Texture2D = load(data["texture"])

	# Crop to top third of the sprite
	var atlas := AtlasTexture.new()
	atlas.atlas = sprite_tex
	var sprite_h: float = sprite_tex.get_height()
	atlas.region = Rect2(0, 0, sprite_tex.get_width(), sprite_h / 3.0)

	# Position sprite: centered horizontally, space on top, flush to bottom
	var cropped_h: float = sprite_tex.get_height() / 3.0
	var aspect: float = sprite_tex.get_width() / cropped_h
	var top_pad: float = 10.0
	var tex_h: float = CARD_H - top_pad          # fills to bottom edge
	var tex_w: float = tex_h * aspect            # width preserving aspect ratio
	var tex := TextureRect.new()
	tex.texture = atlas
	tex.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT
	tex.size = Vector2(tex_w, tex_h)
	tex.position = Vector2((CARD_W - tex_w) / 2.0, top_pad)
	tex.mouse_filter = Control.MOUSE_FILTER_IGNORE
	panel.add_child(tex)

	panel.gui_input.connect(_on_card_input.bind(index, side))
	panel.mouse_entered.connect(_on_card_hover.bind(panel, index, side))
	panel.mouse_exited.connect(_on_card_exit.bind(panel, index, side))

	return panel


func _on_card_input(event: InputEvent, index: int, side: String) -> void:
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		if side == "a":
			selected_a = index
			Global.paddle_a_mode = CHARACTERS[index]["key_a"]
			name_label_a.text = CHARACTERS[index]["name"]
			preview_a.texture = load(CHARACTERS[index]["texture"])
		else:
			selected_b = index
			Global.paddle_b_mode = CHARACTERS[index]["key_b"]
			name_label_b.text = CHARACTERS[index]["name"]
			preview_b.texture = load(CHARACTERS[index]["texture"])
		_refresh_highlights()


func _on_card_hover(panel: Panel, index: int, side: String) -> void:
	var sel: int = selected_a if side == "a" else selected_b
	if index != sel:
		panel.add_theme_stylebox_override("panel", _card_style(Color(0.22, 0.22, 0.22), Color(0.75, 0.75, 0.75), 2))


func _on_card_exit(panel: Panel, index: int, side: String) -> void:
	var sel: int = selected_a if side == "a" else selected_b
	if index != sel:
		panel.add_theme_stylebox_override("panel", _card_style(Color(0.15, 0.15, 0.15), Color(0.45, 0.45, 0.45), 2))


func _refresh_highlights() -> void:
	for i in cards_a.size():
		var on: bool = (i == selected_a)
		cards_a[i].add_theme_stylebox_override("panel",
			_card_style(Color(0.15, 0.15, 0.15), Color(1.0, 0.6, 0.1) if on else Color(0.45, 0.45, 0.45), 3 if on else 2))
	for i in cards_b.size():
		var on: bool = (i == selected_b)
		cards_b[i].add_theme_stylebox_override("panel",
			_card_style(Color(0.15, 0.15, 0.15), Color(1.0, 0.6, 0.1) if on else Color(0.45, 0.45, 0.45), 3 if on else 2))


func _card_style(bg: Color, border: Color, border_w: int) -> StyleBoxFlat:
	var s := StyleBoxFlat.new()
	s.bg_color = bg
	s.border_color = border
	s.set_border_width_all(border_w)
	s.corner_radius_top_left = 6
	s.corner_radius_top_right = 6
	s.corner_radius_bottom_left = 6
	s.corner_radius_bottom_right = 6
	return s


func _add_player_label(text: String, pos: Vector2, width: float) -> Label:
	var lbl := Label.new()
	lbl.text = text
	lbl.position = pos
	lbl.size = Vector2(width, 36.0)
	lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	lbl.add_theme_font_override("font", font)
	lbl.add_theme_font_size_override("font_size", 28)
	lbl.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(lbl)
	return lbl


func _make_preview(texture_path: String, center_x: float, y: float) -> TextureRect:
	var tex := TextureRect.new()
	tex.texture = load(texture_path)
	tex.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	tex.size = Vector2(PREVIEW_H * 0.3, PREVIEW_H)  # narrow tall box matching sprite ratio
	tex.position = Vector2(center_x - tex.size.x / 2.0, y)
	tex.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(tex)
	return tex


func _style_btn(btn: Button, normal: Color, hover: Color) -> void:
	var sn := StyleBoxFlat.new()
	sn.bg_color = normal
	var sh := StyleBoxFlat.new()
	sh.bg_color = hover
	btn.add_theme_stylebox_override("normal", sn)
	btn.add_theme_stylebox_override("hover", sh)
	btn.add_theme_font_override("font", font)
	btn.add_theme_font_size_override("font_size", 16)


func _on_back_pressed() -> void:
	get_tree().change_scene_to_file("res://UI/main_menu/main_menu.tscn")


func _on_play_pressed() -> void:
	get_tree().change_scene_to_file("res://game/game/game.tscn")
