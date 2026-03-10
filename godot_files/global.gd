extends Node

const MAX_SPEED: float = 1600.0
const VIEWPORT_SIZE: Vector2 = Vector2(1280, 720)

const PADDLE_SCENES: Dictionary = {
	"coach":              "res://game/paddle/paddle_coach.tscn",
	"student":   "res://game/paddle/paddle_student.tscn",
	"homing":    "res://game/paddle/paddle_homing.tscn",
	"static":    "res://game/paddle/paddle_static.tscn",
	"off":       "res://game/paddle/paddle_static.tscn",
	"manual_a":  "res://game/paddle/paddle_manual_a.tscn",
	"manual_b":  "res://game/paddle/paddle_manual_b.tscn",
}

var is_host: bool = false
var is_online: bool = false
var join_ip: String = ""
var paddle_a_mode: String = "manual_a"
var paddle_b_mode: String = "manual_b"
# "vs_static" | "vs_homing" | "vs_coach" | "coach"
var training_mode: String = "vs_static"
