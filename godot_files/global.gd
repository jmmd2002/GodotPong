extends Node

const MAX_SPEED: float = 1600.0
const VIEWPORT_SIZE: Vector2 = Vector2(1280, 720)

var PADDLE_SCENES: Dictionary = {
	"coach":     Paddles.PADDLE_COACH,
	"student":   Paddles.PADDLE_STUDENT,
	"homing":    Paddles.PADDLE_HOMING,
	"static":    Paddles.PADDLE_STATIC,
	"off":       Paddles.PADDLE_STATIC,
	"manual_a":  Paddles.PADDLE_MANUAL_A,
	"manual_b":  Paddles.PADDLE_MANUAL_B,
}

var is_host: bool = false
var is_online: bool = false
var join_ip: String = ""
var paddle_a_mode: String = "manual_a"
var paddle_b_mode: String = "manual_b"
# "vs_static" | "vs_homing" | "vs_coach" | "coach"
var training_method: String = "q_learning"
var training_mode: String = "vs_static"
