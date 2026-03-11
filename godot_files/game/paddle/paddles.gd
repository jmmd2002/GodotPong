extends Node

# --- Paddle Scenes ---
const PADDLE_QCOACH:    String = "res://game/paddle/paddle_Qcoach.tscn"
const PADDLE_QSTUDENT:  String = "res://game/paddle/paddle_Qstudent.tscn"
const PADDLE_POLGRADCOACH:    String = "res://game/paddle/paddle_PGcoach.tscn"
const PADDLE_POLGRADSTUDENT:  String = "res://game/paddle/paddle_PGstudent.tscn"
const PADDLE_HOMING:   String = "res://game/paddle/paddle_homing.tscn"
const PADDLE_STATIC:   String = "res://game/paddle/paddle_static.tscn"
const PADDLE_MANUAL_A: String = "res://game/paddle/paddle_manual_a.tscn"
const PADDLE_MANUAL_B: String = "res://game/paddle/paddle_manual_b.tscn"


var PADDLE_SCENES: Dictionary = {
	"Qcoach":     PADDLE_QCOACH,
	"Qstudent":   PADDLE_QSTUDENT,
	"PolGradcoach": PADDLE_POLGRADCOACH,
	"PolGradstudent": PADDLE_POLGRADSTUDENT,
	"homing":    PADDLE_HOMING,
	"static":    PADDLE_STATIC,
	"off":       PADDLE_STATIC,
	"manual_a":  PADDLE_MANUAL_A,
	"manual_b":  PADDLE_MANUAL_B,
}
