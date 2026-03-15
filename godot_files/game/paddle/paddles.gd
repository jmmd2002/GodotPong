extends Node

# --- Paddle Scenes ---
const PADDLE_AI: String = "res://game/paddle/paddle_ai.tscn"
const PADDLE_QCOACH:    String = "res://game/paddle/paddle_Qcoach.tscn"
const PADDLE_QSTUDENT:  String = "res://game/paddle/paddle_Qstudent.tscn"
const PADDLE_POLGRADCOACH:    String = "res://game/paddle/paddle_PGcoach.tscn"
const PADDLE_POLGRADSTUDENT:  String = "res://game/paddle/paddle_PGstudent.tscn"
const PADDLE_POLGRADDNNCOACH: String = "res://game/paddle/paddle_PGDNNcoach.tscn"
const PADDLE_POLGRADDNNSTUDENT: String = "res://game/paddle/paddle_PGDNNstudent.tscn"
const PADDLE_A2CCOACH: String = "res://game/paddle/paddle_A2Ccoach.tscn"
const PADDLE_A2CSTUDENT: String = "res://game/paddle/paddle_A2Cstudent.tscn"
const PADDLE_PPOCOACH: String = "res://game/paddle/paddle_PPOcoach.tscn"
const PADDLE_PPOSTUDENT: String = "res://game/paddle/paddle_PPOstudent.tscn"
const PADDLE_HOMING:   String = "res://game/paddle/paddle_homing.tscn"
const PADDLE_STATIC:   String = "res://game/paddle/paddle_static.tscn"
const PADDLE_MANUAL_A: String = "res://game/paddle/paddle_manual_a.tscn"
const PADDLE_MANUAL_B: String = "res://game/paddle/paddle_manual_b.tscn"


var PADDLE_SCENES: Dictionary = {
	"ai_training": PADDLE_AI,
	"coach": PADDLE_POLGRADCOACH,
	"Qcoach":     PADDLE_QCOACH,
	"Qstudent":   PADDLE_QSTUDENT,
	"PolGradcoach": PADDLE_POLGRADCOACH,
	"PolGradstudent": PADDLE_POLGRADSTUDENT,
	"PolGradDNNcoach": PADDLE_POLGRADDNNCOACH,
	"PolGradDNNstudent": PADDLE_POLGRADDNNSTUDENT,
	"A2Ccoach": PADDLE_A2CCOACH,
	"A2Cstudent": PADDLE_A2CSTUDENT,
	"homing":    PADDLE_HOMING,
	"static":    PADDLE_STATIC,
	"off":       PADDLE_STATIC,
	"manual_a":  PADDLE_MANUAL_A,
	"manual_b":  PADDLE_MANUAL_B,
}
