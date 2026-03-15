extends Node

const MAX_SPEED: float = 1600.0
const VIEWPORT_SIZE: Vector2 = Vector2(1280, 720)
const TARGET_FPS: int = 30
const FRAME_DT: float = 1.0 / TARGET_FPS

var is_host: bool = false
var is_online: bool = false
var join_ip: String = ""
var paddle_a_mode: String = "ai_training"
var paddle_b_mode: String = "coach"
# "vs_static" | "vs_homing" | "vs_coach" | "coach"
var training_method: String = "qvalue"
var training_mode: String = "coach"
