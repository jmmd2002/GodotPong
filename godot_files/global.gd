extends Node

var is_host: bool = false
var join_ip: String = ""
# "manual" | "ai_qlearn" | "homing"
var paddle_a_mode: String = "manual"
# "manual" | "static" | "homing" | "ai_qlearn"
var paddle_b_mode: String = "manual"
# "vs_static" | "vs_homing" | "vs_coach" | "coach"
var training_mode: String = "vs_static"
