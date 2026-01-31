extends CharacterBody2D

@export var speed : float = 400.0

func _physics_process(delta):
	var direction: float = 0.0
	
	if Input.is_action_pressed("move_up"):
		direction -= 1
	if Input.is_action_pressed("move_down"):
		direction += 1
	
	velocity.y = direction * speed
	move_and_slide()
	
	# Clamp to screen
	var screen_height: float = get_viewport_rect().size.y
	global_position.y = clamp(global_position.y, 50, screen_height - 50)
